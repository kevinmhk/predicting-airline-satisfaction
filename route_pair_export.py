"""Fit and export the approved fixed route-pair student blend."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

# Scientific imports intentionally follow the thread-limit environment setup.
# ruff: noqa: E402
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import cached_route_blends as cached
import final_sprint_experiments as route
from compare import prepare_catboost_frame
from distance_profile_experiment import (
    append_distance_profile_features,
    fit_distance_profile_table,
)
from distance_profile_export import (
    _load_export_profile_table,
    _load_frozen_teacher,
    _save_export_profile_table,
)
from lightgbm_experiment import (
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
)
from source_teacher_cv import _verify_teacher_cache
from source_teacher_experiment import append_teacher_feature
from source_teacher_export import (
    _code_hashes,
    _fit_and_save_lightgbm,
    _validate_frozen_catboost,
    _versions,
    build_export_config,
)
from train import (
    ID_COLUMN,
    MAX_THREADS,
    TARGET_COLUMN,
    InputSpec,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
    validate_submission_file,
)

OUTPUT_NAME = "route_pair_export"
METADATA_NAME = "export_metadata.json"
FEATURE_SPEC_NAME = "feature_spec.json"
PROFILE_TABLE_NAME = "distance_profile_table.npz"
CONTEXT_MAPS_NAME = "route_context_maps.json"
HIERARCHICAL_MAPS_NAME = "route_hierarchical_maps.json"
CONTEXT_MODEL_NAME = "context_student_model.txt"
HIERARCHICAL_MODEL_NAME = "hierarchical_student_model.txt"
COMPONENTS_NAME = "component_test_predictions.npz"
SUBMISSION_NAME = "submission_slowcat_route_context_hierarchical_50_25_25.csv"
RELOAD_NAME = "reload_predictions.npy"
SPLIT_SEEDS = (42, 2024)
CONTEXT_EXPERIMENT = "final_sprint_route_context_target_encoding127"
HIERARCHICAL_EXPERIMENT = "final_sprint_route_hierarchical_encoding"
ROUTE_MAP_NAMES = (
    "route_distance_encoding",
    "route_pair_encoding",
    "route_travel_encoding",
    "route_customer_encoding",
)
ROUTE_KEYS = (
    route.ROUTE_KEY_DISTANCE,
    route.ROUTE_KEY_PAIR,
    route.ROUTE_KEY_TRAVEL_PAIR,
    route.ROUTE_KEY_CUSTOMER_PAIR,
)
CATEGORY_NAMES = ("Class", "Type of Travel", "Customer Type")
SMOOTH = 20.0
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_TEST_ROWS = 299844
EXPECTED_TEACHER_ITERATIONS = 531
EXPECTED_CATBOOST_ITERATIONS = 4740
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
ACCEPTED_PUBLIC_CSV_SHA256 = (
    "048abb59dc3ffb20e495dd414f7e7be3ca0e795d3fa66145516957c4b1d148e7"
)


class RoutePairExportTimeout(TimeoutError):
    """Raised when the fixed export reaches its internal deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise RoutePairExportTimeout("route-pair export exceeded its deadline")


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _spec(record: dict[str, Any]) -> InputSpec:
    return InputSpec(
        tuple(record["feature_columns"]),
        tuple(record["numeric_columns"]),
        tuple(record["categorical_columns"]),
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_json(temporary, value)
    temporary.replace(path)


def _output_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _group_stats(keys: list[str], labels: np.ndarray) -> dict[str, dict[str, float]]:
    grouped = (
        pd.DataFrame({"key": keys, "target": labels})
        .groupby("key", sort=False, dropna=False)["target"]
        .agg(["sum", "count"])
    )
    return {
        str(key): {"sum": float(row["sum"]), "count": float(row["count"])}
        for key, row in grouped.iterrows()
    }


def fit_route_maps(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    hierarchical: bool,
    feature_names: tuple[str, ...] = ROUTE_MAP_NAMES,
) -> dict[str, Any]:
    """Fit compact full-training route maps for label-free test inference."""
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or len(labels) != len(features) or set(labels) != {0.0, 1.0}:
        raise ValueError("route-map labels must be binary and match training rows")
    if len(feature_names) != len(ROUTE_KEYS):
        raise ValueError("route-map feature names must contain four fields")
    key_frame = route._route_key_frame(features, include_context=True)
    categories = {
        name: [route._canonical_class_key(value) for value in features[name]]
        for name in CATEGORY_NAMES
    }
    conditional_priors = {
        name: {
            key: value["sum"] / value["count"]
            for key, value in _group_stats(values, labels).items()
        }
        for name, values in categories.items()
    }
    return {
        "version": 1,
        "strategy": "hierarchical_route_prior" if hierarchical else "flat_route_prior",
        "smooth": SMOOTH,
        "folds": 5,
        "shuffle": True,
        "random_state": 42,
        "global_prior": float(np.mean(labels)),
        "conditional_priors": conditional_priors,
        "feature_names": list(feature_names),
        "key_names": list(ROUTE_KEYS),
        "maps": {
            key: _group_stats(key_frame[key].tolist(), labels) for key in ROUTE_KEYS
        },
    }


def apply_route_maps(features: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    """Apply persisted route maps, including explicit unknown and missing fallbacks."""
    if tuple(record.get("key_names", ())) != ROUTE_KEYS:
        raise ValueError("route-map key contract changed")
    names = tuple(record.get("feature_names", ()))
    if len(names) != len(ROUTE_KEYS):
        raise ValueError("route-map feature contract changed")
    smooth = float(record.get("smooth", SMOOTH))
    if smooth != SMOOTH:
        raise ValueError("route-map smoothing changed")
    global_prior = float(record["global_prior"])
    hierarchical = record.get("strategy") == "hierarchical_route_prior"
    key_frame = route._route_key_frame(features, include_context=True)
    categories = {
        name: [route._canonical_class_key(value) for value in features[name]]
        for name in CATEGORY_NAMES
    }
    values = np.empty((len(features), len(ROUTE_KEYS)), dtype=np.float64)
    for column, key_name in enumerate(ROUTE_KEYS):
        stats = record["maps"][key_name]
        for row, key in enumerate(key_frame[key_name]):
            statistic = stats.get(str(key), {"sum": 0.0, "count": 0.0})
            prior = global_prior
            if hierarchical and column:
                prior = float(
                    record["conditional_priors"][CATEGORY_NAMES[column - 1]].get(
                        categories[CATEGORY_NAMES[column - 1]][row], global_prior
                    )
                )
            values[row, column] = (float(statistic["sum"]) + smooth * prior) / (
                float(statistic["count"]) + smooth
            )
    result = pd.DataFrame(values, columns=names, index=features.index)
    for column in result:
        validate_probabilities(result[column].to_numpy(), len(features))
    return result


def save_route_maps(path: Path, record: dict[str, Any]) -> None:
    """Persist maps as compact JSON without raw test rows or labels."""
    _atomic_json(path, record)


def load_route_maps(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    apply_route_maps(
        pd.DataFrame(
            {
                "Flight Distance": [0.0],
                "Class": ["<missing>"],
                "Type of Travel": ["<missing>"],
                "Customer Type": ["<missing>"],
            }
        ),
        record,
    )
    return record


def fit_route_features(
    features: pd.DataFrame,
    spec: InputSpec,
    labels: np.ndarray,
    kind: str,
) -> route.RouteEncodingResult:
    """Use the existing five-fold cross-fit helper on train rows only."""
    labels = np.asarray(labels, dtype=np.int8)
    fit_indices = np.arange(len(labels), dtype=np.int64)
    test_indices = np.arange(len(labels), len(features), dtype=np.int64)
    if kind == "context":
        return route.fit_route_target_encoding(
            features,
            spec,
            labels,
            fit_indices,
            np.array([], dtype=np.int64),
            test_indices,
            include_context=True,
        )
    if kind == "hierarchical":
        return route.fit_hierarchical_route_encoding(
            features,
            spec,
            labels,
            fit_indices,
            np.array([], dtype=np.int64),
            test_indices,
        )
    raise ValueError(f"unknown route encoding kind: {kind}")


def derive_refit_iterations(metadata: dict[str, Any]) -> int:
    """Return ceil(median(native best counts)) for exactly the two approved seeds."""
    if metadata.get("status") != "completed":
        raise RuntimeError("route component metadata is not completed")
    records = metadata.get("validation_results")
    if not isinstance(records, list):
        raise RuntimeError("route component validation results are missing")
    by_seed = {record.get("split_seed"): record for record in records}
    if set(by_seed) != set(SPLIT_SEEDS) or len(records) != len(SPLIT_SEEDS):
        raise RuntimeError("route component must contain exactly two seeds")
    counts: list[int] = []
    for seed in SPLIT_SEEDS:
        record = by_seed[seed]
        best = int(record.get("best_iteration", 0))
        actual = int(record.get("actual_iterations", 0))
        if best <= 0 or actual <= 0 or best != actual:
            raise RuntimeError("iteration provenance is invalid")
        counts.append(best)
    return int(math.ceil(median(counts)))


def build_student_config(iterations: int, num_leaves: int) -> dict[str, Any]:
    """Build the fixed LightGBM config with only the approved two deltas."""
    if iterations <= 0 or num_leaves <= 0:
        raise ValueError("iterations and num_leaves must be positive")
    config = build_export_config(iterations)
    config["num_leaves"] = num_leaves
    return config


def route_pair_blend(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> np.ndarray:
    """Apply the fixed CatBoost/context/hierarchical 50/25/25 blend."""
    catboost = np.asarray(catboost, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    hierarchical = np.asarray(hierarchical, dtype=np.float64)
    if (
        catboost.ndim != 1
        or context.shape != catboost.shape
        or hierarchical.shape != catboost.shape
    ):
        raise ValueError("blend components must have equal one-dimensional shapes")
    for values in (catboost, context, hierarchical):
        validate_probabilities(values, len(values))
    result = 0.5 * catboost + 0.25 * context + 0.25 * hierarchical
    validate_probabilities(result, len(result))
    return result


def _validate_qualification(
    artifacts_dir: Path,
    train: pd.DataFrame,
    labels: np.ndarray,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    metadata_path = (
        artifacts_dir
        / "cached_route_blends_route_pair_average"
        / "compare_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selection = metadata.get("selection", {})
    if (
        metadata.get("input_sha256_before") != input_hashes
        or metadata.get("input_sha256_after") != input_hashes
        or metadata.get("source_sha256_before") != cached._own_source_hashes()
        or metadata.get("source_sha256_after") != cached._own_source_hashes()
    ):
        raise RuntimeError("accepted route-pair input/source lineage changed")
    if metadata.get("status") != "completed" or selection.get("qualifies") is not True:
        raise RuntimeError("accepted route-pair qualification is not complete")
    expected_gains = {
        42: 0.00022361123229974744,
        2024: 0.0002892394910254392,
    }
    comparisons = selection.get("comparison")
    if not isinstance(comparisons, list) or {
        item.get("split_seed") for item in comparisons
    } != set(SPLIT_SEEDS):
        raise RuntimeError("accepted route-pair comparisons are incomplete")
    accepted_records = {
        int(record["split_seed"]): record
        for record in metadata.get("validation_results", [])
        if "split_seed" in record
    }
    if set(accepted_records) != set(SPLIT_SEEDS):
        raise RuntimeError("accepted route-pair validation lineage is incomplete")
    for comparison in comparisons:
        seed = int(comparison.get("split_seed", -1))
        if not np.isclose(
            float(comparison.get("auc_gain", -1.0)),
            expected_gains.get(seed, -2.0),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError("accepted route-pair metric lineage changed")
    cache_artifacts = Path(os.path.relpath(artifacts_dir, Path.cwd()))
    prior = route._load_prior_lightgbm(cache_artifacts, input_hashes)
    checked: list[dict[str, Any]] = []
    for seed in SPLIT_SEEDS:
        _, record, _ = cached._source_components(
            cached.ROUTE_PAIR_AVERAGE,
            cache_artifacts,
            train,
            labels,
            prior,
            input_hashes,
            route._source_hashes(),
            seed,
        )
        accepted = accepted_records[seed]
        if not cached._metric_match(record["candidate"], accepted["candidate"]):
            raise RuntimeError("accepted route-pair candidate metrics changed")
        if not cached._metric_match(record["baseline"], accepted["baseline"]):
            raise RuntimeError("accepted route-pair baseline metrics changed")
        if record["lineage"] != accepted["lineage"]:
            raise RuntimeError("accepted route-pair component lineage changed")
        checked.append(
            {
                "split_seed": seed,
                "candidate": record["candidate"],
                "baseline": record["baseline"],
                "weights": record["weights"],
                "lineage": record["lineage"],
            }
        )
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "selection": selection,
        "checks": checked,
    }


def _predict_lightgbm(
    model: lgb.Booster,
    features: pd.DataFrame,
    spec: InputSpec,
    levels: dict[str, list[str]],
) -> np.ndarray:
    values = np.asarray(
        model.predict(prepare_lightgbm_frame(features, spec, levels)), dtype=np.float64
    )
    validate_probabilities(values, len(features))
    return values


def _fit_student(
    *,
    name: str,
    kind: str,
    base_train: pd.DataFrame,
    base_test: pd.DataFrame,
    base_spec: InputSpec,
    labels: np.ndarray,
    iterations: int,
    num_leaves: int,
    output_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> tuple[np.ndarray, InputSpec, dict[str, list[str]], dict[str, Any]]:
    combined = pd.concat([base_train, base_test], ignore_index=True)
    result = fit_route_features(combined, base_spec, labels, kind)
    train_features = result.features.iloc[: len(base_train)].copy()
    test_features = result.features.iloc[len(base_train) :].copy()
    map_record = fit_route_maps(
        base_train,
        labels,
        hierarchical=kind == "hierarchical",
        feature_names=tuple(result.spec.feature_columns[-4:]),
    )
    persisted_test = apply_route_maps(base_test, map_record)
    helper_test = test_features.loc[
        :, list(result.spec.feature_columns[-4:])
    ].reset_index(drop=True)
    if not np.allclose(
        persisted_test.to_numpy(), helper_test.to_numpy(), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(f"{kind} persisted route maps differ from helper inference")
    map_path = output_dir / (
        CONTEXT_MAPS_NAME if kind == "context" else HIERARCHICAL_MAPS_NAME
    )
    save_route_maps(map_path, map_record)
    encoding_metadata = result.metadata
    student_spec = result.spec
    del combined, result, helper_test, persisted_test
    gc.collect()
    levels = fit_category_levels(train_features, student_spec)
    model_path = output_dir / (
        CONTEXT_MODEL_NAME if kind == "context" else HIERARCHICAL_MODEL_NAME
    )
    model = build_lightgbm_classifier(**build_student_config(iterations, num_leaves))
    record = _fit_and_save_lightgbm(
        name=name,
        model=model,
        features=prepare_lightgbm_frame(train_features, student_spec, levels),
        labels=labels,
        spec=student_spec,
        model_path=model_path,
        metadata=metadata,
        metadata_path=metadata_path,
    )
    record["route_encoding"] = encoding_metadata
    record["test_map_comparison_rows"] = len(base_test)
    metadata.setdefault("files", {})[model_path.name] = sha256_file(model_path)
    metadata["files"][map_path.name] = sha256_file(map_path)
    save_json(metadata_path, metadata)
    predictions = _predict_lightgbm(model.booster_, test_features, student_spec, levels)
    del model, train_features, test_features
    gc.collect()
    return predictions, student_spec, levels, record


def _write_submission(
    path: Path, ids: np.ndarray, values: np.ndarray
) -> dict[str, Any]:
    validate_probabilities(values, len(ids))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ID_COLUMN, TARGET_COLUMN])
        writer.writerows(
            (int(identifier), format(float(value), ".17g"))
            for identifier, value in zip(ids, values, strict=True)
        )
    submitted = validate_submission_file(path, pd.Series(ids, name=ID_COLUMN))
    difference = float(np.max(np.abs(submitted[TARGET_COLUMN].to_numpy() - values)))
    if difference > 1e-12:
        raise RuntimeError("exported submission predictions changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "columns": list(submitted.columns),
        "rows": len(submitted),
        "max_abs_difference": difference,
    }


def _fresh_reload_predictions(
    output_dir: Path, test_path: Path, output_path: Path, *, timeout_seconds: float
) -> None:
    """Reconstruct all components through this importable module in a clean process."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--reload",
            "--output-dir",
            str(output_dir),
            "--test-path",
            str(test_path),
            "--output-path",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def reload_predictions(output_dir: Path, test_path: Path, output_path: Path) -> None:
    record = json.loads((output_dir / FEATURE_SPEC_NAME).read_text(encoding="utf-8"))
    original_spec = _spec(record["original"])
    teacher_spec = _spec(record["teacher"])
    context_spec = _spec(record["context"])
    hierarchical_spec = _spec(record["hierarchical"])
    test = pd.read_csv(test_path)
    original = test.loc[:, list(original_spec.feature_columns)].copy()
    teacher = lgb.Booster(model_file=record["teacher_model_path"])
    teacher_values = _predict_lightgbm(
        teacher, original, original_spec, record["category_levels"]["source_full"]
    )
    teacher_frame, _ = append_teacher_feature(original, teacher_values, original_spec)
    table = _load_export_profile_table(output_dir / PROFILE_TABLE_NAME, original_spec)
    base, _ = append_distance_profile_features(teacher_frame, teacher_spec, table)
    context_values = apply_route_maps(
        base,
        load_route_maps(output_dir / CONTEXT_MAPS_NAME),
    )
    hierarchical_values = apply_route_maps(
        base,
        load_route_maps(output_dir / HIERARCHICAL_MAPS_NAME),
    )
    context_frame = base.copy()
    hierarchical_frame = base.copy()
    for name in context_values:
        context_frame[name] = context_values[name]
    for name in hierarchical_values:
        hierarchical_frame[name] = hierarchical_values[name]
    context = _predict_lightgbm(
        lgb.Booster(model_file=record["context_model_path"]),
        context_frame,
        context_spec,
        record["category_levels"]["context_full"],
    )
    hierarchical = _predict_lightgbm(
        lgb.Booster(model_file=record["hierarchical_model_path"]),
        hierarchical_frame,
        hierarchical_spec,
        record["category_levels"]["hierarchical_full"],
    )
    catboost = CatBoostClassifier()
    catboost.load_model(record["catboost_model_path"])
    cat = np.asarray(
        catboost.predict_proba(prepare_catboost_frame(test, original_spec))[:, 1],
        dtype=np.float64,
    )
    candidate = route_pair_blend(cat, context, hierarchical)
    np.save(output_path, np.column_stack((cat, context, hierarchical, candidate)))


def _invalidate_export(
    metadata: dict[str, Any], csv_path: Path, reload_path: Path
) -> None:
    metadata["selection"] = None
    metadata["submission"] = None
    metadata["reload_check"] = None
    metadata["qualification"] = None
    metadata["current_fit"] = None
    csv_path.unlink(missing_ok=True)
    reload_path.unlink(missing_ok=True)


def export_submission(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the one approved full-training route-pair export."""
    project_dir = Path(__file__).resolve().parent
    data_dir = data_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / METADATA_NAME
    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "selection": None,
        "current_fit": None,
        "completed_fits": [],
        "fit_policy": (
            "full training rows; no eval_set, early stopping, callbacks or "
            "model-validation CV; target features cross-fitted without test labels"
        ),
        "qualification_note": (
            "direct refit from overlapping holdout screen; no new five-fold "
            "model confirmation performed"
        ),
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
    }
    _atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    submission_path = output_dir / SUBMISSION_NAME
    reload_path = output_dir / RELOAD_NAME
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = cached.f._artifact_snapshot(artifacts_dir, output_dir)
        input_before = cached.f._input_hashes(data_dir)
        source_before = _code_hashes(project_dir)
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
            raise RuntimeError("input row counts do not match the approved export")
        labels = target_array(train)
        qualification = _validate_qualification(
            artifacts_dir, train, labels, input_before
        )
        original_train = train.loc[:, list(spec.feature_columns)]
        original_test = test.loc[:, list(spec.feature_columns)]
        (
            teacher_model,
            teacher_path,
            source_levels,
            _competition_levels,
            _teacher_metadata,
        ) = _load_frozen_teacher(project_dir, artifacts_dir, spec)
        if teacher_model.current_iteration() != EXPECTED_TEACHER_ITERATIONS:
            raise RuntimeError("frozen teacher iteration count changed")
        teacher_train = _predict_lightgbm(
            teacher_model, original_train, spec, source_levels
        )
        teacher_test = _predict_lightgbm(
            teacher_model, original_test, spec, source_levels
        )
        cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        difference = float(np.max(np.abs(teacher_train - cache["probabilities"])))
        if difference > 1e-12:
            raise RuntimeError("frozen teacher train cache does not reproduce")
        catboost, catboost_path, _catboost_metadata = _validate_frozen_catboost(
            project_dir, artifacts_dir, spec
        )
        catboost_test = np.asarray(
            catboost.predict_proba(prepare_catboost_frame(test, spec))[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(catboost_test, len(test))
        baseline_path = (
            artifacts_dir
            / "distance_profile_export"
            / "submission_slowcat_teacher_distance_profile_50_50.csv"
        )
        if sha256_file(baseline_path) != ACCEPTED_PUBLIC_CSV_SHA256:
            raise RuntimeError(
                "accepted public baseline submission fingerprint changed"
            )
        validate_submission_file(baseline_path, test[ID_COLUMN])
        del catboost
        metadata.update(
            {
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "threads": MAX_THREADS,
                "python": sys.version,
                "dependencies": _versions(),
                "qualification": qualification,
                "teacher_model_path": str(teacher_path),
                "teacher_model_sha256": sha256_file(teacher_path),
                "teacher_iterations": EXPECTED_TEACHER_ITERATIONS,
                "teacher_train_prediction_check": {
                    "rows": len(train),
                    "max_abs_difference": difference,
                },
                "catboost_model_path": str(catboost_path),
                "catboost_model_sha256": sha256_file(catboost_path),
                "baseline_submission_path": str(baseline_path),
                "baseline_submission_sha256": sha256_file(baseline_path),
            }
        )
        profile = fit_distance_profile_table(original_train, spec)
        profile_path = output_dir / PROFILE_TABLE_NAME
        _save_export_profile_table(profile_path, profile)
        teacher_train_frame, teacher_spec = append_teacher_feature(
            original_train, teacher_train, spec
        )
        teacher_test_frame, _ = append_teacher_feature(
            original_test, teacher_test, spec
        )
        base_train, base_spec = append_distance_profile_features(
            teacher_train_frame, teacher_spec, profile
        )
        base_test, _ = append_distance_profile_features(
            teacher_test_frame, teacher_spec, profile
        )
        context_iterations = derive_refit_iterations(
            json.loads(
                (
                    artifacts_dir / CONTEXT_EXPERIMENT / "compare_metadata.json"
                ).read_text(encoding="utf-8")
            )
        )
        hierarchical_iterations = derive_refit_iterations(
            json.loads(
                (
                    artifacts_dir / HIERARCHICAL_EXPERIMENT / "compare_metadata.json"
                ).read_text(encoding="utf-8")
            )
        )
        if (context_iterations, hierarchical_iterations) != (219, 612):
            raise RuntimeError("announced fixed refit counts changed")
        metadata.update(
            {
                "refit_iterations": {
                    "context": context_iterations,
                    "hierarchical": hierarchical_iterations,
                },
                "profile_table_sha256": sha256_file(profile_path),
                "feature_specs": {
                    "original": _feature_spec(spec),
                    "teacher": _feature_spec(teacher_spec),
                    "base": _feature_spec(base_spec),
                },
                "files": {PROFILE_TABLE_NAME: sha256_file(profile_path)},
            }
        )
        _atomic_json(metadata_path, metadata)
        context_predictions, context_spec, context_levels, context_record = (
            _fit_student(
                name="context_student",
                kind="context",
                base_train=base_train,
                base_test=base_test,
                base_spec=base_spec,
                labels=labels,
                iterations=context_iterations,
                num_leaves=127,
                output_dir=output_dir,
                metadata=metadata,
                metadata_path=metadata_path,
            )
        )
        metadata["feature_specs"]["context"] = _feature_spec(context_spec)
        metadata["category_levels"] = {
            "source_full": source_levels,
            "context_full": context_levels,
        }
        _atomic_json(metadata_path, metadata)
        (
            hierarchical_predictions,
            hierarchical_spec,
            hierarchical_levels,
            hierarchical_record,
        ) = _fit_student(
            name="hierarchical_student",
            kind="hierarchical",
            base_train=base_train,
            base_test=base_test,
            base_spec=base_spec,
            labels=labels,
            iterations=hierarchical_iterations,
            num_leaves=31,
            output_dir=output_dir,
            metadata=metadata,
            metadata_path=metadata_path,
        )
        metadata["feature_specs"]["hierarchical"] = _feature_spec(hierarchical_spec)
        metadata["category_levels"]["hierarchical_full"] = hierarchical_levels
        feature_record = {
            "original": _feature_spec(spec),
            "teacher": _feature_spec(teacher_spec),
            "context": _feature_spec(context_spec),
            "hierarchical": _feature_spec(hierarchical_spec),
            "category_levels": metadata["category_levels"],
            "teacher_model_path": str(teacher_path),
            "context_model_path": str(output_dir / CONTEXT_MODEL_NAME),
            "hierarchical_model_path": str(output_dir / HIERARCHICAL_MODEL_NAME),
            "catboost_model_path": str(catboost_path),
        }
        _atomic_json(output_dir / FEATURE_SPEC_NAME, feature_record)
        metadata["files"][FEATURE_SPEC_NAME] = sha256_file(
            output_dir / FEATURE_SPEC_NAME
        )
        components = route_pair_blend(
            catboost_test, context_predictions, hierarchical_predictions
        )
        np.savez_compressed(
            output_dir / COMPONENTS_NAME,
            id=test[ID_COLUMN].to_numpy(dtype=np.int64),
            catboost=catboost_test,
            context=context_predictions,
            hierarchical=hierarchical_predictions,
            candidate=components,
        )
        metadata["files"][COMPONENTS_NAME] = sha256_file(output_dir / COMPONENTS_NAME)
        metadata["context_model"] = context_record
        metadata["hierarchical_model"] = hierarchical_record
        metadata["blend_components"] = {
            "weights": {"catboost": 0.5, "context": 0.25, "hierarchical": 0.25}
        }
        _atomic_json(metadata_path, metadata)
        submission = _write_submission(
            submission_path, test[ID_COLUMN].to_numpy(dtype=np.int64), components
        )
        del (
            teacher_model,
            base_train,
            base_test,
            teacher_train_frame,
            teacher_test_frame,
            original_train,
            original_test,
            train,
        )
        gc.collect()
        _fresh_reload_predictions(
            output_dir,
            data_dir / "test.csv",
            reload_path,
            timeout_seconds=max(
                0.1, INTERNAL_TIMEOUT_SECONDS - (time.perf_counter() - started)
            ),
        )
        reloaded = np.asarray(np.load(reload_path), dtype=np.float64)
        reload_path.unlink()
        reload_differences = {
            "catboost_max_abs": float(np.max(np.abs(catboost_test - reloaded[:, 0]))),
            "context_max_abs": float(
                np.max(np.abs(context_predictions - reloaded[:, 1]))
            ),
            "hierarchical_max_abs": float(
                np.max(np.abs(hierarchical_predictions - reloaded[:, 2]))
            ),
            "blend_max_abs": float(np.max(np.abs(components - reloaded[:, 3]))),
        }
        if max(reload_differences.values()) > 1e-12:
            raise RuntimeError(
                f"fresh-interpreter reload verification failed: {reload_differences}"
            )
        metadata.update(
            {
                "status": "completed",
                "selection": {
                    "qualifies": True,
                    "decision": "direct_route_pair_refit",
                    "requires_five_fold_confirmation": False,
                },
                "submission": submission,
                "reload_check": {
                    "fresh_interpreter": True,
                    "rows": len(test),
                    "max_abs_difference": reload_differences,
                    "temporary_path_removed": not reload_path.exists(),
                },
            }
        )
    except BaseException as error:
        caught = error
        metadata["status"] = (
            "timeout" if isinstance(error, RoutePairExportTimeout) else "failed"
        )
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            _invalidate_export(metadata, submission_path, reload_path)
        except OSError as cleanup_error:
            finalization_errors.append(cleanup_error)
    finally:
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                finalization_errors.append(error)
        if input_before is not None:
            try:
                metadata["input_sha256_after"] = cached.f._input_hashes(data_dir)
                if metadata["input_sha256_after"] != input_before:
                    finalization_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
            except BaseException as error:
                finalization_errors.append(error)
        if source_before is not None:
            try:
                metadata["source_sha256_after"] = _code_hashes(project_dir)
                if metadata["source_sha256_after"] != source_before:
                    finalization_errors.append(
                        RuntimeError("source fingerprint changed")
                    )
            except BaseException as error:
                finalization_errors.append(error)
        try:
            protected_after = cached.f._artifact_snapshot(artifacts_dir, output_dir)
            metadata["protected_artifact_sha256_after"] = protected_after
            metadata["protected_artifacts_unchanged"] = (
                protected_before is not None and protected_after == protected_before
            )
            if protected_before is not None and protected_after != protected_before:
                finalization_errors.append(
                    RuntimeError("pre-existing artifact changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        if finalization_errors:
            integrity = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["integrity_error"] = {
                "type": type(integrity).__name__,
                "message": str(integrity),
            }
            if caught is None:
                caught = integrity
                metadata["error"] = metadata["integrity_error"]
        metadata["current_fit"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = _output_size(output_dir)
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget = RuntimeError("output directory exceeded 100 MiB budget")
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["output_budget_error"] = {
                "type": type(budget).__name__,
                "message": str(budget),
            }
            if caught is None:
                caught = budget
                metadata["error"] = metadata["output_budget_error"]
        if metadata["status"] != "completed":
            try:
                _invalidate_export(metadata, submission_path, reload_path)
            except OSError as cleanup_error:
                metadata["cleanup_error"] = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
                if caught is None:
                    caught = cleanup_error
        try:
            cached._save_sized_metadata(metadata_path, metadata)
        except BaseException as persistence_error:
            metadata["status"] = "failed"
            metadata["metadata_persistence_error"] = {
                "type": type(persistence_error).__name__,
                "message": str(persistence_error),
            }
            if caught is None:
                caught = persistence_error
            try:
                _invalidate_export(metadata, submission_path, reload_path)
                cached._save_sized_metadata(metadata_path, metadata)
            except BaseException:
                pass  # The earlier incomplete record cannot authorize upload.
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    if args.reload:
        if (
            args.output_dir is None
            or args.test_path is None
            or args.output_path is None
        ):
            parser.error(
                "--reload requires --output-dir, --test-path, and --output-path"
            )
        reload_predictions(args.output_dir, args.test_path, args.output_path)
        return
    result = export_submission(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "submission": result.get("submission"),
                "output_dir": result["output_dir"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
