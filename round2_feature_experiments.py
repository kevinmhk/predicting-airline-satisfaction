"""Run the fixed conditional-feature batch and its conditional cached reweights."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

# ruff: noqa: E402
import numpy as np
import pandas as pd

import cached_route_blends as cached
import final_sprint_experiments as f
import round2_cached_experiments as r
from distance_profile_experiment import append_distance_profile_features
from source_teacher_experiment import append_teacher_feature
from train import (
    ID_COLUMN,
    MAX_THREADS,
    InputSpec,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)

BOARDING_ROUTE = "boarding_route"
TEACHER_ROUTE = "teacher_route"
JOINT_PASSENGER = "joint_passenger"
BASE_MODES = (BOARDING_ROUTE, TEACHER_ROUTE, JOINT_PASSENGER)
LIGHT60_SUFFIX = "_light60"
CACHED_MODES = tuple(f"{mode}{LIGHT60_SUFFIX}" for mode in BASE_MODES)
MODES = BASE_MODES + CACHED_MODES

CONDITIONAL_FEATURE_NAMES = {
    BOARDING_ROUTE: "boarding_route_target_mean",
    TEACHER_ROUTE: "teacher_route_target_mean",
    JOINT_PASSENGER: "joint_passenger_target_mean",
}
CONDITIONAL_COLUMNS = {
    BOARDING_ROUTE: "(Flight Distance, Online boarding)",
    TEACHER_ROUTE: "(Flight Distance, teacherBin)",
    JOINT_PASSENGER: "(Flight Distance, Class, Type of Travel, Customer Type)",
}
LATEST_BASELINE_AUC = dict(r.LATEST_BASELINE_AUC)
SPLIT_SEEDS = tuple(r.SPLIT_SEEDS)
INTERNAL_SEED = 42
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_BASE_FEATURES = (39, 35, 4)
EXPECTED_STUDENT_FEATURES = (44, 40, 4)
SMOOTHING = 20.0
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
VALIDATION_COLUMNS = r.VALIDATION_COLUMNS


class _Round2FeatureTimeout(TimeoutError):
    """Raised when one fixed round-two candidate reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise _Round2FeatureTimeout("round-two feature experiment exceeded its deadline")


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _tuple_key(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _teacher_bin(value: Any) -> int:
    """Map a frozen teacher probability to one of five fixed bins."""
    if pd.isna(value):
        raise ValueError("source teacher probability must not be missing")
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("source teacher probability must be finite in [0, 1]")
    return min(4, max(0, int(np.floor(probability * 5.0))))


def _conditional_keys(features: pd.DataFrame, mode: str) -> tuple[list[str], list[str]]:
    distances = [
        f.canonical_distance_key(value) for value in features["Flight Distance"]
    ]
    if mode == BOARDING_ROUTE:
        ratings = [
            f.canonical_distance_key(value) for value in features["Online boarding"]
        ]
        parents = ratings
        keys = [
            _tuple_key((distance, rating))
            for distance, rating in zip(distances, ratings, strict=True)
        ]
    elif mode == TEACHER_ROUTE:
        bins = [
            str(_teacher_bin(value)) for value in features["source_teacher_probability"]
        ]
        parents = bins
        keys = [
            _tuple_key((distance, teacher_bin))
            for distance, teacher_bin in zip(distances, bins, strict=True)
        ]
    elif mode == JOINT_PASSENGER:
        categories = {
            column: [f._canonical_class_key(value) for value in features[column]]
            for column in ("Class", "Type of Travel", "Customer Type")
        }
        parents = [
            _tuple_key(
                (
                    categories["Class"][row],
                    categories["Type of Travel"][row],
                    categories["Customer Type"][row],
                )
            )
            for row in range(len(features))
        ]
        keys = [
            _tuple_key(
                (
                    distances[row],
                    categories["Class"][row],
                    categories["Type of Travel"][row],
                    categories["Customer Type"][row],
                )
            )
            for row in range(len(features))
        ]
    else:
        raise ValueError(f"unknown conditional feature mode: {mode}")
    return keys, parents


def fit_conditional_feature(
    features: pd.DataFrame,
    spec: InputSpec,
    fit_labels: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    holdout_indices: np.ndarray,
    mode: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Cross-fit one fixed conditional target mean from inner-fit labels only."""
    if mode not in BASE_MODES:
        raise ValueError(f"unknown conditional feature mode: {mode}")
    labels, fit_indices, validation_indices, holdout_indices = (
        f._validate_new_route_encoding_inputs(
            features,
            spec,
            fit_labels,
            fit_indices,
            validation_indices,
            holdout_indices,
        )
    )
    keys, parents = _conditional_keys(features, mode)
    labels_by_row = np.full(len(features), np.nan, dtype=np.float64)
    labels_by_row[fit_indices] = labels
    transformed = np.full(len(features), np.nan, dtype=np.float64)

    def map_rows(statistics: np.ndarray, queries: np.ndarray) -> None:
        prior = float(np.mean(labels_by_row[statistics]))
        key_sums, key_counts = f._smoothed_group_stats(keys, labels_by_row, statistics)
        parent_sums, parent_counts = f._smoothed_group_stats(
            parents, labels_by_row, statistics
        )
        query_keys = pd.Series(keys, dtype=object).iloc[queries]
        query_parents = pd.Series(parents, dtype=object).iloc[queries]
        sums = query_keys.map(key_sums).fillna(0.0).to_numpy(dtype=np.float64)
        counts = query_keys.map(key_counts).fillna(0.0).to_numpy(dtype=np.float64)
        parent_prior = query_parents.map(parent_sums / parent_counts).fillna(prior)
        transformed[queries] = (sums + SMOOTHING * parent_prior.to_numpy()) / (
            counts + SMOOTHING
        )

    folds = f._route_fold_queries(fit_indices, labels)
    for statistics, queries in folds:
        map_rows(statistics, queries)
    map_rows(fit_indices, np.concatenate((validation_indices, holdout_indices)))
    validate_probabilities(transformed, len(features))
    fold_membership = f._route_fold_membership(folds)
    metadata = {
        "feature": CONDITIONAL_FEATURE_NAMES[mode],
        "key_policy": mode,
        "key_definition": CONDITIONAL_COLUMNS[mode],
        "smooth": SMOOTHING,
        "fit_rows": len(fit_indices),
        "fit_row_fingerprint": f._array_fingerprint(fit_indices),
        "fold_membership": fold_membership,
        "fold_hashes": [
            {
                "fit_sha256": item["fit_sha256"],
                "query_sha256": item["query_sha256"],
            }
            for item in fold_membership
        ],
        "prior_policy": "fit-label parent prevalence",
        "unseen_key_policy": "parent prevalence",
        "unseen_parent_policy": "global fit-label prevalence",
        "source": "inner-fit satisfaction labels only",
        "timing": "cross-fit before both student fits; no outer-holdout labels",
        "global_fit_prevalence": float(np.mean(labels)),
    }
    return (
        pd.Series(
            transformed,
            index=features.index,
            name=CONDITIONAL_FEATURE_NAMES[mode],
        ),
        metadata,
    )


def append_conditional_feature(
    features: pd.DataFrame,
    spec: InputSpec,
    values: pd.Series | np.ndarray,
    mode: str,
) -> tuple[pd.DataFrame, InputSpec]:
    """Append one validated conditional feature without changing prior columns."""
    if mode not in BASE_MODES:
        raise ValueError(f"unknown conditional feature mode: {mode}")
    f._validate_feature_frame(features, spec)
    name = CONDITIONAL_FEATURE_NAMES[mode]
    if name in features.columns:
        raise ValueError(f"feature already exists: {name}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(features):
        raise ValueError("conditional feature values do not match the feature frame")
    validate_probabilities(array, len(features))
    result = features.copy()
    result[name] = array
    extended = InputSpec(
        spec.feature_columns + (name,),
        spec.numeric_columns + (name,),
        spec.categorical_columns,
    )
    return result, extended


def _base_blend(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> np.ndarray:
    return 0.5 * catboost + 0.25 * context + 0.25 * hierarchical


def _light60_blend(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> np.ndarray:
    return 0.4 * catboost + 0.3 * context + 0.3 * hierarchical


def _own_source_hashes() -> dict[str, str]:
    hashes = r._own_source_hashes()
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _save_component(
    path: Path, row_index: np.ndarray, ids: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    np.savez_compressed(path, row_index=row_index, id=ids, probability=probabilities)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(row_index),
        "row_index_sha256": f._array_fingerprint(row_index),
        "id_sha256": f._array_fingerprint(ids),
    }


def _validation_frame(
    row_index: np.ndarray,
    ids: np.ndarray,
    catboost: np.ndarray,
    context: np.ndarray,
    hierarchical: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": row_index,
            ID_COLUMN: ids,
            "catboost_probability": catboost,
            "context_probability": context,
            "hierarchical_probability": hierarchical,
            "baseline_probability": baseline,
            "candidate_probability": candidate,
        }
    )


def _assert_prediction_schema(
    frame: pd.DataFrame, train: pd.DataFrame, holdout: np.ndarray
) -> None:
    if list(frame.columns) != list(VALIDATION_COLUMNS):
        raise RuntimeError("round-two validation schema changed")
    expected_ids = train.iloc[holdout][ID_COLUMN].to_numpy(dtype=np.int64)
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), holdout):
        raise RuntimeError("round-two validation rows changed")
    if not np.array_equal(frame[ID_COLUMN].to_numpy(dtype=np.int64), expected_ids):
        raise RuntimeError("round-two validation IDs changed")
    for column in VALIDATION_COLUMNS[2:]:
        validate_probabilities(frame[column].to_numpy(dtype=np.float64), len(frame))


def _read_prediction_npz(
    record: dict[str, Any], path: Path, train: pd.DataFrame, holdout: np.ndarray
) -> np.ndarray:
    if record.get("path") != str(path) or record.get("sha256") != sha256_file(path):
        raise RuntimeError(f"component prediction cache changed: {path}")
    with np.load(path) as saved:
        if set(saved.files) != {"row_index", "id", "probability"}:
            raise RuntimeError(f"component prediction fields changed: {path}")
        rows = saved["row_index"].astype(np.int64, copy=False)
        ids = saved["id"].astype(np.int64, copy=False)
        probabilities = saved["probability"].astype(np.float64, copy=False)
    expected_ids = train.iloc[holdout][ID_COLUMN].to_numpy(dtype=np.int64)
    if not np.array_equal(rows, holdout) or not np.array_equal(ids, expected_ids):
        raise RuntimeError(f"component prediction IDs changed: {path}")
    validate_probabilities(probabilities, len(holdout))
    if record.get("row_index_sha256") != f._array_fingerprint(rows):
        raise RuntimeError(f"component row fingerprint changed: {path}")
    if record.get("id_sha256") != f._array_fingerprint(ids):
        raise RuntimeError(f"component ID fingerprint changed: {path}")
    return probabilities


def _persist_metadata(path: Path, metadata: dict[str, Any], output_dir: Path) -> None:
    metadata["output_bytes"] = f._output_size(output_dir)
    cached._save_sized_metadata(path, metadata)
    if metadata["output_bytes"] != f._output_size(output_dir):
        raise RuntimeError("round-two metadata byte count did not stabilize")


def _common_preflight(
    data_dir: Path,
    artifacts_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    InputSpec,
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[str, str],
    dict[str, str],
]:
    input_before = f._input_hashes(data_dir)
    source_before = _own_source_hashes()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    spec = validate_inputs(train, test)
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("training row count does not match the approved screen")
    y = target_array(train)
    prior = f._load_prior_lightgbm(artifacts_dir, input_before)
    teacher_cache = f._verify_teacher_cache(artifacts_dir, train, spec, input_before)
    profile_path = artifacts_dir / "distance_profile_feature" / "compare_metadata.json"
    profile_metadata = json.loads(profile_path.read_text(encoding="utf-8"))
    f._validate_prior_profile(profile_metadata, input_before, spec)
    incumbent_refs = r._validate_incumbent(artifacts_dir, input_before)
    metadata.update(
        {
            "python": sys.version,
            "threads": MAX_THREADS,
            "input_sha256_before": input_before,
            "source_sha256_before": source_before,
            "feature_spec_original": _feature_spec(spec),
            "feature_spec": _feature_spec(spec),
            "split_seeds": list(SPLIT_SEEDS),
            "internal_seed": INTERNAL_SEED,
            "early_stopping": {"metric": "auc", "rounds": 100},
            "baseline_reference": {
                "artifact": "cached_route_blends_route_pair_average",
                "auc_by_seed": LATEST_BASELINE_AUC,
                "weights": {"catboost": 0.5, "context": 0.25, "hierarchical": 0.25},
            },
            "reused_artifacts": {
                "source_teacher_metadata": sha256_file(teacher_cache["metadata_path"]),
                "distance_profile_metadata": sha256_file(profile_path),
                "latest_incumbent_metadata": sha256_file(
                    Path(next(iter(incumbent_refs.values()))["metadata_path"])
                ),
            },
        }
    )
    save_json(metadata_path, metadata)
    return (
        train,
        y,
        spec,
        prior,
        profile_metadata,
        incumbent_refs,
        input_before,
        source_before,
    )


def _base_features_for_seed(
    train: pd.DataFrame,
    spec: InputSpec,
    teacher_cache: dict[str, Any],
    profile_metadata: dict[str, Any],
    artifacts_dir: Path,
    seed: int,
    inner_fit: np.ndarray,
) -> tuple[pd.DataFrame, InputSpec]:
    original = train.loc[:, list(spec.feature_columns)]
    teacher_features, teacher_spec = append_teacher_feature(
        original, teacher_cache["probabilities"], spec
    )
    profile_record = f._profile_cache_record(profile_metadata, seed, inner_fit)
    profile_path = (
        artifacts_dir / "distance_profile_feature" / f"distance_profile_seed{seed}.npz"
    )
    table = f._load_profile_table(
        profile_record, profile_path, original, spec, inner_fit
    )
    base_features, base_spec = append_distance_profile_features(
        teacher_features, teacher_spec, table
    )
    actual = (
        len(base_spec.feature_columns),
        len(base_spec.numeric_columns),
        len(base_spec.categorical_columns),
    )
    if actual != EXPECTED_BASE_FEATURES:
        raise RuntimeError("round-two base feature schema is not 39/35/4")
    return base_features, base_spec


def _run_base(
    mode: str,
    artifacts_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    spec: InputSpec,
    prior: dict[str, Any],
    profile_metadata: dict[str, Any],
    incumbent_refs: dict[int, dict[str, Any]],
    input_before: dict[str, str],
) -> None:
    teacher_cache = f._verify_teacher_cache(artifacts_dir, train, spec, input_before)
    source_hashes = f._source_hashes()
    ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
    baseline_auc: dict[int, float] = {}
    candidate_auc: dict[int, float] = {}
    for seed in SPLIT_SEEDS:
        _, holdout, inner_fit, inner_validation, split_record = f._load_split_arrays(
            artifacts_dir, train, y, prior, seed
        )
        assembled, checked_holdout, baseline_metrics, lineage = r._load_seed_components(
            r.PAIR_LIGHT60,
            artifacts_dir,
            train,
            y,
            prior,
            input_before,
            source_hashes,
            incumbent_refs,
            seed,
        )
        if not np.array_equal(holdout, checked_holdout):
            raise RuntimeError("round-two split changed while loading incumbent")
        base_features, base_spec = _base_features_for_seed(
            train,
            spec,
            teacher_cache,
            profile_metadata,
            artifacts_dir,
            seed,
            inner_fit,
        )
        extra, extra_metadata = fit_conditional_feature(
            base_features,
            base_spec,
            y[inner_fit],
            inner_fit,
            inner_validation,
            holdout,
            mode,
        )
        context = f.fit_route_target_encoding(
            base_features,
            base_spec,
            y[inner_fit],
            inner_fit,
            inner_validation,
            holdout,
            include_context=True,
        )
        context_features, context_spec = append_conditional_feature(
            context.features, context.spec, extra, mode
        )
        del context
        gc.collect()
        if (
            len(context_spec.feature_columns),
            len(context_spec.numeric_columns),
            len(context_spec.categorical_columns),
        ) != EXPECTED_STUDENT_FEATURES:
            raise RuntimeError("round-two context schema is not 44/40/4")
        split_name = "outer" if seed == 42 else "confirmation"
        metadata.setdefault("recorded_splits", {})[split_name] = split_record
        metadata["current_fit"] = {
            "split_seed": seed,
            "status": "running",
            "inner_fit_rows": len(inner_fit),
            "inner_validation_rows": len(inner_validation),
            "outer_holdout_rows": len(holdout),
            "outer_holdout_used_for_training_or_early_stopping": False,
            "conditional_feature": extra_metadata,
        }
        save_json(metadata_path, metadata)
        component_records: dict[str, Any] = {}
        fit_started = time.perf_counter()
        context_predictions, context_record = f._fit_student(
            f.ROUTE_CONTEXT_TARGET_127,
            context_features,
            context_spec,
            y,
            holdout,
            inner_fit,
            inner_validation,
            metadata,
            metadata_path,
            seed,
        )
        context_path = (
            artifacts_dir / f"round2_{mode}" / f"predictions_context_seed{seed}.npz"
        )
        component_records["context"] = context_record | {
            "fit_mode": f.ROUTE_CONTEXT_TARGET_127,
            "feature_spec": _feature_spec(context_spec),
            "prediction": _save_component(
                context_path, holdout, ids[holdout], context_predictions
            ),
        }
        metadata.setdefault("component_fits", {}).setdefault(str(seed), {})[
            "context"
        ] = component_records["context"]
        metadata["fits"] = int(metadata.get("fits", 0)) + 1
        metadata["completed_fits"].append(f"context_seed_{seed}")
        metadata["training_seconds"] = round(
            float(metadata.get("training_seconds", 0.0))
            + float(context_record.get("fit_seconds", 0.0)),
            3,
        )
        metadata["current_fit"] = None
        save_json(metadata_path, metadata)
        del context_features
        gc.collect()

        hierarchical = f.fit_hierarchical_route_encoding(
            base_features,
            base_spec,
            y[inner_fit],
            inner_fit,
            inner_validation,
            holdout,
        )
        hierarchical_features, hierarchical_spec = append_conditional_feature(
            hierarchical.features, hierarchical.spec, extra, mode
        )
        del hierarchical
        gc.collect()
        if (
            len(hierarchical_spec.feature_columns),
            len(hierarchical_spec.numeric_columns),
            len(hierarchical_spec.categorical_columns),
        ) != EXPECTED_STUDENT_FEATURES:
            raise RuntimeError("round-two hierarchical schema is not 44/40/4")
        hierarchical_predictions, hierarchical_record = f._fit_student(
            f.ROUTE_HIERARCHICAL,
            hierarchical_features,
            hierarchical_spec,
            y,
            holdout,
            inner_fit,
            inner_validation,
            metadata,
            metadata_path,
            seed,
        )
        hierarchical_path = (
            artifacts_dir
            / f"round2_{mode}"
            / f"predictions_hierarchical_seed{seed}.npz"
        )
        component_records["hierarchical"] = hierarchical_record | {
            "fit_mode": f.ROUTE_HIERARCHICAL,
            "feature_spec": _feature_spec(hierarchical_spec),
            "prediction": _save_component(
                hierarchical_path, holdout, ids[holdout], hierarchical_predictions
            ),
        }
        metadata.setdefault("component_fits", {}).setdefault(str(seed), {})[
            "hierarchical"
        ] = component_records["hierarchical"]
        metadata["fits"] = int(metadata.get("fits", 0)) + 1
        metadata["completed_fits"].append(f"hierarchical_seed_{seed}")
        metadata["training_seconds"] = round(
            float(metadata.get("training_seconds", 0.0))
            + float(hierarchical_record.get("fit_seconds", 0.0)),
            3,
        )
        metadata["current_fit"] = None
        save_json(metadata_path, metadata)

        catboost = assembled["catboost_probability"].to_numpy(dtype=np.float64)
        baseline = assembled["baseline_probability"].to_numpy(dtype=np.float64)
        candidate = _base_blend(catboost, context_predictions, hierarchical_predictions)
        candidate_metrics = f._metrics(y[holdout], candidate)
        baseline_auc[seed] = baseline_metrics["roc_auc"]
        candidate_auc[seed] = candidate_metrics["roc_auc"]
        validation = _validation_frame(
            holdout,
            ids[holdout],
            catboost,
            context_predictions,
            hierarchical_predictions,
            baseline,
            candidate,
        )
        _assert_prediction_schema(validation, train, holdout)
        validation_path = (
            artifacts_dir / f"round2_{mode}" / f"validation_{mode}_seed{seed}.csv"
        )
        validation.to_csv(validation_path, index=False, float_format="%.17g")
        metadata["validation_results"].append(
            {
                "split_seed": seed,
                "split": split_record,
                "feature_spec_context": _feature_spec(context_spec),
                "feature_spec_hierarchical": _feature_spec(hierarchical_spec),
                "conditional_feature": extra_metadata,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "training_seconds": round(
                    context_record["fit_seconds"] + hierarchical_record["fit_seconds"],
                    3,
                ),
                "elapsed_components_seconds": round(
                    time.perf_counter() - fit_started, 3
                ),
                "fits": 2,
                "validation_predictions_path": str(validation_path),
                "validation_predictions_sha256": sha256_file(validation_path),
                "component_predictions": component_records,
                "lineage": lineage,
            }
        )
        save_json(metadata_path, metadata)
        del (
            hierarchical_predictions,
            hierarchical_features,
            context_predictions,
            base_features,
            extra,
            validation,
        )
        gc.collect()
    metadata["selection"] = f.final_sprint_gate(mode, baseline_auc, candidate_auc)


def _validate_base_trial(
    base_mode: str,
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    prior: dict[str, Any],
    input_before: dict[str, str],
    source_before: dict[str, str],
    incumbent_refs: dict[int, dict[str, Any]],
) -> tuple[
    list[tuple[int, pd.DataFrame, np.ndarray, dict[str, float], dict[str, Any]]],
    dict[str, Any],
]:
    base_dir = artifacts_dir / f"round2_{base_mode}"
    metadata_path = base_dir / "compare_metadata.json"
    base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if base_metadata.get("status") != "completed":
        raise RuntimeError("conditional feature trial is not completed")
    if (
        base_metadata.get("input_sha256_before") != input_before
        or base_metadata.get("input_sha256_after") != input_before
    ):
        raise RuntimeError("conditional feature trial inputs changed")
    if (
        base_metadata.get("source_sha256_before") != source_before
        or base_metadata.get("source_sha256_after") != source_before
    ):
        raise RuntimeError("conditional feature trial source fingerprints changed")
    selection = base_metadata.get("selection")
    if not isinstance(selection, dict) or selection.get("qualifies") is True:
        raise RuntimeError("conditional feature trial passed; skip its cached reweight")
    if int(base_metadata.get("fits", 0)) != 4:
        raise RuntimeError("conditional feature trial did not complete four fits")
    source_hashes = f._source_hashes()
    checked: list[
        tuple[int, pd.DataFrame, np.ndarray, dict[str, float], dict[str, Any]]
    ] = []
    for seed in SPLIT_SEEDS:
        _, holdout, _, _, _ = f._load_split_arrays(artifacts_dir, train, y, prior, seed)
        records = [
            item
            for item in base_metadata.get("validation_results", [])
            if item.get("split_seed") == seed
        ]
        if len(records) != 1:
            raise RuntimeError("conditional feature trial seed record is missing")
        record = records[0]
        gain = float(record.get("auc_gain_vs_baseline", 0.0))
        if gain <= 0.0:
            raise RuntimeError(
                "conditional feature trial was harmed; skip its cached reweight"
            )
        path = Path(record["validation_predictions_path"])
        if (
            path.resolve()
            != (base_dir / f"validation_{base_mode}_seed{seed}.csv").resolve()
        ):
            raise RuntimeError("conditional feature validation path changed")
        if record.get("validation_predictions_sha256") != sha256_file(path):
            raise RuntimeError("conditional feature validation CSV changed")
        frame = pd.read_csv(path, float_precision="round_trip")
        _assert_prediction_schema(frame, train, holdout)
        catboost = frame["catboost_probability"].to_numpy(dtype=np.float64)
        context = frame["context_probability"].to_numpy(dtype=np.float64)
        hierarchical = frame["hierarchical_probability"].to_numpy(dtype=np.float64)
        recomputed = _base_blend(catboost, context, hierarchical)
        recomputed_metrics = f._metrics(y[holdout], recomputed)
        if not np.isclose(
            recomputed_metrics["roc_auc"],
            record["candidate"]["roc_auc"],
            rtol=0.0,
            atol=1e-12,
        ) or not np.isclose(
            recomputed_metrics["log_loss"],
            record["candidate"]["log_loss"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("conditional feature base blend metrics changed")
        if not np.allclose(
            frame["candidate_probability"].to_numpy(dtype=np.float64),
            recomputed,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("conditional feature candidate arithmetic changed")
        for name, suffix in (
            (
                "context",
                [
                    f.FLIGHT_DISTANCE_TARGET,
                    f.FLIGHT_DISTANCE_CLASS_TARGET,
                    f.FLIGHT_DISTANCE_TRAVEL_TARGET,
                    f.FLIGHT_DISTANCE_CUSTOMER_TARGET,
                    CONDITIONAL_FEATURE_NAMES[base_mode],
                ],
            ),
            (
                "hierarchical",
                [
                    f.FLIGHT_DISTANCE_HIERARCHICAL,
                    f.FLIGHT_DISTANCE_CLASS_HIERARCHICAL,
                    f.FLIGHT_DISTANCE_TRAVEL_HIERARCHICAL,
                    f.FLIGHT_DISTANCE_CUSTOMER_HIERARCHICAL,
                    CONDITIONAL_FEATURE_NAMES[base_mode],
                ],
            ),
        ):
            feature_spec = record.get("feature_spec_" + name, {})
            if (
                len(feature_spec.get("feature_columns", [])) != 44
                or len(feature_spec.get("numeric_columns", [])) != 40
                or feature_spec.get("categorical_columns")
                != ["Gender", "Customer Type", "Type of Travel", "Class"]
                or feature_spec.get("feature_columns", [])[-5:] != suffix
                or feature_spec.get("numeric_columns", [])[-5:] != suffix
            ):
                raise RuntimeError("conditional feature student schema changed")
        components = base_metadata.get("component_fits", {}).get(str(seed), {})
        for name in ("context", "hierarchical"):
            component = components.get(name, {}).get("prediction")
            if not isinstance(component, dict):
                raise RuntimeError("conditional feature component record is missing")
            expected_mode = (
                f.ROUTE_CONTEXT_TARGET_127
                if name == "context"
                else f.ROUTE_HIERARCHICAL
            )
            if (
                components[name].get("settings")
                != f.build_trial_classifier(expected_mode).get_params()
            ):
                raise RuntimeError("cached conditional model settings changed")
            probabilities = _read_prediction_npz(
                component,
                base_dir / f"predictions_{name}_seed{seed}.npz",
                train,
                holdout,
            )
            if not np.allclose(
                probabilities, frame[name + "_probability"], rtol=0, atol=1e-15
            ):
                raise RuntimeError("cached component CSV/NPZ probabilities differ")
        assembled, checked_holdout, baseline_metrics, lineage = r._load_seed_components(
            r.PAIR_LIGHT60,
            artifacts_dir,
            train,
            y,
            prior,
            input_before,
            source_hashes,
            incumbent_refs,
            seed,
        )
        if not np.array_equal(holdout, checked_holdout):
            raise RuntimeError("conditional feature holdout changed")
        if not np.isclose(
            baseline_metrics["roc_auc"],
            record["baseline"]["roc_auc"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("conditional feature incumbent baseline changed")
        if not np.allclose(
            frame["baseline_probability"].to_numpy(dtype=np.float64),
            assembled["baseline_probability"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("conditional feature incumbent probabilities changed")
        recomputed_gain = recomputed_metrics["roc_auc"] - baseline_metrics["roc_auc"]
        if recomputed_gain <= 0 or not np.isclose(
            recomputed_gain, gain, rtol=0, atol=1e-14
        ):
            raise RuntimeError("cached conditional positive-gain criterion changed")
        checked.append((seed, frame, holdout, baseline_metrics, lineage))
    expected_gate = f.final_sprint_gate(
        base_mode,
        {seed: baseline["roc_auc"] for seed, _, _, baseline, _ in checked},
        {
            seed: f._metrics(y[hold], frame["candidate_probability"].to_numpy())[
                "roc_auc"
            ]
            for seed, frame, hold, _, _ in checked
        },
    )
    if expected_gate != selection:
        raise RuntimeError("cached conditional gate changed")
    return checked, base_metadata


def _run_cached_reweight(
    mode: str,
    artifacts_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    prior: dict[str, Any],
    input_before: dict[str, str],
    source_before: dict[str, str],
    incumbent_refs: dict[int, dict[str, Any]],
) -> None:
    base_mode = mode.removesuffix(LIGHT60_SUFFIX)
    checked, base_metadata = _validate_base_trial(
        base_mode,
        artifacts_dir,
        train,
        y,
        prior,
        input_before,
        source_before,
        incumbent_refs,
    )
    baseline_auc: dict[int, float] = {}
    candidate_auc: dict[int, float] = {}
    metadata["base_feature_trial"] = {
        "mode": base_mode,
        "metadata_path": str(
            artifacts_dir / f"round2_{base_mode}" / "compare_metadata.json"
        ),
        "metadata_sha256": sha256_file(
            artifacts_dir / f"round2_{base_mode}" / "compare_metadata.json"
        ),
        "selection": base_metadata["selection"],
    }
    for seed, frame, holdout, baseline_metrics, lineage in checked:
        record = next(
            item
            for item in base_metadata["validation_results"]
            if item["split_seed"] == seed
        )
        catboost = frame["catboost_probability"].to_numpy(dtype=np.float64)
        context = frame["context_probability"].to_numpy(dtype=np.float64)
        hierarchical = frame["hierarchical_probability"].to_numpy(dtype=np.float64)
        baseline = frame["baseline_probability"].to_numpy(dtype=np.float64)
        candidate = _light60_blend(catboost, context, hierarchical)
        candidate_metrics = f._metrics(y[holdout], candidate)
        baseline_auc[seed] = baseline_metrics["roc_auc"]
        candidate_auc[seed] = candidate_metrics["roc_auc"]
        validation = _validation_frame(
            holdout,
            frame[ID_COLUMN].to_numpy(dtype=np.int64),
            catboost,
            context,
            hierarchical,
            baseline,
            candidate,
        )
        _assert_prediction_schema(validation, train, holdout)
        path = artifacts_dir / f"round2_{mode}" / f"validation_{mode}_seed{seed}.csv"
        validation.to_csv(path, index=False, float_format="%.17g")
        metadata["validation_results"].append(
            {
                "split_seed": seed,
                "baseline": baseline_metrics,
                "base_candidate": record["candidate"],
                "candidate": candidate_metrics,
                "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "training_seconds": 0.0,
                "fits": 0,
                "validation_predictions_path": str(path),
                "validation_predictions_sha256": sha256_file(path),
                "lineage": lineage,
                "base_feature_validation_path": record["validation_predictions_path"],
                "base_component_predictions": {
                    name: base_metadata["component_fits"][str(seed)][name]["prediction"]
                    for name in ("context", "hierarchical")
                },
            }
        )
        save_json(metadata_path, metadata)
    metadata["selection"] = f.final_sprint_gate(mode, baseline_auc, candidate_auc)


def run_experiment(mode: str, data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run one fixed feature or conditional cached-reweight choice."""
    if mode not in MODES:
        raise ValueError(f"unknown round-two mode: {mode}")
    started = time.perf_counter()
    output_dir = artifacts_dir / f"round2_{mode}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "experiment": mode,
        "mode_definition": {
            "base_feature": mode
            if mode in BASE_MODES
            else mode.removesuffix(LIGHT60_SUFFIX),
            "conditional": mode in BASE_MODES,
            "weights": {"catboost": 0.5, "context": 0.25, "hierarchical": 0.25}
            if mode in BASE_MODES
            else {"catboost": 0.4, "context": 0.3, "hierarchical": 0.3},
        },
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "completed_fits": [],
        "component_fits": {},
        "fits": 0,
        "training_seconds": 0.0,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "output_dir": str(output_dir),
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    integrity_errors: list[BaseException] = []
    cleanup_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = f._artifact_snapshot(artifacts_dir, output_dir)
        metadata["protected_artifact_sha256_before"] = protected_before
        save_json(metadata_path, metadata)
        (
            train,
            y,
            spec,
            prior,
            profile_metadata,
            incumbent_refs,
            input_before,
            source_before,
        ) = _common_preflight(data_dir, artifacts_dir, metadata, metadata_path)
        if mode in BASE_MODES:
            _run_base(
                mode,
                artifacts_dir,
                metadata,
                metadata_path,
                train,
                y,
                spec,
                prior,
                profile_metadata,
                incumbent_refs,
                input_before,
            )
        else:
            _run_cached_reweight(
                mode,
                artifacts_dir,
                metadata,
                metadata_path,
                train,
                y,
                prior,
                input_before,
                source_before,
                incumbent_refs,
            )
    except BaseException as error:
        caught = error
        metadata["error"] = _error_record(error)
    finally:
        metadata["current_fit"] = None
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                cleanup_errors.append(error)
        if input_before is not None:
            try:
                metadata["input_sha256_after"] = f._input_hashes(data_dir)
                if metadata["input_sha256_after"] != input_before:
                    integrity_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if source_before is not None:
            try:
                metadata["source_sha256_after"] = _own_source_hashes()
                if metadata["source_sha256_after"] != source_before:
                    integrity_errors.append(RuntimeError("source fingerprint changed"))
            except BaseException as error:
                integrity_errors.append(error)
        if protected_before is not None:
            try:
                protected_after = f._artifact_snapshot(artifacts_dir, output_dir)
                metadata["protected_artifact_sha256_after"] = protected_after
                metadata["protected_artifacts_unchanged"] = (
                    protected_after == protected_before
                )
                if protected_after != protected_before:
                    integrity_errors.append(
                        RuntimeError("pre-existing artifact fingerprint changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if integrity_errors:
            metadata["integrity_error"] = _error_record(
                RuntimeError("; ".join(str(error) for error in integrity_errors))
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = RuntimeError("integrity checks failed")
                metadata["error"] = _error_record(caught)
        if cleanup_errors:
            metadata["cleanup_error"] = _error_record(
                RuntimeError("; ".join(str(error) for error in cleanup_errors))
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = RuntimeError("cleanup checks failed")
                metadata["error"] = _error_record(caught)
        elif caught is not None:
            metadata["status"] = (
                "timeout" if isinstance(caught, _Round2FeatureTimeout) else "failed"
            )
            metadata["selection"] = None
        else:
            metadata["status"] = "completed"
        metadata["elapsed_wall_seconds"] = round(time.perf_counter() - started, 3)
        if metadata["status"] != "completed":
            metadata["selection"] = None
        try:
            _persist_metadata(metadata_path, metadata, output_dir)
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["metadata_persistence_error"] = _error_record(error)
            if caught is None:
                caught = error
                metadata["error"] = _error_record(error)
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=MODES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.experiment, args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
