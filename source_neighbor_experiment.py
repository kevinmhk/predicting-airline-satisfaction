"""Run the fixed source-neighbour feature two-holdout diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

# Keep scientific imports after the thread limits are set.
# ruff: noqa: E402
import numpy as np
import pandas as pd

from distance_profile_experiment import (
    DISTANCE_COLUMN,
    append_distance_profile_features,
    fit_distance_profile_table,
)
from five_fold_cv import _metrics
from lightgbm_capacity import (
    _array_fingerprint,
    _artifact_snapshot,
    _input_hashes,
    _load_components,
    _load_split_arrays,
)
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _fit_callbacks,
    _fit_with_inner_validation,
    build_lightgbm_classifier,
    prepare_lightgbm_frame,
)
from source_augmentation import clean_source_rows
from source_teacher_cv import _verify_teacher_cache
from source_teacher_experiment import (
    _load_prior_lightgbm,
    append_teacher_feature,
)
from train import (
    ID_COLUMN,
    MAX_THREADS,
    TARGET_COLUMN,
    InputSpec,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)
from weighted_blend_experiment import weighted_blend_predictions

SOURCE_NEIGHBOR_FEATURE = "source_neighbor_probability"
OUTPUT_NAME = "source_neighbor_feature"
SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
SPLIT_SEEDS = tuple(SPLIT_FILES)
INTERNAL_SEED = 42
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_CLEAN_SOURCE_ROWS = 129859
EXPECTED_FEATURE_COLUMNS = 21
EXPECTED_NUMERIC_COLUMNS = 17
EXPECTED_CATEGORICAL_COLUMNS = 4
MAX_PAIR_ELEMENTS = 250_000
NEIGHBOR_QUERY_BLOCK_SIZE = 1_024
PROGRESS_INTERVAL = 10_000
INTERNAL_TIMEOUT_SECONDS = 1_680
EXTERNAL_TIMEOUT_SECONDS = 1_800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
MINIMUM_AUC_GAIN = 0.0002
BASELINE_AUC = {42: 0.9593360799332391, 2024: 0.9595753150002255}
VALIDATION_COLUMNS = (
    "row_index",
    ID_COLUMN,
    "student_probability",
    "catboost_probability",
    "baseline_probability",
    "candidate_probability",
)


class SourceNeighborTimeout(TimeoutError):
    """Raised when the fixed source-neighbour diagnostic reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise SourceNeighborTimeout("source-neighbour diagnostic exceeded its deadline")


def _validate_predictor_frame(frame: pd.DataFrame, spec: InputSpec) -> None:
    if list(frame.columns) != list(spec.feature_columns):
        raise ValueError("features must match the declared predictor schema exactly")
    if frame.columns.duplicated().any():
        raise ValueError("features must not contain duplicate columns")
    if ID_COLUMN in frame or TARGET_COLUMN in frame:
        raise ValueError("features must match the declared predictor schema exactly")


def _canonical_columns(
    frame: pd.DataFrame, spec: InputSpec
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    numeric: dict[str, np.ndarray] = {}
    categorical: dict[str, np.ndarray] = {}
    for column in spec.numeric_columns:
        numeric[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64, copy=True
        )
    for column in spec.categorical_columns:
        categorical[column] = (
            frame[column].astype("string").fillna("<missing>").astype(str).to_numpy()
        )
    return numeric, categorical


def _numeric_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left == right) | (np.isnan(left) & np.isnan(right))


def _distance_groups(values: np.ndarray) -> dict[float, np.ndarray]:
    groups: dict[float, list[int]] = {}
    for index, value in enumerate(values):
        if not np.isnan(value):
            groups.setdefault(float(value), []).append(index)
    return {
        value: np.asarray(indices, dtype=np.int64) for value, indices in groups.items()
    }


def compute_source_neighbor_probabilities(
    query_features: pd.DataFrame,
    source_features: pd.DataFrame,
    source_labels: np.ndarray,
    spec: InputSpec,
    *,
    query_block_size: int = NEIGHBOR_QUERY_BLOCK_SIZE,
    max_pair_elements: int = MAX_PAIR_ELEMENTS,
    progress_callback: Any | None = None,
) -> np.ndarray:
    """Compute the fixed all-source weighted Hamming probability per row.

    The caller supplies only predictor frames and cleaned source labels. The
    source labels are never joined to competition rows or used as features.
    """
    _validate_predictor_frame(query_features, spec)
    _validate_predictor_frame(source_features, spec)
    if DISTANCE_COLUMN not in spec.numeric_columns:
        raise ValueError("Flight Distance must be a declared numeric predictor")
    labels = np.asarray(source_labels, dtype=np.float64)
    if labels.ndim != 1 or len(labels) != len(source_features) or len(labels) == 0:
        raise ValueError("source labels must match non-empty source predictors")
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("source labels must be finite binary values")
    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")
    if max_pair_elements < 1 or max_pair_elements > MAX_PAIR_ELEMENTS:
        raise ValueError("max_pair_elements must be between 1 and 250000")

    query_numeric, query_categorical = _canonical_columns(query_features, spec)
    source_numeric, source_categorical = _canonical_columns(source_features, spec)
    query_distance = query_numeric[DISTANCE_COLUMN]
    source_distance = source_numeric[DISTANCE_COLUMN]
    source_groups = _distance_groups(source_distance)
    query_groups = _distance_groups(query_distance)
    fallback = float(np.mean(labels))
    result = np.full(len(query_features), fallback, dtype=np.float64)
    compare_numeric = tuple(
        column for column in spec.numeric_columns if column != DISTANCE_COLUMN
    )
    compare_categorical = tuple(spec.categorical_columns)
    processed = 0

    for distance, query_indices in query_groups.items():
        source_indices = source_groups.get(distance)
        if source_indices is None:
            processed += len(query_indices)
            if progress_callback is not None:
                progress_callback(processed, len(query_features), distance)
            continue
        query_step = min(query_block_size, max_pair_elements)
        for query_start in range(0, len(query_indices), query_step):
            query_block = query_indices[query_start : query_start + query_step]
            source_block_size = max(1, max_pair_elements // len(query_block))
            numerator = np.zeros(len(query_block), dtype=np.float64)
            denominator = np.zeros(len(query_block), dtype=np.float64)
            for source_start in range(0, len(source_indices), source_block_size):
                source_block = source_indices[
                    source_start : source_start + source_block_size
                ]
                mismatch = np.zeros(
                    (len(query_block), len(source_block)), dtype=np.uint8
                )
                for column in compare_numeric:
                    mismatch += (
                        ~_numeric_equal(
                            query_numeric[column][query_block, None],
                            source_numeric[column][source_block][None, :],
                        )
                    ).astype(np.uint8)
                for column in compare_categorical:
                    mismatch += (
                        query_categorical[column][query_block, None]
                        != source_categorical[column][source_block][None, :]
                    ).astype(np.uint8)
                weights = np.exp(-mismatch.astype(np.float64))
                denominator += weights.sum(axis=1, dtype=np.float64)
                numerator += weights @ labels[source_block]
            if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
                raise RuntimeError("source-neighbour normalization weights are invalid")
            # Positive weights and binary labels imply [0, 1]; clamp roundoff only.
            result[query_block] = np.clip(numerator / denominator, 0.0, 1.0)
            processed += len(query_block)
            if progress_callback is not None:
                progress_callback(processed, len(query_features), distance)
    validate_probabilities(result, len(query_features))
    return result


def append_source_neighbor_feature(
    features: pd.DataFrame, probabilities: np.ndarray, spec: InputSpec
) -> tuple[pd.DataFrame, InputSpec]:
    """Append one numeric source-neighbour probability without mutation."""
    _validate_predictor_frame(features, spec)
    values = np.asarray(probabilities, dtype=np.float64)
    validate_probabilities(values, len(features))
    result = features.loc[:, list(spec.feature_columns)].copy()
    result[SOURCE_NEIGHBOR_FEATURE] = values
    extended = InputSpec(
        feature_columns=spec.feature_columns + (SOURCE_NEIGHBOR_FEATURE,),
        numeric_columns=spec.numeric_columns + (SOURCE_NEIGHBOR_FEATURE,),
        categorical_columns=spec.categorical_columns,
    )
    return result, extended


def _array_record(*arrays: tuple[str, np.ndarray]) -> dict[str, Any]:
    return {
        "rows": {name: len(value) for name, value in arrays},
        "sha256": {name: _array_fingerprint(value) for name, value in arrays},
    }


def _source_hashes() -> dict[str, str]:
    names = (
        "source_neighbor_experiment.py",
        "source_augmentation.py",
        "source_teacher_cv.py",
        "source_teacher_experiment.py",
        "distance_profile_experiment.py",
        "catboost_teacher_feature.py",
        "five_fold_cv.py",
        "lightgbm_experiment.py",
        "lightgbm_capacity.py",
        "compare.py",
        "train.py",
        "weighted_blend_experiment.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _config_hash() -> str:
    return hashlib.sha256(
        json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _validate_fixed_schema(spec: InputSpec) -> None:
    if len(spec.feature_columns) != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError("original feature count is not 21")
    if len(spec.numeric_columns) != EXPECTED_NUMERIC_COLUMNS:
        raise RuntimeError("original numeric feature count is not 17")
    if len(spec.categorical_columns) != EXPECTED_CATEGORICAL_COLUMNS:
        raise RuntimeError("original categorical feature count is not 4")
    if DISTANCE_COLUMN not in spec.numeric_columns:
        raise RuntimeError("Flight Distance is not an original numeric predictor")


def _progress(
    processed: int, total: int, distance: float | None, *, last: list[int]
) -> None:
    if processed == total or processed - last[0] >= PROGRESS_INTERVAL:
        last[0] = processed
        print(
            f"{datetime.now(timezone.utc).isoformat()} "
            f"source-neighbour rows={processed}/{total} distance={distance}",
            flush=True,
        )


def _save_neighbor_cache(
    path: Path, ids: np.ndarray, probabilities: np.ndarray
) -> None:
    np.savez_compressed(
        path,
        row_index=np.arange(len(ids), dtype=np.int64),
        id=np.asarray(ids, dtype=np.int64),
        source_neighbor_probability=np.asarray(probabilities, dtype=np.float64),
    )


def _load_neighbor_cache(
    path: Path, train: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path) as saved:
        if set(saved.files) != {
            "row_index",
            "id",
            "source_neighbor_probability",
        }:
            raise RuntimeError("source-neighbour cache fields changed")
        row_index = np.asarray(saved["row_index"], dtype=np.int64)
        ids = np.asarray(saved["id"], dtype=np.int64)
        probabilities = np.asarray(
            saved["source_neighbor_probability"], dtype=np.float64
        )
    expected_rows = np.arange(len(train), dtype=np.int64)
    expected_ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
    if not np.array_equal(row_index, expected_rows) or not np.array_equal(
        ids, expected_ids
    ):
        raise RuntimeError("source-neighbour cache row or ID order changed")
    validate_probabilities(probabilities, len(train))
    return probabilities, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(probabilities),
        "row_index": _array_record(("row_index", row_index)),
    }


def _profile_cache_record(
    metadata: dict[str, Any], seed: int, inner_fit: np.ndarray
) -> dict[str, Any]:
    results = metadata.get("validation_results", [])
    for result in results:
        if result.get("split_seed") == seed:
            record = result.get("profile_table")
            membership = result.get("table_fit_membership")
            expected_hash = _array_fingerprint(inner_fit)
            if not isinstance(record, dict) or membership != {
                "rows": len(inner_fit),
                "sha256": expected_hash,
            }:
                raise RuntimeError("saved distance profile fit membership changed")
            return record
    raise RuntimeError(f"saved distance profile record is missing for seed {seed}")


def _load_verified_profile_table(
    path: Path,
    record: dict[str, Any],
    original_features: pd.DataFrame,
    spec: InputSpec,
    inner_fit: np.ndarray,
) -> Any:
    if record.get("path") != str(path) or record.get("sha256") != sha256_file(path):
        raise RuntimeError(f"saved distance profile cache changed: {path}")
    table = fit_distance_profile_table(original_features.iloc[inner_fit], spec)
    if record.get("distance_column") != table.distance_column or record.get(
        "mean_columns"
    ) != list(table.mean_columns):
        raise RuntimeError("saved distance profile schema changed")
    with np.load(path) as saved:
        if set(saved.files) != {"distance", "group_count", "group_means"}:
            raise RuntimeError("saved distance profile fields changed")
        if not np.array_equal(saved["distance"], table.group_means.index.to_numpy()):
            raise RuntimeError("saved distance profile distances changed")
        if not np.array_equal(saved["group_count"], table.group_counts.to_numpy()):
            raise RuntimeError("saved distance profile counts changed")
        if not np.array_equal(
            saved["group_means"], table.group_means.to_numpy(), equal_nan=True
        ):
            raise RuntimeError("saved distance profile means changed")
    recorded_means = record.get("fallback_global_means")
    actual_means = {
        column: None
        if pd.isna(table.global_means[column])
        else float(table.global_means[column])
        for column in table.mean_columns
    }
    if recorded_means != actual_means:
        raise RuntimeError("saved distance profile global means changed")
    return table


def source_neighbor_gate(
    baseline: dict[int, float], candidate: dict[int, float]
) -> dict[str, Any]:
    """Require the fixed candidate AUC gain independently on both holdouts."""
    comparison = [
        {
            "split_seed": seed,
            "baseline_roc_auc": baseline[seed],
            "candidate_roc_auc": candidate[seed],
            "auc_gain": candidate[seed] - baseline[seed],
            "qualifies": candidate[seed] - baseline[seed] >= MINIMUM_AUC_GAIN,
        }
        for seed in SPLIT_SEEDS
    ]
    qualifies = all(row["qualifies"] for row in comparison)
    return {
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "comparison": comparison,
        "qualifies": qualifies,
        "decision": (
            "select_source_neighbor_feature" if qualifies else "retain_current_blend"
        ),
        "holdout_note": (
            "overlapping holdouts are stability evidence, not independent confirmation"
        ),
    }


def _read_saved_distance_validation(
    path: Path, train: pd.DataFrame, holdout: np.ndarray
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != list(VALIDATION_COLUMNS):
        raise RuntimeError(f"saved distance profile validation schema changed: {path}")
    expected_rows = np.asarray(holdout, dtype=np.int64)
    expected_ids = train.iloc[holdout][ID_COLUMN].to_numpy(dtype=np.int64)
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise RuntimeError(f"saved distance profile rows changed: {path}")
    if not np.array_equal(frame[ID_COLUMN].to_numpy(dtype=np.int64), expected_ids):
        raise RuntimeError(f"saved distance profile IDs changed: {path}")
    if frame["row_index"].duplicated().any() or frame[ID_COLUMN].duplicated().any():
        raise RuntimeError(f"saved distance profile rows or IDs are not unique: {path}")
    for column in VALIDATION_COLUMNS[2:]:
        validate_probabilities(frame[column].to_numpy(dtype=np.float64), len(frame))
    return frame


def _metrics_from_saved_baseline(
    y: np.ndarray,
    holdout: np.ndarray,
    slow: pd.DataFrame,
    profile: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    expected_ids = profile[ID_COLUMN].to_numpy(dtype=np.int64)
    if not np.array_equal(slow[ID_COLUMN].to_numpy(dtype=np.int64), expected_ids):
        raise RuntimeError(
            "saved slow-CatBoost IDs do not align with profile validation"
        )
    if not np.allclose(
        slow["probability"].to_numpy(dtype=np.float64),
        profile["catboost_probability"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("saved slow-CatBoost probabilities changed")
    baseline = weighted_blend_predictions(
        expected_ids,
        expected_ids,
        slow["probability"].to_numpy(dtype=np.float64),
        expected_ids,
        profile["student_probability"].to_numpy(dtype=np.float64),
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )["blend_probability"].to_numpy(dtype=np.float64)
    if not np.allclose(
        baseline,
        profile["candidate_probability"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("saved distance profile candidate is not current baseline")
    metrics = _metrics(y[holdout], baseline)
    if abs(metrics["roc_auc"] - BASELINE_AUC[seed]) > 1e-12:
        raise RuntimeError("recomputed current baseline changed")
    return baseline, metrics


def _progress_callback(
    metadata: dict[str, Any], metadata_path: Path, split_seed: int
) -> Any:
    def callback(environment: Any) -> None:
        iteration = int(environment.iteration) + 1
        if iteration % 100:
            return
        current = metadata.get("current_fit")
        if isinstance(current, dict):
            current.setdefault("progress_log", []).append(
                {"utc": datetime.now(timezone.utc).isoformat(), "tree": iteration}
            )
            current["last_progress_tree"] = iteration
            save_json(metadata_path, metadata)
        print(
            f"{datetime.now(timezone.utc).isoformat()} split_seed={split_seed} "
            f"tree={iteration}",
            flush=True,
        )

    callback.order = 20  # type: ignore[attr-defined]
    return callback


def _run_fit(
    *,
    features: pd.DataFrame,
    spec: InputSpec,
    y: np.ndarray,
    holdout: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    metadata: dict[str, Any],
    metadata_path: Path,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata["current_fit"] = {
        "split_seed": seed,
        "model": "lightgbm_source_neighbor_student",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "outer_holdout_rows": len(holdout),
        "outer_holdout_used_for_training_or_early_stopping": False,
        "progress_log": [],
    }
    save_json(metadata_path, metadata)
    model = build_lightgbm_classifier()
    fit_started = time.perf_counter()
    levels, fit_seconds, details = _fit_with_inner_validation(
        model,
        features,
        y,
        inner_fit,
        inner_validation,
        spec,
        callbacks=_fit_callbacks()
        + [_progress_callback(metadata, metadata_path, seed)],
    )
    predictions = np.asarray(
        model.predict_proba(
            prepare_lightgbm_frame(features.iloc[holdout], spec, levels)
        )[:, 1],
        dtype=np.float64,
    )
    validate_probabilities(predictions, len(holdout))
    result = {
        "fit_seconds": round(float(fit_seconds), 3),
        "fit_elapsed_seconds": round(time.perf_counter() - fit_started, 3),
        "settings": model.get_params(),
        "best_iteration": details["best_iteration"],
        "actual_iterations": details["actual_iterations"],
        "resources": details["resources"],
        "standalone": _metrics(y[holdout], predictions),
        "category_levels_from_inner_fit": {
            name: len(values) for name, values in levels.items()
        },
    }
    del model
    gc.collect()
    return predictions, result


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly two approved source-neighbour LightGBM student fits."""
    started = time.perf_counter()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "completed_fits": [],
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "progress_interval": PROGRESS_INTERVAL,
        "predeclared_settings": dict(PREDECLARED_CONFIG),
        "feature_change": {
            "only": SOURCE_NEIGHBOR_FEATURE,
            "original_feature_count": EXPECTED_FEATURE_COLUMNS,
            "teacher_feature_count": 1,
            "distance_profile_feature_count": 17,
            "student_feature_count": 40,
            "student_numeric_count": 36,
            "categorical_count": EXPECTED_CATEGORICAL_COLUMNS,
        },
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        source_before = _source_hashes()
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        source_raw = pd.read_csv(data_dir / "source_airline" / "data.csv")
        spec = validate_inputs(train, test)
        _validate_fixed_schema(spec)
        if len(train) != EXPECTED_TRAIN_ROWS:
            raise RuntimeError("training row count does not match the approved screen")
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        original_features = train.loc[:, list(spec.feature_columns)]
        source_features, source_labels, cleaning = clean_source_rows(
            source_raw, train, test, spec
        )
        source_metadata_path = (
            artifacts_dir / "source_augmentation" / "compare_metadata.json"
        )
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if cleaning != source_metadata.get("source_rows"):
            raise RuntimeError("source cleaning report does not match prior metadata")
        if cleaning.get("clean_rows") != EXPECTED_CLEAN_SOURCE_ROWS:
            raise RuntimeError("source cleaning report does not contain 129859 rows")
        input_after_preflight = _input_hashes(data_dir)
        if input_after_preflight != input_before:
            raise RuntimeError("input CSV fingerprint changed during preflight")
        prior_lightgbm = _load_prior_lightgbm(artifacts_dir, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        profile_metadata_path = (
            artifacts_dir
            / OUTPUT_NAME.replace("source_neighbor", "distance_profile")
            / "compare_metadata.json"
        )
        profile_metadata = json.loads(profile_metadata_path.read_text(encoding="utf-8"))
        if profile_metadata.get("status") != "completed":
            raise RuntimeError("distance profile metadata is not completed")
        if profile_metadata.get("input_sha256_before") != input_before:
            raise RuntimeError("distance profile inputs do not match current inputs")
        if profile_metadata.get("predeclared_settings") != dict(PREDECLARED_CONFIG):
            raise RuntimeError("distance profile LightGBM configuration changed")
        recorded_profile_sources = profile_metadata.get("source_sha256_before", {})
        current_sources = _source_hashes()
        if any(
            current_sources.get(name) != digest
            for name, digest in recorded_profile_sources.items()
        ):
            raise RuntimeError("distance profile source fingerprints changed")
        if profile_metadata.get("config_sha256") != _config_hash():
            raise RuntimeError("distance profile configuration fingerprint changed")
        if profile_metadata.get("feature_spec_original") != _feature_spec(spec):
            raise RuntimeError("distance profile original feature schema changed")
        if profile_metadata.get("feature_spec_teacher") is None:
            raise RuntimeError("distance profile teacher feature schema is missing")
        metadata.update(
            {
                "python": sys.version,
                "dependencies": package_versions(),
                "threads": MAX_THREADS,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "config_sha256": _config_hash(),
                "feature_spec_original": _feature_spec(spec),
                "source_cleaning": {
                    "actual": cleaning,
                    "recorded": source_metadata.get("source_rows"),
                    "source_metadata_path": str(source_metadata_path),
                    "source_metadata_sha256": sha256_file(source_metadata_path),
                    "source_provenance": source_metadata.get("source_provenance"),
                    "source_uncertainty": source_metadata.get("source_uncertainty"),
                    "license": source_metadata.get("license"),
                },
                "source_labels": {
                    "source_only": True,
                    "competition_labels_accessible": False,
                    "rows": len(source_labels),
                },
                "reused_artifacts": {
                    "source_teacher_cache": sha256_file(teacher_cache["metadata_path"]),
                    "distance_profile_metadata": sha256_file(profile_metadata_path),
                },
                "neighbor_parameters": {
                    "distance_column": DISTANCE_COLUMN,
                    "comparison_columns": [
                        column
                        for column in spec.feature_columns
                        if column != DISTANCE_COLUMN
                    ],
                    "alpha": 1.0,
                    "distance": "exact Hamming over 20 predictors",
                    "numeric_nan_equal": True,
                    "categorical_missing": "<missing>",
                    "all_source_rows_in_exact_distance_group": True,
                    "fallback": "clean source global positive prevalence",
                    "max_pair_elements": MAX_PAIR_ELEMENTS,
                    "query_block_size": NEIGHBOR_QUERY_BLOCK_SIZE,
                },
            }
        )
        save_json(metadata_path, metadata)

        progress_state = [0]
        neighbor_probability = compute_source_neighbor_probabilities(
            original_features,
            source_features,
            source_labels,
            spec,
            progress_callback=lambda done, total, distance: _progress(
                done, total, distance, last=progress_state
            ),
        )
        cache_path = output_dir / "source_neighbor_train_predictions.npz"
        _save_neighbor_cache(cache_path, ids, neighbor_probability)
        metadata["neighbor_cache"] = {
            **_load_neighbor_cache(cache_path, train)[1],
            "source_rows": len(source_features),
            "source_csv_sha256": input_before["source_airline/data.csv"],
            "source_labels_sha256": _array_fingerprint(source_labels),
        }
        save_json(metadata_path, metadata)

        teacher_features, teacher_spec = append_teacher_feature(
            original_features, teacher_cache["probabilities"], spec
        )
        baseline_aucs: dict[int, float] = {}
        candidate_aucs: dict[int, float] = {}
        for fit_number, seed in enumerate(SPLIT_SEEDS, 1):
            outer_fit, holdout, inner_fit, inner_validation, split_record = (
                _load_split_arrays(artifacts_dir, train, y, prior_lightgbm, seed)
            )
            if (
                len(inner_fit) != 503737
                or len(inner_validation) != 55971
                or len(holdout) != 139927
            ):
                raise RuntimeError("recorded split row counts changed")
            split_name = "outer" if seed == 42 else "confirmation"
            profile_record = _profile_cache_record(profile_metadata, seed, inner_fit)
            profile_path = (
                artifacts_dir
                / "distance_profile_feature"
                / f"distance_profile_seed{seed}.npz"
            )
            profile_table = _load_verified_profile_table(
                profile_path, profile_record, original_features, spec, inner_fit
            )
            profile_features, profile_spec = append_distance_profile_features(
                teacher_features, teacher_spec, profile_table
            )
            student_features, student_spec = append_source_neighbor_feature(
                profile_features, neighbor_probability, profile_spec
            )
            if (
                len(student_spec.feature_columns) != 40
                or len(student_spec.numeric_columns) != 36
            ):
                raise RuntimeError("source-neighbour student schema is not 40/36/4")
            metadata["feature_spec_student"] = _feature_spec(student_spec)

            slow_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            slow = _load_components(slow_path, train, holdout)
            saved_profile_path = (
                artifacts_dir
                / "distance_profile_feature"
                / f"validation_distance_profile_seed{seed}.csv"
            )
            saved_profile = _read_saved_distance_validation(
                saved_profile_path, train, holdout
            )
            baseline, baseline_metrics = _metrics_from_saved_baseline(
                y, holdout, slow, saved_profile, seed
            )
            baseline_aucs[seed] = baseline_metrics["roc_auc"]
            metadata["recorded_splits"] = metadata.get("recorded_splits", {})
            metadata["recorded_splits"][split_name] = split_record
            metadata["current_fit"] = {
                "fit_number": fit_number,
                "split_seed": seed,
                "status": "running",
                "split": split_record,
                "profile_fit_membership": {
                    "rows": len(inner_fit),
                    "sha256": _array_fingerprint(inner_fit),
                },
                "baseline": baseline_metrics,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)
            predictions, fit_record = _run_fit(
                features=student_features,
                spec=student_spec,
                y=y,
                holdout=holdout,
                inner_fit=inner_fit,
                inner_validation=inner_validation,
                metadata=metadata,
                metadata_path=metadata_path,
                seed=seed,
            )
            candidate = weighted_blend_predictions(
                train.iloc[holdout][ID_COLUMN].to_numpy(),
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                train.iloc[holdout][ID_COLUMN].to_numpy(),
                predictions,
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )["blend_probability"].to_numpy(dtype=np.float64)
            candidate_metrics = _metrics(y[holdout], candidate)
            candidate_aucs[seed] = candidate_metrics["roc_auc"]
            validation = pd.DataFrame(
                {
                    "row_index": holdout,
                    ID_COLUMN: train.iloc[holdout][ID_COLUMN].to_numpy(),
                    "student_probability": predictions,
                    "catboost_probability": slow["probability"].to_numpy(),
                    "baseline_probability": baseline,
                    "candidate_probability": candidate,
                }
            )
            validation_path = output_dir / f"validation_source_neighbor_seed{seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            fit_record.update(
                {
                    "split_seed": seed,
                    "split": split_record,
                    "profile_fit_membership": {
                        "rows": len(inner_fit),
                        "sha256": _array_fingerprint(inner_fit),
                    },
                    "profile_table": {
                        "path": str(profile_path),
                        "sha256": sha256_file(profile_path),
                        "mean_columns": list(profile_table.mean_columns),
                    },
                    "baseline": baseline_metrics,
                    "new_baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "new_candidate": candidate_metrics,
                    "auc_gain_vs_current_baseline": candidate_metrics["roc_auc"]
                    - baseline_metrics["roc_auc"],
                    "validation_predictions_path": str(validation_path),
                    "validation_predictions_sha256": sha256_file(validation_path),
                }
            )
            metadata["validation_results"].append(fit_record)
            metadata["completed_fits"].append(f"student_seed_{seed}")
            metadata["completed_prediction_hashes"] = [
                result["validation_predictions_sha256"]
                for result in metadata["validation_results"]
            ]
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del predictions, candidate, validation, profile_features, student_features
            gc.collect()

        metadata["selection"] = source_neighbor_gate(baseline_aucs, candidate_aucs)
    except BaseException as error:
        caught = error
        metadata["error"] = _error_record(error)
    finally:
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                finalization_errors.append(error)
        metadata["current_fit"] = None
        try:
            if input_before is not None:
                input_after = _input_hashes(data_dir)
                metadata["input_sha256_after"] = input_after
                if input_after != input_before:
                    finalization_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
        except BaseException as error:
            finalization_errors.append(error)
        try:
            if source_before is not None:
                source_after = _source_hashes()
                metadata["source_sha256_after"] = source_after
                if source_after != source_before:
                    finalization_errors.append(
                        RuntimeError("source fingerprint changed")
                    )
        except BaseException as error:
            finalization_errors.append(error)
        try:
            protected_after = _artifact_snapshot(artifacts_dir, output_dir)
            metadata["protected_artifact_sha256_after"] = protected_after
            metadata["protected_artifacts_unchanged"] = (
                protected_before is not None and protected_after == protected_before
            )
            if protected_before is not None and protected_after != protected_before:
                finalization_errors.append(
                    RuntimeError("pre-existing artifact fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        if finalization_errors:
            integrity_error = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["integrity_error"] = _error_record(integrity_error)
            if caught is None:
                caught = integrity_error
                metadata["error"] = _error_record(integrity_error)
            else:
                metadata["finalization_error"] = _error_record(integrity_error)
        elif caught is not None:
            metadata["status"] = (
                "timeout" if isinstance(caught, SourceNeighborTimeout) else "failed"
            )
            metadata["selection"] = None
            metadata["error"] = _error_record(caught)
        else:
            metadata["status"] = "completed"
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("output exceeded 100 MiB")
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = budget_error
                metadata["error"] = _error_record(budget_error)
            else:
                metadata["output_budget_error"] = _error_record(budget_error)
        if metadata["status"] != "completed":
            metadata["selection"] = None
        try:
            for _ in range(4):
                save_json(metadata_path, metadata)
                actual_bytes = sum(
                    path.stat().st_size
                    for path in output_dir.rglob("*")
                    if path.is_file()
                )
                if actual_bytes == metadata["output_bytes"]:
                    break
                metadata["output_bytes"] = actual_bytes
        except BaseException as error:
            if caught is None:
                caught = error
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
