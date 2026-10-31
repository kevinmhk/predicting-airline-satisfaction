"""Run one of the eight fixed final-sprint student diagnostics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
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

# Scientific imports intentionally follow the thread-limit environment setup.
# ruff: noqa: E402
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from distance_profile_experiment import (
    DISTANCE_COLUMN,
    append_distance_profile_features,
)
from distance_residual_experiment import _source_hashes as residual_source_hashes
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
from source_neighbor_experiment import (
    _load_verified_profile_table,
    _metrics_from_saved_baseline,
    _profile_cache_record,
    _read_saved_distance_validation,
)
from source_teacher_cv import _verify_teacher_cache
from source_teacher_experiment import (
    SOURCE_TEACHER_FEATURE,
    _load_prior_lightgbm,
    append_teacher_feature,
    teacher_feature_gate,
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

DISTANCE_CATEGORY = "distance_category"
LEAVES127 = "leaves127"
ROUTE_DISTANCE_TARGET = "route_target_encoding"
ROUTE_DISTANCE_TARGET_127 = "route_target_encoding127"
ROUTE_CONTEXT_TARGET = "route_context_target_encoding"
ROUTE_CONTEXT_TARGET_127 = "route_context_target_encoding127"
ROUTE_HIERARCHICAL = "route_hierarchical_encoding"
ROUTE_TEACHER_RESIDUAL = "route_teacher_residual_encoding"
EXPERIMENTS = (
    DISTANCE_CATEGORY,
    LEAVES127,
    ROUTE_DISTANCE_TARGET,
    ROUTE_DISTANCE_TARGET_127,
    ROUTE_CONTEXT_TARGET,
    ROUTE_CONTEXT_TARGET_127,
    ROUTE_HIERARCHICAL,
    ROUTE_TEACHER_RESIDUAL,
)
FLIGHT_DISTANCE_CATEGORY = "flight_distance_category"
FLIGHT_DISTANCE_TARGET = "flight_distance_target_mean"
FLIGHT_DISTANCE_CLASS_TARGET = "flight_distance_class_target_mean"
FLIGHT_DISTANCE_TRAVEL_TARGET = "flight_distance_travel_target_mean"
FLIGHT_DISTANCE_CUSTOMER_TARGET = "flight_distance_customer_target_mean"
FLIGHT_DISTANCE_HIERARCHICAL = "flight_distance_hierarchical_target_mean"
FLIGHT_DISTANCE_CLASS_HIERARCHICAL = "flight_distance_class_hierarchical_target_mean"
FLIGHT_DISTANCE_TRAVEL_HIERARCHICAL = "flight_distance_travel_hierarchical_target_mean"
FLIGHT_DISTANCE_CUSTOMER_HIERARCHICAL = (
    "flight_distance_customer_hierarchical_target_mean"
)
FLIGHT_DISTANCE_TEACHER_RESIDUAL = "flight_distance_teacher_residual_mean"
FLIGHT_DISTANCE_CLASS_TEACHER_RESIDUAL = "flight_distance_class_teacher_residual_mean"
FLIGHT_DISTANCE_TRAVEL_TEACHER_RESIDUAL = "flight_distance_travel_teacher_residual_mean"
FLIGHT_DISTANCE_CUSTOMER_TEACHER_RESIDUAL = (
    "flight_distance_customer_teacher_residual_mean"
)
ROUTE_KEY_DISTANCE = "__route_distance_key"
ROUTE_KEY_PAIR = "__route_distance_class_key"
ROUTE_KEY_TRAVEL_PAIR = "__route_distance_travel_key"
ROUTE_KEY_CUSTOMER_PAIR = "__route_distance_customer_key"
SPLIT_SEEDS = (42, 2024)
INTERNAL_SEED = 42
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_FEATURE_COLUMNS = 21
EXPECTED_NUMERIC_COLUMNS = 17
EXPECTED_CATEGORICAL_COLUMNS = 4
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 100
MINIMUM_AUC_GAIN = 0.0002
VALIDATION_COLUMNS = (
    "row_index",
    ID_COLUMN,
    "student_probability",
    "catboost_probability",
    "baseline_probability",
    "candidate_probability",
)


class FinalSprintTimeout(TimeoutError):
    """Raised when one fixed final-sprint run reaches its deadline."""


@dataclass(frozen=True)
class RouteEncodingResult:
    """Route encodings and their resulting feature contract."""

    features: pd.DataFrame
    spec: InputSpec
    metadata: dict[str, Any]


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise FinalSprintTimeout("final-sprint diagnostic exceeded its deadline")


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _validate_feature_frame(features: pd.DataFrame, spec: InputSpec) -> None:
    if list(features.columns) != list(spec.feature_columns):
        raise ValueError("features must match the declared predictor schema exactly")
    if features.columns.duplicated().any():
        raise ValueError("features must not contain duplicate columns")
    if ID_COLUMN in features or TARGET_COLUMN in features:
        raise ValueError("final-sprint features must not receive id or satisfaction")


def canonical_distance_key(value: Any) -> str:
    """Return one stable string key for numeric, zero, missing, or unseen distance."""
    if pd.isna(value):
        return "<missing>"
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("Flight Distance must be finite or missing")
    if numeric == 0.0:
        return "0"
    return format(numeric, ".17g")


def _canonical_class_key(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value)


def _route_key_frame(
    features: pd.DataFrame, *, include_context: bool = False
) -> pd.DataFrame:
    distances = [canonical_distance_key(value) for value in features[DISTANCE_COLUMN]]
    classes = [_canonical_class_key(value) for value in features["Class"]]
    pairs = [
        json.dumps((distance, category), ensure_ascii=True, separators=(",", ":"))
        for distance, category in zip(distances, classes, strict=True)
    ]
    result = pd.DataFrame(
        {ROUTE_KEY_DISTANCE: distances, ROUTE_KEY_PAIR: pairs}, index=features.index
    )
    if include_context:
        travels = [_canonical_class_key(value) for value in features["Type of Travel"]]
        customers = [_canonical_class_key(value) for value in features["Customer Type"]]
        result[ROUTE_KEY_TRAVEL_PAIR] = [
            json.dumps((distance, travel), ensure_ascii=True, separators=(",", ":"))
            for distance, travel in zip(distances, travels, strict=True)
        ]
        result[ROUTE_KEY_CUSTOMER_PAIR] = [
            json.dumps((distance, customer), ensure_ascii=True, separators=(",", ":"))
            for distance, customer in zip(distances, customers, strict=True)
        ]
    return result


def append_distance_category(
    features: pd.DataFrame, spec: InputSpec
) -> tuple[pd.DataFrame, InputSpec]:
    """Append the canonical Flight Distance duplicate as one native category."""
    _validate_feature_frame(features, spec)
    if DISTANCE_COLUMN not in spec.numeric_columns:
        raise ValueError("Flight Distance must remain an original numeric predictor")
    if FLIGHT_DISTANCE_CATEGORY in features.columns:
        raise ValueError(f"feature already exists: {FLIGHT_DISTANCE_CATEGORY}")
    result = features.loc[:, list(spec.feature_columns)].copy()
    result[FLIGHT_DISTANCE_CATEGORY] = [
        canonical_distance_key(value) for value in features[DISTANCE_COLUMN]
    ]
    extended = InputSpec(
        feature_columns=spec.feature_columns + (FLIGHT_DISTANCE_CATEGORY,),
        numeric_columns=spec.numeric_columns,
        categorical_columns=spec.categorical_columns + (FLIGHT_DISTANCE_CATEGORY,),
    )
    return result, extended


def fit_route_target_encoding(
    features: pd.DataFrame,
    spec: InputSpec,
    fit_labels: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    holdout_indices: np.ndarray,
    *,
    include_context: bool = False,
) -> RouteEncodingResult:
    """Cross-fit the approved route keys using only inner model-fit labels."""
    _validate_feature_frame(features, spec)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    holdout_indices = np.asarray(holdout_indices, dtype=np.int64)
    labels = np.asarray(fit_labels, dtype=np.int8)
    if len(labels) != len(fit_indices):
        raise ValueError("route encoder labels must match inner-fit rows")
    if len(np.unique(fit_indices)) != len(fit_indices):
        raise ValueError("route encoder fit rows must be unique")
    if (
        np.intersect1d(fit_indices, validation_indices).size
        or np.intersect1d(fit_indices, holdout_indices).size
        or np.intersect1d(validation_indices, holdout_indices).size
    ):
        raise ValueError("route encoder partitions must be disjoint")
    if set(labels) != {0, 1}:
        raise ValueError("route encoder labels must contain both binary classes")
    keys = _route_key_frame(features, include_context=include_context)
    encoder = TargetEncoder(
        target_type="binary",
        smooth=20.0,
        cv=5,
        shuffle=True,
        random_state=42,
    )
    crossfit = encoder.fit_transform(keys.iloc[fit_indices], labels)
    transformed = np.full((len(features), len(keys.columns)), np.nan, dtype=np.float64)
    transformed[fit_indices] = crossfit
    for indices in (validation_indices, holdout_indices):
        if len(indices):
            transformed[indices] = encoder.transform(keys.iloc[indices])
    for column in range(transformed.shape[1]):
        validate_probabilities(transformed[:, column], len(features))
    encoded_feature_names: tuple[str, ...] = (
        FLIGHT_DISTANCE_TARGET,
        FLIGHT_DISTANCE_CLASS_TARGET,
    )
    if include_context:
        encoded_feature_names += (
            FLIGHT_DISTANCE_TRAVEL_TARGET,
            FLIGHT_DISTANCE_CUSTOMER_TARGET,
        )
    result = features.copy()
    for column, name in enumerate(encoded_feature_names):
        result[name] = transformed[:, column]
    extended_spec = InputSpec(
        feature_columns=spec.feature_columns + encoded_feature_names,
        numeric_columns=spec.numeric_columns + encoded_feature_names,
        categorical_columns=spec.categorical_columns,
    )
    metadata = {
        "features": list(encoded_feature_names),
        "fit_transform_rows": len(fit_indices),
        "fit_row_fingerprint": _array_fingerprint(fit_indices),
        "crossfit_policy": "TargetEncoder.fit_transform on inner model-fit rows only",
        "fold_membership": _route_fold_membership(
            _route_fold_queries(fit_indices, labels)
        ),
        "target_strategy": "binary labels with global route priors",
        "data": "training labels from inner-fit rows only",
        "source": "inner-fit satisfaction labels",
        "timing": "cross-fit before the student fit; no outer-holdout labels",
        "prior_policy": "encoding-fit global prevalence",
        "target_type": "binary",
        "smooth": 20.0,
        "cv": 5,
        "shuffle": True,
        "random_state": 42,
        "missing_key_policy": "explicit missing category",
        "unseen_key_fallback": "inner-fit prevalence",
    }
    return RouteEncodingResult(result, extended_spec, metadata)


def _validate_new_route_encoding_inputs(
    features: pd.DataFrame,
    spec: InputSpec,
    fit_labels: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_feature_frame(features, spec)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    holdout_indices = np.asarray(holdout_indices, dtype=np.int64)
    labels = np.asarray(fit_labels, dtype=np.int8)
    if labels.ndim != 1 or len(labels) != len(fit_indices):
        raise ValueError("route encoder labels must match inner-fit rows")
    if set(labels) != {0, 1}:
        raise ValueError("route encoder labels must contain both binary classes")
    partitions = (fit_indices, validation_indices, holdout_indices)
    if any(
        np.any(indices < 0) or np.any(indices >= len(features))
        for indices in partitions
    ):
        raise ValueError("route encoder indices must address feature rows")
    if any(len(np.unique(indices)) != len(indices) for indices in partitions):
        raise ValueError("route encoder partitions must contain unique rows")
    if any(
        np.intersect1d(left, right).size
        for number, left in enumerate(partitions)
        for right in partitions[number + 1 :]
    ):
        raise ValueError("route encoder partitions must be disjoint")
    return labels, fit_indices, validation_indices, holdout_indices


def _route_encoding_keys(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[list[str], ...]]:
    key_frame = _route_key_frame(features, include_context=True)
    categories = (
        [_canonical_class_key(value) for value in features["Class"]],
        [_canonical_class_key(value) for value in features["Type of Travel"]],
        [_canonical_class_key(value) for value in features["Customer Type"]],
    )
    return key_frame, categories


def _smoothed_group_stats(
    keys: list[str], targets: np.ndarray, fit_indices: np.ndarray
) -> tuple[pd.Series, pd.Series]:
    grouped = (
        pd.DataFrame(
            {
                "key": np.asarray(keys, dtype=object)[fit_indices],
                "target": np.asarray(targets, dtype=np.float64)[fit_indices],
            }
        )
        .groupby("key", sort=False)["target"]
        .agg(["sum", "count"])
    )
    sums = pd.Series(grouped["sum"], dtype=np.float64)
    counts = pd.Series(grouped["count"], dtype=np.float64)
    return sums, counts


def _route_fold_queries(
    fit_indices: np.ndarray, labels: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return [
        (fit_indices[fit], fit_indices[query])
        for fit, query in folds.split(fit_indices, labels)
    ]


def _route_fold_membership(
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    return [
        {
            "fit_rows": len(statistic_indices),
            "fit_sha256": _array_fingerprint(statistic_indices),
            "query_rows": len(query_indices),
            "query_sha256": _array_fingerprint(query_indices),
        }
        for statistic_indices, query_indices in folds
    ]


def _append_route_numeric_encodings(
    features: pd.DataFrame,
    spec: InputSpec,
    names: tuple[str, ...],
    values: np.ndarray,
    metadata: dict[str, Any],
) -> RouteEncodingResult:
    if values.shape != (len(features), len(names)):
        raise ValueError("route encoding values do not match the feature contract")
    if any(name in features.columns for name in names):
        raise ValueError("route encoding feature already exists")
    result = features.copy()
    for column, name in enumerate(names):
        result[name] = values[:, column]
    extended_spec = InputSpec(
        feature_columns=spec.feature_columns + names,
        numeric_columns=spec.numeric_columns + names,
        categorical_columns=spec.categorical_columns,
    )
    metadata["features"] = list(names)
    return RouteEncodingResult(result, extended_spec, metadata)


def fit_hierarchical_route_encoding(
    features: pd.DataFrame,
    spec: InputSpec,
    fit_labels: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> RouteEncodingResult:
    """Cross-fit route means with distance-global and category-conditional priors."""
    labels, fit_indices, validation_indices, holdout_indices = (
        _validate_new_route_encoding_inputs(
            features,
            spec,
            fit_labels,
            fit_indices,
            validation_indices,
            holdout_indices,
        )
    )
    key_frame, categories = _route_encoding_keys(features)
    key_columns = (
        ROUTE_KEY_DISTANCE,
        ROUTE_KEY_PAIR,
        ROUTE_KEY_TRAVEL_PAIR,
        ROUTE_KEY_CUSTOMER_PAIR,
    )
    names = (
        FLIGHT_DISTANCE_HIERARCHICAL,
        FLIGHT_DISTANCE_CLASS_HIERARCHICAL,
        FLIGHT_DISTANCE_TRAVEL_HIERARCHICAL,
        FLIGHT_DISTANCE_CUSTOMER_HIERARCHICAL,
    )
    labels_by_row = np.full(len(features), np.nan, dtype=np.float64)
    labels_by_row[fit_indices] = labels
    transformed = np.full((len(features), len(names)), np.nan, dtype=np.float64)

    def map_rows(
        statistic_indices: np.ndarray, query_indices: np.ndarray, column: int
    ) -> None:
        global_prior = float(np.mean(labels_by_row[statistic_indices]))
        key_values = key_frame[key_columns[column]].tolist()
        sums, counts = _smoothed_group_stats(
            key_values, labels_by_row, statistic_indices
        )
        if column == 0:
            category_priors = pd.Series(
                global_prior, index=query_indices, dtype=np.float64
            )
        else:
            category_values = categories[column - 1]
            category_sums, category_counts = _smoothed_group_stats(
                category_values, labels_by_row, statistic_indices
            )
            category_priors = pd.Series(category_values, dtype=object).map(
                category_sums / category_counts
            )
            category_priors = category_priors.fillna(global_prior).iloc[query_indices]
        query_keys = pd.Series(key_values, dtype=object).iloc[query_indices]
        mapped_sums = query_keys.map(sums).fillna(0.0).to_numpy(dtype=np.float64)
        mapped_counts = query_keys.map(counts).fillna(0.0).to_numpy(dtype=np.float64)
        transformed[query_indices, column] = (
            mapped_sums + 20.0 * category_priors.to_numpy(dtype=np.float64)
        ) / (mapped_counts + 20.0)

    folds = _route_fold_queries(fit_indices, labels)
    for statistic_indices, query_indices in folds:
        for column in range(len(names)):
            map_rows(statistic_indices, query_indices, column)
    query_indices = np.concatenate((validation_indices, holdout_indices))
    for column in range(len(names)):
        map_rows(fit_indices, query_indices, column)
    for column in range(transformed.shape[1]):
        validate_probabilities(transformed[:, column], len(features))
    return _append_route_numeric_encodings(
        features,
        spec,
        names,
        transformed,
        {
            "fit_transform_rows": len(fit_indices),
            "fit_row_fingerprint": _array_fingerprint(fit_indices),
            "fold_membership": _route_fold_membership(folds),
            "target_strategy": "binary labels with hierarchical route priors",
            "keys": [
                "Flight Distance",
                "Distance x Class",
                "Distance x Type of Travel",
                "Distance x Customer Type",
            ],
            "smooth": 20.0,
            "config": {"smooth": 20.0, "folds": 5, "shuffle": True, "random_state": 42},
            "folds": {"count": 5, "shuffle": True, "random_state": 42},
            "data": "training labels from inner-fit rows only",
            "source": "inner-fit satisfaction labels",
            "timing": "cross-fit before the student fit; no outer-holdout labels",
            "prior_policy": "global; pair category conditional",
            "unseen_policy": (
                "missing explicit; unseen pair uses category prior; "
                "unseen category uses global prior"
            ),
        },
    )


def fit_route_teacher_residual_encoding(
    features: pd.DataFrame,
    spec: InputSpec,
    fit_labels: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    holdout_indices: np.ndarray,
) -> RouteEncodingResult:
    """Cross-fit route means of labels minus the frozen source-teacher probability."""
    labels, fit_indices, validation_indices, holdout_indices = (
        _validate_new_route_encoding_inputs(
            features,
            spec,
            fit_labels,
            fit_indices,
            validation_indices,
            holdout_indices,
        )
    )
    if SOURCE_TEACHER_FEATURE not in spec.numeric_columns:
        raise ValueError(
            "teacher residual encoding requires the frozen teacher feature"
        )
    teacher = features[SOURCE_TEACHER_FEATURE].to_numpy(dtype=np.float64)
    validate_probabilities(teacher, len(features))
    key_frame, _ = _route_encoding_keys(features)
    key_columns = (
        ROUTE_KEY_DISTANCE,
        ROUTE_KEY_PAIR,
        ROUTE_KEY_TRAVEL_PAIR,
        ROUTE_KEY_CUSTOMER_PAIR,
    )
    names = (
        FLIGHT_DISTANCE_TEACHER_RESIDUAL,
        FLIGHT_DISTANCE_CLASS_TEACHER_RESIDUAL,
        FLIGHT_DISTANCE_TRAVEL_TEACHER_RESIDUAL,
        FLIGHT_DISTANCE_CUSTOMER_TEACHER_RESIDUAL,
    )
    residual_by_row = np.full(len(features), np.nan, dtype=np.float64)
    residual_by_row[fit_indices] = labels - teacher[fit_indices]
    transformed = np.full((len(features), len(names)), np.nan, dtype=np.float64)

    def map_rows(statistic_indices: np.ndarray, query_indices: np.ndarray) -> None:
        prior = float(np.mean(residual_by_row[statistic_indices]))
        for column, key in enumerate(key_columns):
            key_values = key_frame[key].tolist()
            sums, counts = _smoothed_group_stats(
                key_values, residual_by_row, statistic_indices
            )
            query_keys = pd.Series(key_values, dtype=object).iloc[query_indices]
            mapped_sums = query_keys.map(sums).fillna(0.0).to_numpy(dtype=np.float64)
            mapped_counts = (
                query_keys.map(counts).fillna(0.0).to_numpy(dtype=np.float64)
            )
            transformed[query_indices, column] = (mapped_sums + 20.0 * prior) / (
                mapped_counts + 20.0
            )

    folds = _route_fold_queries(fit_indices, labels)
    for statistic_indices, query_indices in folds:
        map_rows(statistic_indices, query_indices)
    query_indices = np.concatenate((validation_indices, holdout_indices))
    map_rows(fit_indices, query_indices)
    if (
        not np.isfinite(transformed).all()
        or (transformed < -1.0).any()
        or (transformed > 1.0).any()
    ):
        raise ValueError("teacher residual encodings must be finite and within [-1, 1]")
    return _append_route_numeric_encodings(
        features,
        spec,
        names,
        transformed,
        {
            "fit_transform_rows": len(fit_indices),
            "fit_row_fingerprint": _array_fingerprint(fit_indices),
            "fold_membership": _route_fold_membership(folds),
            "target_strategy": "label minus frozen teacher probability",
            "keys": [
                "Flight Distance",
                "Distance x Class",
                "Distance x Type of Travel",
                "Distance x Customer Type",
            ],
            "smooth": 20.0,
            "config": {"smooth": 20.0, "folds": 5, "shuffle": True, "random_state": 42},
            "folds": {"count": 5, "shuffle": True, "random_state": 42},
            "data": (
                "training labels and frozen teacher probabilities from "
                "inner-fit rows only"
            ),
            "source": SOURCE_TEACHER_FEATURE,
            "timing": "cross-fit before the student fit; no outer-holdout labels",
            "teacher_retrained": False,
            "unseen_policy": "current inner-fit mean residual",
            "value_range": [-1.0, 1.0],
        },
    )


def build_trial_classifier(experiment: str) -> Any:
    """Build the fixed student model for one declared experiment."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown final-sprint experiment: {experiment}")
    if experiment in (
        LEAVES127,
        ROUTE_DISTANCE_TARGET_127,
        ROUTE_CONTEXT_TARGET_127,
    ):
        return build_lightgbm_classifier(num_leaves=127)
    return build_lightgbm_classifier()


def final_sprint_gate(
    experiment: str, baseline: dict[int, float], candidate: dict[int, float]
) -> dict[str, Any]:
    """Apply the existing independent +0.0002 gate with a mode-specific decision."""
    result = teacher_feature_gate(baseline, candidate)
    result["decision"] = (
        f"select_final_sprint_{experiment}"
        if result["qualifies"]
        else "retain_current_blend"
    )
    return result


def _source_hashes() -> dict[str, str]:
    hashes = dict(residual_source_hashes())
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _config_hash() -> str:
    return hashlib.sha256(
        json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _output_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _profile_path(record: dict[str, Any], expected: Path) -> Path:
    recorded = record.get("path")
    if not isinstance(recorded, str):
        raise RuntimeError("saved distance profile path is missing")
    recorded_path = Path(recorded)
    recorded_absolute = (
        recorded_path if recorded_path.is_absolute() else Path.cwd() / recorded_path
    ).resolve()
    if recorded_absolute != expected.resolve():
        raise RuntimeError(
            "saved distance profile path is not the expected project path"
        )
    return expected


def _load_profile_table(
    record: dict[str, Any],
    path: Path,
    features: pd.DataFrame,
    spec: InputSpec,
    inner_fit: np.ndarray,
) -> Any:
    actual_path = _profile_path(record, path)
    normalized_record = dict(record)
    normalized_record["path"] = str(actual_path)
    return _load_verified_profile_table(
        actual_path, normalized_record, features, spec, inner_fit
    )


def _progress_callback(
    metadata: dict[str, Any], metadata_path: Path, split_seed: int
) -> Any:
    def callback(environment: Any) -> None:
        tree = int(environment.iteration) + 1
        if tree % PROGRESS_INTERVAL:
            return
        current = metadata.get("current_fit")
        if isinstance(current, dict):
            current.setdefault("progress_log", []).append(
                {"utc": datetime.now(timezone.utc).isoformat(), "tree": tree}
            )
            current["last_progress_tree"] = tree
            save_json(metadata_path, metadata)
        print(
            f"{datetime.now(timezone.utc).isoformat()} "
            f"split_seed={split_seed} tree={tree}",
            flush=True,
        )

    callback.order = 20  # type: ignore[attr-defined]
    return callback


def _fit_student(
    experiment: str,
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
        "experiment": experiment,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "outer_holdout_rows": len(holdout),
        "outer_holdout_used_for_training_or_early_stopping": False,
        "progress_log": [],
    }
    save_json(metadata_path, metadata)
    model = build_trial_classifier(experiment)
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


def _validate_prior_profile(
    profile_metadata: dict[str, Any],
    input_hashes: dict[str, str],
    spec: InputSpec,
) -> None:
    if profile_metadata.get("status") != "completed":
        raise RuntimeError("distance profile metadata is not completed")
    if profile_metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError("distance profile inputs do not match current inputs")
    if profile_metadata.get("predeclared_settings") != dict(PREDECLARED_CONFIG):
        raise RuntimeError("distance profile LightGBM configuration changed")
    current_sources = _source_hashes()
    for name, digest in profile_metadata.get("source_sha256_before", {}).items():
        if current_sources.get(name) != digest:
            raise RuntimeError(f"distance profile source fingerprint changed: {name}")
    if profile_metadata.get("config_sha256") != _config_hash():
        raise RuntimeError("distance profile configuration fingerprint changed")
    expected = _feature_spec(spec)
    if profile_metadata.get("feature_spec_original") != expected:
        raise RuntimeError("distance profile original feature schema changed")
    if profile_metadata.get("feature_spec_teacher") is None:
        raise RuntimeError("distance profile teacher feature schema is missing")


def run_experiment(
    experiment: str, data_dir: Path, artifacts_dir: Path
) -> dict[str, Any]:
    """Run exactly two sequential student fits for one fixed experiment."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown final-sprint experiment: {experiment}")
    started = time.perf_counter()
    output_name = f"final_sprint_{experiment}"
    output_dir = artifacts_dir / output_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "experiment": experiment,
        "predeclared_experiments": list(EXPERIMENTS),
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "completed_fits": [],
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "progress_interval": PROGRESS_INTERVAL,
        "baseline_settings": dict(PREDECLARED_CONFIG),
        "predeclared_settings": {
            **PREDECLARED_CONFIG,
            "num_leaves": (
                127
                if experiment
                in (LEAVES127, ROUTE_DISTANCE_TARGET_127, ROUTE_CONTEXT_TARGET_127)
                else 31
            ),
        },
        "feature_change": {
            DISTANCE_CATEGORY: "append canonical Flight Distance category",
            LEAVES127: "num_leaves 31 -> 127",
            ROUTE_DISTANCE_TARGET: "append two fixed inner-fit cross-fit encodings",
            ROUTE_DISTANCE_TARGET_127: (
                "append two fixed inner-fit cross-fit encodings and use num_leaves 127"
            ),
            ROUTE_CONTEXT_TARGET: (
                "append four fixed inner-fit route/context cross-fit encodings"
            ),
            ROUTE_CONTEXT_TARGET_127: (
                "append four fixed inner-fit route/context cross-fit encodings "
                "and use num_leaves 127"
            ),
            ROUTE_HIERARCHICAL: (
                "append four fixed inner-fit hierarchical route encodings"
            ),
            ROUTE_TEACHER_RESIDUAL: (
                "append four fixed inner-fit frozen-teacher residual route encodings"
            ),
        }[experiment],
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
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
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        source_before = _source_hashes()
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        if len(train) != EXPECTED_TRAIN_ROWS:
            raise RuntimeError("training row count does not match the approved screen")
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        original_features = train.loc[:, list(spec.feature_columns)]
        prior_lightgbm = _load_prior_lightgbm(artifacts_dir, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        teacher_features, teacher_spec = append_teacher_feature(
            original_features, teacher_cache["probabilities"], spec
        )
        profile_metadata_path = (
            artifacts_dir / "distance_profile_feature" / "compare_metadata.json"
        )
        profile_metadata = json.loads(profile_metadata_path.read_text(encoding="utf-8"))
        _validate_prior_profile(profile_metadata, input_before, spec)
        metadata.update(
            {
                "python": sys.version,
                "packages": package_versions(),
                "threads": MAX_THREADS,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "baseline_config_sha256": _config_hash(),
                "feature_spec_original": _feature_spec(spec),
                "feature_spec_teacher": _feature_spec(teacher_spec),
                "reused_artifacts": {
                    "source_teacher_metadata": sha256_file(
                        teacher_cache["metadata_path"]
                    ),
                    "distance_profile_metadata": sha256_file(profile_metadata_path),
                },
                "split_seeds": list(SPLIT_SEEDS),
                "internal_seed": INTERNAL_SEED,
                "early_stopping": {
                    "metric": "auc",
                    "rounds": 100,
                    "callbacks": ["early_stopping", "log_evaluation", "progress"],
                },
            }
        )
        save_json(metadata_path, metadata)
        if _input_hashes(data_dir) != input_before:
            raise RuntimeError("input CSV fingerprint changed during preflight")

        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        for fit_number, seed in enumerate(SPLIT_SEEDS, 1):
            outer_fit, holdout, inner_fit, inner_validation, split_record = (
                _load_split_arrays(artifacts_dir, train, y, prior_lightgbm, seed)
            )
            if (len(inner_fit), len(inner_validation), len(holdout)) != (
                503737,
                55971,
                139927,
            ):
                raise RuntimeError("recorded split row counts changed")
            split_name = "outer" if seed == 42 else "confirmation"
            profile_record = _profile_cache_record(profile_metadata, seed, inner_fit)
            profile_path = (
                artifacts_dir
                / "distance_profile_feature"
                / f"distance_profile_seed{seed}.npz"
            )
            profile_table = _load_profile_table(
                profile_record,
                profile_path,
                original_features,
                spec,
                inner_fit,
            )
            base_features, base_spec = append_distance_profile_features(
                teacher_features, teacher_spec, profile_table
            )
            route_metadata: dict[str, Any] | None = None
            if experiment == DISTANCE_CATEGORY:
                student_features, student_spec = append_distance_category(
                    base_features, base_spec
                )
            elif experiment in (
                ROUTE_DISTANCE_TARGET,
                ROUTE_DISTANCE_TARGET_127,
                ROUTE_CONTEXT_TARGET,
                ROUTE_CONTEXT_TARGET_127,
            ):
                route = fit_route_target_encoding(
                    base_features,
                    base_spec,
                    y[inner_fit],
                    inner_fit,
                    inner_validation,
                    holdout,
                    include_context=experiment
                    in (ROUTE_CONTEXT_TARGET, ROUTE_CONTEXT_TARGET_127),
                )
                student_features, student_spec = route.features, route.spec
                route_metadata = route.metadata
            elif experiment == ROUTE_HIERARCHICAL:
                route = fit_hierarchical_route_encoding(
                    base_features,
                    base_spec,
                    y[inner_fit],
                    inner_fit,
                    inner_validation,
                    holdout,
                )
                student_features, student_spec = route.features, route.spec
                route_metadata = route.metadata
            elif experiment == ROUTE_TEACHER_RESIDUAL:
                route = fit_route_teacher_residual_encoding(
                    base_features,
                    base_spec,
                    y[inner_fit],
                    inner_fit,
                    inner_validation,
                    holdout,
                )
                student_features, student_spec = route.features, route.spec
                route_metadata = route.metadata
            else:
                student_features, student_spec = base_features, base_spec

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
            baseline_auc[seed] = baseline_metrics["roc_auc"]
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
            predictions, fit_record = _fit_student(
                experiment,
                student_features,
                student_spec,
                y,
                holdout,
                inner_fit,
                inner_validation,
                metadata,
                metadata_path,
                seed,
            )
            reference_ids = ids[holdout]
            candidate = weighted_blend_predictions(
                reference_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(dtype=np.float64),
                reference_ids,
                predictions,
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )["blend_probability"].to_numpy(dtype=np.float64)
            candidate_metrics = _metrics(y[holdout], candidate)
            candidate_auc[seed] = candidate_metrics["roc_auc"]
            validation = pd.DataFrame(
                {
                    "row_index": holdout,
                    ID_COLUMN: reference_ids,
                    "student_probability": predictions,
                    "catboost_probability": slow["probability"].to_numpy(),
                    "baseline_probability": baseline,
                    "candidate_probability": candidate,
                }
            )
            if list(validation.columns) != list(VALIDATION_COLUMNS):
                raise RuntimeError("final-sprint validation schema changed")
            validation_path = output_dir / f"validation_{experiment}_seed{seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            fit_record.update(
                {
                    "split_seed": seed,
                    "split": split_record,
                    "feature_spec": _feature_spec(student_spec),
                    "profile_fit_membership": {
                        "rows": len(inner_fit),
                        "sha256": _array_fingerprint(inner_fit),
                    },
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                    - baseline_metrics["roc_auc"],
                    "validation_predictions_path": str(validation_path),
                    "validation_predictions_sha256": sha256_file(validation_path),
                }
            )
            if route_metadata is not None:
                fit_record["route_target_encoding"] = route_metadata
            metadata["validation_results"].append(fit_record)
            metadata["completed_fits"].append(f"student_seed_{seed}")
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del predictions, candidate, validation, base_features, student_features
            gc.collect()

        metadata["selection"] = final_sprint_gate(
            experiment, baseline_auc, candidate_auc
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
                finalization_errors.append(error)
        if input_before is not None:
            try:
                metadata["input_sha256_after"] = _input_hashes(data_dir)
                if metadata["input_sha256_after"] != input_before:
                    finalization_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
            except BaseException as error:
                finalization_errors.append(error)
        if source_before is not None:
            try:
                metadata["source_sha256_after"] = _source_hashes()
                if metadata["source_sha256_after"] != source_before:
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
                "timeout" if isinstance(caught, FinalSprintTimeout) else "failed"
            )
            metadata["selection"] = None
        else:
            metadata["status"] = "completed"
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = _output_size(output_dir)
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
                actual_bytes = _output_size(output_dir)
                if actual_bytes == metadata["output_bytes"]:
                    break
                metadata["output_bytes"] = actual_bytes
        except BaseException as error:
            if caught is None:
                caught = error
                metadata["status"] = "failed"
                metadata["selection"] = None
                metadata["error"] = _error_record(error)
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.experiment, args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
