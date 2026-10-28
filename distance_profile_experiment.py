"""Run the fixed label-free distance-group profile LightGBM diagnostic."""

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

from catboost_teacher_feature import _load_teacher_student_predictions
from five_fold_cv import _metrics
from lightgbm_capacity import (
    _array_fingerprint,
    _artifact_snapshot,
    _error_record,
    _input_hashes,
    _load_components,
    _load_split_arrays,
    _package_versions,
)
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _fit_callbacks,
    _fit_with_inner_validation,
    build_lightgbm_classifier,
    prepare_lightgbm_frame,
)
from source_teacher_cv import _verify_teacher_cache
from source_teacher_experiment import (
    _load_prior_lightgbm,
    append_teacher_feature,
    teacher_feature_gate,
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
)
from weighted_blend_experiment import weighted_blend_predictions

DISTANCE_COLUMN = "Flight Distance"
PROFILE_COUNT_COLUMN = "distance_group_count"
PROFILE_MEAN_PREFIX = "distance_group_mean__"
SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
SPLIT_SEEDS = tuple(SPLIT_FILES)
INTERNAL_SEED = 42
BASELINE_AUC = {42: 0.9580277365442882, 2024: 0.9584612650032445}
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_NAME = "distance_profile_feature"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 100
EXPECTED_NUMERIC_COLUMNS = 17
EXPECTED_PROFILE_MEAN_COLUMNS = 16
EXPECTED_FEATURE_COLUMNS = 21
EXPECTED_PROFILE_FEATURES = 17


class DistanceProfileTimeout(TimeoutError):
    """Raised when the fixed distance-profile diagnostic reaches its deadline."""


@dataclass(frozen=True)
class DistanceProfileTable:
    """Label-free profile statistics fitted from one model-fit partition."""

    distance_column: str
    mean_columns: tuple[str, ...]
    group_counts: pd.Series
    group_means: pd.DataFrame
    global_means: pd.Series


def distance_profile_feature_names(mean_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return the fixed count-plus-mean profile column names."""
    if len(set(mean_columns)) != len(mean_columns):
        raise ValueError("distance profile mean columns must be unique")
    names = (PROFILE_COUNT_COLUMN,) + tuple(
        f"{PROFILE_MEAN_PREFIX}{column}" for column in mean_columns
    )
    if len(set(names)) != len(names):
        raise ValueError("distance profile feature names collide")
    return names


def _validate_feature_frame(features: pd.DataFrame, spec: InputSpec) -> None:
    if list(features.columns) != list(spec.feature_columns):
        raise ValueError("features must match the declared predictor schema exactly")
    if features.columns.duplicated().any():
        raise ValueError("features must not contain duplicate columns")
    if ID_COLUMN in features or TARGET_COLUMN in features:
        raise ValueError("distance profiles must not receive id or satisfaction")


def _profile_mean_columns(spec: InputSpec) -> tuple[str, ...]:
    if DISTANCE_COLUMN not in spec.numeric_columns:
        raise ValueError("Flight Distance must be a declared numeric predictor")
    mean_columns = tuple(
        column for column in spec.numeric_columns if column != DISTANCE_COLUMN
    )
    distance_profile_feature_names(mean_columns)
    return mean_columns


def fit_distance_profile_table(
    features: pd.DataFrame, spec: InputSpec
) -> DistanceProfileTable:
    """Fit distance-group count and numeric means from predictors only."""
    _validate_feature_frame(features, spec)
    if features.empty:
        raise ValueError("distance profile fit rows must not be empty")
    mean_columns = _profile_mean_columns(spec)
    for column in (DISTANCE_COLUMN, *mean_columns):
        if not pd.api.types.is_numeric_dtype(features[column]):
            raise TypeError(f"distance profile column must be numeric: {column}")
    grouped = features.groupby(DISTANCE_COLUMN, sort=True, dropna=True)
    group_counts = grouped.size().astype(np.float64)
    group_means = grouped[list(mean_columns)].mean()
    global_means = features.loc[:, list(mean_columns)].mean(skipna=True)
    return DistanceProfileTable(
        distance_column=DISTANCE_COLUMN,
        mean_columns=mean_columns,
        group_counts=group_counts,
        group_means=group_means,
        global_means=global_means,
    )


def _validate_profile_table(table: DistanceProfileTable) -> None:
    if table.distance_column != DISTANCE_COLUMN:
        raise ValueError("distance profile grouping column changed")
    distance_profile_feature_names(table.mean_columns)
    if list(table.group_means.columns) != list(table.mean_columns):
        raise ValueError("distance profile mean table schema changed")
    if not table.group_counts.index.equals(table.group_means.index):
        raise ValueError("distance profile count and mean keys differ")
    if list(table.global_means.index) != list(table.mean_columns):
        raise ValueError("distance profile fallback schema changed")


def apply_distance_profile(
    features: pd.DataFrame, table: DistanceProfileTable
) -> pd.DataFrame:
    """Map a fitted profile table, using global means for unknown distance keys."""
    _validate_profile_table(table)
    if DISTANCE_COLUMN not in features.columns:
        raise ValueError("features are missing Flight Distance")
    if any(column not in features.columns for column in table.mean_columns):
        raise ValueError("features are missing distance profile mean columns")
    for column in (DISTANCE_COLUMN, *table.mean_columns):
        if not pd.api.types.is_numeric_dtype(features[column]):
            raise TypeError(f"distance profile column must be numeric: {column}")

    keys = features[DISTANCE_COLUMN]
    known = keys.notna() & keys.isin(table.group_means.index)
    count_values = keys.map(table.group_counts).fillna(0.0).to_numpy(dtype=np.float64)
    result = pd.DataFrame({PROFILE_COUNT_COLUMN: count_values}, index=features.index)
    known_mask = known.to_numpy()
    for column in table.mean_columns:
        values = np.full(
            len(features),
            table.global_means[column],
            dtype=np.float64,
        )
        if known_mask.any():
            values[known_mask] = (
                keys[known].map(table.group_means[column]).to_numpy(dtype=np.float64)
            )
        result[f"{PROFILE_MEAN_PREFIX}{column}"] = values
    return result


def append_distance_profile_features(
    features: pd.DataFrame, spec: InputSpec, table: DistanceProfileTable
) -> tuple[pd.DataFrame, InputSpec]:
    """Append profile columns without mutating the supplied frame or spec."""
    _validate_feature_frame(features, spec)
    profile_columns = distance_profile_feature_names(table.mean_columns)
    collisions = set(profile_columns).intersection(features.columns)
    if collisions:
        raise ValueError(
            f"distance profile columns already exist: {sorted(collisions)}"
        )
    if any(column not in spec.numeric_columns for column in table.mean_columns):
        raise ValueError("distance profile table is not based on declared numerics")
    profile = apply_distance_profile(features, table)
    extended = InputSpec(
        feature_columns=spec.feature_columns + profile_columns,
        numeric_columns=spec.numeric_columns + profile_columns,
        categorical_columns=spec.categorical_columns,
    )
    return pd.concat([features.copy(), profile], axis=1), extended


def _save_profile_table(path: Path, table: DistanceProfileTable) -> None:
    """Save one profile table without labels, IDs, or model state."""
    _validate_profile_table(table)
    np.savez_compressed(
        path,
        distance=np.asarray(table.group_means.index, dtype=np.float64),
        group_count=table.group_counts.to_numpy(dtype=np.float64),
        group_means=table.group_means.to_numpy(dtype=np.float64),
    )


def _profile_record(table: DistanceProfileTable, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "distance_column": table.distance_column,
        "mean_columns": list(table.mean_columns),
        "group_rows": len(table.group_means),
        "fallback_global_means": {
            column: (
                None
                if pd.isna(table.global_means[column])
                else float(table.global_means[column])
            )
            for column in table.mean_columns
        },
    }


def distance_profile_gate(
    baseline_auc: dict[int, float], candidate_auc: dict[int, float]
) -> dict[str, Any]:
    """Require the fixed candidate gain independently on both holdouts."""
    result = teacher_feature_gate(baseline_auc, candidate_auc)
    if result["qualifies"]:
        result["decision"] = "select_distance_profile_feature"
    return result


def _source_hashes() -> dict[str, str]:
    names = (
        "distance_profile_experiment.py",
        "catboost_teacher_feature.py",
        "compare.py",
        "five_fold_cv.py",
        "lightgbm_capacity.py",
        "lightgbm_experiment.py",
        "source_teacher_cv.py",
        "source_teacher_experiment.py",
        "train.py",
        "weighted_blend_experiment.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _config_hash() -> str:
    encoded = json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _progress_callback(
    metadata: dict[str, Any], metadata_path: Path, split_seed: int
) -> Any:
    def callback(environment: Any) -> None:
        iteration = int(environment.iteration) + 1
        if iteration % PROGRESS_INTERVAL:
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


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _validate_fixed_schema(spec: InputSpec) -> None:
    mean_columns = _profile_mean_columns(spec)
    if len(spec.feature_columns) != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError("original feature count is not 21")
    if len(spec.numeric_columns) != EXPECTED_NUMERIC_COLUMNS:
        raise RuntimeError("original numeric feature count is not 17")
    if len(mean_columns) != EXPECTED_PROFILE_MEAN_COLUMNS:
        raise RuntimeError("distance profile mean count is not 16")
    if len(spec.categorical_columns) != 4:
        raise RuntimeError("original categorical feature count is not 4")


def _profile_feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    expected = _feature_spec(spec)
    if (
        len(spec.feature_columns)
        != EXPECTED_FEATURE_COLUMNS + 1 + EXPECTED_PROFILE_FEATURES
    ):
        raise RuntimeError("distance profile student feature count is not 39")
    if (
        len(spec.numeric_columns)
        != EXPECTED_NUMERIC_COLUMNS + 1 + EXPECTED_PROFILE_FEATURES
    ):
        raise RuntimeError("distance profile student numeric count is not 35")
    return expected


def _metrics_from_components(
    ids: np.ndarray,
    slow: pd.DataFrame,
    baseline_student: np.ndarray,
    candidate_student: np.ndarray,
    y: np.ndarray,
    holdout: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    reference_ids = ids[holdout]
    baseline = weighted_blend_predictions(
        reference_ids,
        slow[ID_COLUMN].to_numpy(),
        slow["probability"].to_numpy(),
        reference_ids,
        baseline_student,
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    baseline_values = baseline["blend_probability"].to_numpy()
    candidate = weighted_blend_predictions(
        reference_ids,
        slow[ID_COLUMN].to_numpy(),
        slow["probability"].to_numpy(),
        reference_ids,
        candidate_student,
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    candidate_values = candidate["blend_probability"].to_numpy()
    result = pd.DataFrame(
        {
            "row_index": holdout,
            ID_COLUMN: reference_ids,
            "student_probability": candidate_student,
            "catboost_probability": slow["probability"].to_numpy(),
            "baseline_probability": baseline_values,
            "candidate_probability": candidate_values,
        }
    )
    return (
        result,
        _metrics(y[holdout], baseline_values),
        _metrics(y[holdout], candidate_values),
    )


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly two bounded LightGBM student fits with profile tables."""
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
        "completed_split_seeds": [],
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "progress_interval": PROGRESS_INTERVAL,
        "predeclared_settings": dict(PREDECLARED_CONFIG),
    }
    save_json(metadata_path, metadata)

    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    signals_installed = False
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)

    def on_timeout(_signum: int, _frame: Any) -> None:
        raise DistanceProfileTimeout("distance profile diagnostic timed out")

    try:
        signal.signal(signal.SIGALRM, on_timeout)
        signal.signal(signal.SIGTERM, on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        source_before = _source_hashes()
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        _validate_fixed_schema(spec)
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        original_features = train.loc[:, list(spec.feature_columns)]
        prior_lightgbm = _load_prior_lightgbm(artifacts_dir, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        teacher_features, teacher_spec = append_teacher_feature(
            original_features, teacher_cache["probabilities"], spec
        )
        metadata.update(
            {
                "python": sys.version,
                "packages": _package_versions(),
                "threads": MAX_THREADS,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "feature_spec_original": _feature_spec(spec),
                "feature_spec_teacher": _feature_spec(teacher_spec),
                "teacher_cache": {
                    "metadata_path": str(teacher_cache["metadata_path"]),
                    "metadata_sha256": sha256_file(teacher_cache["metadata_path"]),
                    "prediction_path": str(teacher_cache["cache_path"]),
                    "prediction_sha256": teacher_cache["cache_sha256"],
                    "rows": len(teacher_cache["probabilities"]),
                },
                "config_sha256": _config_hash(),
                "split_seeds": list(SPLIT_SEEDS),
                "internal_seed": INTERNAL_SEED,
                "feature_change": {
                    "only": "distance_group_count_and_means",
                    "distance_column": DISTANCE_COLUMN,
                    "profile_feature_count": EXPECTED_PROFILE_FEATURES,
                    "profile_mean_columns": list(_profile_mean_columns(spec)),
                },
                "recorded_splits": {},
            }
        )
        save_json(metadata_path, metadata)

        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            outer_fit, holdout, inner_fit, inner_validation, split_record = (
                _load_split_arrays(artifacts_dir, train, y, prior_lightgbm, seed)
            )
            split_name = "outer" if seed == 42 else "confirmation"
            metadata["recorded_splits"][split_name] = split_record
            slow_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            teacher_path = (
                artifacts_dir
                / "source_teacher_feature"
                / f"validation_source_teacher_seed{seed}.csv"
            )
            slow = _load_components(slow_path, train, holdout)
            teacher_student = _load_teacher_student_predictions(
                teacher_path, holdout, ids[holdout]
            )
            if not np.allclose(
                teacher_student["teacher_probability"].to_numpy(),
                teacher_cache["probabilities"][holdout],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError("saved holdout teacher features differ from cache")
            baseline_blend = weighted_blend_predictions(
                ids[holdout],
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                teacher_student[ID_COLUMN].to_numpy(),
                teacher_student["student_probability"].to_numpy(),
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )
            baseline_values = baseline_blend["blend_probability"].to_numpy()
            baseline_metrics = _metrics(y[holdout], baseline_values)
            if abs(baseline_metrics["roc_auc"] - BASELINE_AUC[seed]) > 1e-12:
                raise RuntimeError(f"recomputed baseline changed for split seed {seed}")
            if not np.allclose(
                baseline_values,
                teacher_student["candidate_probability"].to_numpy(),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "saved teacher candidate is not the current baseline"
                )

            table = fit_distance_profile_table(original_features.iloc[inner_fit], spec)
            table_path = output_dir / f"distance_profile_seed{seed}.npz"
            _save_profile_table(table_path, table)
            profile_features, profile_spec = append_distance_profile_features(
                teacher_features, teacher_spec, table
            )
            _profile_feature_spec(profile_spec)
            metadata["current_fit"] = {
                "split_seed": seed,
                "status": "running",
                "fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_holdout_rows": len(holdout),
                "profile_fit_rows": len(inner_fit),
                "profile_fit_membership": {
                    "rows": len(inner_fit),
                    "sha256": _array_fingerprint(inner_fit),
                },
                "table": _profile_record(table, table_path),
                "baseline": baseline_metrics,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)

            model = build_lightgbm_classifier()
            _levels, fit_seconds, fit_details = _fit_with_inner_validation(
                model,
                profile_features,
                y,
                inner_fit,
                inner_validation,
                profile_spec,
                callbacks=_fit_callbacks()
                + [_progress_callback(metadata, metadata_path, seed)],
            )
            predictions = np.asarray(
                model.predict_proba(
                    prepare_lightgbm_frame(
                        profile_features.iloc[holdout], profile_spec, _levels
                    )
                )[:, 1],
                dtype=np.float64,
            )
            validate_probabilities(predictions, len(holdout))
            output, _baseline_check, candidate_metrics = _metrics_from_components(
                ids,
                slow,
                teacher_student["student_probability"].to_numpy(),
                predictions,
                y,
                holdout,
            )
            if not np.allclose(
                output["baseline_probability"].to_numpy(),
                baseline_values,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError("baseline changed while constructing candidate")
            validation_path = output_dir / f"validation_distance_profile_seed{seed}.csv"
            output.to_csv(validation_path, index=False, float_format="%.17g")
            result = {
                "split_seed": seed,
                "student": _metrics(y[holdout], predictions),
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "gain_vs_current": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "fit_seconds": round(fit_seconds, 3),
                "best_iteration": fit_details["best_iteration"],
                "actual_iterations": fit_details["actual_iterations"],
                "settings": dict(PREDECLARED_CONFIG),
                "resources": fit_details["resources"],
                "validation_predictions_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "profile_table": _profile_record(table, table_path),
                "split": split_record,
                "table_fit_membership": {
                    "rows": len(inner_fit),
                    "sha256": _array_fingerprint(inner_fit),
                },
                "internal_validation": {
                    "seed": INTERNAL_SEED,
                    "fit_rows": len(inner_fit),
                    "validation_rows": len(inner_validation),
                    "outer_holdout_used": False,
                },
            }
            metadata["validation_results"].append(result)
            metadata["completed_split_seeds"].append(seed)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            baseline_auc[seed] = baseline_metrics["roc_auc"]
            candidate_auc[seed] = candidate_metrics["roc_auc"]
            del model, predictions, profile_features, table
            gc.collect()

        metadata["selection"] = distance_profile_gate(baseline_auc, candidate_auc)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, DistanceProfileTimeout) else "failed"
        )
        metadata["error"] = _error_record(error)
        metadata["selection"] = None
    finally:
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                finalization_errors.append(error)
        try:
            input_after = _input_hashes(data_dir)
            metadata["input_sha256_after"] = input_after
            if input_before is not None and input_after != input_before:
                finalization_errors.append(
                    RuntimeError("input CSV fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        try:
            source_after = _source_hashes()
            metadata["source_sha256_after"] = source_after
            if source_before is not None and source_after != source_before:
                finalization_errors.append(RuntimeError("source fingerprint changed"))
        except BaseException as error:
            finalization_errors.append(error)
        try:
            protected_after = _artifact_snapshot(artifacts_dir, output_dir)
            metadata["protected_artifact_sha256_before"] = protected_before
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
        metadata["current_fit"] = None
        if finalization_errors:
            integrity_error = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata["status"] = "failed"
            metadata["integrity_error"] = _error_record(integrity_error)
            if caught_error is None:
                caught_error = integrity_error
                metadata["error"] = _error_record(integrity_error)
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("output exceeded 100 MiB")
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught_error is None:
                caught_error = budget_error
                metadata["error"] = _error_record(budget_error)
        if metadata["status"] != "completed":
            metadata["selection"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        try:
            save_json(
                output_dir / "protected_baseline_sha256.json",
                metadata.get("protected_artifact_sha256_after", protected_before or {}),
            )
            for _ in range(4):
                save_json(metadata_path, metadata)
                actual_bytes = sum(
                    path.stat().st_size
                    for path in output_dir.rglob("*")
                    if path.is_file()
                )
                if actual_bytes > OUTPUT_BUDGET_BYTES:
                    raise RuntimeError("output exceeded 100 MiB after finalization")
                if actual_bytes == metadata["output_bytes"]:
                    break
                metadata["output_bytes"] = actual_bytes
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught_error is None:
                caught_error = error
                metadata["error"] = _error_record(error)
            else:
                metadata["finalization_error"] = _error_record(error)
            save_json(metadata_path, metadata)
    if caught_error is not None:
        raise caught_error
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
