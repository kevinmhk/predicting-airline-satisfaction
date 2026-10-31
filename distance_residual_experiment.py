"""Run the fixed distance-group residual LightGBM diagnostic."""

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

# Scientific imports intentionally follow the thread-limit environment setup.
# ruff: noqa: E402
import numpy as np
import pandas as pd

from distance_profile_experiment import (
    DISTANCE_COLUMN,
    _profile_mean_columns,
    append_distance_profile_features,
    apply_distance_profile,
)
from distance_profile_experiment import (
    _source_hashes as profile_source_hashes,
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
from source_neighbor_experiment import (
    _load_verified_profile_table,
    _metrics_from_saved_baseline,
    _profile_cache_record,
    _read_saved_distance_validation,
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
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)
from weighted_blend_experiment import weighted_blend_predictions

RESIDUAL_PREFIX = "distance_group_residual__"
OUTPUT_NAME = "distance_residual_feature"
SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
SPLIT_SEEDS = tuple(SPLIT_FILES)
INTERNAL_SEED = 42
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_FEATURE_COLUMNS = 21
EXPECTED_NUMERIC_COLUMNS = 17
EXPECTED_CATEGORICAL_COLUMNS = 4
EXPECTED_RESIDUAL_FEATURES = 16
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 100
VALIDATION_COLUMNS = (
    "row_index",
    ID_COLUMN,
    "student_probability",
    "catboost_probability",
    "baseline_probability",
    "candidate_probability",
)


class DistanceResidualTimeout(TimeoutError):
    """Raised when the fixed residual diagnostic reaches its deadline."""


def distance_residual_feature_names(mean_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return residual names for the non-distance original numeric predictors."""
    if DISTANCE_COLUMN in mean_columns or len(set(mean_columns)) != len(mean_columns):
        raise ValueError(
            "distance residual columns must exclude duplicate Flight Distance"
        )
    names = tuple(f"{RESIDUAL_PREFIX}{column}" for column in mean_columns)
    if len(set(names)) != len(names):
        raise ValueError("distance residual feature names collide")
    return names


def _validate_feature_frame(features: pd.DataFrame, spec: InputSpec) -> None:
    if list(features.columns) != list(spec.feature_columns):
        raise ValueError("features must match the declared predictor schema exactly")
    if features.columns.duplicated().any():
        raise ValueError("features must not contain duplicate columns")
    if ID_COLUMN in features or TARGET_COLUMN in features:
        raise ValueError("distance residuals must not receive id or satisfaction")


def append_distance_residual_features(
    features: pd.DataFrame, spec: InputSpec, table: Any
) -> tuple[pd.DataFrame, InputSpec]:
    """Append original-numeric minus mapped distance-group-mean features."""
    _validate_feature_frame(features, spec)
    mean_columns = tuple(table.mean_columns)
    residual_columns = distance_residual_feature_names(mean_columns)
    if any(column not in spec.numeric_columns for column in mean_columns):
        raise ValueError("distance residual table is not based on declared numerics")
    collisions = set(residual_columns).intersection(features.columns)
    if collisions:
        raise ValueError(
            f"distance residual columns already exist: {sorted(collisions)}"
        )
    mapped = apply_distance_profile(features, table)
    residual = pd.DataFrame(
        {
            name: features[column].to_numpy(dtype=np.float64)
            - mapped[f"distance_group_mean__{column}"].to_numpy(dtype=np.float64)
            for column, name in zip(mean_columns, residual_columns, strict=True)
        },
        index=features.index,
    )
    extended = InputSpec(
        feature_columns=spec.feature_columns + residual_columns,
        numeric_columns=spec.numeric_columns + residual_columns,
        categorical_columns=spec.categorical_columns,
    )
    return pd.concat([features.copy(), residual], axis=1), extended


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _validate_fixed_schema(spec: InputSpec) -> None:
    mean_columns = _profile_mean_columns(spec)
    if (
        len(spec.feature_columns) != EXPECTED_FEATURE_COLUMNS
        or len(spec.numeric_columns) != EXPECTED_NUMERIC_COLUMNS
        or len(spec.categorical_columns) != EXPECTED_CATEGORICAL_COLUMNS
        or len(mean_columns) != EXPECTED_RESIDUAL_FEATURES
    ):
        raise RuntimeError("input schema is not the approved 21/17/4/16 contract")


def _source_hashes() -> dict[str, str]:
    hashes = dict(profile_source_hashes())
    from source_neighbor_experiment import _source_hashes as neighbor_source_hashes

    hashes.update(neighbor_source_hashes())
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _config_hash() -> str:
    return hashlib.sha256(
        json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


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
            f"{datetime.now(timezone.utc).isoformat()} "
            f"split_seed={split_seed} tree={iteration}",
            flush=True,
        )

    callback.order = 20  # type: ignore[attr-defined]
    return callback


def _output_size(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())


def _fit_residual_student(
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
        "status": "running",
        "model": "lightgbm_distance_residual_student",
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "outer_holdout_rows": len(holdout),
        "outer_holdout_used_for_training_or_early_stopping": False,
        "progress_log": [],
    }
    save_json(metadata_path, metadata)
    model = build_lightgbm_classifier()
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
        "standalone": _metrics(y[holdout], predictions),
        "fit_seconds": round(float(fit_seconds), 3),
        "settings": dict(PREDECLARED_CONFIG),
        "actual_iterations": details["actual_iterations"],
        "best_iteration": details["best_iteration"],
        "native_category_levels_from_inner_fit": {
            name: len(values) for name, values in levels.items()
        },
        "resources": details["resources"],
        "early_stopping": {
            "metric": "auc",
            "rounds": 100,
            "callbacks": ["early_stopping", "log_evaluation", "progress"],
        },
    }
    del model
    gc.collect()
    return predictions, result


def distance_residual_gate(
    baseline_auc: dict[int, float], candidate_auc: dict[int, float]
) -> dict[str, Any]:
    """Apply the existing two-seed gate with the residual decision name."""
    result = teacher_feature_gate(baseline_auc, candidate_auc)
    if result["qualifies"]:
        result["decision"] = "select_distance_residual_feature"
    return result


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly two sequential residual-feature LightGBM student fits."""
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
        "feature_change": {
            "only": "distance_group_residual__<original_numeric_column>",
            "distance_column": DISTANCE_COLUMN,
            "residual_feature_count": EXPECTED_RESIDUAL_FEATURES,
            "excluded": [
                "Flight Distance",
                "distance_group_count",
                "source_teacher_probability",
                "categorical_columns",
                ID_COLUMN,
                TARGET_COLUMN,
            ],
        },
    }
    save_json(metadata_path, metadata)

    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False

    def on_timeout(_signum: int, _frame: Any) -> None:
        raise DistanceResidualTimeout("distance residual diagnostic timed out")

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
        if profile_metadata.get("status") != "completed":
            raise RuntimeError("distance profile metadata is not completed")
        if profile_metadata.get("input_sha256_before") != input_before:
            raise RuntimeError("distance profile inputs do not match current inputs")
        if profile_metadata.get("predeclared_settings") != dict(PREDECLARED_CONFIG):
            raise RuntimeError("distance profile LightGBM configuration changed")
        recorded_sources = profile_metadata.get("source_sha256_before", {})
        current_sources = _source_hashes()
        if any(
            current_sources.get(name) != digest
            for name, digest in recorded_sources.items()
        ):
            raise RuntimeError("distance profile source fingerprints changed")
        if profile_metadata.get("config_sha256") != _config_hash():
            raise RuntimeError("distance profile configuration fingerprint changed")
        if profile_metadata.get("feature_spec_original") != _feature_spec(spec):
            raise RuntimeError("distance profile original feature schema changed")
        if profile_metadata.get("feature_spec_teacher") is None:
            raise RuntimeError("distance profile teacher feature schema is missing")

        residual_columns = distance_residual_feature_names(_profile_mean_columns(spec))
        metadata.update(
            {
                "python": sys.version,
                "packages": package_versions(),
                "threads": MAX_THREADS,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "config_sha256": _config_hash(),
                "feature_spec_original": _feature_spec(spec),
                "feature_spec_teacher": _feature_spec(teacher_spec),
                "residual_feature_definitions": residual_columns,
                "split_seeds": list(SPLIT_SEEDS),
                "internal_seed": INTERNAL_SEED,
                "recorded_splits": {},
                "reused_artifacts": {
                    "source_teacher_metadata": sha256_file(
                        teacher_cache["metadata_path"]
                    ),
                    "distance_profile_metadata": sha256_file(profile_metadata_path),
                },
            }
        )
        save_json(metadata_path, metadata)
        if _input_hashes(data_dir) != input_before:
            raise RuntimeError("input CSV fingerprint changed during preflight")

        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
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
            metadata["recorded_splits"][split_name] = split_record
            profile_record = _profile_cache_record(profile_metadata, seed, inner_fit)
            profile_path = _profile_path(
                profile_record,
                artifacts_dir
                / "distance_profile_feature"
                / f"distance_profile_seed{seed}.npz",
            )
            profile_table = _load_verified_profile_table(
                profile_path, profile_record, original_features, spec, inner_fit
            )
            profile_features, profile_spec = append_distance_profile_features(
                teacher_features, teacher_spec, profile_table
            )
            student_features, student_spec = append_distance_residual_features(
                profile_features, profile_spec, profile_table
            )
            if (
                len(student_spec.feature_columns) != 55
                or len(student_spec.numeric_columns) != 51
                or len(student_spec.categorical_columns) != 4
            ):
                raise RuntimeError("residual student schema is not 55/51/4")
            metadata["feature_spec_profile"] = _feature_spec(profile_spec)
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
            baseline_auc[seed] = baseline_metrics["roc_auc"]
            metadata["current_fit"] = {
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
            predictions, fit_record = _fit_residual_student(
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
                raise RuntimeError("residual validation schema changed")
            validation_path = (
                output_dir / f"validation_distance_residual_seed{seed}.csv"
            )
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            fit_record.update(
                {
                    "split_seed": seed,
                    "split": split_record,
                    "profile_fit_membership": {
                        "rows": len(inner_fit),
                        "sha256": _array_fingerprint(inner_fit),
                    },
                    "feature_spec": _feature_spec(student_spec),
                    "residual_feature_definitions": residual_columns,
                    "profile_table_path": str(profile_path),
                    "profile_table_sha256": sha256_file(profile_path),
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                    - baseline_metrics["roc_auc"],
                    "validation_predictions_path": str(validation_path),
                    "validation_predictions_sha256": sha256_file(validation_path),
                    "internal_validation": {
                        "seed": INTERNAL_SEED,
                        "fit_rows": len(inner_fit),
                        "validation_rows": len(inner_validation),
                        "outer_holdout_used": False,
                    },
                }
            )
            metadata["validation_results"].append(fit_record)
            metadata["completed_split_seeds"].append(seed)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del predictions, candidate, validation, profile_features, student_features
            gc.collect()

        metadata["selection"] = distance_residual_gate(baseline_auc, candidate_auc)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, DistanceResidualTimeout) else "failed"
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
        metadata["current_fit"] = None
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
            if caught_error is None:
                caught_error = integrity_error
                metadata["error"] = _error_record(integrity_error)
            else:
                metadata["finalization_error"] = _error_record(integrity_error)
        if metadata["status"] != "completed":
            metadata["selection"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        try:
            save_json(
                output_dir / "protected_baseline_sha256.json",
                metadata.get("protected_artifact_sha256_after", protected_before or {}),
            )
            for _ in range(4):
                metadata["output_bytes"] = _output_size(output_dir)
                if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
                    raise RuntimeError("output exceeded 100 MiB")
                save_json(metadata_path, metadata)
                actual_size = _output_size(output_dir)
                if actual_size == metadata["output_bytes"]:
                    break
            else:
                metadata["output_bytes"] = _output_size(output_dir)
            if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
                raise RuntimeError("output exceeded 100 MiB after finalization")
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
    result = run_experiment(args.data_dir, args.artifacts_dir)
    print(json.dumps(result["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
