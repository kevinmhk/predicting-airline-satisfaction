"""Run the fixed external-source LightGBM teacher-feature diagnostic."""

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
from importlib.metadata import version
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
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from compare import (
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
)
from lightgbm_capacity import (
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
    fit_category_levels,
    prepare_lightgbm_frame,
)
from source_augmentation import clean_source_rows
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

SPLIT_SEEDS = (42, 2024)
INTERNAL_SEED = 42
SOURCE_TEACHER_FEATURE = "source_teacher_probability"
OUTPUT_NAME = "source_teacher_feature"
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 500
MINIMUM_AUC_GAIN = 0.0002
BASELINE_AUC = {42: 0.9577942088132286, 2024: 0.9581967799255812}
EXPECTED_CLEAN_SOURCE_ROWS = 129859
SOURCE_AUGMENTATION_METADATA = "source_augmentation/compare_metadata.json"


class SourceTeacherTimeout(TimeoutError):
    """Raised when the fixed source-teacher diagnostic reaches its deadline."""


@dataclass(frozen=True)
class SourceTeacherFit:
    """The fitted source-only teacher and the levels used for prediction."""

    model: lgb.LGBMClassifier
    category_levels: dict[str, list[str]]
    record: dict[str, Any]


def _array_fingerprint(array: np.ndarray) -> str:
    value = np.asarray(array, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _array_record(*arrays: tuple[str, np.ndarray]) -> dict[str, Any]:
    return {
        "rows": {name: len(value) for name, value in arrays},
        "sha256": {name: _array_fingerprint(value) for name, value in arrays},
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _source_hashes() -> dict[str, str]:
    names = (
        "source_teacher_experiment.py",
        "source_augmentation.py",
        "lightgbm_capacity.py",
        "lightgbm_experiment.py",
        "weighted_blend_experiment.py",
        "compare.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _config_hash() -> str:
    return hashlib.sha256(
        json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _versions() -> dict[str, str]:
    result = package_versions()
    for name in ("lightgbm", "threadpoolctl"):
        result[name] = version(name)
    return result


def _progress_callback(
    metadata: dict[str, Any], metadata_path: Path, fit_name: str
) -> Any:
    def callback(environment: Any) -> None:
        tree = int(environment.iteration) + 1
        if tree % PROGRESS_INTERVAL:
            return
        current = metadata.get("current_fit")
        if isinstance(current, dict):
            event = {"utc": datetime.now(timezone.utc).isoformat(), "tree": tree}
            current.setdefault("progress_log", []).append(event)
            current["last_progress_tree"] = tree
            save_json(metadata_path, metadata)
        print(
            f"{datetime.now(timezone.utc).isoformat()} {fit_name} tree={tree}",
            flush=True,
        )

    callback.order = 20  # type: ignore[attr-defined]
    return callback


def _validate_teacher_inputs(
    source_x: pd.DataFrame, source_y: np.ndarray, spec: InputSpec
) -> None:
    if set(source_x.columns) != set(spec.feature_columns):
        raise ValueError("source teacher features must match the original predictors")
    if ID_COLUMN in source_x or TARGET_COLUMN in source_x:
        raise ValueError("source teacher must not receive competition IDs or labels")
    if len(source_x) != len(source_y):
        raise ValueError("source teacher features and labels must have matching rows")
    labels = np.asarray(source_y, dtype=np.int8)
    if labels.ndim != 1 or set(labels) != {0, 1}:
        raise ValueError("source teacher labels must contain both binary classes")


def fit_source_teacher(
    source_x: pd.DataFrame,
    source_y: np.ndarray,
    spec: InputSpec,
    *,
    iteration_cap: int | None = None,
    progress_callback: Any | None = None,
) -> SourceTeacherFit:
    """Fit the source-only probe, then refit at its selected round count.

    ``iteration_cap`` exists only to keep synthetic tests small. The production
    runner leaves it unset, so the approved LightGBM configuration is used.
    """
    _validate_teacher_inputs(source_x, source_y, spec)
    source_indices = np.arange(len(source_x), dtype=np.int64)
    source_fit, source_validation = make_catboost_internal_split(
        source_indices, source_y, seed=INTERNAL_SEED
    )
    probe_settings: dict[str, Any] = {}
    if iteration_cap is not None:
        if iteration_cap < 1:
            raise ValueError("iteration_cap must be positive")
        probe_settings["n_estimators"] = iteration_cap
    probe = build_lightgbm_classifier(**probe_settings)
    callbacks = _fit_callbacks()
    if progress_callback is not None:
        callbacks.append(progress_callback)
    probe_levels, probe_seconds, probe_details = _fit_with_inner_validation(
        probe,
        source_x,
        np.asarray(source_y, dtype=np.int8),
        source_fit,
        source_validation,
        spec,
        callbacks=callbacks,
    )
    probe_predictions = probe.predict_proba(
        prepare_lightgbm_frame(source_x.iloc[source_validation], spec, probe_levels)
    )[:, 1]
    validate_probabilities(probe_predictions, len(source_validation))
    selected_best = int(probe_details["best_iteration"])
    full_levels = fit_category_levels(source_x, spec)
    full_model = build_lightgbm_classifier(n_estimators=selected_best)
    full_resources_before = _resource_snapshot()
    full_started = time.perf_counter()
    full_model.fit(
        prepare_lightgbm_frame(source_x, spec, full_levels),
        np.asarray(source_y, dtype=np.int8),
        categorical_feature=list(spec.categorical_columns),
    )
    full_seconds = time.perf_counter() - full_started
    full_resources = _resource_delta(full_resources_before, _resource_snapshot())
    full_rounds = int(full_model.booster_.current_iteration())
    record = {
        "source_rows": len(source_x),
        "membership": {
            "source_only": True,
            "competition_labels_accessible": False,
            "full_fit_rows": len(source_x),
        },
        "split": {
            "seed": INTERNAL_SEED,
            "fit_rows": len(source_fit),
            "validation_rows": len(source_validation),
            "fit_indices": _array_record(("fit", source_fit)),
            "validation_indices": _array_record(("validation", source_validation)),
            "disjoint": not bool(np.intersect1d(source_fit, source_validation).size),
        },
        "probe": {
            "settings": probe.get_params(),
            "iteration_cap": iteration_cap,
            "fit_seconds": round(probe_seconds, 3),
            "selected_best_iteration": selected_best,
            "actual_rounds": probe_details["actual_iterations"],
            "source_validation_auc": float(
                roc_auc_score(source_y[source_validation], probe_predictions)
            ),
            "resources": probe_details["resources"],
            "early_stopping": {
                "eval_metric": "auc",
                "rounds": 100,
                "callbacks": ["early_stopping", "log_evaluation"],
            },
        },
        "full_fit": {
            "settings": full_model.get_params(),
            "fit_rows": len(source_x),
            "fit_seconds": round(full_seconds, 3),
            "actual_rounds": full_rounds,
            "resources": full_resources,
            "category_levels_from_all_source_rows": True,
            "eval_set": None,
            "callbacks": [],
        },
    }
    del probe, probe_predictions
    gc.collect()
    return SourceTeacherFit(full_model, full_levels, record)


def predict_source_teacher(
    teacher: SourceTeacherFit, features: pd.DataFrame, spec: InputSpec
) -> np.ndarray:
    """Predict the source teacher probability for an original feature frame."""
    if set(features.columns) != set(spec.feature_columns):
        raise ValueError(
            "teacher prediction features must match the original predictors"
        )
    predictions = teacher.model.predict_proba(
        prepare_lightgbm_frame(features, spec, teacher.category_levels)
    )[:, 1]
    validate_probabilities(predictions, len(features))
    return predictions


def append_teacher_feature(
    features: pd.DataFrame, probabilities: np.ndarray, spec: InputSpec
) -> tuple[pd.DataFrame, InputSpec]:
    """Copy original features and append one numeric teacher probability column."""
    if set(features.columns) != set(spec.feature_columns):
        raise ValueError("features must match the original predictors")
    values = np.asarray(probabilities, dtype=np.float64)
    validate_probabilities(values, len(features))
    result = features.loc[:, list(spec.feature_columns)].copy()
    result[SOURCE_TEACHER_FEATURE] = values
    extended = InputSpec(
        feature_columns=spec.feature_columns + (SOURCE_TEACHER_FEATURE,),
        numeric_columns=spec.numeric_columns + (SOURCE_TEACHER_FEATURE,),
        categorical_columns=spec.categorical_columns,
    )
    return result, extended


def teacher_feature_gate(
    baseline: dict[int, float], candidate: dict[int, float]
) -> dict[str, Any]:
    """Require the fixed candidate AUC gain on both overlapping holdouts."""
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
            "select_source_teacher_feature_blend"
            if qualifies
            else "retain_current_blend"
        ),
        "holdout_note": (
            "overlapping holdouts are stability evidence, not independent confirmation"
        ),
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    validate_probabilities(probabilities, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def _save_teacher_predictions(
    path: Path, train: pd.DataFrame, probabilities: np.ndarray
) -> None:
    if len(probabilities) != len(train):
        raise ValueError("teacher prediction rows do not match competition train")
    np.savez(
        path,
        row_index=np.arange(len(train), dtype=np.int64),
        id=train[ID_COLUMN].to_numpy(),
        teacher_probability=np.asarray(probabilities, dtype=np.float64),
    )


def _load_source_metadata(
    artifacts_dir: Path, input_hashes: dict[str, str]
) -> tuple[dict[str, Any], Path]:
    path = artifacts_dir / SOURCE_AUGMENTATION_METADATA
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("source augmentation metadata is not completed")
    if metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError("source augmentation inputs do not match current inputs")
    return metadata, path


def _load_prior_lightgbm(
    artifacts_dir: Path, input_hashes: dict[str, str]
) -> dict[str, Any]:
    path = artifacts_dir / "lightgbm" / "compare_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("prior corrected LightGBM metadata is not completed")
    expected = {name: input_hashes[name] for name in ("train.csv", "test.csv")}
    if metadata.get("input_sha256_before") != expected:
        raise RuntimeError("prior LightGBM inputs do not match current inputs")
    if metadata.get("predeclared_config") != dict(PREDECLARED_CONFIG):
        raise RuntimeError("prior LightGBM metadata is not the original 31-leaf run")
    return metadata


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Fit the source-only teacher and two competition-fold students."""
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
        "feature_change": {
            "only": SOURCE_TEACHER_FEATURE,
            "original_feature_count": 21,
            "student_feature_count": 22,
        },
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: str | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)

    def on_timeout(_signum: int, _frame: Any) -> None:
        raise SourceTeacherTimeout("source teacher diagnostic timed out")

    try:
        signal.signal(signal.SIGALRM, on_timeout)
        signal.signal(signal.SIGTERM, on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        source_before = input_before["source_airline/data.csv"]
        metadata.update(
            {
                "python": sys.version,
                "dependencies": _versions(),
                "threads": MAX_THREADS,
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
                "input_sha256_before": input_before,
                "source_csv_sha256_before": source_before,
                "source_sha256_before": _source_hashes(),
                "config_sha256": _config_hash(),
                "protected_artifact_sha256_before": protected_before,
                "model_seed": PREDECLARED_CONFIG["random_state"],
                "split_seeds": list(SPLIT_SEEDS),
                "internal_seed": INTERNAL_SEED,
                "student_settings": dict(PREDECLARED_CONFIG),
                "no_additional_download": True,
            }
        )
        save_json(metadata_path, metadata)

        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        source_raw = pd.read_csv(data_dir / "source_airline" / "data.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        source_metadata, source_metadata_path = _load_source_metadata(
            artifacts_dir, input_before
        )
        prior_lightgbm = _load_prior_lightgbm(artifacts_dir, input_before)
        source_x, source_y, cleaning = clean_source_rows(source_raw, train, test, spec)
        recorded_cleaning = source_metadata.get("source_rows")
        if cleaning != recorded_cleaning:
            raise RuntimeError(
                "source cleaning report does not match recorded metadata"
            )
        if cleaning.get("clean_rows") != EXPECTED_CLEAN_SOURCE_ROWS:
            raise RuntimeError(
                "source cleaning report does not contain the approved clean-row count"
            )
        metadata.update(
            {
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "source_cleaning": {
                    "actual": cleaning,
                    "recorded": recorded_cleaning,
                    "source_metadata_path": str(source_metadata_path),
                    "source_metadata_sha256": sha256_file(source_metadata_path),
                    "source_provenance": source_metadata.get("source_provenance"),
                    "source_uncertainty": source_metadata.get("source_uncertainty"),
                    "license": source_metadata.get("license"),
                    "no_additional_download": True,
                },
                "reused_artifacts": {
                    SOURCE_AUGMENTATION_METADATA: sha256_file(source_metadata_path),
                    "lightgbm/compare_metadata.json": sha256_file(
                        artifacts_dir / "lightgbm" / "compare_metadata.json"
                    ),
                },
            }
        )
        save_json(metadata_path, metadata)

        metadata["current_fit"] = {
            "name": "source_teacher_probe_and_full_fit",
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(metadata_path, metadata)
        teacher = fit_source_teacher(
            source_x,
            source_y,
            spec,
            progress_callback=_progress_callback(
                metadata, metadata_path, "source_teacher_probe"
            ),
        )
        metadata["teacher"] = teacher.record
        metadata["completed_fits"].append("source_teacher_probe_and_full_fit")
        metadata["current_fit"] = None
        teacher_train = predict_source_teacher(
            teacher, train.loc[:, list(spec.feature_columns)], spec
        )
        _save_teacher_predictions(
            output_dir / "source_teacher_train_predictions.npz", train, teacher_train
        )
        metadata["teacher_train_predictions"] = {
            "path": str(output_dir / "source_teacher_train_predictions.npz"),
            "rows": len(teacher_train),
            "row_index": _array_record(
                ("row_index", np.arange(len(train), dtype=np.int64))
            ),
            "sha256": sha256_file(output_dir / "source_teacher_train_predictions.npz"),
        }
        save_json(metadata_path, metadata)

        student_features, student_spec = append_teacher_feature(
            train.loc[:, list(spec.feature_columns)], teacher_train, spec
        )
        metadata["student"] = {
            "feature_columns": list(student_spec.feature_columns),
            "numeric_columns": list(student_spec.numeric_columns),
            "categorical_columns": list(student_spec.categorical_columns),
            "fit_policy": (
                "original LightGBM31 settings with one numeric teacher feature"
            ),
            "outer_holdout_used_for_training_or_early_stopping": False,
        }
        metadata["recorded_splits"] = {}
        baseline_aucs: dict[int, float] = {}
        candidate_aucs: dict[int, float] = {}
        for fit_number, seed in enumerate(SPLIT_SEEDS, 1):
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
            original_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            slow = _load_components(slow_path, train, holdout)
            original = _load_components(original_path, train, holdout)
            reference_ids = train.iloc[holdout][ID_COLUMN].to_numpy()
            baseline = weighted_blend_predictions(
                reference_ids,
                slow[ID_COLUMN],
                slow["probability"].to_numpy(),
                original[ID_COLUMN],
                original["probability"].to_numpy(),
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )
            baseline_metrics = _metrics(
                y[holdout], baseline["blend_probability"].to_numpy()
            )
            if abs(baseline_metrics["roc_auc"] - BASELINE_AUC[seed]) > 1e-9:
                raise RuntimeError(f"current baseline changed for split seed {seed}")
            baseline_aucs[seed] = baseline_metrics["roc_auc"]
            metadata["current_fit"] = {
                "fit_number": fit_number,
                "split_seed": seed,
                "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "outer_fit_rows": len(outer_fit),
                "inner_fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_holdout_rows": len(holdout),
                "outer_holdout_used_for_early_stopping": False,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)
            model = build_lightgbm_classifier()
            fit_started = time.perf_counter()
            category_levels, fit_seconds, details = _fit_with_inner_validation(
                model,
                student_features,
                y,
                inner_fit,
                inner_validation,
                student_spec,
                callbacks=_fit_callbacks()
                + [_progress_callback(metadata, metadata_path, f"student_seed_{seed}")],
            )
            student_predictions = model.predict_proba(
                prepare_lightgbm_frame(
                    student_features.iloc[holdout], student_spec, category_levels
                )
            )[:, 1]
            validate_probabilities(student_predictions, len(holdout))
            student_metrics = _metrics(y[holdout], student_predictions)
            candidate = weighted_blend_predictions(
                reference_ids,
                slow[ID_COLUMN],
                slow["probability"].to_numpy(),
                reference_ids,
                student_predictions,
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )
            candidate_metrics = _metrics(
                y[holdout], candidate["blend_probability"].to_numpy()
            )
            candidate_aucs[seed] = candidate_metrics["roc_auc"]
            validation = pd.DataFrame(
                {
                    "row_index": holdout,
                    ID_COLUMN: candidate[ID_COLUMN],
                    "teacher_probability": teacher_train[holdout],
                    "student_probability": candidate["lightgbm_probability"],
                    "baseline_probability": baseline["blend_probability"],
                    "candidate_probability": candidate["blend_probability"],
                }
            )
            validation_path = output_dir / f"validation_source_teacher_seed{seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            fit_record = {
                "split_seed": seed,
                "fit_seconds": round(fit_seconds, 3),
                "fit_elapsed_seconds": round(time.perf_counter() - fit_started, 3),
                "settings": model.get_params(),
                "actual_rounds": details["actual_iterations"],
                "best_iteration": details["best_iteration"],
                "resources": details["resources"],
                "standalone": student_metrics,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "validation_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "split": split_record,
            }
            metadata["validation_results"].append(fit_record)
            metadata["completed_fits"].append(f"student_seed_{seed}")
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del model, student_predictions, validation, baseline, candidate
            gc.collect()

        metadata["selection"] = teacher_feature_gate(baseline_aucs, candidate_aucs)
    except BaseException as error:
        caught = error
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
            signal.signal(signal.SIGTERM, previous_term)
        except BaseException as error:
            finalization_errors.append(error)
        metadata["current_fit"] = None
        metadata["status"] = (
            "timeout"
            if isinstance(caught, SourceTeacherTimeout)
            else "failed"
            if caught is not None
            else "completed"
        )
        if caught is not None:
            metadata["error"] = _error_record(caught)
        try:
            if input_before is not None:
                input_after = _input_hashes(data_dir)
                metadata["input_sha256_after"] = input_after
                if input_after != input_before:
                    finalization_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
                metadata["source_csv_sha256_after"] = input_after[
                    "source_airline/data.csv"
                ]
                if (
                    source_before is not None
                    and metadata["source_csv_sha256_after"] != source_before
                ):
                    finalization_errors.append(
                        RuntimeError("source CSV fingerprint changed")
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
        try:
            source_after = _source_hashes()
            metadata["source_sha256_after"] = source_after
            if metadata.get("source_sha256_before") != source_after:
                finalization_errors.append(RuntimeError("source fingerprint changed"))
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
        metadata["selection"] = (
            metadata.get("selection") if metadata["status"] == "completed" else None
        )
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        for _ in range(4):
            save_json(metadata_path, metadata)
            actual_bytes = sum(
                path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
            )
            if actual_bytes == metadata["output_bytes"]:
                break
            metadata["output_bytes"] = actual_bytes
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("output exceeded 100 MiB")
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["error"] = _error_record(budget_error)
            if caught is None:
                caught = budget_error
            save_json(metadata_path, metadata)
    if caught is not None:
        raise caught
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
