"""Run the fixed five-fold source-teacher student diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import math
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

from compare import make_catboost_internal_split
from eco_business_specialist import _verify_prior_oof
from five_fold_cv import (
    _metrics,
    _record_split,
    five_fold_gate,
)
from lightgbm_capacity import _artifact_snapshot, _input_hashes
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _fit_callbacks,
    _fit_with_inner_validation,
    build_lightgbm_classifier,
    prepare_lightgbm_frame,
)
from source_teacher_experiment import (
    SOURCE_TEACHER_FEATURE,
    _load_prior_lightgbm,
    _progress_callback,
    append_teacher_feature,
)
from source_teacher_experiment import (
    _config_hash as teacher_config_hash,
)
from source_teacher_experiment import (
    _source_hashes as teacher_source_hashes,
)
from source_teacher_experiment import (
    _versions as teacher_versions,
)
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

FOLD_COUNT = 5
FOLD_SEED = 2026
INTERNAL_SEED = 42
TEACHER_ROUNDS = 531
EXPECTED_TRAIN_ROWS = 699635
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_NAME = "source_teacher_cv"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 500
MINIMUM_AUC_GAIN = 0.0002


class SourceTeacherCVTimeout(TimeoutError):
    """Raised when the fixed five-fold diagnostic reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise SourceTeacherCVTimeout("source-teacher five-fold CV exceeded its deadline")


def _source_hashes() -> dict[str, str]:
    names = (
        "source_teacher_cv.py",
        "source_teacher_experiment.py",
        "five_fold_cv.py",
        "eco_business_specialist.py",
        "compare.py",
        "lightgbm_experiment.py",
        "lightgbm_capacity.py",
        "source_augmentation.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _feature_spec(spec: InputSpec, *, include_teacher: bool) -> dict[str, list[str]]:
    feature_columns = list(spec.feature_columns)
    numeric_columns = list(spec.numeric_columns)
    if include_teacher:
        feature_columns.append(SOURCE_TEACHER_FEATURE)
        numeric_columns.append(SOURCE_TEACHER_FEATURE)
    return {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": list(spec.categorical_columns),
    }


def _validate_feature_spec(
    recorded: dict[str, Any], spec: InputSpec, *, include_teacher: bool
) -> None:
    expected = _feature_spec(spec, include_teacher=include_teacher)
    actual = {name: recorded.get(name) for name in expected}
    if actual != expected:
        suffix = " with teacher feature" if include_teacher else ""
        raise RuntimeError(f"recorded feature definition changed{suffix}")


def _validate_teacher_cache_arrays(
    row_index: np.ndarray,
    ids: np.ndarray,
    probabilities: np.ndarray,
    expected_ids: np.ndarray,
) -> None:
    expected_rows = np.arange(len(expected_ids), dtype=np.int64)
    if not np.array_equal(row_index, expected_rows):
        raise RuntimeError("source-teacher cache row order changed")
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError("source-teacher cache ID order changed")
    validate_probabilities(probabilities, len(expected_ids))


def _validate_oof_alignment(
    row_index: np.ndarray,
    ids: np.ndarray,
    fold_ids: np.ndarray,
    expected_ids: np.ndarray,
    expected_fold_ids: np.ndarray,
) -> None:
    expected_rows = np.arange(len(expected_ids), dtype=np.int64)
    if not np.array_equal(row_index, expected_rows):
        raise RuntimeError("OOF row order changed")
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError("OOF ID order changed")
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise RuntimeError("OOF fold membership changed")


def _candidate_blend(
    catboost_probability: np.ndarray, student_probability: np.ndarray
) -> np.ndarray:
    catboost = np.asarray(catboost_probability, dtype=np.float64)
    student = np.asarray(student_probability, dtype=np.float64)
    if len(catboost) != len(student):
        raise ValueError("blend components must have equal lengths")
    validate_probabilities(catboost, len(catboost))
    validate_probabilities(student, len(student))
    candidate = 0.5 * catboost + 0.5 * student
    validate_probabilities(candidate, len(candidate))
    return candidate


def _verify_teacher_cache(
    artifacts_dir: Path,
    train: pd.DataFrame,
    spec: InputSpec,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    metadata_path = artifacts_dir / "source_teacher_feature" / "compare_metadata.json"
    cache_path = (
        artifacts_dir
        / "source_teacher_feature"
        / "source_teacher_train_predictions.npz"
    )
    if not metadata_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError("completed source-teacher cache is required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("source-teacher metadata is not completed")
    if metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError("source-teacher inputs do not match current inputs")
    current_teacher_sources = teacher_source_hashes()
    recorded_teacher_sources = metadata.get("source_sha256_before")
    if recorded_teacher_sources != current_teacher_sources:
        raise RuntimeError("source-teacher source fingerprints changed")
    _validate_feature_spec(metadata, spec, include_teacher=False)
    student_record = metadata.get("student")
    if not isinstance(student_record, dict):
        raise RuntimeError("source-teacher student feature definition is missing")
    _validate_feature_spec(student_record, spec, include_teacher=True)
    if metadata.get("config_sha256") != teacher_config_hash():
        raise RuntimeError("source-teacher configuration fingerprint changed")

    teacher = metadata.get("teacher", {})
    membership = teacher.get("membership", {})
    if teacher.get("source_rows") != 129859 or membership != {
        "source_only": True,
        "competition_labels_accessible": False,
        "full_fit_rows": 129859,
    }:
        raise RuntimeError("source-teacher provenance is not source-only")
    source_cleaning = metadata.get("source_cleaning", {})
    if source_cleaning.get("actual", {}).get("clean_rows") != 129859:
        raise RuntimeError("source-teacher clean source row count changed")
    if source_cleaning.get("actual") != source_cleaning.get("recorded"):
        raise RuntimeError("source-teacher cleaning report changed")
    if not source_cleaning.get("source_provenance") or not source_cleaning.get(
        "license"
    ):
        raise RuntimeError("source-teacher source provenance is incomplete")
    probe = teacher.get("probe", {})
    full_fit = teacher.get("full_fit", {})
    if probe.get("selected_best_iteration") != TEACHER_ROUNDS:
        raise RuntimeError("source-teacher selected round count changed")
    if full_fit.get("actual_rounds") != TEACHER_ROUNDS:
        raise RuntimeError("source-teacher full fit round count changed")
    if full_fit.get("fit_rows") != 129859 or full_fit.get("eval_set") is not None:
        raise RuntimeError("source-teacher full fit provenance changed")
    if full_fit.get("callbacks") != []:
        raise RuntimeError("source-teacher full fit callbacks changed")
    for name, value in PREDECLARED_CONFIG.items():
        if probe.get("settings", {}).get(name) != value:
            raise RuntimeError(f"source-teacher probe setting changed: {name}")
        expected = TEACHER_ROUNDS if name == "n_estimators" else value
        if full_fit.get("settings", {}).get(name) != expected:
            raise RuntimeError(f"source-teacher full setting changed: {name}")

    prediction_record = metadata.get("teacher_train_predictions", {})
    if prediction_record.get("rows") != len(train):
        raise RuntimeError("source-teacher cache row count changed")
    cache_sha256 = sha256_file(cache_path)
    if prediction_record.get("sha256") != cache_sha256:
        raise RuntimeError("source-teacher cache fingerprint changed")
    expected_ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
    with np.load(cache_path) as saved:
        if set(saved.files) != {"row_index", "id", "teacher_probability"}:
            raise RuntimeError("source-teacher cache fields changed")
        row_index = np.asarray(saved["row_index"], dtype=np.int64)
        ids = np.asarray(saved["id"], dtype=np.int64)
        probabilities = np.asarray(saved["teacher_probability"], dtype=np.float64)
    if len(expected_ids) != EXPECTED_TRAIN_ROWS or not pd.Index(expected_ids).is_unique:
        raise RuntimeError("training IDs do not match the approved cache contract")
    _validate_teacher_cache_arrays(row_index, ids, probabilities, expected_ids)
    _load_prior_lightgbm(artifacts_dir, input_hashes)
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "cache_path": cache_path,
        "cache_sha256": cache_sha256,
        "probabilities": probabilities,
    }


def _verify_prior_baseline(prior: dict[str, Any], y: np.ndarray) -> None:
    results = prior["metadata"].get("fold_results", [])
    if len(results) != FOLD_COUNT:
        raise RuntimeError("frozen baseline does not contain five fold results")
    for fold_id, (_fit, holdout) in enumerate(prior["folds"], start=1):
        result = results[fold_id - 1]
        if result.get("fold_id") != fold_id:
            raise RuntimeError(f"frozen baseline fold numbering changed: {fold_id}")
        actual = _metrics(y[holdout], prior["baseline"][holdout])
        recorded = result.get("baseline", {})
        for metric in ("roc_auc", "log_loss"):
            if not math.isclose(
                actual[metric],
                float(recorded.get(metric, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"frozen baseline {metric} changed before fold {fold_id}"
                )


def _verify_split_records(
    prior: dict[str, Any], y: np.ndarray
) -> tuple[list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    recorded = prior["metadata"].get("splits", [])
    if len(recorded) != FOLD_COUNT:
        raise RuntimeError("frozen baseline does not contain five split records")
    split_records: list[dict[str, Any]] = []
    inner_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id, (outer_fit, holdout) in enumerate(prior["folds"], start=1):
        inner_fit, inner_validation = make_catboost_internal_split(
            outer_fit, y, seed=INTERNAL_SEED
        )
        split = _record_split(fold_id, outer_fit, holdout, inner_fit, inner_validation)
        if split != recorded[fold_id - 1]:
            raise RuntimeError(f"frozen split record changed for fold {fold_id}")
        split_records.append(split)
        inner_splits.append((inner_fit, inner_validation))
    return split_records, inner_splits


def _run_student_folds(
    *,
    student_features: pd.DataFrame,
    student_spec: InputSpec,
    y: np.ndarray,
    ids: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    split_records: list[dict[str, Any]],
    catboost_probability: np.ndarray,
    lightgbm_probability: np.ndarray,
    metadata: dict[str, Any],
    metadata_path: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    student_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_results: list[dict[str, Any]] = []
    for fold_id, ((outer_fit, holdout), (inner_fit, inner_validation)) in enumerate(
        zip(folds, inner_splits, strict=True), start=1
    ):
        if (
            np.intersect1d(inner_fit, inner_validation).size
            or np.intersect1d(inner_fit, holdout).size
            or np.intersect1d(inner_validation, holdout).size
        ):
            raise RuntimeError(f"student inner split overlaps in fold {fold_id}")
        metadata["current_fit"] = {
            "fold_id": fold_id,
            "model": "lightgbm_source_teacher_student",
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "outer_fit_rows": len(outer_fit),
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
            student_features,
            y,
            inner_fit,
            inner_validation,
            student_spec,
            callbacks=_fit_callbacks()
            + [_progress_callback(metadata, metadata_path, f"student_fold_{fold_id}")],
        )
        probabilities = np.asarray(
            model.predict_proba(
                prepare_lightgbm_frame(
                    student_features.iloc[holdout], student_spec, levels
                )
            )[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(probabilities, len(holdout))
        baseline = (
            0.5 * catboost_probability[holdout] + 0.5 * lightgbm_probability[holdout]
        )
        candidate = _candidate_blend(catboost_probability[holdout], probabilities)
        baseline_metrics = _metrics(y[holdout], baseline)
        candidate_metrics = _metrics(y[holdout], candidate)
        component_path = output_dir / f"fold{fold_id}_student.npz"
        np.savez_compressed(
            component_path,
            row_index=np.asarray(holdout, dtype=np.int64),
            id=np.asarray(ids[holdout], dtype=np.int64),
            fold_id=np.full(len(holdout), fold_id, dtype=np.int8),
            student_probability=probabilities,
        )
        student_oof[holdout] = probabilities
        best_iteration_1based = int(details["best_iteration"])
        fit_record = {
            "fold_id": fold_id,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "paired_auc_gain": candidate_metrics["roc_auc"]
            - baseline_metrics["roc_auc"],
            "student": {
                "fit_seconds": round(float(fit_seconds), 3),
                "fit_elapsed_seconds": round(time.perf_counter() - fit_started, 3),
                "settings": model.get_params(),
                "best_iteration_0based": best_iteration_1based - 1,
                "best_iteration_1based": best_iteration_1based,
                "actual_iterations": int(details["actual_iterations"]),
                "resources": details["resources"],
                "category_levels_from_inner_fit": {
                    name: len(values) for name, values in levels.items()
                },
            },
            "split": split_records[fold_id - 1],
            "prediction_path": str(component_path),
            "prediction_sha256": sha256_file(component_path),
        }
        fold_results.append(fit_record)
        metadata["fold_results"] = fold_results
        metadata["completed_fits"].append(f"student_fold_{fold_id}")
        metadata["current_fit"] = None
        save_json(metadata_path, metadata)
        del model, probabilities, baseline, candidate
        gc.collect()
    validate_probabilities(student_oof, len(y))
    return fold_results, student_oof


def _save_final_oof(
    path: Path,
    ids: np.ndarray,
    fold_ids: np.ndarray,
    teacher_probability: np.ndarray,
    catboost_probability: np.ndarray,
    lightgbm_probability: np.ndarray,
    student_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline = _candidate_blend(catboost_probability, lightgbm_probability)
    candidate = _candidate_blend(catboost_probability, student_probability)
    for values in (teacher_probability, baseline, candidate):
        validate_probabilities(values, len(ids))
    _validate_oof_alignment(
        np.arange(len(ids), dtype=np.int64),
        ids,
        fold_ids,
        ids,
        fold_ids,
    )
    np.savez_compressed(
        path,
        row_index=np.arange(len(ids), dtype=np.int64),
        id=ids,
        fold_id=fold_ids,
        teacher_probability=teacher_probability,
        catboost_probability=catboost_probability,
        student_probability=student_probability,
        baseline_probability=baseline,
        candidate_probability=candidate,
    )
    with np.load(path) as saved:
        expected_fields = {
            "row_index",
            "id",
            "fold_id",
            "teacher_probability",
            "catboost_probability",
            "student_probability",
            "baseline_probability",
            "candidate_probability",
        }
        if set(saved.files) != expected_fields:
            raise RuntimeError("saved OOF fields changed")
        _validate_oof_alignment(
            np.asarray(saved["row_index"], dtype=np.int64),
            np.asarray(saved["id"], dtype=np.int64),
            np.asarray(saved["fold_id"], dtype=np.int8),
            ids,
            fold_ids,
        )
        for name in (
            "teacher_probability",
            "catboost_probability",
            "student_probability",
            "baseline_probability",
            "candidate_probability",
        ):
            validate_probabilities(np.asarray(saved[name], dtype=np.float64), len(ids))
    return {
        "baseline": baseline,
        "candidate": candidate,
        "student": student_probability,
    }


def _summary(
    y: np.ndarray, fold_results: list[dict[str, Any]], oof: dict[str, np.ndarray]
) -> dict[str, Any]:
    baseline = _metrics(y, oof["baseline"])
    candidate = _metrics(y, oof["candidate"])
    mean_fold_baseline = {
        metric: float(np.mean([result["baseline"][metric] for result in fold_results]))
        for metric in ("roc_auc", "log_loss")
    }
    mean_fold_candidate = {
        metric: float(np.mean([result["candidate"][metric] for result in fold_results]))
        for metric in ("roc_auc", "log_loss")
    }
    gains = [float(result["paired_auc_gain"]) for result in fold_results]
    return {
        "fold_count": FOLD_COUNT,
        "oof_rows": len(y),
        "oof_unique_rows": len(y),
        "mean_fold_baseline": mean_fold_baseline,
        "mean_fold_candidate": mean_fold_candidate,
        "mean_paired_auc_gain": float(np.mean(gains)),
        "positive_fold_count": int(sum(gain > 0 for gain in gains)),
        "pooled_oof": {
            "baseline": baseline,
            "candidate": candidate,
            "roc_auc_gain": candidate["roc_auc"] - baseline["roc_auc"],
            "log_loss_gain": candidate["log_loss"] - baseline["log_loss"],
        },
        "interpretation": (
            "These folds were used during development; this is diagnostic evidence, "
            "not untouched confirmation or a public-score forecast."
        ),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly five source-teacher-feature LightGBM student fits."""
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
        "completed_fits": [],
        "fold_results": [],
        "selection": None,
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "progress_interval": PROGRESS_INTERVAL,
        "fixed_hypothesis": {
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "inner_validation_seed": INTERNAL_SEED,
            "feature": SOURCE_TEACHER_FEATURE,
            "student_configuration": dict(PREDECLARED_CONFIG),
            "baseline_blend": {"catboost_seed42": 0.5, "original_lightgbm31": 0.5},
            "candidate_blend": {"catboost_seed42": 0.5, "student": 0.5},
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
        spec = validate_inputs(train, test)
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        prior = _verify_prior_oof(data_dir, artifacts_dir, train, y, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        _verify_prior_baseline(prior, y)
        split_records, inner_splits = _verify_split_records(prior, y)
        metadata.update(
            {
                "python": sys.version,
                "dependencies": teacher_versions(),
                "threads": MAX_THREADS,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "feature_spec": _feature_spec(spec, include_teacher=True),
                "numeric_nan_policy": "preserve numeric NaN; no imputation",
                "teacher_cache": {
                    "metadata_path": str(teacher_cache["metadata_path"]),
                    "metadata_sha256": sha256_file(teacher_cache["metadata_path"]),
                    "prediction_path": str(teacher_cache["cache_path"]),
                    "prediction_sha256": teacher_cache["cache_sha256"],
                    "rows": len(teacher_cache["probabilities"]),
                },
                "prior_oof": {
                    "metadata_path": str(prior["metadata_path"]),
                    "metadata_sha256": sha256_file(prior["metadata_path"]),
                    "oof_path": str(prior["oof_path"]),
                    "oof_sha256": sha256_file(prior["oof_path"]),
                },
                "splits": split_records,
                "fold_membership": {
                    "rows": len(prior["fold_ids"]),
                    "counts": {
                        str(fold_id): int(np.sum(prior["fold_ids"] == fold_id))
                        for fold_id in range(1, FOLD_COUNT + 1)
                    },
                },
            }
        )
        save_json(metadata_path, metadata)
        student_features, student_spec = append_teacher_feature(
            train.loc[:, list(spec.feature_columns)],
            teacher_cache["probabilities"],
            spec,
        )
        fold_results, student_oof = _run_student_folds(
            student_features=student_features,
            student_spec=student_spec,
            y=y,
            ids=ids,
            folds=prior["folds"],
            inner_splits=inner_splits,
            split_records=split_records,
            catboost_probability=prior["catboost_seed42"],
            lightgbm_probability=prior["lightgbm"],
            metadata=metadata,
            metadata_path=metadata_path,
            output_dir=output_dir,
        )
        oof = _save_final_oof(
            output_dir / "oof_predictions.npz",
            ids,
            prior["fold_ids"],
            teacher_cache["probabilities"],
            prior["catboost_seed42"],
            prior["lightgbm"],
            student_oof,
        )
        metadata["oof_predictions"] = {
            "path": str(output_dir / "oof_predictions.npz"),
            "sha256": sha256_file(output_dir / "oof_predictions.npz"),
            "rows": len(y),
        }
        summary = _summary(y, fold_results, oof)
        metadata["summary"] = summary
        metadata["selection"] = five_fold_gate(
            summary["mean_paired_auc_gain"],
            summary["positive_fold_count"],
            summary["pooled_oof"]["roc_auc_gain"],
        )
    except BaseException as error:
        caught = error
        metadata["status"] = (
            "timeout" if isinstance(error, SourceTeacherCVTimeout) else "failed"
        )
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
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
        if finalization_errors:
            integrity_error = RuntimeError("; ".join(map(str, finalization_errors)))
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["integrity_error"] = {
                "type": type(integrity_error).__name__,
                "message": str(integrity_error),
            }
            if caught is None:
                caught = integrity_error
                metadata["error"] = metadata["integrity_error"]
            else:
                metadata["finalization_error"] = metadata["integrity_error"]
        metadata["current_fit"] = None
        if metadata.get("status") not in {"failed", "timeout"}:
            metadata["status"] = "completed"
        if metadata["status"] != "completed":
            metadata["selection"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        try:
            for _ in range(4):
                save_json(metadata_path, metadata)
                output_bytes = sum(
                    path.stat().st_size
                    for path in output_dir.rglob("*")
                    if path.is_file()
                )
                if metadata.get("output_bytes") == output_bytes:
                    break
                metadata["output_bytes"] = output_bytes
            if metadata.get("output_bytes", 0) > OUTPUT_BUDGET_BYTES:
                budget_error = RuntimeError("output directory exceeded 100 MiB budget")
                metadata["status"] = "failed"
                metadata["selection"] = None
                metadata["error"] = {
                    "type": type(budget_error).__name__,
                    "message": str(budget_error),
                }
                if caught is None:
                    caught = budget_error
                save_json(metadata_path, metadata)
        except BaseException as error:
            if caught is None:
                caught = error
                metadata["status"] = "failed"
                metadata["selection"] = None
                metadata["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            else:
                metadata["finalization_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
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
