"""Run the approved five-fold distance-profile validation diagnostic."""

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

# Scientific imports intentionally follow the thread-limit environment setup.
# ruff: noqa: E402
import numpy as np
import pandas as pd

from distance_profile_experiment import (
    _profile_record,
    _save_profile_table,
    append_distance_profile_features,
    fit_distance_profile_table,
)
from eco_business_specialist import _verify_prior_oof
from five_fold_cv import _metrics, five_fold_gate
from lightgbm_capacity import (
    _array_fingerprint,
    _artifact_snapshot,
    _input_hashes,
    _package_versions,
)
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _fit_callbacks,
    _fit_with_inner_validation,
    build_lightgbm_classifier,
    prepare_lightgbm_frame,
)
from source_teacher_cv import (
    FOLD_COUNT,
    _candidate_blend,
    _save_final_oof,
    _summary,
    _verify_split_records,
    _verify_teacher_cache,
)
from source_teacher_experiment import append_teacher_feature
from source_teacher_export import validate_cv_oof
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

FOLD_SEED = 2026
INTERNAL_SEED = 42
EXPECTED_TRAIN_ROWS = 699635
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_NAME = "distance_profile_cv"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 100
EXPECTED_FEATURE_COLUMNS = 21
EXPECTED_NUMERIC_COLUMNS = 17
EXPECTED_CATEGORICAL_COLUMNS = 4
EXPECTED_PROFILE_FEATURES = 17
EXPECTED_PROFILE_STUDENT_FEATURES = 39
EXPECTED_PROFILE_STUDENT_NUMERICS = 35
EXPECTED_CURRENT_POOLED_AUC = 0.9591080056460375
EXPECTED_CURRENT_MEAN_AUC = 0.9591155782004288


class DistanceProfileCVTimeout(TimeoutError):
    """Raised when the fixed five-fold diagnostic reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise DistanceProfileCVTimeout(
        "distance-profile five-fold CV exceeded its deadline"
    )


def _source_hashes() -> dict[str, str]:
    names = (
        "distance_profile_cv.py",
        "distance_profile_experiment.py",
        "source_teacher_cv.py",
        "source_teacher_export.py",
        "source_teacher_experiment.py",
        "five_fold_cv.py",
        "eco_business_specialist.py",
        "lightgbm_experiment.py",
        "lightgbm_capacity.py",
        "compare.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


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


def _validate_profile_student_schema(spec: InputSpec) -> None:
    if len(spec.feature_columns) != EXPECTED_PROFILE_STUDENT_FEATURES:
        raise RuntimeError("distance-profile student feature count is not 39")
    if len(spec.numeric_columns) != EXPECTED_PROFILE_STUDENT_NUMERICS:
        raise RuntimeError("distance-profile student numeric count is not 35")
    if len(spec.categorical_columns) != EXPECTED_CATEGORICAL_COLUMNS:
        raise RuntimeError("distance-profile student categorical count changed")


def _current_baseline_probability(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return the current blend, which is the prior teacher-CV candidate."""
    required = {
        "row_index",
        "id",
        "fold_id",
        "teacher_probability",
        "catboost_probability",
        "student_probability",
        "baseline_probability",
        "candidate_probability",
    }
    if set(arrays) != required:
        raise RuntimeError("source-teacher CV OOF fields changed")
    candidate = np.asarray(arrays["candidate_probability"], dtype=np.float64)
    validate_probabilities(candidate, len(candidate))
    return candidate


def _validate_teacher_cv_oof(
    arrays: dict[str, np.ndarray],
    ids: np.ndarray,
    prior: dict[str, Any],
    teacher_probability: np.ndarray,
) -> None:
    """Validate the prior teacher-CV eight-field contract before fitting."""
    validate_cv_oof(arrays, ids, prior, teacher_probability)
    _current_baseline_probability(arrays)


def _resolve_recorded_path(
    recorded: str, project_dir: Path, artifacts_dir: Path
) -> Path:
    path = Path(recorded)
    if path.is_absolute() and path.is_file():
        return path
    candidates = (project_dir / path, artifacts_dir / path, path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"recorded artifact is missing: {recorded}")


def _load_teacher_cv_artifacts(
    artifacts_dir: Path,
    project_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    ids: np.ndarray,
    prior: dict[str, Any],
    teacher_probability: np.ndarray,
    input_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = artifacts_dir / "source_teacher_cv" / "compare_metadata.json"
    oof_path = artifacts_dir / "source_teacher_cv" / "oof_predictions.npz"
    if not metadata_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError("completed source-teacher CV artifacts are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("source-teacher CV metadata is not completed")
    if metadata.get("selection", {}).get("qualifies") is not True:
        raise RuntimeError("source-teacher CV gate is not qualified")
    if metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError("source-teacher CV inputs do not match current inputs")
    oof_record = metadata.get("oof_predictions", {})
    if oof_record.get("rows") != len(train):
        raise RuntimeError("source-teacher CV OOF row count changed")
    if oof_record.get("sha256") != sha256_file(oof_path):
        raise RuntimeError("source-teacher CV OOF fingerprint changed")

    with np.load(oof_path) as saved:
        arrays = {
            name: np.asarray(
                saved[name],
                dtype=np.float64
                if name not in {"row_index", "id", "fold_id"}
                else np.int64,
            )
            for name in saved.files
        }
    arrays["fold_id"] = np.asarray(arrays["fold_id"], dtype=np.int8)
    _validate_teacher_cv_oof(arrays, ids, prior, teacher_probability)

    split_records, _inner_splits = _verify_split_records(prior, y)
    if metadata.get("splits") != split_records:
        raise RuntimeError("source-teacher CV split records changed")
    fold_results = metadata.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != FOLD_COUNT:
        raise RuntimeError("source-teacher CV does not contain five fold results")
    if len(metadata.get("completed_fits", [])) != FOLD_COUNT:
        raise RuntimeError("source-teacher CV does not contain five completed fits")

    current_baseline = _current_baseline_probability(arrays)
    for fold_id, (_outer_fit, holdout) in enumerate(prior["folds"], start=1):
        result = fold_results[fold_id - 1]
        if result.get("fold_id") != fold_id:
            raise RuntimeError(f"source-teacher CV fold numbering changed: {fold_id}")
        if result.get("split") != split_records[fold_id - 1]:
            raise RuntimeError(f"source-teacher CV split changed in fold {fold_id}")
        actual = _metrics(y[holdout], current_baseline[holdout])
        recorded = result.get("candidate", {})
        for metric in ("roc_auc", "log_loss"):
            if not math.isclose(
                actual[metric],
                float(recorded.get(metric, math.nan)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "source-teacher CV current baseline "
                    f"{metric} changed in fold {fold_id}"
                )

        prediction_record = result.get("prediction_sha256")
        prediction_path = _resolve_recorded_path(
            str(result.get("prediction_path", "")), project_dir, artifacts_dir
        )
        if prediction_record != sha256_file(prediction_path):
            raise RuntimeError(
                f"source-teacher CV student fingerprint changed in fold {fold_id}"
            )
        with np.load(prediction_path) as saved:
            if set(saved.files) != {
                "row_index",
                "id",
                "fold_id",
                "student_probability",
            }:
                raise RuntimeError(
                    f"source-teacher CV student fields changed in fold {fold_id}"
                )
            row_index = np.asarray(saved["row_index"], dtype=np.int64)
            saved_ids = np.asarray(saved["id"], dtype=np.int64)
            saved_fold_ids = np.asarray(saved["fold_id"], dtype=np.int8)
            student = np.asarray(saved["student_probability"], dtype=np.float64)
        expected_rows = np.asarray(holdout, dtype=np.int64)
        if not np.array_equal(row_index, expected_rows):
            raise RuntimeError(
                f"source-teacher CV student row order changed in fold {fold_id}"
            )
        if not np.array_equal(saved_ids, ids[holdout]):
            raise RuntimeError(
                f"source-teacher CV student ID order changed in fold {fold_id}"
            )
        if not np.array_equal(
            saved_fold_ids, np.full(len(holdout), fold_id, dtype=np.int8)
        ):
            raise RuntimeError(
                f"source-teacher CV student fold order changed in fold {fold_id}"
            )
        if not np.array_equal(student, arrays["student_probability"][holdout]):
            raise RuntimeError(
                f"source-teacher CV student OOF changed in fold {fold_id}"
            )

    summary = metadata.get("summary", {})
    pooled = _metrics(y, current_baseline)
    mean_auc = float(
        np.mean(
            [
                _metrics(y[holdout], current_baseline[holdout])["roc_auc"]
                for _fit, holdout in prior["folds"]
            ]
        )
    )
    if not math.isclose(
        float(
            summary.get("pooled_oof", {}).get("candidate", {}).get("roc_auc", math.nan)
        ),
        pooled["roc_auc"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(summary.get("mean_fold_candidate", {}).get("roc_auc", math.nan)),
        mean_auc,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("source-teacher CV current baseline summary changed")
    if not math.isclose(
        pooled["roc_auc"], EXPECTED_CURRENT_POOLED_AUC, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("current baseline pooled AUC is not the approved value")
    if not math.isclose(
        mean_auc, EXPECTED_CURRENT_MEAN_AUC, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("current baseline mean AUC is not the approved value")
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "oof_path": oof_path,
        "oof_sha256": sha256_file(oof_path),
        "arrays": arrays,
        "split_records": split_records,
    }, arrays


def _profile_record_with_membership(
    table: Any, path: Path, inner_fit: np.ndarray, profile_spec: InputSpec
) -> dict[str, Any]:
    record = _profile_record(table, path)
    record.update(
        {
            "column_order": list(profile_spec.feature_columns),
            "table_fit_membership": {
                "rows": len(inner_fit),
                "sha256": _array_fingerprint(inner_fit),
            },
        }
    )
    return record


def _progress_callback(
    metadata: dict[str, Any], metadata_path: Path, fold_id: int
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
            f"{datetime.now(timezone.utc).isoformat()} fold={fold_id} tree={tree}",
            flush=True,
        )

    callback.order = 20  # type: ignore[attr-defined]
    return callback


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly five profile-feature LightGBM student fits."""
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
            "student_configuration": dict(PREDECLARED_CONFIG),
            "feature_change": "distance_group_count_and_means",
            "baseline_blend": {
                "catboost_seed42": 0.5,
                "source_teacher_cv_student": 0.5,
            },
            "candidate_blend": {"catboost_seed42": 0.5, "student": 0.5},
        },
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
        _validate_fixed_schema(spec)
        if len(train) != EXPECTED_TRAIN_ROWS:
            raise RuntimeError(
                "training row count does not match the approved contract"
            )
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        if not pd.Index(ids).is_unique:
            raise RuntimeError("training IDs are not unique")

        prior = _verify_prior_oof(data_dir, artifacts_dir, train, y, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        teacher_cv, teacher_cv_arrays = _load_teacher_cv_artifacts(
            artifacts_dir,
            Path(__file__).resolve().parent,
            train,
            y,
            ids,
            prior,
            teacher_cache["probabilities"],
            input_before,
        )
        split_records, inner_splits = _verify_split_records(prior, y)
        original_features = train.loc[:, list(spec.feature_columns)]
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
                "prior_oof": {
                    "metadata_path": str(prior["metadata_path"]),
                    "metadata_sha256": sha256_file(prior["metadata_path"]),
                    "oof_path": str(prior["oof_path"]),
                    "oof_sha256": sha256_file(prior["oof_path"]),
                },
                "teacher_cv_oof": {
                    "metadata_path": str(teacher_cv["metadata_path"]),
                    "metadata_sha256": sha256_file(teacher_cv["metadata_path"]),
                    "oof_path": str(teacher_cv["oof_path"]),
                    "oof_sha256": teacher_cv["oof_sha256"],
                },
                "splits": split_records,
                "fold_membership": {
                    "rows": len(prior["fold_ids"]),
                    "counts": {
                        str(fold_id): int(np.sum(prior["fold_ids"] == fold_id))
                        for fold_id in range(1, FOLD_COUNT + 1)
                    },
                },
                "current_baseline": {
                    "source": "source_teacher_cv.candidate_probability",
                    "pooled": _metrics(
                        y, _current_baseline_probability(teacher_cv_arrays)
                    ),
                    "mean_fold_auc": float(
                        np.mean(
                            [
                                _metrics(
                                    y[holdout],
                                    _current_baseline_probability(teacher_cv_arrays)[
                                        holdout
                                    ],
                                )["roc_auc"]
                                for _fit, holdout in prior["folds"]
                            ]
                        )
                    ),
                },
            }
        )
        save_json(metadata_path, metadata)

        current_baseline = _current_baseline_probability(teacher_cv_arrays)
        student_oof = np.full(len(y), np.nan, dtype=np.float64)
        for fold_id, ((outer_fit, holdout), (inner_fit, inner_validation)) in enumerate(
            zip(prior["folds"], inner_splits, strict=True), start=1
        ):
            if (
                np.intersect1d(inner_fit, inner_validation).size
                or np.intersect1d(inner_fit, holdout).size
                or np.intersect1d(inner_validation, holdout).size
            ):
                raise RuntimeError(f"student inner split overlaps in fold {fold_id}")
            table = fit_distance_profile_table(original_features.iloc[inner_fit], spec)
            table_path = output_dir / f"fold{fold_id}_distance_profile.npz"
            _save_profile_table(table_path, table)
            profile_features, profile_spec = append_distance_profile_features(
                teacher_features, teacher_spec, table
            )
            _validate_profile_student_schema(profile_spec)
            table_record = _profile_record_with_membership(
                table, table_path, inner_fit, profile_spec
            )
            metadata["feature_spec_profile_student"] = _feature_spec(profile_spec)
            baseline_values = current_baseline[holdout]
            baseline_metrics = _metrics(y[holdout], baseline_values)
            metadata["current_fit"] = {
                "fold_id": fold_id,
                "model": "lightgbm_distance_profile_student",
                "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "outer_fit_rows": len(outer_fit),
                "inner_fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_holdout_rows": len(holdout),
                "outer_holdout_used_for_training_or_early_stopping": False,
                "baseline": baseline_metrics,
                "table": table_record,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)

            model = build_lightgbm_classifier()
            fit_started = time.perf_counter()
            levels, fit_seconds, details = _fit_with_inner_validation(
                model,
                profile_features,
                y,
                inner_fit,
                inner_validation,
                profile_spec,
                callbacks=_fit_callbacks()
                + [_progress_callback(metadata, metadata_path, fold_id)],
            )
            predictions = np.asarray(
                model.predict_proba(
                    prepare_lightgbm_frame(
                        profile_features.iloc[holdout], profile_spec, levels
                    )
                )[:, 1],
                dtype=np.float64,
            )
            validate_probabilities(predictions, len(holdout))
            candidate = _candidate_blend(prior["catboost_seed42"][holdout], predictions)
            candidate_metrics = _metrics(y[holdout], candidate)
            student_path = output_dir / f"fold{fold_id}_student.npz"
            np.savez_compressed(
                student_path,
                row_index=np.asarray(holdout, dtype=np.int64),
                id=np.asarray(ids[holdout], dtype=np.int64),
                fold_id=np.full(len(holdout), fold_id, dtype=np.int8),
                student_probability=predictions,
            )
            student_oof[holdout] = predictions
            best_iteration_1based = int(details["best_iteration"])
            fold_result = {
                "fold_id": fold_id,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "paired_auc_gain": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "student": {
                    "metrics": _metrics(y[holdout], predictions),
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
                "prediction_path": str(student_path),
                "prediction_sha256": sha256_file(student_path),
                "profile_table": table_record,
            }
            metadata["fold_results"].append(fold_result)
            metadata["completed_fits"].append(f"student_fold_{fold_id}")
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del model, predictions, candidate, profile_features, table
            gc.collect()

        validate_probabilities(student_oof, len(y))
        oof = _save_final_oof(
            output_dir / "oof_predictions.npz",
            ids,
            prior["fold_ids"],
            teacher_cache["probabilities"],
            prior["catboost_seed42"],
            teacher_cv_arrays["student_probability"],
            student_oof,
        )
        if not np.array_equal(oof["baseline"], current_baseline):
            raise RuntimeError("final OOF current baseline changed")
        metadata["oof_predictions"] = {
            "path": str(output_dir / "oof_predictions.npz"),
            "sha256": sha256_file(output_dir / "oof_predictions.npz"),
            "rows": len(y),
        }
        summary = _summary(y, metadata["fold_results"], oof)
        metadata["summary"] = summary
        metadata["selection"] = five_fold_gate(
            summary["mean_paired_auc_gain"],
            summary["positive_fold_count"],
            summary["pooled_oof"]["roc_auc_gain"],
        )
    except BaseException as error:
        caught = error
        metadata["status"] = (
            "timeout" if isinstance(error, DistanceProfileCVTimeout) else "failed"
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
                if caught is None:
                    caught = budget_error
                    metadata["error"] = {
                        "type": type(budget_error).__name__,
                        "message": str(budget_error),
                    }
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
