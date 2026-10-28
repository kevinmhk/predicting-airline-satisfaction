"""Run the fixed two-holdout CatBoost source-teacher-feature diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import signal
import sys
import time
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
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from compare import (
    _resource_delta,
    _resource_snapshot,
    prepare_catboost_frame,
)
from lightgbm_capacity import (
    _artifact_snapshot,
    _input_hashes,
    _load_components,
    _load_split_arrays,
)
from slower_learning import build_slow_catboost
from source_teacher_cv import _verify_teacher_cache
from source_teacher_experiment import (
    SOURCE_TEACHER_FEATURE,
    _load_prior_lightgbm,
    append_teacher_feature,
)
from train import (
    ID_COLUMN,
    MAX_THREADS,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)
from weighted_blend_experiment import weighted_blend_predictions

SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
SPLIT_SEEDS = tuple(SPLIT_FILES)
INTERNAL_SEED = 42
OUTPUT_NAME = "catboost_teacher_feature"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
INTERNAL_TIMEOUT_SECONDS = 2580
EXTERNAL_TIMEOUT_SECONDS = 2700
PROGRESS_INTERVAL = 500
MINIMUM_AUC_GAIN = 0.0002
BASELINE_AUC = {42: 0.9580277365442882, 2024: 0.9584612650032445}
TEACHER_COMPONENT_COLUMNS = [
    "row_index",
    ID_COLUMN,
    "teacher_probability",
    "student_probability",
    "baseline_probability",
    "candidate_probability",
]


class CatBoostTeacherFeatureTimeout(TimeoutError):
    """Raised when the bounded diagnostic reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise CatBoostTeacherFeatureTimeout(
        "CatBoost teacher-feature diagnostic exceeded its deadline"
    )


def build_catboost_teacher_classifier() -> CatBoostClassifier:
    """Build the unchanged slower-learning CatBoost configuration."""
    return build_slow_catboost()


def teacher_feature_gate(
    baseline: dict[int, float], candidate: dict[int, float]
) -> dict[str, Any]:
    """Require the fixed candidate gain independently on both holdouts."""
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
            "select_catboost_teacher_feature" if qualifies else "retain_current_blend"
        ),
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    validate_probabilities(probabilities, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _source_hashes() -> dict[str, str]:
    names = (
        "catboost_teacher_feature.py",
        "compare.py",
        "lightgbm_capacity.py",
        "lightgbm_experiment.py",
        "slower_learning.py",
        "source_teacher_cv.py",
        "source_teacher_experiment.py",
        "train.py",
        "weighted_blend_experiment.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _config_hash() -> str:
    settings = build_catboost_teacher_classifier().get_params()
    return hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prediction_paths(artifacts_dir: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for seed in SPLIT_SEEDS
        for path in (
            artifacts_dir
            / "slower_learning"
            / f"validation_slow_catboost_seed{seed}.csv",
            artifacts_dir
            / "source_teacher_feature"
            / f"validation_source_teacher_seed{seed}.csv",
        )
    )


def _prediction_hashes(artifacts_dir: Path) -> dict[str, str]:
    paths = _prediction_paths(artifacts_dir)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required prediction CSVs are missing: {missing}")
    return {str(path.relative_to(artifacts_dir)): sha256_file(path) for path in paths}


def _load_teacher_student_predictions(
    path: Path, expected_rows: np.ndarray, expected_ids: np.ndarray
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != TEACHER_COMPONENT_COLUMNS:
        raise ValueError(
            f"saved teacher-feature predictions have unexpected schema: {path}"
        )
    if frame["row_index"].duplicated().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(
            f"saved teacher-feature predictions contain duplicate keys: {path}"
        )
    if set(frame["row_index"].to_numpy(dtype=np.int64)) != set(expected_rows):
        raise ValueError(f"saved teacher-feature rows do not match holdout: {path}")
    frame = frame.set_index("row_index").loc[expected_rows].reset_index()
    if not np.array_equal(frame[ID_COLUMN].to_numpy(), expected_ids):
        raise ValueError(f"saved teacher-feature IDs do not match holdout: {path}")
    for name in TEACHER_COMPONENT_COLUMNS[2:]:
        validate_probabilities(frame[name].to_numpy(), len(frame))
    return frame


def teacher_feature_blend_predictions(
    reference_ids: Any,
    original_catboost_ids: Any,
    original_catboost_values: np.ndarray,
    teacher_student: pd.DataFrame,
    candidate_catboost_ids: Any,
    candidate_catboost_values: np.ndarray,
) -> pd.DataFrame:
    """Use only the saved student probability for both fixed 50/50 blends."""
    expected = TEACHER_COMPONENT_COLUMNS
    if list(teacher_student.columns) != expected:
        raise ValueError("teacher-feature predictions have unexpected schema")
    baseline = weighted_blend_predictions(
        reference_ids,
        original_catboost_ids,
        original_catboost_values,
        teacher_student[ID_COLUMN],
        teacher_student["student_probability"].to_numpy(),
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    candidate = weighted_blend_predictions(
        reference_ids,
        candidate_catboost_ids,
        candidate_catboost_values,
        teacher_student[ID_COLUMN],
        teacher_student["student_probability"].to_numpy(),
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    if not np.array_equal(baseline[ID_COLUMN], candidate[ID_COLUMN]):
        raise ValueError("baseline and candidate blend ID orders differ")
    result = pd.DataFrame(
        {
            ID_COLUMN: candidate[ID_COLUMN],
            "lightgbm_teacher_probability": candidate["lightgbm_probability"],
            "baseline_probability": baseline["blend_probability"],
            "candidate_probability": candidate["blend_probability"],
        }
    )
    validate_probabilities(result["baseline_probability"].to_numpy(), len(result))
    validate_probabilities(result["candidate_probability"].to_numpy(), len(result))
    return result


def _progress_logger(metadata: dict[str, Any], path: Path, split_seed: int) -> Any:
    pattern = re.compile(r"^\s*(\d+):")

    def log(message: str) -> None:
        for line in str(message).splitlines():
            match = pattern.match(line)
            if match is None:
                continue
            tree = int(match.group(1))
            if tree == 0 or tree % PROGRESS_INTERVAL:
                continue
            current = metadata.get("current_fit")
            if isinstance(current, dict):
                event = {"utc": datetime.now(timezone.utc).isoformat(), "tree": tree}
                current.setdefault("progress_log", []).append(event)
                current["last_progress_tree"] = tree
                save_json(path, metadata)
            print(
                f"{datetime.now(timezone.utc).isoformat()} split_seed={split_seed} "
                f"tree={tree}",
                flush=True,
            )

    return log


def _feature_spec(spec: Any) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _output_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly two sequential CatBoost teacher-feature fits."""
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
        "validation_results": [],
        "selection": None,
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "progress_interval": PROGRESS_INTERVAL,
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    prediction_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    signals_installed = False
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)

    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        prediction_before = _prediction_hashes(artifacts_dir)
        source_before = _source_hashes()
        metadata.update(
            {
                "python": sys.version,
                "packages": {**package_versions(), "catboost": version("catboost")},
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
                "prediction_input_sha256_before": prediction_before,
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "config_sha256": _config_hash(),
                "model_parameters": build_catboost_teacher_classifier().get_params(),
                "split_seeds": list(SPLIT_SEEDS),
                "internal_seed": INTERNAL_SEED,
                "feature_change": {
                    "only": SOURCE_TEACHER_FEATURE,
                    "original_feature_count": 21,
                    "teacher_feature_count": 22,
                    "categorical_feature_count": 4,
                },
                "no_source_rows_in_fit": True,
                "no_additional_download": True,
            }
        )
        save_json(metadata_path, metadata)

        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        prior_lightgbm = _load_prior_lightgbm(artifacts_dir, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        student_features, student_spec = append_teacher_feature(
            train.loc[:, list(spec.feature_columns)],
            teacher_cache["probabilities"],
            spec,
        )
        X_cat = prepare_catboost_frame(student_features, student_spec)
        metadata.update(
            {
                "feature_spec": _feature_spec(student_spec),
                "teacher_cache": {
                    "metadata_path": str(teacher_cache["metadata_path"]),
                    "metadata_sha256": sha256_file(teacher_cache["metadata_path"]),
                    "prediction_path": str(teacher_cache["cache_path"]),
                    "prediction_sha256": teacher_cache["cache_sha256"],
                    "rows": len(teacher_cache["probabilities"]),
                    "holdout_values_from_full_cache": True,
                },
                "recorded_splits": {},
            }
        )
        save_json(metadata_path, metadata)

        baseline_aucs: dict[int, float] = {}
        candidate_aucs: dict[int, float] = {}
        for fit_number, seed in enumerate(SPLIT_SEEDS, 1):
            outer_fit, holdout, inner_fit, inner_validation, split_record = (
                _load_split_arrays(artifacts_dir, train, y, prior_lightgbm, seed)
            )
            split_name = "outer" if seed == 42 else "confirmation"
            metadata["recorded_splits"][split_name] = split_record
            reference_ids = ids[holdout]
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
                teacher_path, holdout, reference_ids
            )
            if not np.allclose(
                teacher_student["teacher_probability"].to_numpy(),
                teacher_cache["probabilities"][holdout],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "saved holdout teacher features differ from frozen cache"
                )
            baseline_blend = weighted_blend_predictions(
                reference_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                reference_ids,
                teacher_student["student_probability"].to_numpy(),
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )
            baseline_probabilities = baseline_blend["blend_probability"].to_numpy()
            baseline_metrics = _metrics(y[holdout], baseline_probabilities)
            if abs(baseline_metrics["roc_auc"] - BASELINE_AUC[seed]) > 1e-12:
                raise RuntimeError(f"recomputed baseline changed for split seed {seed}")
            if not np.allclose(
                baseline_probabilities,
                teacher_student["candidate_probability"].to_numpy(),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"source-teacher baseline candidate mismatch for split seed {seed}"
                )
            baseline_aucs[seed] = baseline_metrics["roc_auc"]
            metadata["current_fit"] = {
                "fit_number": fit_number,
                "split_seed": seed,
                "model": "catboost_teacher_feature",
                "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "outer_fit_rows": len(outer_fit),
                "inner_fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_holdout_rows": len(holdout),
                "outer_holdout_used_for_training_or_early_stopping": False,
                "baseline_current_blend_50_50": baseline_metrics,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)

            model = build_catboost_teacher_classifier()
            resources_before = _resource_snapshot()
            fit_started = time.perf_counter()
            model.fit(
                X_cat.iloc[inner_fit],
                y[inner_fit],
                eval_set=(X_cat.iloc[inner_validation], y[inner_validation]),
                cat_features=list(student_spec.categorical_columns),
                verbose=PROGRESS_INTERVAL,
                log_cout=_progress_logger(metadata, metadata_path, seed),
            )
            fit_seconds = time.perf_counter() - fit_started
            predictions = np.asarray(
                model.predict_proba(X_cat.iloc[holdout])[:, 1], dtype=np.float64
            )
            validate_probabilities(predictions, len(holdout))
            blended = teacher_feature_blend_predictions(
                reference_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                teacher_student,
                reference_ids,
                predictions,
            )
            standalone_metrics = _metrics(y[holdout], predictions)
            candidate_metrics = _metrics(
                y[holdout], blended["candidate_probability"].to_numpy()
            )
            candidate_aucs[seed] = candidate_metrics["roc_auc"]
            validation = pd.DataFrame(
                {
                    "row_index": holdout,
                    ID_COLUMN: blended[ID_COLUMN],
                    "catboost_teacher_probability": predictions,
                    "lightgbm_teacher_probability": blended[
                        "lightgbm_teacher_probability"
                    ],
                    "baseline_probability": blended["baseline_probability"],
                    "candidate_probability": blended["candidate_probability"],
                }
            )
            validation_path = output_dir / f"validation_catboost_teacher_seed{seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            fit_record = {
                "split_seed": seed,
                "standalone": standalone_metrics,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "fit_seconds": round(fit_seconds, 3),
                "fit_elapsed_seconds": round(time.perf_counter() - fit_started, 3),
                "settings": model.get_params(),
                "best_iteration_0based": int(model.get_best_iteration()),
                "best_iteration_1based": int(model.get_best_iteration()) + 1,
                "actual_tree_count": int(model.tree_count_),
                "resources": _resource_delta(resources_before, _resource_snapshot()),
                "internal_validation": {
                    "seed": INTERNAL_SEED,
                    "fit_rows": len(inner_fit),
                    "validation_rows": len(inner_validation),
                    "outer_holdout_used": False,
                },
                "split": split_record,
                "validation_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
            }
            metadata["validation_results"].append(fit_record)
            metadata["completed_fits"].append(f"catboost_teacher_seed_{seed}")
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del model, predictions, validation, blended, slow, teacher_student
            gc.collect()

        metadata["selection"] = teacher_feature_gate(baseline_aucs, candidate_aucs)
    except BaseException as error:
        caught = error
        metadata["status"] = (
            "timeout" if isinstance(error, CatBoostTeacherFeatureTimeout) else "failed"
        )
        metadata["error"] = _error_record(error)
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
            prediction_after = _prediction_hashes(artifacts_dir)
            metadata["prediction_input_sha256_after"] = prediction_after
            if prediction_before is not None and prediction_after != prediction_before:
                finalization_errors.append(
                    RuntimeError("prediction input CSV fingerprint changed")
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
        metadata["current_fit"] = None
        if finalization_errors:
            integrity_error = RuntimeError("; ".join(map(str, finalization_errors)))
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["integrity_error"] = _error_record(integrity_error)
            if caught is None:
                caught = integrity_error
                metadata["error"] = _error_record(integrity_error)
            else:
                metadata["finalization_error"] = _error_record(integrity_error)
        elif caught is None:
            metadata["status"] = "completed"
        else:
            metadata["status"] = (
                "timeout"
                if isinstance(caught, CatBoostTeacherFeatureTimeout)
                else "failed"
            )
        if metadata["status"] != "completed":
            metadata["selection"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = _output_bytes(output_dir)
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("output directory exceeded 100 MiB budget")
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["error"] = _error_record(budget_error)
            if caught is None:
                caught = budget_error
        try:
            for _ in range(4):
                save_json(metadata_path, metadata)
                output_bytes = _output_bytes(output_dir)
                if output_bytes == metadata["output_bytes"]:
                    break
                metadata["output_bytes"] = output_bytes
            if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
                raise RuntimeError("output directory exceeded 100 MiB budget")
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = error
                metadata["error"] = _error_record(error)
            else:
                metadata["finalization_error"] = _error_record(error)
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
