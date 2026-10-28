"""Run the qualified full-data distance-profile student export."""

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

# ruff: noqa: E402
import lightgbm as lgb
import numpy as np
import pandas as pd

from compare import prepare_catboost_frame
from distance_profile_cv import _load_teacher_cv_artifacts
from distance_profile_experiment import (
    DISTANCE_COLUMN,
    DistanceProfileTable,
    _profile_mean_columns,
    _profile_record,
    _save_profile_table,
    _validate_profile_table,
    append_distance_profile_features,
    fit_distance_profile_table,
)
from eco_business_specialist import _verify_prior_oof
from five_fold_cv import five_fold_gate
from lightgbm_capacity import _array_fingerprint, _artifact_snapshot, _input_hashes
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
)
from source_teacher_cv import _verify_split_records, _verify_teacher_cache
from source_teacher_experiment import append_teacher_feature
from source_teacher_export import (
    EXPECTED_BASELINE_SHA256,
    _code_hashes,
    _fit_and_save_lightgbm,
    _hash_json,
    _output_bytes,
    _resolve_recorded_path,
    _validate_frozen_catboost,
    _versions,
    blend_probabilities,
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

OUTPUT_NAME = "distance_profile_export"
STUDENT_MODEL_NAME = "student_model.txt"
PROFILE_TABLE_NAME = "distance_profile_table.npz"
FEATURE_SPEC_NAME = "feature_spec.json"
SUBMISSION_NAME = "submission_slowcat_teacher_distance_profile_50_50.csv"
METADATA_NAME = "export_metadata.json"
RELOAD_NAME = "reload_predictions.npy"
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_TEST_ROWS = 299844
EXPECTED_TEACHER_ITERATIONS = 531
EXPECTED_STUDENT_ITERATIONS = 542
EXPECTED_CATBOOST_ITERATIONS = 4740
EXPECTED_FEATURE_COLUMNS = 21
EXPECTED_NUMERIC_COLUMNS = 17
EXPECTED_CATEGORICAL_COLUMNS = 4
EXPECTED_PROFILE_FEATURES = 17
EXPECTED_PROFILE_STUDENT_FEATURES = 39
EXPECTED_PROFILE_STUDENT_NUMERICS = 35
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
BLEND_WEIGHTS = {"catboost": 0.5, "student": 0.5}


class DistanceProfileExportTimeout(TimeoutError):
    """Raised when the bounded export reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise DistanceProfileExportTimeout("distance-profile export exceeded its deadline")


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _spec_from_record(record: dict[str, Any]) -> InputSpec:
    return InputSpec(
        tuple(record["feature_columns"]),
        tuple(record["numeric_columns"]),
        tuple(record["categorical_columns"]),
    )


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


def derive_profile_student_iterations(metadata: dict[str, Any]) -> int:
    """Derive the approved full-fit count from five native CV records."""
    if metadata.get("status") != "completed":
        raise RuntimeError("distance-profile CV metadata is not completed")
    if metadata.get("selection", {}).get("qualifies") is not True:
        raise RuntimeError("distance-profile CV gate is not qualified")
    hypothesis = metadata.get("fixed_hypothesis", {})
    if hypothesis.get("fold_count") != 5:
        raise RuntimeError("distance-profile CV does not contain five folds")
    if hypothesis.get("student_configuration") != dict(PREDECLARED_CONFIG):
        raise RuntimeError("distance-profile CV configuration changed")
    fold_results = metadata.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != 5:
        raise RuntimeError("distance-profile CV must contain five fold results")
    fold_ids = [int(row.get("fold_id", -1)) for row in fold_results]
    if sorted(fold_ids) != [1, 2, 3, 4, 5]:
        raise RuntimeError("distance-profile CV folds are not five distinct folds")
    counts: list[int] = []
    for row in sorted(fold_results, key=lambda item: int(item["fold_id"])):
        student = row.get("student", {})
        best = int(student.get("best_iteration_1based", 0))
        actual = int(student.get("actual_iterations", 0))
        if best <= 0 or actual <= 0 or best != actual:
            raise RuntimeError(
                "distance-profile student iteration provenance is invalid"
            )
        counts.append(best)
    selected = int(median(counts))
    if selected != EXPECTED_STUDENT_ITERATIONS:
        raise RuntimeError("approved distance-profile iteration count changed")
    return selected


def validate_profile_cv_oof(
    arrays: dict[str, np.ndarray],
    ids: np.ndarray,
    fold_ids: np.ndarray,
    teacher_probability: np.ndarray,
    catboost_probability: np.ndarray,
    baseline_probability: np.ndarray,
) -> None:
    """Validate the eight-field profile-CV OOF contract and blend formulas."""
    expected = {
        "row_index",
        "id",
        "fold_id",
        "teacher_probability",
        "catboost_probability",
        "student_probability",
        "baseline_probability",
        "candidate_probability",
    }
    if set(arrays) != expected:
        raise RuntimeError("distance-profile CV OOF fields changed")
    if not np.array_equal(arrays["row_index"], np.arange(len(ids))):
        raise RuntimeError("distance-profile CV OOF row coverage changed")
    if not np.array_equal(arrays["id"], ids):
        raise RuntimeError("distance-profile CV OOF IDs changed")
    if not np.array_equal(arrays["fold_id"], fold_ids):
        raise RuntimeError("distance-profile CV OOF folds changed")
    if not np.array_equal(arrays["teacher_probability"], teacher_probability):
        raise RuntimeError("distance-profile CV teacher feature changed")
    if not np.array_equal(arrays["catboost_probability"], catboost_probability):
        raise RuntimeError("distance-profile CV CatBoost component changed")
    if not np.array_equal(arrays["baseline_probability"], baseline_probability):
        raise RuntimeError(
            "distance-profile CV baseline is not source-teacher candidate"
        )
    for name in (
        "teacher_probability",
        "catboost_probability",
        "student_probability",
        "baseline_probability",
        "candidate_probability",
    ):
        validate_probabilities(arrays[name], len(ids))
    expected_candidate = (
        0.5 * arrays["catboost_probability"] + 0.5 * arrays["student_probability"]
    )
    if not np.array_equal(arrays["candidate_probability"], expected_candidate):
        raise RuntimeError("distance-profile CV candidate formula changed")


def _same_value(actual: Any, expected: Any, path: str = "value") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise RuntimeError(f"{path} fields changed")
        for name in actual:
            _same_value(actual[name], expected[name], f"{path}.{name}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f"{path} length changed")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same_value(left, right, f"{path}[{index}]")
        return
    if isinstance(actual, int | float) and isinstance(expected, int | float):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{path} changed")
        return
    if actual != expected:
        raise RuntimeError(f"{path} changed")


def _load_profile_cv_artifacts(
    artifacts_dir: Path,
    project_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    ids: np.ndarray,
    prior: dict[str, Any],
    teacher_cache: dict[str, Any],
    teacher_cv_arrays: dict[str, np.ndarray],
    input_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = artifacts_dir / "distance_profile_cv" / "compare_metadata.json"
    oof_path = artifacts_dir / "distance_profile_cv" / "oof_predictions.npz"
    if not metadata_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError("completed distance-profile CV artifacts are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("distance-profile CV metadata is not completed")
    if metadata.get("selection", {}).get("qualifies") is not True:
        raise RuntimeError("distance-profile CV gate is not qualified")
    if metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError("distance-profile CV inputs do not match current inputs")
    if metadata.get("oof_predictions", {}).get("rows") != len(train):
        raise RuntimeError("distance-profile CV OOF row count changed")
    if metadata["oof_predictions"].get("sha256") != sha256_file(oof_path):
        raise RuntimeError("distance-profile CV OOF fingerprint changed")

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
    fold_ids = np.asarray(teacher_cv_arrays["fold_id"], dtype=np.int8)
    validate_profile_cv_oof(
        arrays,
        ids,
        fold_ids,
        np.asarray(teacher_cache["probabilities"], dtype=np.float64),
        np.asarray(prior["catboost_seed42"], dtype=np.float64),
        np.asarray(teacher_cv_arrays["candidate_probability"], dtype=np.float64),
    )

    split_records, inner_splits = _verify_split_records(prior, y)
    if metadata.get("splits") != split_records:
        raise RuntimeError("distance-profile CV split records changed")
    fold_results = metadata.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != 5:
        raise RuntimeError("distance-profile CV does not contain five fold results")
    for fold_id, ((_outer_fit, holdout), (inner_fit, _inner_validation)) in enumerate(
        zip(prior["folds"], inner_splits, strict=True), start=1
    ):
        result = fold_results[fold_id - 1]
        if (
            result.get("fold_id") != fold_id
            or result.get("split") != split_records[fold_id - 1]
        ):
            raise RuntimeError(f"distance-profile CV split changed in fold {fold_id}")
        table = result.get("profile_table", {})
        membership = table.get("table_fit_membership", {})
        if membership.get("rows") != len(inner_fit) or membership.get(
            "sha256"
        ) != _array_fingerprint(inner_fit):
            raise RuntimeError(
                f"distance-profile table membership changed in fold {fold_id}"
            )
        table_path = _resolve_recorded_path(
            str(table.get("path", "")), project_dir, artifacts_dir
        )
        if table.get("sha256") != sha256_file(table_path):
            raise RuntimeError(
                f"distance-profile table fingerprint changed in fold {fold_id}"
            )
        prediction_path = _resolve_recorded_path(
            str(result.get("prediction_path", "")), project_dir, artifacts_dir
        )
        if result.get("prediction_sha256") != sha256_file(prediction_path):
            raise RuntimeError(
                f"distance-profile prediction fingerprint changed in fold {fold_id}"
            )
        with np.load(prediction_path) as saved:
            expected_fields = {"row_index", "id", "fold_id", "student_probability"}
            if set(saved.files) != expected_fields:
                raise RuntimeError(
                    f"distance-profile prediction fields changed in fold {fold_id}"
                )
            if not np.array_equal(saved["row_index"], holdout):
                raise RuntimeError(
                    f"distance-profile prediction rows changed in fold {fold_id}"
                )
            if not np.array_equal(saved["id"], ids[holdout]):
                raise RuntimeError(
                    f"distance-profile prediction IDs changed in fold {fold_id}"
                )
            if not np.array_equal(
                saved["fold_id"], np.full(len(holdout), fold_id, dtype=np.int8)
            ):
                raise RuntimeError(
                    f"distance-profile prediction folds changed in fold {fold_id}"
                )
            if not np.array_equal(
                saved["student_probability"], arrays["student_probability"][holdout]
            ):
                raise RuntimeError(
                    f"distance-profile student OOF changed in fold {fold_id}"
                )

    oof = {
        "baseline": arrays["baseline_probability"],
        "candidate": arrays["candidate_probability"],
        "student": arrays["student_probability"],
    }
    from source_teacher_cv import _summary

    actual_summary = _summary(y, fold_results, oof)
    _same_value(
        actual_summary, metadata.get("summary", {}), "distance-profile CV summary"
    )
    expected_selection = five_fold_gate(
        actual_summary["mean_paired_auc_gain"],
        actual_summary["positive_fold_count"],
        actual_summary["pooled_oof"]["roc_auc_gain"],
    )
    _same_value(
        expected_selection, metadata["selection"], "distance-profile CV selection"
    )
    return {
        "metadata": metadata,
        "metadata_path": metadata_path,
        "oof_path": oof_path,
    }, arrays


def _save_export_profile_table(path: Path, table: DistanceProfileTable) -> None:
    """Persist the normal table plus the fallback means needed for reload."""
    _save_profile_table(path, table)
    with np.load(path) as saved:
        distance = np.asarray(saved["distance"], dtype=np.float64)
        group_count = np.asarray(saved["group_count"], dtype=np.float64)
        group_means = np.asarray(saved["group_means"], dtype=np.float64)
    np.savez_compressed(
        path,
        distance=distance,
        group_count=group_count,
        group_means=group_means,
        global_means=table.global_means.to_numpy(dtype=np.float64),
    )


def _load_export_profile_table(
    path: Path, original_spec: InputSpec
) -> DistanceProfileTable:
    mean_columns = _profile_mean_columns(original_spec)
    with np.load(path) as saved:
        if set(saved.files) != {
            "distance",
            "group_count",
            "group_means",
            "global_means",
        }:
            raise RuntimeError("distance-profile table fields changed")
        distance = np.asarray(saved["distance"], dtype=np.float64)
        group_count = np.asarray(saved["group_count"], dtype=np.float64)
        group_means = np.asarray(saved["group_means"], dtype=np.float64)
        global_means = np.asarray(saved["global_means"], dtype=np.float64)
    if (
        distance.ndim != 1
        or group_count.shape != distance.shape
        or group_means.shape != (len(distance), len(mean_columns))
        or global_means.shape != (len(mean_columns),)
    ):
        raise RuntimeError("distance-profile table shapes changed")
    index = pd.Index(distance, name=DISTANCE_COLUMN)
    table = DistanceProfileTable(
        distance_column=DISTANCE_COLUMN,
        mean_columns=mean_columns,
        group_counts=pd.Series(group_count, index=index),
        group_means=pd.DataFrame(group_means, index=index, columns=mean_columns),
        global_means=pd.Series(global_means, index=mean_columns),
    )
    _validate_profile_table(table)
    return table


def _profile_feature_spec(
    original: InputSpec,
    teacher: InputSpec,
    profile_student: InputSpec,
    source_levels: dict[str, list[str]],
    competition_levels: dict[str, list[str]],
    table: DistanceProfileTable,
) -> dict[str, Any]:
    return {
        "original": _feature_spec(original),
        "teacher": _feature_spec(teacher),
        "profile_student": _feature_spec(profile_student),
        "category_levels": {
            "source_full": source_levels,
            "competition_full": competition_levels,
        },
        "distance_profile": {
            "distance_column": table.distance_column,
            "mean_columns": list(table.mean_columns),
            "feature_columns": [
                "distance_group_count",
                *[f"distance_group_mean__{name}" for name in table.mean_columns],
            ],
        },
        "numeric_missing_policy": "preserve numeric NaN; no imputation",
        "categorical_missing_policy": "native categorical levels for LightGBM",
        "id_column_excluded": ID_COLUMN,
        "target_column_excluded": TARGET_COLUMN,
        "weights": dict(BLEND_WEIGHTS),
    }


def _load_frozen_teacher(
    project_dir: Path, artifacts_dir: Path, spec: InputSpec
) -> tuple[
    lgb.Booster, Path, dict[str, list[str]], dict[str, list[str]], dict[str, Any]
]:
    export_dir = artifacts_dir / "source_teacher_export"
    metadata_path = export_dir / "export_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("completed source-teacher export is required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("source-teacher export metadata is not completed")
    recorded_spec = metadata.get("feature_spec", {})
    if recorded_spec.get("original") != _feature_spec(spec):
        raise RuntimeError("source-teacher original feature definition changed")
    teacher_record = metadata.get("teacher_model", {})
    if teacher_record.get("actual_iterations") != EXPECTED_TEACHER_ITERATIONS:
        raise RuntimeError("frozen source teacher round count changed")
    teacher_path = _resolve_recorded_path(
        str(teacher_record.get("model_path", "")), project_dir, artifacts_dir
    )
    expected_hash = metadata.get("files", {}).get("teacher_model.txt")
    if (
        teacher_record.get("model_sha256") != expected_hash
        or sha256_file(teacher_path) != expected_hash
    ):
        raise RuntimeError("frozen source teacher model fingerprint changed")
    levels = recorded_spec.get("category_levels", {})
    source_levels = levels.get("source_full")
    competition_levels = levels.get("competition_full")
    if not isinstance(source_levels, dict) or not isinstance(competition_levels, dict):
        raise RuntimeError("source-teacher category levels are missing")
    levels_path = export_dir / "teacher_category_levels.json"
    if metadata.get("files", {}).get("teacher_category_levels.json") != sha256_file(
        levels_path
    ):
        raise RuntimeError("frozen source teacher category levels changed")
    model = lgb.Booster(model_file=str(teacher_path))
    if int(model.current_iteration()) != EXPECTED_TEACHER_ITERATIONS:
        raise RuntimeError("frozen source teacher native round count changed")
    # Native LightGBM metadata normalizes spaces; inference keeps original names.
    expected_native_names = [name.replace(" ", "_") for name in spec.feature_columns]
    if list(model.feature_name()) != expected_native_names:
        raise RuntimeError("frozen source teacher feature order changed")
    return model, teacher_path, source_levels, competition_levels, metadata


def _fresh_reload_predictions(
    *,
    model_path: Path,
    teacher_path: Path,
    catboost_path: Path,
    feature_spec_path: Path,
    profile_table_path: Path,
    test_path: Path,
    output_path: Path,
) -> None:
    """Reconstruct every test component in a clean Python process."""
    code = r"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from compare import prepare_catboost_frame
from distance_profile_experiment import append_distance_profile_features
from distance_profile_export import _load_export_profile_table
from lightgbm_experiment import prepare_lightgbm_frame
from source_teacher_experiment import append_teacher_feature
from train import InputSpec

(
    student_path,
    teacher_path,
    catboost_path,
    feature_path,
    table_path,
    test_path,
    output_path,
) = sys.argv[1:]
record = json.loads(open(feature_path, encoding="utf-8").read())
original = InputSpec(
    tuple(record["original"]["feature_columns"]),
    tuple(record["original"]["numeric_columns"]),
    tuple(record["original"]["categorical_columns"]),
)
teacher_spec = InputSpec(
    tuple(record["teacher"]["feature_columns"]),
    tuple(record["teacher"]["numeric_columns"]),
    tuple(record["teacher"]["categorical_columns"]),
)
student_spec = InputSpec(
    tuple(record["profile_student"]["feature_columns"]),
    tuple(record["profile_student"]["numeric_columns"]),
    tuple(record["profile_student"]["categorical_columns"]),
)
levels = record["category_levels"]
test = pd.read_csv(test_path)
original_frame = test.loc[:, list(original.feature_columns)].copy()
teacher = lgb.Booster(model_file=teacher_path)
teacher_values = np.asarray(
    teacher.predict(
        prepare_lightgbm_frame(original_frame, original, levels["source_full"])
    ),
    dtype=np.float64,
)
table = _load_export_profile_table(table_path, original)
teacher_frame, _ = append_teacher_feature(original_frame, teacher_values, original)
student_frame, _ = append_distance_profile_features(teacher_frame, teacher_spec, table)
student = lgb.Booster(model_file=student_path)
student_values = np.asarray(
    student.predict(
        prepare_lightgbm_frame(
            student_frame, student_spec, levels["competition_full"]
        )
    ),
    dtype=np.float64,
)
catboost = CatBoostClassifier()
catboost.load_model(catboost_path)
cat_values = np.asarray(
    catboost.predict_proba(prepare_catboost_frame(test, original))[:, 1],
    dtype=np.float64,
)
candidate = 0.5 * cat_values + 0.5 * student_values
np.save(
    output_path,
    np.column_stack((teacher_values, student_values, cat_values, candidate)),
)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(model_path),
            str(teacher_path),
            str(catboost_path),
            str(feature_spec_path),
            str(profile_table_path),
            str(test_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=INTERNAL_TIMEOUT_SECONDS,
    )


def _write_submission(path: Path, ids: np.ndarray, probabilities: np.ndarray) -> None:
    validate_probabilities(probabilities, len(ids))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ID_COLUMN, TARGET_COLUMN])
        writer.writerows(
            (int(identifier), format(float(value), ".17g"))
            for identifier, value in zip(ids, probabilities, strict=True)
        )


def export_submission(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Fit only the qualified 542-round profile student and write the export."""
    project_dir = Path(__file__).resolve().parent
    data_dir = data_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / METADATA_NAME
    submission_path = output_dir / SUBMISSION_NAME
    reload_path = output_dir / RELOAD_NAME
    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "authorized_qualified_export": True,
        "qualification_source": "artifacts/distance_profile_cv/compare_metadata.json",
        "current_fit": None,
        "completed_fits": [],
        "selection": None,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "submission_path": str(submission_path),
    }
    save_json(metadata_path, metadata)

    started = time.perf_counter()
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
        source_before = _code_hashes(project_dir)
        metadata.update(
            {
                "protected_artifact_sha256_before": protected_before,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
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
            }
        )
        save_json(metadata_path, metadata)

        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        _validate_fixed_schema(spec)
        if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
            raise RuntimeError("competition row counts changed")
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        test_ids = test[ID_COLUMN].to_numpy(dtype=np.int64)
        prior = _verify_prior_oof(data_dir, artifacts_dir, train, y, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        teacher_cv, teacher_cv_arrays = _load_teacher_cv_artifacts(
            artifacts_dir,
            project_dir,
            train,
            y,
            ids,
            prior,
            teacher_cache["probabilities"],
            input_before,
        )
        profile_cv, _profile_cv_arrays = _load_profile_cv_artifacts(
            artifacts_dir,
            project_dir,
            train,
            y,
            ids,
            prior,
            teacher_cache,
            teacher_cv_arrays,
            input_before,
        )
        student_iterations = derive_profile_student_iterations(profile_cv["metadata"])
        (
            teacher_model,
            teacher_path,
            source_levels,
            competition_levels,
            _teacher_export,
        ) = _load_frozen_teacher(project_dir, artifacts_dir, spec)
        catboost_model, catboost_path, baseline_metadata = _validate_frozen_catboost(
            project_dir, artifacts_dir, spec
        )
        baseline_path = _resolve_recorded_path(
            baseline_metadata["final"]["submission_path"], project_dir, artifacts_dir
        )
        baseline_hash = sha256_file(baseline_path)
        if (
            baseline_hash != EXPECTED_BASELINE_SHA256
            or baseline_hash != baseline_metadata["final"]["submission_sha256"]
        ):
            raise RuntimeError(
                "frozen baseline submission fingerprint is not authorized"
            )
        validate_submission_file(baseline_path, test[ID_COLUMN])

        original_train = train.loc[:, list(spec.feature_columns)]
        original_test = test.loc[:, list(spec.feature_columns)]
        teacher_train = np.asarray(
            teacher_model.predict(
                prepare_lightgbm_frame(original_train, spec, source_levels)
            ),
            dtype=np.float64,
        )
        teacher_test = np.asarray(
            teacher_model.predict(
                prepare_lightgbm_frame(original_test, spec, source_levels)
            ),
            dtype=np.float64,
        )
        validate_probabilities(teacher_train, len(train))
        validate_probabilities(teacher_test, len(test))
        teacher_difference = float(
            np.max(np.abs(teacher_train - teacher_cache["probabilities"]))
        )
        if teacher_difference > 1e-12:
            raise RuntimeError(
                f"refrozen teacher does not reproduce cache: {teacher_difference}"
            )

        table = fit_distance_profile_table(original_train, spec)
        table_path = output_dir / PROFILE_TABLE_NAME
        _save_export_profile_table(table_path, table)
        teacher_features, teacher_spec = append_teacher_feature(
            original_train, teacher_train, spec
        )
        student_features, student_spec = append_distance_profile_features(
            teacher_features, teacher_spec, table
        )
        _validate_profile_student_schema(student_spec)
        student_levels = fit_category_levels(student_features, student_spec)
        if student_levels != competition_levels:
            raise RuntimeError("competition category levels changed")
        feature_spec = _profile_feature_spec(
            spec, teacher_spec, student_spec, source_levels, competition_levels, table
        )
        feature_path = output_dir / FEATURE_SPEC_NAME
        save_json(feature_path, feature_spec)
        metadata.update(
            {
                "student_iterations": student_iterations,
                "iteration_lineage": {
                    "student": (
                        "distance_profile_cv five student actual_iterations median"
                    ),
                    "student_cv_counts": [
                        int(row["student"]["actual_iterations"])
                        for row in sorted(
                            profile_cv["metadata"]["fold_results"],
                            key=lambda item: int(item["fold_id"]),
                        )
                    ],
                },
                "qualification": {
                    "distance_profile_cv_selection": profile_cv["metadata"][
                        "selection"
                    ],
                    "distance_profile_cv_summary": profile_cv["metadata"]["summary"],
                    "distance_profile_cv_metadata_sha256": sha256_file(
                        profile_cv["metadata_path"]
                    ),
                    "distance_profile_cv_oof_sha256": sha256_file(
                        profile_cv["oof_path"]
                    ),
                    "distance_profile_cv_oof_rows": len(train),
                    "source_teacher_cv_metadata_sha256": sha256_file(
                        teacher_cv["metadata_path"]
                    ),
                    "source_teacher_cv_oof_sha256": sha256_file(teacher_cv["oof_path"]),
                },
                "feature_spec": feature_spec,
                "profile_table": _profile_record(table, table_path),
                "configuration": {
                    "student": build_export_config(student_iterations),
                    "fit_policy": (
                        "full-data fit with no eval_set, no early stopping, "
                        "no callbacks"
                    ),
                },
                "configuration_sha256": _hash_json(
                    {"student": build_export_config(student_iterations)}
                ),
                "teacher_train_prediction_check": {
                    "rows": len(train),
                    "max_abs_difference": teacher_difference,
                    "tolerance": 1e-12,
                },
                "reused_artifacts": {
                    "teacher_export_metadata_path": str(
                        artifacts_dir / "source_teacher_export" / METADATA_NAME
                    ),
                    "teacher_export_metadata_sha256": sha256_file(
                        artifacts_dir / "source_teacher_export" / METADATA_NAME
                    ),
                    "teacher_model_path": str(teacher_path),
                    "teacher_model_sha256": sha256_file(teacher_path),
                    "teacher_category_levels_path": str(
                        artifacts_dir
                        / "source_teacher_export"
                        / "teacher_category_levels.json"
                    ),
                    "catboost_model_path": str(catboost_path),
                    "catboost_model_sha256": sha256_file(catboost_path),
                    "catboost_tree_count": EXPECTED_CATBOOST_ITERATIONS,
                    "baseline_submission_path": str(baseline_path),
                    "baseline_submission_sha256": baseline_hash,
                    "teacher_cache_sha256": teacher_cache["cache_sha256"],
                },
                "files": {
                    FEATURE_SPEC_NAME: sha256_file(feature_path),
                    PROFILE_TABLE_NAME: sha256_file(table_path),
                },
            }
        )
        save_json(metadata_path, metadata)

        student_model = build_lightgbm_classifier(n_estimators=student_iterations)
        student_record = _fit_and_save_lightgbm(
            name="student",
            model=student_model,
            features=prepare_lightgbm_frame(
                student_features, student_spec, student_levels
            ),
            labels=y,
            spec=student_spec,
            model_path=output_dir / STUDENT_MODEL_NAME,
            metadata=metadata,
            metadata_path=metadata_path,
        )
        metadata["files"][STUDENT_MODEL_NAME] = sha256_file(
            output_dir / STUDENT_MODEL_NAME
        )
        student_test_features, _ = append_teacher_feature(
            original_test, teacher_test, spec
        )
        student_test_features, _ = append_distance_profile_features(
            student_test_features, teacher_spec, table
        )
        student_test = np.asarray(
            student_model.predict_proba(
                prepare_lightgbm_frame(
                    student_test_features, student_spec, student_levels
                )
            )[:, 1],
            dtype=np.float64,
        )
        catboost_test = np.asarray(
            catboost_model.predict_proba(prepare_catboost_frame(test, spec))[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(student_test, len(test))
        validate_probabilities(catboost_test, len(test))
        candidate = blend_probabilities(catboost_test, student_test)
        _write_submission(submission_path, test_ids, candidate)
        submitted = validate_submission_file(submission_path, test[ID_COLUMN])
        schema: dict[str, Any] = {
            "columns": list(submitted.columns),
            "rows": len(submitted),
            "unique_ids": bool(submitted[ID_COLUMN].is_unique),
            "probability_min": float(submitted[TARGET_COLUMN].min()),
            "probability_max": float(submitted[TARGET_COLUMN].max()),
            "max_abs_difference": float(
                np.max(np.abs(submitted[TARGET_COLUMN].to_numpy() - candidate))
            ),
        }
        if schema["max_abs_difference"] > 1e-12:
            raise RuntimeError("exported submission predictions changed")
        _fresh_reload_predictions(
            model_path=output_dir / STUDENT_MODEL_NAME,
            teacher_path=teacher_path,
            catboost_path=catboost_path,
            feature_spec_path=feature_path,
            profile_table_path=table_path,
            test_path=data_dir / "test.csv",
            output_path=reload_path,
        )
        reloaded = np.asarray(np.load(reload_path), dtype=np.float64)
        reload_path.unlink()
        reload_differences = {
            "teacher_max_abs": float(np.max(np.abs(teacher_test - reloaded[:, 0]))),
            "student_max_abs": float(np.max(np.abs(student_test - reloaded[:, 1]))),
            "catboost_max_abs": float(np.max(np.abs(catboost_test - reloaded[:, 2]))),
            "blend_max_abs": float(np.max(np.abs(candidate - reloaded[:, 3]))),
        }
        if max(reload_differences.values()) > 1e-12:
            raise RuntimeError(
                f"fresh-interpreter reload verification failed: {reload_differences}"
            )
        baseline_after = sha256_file(baseline_path)
        if baseline_after != baseline_hash:
            raise RuntimeError("frozen baseline submission changed during export")
        metadata.update(
            {
                "status": "completed",
                "selection": profile_cv["metadata"]["selection"],
                "current_fit": None,
                "student_model": student_record,
                "blend_components": {
                    "catboost_model": str(catboost_path),
                    "student_model": str(output_dir / STUDENT_MODEL_NAME),
                    "weights": dict(BLEND_WEIGHTS),
                    "teacher_model": str(teacher_path),
                },
                "submission": {
                    "path": str(submission_path),
                    "sha256": sha256_file(submission_path),
                    **schema,
                },
                "reload_check": {
                    "fresh_interpreter": True,
                    "rows": len(test),
                    "max_abs_difference": reload_differences,
                    "temporary_path_removed": not reload_path.exists(),
                },
            }
        )
        del teacher_model, student_model, catboost_model
        gc.collect()
    except BaseException as error:
        caught = error
        metadata["status"] = (
            "timeout" if isinstance(error, DistanceProfileExportTimeout) else "failed"
        )
        metadata["selection"] = None
        metadata["current_fit"] = None
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        if submission_path.exists():
            submission_path.unlink()
        if reload_path.exists():
            reload_path.unlink()
    finally:
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                finalization_errors.append(error)
        try:
            metadata["input_sha256_after"] = _input_hashes(data_dir)
            if (
                input_before is not None
                and metadata["input_sha256_after"] != input_before
            ):
                finalization_errors.append(
                    RuntimeError("input CSV fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        try:
            metadata["source_sha256_after"] = _code_hashes(project_dir)
            if (
                source_before is not None
                and metadata["source_sha256_after"] != source_before
            ):
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
                    RuntimeError("pre-existing artifact changed")
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
        if reload_path.exists():
            reload_path.unlink()
        metadata["current_fit"] = None
        if metadata.get("status") != "completed" and submission_path.exists():
            submission_path.unlink()
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = _output_bytes(output_dir)
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            if submission_path.exists():
                submission_path.unlink()
            budget_error = RuntimeError("output directory exceeded 100 MiB budget")
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = budget_error
                metadata["error"] = {
                    "type": type(budget_error).__name__,
                    "message": str(budget_error),
                }
        for _ in range(4):
            save_json(metadata_path, metadata)
            actual_bytes = _output_bytes(output_dir)
            if actual_bytes == metadata["output_bytes"]:
                break
            metadata["output_bytes"] = actual_bytes
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    result = export_submission(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {"status": result["status"], "submission": result.get("submission")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
