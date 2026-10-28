"""Run the approved qualified source-teacher refit and local export."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
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
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from compare import _resource_delta, _resource_snapshot, prepare_catboost_frame
from eco_business_specialist import _verify_prior_oof
from lightgbm_capacity import _artifact_snapshot, _input_hashes
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
)
from source_augmentation import clean_source_rows
from source_teacher_cv import _source_hashes as source_teacher_cv_source_hashes
from source_teacher_cv import _verify_teacher_cache
from source_teacher_experiment import (
    SOURCE_TEACHER_FEATURE,
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
    validate_submission_file,
)

OUTPUT_NAME = "source_teacher_export"
TEACHER_MODEL_NAME = "teacher_model.txt"
STUDENT_MODEL_NAME = "student_model.txt"
TEACHER_LEVELS_NAME = "teacher_category_levels.json"
STUDENT_LEVELS_NAME = "student_category_levels.json"
FEATURE_SPEC_NAME = "feature_spec.json"
SUBMISSION_NAME = "submission_slowcat_source_teacher_50_50.csv"
METADATA_NAME = "export_metadata.json"
RELOAD_NAME = "reload_predictions.npy"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_TEST_ROWS = 299844
EXPECTED_SOURCE_ROWS = 129859
EXPECTED_TEACHER_ITERATIONS = 531
EXPECTED_CATBOOST_ITERATIONS = 4740
EXPECTED_CATBOOST_MODEL_SHA256 = (
    "85466352ce440da5aea2234b034a55df1b8c7e35c33c2fed1b66ea8377924653"
)
EXPECTED_BASELINE_SHA256 = (
    "5a5f985fd98a78bce5cf93e0a2dde6f228b36d25cc4d60c54b859be6d7262059"
)
BLEND_WEIGHTS = {"catboost": 0.5, "student": 0.5}


class SourceTeacherExportTimeout(TimeoutError):
    """Raised when the bounded export reaches its internal deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise SourceTeacherExportTimeout("source-teacher export exceeded its deadline")


def derive_student_iterations(metadata: dict[str, Any]) -> int:
    """Derive the full student count from five validated one-based CV records."""
    if metadata.get("status") != "completed":
        raise RuntimeError("source-teacher CV metadata is not completed")
    if metadata.get("selection", {}).get("qualifies") is not True:
        raise RuntimeError("source-teacher CV gate is not qualified")
    hypothesis = metadata.get("fixed_hypothesis", {})
    if hypothesis.get("fold_count") != 5:
        raise RuntimeError("source-teacher CV does not contain five folds")
    fold_results = metadata.get("fold_results")
    if not isinstance(fold_results, list) or len(fold_results) != 5:
        raise RuntimeError("source-teacher CV must contain five fold results")
    fold_ids = [int(row.get("fold_id", -1)) for row in fold_results]
    if sorted(fold_ids) != [1, 2, 3, 4, 5]:
        raise RuntimeError("source-teacher CV folds are not five distinct folds")
    counts: list[int] = []
    for row in sorted(fold_results, key=lambda item: int(item["fold_id"])):
        student = row.get("student", {})
        best = int(student.get("best_iteration_1based", 0))
        actual = int(student.get("actual_iterations", 0))
        if best <= 0 or actual <= 0 or best != actual:
            raise RuntimeError("student iteration provenance is invalid")
        counts.append(best)
    selected = int(median(counts))
    if selected <= 0:
        raise RuntimeError("derived student iteration count is invalid")
    return selected


def validate_cv_oof(
    arrays: dict[str, np.ndarray],
    ids: np.ndarray,
    prior: dict[str, Any],
    teacher_probability: np.ndarray,
) -> None:
    """Validate the actual eight-column cached CV contract before any refit."""
    if set(arrays) != {
        "row_index",
        "id",
        "fold_id",
        "teacher_probability",
        "catboost_probability",
        "student_probability",
        "baseline_probability",
        "candidate_probability",
    }:
        raise RuntimeError("source-teacher CV OOF fields changed")
    if not np.array_equal(arrays["row_index"], np.arange(len(ids))):
        raise RuntimeError("source-teacher CV OOF row coverage changed")
    validate_test_ids(arrays["id"], ids)
    if not np.array_equal(arrays["fold_id"], prior["fold_ids"]):
        raise RuntimeError("source-teacher CV OOF fold coverage changed")
    if not np.array_equal(arrays["teacher_probability"], teacher_probability):
        raise RuntimeError("source-teacher CV teacher feature changed")
    if not np.array_equal(arrays["catboost_probability"], prior["catboost_seed42"]):
        raise RuntimeError("source-teacher CV CatBoost component changed")
    for name in (
        "teacher_probability",
        "catboost_probability",
        "student_probability",
        "baseline_probability",
        "candidate_probability",
    ):
        validate_probabilities(arrays[name], len(ids))
    cat = arrays["catboost_probability"]
    if not np.array_equal(
        arrays["baseline_probability"], 0.5 * cat + 0.5 * prior["lightgbm"]
    ):
        raise RuntimeError("source-teacher CV baseline formula changed")
    if not np.array_equal(
        arrays["candidate_probability"], 0.5 * cat + 0.5 * arrays["student_probability"]
    ):
        raise RuntimeError("source-teacher CV candidate formula changed")


def derive_teacher_iterations(metadata: dict[str, Any]) -> int:
    """Validate and return the recorded source-only teacher round count."""
    teacher = metadata.get("teacher", {})
    probe = teacher.get("probe", {})
    full_fit = teacher.get("full_fit", {})
    if teacher.get("source_rows") != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("source-teacher source row count changed")
    selected = int(probe.get("selected_best_iteration", 0))
    actual = int(full_fit.get("actual_rounds", 0))
    if selected != EXPECTED_TEACHER_ITERATIONS or actual != selected:
        raise RuntimeError("source-teacher round provenance changed")
    if full_fit.get("fit_rows") != EXPECTED_SOURCE_ROWS:
        raise RuntimeError("source-teacher full-fit row count changed")
    if full_fit.get("eval_set") is not None or full_fit.get("callbacks") != []:
        raise RuntimeError("source-teacher full fit uses evaluation or callbacks")
    return selected


def build_export_config(iterations: int) -> dict[str, Any]:
    """Return the approved config with only the derived iteration count changed."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return {**PREDECLARED_CONFIG, "n_estimators": iterations}


def blend_probabilities(catboost: np.ndarray, student: np.ndarray) -> np.ndarray:
    """Return the fixed 50/50 CatBoost/student blend."""
    catboost = np.asarray(catboost, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if catboost.ndim != 1 or student.shape != catboost.shape:
        raise ValueError("blend components must have equal one-dimensional shapes")
    validate_probabilities(catboost, len(catboost))
    validate_probabilities(student, len(student))
    result = BLEND_WEIGHTS["catboost"] * catboost + BLEND_WEIGHTS["student"] * student
    validate_probabilities(result, len(result))
    return result


def validate_test_ids(actual: Any, expected: Any) -> None:
    """Require unique IDs in the exact expected test order."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if len(actual_array) != len(expected_array) or not np.array_equal(
        actual_array, expected_array
    ):
        raise ValueError("prediction IDs do not exactly match test order")
    if pd.Index(actual_array).duplicated().any():
        raise ValueError("prediction IDs are not unique")


def build_feature_spec(
    original: InputSpec,
    source_levels: dict[str, list[str]],
    competition_levels: dict[str, list[str]],
) -> dict[str, Any]:
    """Record the original and one-feature-extended model contracts."""
    original_spec = {
        "feature_columns": list(original.feature_columns),
        "numeric_columns": list(original.numeric_columns),
        "categorical_columns": list(original.categorical_columns),
    }
    extended_spec = {
        "feature_columns": [*original_spec["feature_columns"], SOURCE_TEACHER_FEATURE],
        "numeric_columns": [*original_spec["numeric_columns"], SOURCE_TEACHER_FEATURE],
        "categorical_columns": list(original_spec["categorical_columns"]),
    }
    return {
        "original": original_spec,
        "extended": extended_spec,
        "category_levels": {
            "source_full": source_levels,
            "competition_full": competition_levels,
        },
        "numeric_missing_policy": "preserve numeric NaN; no imputation",
        "categorical_missing_policy": (
            "string <missing> for CatBoost; native categorical levels for LightGBM"
        ),
        "id_column_excluded": ID_COLUMN,
        "target_column_excluded": TARGET_COLUMN,
        "weights": dict(BLEND_WEIGHTS),
    }


def fit_full_lightgbm(
    model: Any, features: pd.DataFrame, labels: np.ndarray, spec: InputSpec
) -> None:
    """Fit one full-data LightGBM model without eval data or early stopping."""
    model.fit(
        features,
        labels,
        categorical_feature=list(spec.categorical_columns),
    )


def _code_hashes(project_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(project_dir)): sha256_file(path)
        for path in sorted(project_dir.glob("*.py"))
    }


def _versions() -> dict[str, str]:
    result = package_versions()
    for name in ("catboost", "lightgbm", "threadpoolctl"):
        result[name] = version(name)
    return result


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve_recorded_path(value: str, project_dir: Path, artifacts_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    for candidate in (project_dir / path, artifacts_dir / path):
        if candidate.is_file():
            return candidate.resolve()
    return (project_dir / path).resolve()


def _save_model_metadata(
    output_dir: Path,
    feature_spec: dict[str, Any],
    teacher_levels: dict[str, list[str]],
    student_levels: dict[str, list[str]],
) -> dict[str, str]:
    feature_path = output_dir / FEATURE_SPEC_NAME
    teacher_levels_path = output_dir / TEACHER_LEVELS_NAME
    student_levels_path = output_dir / STUDENT_LEVELS_NAME
    save_json(feature_path, feature_spec)
    save_json(teacher_levels_path, teacher_levels)
    save_json(student_levels_path, student_levels)
    return {
        FEATURE_SPEC_NAME: sha256_file(feature_path),
        TEACHER_LEVELS_NAME: sha256_file(teacher_levels_path),
        STUDENT_LEVELS_NAME: sha256_file(student_levels_path),
    }


def _fit_and_save_lightgbm(
    *,
    name: str,
    model: Any,
    features: pd.DataFrame,
    labels: np.ndarray,
    spec: InputSpec,
    model_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> dict[str, Any]:
    current: dict[str, Any] = {
        "name": name,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fit_rows": len(features),
        "progress_log": [{"event": "fit_started"}],
    }
    metadata["current_fit"] = current
    save_json(metadata_path, metadata)
    before = _resource_snapshot()
    fit_started = time.perf_counter()
    fit_full_lightgbm(model, features, labels, spec)
    fit_seconds = time.perf_counter() - fit_started
    after = _resource_snapshot()
    if int(model.booster_.current_iteration()) != int(
        model.get_params()["n_estimators"]
    ):
        raise RuntimeError(f"{name} did not retain its fixed iteration count")
    model.booster_.save_model(str(model_path))
    record = {
        "fit_rows": len(features),
        "fit_seconds": round(fit_seconds, 3),
        "actual_iterations": int(model.booster_.current_iteration()),
        "settings": model.get_params(),
        "eval_set": None,
        "callbacks": [],
        "early_stopping": False,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "resources": _resource_delta(before, after),
    }
    current["status"] = "completed"
    current["completed_utc"] = datetime.now(timezone.utc).isoformat()
    current["progress_log"].append({"event": "fit_completed"})
    metadata["current_fit"] = None
    metadata.setdefault("completed_fits", []).append(name)
    metadata[f"{name}_fit"] = record
    save_json(metadata_path, metadata)
    return record


def _validate_frozen_catboost(
    project_dir: Path,
    artifacts_dir: Path,
    spec: InputSpec,
) -> tuple[CatBoostClassifier, Path, dict[str, Any]]:
    metadata_path = artifacts_dir / "slower_learning" / "compare_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final = metadata.get("final", {})
    if metadata.get("status") != "completed":
        raise RuntimeError("frozen slower-learning metadata is not completed")
    if final.get("selected_iterations") != EXPECTED_CATBOOST_ITERATIONS:
        raise RuntimeError("frozen CatBoost selected iteration count changed")
    model_path = _resolve_recorded_path(
        final["catboost_model_path"], project_dir, artifacts_dir
    )
    if (
        final.get("catboost_model_sha256") != EXPECTED_CATBOOST_MODEL_SHA256
        or sha256_file(model_path) != EXPECTED_CATBOOST_MODEL_SHA256
    ):
        raise RuntimeError("frozen CatBoost model fingerprint is not authorized")
    model = CatBoostClassifier()
    model.load_model(model_path)
    if model.tree_count_ != EXPECTED_CATBOOST_ITERATIONS:
        raise RuntimeError("frozen CatBoost tree count changed")
    if len(model.feature_names_) != len(spec.feature_columns):
        raise RuntimeError("frozen CatBoost feature count changed")
    names = list(model.feature_names_)
    if names != list(spec.feature_columns):
        raise RuntimeError("frozen CatBoost feature order changed")
    return model, model_path, metadata


def _clean_reload_predictions(
    *,
    model_paths: dict[str, Path],
    level_paths: dict[str, Path],
    feature_spec_path: Path,
    test_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> None:
    """Reproduce all test predictions in a fresh interpreter."""
    code = r"""
import csv
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

(
    teacher_path, student_path, cat_path, teacher_levels_path, student_levels_path,
    feature_path, test_path, baseline_path, output_path,
) = sys.argv[1:]
feature_spec = json.loads(open(feature_path, encoding="utf-8").read())
teacher_levels = json.loads(open(teacher_levels_path, encoding="utf-8").read())
student_levels = json.loads(open(student_levels_path, encoding="utf-8").read())
test = pd.read_csv(test_path)
original = feature_spec["original"]["feature_columns"]
extended = feature_spec["extended"]["feature_columns"]
categorical = feature_spec["original"]["categorical_columns"]
source = test.loc[:, original].copy()
for column in categorical:
    source[column] = pd.Categorical(
        source[column].astype("string"), categories=teacher_levels[column]
    )
teacher = lgb.Booster(model_file=teacher_path)
teacher_values = np.asarray(teacher.predict(source), dtype=np.float64)
student_frame = test.loc[:, original].copy()
for column in categorical:
    student_frame[column] = pd.Categorical(
        student_frame[column].astype("string"), categories=student_levels[column]
    )
student_frame["source_teacher_probability"] = teacher_values
student_frame = student_frame.loc[:, extended]
student = lgb.Booster(model_file=student_path)
student_values = np.asarray(student.predict(student_frame), dtype=np.float64)
cat_frame = test.loc[:, original].copy()
for column in categorical:
    cat_frame[column] = (
        cat_frame[column].astype("string").fillna("<missing>").astype(object)
    )
cat = CatBoostClassifier()
cat.load_model(cat_path)
cat_values = np.asarray(cat.predict_proba(cat_frame)[:, 1], dtype=np.float64)
with open(baseline_path, encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    next(reader)
    baseline = np.asarray([float(row[1]) for row in reader], dtype=np.float64)
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
            str(model_paths["teacher"]),
            str(model_paths["student"]),
            str(model_paths["catboost"]),
            str(level_paths["teacher"]),
            str(level_paths["student"]),
            str(feature_spec_path),
            str(test_path),
            str(baseline_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _output_bytes(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())


def export_submission(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Refit the qualified teacher/student pair and write the local submission."""
    project_dir = Path(__file__).resolve().parent
    data_dir = data_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / METADATA_NAME
    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "authorized_qualified_export": True,
        "qualification_source": "artifacts/source_teacher_cv/compare_metadata.json",
        "current_fit": None,
        "completed_fits": [],
        "selection": None,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "submission_path": str(output_dir / SUBMISSION_NAME),
    }
    save_json(metadata_path, metadata)
    started = time.perf_counter()
    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    protected_before: dict[str, str] | None = None
    input_before: dict[str, str] | None = None
    code_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    submission_path = output_dir / SUBMISSION_NAME
    reload_path = output_dir / RELOAD_NAME
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        code_before = _code_hashes(project_dir)
        metadata.update(
            {
                "protected_artifact_sha256_before": protected_before,
                "input_sha256_before": input_before,
                "source_sha256_before": code_before,
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
        if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
            raise RuntimeError("competition row counts changed")
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        test_ids = test[ID_COLUMN].to_numpy(dtype=np.int64)
        prior_oof = _verify_prior_oof(data_dir, artifacts_dir, train, y, input_before)
        teacher_cache = _verify_teacher_cache(artifacts_dir, train, spec, input_before)
        teacher_metadata = teacher_cache["metadata"]
        teacher_iterations = derive_teacher_iterations(teacher_metadata)
        cv_path = artifacts_dir / "source_teacher_cv" / "compare_metadata.json"
        cv_metadata = json.loads(cv_path.read_text(encoding="utf-8"))
        if (
            cv_metadata.get("input_sha256_before") != input_before
            or cv_metadata.get("input_sha256_after") != input_before
        ):
            raise RuntimeError("source-teacher CV input fingerprints changed")
        current_cv_sources = source_teacher_cv_source_hashes()
        if (
            cv_metadata.get("source_sha256_before") != current_cv_sources
            or cv_metadata.get("source_sha256_after") != current_cv_sources
        ):
            raise RuntimeError("source-teacher CV source fingerprints changed")
        if cv_metadata.get("fixed_hypothesis", {}).get("student_configuration") != dict(
            PREDECLARED_CONFIG
        ):
            raise RuntimeError("source-teacher CV configuration changed")
        student_iterations = derive_student_iterations(cv_metadata)
        if student_iterations != 533:
            raise RuntimeError("approved student iteration derivation changed")
        if cv_metadata.get("fold_membership", {}).get("rows") != EXPECTED_TRAIN_ROWS:
            raise RuntimeError("source-teacher CV row coverage changed")

        oof_path = _resolve_recorded_path(
            cv_metadata["oof_predictions"]["path"], project_dir, artifacts_dir
        )
        with np.load(oof_path) as saved:
            validate_cv_oof(
                {name: saved[name] for name in saved.files},
                ids,
                prior_oof,
                teacher_cache["probabilities"],
            )
        if sha256_file(oof_path) != cv_metadata["oof_predictions"]["sha256"]:
            raise RuntimeError("source-teacher CV OOF fingerprint changed")

        source_raw = pd.read_csv(data_dir / "source_airline" / "data.csv")
        source_x, source_y, cleaning = clean_source_rows(source_raw, train, test, spec)
        recorded_cleaning = teacher_metadata.get("source_cleaning", {}).get("actual")
        if (
            cleaning != recorded_cleaning
            or cleaning.get("clean_rows") != EXPECTED_SOURCE_ROWS
        ):
            raise RuntimeError(
                "current clean source rows do not match approved provenance"
            )
        feature_spec = build_feature_spec(
            spec,
            fit_category_levels(source_x, spec),
            fit_category_levels(train.loc[:, list(spec.feature_columns)], spec),
        )
        teacher_levels = feature_spec["category_levels"]["source_full"]
        competition_levels = feature_spec["category_levels"]["competition_full"]
        metadata.update(
            {
                "teacher_iterations": teacher_iterations,
                "student_iterations": student_iterations,
                "iteration_lineage": {
                    "teacher": "source_teacher_feature teacher.full_fit.actual_rounds",
                    "student": (
                        "source_teacher_cv five student best_iteration_1based median"
                    ),
                    "student_cv_counts": [
                        int(row["student"]["best_iteration_1based"])
                        for row in sorted(
                            cv_metadata["fold_results"],
                            key=lambda item: int(item["fold_id"]),
                        )
                    ],
                },
                "qualification": {
                    "source_teacher_cv_selection": cv_metadata["selection"],
                    "source_teacher_cv_summary": cv_metadata.get("summary"),
                    "source_teacher_cv_metadata_sha256": sha256_file(cv_path),
                    "source_teacher_cv_oof_sha256": sha256_file(oof_path),
                    "source_teacher_cv_oof_rows": len(train),
                },
                "reused_artifact_fingerprints": {
                    "teacher_metadata_sha256": sha256_file(
                        teacher_cache["metadata_path"]
                    ),
                    "teacher_cache_sha256": teacher_cache["cache_sha256"],
                    "prior_oof_metadata_sha256": sha256_file(
                        prior_oof["metadata_path"]
                    ),
                    "prior_oof_sha256": sha256_file(prior_oof["oof_path"]),
                },
                "source_cleaning": cleaning,
                "feature_spec": feature_spec,
                "configuration": {
                    "teacher": build_export_config(teacher_iterations),
                    "student": build_export_config(student_iterations),
                    "fit_policy": (
                        "full-data fit with no eval_set, no early stopping, "
                        "no callbacks"
                    ),
                },
                "configuration_sha256": _hash_json(
                    {
                        "teacher": build_export_config(teacher_iterations),
                        "student": build_export_config(student_iterations),
                    }
                ),
            }
        )
        feature_hashes = _save_model_metadata(
            output_dir, feature_spec, teacher_levels, competition_levels
        )
        metadata["files"] = feature_hashes
        save_json(metadata_path, metadata)

        teacher_features = prepare_lightgbm_frame(source_x, spec, teacher_levels)
        teacher_model = build_lightgbm_classifier(n_estimators=teacher_iterations)
        teacher_record = _fit_and_save_lightgbm(
            name="teacher",
            model=teacher_model,
            features=teacher_features,
            labels=source_y,
            spec=spec,
            model_path=output_dir / TEACHER_MODEL_NAME,
            metadata=metadata,
            metadata_path=metadata_path,
        )
        metadata["files"][TEACHER_MODEL_NAME] = sha256_file(
            output_dir / TEACHER_MODEL_NAME
        )
        save_json(metadata_path, metadata)

        competition_original = train.loc[:, list(spec.feature_columns)]
        test_original = test.loc[:, list(spec.feature_columns)]
        teacher_train = np.asarray(
            teacher_model.predict_proba(
                prepare_lightgbm_frame(competition_original, spec, teacher_levels)
            )[:, 1],
            dtype=np.float64,
        )
        teacher_test = np.asarray(
            teacher_model.predict_proba(
                prepare_lightgbm_frame(test_original, spec, teacher_levels)
            )[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(teacher_train, len(train))
        validate_probabilities(teacher_test, len(test))
        cached_teacher = teacher_cache["probabilities"]
        teacher_cache_difference = float(np.max(np.abs(teacher_train - cached_teacher)))
        if teacher_cache_difference > 1e-12:
            raise RuntimeError(
                "refit teacher does not reproduce frozen train predictions: "
                f"{teacher_cache_difference}"
            )
        metadata["teacher_train_prediction_check"] = {
            "rows": len(train),
            "max_abs_difference": teacher_cache_difference,
            "tolerance": 1e-12,
        }
        save_json(metadata_path, metadata)

        student_features, student_spec = append_teacher_feature(
            competition_original, teacher_train, spec
        )
        student_test_features, _ = append_teacher_feature(
            test_original, teacher_test, spec
        )
        student_model = build_lightgbm_classifier(n_estimators=student_iterations)
        student_levels = fit_category_levels(student_features, student_spec)
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
        save_json(metadata_path, metadata)
        student_test = np.asarray(
            student_model.predict_proba(
                prepare_lightgbm_frame(
                    student_test_features, student_spec, student_levels
                )
            )[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(student_test, len(test))
        catboost_model, catboost_path, baseline_metadata = _validate_frozen_catboost(
            project_dir, artifacts_dir, spec
        )
        catboost_test = np.asarray(
            catboost_model.predict_proba(prepare_catboost_frame(test, spec))[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(catboost_test, len(test))
        baseline_path = _resolve_recorded_path(
            baseline_metadata["final"]["submission_path"], project_dir, artifacts_dir
        )
        baseline_hash_before = sha256_file(baseline_path)
        if (
            baseline_hash_before != EXPECTED_BASELINE_SHA256
            or baseline_hash_before != baseline_metadata["final"]["submission_sha256"]
        ):
            raise RuntimeError(
                "frozen baseline submission fingerprint is not authorized"
            )
        validate_submission_file(baseline_path, test[ID_COLUMN])
        candidate = blend_probabilities(catboost_test, student_test)
        if len(candidate) != EXPECTED_TEST_ROWS:
            raise RuntimeError("candidate row count changed")
        with submission_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([ID_COLUMN, TARGET_COLUMN])
            writer.writerows(
                (int(identifier), format(float(value), ".17g"))
                for identifier, value in zip(test_ids, candidate, strict=True)
            )
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
        baseline_hash_after = sha256_file(baseline_path)
        if baseline_hash_after != baseline_hash_before:
            raise RuntimeError("frozen baseline submission changed during export")

        _clean_reload_predictions(
            model_paths={
                "teacher": output_dir / TEACHER_MODEL_NAME,
                "student": output_dir / STUDENT_MODEL_NAME,
                "catboost": catboost_path,
            },
            level_paths={
                "teacher": output_dir / TEACHER_LEVELS_NAME,
                "student": output_dir / STUDENT_LEVELS_NAME,
            },
            feature_spec_path=output_dir / FEATURE_SPEC_NAME,
            test_path=data_dir / "test.csv",
            baseline_path=baseline_path,
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
        metadata.update(
            {
                "status": "completed",
                "current_fit": None,
                "blend_components": {
                    "catboost_model": str(catboost_path),
                    "student_model": str(output_dir / STUDENT_MODEL_NAME),
                    "weights": dict(BLEND_WEIGHTS),
                    "original_lightgbm_component_used": False,
                },
                "reused_artifacts": {
                    "catboost_model_path": str(catboost_path),
                    "catboost_model_sha256": sha256_file(catboost_path),
                    "catboost_tree_count": int(catboost_model.tree_count_),
                    "catboost_feature_count": len(catboost_model.feature_names_),
                    "baseline_submission_path": str(baseline_path),
                    "baseline_submission_sha256": baseline_hash_before,
                    "baseline_submission_sha256_after": baseline_hash_after,
                    "baseline_metadata_sha256": sha256_file(
                        artifacts_dir / "slower_learning" / "compare_metadata.json"
                    ),
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
                "input_sha256_before": input_before,
                "source_sha256_before": code_before,
                "fit_records": {"teacher": teacher_record, "student": student_record},
                "teacher_model": teacher_record,
                "student_model": student_record,
                "output_budget_bytes": OUTPUT_BUDGET_BYTES,
            }
        )
        del teacher_model, student_model, catboost_model
        gc.collect()
    except BaseException as error:
        caught = error
        metadata["status"] = (
            "timeout" if isinstance(error, SourceTeacherExportTimeout) else "failed"
        )
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
                code_before is not None
                and metadata["source_sha256_after"] != code_before
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
        if metadata.get("status") != "completed":
            metadata["selection"] = None
            if submission_path.exists():
                submission_path.unlink()
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = _output_bytes(output_dir)
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            if submission_path.exists():
                submission_path.unlink()
            budget_error = RuntimeError("output directory exceeded 100 MiB budget")
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["error"] = {
                "type": type(budget_error).__name__,
                "message": str(budget_error),
            }
            if caught is None:
                caught = budget_error
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
            {
                "status": result["status"],
                "submission": result.get("submission"),
                "output_dir": result["output_dir"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
