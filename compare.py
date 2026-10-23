from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Keep numerical libraries bounded before importing NumPy, pandas, or CatBoost.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_info

from train import (
    CATEGORICAL_COLUMNS,
    CONFIRMATION_SPLIT_SEED,
    HGB_L2_REGULARIZATION,
    HGB_LEARNING_RATE,
    HGB_MAX_LEAF_NODES,
    HGB_RANDOM_STATE,
    ID_COLUMN,
    MAX_THREADS,
    TARGET_COLUMN,
    _model_settings,
    build_hgb_pipeline,
    make_split,
    package_versions,
    probabilities,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
    validate_submission_file,
)

CHALLENGER_MIN_GAIN = 0.0002
HGB_CHALLENGER_MAX_ITER = 1000
CATBOOST_ITERATIONS = 1200
CATBOOST_LEARNING_RATE = 0.06
CATBOOST_DEPTH = 6
DEEPER_CATBOOST_DEPTH = 8
CATBOOST_L2_REGULARIZATION = 3.0
CATBOOST_RANDOM_SEED = 42
CATBOOST_EARLY_STOPPING_ROUNDS = 100
CATBOOST_INTERNAL_SEED = 42
CATBOOST_INTERNAL_VALIDATION_FRACTION = 0.1
DEEPER_CATBOOST_TIMEOUT_SECONDS = 25 * 60

PROTECTED_ARTIFACTS = (
    "submission.csv",
    "final_model.joblib",
    "validation_logistic.csv",
    "validation_hist_gradient_boosting.csv",
    "run_metadata.json",
    "outer_split.npz",
    "confirmation_split.npz",
)


def prepare_catboost_frame(frame: pd.DataFrame, spec: Any) -> pd.DataFrame:
    """Return the validated feature frame for CatBoost native categories."""
    prepared = frame.loc[:, list(spec.feature_columns)].copy()
    for column in spec.categorical_columns:
        prepared[column] = (
            prepared[column].astype("string").fillna("<missing>").astype(object)
        )
    return prepared


def build_catboost_classifier(
    *,
    iterations: int = CATBOOST_ITERATIONS,
    early_stopping_rounds: int | None = CATBOOST_EARLY_STOPPING_ROUNDS,
    depth: int = CATBOOST_DEPTH,
) -> CatBoostClassifier:
    """Build the predeclared CatBoost configuration with an explicit depth."""
    settings: dict[str, Any] = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": CATBOOST_LEARNING_RATE,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": CATBOOST_L2_REGULARIZATION,
        "random_seed": CATBOOST_RANDOM_SEED,
        "thread_count": MAX_THREADS,
        "allow_writing_files": False,
        "verbose": False,
    }
    if early_stopping_rounds is not None:
        settings["early_stopping_rounds"] = early_stopping_rounds
    return CatBoostClassifier(**settings)


def make_catboost_internal_split(
    outer_fit_indices: np.ndarray, y: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split only an outer-fit partition for CatBoost early stopping."""
    outer_fit_indices = np.asarray(outer_fit_indices, dtype=np.int64)
    internal_fit, internal_validation = train_test_split(
        outer_fit_indices,
        test_size=CATBOOST_INTERNAL_VALIDATION_FRACTION,
        random_state=seed,
        stratify=y[outer_fit_indices],
    )
    if np.intersect1d(internal_fit, internal_validation).size:
        raise RuntimeError("CatBoost internal split overlaps")
    if not np.array_equal(
        np.sort(np.concatenate((internal_fit, internal_validation))),
        np.sort(outer_fit_indices),
    ):
        raise RuntimeError("CatBoost internal split does not cover outer fit rows")
    return internal_fit, internal_validation


def _resource_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "max_rss_mb": round(float(usage.ru_maxrss) / 1024.0, 1),
        "process_max_rss_mb": round(float(usage.ru_maxrss) / 1024.0, 1),
        "cpu_count": os.cpu_count(),
        "thread_limit": MAX_THREADS,
        "threadpools": [
            {
                "user_api": info.get("user_api"),
                "internal_api": info.get("internal_api"),
                "num_threads": info.get("num_threads"),
            }
            for info in threadpool_info()
        ],
    }


def _resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result = dict(after)
    result["max_rss_delta_mb"] = round(
        max(0.0, float(after["max_rss_mb"]) - float(before["max_rss_mb"])), 1
    )
    result["process_max_rss_delta_mb"] = result["max_rss_delta_mb"]
    return result


def _validation_result(
    *,
    candidate: str,
    split_seed: int,
    predictions: np.ndarray,
    y_holdout: np.ndarray,
    started: float,
    fit_started: float,
    before_resources: dict[str, Any],
    actual_iterations: int,
    best_iteration: int | None,
    settings: dict[str, Any],
    internal_validation: dict[str, Any] | None,
    fit_seconds: float | None = None,
) -> dict[str, Any]:
    after_resources = _resource_snapshot()
    return {
        "candidate": candidate,
        "split_seed": split_seed,
        "roc_auc": float(roc_auc_score(y_holdout, predictions)),
        "log_loss": float(log_loss(y_holdout, predictions)),
        "fit_seconds": round(
            time.perf_counter() - fit_started if fit_seconds is None else fit_seconds,
            3,
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "actual_iterations": actual_iterations,
        "best_iteration": best_iteration,
        "settings": settings,
        "internal_validation": internal_validation,
        "resources": _resource_delta(before_resources, after_resources),
    }


def _fit_hgb(
    spec: Any,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    *,
    split_seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    fit_started = time.perf_counter()
    before_resources = _resource_snapshot()
    model = build_hgb_pipeline(spec, max_iter=HGB_CHALLENGER_MAX_ITER)
    model.fit(X_fit, y_fit)
    predictions = probabilities(model, X_holdout)
    estimator = model.named_steps["model"]
    result = _validation_result(
        candidate="hgb_max_iter_1000",
        split_seed=split_seed,
        predictions=predictions,
        y_holdout=y_holdout,
        started=started,
        fit_started=fit_started,
        before_resources=before_resources,
        actual_iterations=int(estimator.n_iter_),
        best_iteration=int(estimator.n_iter_),
        settings=_model_settings(model, "hist_gradient_boosting"),
        internal_validation={
            "method": "scikit-learn validation_fraction",
            "fraction": 0.1,
            "n_iter_no_change": 20,
            "outer_holdout_used": False,
        },
    )
    return model, predictions, result


def _fit_catboost(
    spec: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    outer_fit: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    *,
    split_seed: int,
    depth: int = CATBOOST_DEPTH,
    iterations: int = CATBOOST_ITERATIONS,
    candidate: str = "catboost_native",
) -> tuple[CatBoostClassifier, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    internal_fit, internal_validation = make_catboost_internal_split(
        outer_fit,
        y,
        seed=CATBOOST_INTERNAL_SEED,
    )
    X_cat = prepare_catboost_frame(X, spec)
    started = time.perf_counter()
    fit_started = time.perf_counter()
    before_resources = _resource_snapshot()
    model = build_catboost_classifier(iterations=iterations, depth=depth)
    model.fit(
        X_cat.iloc[internal_fit],
        y[internal_fit],
        eval_set=(X_cat.iloc[internal_validation], y[internal_validation]),
        cat_features=list(spec.categorical_columns),
        verbose=False,
    )
    fit_seconds = time.perf_counter() - fit_started
    predictions = model.predict_proba(prepare_catboost_frame(X_holdout, spec))[:, 1]
    validate_probabilities(predictions, len(X_holdout))
    best_iteration = int(model.get_best_iteration())
    settings = {
        "depth": depth,
        "iterations": iterations,
        "learning_rate": CATBOOST_LEARNING_RATE,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": CATBOOST_L2_REGULARIZATION,
        "random_seed": CATBOOST_RANDOM_SEED,
        "thread_count": MAX_THREADS,
        "allow_writing_files": False,
        "early_stopping_rounds": CATBOOST_EARLY_STOPPING_ROUNDS,
        "cat_features": list(spec.categorical_columns),
    }
    result = _validation_result(
        candidate=candidate,
        split_seed=split_seed,
        predictions=predictions,
        y_holdout=y_holdout,
        started=started,
        fit_started=fit_started,
        before_resources=before_resources,
        actual_iterations=int(model.tree_count_),
        best_iteration=best_iteration,
        settings=settings,
        internal_validation={
            "seed": CATBOOST_INTERNAL_SEED,
            "fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
            "fit_rows": len(internal_fit),
            "validation_rows": len(internal_validation),
            "outer_holdout_used": False,
        },
        fit_seconds=fit_seconds,
    )
    return (
        model,
        predictions,
        result,
        {
            "fit_indices": internal_fit,
            "validation_indices": internal_validation,
        },
    )


def _load_recorded_split(
    path: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing recorded split: {path}")
    with np.load(path) as recorded:
        fit_indices = np.asarray(recorded["fit_indices"], dtype=np.int64)
        holdout_indices = np.asarray(recorded["holdout_indices"], dtype=np.int64)
    expected_fit, expected_holdout = make_split(train, y, seed)
    if not np.array_equal(fit_indices, expected_fit) or not np.array_equal(
        holdout_indices, expected_holdout
    ):
        raise RuntimeError(f"recorded split does not match train.py seed {seed}")
    return fit_indices, holdout_indices


def _save_validation_predictions(
    path: Path,
    train: pd.DataFrame,
    holdout_indices: np.ndarray,
    predictions: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "row_index": holdout_indices,
            ID_COLUMN: train.iloc[holdout_indices][ID_COLUMN].to_numpy(),
            "probability": predictions,
        }
    ).to_csv(path, index=False, float_format="%.17g")


def _fit_smoke_models(
    spec: Any, X: pd.DataFrame, y: np.ndarray, outer_fit: np.ndarray
) -> dict[str, Any]:
    class_indices = [outer_fit[y[outer_fit] == label][:2000] for label in (0, 1)]
    smoke_indices = np.concatenate(class_indices)
    smoke_internal_fit, smoke_internal_validation = make_catboost_internal_split(
        smoke_indices, y, seed=CATBOOST_INTERNAL_SEED
    )
    records: dict[str, Any] = {"rows": len(smoke_indices), "models": {}}

    started = time.perf_counter()
    hgb = build_hgb_pipeline(spec, max_iter=HGB_CHALLENGER_MAX_ITER)
    hgb.fit(X.iloc[smoke_indices], y[smoke_indices])
    records["models"]["hgb_max_iter_1000"] = {
        "completed": True,
        "actual_iterations": int(hgb.named_steps["model"].n_iter_),
        "seconds": round(time.perf_counter() - started, 3),
    }

    started = time.perf_counter()
    catboost = build_catboost_classifier()
    X_cat = prepare_catboost_frame(X, spec)
    catboost.fit(
        X_cat.iloc[smoke_internal_fit],
        y[smoke_internal_fit],
        eval_set=(X_cat.iloc[smoke_internal_validation], y[smoke_internal_validation]),
        cat_features=list(spec.categorical_columns),
        verbose=False,
    )
    records["models"]["catboost_native"] = {
        "completed": True,
        "actual_iterations": int(catboost.tree_count_),
        "best_iteration": int(catboost.get_best_iteration()),
        "seconds": round(time.perf_counter() - started, 3),
    }
    return records


def _package_versions() -> dict[str, str]:
    packages = package_versions()
    packages["catboost"] = version("catboost")
    packages["threadpoolctl"] = version("threadpoolctl")
    return packages


def _protected_artifact_hashes(artifacts_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(artifacts_dir / name)
        for name in PROTECTED_ARTIFACTS
        if (artifacts_dir / name).is_file()
    }


def _all_existing_artifact_hashes(artifacts_dir: Path) -> dict[str, str]:
    """Hash every existing artifact except the isolated experiment directory."""
    hashes: dict[str, str] = {}
    for path in sorted(artifacts_dir.rglob("*")):
        if (
            not path.is_file()
            or "deeper_catboost" in path.relative_to(artifacts_dir).parts
        ):
            continue
        hashes[str(path.relative_to(artifacts_dir))] = sha256_file(path)
    return hashes


class DeeperCatBoostTimeout(TimeoutError):
    """Raised when the bounded deeper-CatBoost experiment reaches its deadline."""


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _deeper_split_record(
    *,
    split_name: str,
    split_seed: int,
    split_path: Path,
    train: pd.DataFrame,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    internal_fit: np.ndarray,
    internal_validation: np.ndarray,
) -> dict[str, Any]:
    return {
        "name": split_name,
        "seed": split_seed,
        "path": str(split_path),
        "sha256": sha256_file(split_path),
        "fit_indices": fit_indices.tolist(),
        "holdout_indices": holdout_indices.tolist(),
        "fit_ids": train.iloc[fit_indices][ID_COLUMN].tolist(),
        "holdout_ids": train.iloc[holdout_indices][ID_COLUMN].tolist(),
        "internal_validation": {
            "seed": CATBOOST_INTERNAL_SEED,
            "fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
            "fit_indices": internal_fit.tolist(),
            "validation_indices": internal_validation.tolist(),
            "fit_ids": train.iloc[internal_fit][ID_COLUMN].tolist(),
            "validation_ids": train.iloc[internal_validation][ID_COLUMN].tolist(),
            "outer_holdout_used": False,
        },
    }


def _deeper_config() -> dict[str, Any]:
    """Return the one new, predeclared depth-8 CatBoost configuration."""
    return {
        "depth": DEEPER_CATBOOST_DEPTH,
        "iterations": CATBOOST_ITERATIONS,
        "learning_rate": CATBOOST_LEARNING_RATE,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": CATBOOST_L2_REGULARIZATION,
        "random_seed": CATBOOST_RANDOM_SEED,
        "thread_count": MAX_THREADS,
        "allow_writing_files": False,
        "early_stopping_rounds": CATBOOST_EARLY_STOPPING_ROUNDS,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "internal_validation_seed": CATBOOST_INTERNAL_SEED,
        "internal_validation_fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
    }


def run_deeper_catboost(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run only the predeclared depth-8 CatBoost comparison."""
    started = time.perf_counter()
    deeper_dir = artifacts_dir / "deeper_catboost"
    if deeper_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {deeper_dir}")
    deeper_dir.mkdir(parents=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    input_hashes_before = {
        "train.csv": sha256_file(train_path),
        "test.csv": sha256_file(test_path),
    }
    protected_before = _all_existing_artifact_hashes(artifacts_dir)
    prior_metadata_path = artifacts_dir / "challengers" / "compare_metadata.json"
    prior_metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
    if prior_metadata["input_sha256"] != input_hashes_before:
        raise RuntimeError("input fingerprints do not match prior CatBoost metadata")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    spec = validate_inputs(train, test)
    y = target_array(train)
    X = train.loc[:, list(spec.feature_columns)]
    X_test = test.loc[:, list(spec.feature_columns)]
    split_definitions = (
        ("outer", 42, "outer_split.npz"),
        ("confirmation", CONFIRMATION_SPLIT_SEED, "confirmation_split.npz"),
    )
    split_records: dict[str, dict[str, Any]] = {}
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name, split_seed, split_filename in split_definitions:
        split_path = artifacts_dir / split_filename
        fit_indices, holdout_indices = _load_recorded_split(
            split_path, train, y, split_seed
        )
        prior_internal = prior_metadata["internal_validation_splits"][split_name]
        internal_fit = np.asarray(prior_internal["fit_indices"], dtype=np.int64)
        internal_validation = np.asarray(
            prior_internal["validation_indices"], dtype=np.int64
        )
        expected_internal_fit, expected_internal_validation = (
            make_catboost_internal_split(
                fit_indices,
                y,
                seed=CATBOOST_INTERNAL_SEED,
            )
        )
        if not np.array_equal(
            internal_fit, expected_internal_fit
        ) or not np.array_equal(internal_validation, expected_internal_validation):
            raise RuntimeError("prior CatBoost internal split does not match seed 42")
        if (
            np.intersect1d(internal_fit, holdout_indices).size
            or np.intersect1d(internal_validation, holdout_indices).size
        ):
            raise RuntimeError("CatBoost internal split overlaps outer holdout")
        split_records[split_name] = _deeper_split_record(
            split_name=split_name,
            split_seed=split_seed,
            split_path=split_path,
            train=train,
            fit_indices=fit_indices,
            holdout_indices=holdout_indices,
            internal_fit=internal_fit,
            internal_validation=internal_validation,
        )
        split_arrays[split_name] = (
            fit_indices,
            holdout_indices,
            internal_fit,
            internal_validation,
        )

    config = _deeper_config()
    candidate = "catboost_depth8"
    source_hashes = {
        "compare.py": sha256_file(Path(__file__)),
        "train.py": sha256_file(Path(__file__).with_name("train.py")),
    }
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "python": sys.version,
        "packages": _package_versions(),
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
        "timeout_seconds": DEEPER_CATBOOST_TIMEOUT_SECONDS,
        "input_sha256_before": input_hashes_before,
        "source_sha256": source_hashes,
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "predeclared_config": config,
        "baseline": {
            "candidate": "catboost_native",
            "depth": CATBOOST_DEPTH,
            "metadata_path": str(prior_metadata_path),
            "metadata_sha256": sha256_file(prior_metadata_path),
            "validation_results": prior_metadata["validation_results"][
                "catboost_native"
            ],
        },
        "recorded_splits": split_records,
        "validation_results": [],
        "completed_split_names": [],
        "current_fit": None,
        "selection": None,
        "final": None,
        "artifacts_dir": str(deeper_dir),
    }
    metadata_path = deeper_dir / "compare_metadata.json"
    save_json(metadata_path, metadata)

    previous_alarm = signal.getitimer(signal.ITIMER_REAL)
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    caught_error: Exception | None = None
    cleanup_error: Exception | None = None

    def stop_experiment(_signum: int, _frame: Any) -> None:
        raise DeeperCatBoostTimeout("deeper CatBoost experiment timed out")

    try:
        signal.signal(signal.SIGALRM, stop_experiment)
        signal.signal(signal.SIGTERM, stop_experiment)
        signal.setitimer(signal.ITIMER_REAL, DEEPER_CATBOOST_TIMEOUT_SECONDS)
        for split_name, split_seed, _split_filename in split_definitions:
            fit_indices, holdout_indices, internal_fit, internal_validation = (
                split_arrays[split_name]
            )
            metadata["current_fit"] = {
                "split_name": split_name,
                "split_seed": split_seed,
                "status": "running",
            }
            save_json(metadata_path, metadata)
            model, predictions, result, _ = _fit_catboost(
                spec,
                X,
                y,
                fit_indices,
                X.iloc[holdout_indices],
                y[holdout_indices],
                split_seed=split_seed,
                depth=DEEPER_CATBOOST_DEPTH,
                iterations=CATBOOST_ITERATIONS,
                candidate=candidate,
            )
            del model, internal_fit, internal_validation
            validation_path = (
                deeper_dir / f"validation_catboost_depth8_seed{split_seed}.csv"
            )
            _save_validation_predictions(
                validation_path, train, holdout_indices, predictions
            )
            result["validation_predictions_path"] = str(validation_path)
            metadata["validation_results"].append(result)
            metadata["completed_split_names"].append(split_name)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)

        baseline_by_seed = {
            int(result["split_seed"]): result
            for result in metadata["baseline"]["validation_results"]
        }
        comparison = []
        for result in metadata["validation_results"]:
            baseline_result = baseline_by_seed[int(result["split_seed"])]
            gain = float(result["roc_auc"]) - float(baseline_result["roc_auc"])
            comparison.append(
                {
                    "split_seed": result["split_seed"],
                    "deeper_roc_auc": result["roc_auc"],
                    "depth6_roc_auc": baseline_result["roc_auc"],
                    "auc_gain": gain,
                    "qualifies": gain >= CHALLENGER_MIN_GAIN,
                    "deeper_log_loss": result["log_loss"],
                    "depth6_log_loss": baseline_result["log_loss"],
                    "actual_iterations": result["actual_iterations"],
                    "best_iteration": result["best_iteration"],
                    "fit_seconds": result["fit_seconds"],
                }
            )
        qualifies = len(comparison) == 2 and all(
            item["qualifies"] for item in comparison
        )
        metadata["selection"] = {
            "baseline_candidate": "catboost_native",
            "baseline_depth": CATBOOST_DEPTH,
            "minimum_auc_gain_each_seed": CHALLENGER_MIN_GAIN,
            "comparison": comparison,
            "qualifies": qualifies,
            "decision": ("refit_depth8" if qualifies else "retain_depth6_candidate"),
        }
        if qualifies:
            metadata["current_fit"] = {"status": "refit_running"}
            save_json(metadata_path, metadata)
            final = _refit_winner(
                candidate,
                spec,
                X,
                y,
                X_test,
                test[ID_COLUMN],
                {candidate: metadata["validation_results"]},
                deeper_dir,
                depth=DEEPER_CATBOOST_DEPTH,
                output_stem="catboost_depth8",
            )
            metadata["final"] = final
            metadata["current_fit"] = None
            metadata["status"] = "completed"
        else:
            metadata["status"] = "completed"
    except DeeperCatBoostTimeout as error:
        caught_error = error
        metadata["status"] = "timeout"
        metadata["error"] = _error_record(error)
        metadata["timeout_note"] = (
            "Only validation fits written before the timeout are reported; the "
            "interrupted fit and any later fit are absent."
        )
    except Exception as error:
        caught_error = error
        metadata["status"] = "failed"
        metadata["error"] = _error_record(error)
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, *previous_alarm)
        except Exception as error:
            cleanup_error = error
        try:
            signal.signal(signal.SIGALRM, previous_alarm_handler)
        except Exception as error:
            cleanup_error = cleanup_error or error
        try:
            signal.signal(signal.SIGTERM, previous_term_handler)
        except Exception as error:
            cleanup_error = cleanup_error or error

    finalization_errors: list[Exception] = []
    if cleanup_error is not None:
        finalization_errors.append(cleanup_error)

    input_hashes_after: dict[str, str] | None = None
    try:
        input_hashes_after = {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        }
    except Exception as error:
        finalization_errors.append(error)

    protected_after: dict[str, str] | None = None
    try:
        protected_after = _all_existing_artifact_hashes(artifacts_dir)
    except Exception as error:
        finalization_errors.append(error)

    if input_hashes_after is not None:
        metadata["input_sha256_after"] = input_hashes_after
        if input_hashes_before != input_hashes_after:
            finalization_errors.append(
                RuntimeError("input CSV fingerprint changed during deeper experiment")
            )
    metadata["protected_baseline_sha256_before"] = protected_before
    if protected_after is not None:
        metadata["protected_baseline_sha256_after"] = protected_after
        metadata["protected_baseline_unchanged"] = protected_before == protected_after
        if protected_before != protected_after:
            finalization_errors.append(
                RuntimeError("existing artifact changed during deeper experiment")
            )

    def mark_finalization_failure() -> RuntimeError | None:
        if not finalization_errors:
            return None
        finalization_error = RuntimeError(
            "; ".join(str(error) for error in finalization_errors)
        )
        metadata["status"] = "failed"
        record = _error_record(finalization_error)
        if caught_error is None:
            metadata["error"] = record
        else:
            metadata["finalization_error"] = record
        return finalization_error

    finalization_error = mark_finalization_failure()
    metadata["total_seconds"] = round(time.perf_counter() - started, 3)
    if protected_after is not None:
        try:
            save_json(deeper_dir / "protected_baseline_sha256.json", protected_after)
        except Exception as error:
            finalization_errors.append(error)
            finalization_error = mark_finalization_failure()
    try:
        save_json(metadata_path, metadata)
    except Exception as error:
        finalization_errors.append(error)
        finalization_error = mark_finalization_failure()
        try:
            save_json(metadata_path, metadata)
        except Exception as second_error:
            finalization_errors.append(second_error)
            finalization_error = mark_finalization_failure()

    if caught_error is not None and not isinstance(caught_error, DeeperCatBoostTimeout):
        if finalization_error is not None:
            raise caught_error from finalization_error
        raise caught_error
    if finalization_error is not None:
        if caught_error is not None:
            raise finalization_error from caught_error
        raise finalization_error
    return metadata


def _clean_reload_predictions(
    model_path: Path,
    model_kind: str,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        features_path = directory_path / "features.csv"
        output_path = directory_path / "predictions.npy"
        feature_frame.to_csv(features_path, index=False, float_format="%.17g")
        if model_kind == "catboost":
            code = (
                "import numpy as np, pandas as pd, sys; "
                "from catboost import CatBoostClassifier; "
                "model = CatBoostClassifier(); "
                "model.load_model(sys.argv[1]); "
                "features = pd.read_csv(sys.argv[2]); "
                "np.save(sys.argv[3], model.predict_proba(features)[:, 1])"
            )
        else:
            code = (
                "import joblib, numpy as np, pandas as pd, sys; "
                "model = joblib.load(sys.argv[1]); "
                "features = pd.read_csv(sys.argv[2]); "
                "np.save(sys.argv[3], model.predict_proba(features)[:, 1])"
            )
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(model_path),
                str(features_path),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.load(output_path)


def _refit_winner(
    candidate: str,
    spec: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    validation_results: dict[str, list[dict[str, Any]]],
    challengers_dir: Path,
    *,
    depth: int = CATBOOST_DEPTH,
    output_stem: str | None = None,
) -> dict[str, Any]:
    is_catboost = candidate.startswith("catboost")
    selected_counts = [
        int(result["best_iteration"] + 1)
        if is_catboost
        else int(result["actual_iterations"])
        for result in validation_results[candidate]
    ]
    selected_iterations = int(round(float(np.median(selected_counts))))
    started = time.perf_counter()
    before_resources = _resource_snapshot()
    fit_started = time.perf_counter()
    if is_catboost:
        model = build_catboost_classifier(
            iterations=selected_iterations,
            early_stopping_rounds=None,
            depth=depth,
        )
        model.fit(
            prepare_catboost_frame(X, spec),
            y,
            cat_features=list(spec.categorical_columns),
            verbose=False,
        )
        fit_seconds = time.perf_counter() - fit_started
        predictions = model.predict_proba(prepare_catboost_frame(X_test, spec))[:, 1]
        model_filename = (
            "final_model_catboost.cbm"
            if output_stem is None
            else f"final_model_{output_stem}.cbm"
        )
        model_path = challengers_dir / model_filename
        model.save_model(model_path)
        model_kind = "catboost"
    else:
        model = build_hgb_pipeline(
            spec,
            max_iter=selected_iterations,
            early_stopping=False,
        )
        model.fit(X, y)
        fit_seconds = time.perf_counter() - fit_started
        predictions = probabilities(model, X_test)
        model_path = challengers_dir / "final_model_hgb_max_iter_1000.joblib"
        joblib.dump(model, model_path, compress=3)
        model_kind = "joblib"
    validate_probabilities(predictions, len(X_test))

    feature_spec_path = challengers_dir / (
        "feature_spec.json"
        if output_stem is None
        else f"feature_spec_{output_stem}.json"
    )
    save_json(
        feature_spec_path,
        {
            "feature_columns": list(spec.feature_columns),
            "numeric_columns": list(spec.numeric_columns),
            "categorical_columns": list(spec.categorical_columns),
        },
    )
    submission_path = challengers_dir / (
        "submission.csv" if output_stem is None else f"submission_{output_stem}.csv"
    )
    pd.DataFrame({ID_COLUMN: test_ids.to_numpy(), TARGET_COLUMN: predictions}).to_csv(
        submission_path, index=False, float_format="%.17g"
    )
    submission = validate_submission_file(submission_path, test_ids)
    reloaded_predictions = _clean_reload_predictions(
        model_path,
        model_kind,
        prepare_catboost_frame(X_test, spec) if is_catboost else X_test,
    )
    reload_difference = float(np.max(np.abs(predictions - reloaded_predictions)))
    csv_difference = float(
        np.max(np.abs(predictions - submission[TARGET_COLUMN].to_numpy()))
    )
    if reload_difference > 1e-12 or csv_difference > 1e-12:
        raise RuntimeError("challenger reload or CSV predictions differ")
    return {
        "model": candidate,
        "model_path": str(model_path),
        "feature_spec_path": str(feature_spec_path),
        "submission_path": str(submission_path),
        "submission_rows": len(submission),
        "selected_iterations": selected_iterations,
        "iteration_policy": (
            "median(best_iteration + 1) from completed validation fits"
            if is_catboost
            else "median actual iterations from two validation fits"
        ),
        "early_stopping": False,
        "actual_iterations": int(model.tree_count_)
        if is_catboost
        else int(model.named_steps["model"].n_iter_),
        "best_iteration": None,
        "fit_seconds": round(fit_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "settings": (
            {
                "depth": depth,
                "iterations": selected_iterations,
                "learning_rate": CATBOOST_LEARNING_RATE,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "l2_leaf_reg": CATBOOST_L2_REGULARIZATION,
                "random_seed": CATBOOST_RANDOM_SEED,
                "thread_count": MAX_THREADS,
                "allow_writing_files": False,
                "early_stopping_rounds": None,
                "cat_features": list(spec.categorical_columns),
            }
            if is_catboost
            else _model_settings(model, "hist_gradient_boosting")
        ),
        "resources": _resource_delta(before_resources, _resource_snapshot()),
        "reload_check": {
            "rows": len(X_test),
            "max_absolute_difference": reload_difference,
            "csv_max_absolute_difference": csv_difference,
        },
    }


def run_compare(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the two fixed challenger configurations on the recorded splits."""
    started = time.perf_counter()
    challengers_dir = artifacts_dir / "challengers"
    challengers_dir.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_artifact_hashes(artifacts_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    spec = validate_inputs(train, test)
    y = target_array(train)
    X = train.loc[:, list(spec.feature_columns)]
    X_test = test.loc[:, list(spec.feature_columns)]
    input_hashes = {
        "train.csv": sha256_file(train_path),
        "test.csv": sha256_file(test_path),
    }
    baseline = json.loads((artifacts_dir / "run_metadata.json").read_text())
    if baseline["input_sha256"] != input_hashes:
        raise RuntimeError("input fingerprints do not match baseline metadata")

    smoke = _fit_smoke_models(
        spec,
        X,
        y,
        _load_recorded_split(artifacts_dir / "outer_split.npz", train, y, 42)[0],
    )
    split_definitions = (
        ("outer", 42, "outer_split.npz"),
        ("confirmation", CONFIRMATION_SPLIT_SEED, "confirmation_split.npz"),
    )
    validation_results: dict[str, list[dict[str, Any]]] = {
        "hgb_max_iter_1000": [],
        "catboost_native": [],
    }
    internal_splits: dict[str, Any] = {}
    for split_name, split_seed, split_filename in split_definitions:
        fit_indices, holdout_indices = _load_recorded_split(
            artifacts_dir / split_filename,
            train,
            y,
            split_seed,
        )
        hgb_model, hgb_predictions, hgb_result = _fit_hgb(
            spec,
            X.iloc[fit_indices],
            y[fit_indices],
            X.iloc[holdout_indices],
            y[holdout_indices],
            split_seed=split_seed,
        )
        _save_validation_predictions(
            challengers_dir / f"validation_hgb_max_iter_1000_seed{split_seed}.csv",
            train,
            holdout_indices,
            hgb_predictions,
        )
        validation_results["hgb_max_iter_1000"].append(hgb_result)
        del hgb_model

        catboost_model, catboost_predictions, catboost_result, internal = _fit_catboost(
            spec,
            X,
            y,
            fit_indices,
            X.iloc[holdout_indices],
            y[holdout_indices],
            split_seed=split_seed,
        )
        _save_validation_predictions(
            challengers_dir / f"validation_catboost_native_seed{split_seed}.csv",
            train,
            holdout_indices,
            catboost_predictions,
        )
        validation_results["catboost_native"].append(catboost_result)
        internal_splits[split_name] = {
            "seed": CATBOOST_INTERNAL_SEED,
            "fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
            "fit_indices": internal["fit_indices"].tolist(),
            "validation_indices": internal["validation_indices"].tolist(),
            "fit_ids": train.iloc[internal["fit_indices"]][ID_COLUMN].tolist(),
            "validation_ids": train.iloc[internal["validation_indices"]][
                ID_COLUMN
            ].tolist(),
            "outer_fit_rows": len(fit_indices),
            "outer_holdout_rows": len(holdout_indices),
            "disjoint_from_outer_holdout": not bool(
                np.intersect1d(internal["validation_indices"], holdout_indices).size
            ),
        }
        del catboost_model

    baseline_scores = {
        "outer": baseline["outer_results"]["hist_gradient_boosting"]["roc_auc"],
        "confirmation": baseline["confirmation_results"]["hist_gradient_boosting"][
            "roc_auc"
        ],
    }
    selection_candidates: dict[str, Any] = {}
    for candidate, results in validation_results.items():
        scores = [float(result["roc_auc"]) for result in results]
        gains = [
            scores[0] - float(baseline_scores["outer"]),
            scores[1] - float(baseline_scores["confirmation"]),
        ]
        selection_candidates[candidate] = {
            "scores": scores,
            "mean_roc_auc": float(np.mean(scores)),
            "gains_vs_original_hgb": gains,
            "qualifies": all(gain >= CHALLENGER_MIN_GAIN for gain in gains),
        }
    qualifying = [
        candidate
        for candidate, details in selection_candidates.items()
        if details["qualifies"]
    ]
    selected_candidate = (
        max(qualifying, key=lambda name: selection_candidates[name]["mean_roc_auc"])
        if qualifying
        else None
    )
    final = None
    if selected_candidate is not None:
        final = _refit_winner(
            selected_candidate,
            spec,
            X,
            y,
            X_test,
            test[ID_COLUMN],
            validation_results,
            challengers_dir,
        )

    protected_after = _protected_artifact_hashes(artifacts_dir)
    if protected_before != protected_after:
        raise RuntimeError("protected baseline artifact changed during comparison")
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "packages": _package_versions(),
        "threads": MAX_THREADS,
        "input_sha256": input_hashes,
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "protected_baseline_sha256": protected_after,
        "recorded_splits": {
            "outer": {"seed": 42, "path": str(artifacts_dir / "outer_split.npz")},
            "confirmation": {
                "seed": CONFIRMATION_SPLIT_SEED,
                "path": str(artifacts_dir / "confirmation_split.npz"),
            },
        },
        "smoke_fit": smoke,
        "predeclared_configs": {
            "hgb_max_iter_1000": {
                "max_iter": HGB_CHALLENGER_MAX_ITER,
                "max_leaf_nodes": HGB_MAX_LEAF_NODES,
                "learning_rate": HGB_LEARNING_RATE,
                "l2_regularization": HGB_L2_REGULARIZATION,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 20,
                "random_state": HGB_RANDOM_STATE,
            },
            "catboost_native": {
                "depth": CATBOOST_DEPTH,
                "iterations": CATBOOST_ITERATIONS,
                "learning_rate": CATBOOST_LEARNING_RATE,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "l2_leaf_reg": CATBOOST_L2_REGULARIZATION,
                "random_seed": CATBOOST_RANDOM_SEED,
                "thread_count": MAX_THREADS,
                "allow_writing_files": False,
                "early_stopping_rounds": CATBOOST_EARLY_STOPPING_ROUNDS,
                "categorical_columns": list(CATEGORICAL_COLUMNS),
                "internal_validation_seed": CATBOOST_INTERNAL_SEED,
                "internal_validation_fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
            },
        },
        "internal_validation_splits": internal_splits,
        "validation_results": validation_results,
        "baseline_original_hgb": {
            "outer_roc_auc": baseline_scores["outer"],
            "confirmation_roc_auc": baseline_scores["confirmation"],
            "source": str(artifacts_dir / "run_metadata.json"),
        },
        "selection": {
            "minimum_roc_auc_gain": CHALLENGER_MIN_GAIN,
            "candidates": selection_candidates,
            "qualifying_candidates": qualifying,
            "selected_candidate": selected_candidate,
            "policy": (
                "A candidate must gain at least 0.0002 ROC AUC over the original "
                "HGB on both overlapping fixed seeded holdouts; choose the highest "
                "mean of those two AUCs. Otherwise retain the original model."
            ),
            "holdout_note": (
                "The seeded holdouts overlap; agreement is a stability diagnostic, "
                "not independent confirmation."
            ),
        },
        "final": final,
        "total_seconds": round(time.perf_counter() - started, 3),
        "artifacts_dir": str(challengers_dir),
    }
    save_json(challengers_dir / "compare_metadata.json", metadata)
    save_json(challengers_dir / "protected_baseline_sha256.json", protected_after)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded airline satisfaction challenger comparison."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--deeper-catboost",
        action="store_true",
        help="Run only the predeclared depth-8 CatBoost comparison.",
    )
    args = parser.parse_args()
    if args.deeper_catboost:
        metadata = run_deeper_catboost(args.data_dir, args.artifacts_dir)
        output = {
            "status": metadata["status"],
            "selection": metadata["selection"],
            "final": metadata["final"],
            "total_seconds": metadata.get("total_seconds"),
        }
    else:
        metadata = run_compare(args.data_dir, args.artifacts_dir)
        output = {
            "selected_candidate": metadata["selection"]["selected_candidate"],
            "selection": metadata["selection"]["candidates"],
            "final": metadata["final"],
            "total_seconds": metadata["total_seconds"],
        }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
