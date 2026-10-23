from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Set numerical thread limits before importing NumPy, pandas, or LightGBM.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from compare import (
    _load_recorded_split,
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
)
from train import (
    CONFIRMATION_SPLIT_SEED,
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

LIGHTGBM_ITERATIONS = 1500
LIGHTGBM_LEARNING_RATE = 0.04
LIGHTGBM_NUM_LEAVES = 31
LIGHTGBM_MAX_DEPTH = -1
LIGHTGBM_MIN_CHILD_SAMPLES = 40
LIGHTGBM_REG_LAMBDA = 1.0
LIGHTGBM_SUBSAMPLE = 1.0
LIGHTGBM_COLSAMPLE_BYTREE = 1.0
LIGHTGBM_RANDOM_STATE = 42
LIGHTGBM_EARLY_STOPPING_ROUNDS = 100
LIGHTGBM_INTERNAL_SEED = 42
LIGHTGBM_INTERNAL_VALIDATION_FRACTION = 0.1
MINIMUM_AUC_GAIN = 0.0002
EXPERIMENT_TIMEOUT_SECONDS = 15 * 60

PREDECLARED_CONFIG: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": LIGHTGBM_ITERATIONS,
    "learning_rate": LIGHTGBM_LEARNING_RATE,
    "num_leaves": LIGHTGBM_NUM_LEAVES,
    "max_depth": LIGHTGBM_MAX_DEPTH,
    "min_child_samples": LIGHTGBM_MIN_CHILD_SAMPLES,
    "reg_lambda": LIGHTGBM_REG_LAMBDA,
    "subsample": LIGHTGBM_SUBSAMPLE,
    "colsample_bytree": LIGHTGBM_COLSAMPLE_BYTREE,
    "random_state": LIGHTGBM_RANDOM_STATE,
    "n_jobs": MAX_THREADS,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


class LightGBMExperimentTimeout(TimeoutError):
    """Raised when the external bounded run asks this process to stop."""


def build_lightgbm_classifier(**overrides: Any) -> lgb.LGBMClassifier:
    """Build the one predeclared classifier, with test-only overrides allowed."""
    settings = dict(PREDECLARED_CONFIG)
    settings.update(overrides)
    return lgb.LGBMClassifier(**settings)


def fit_category_levels(frame: pd.DataFrame, spec: InputSpec) -> dict[str, list[str]]:
    """Record category levels from only the rows used to fit a model."""
    levels: dict[str, list[str]] = {}
    for column in spec.categorical_columns:
        values = frame[column].astype("string").dropna()
        levels[column] = [str(value) for value in pd.unique(values)]
    return levels


def prepare_lightgbm_frame(
    frame: pd.DataFrame,
    spec: InputSpec,
    category_levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Align categories to fit-only levels while preserving numeric NaNs."""
    prepared = frame.loc[:, list(spec.feature_columns)].copy()
    for column in spec.categorical_columns:
        prepared[column] = pd.Categorical(
            prepared[column].astype("string"),
            categories=category_levels[column],
        )
    return prepared


def _fit_callbacks() -> list[Any]:
    return [
        lgb.early_stopping(LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(0),
    ]


def _fit_with_inner_validation(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    spec: InputSpec,
    *,
    callbacks: list[Any],
) -> tuple[dict[str, list[str]], float, dict[str, Any]]:
    category_levels = fit_category_levels(X.iloc[fit_indices], spec)
    X_fit = prepare_lightgbm_frame(X.iloc[fit_indices], spec, category_levels)
    X_validation = prepare_lightgbm_frame(
        X.iloc[validation_indices], spec, category_levels
    )
    before_resources = _resource_snapshot()
    fit_started = time.perf_counter()
    model.fit(
        X_fit,
        y[fit_indices],
        eval_set=[(X_validation, y[validation_indices])],
        eval_names=["inner_validation"],
        eval_metric="auc",
        categorical_feature=list(spec.categorical_columns),
        callbacks=callbacks,
    )
    fit_seconds = time.perf_counter() - fit_started
    after_resources = _resource_snapshot()
    best_iteration = int(model.best_iteration_)
    actual_iterations = int(model.booster_.current_iteration())
    return (
        category_levels,
        fit_seconds,
        {
            "best_iteration": best_iteration,
            "actual_iterations": actual_iterations,
            "resources": _resource_delta(before_resources, after_resources),
        },
    )


def _validation_result(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    holdout_indices: np.ndarray,
    category_levels: dict[str, list[str]],
    spec: InputSpec,
    *,
    split_seed: int,
    fit_seconds: float,
    fit_details: dict[str, Any],
    internal_fit: np.ndarray,
    internal_validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = model.predict_proba(
        prepare_lightgbm_frame(X.iloc[holdout_indices], spec, category_levels)
    )[:, 1]
    validate_probabilities(predictions, len(holdout_indices))
    result = {
        "candidate": "lightgbm_native",
        "split_seed": split_seed,
        "roc_auc": float(roc_auc_score(y[holdout_indices], predictions)),
        "log_loss": float(log_loss(y[holdout_indices], predictions)),
        "fit_seconds": round(fit_seconds, 3),
        "actual_iterations": fit_details["actual_iterations"],
        "best_iteration": fit_details["best_iteration"],
        "settings": dict(PREDECLARED_CONFIG),
        "early_stopping": {
            "eval_metric": "auc",
            "rounds": LIGHTGBM_EARLY_STOPPING_ROUNDS,
            "callbacks": ["early_stopping", "log_evaluation"],
        },
        "internal_validation": {
            "seed": LIGHTGBM_INTERNAL_SEED,
            "fraction": LIGHTGBM_INTERNAL_VALIDATION_FRACTION,
            "fit_rows": len(internal_fit),
            "validation_rows": len(internal_validation),
            "outer_holdout_used": False,
        },
        "resources": fit_details["resources"],
    }
    return predictions, result


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


def _selected_iterations(results: list[dict[str, Any]]) -> int:
    """Use LightGBM's 1-based best_iteration_ values without an index offset."""
    return int(
        round(float(np.median([result["best_iteration"] for result in results])))
    )


def _selection(
    validation_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_seed = {
        int(result["split_seed"]): result for result in baseline_results
    }
    comparison = []
    for result in validation_results:
        baseline = baseline_by_seed[int(result["split_seed"])]
        gain = float(result["roc_auc"]) - float(baseline["roc_auc"])
        comparison.append(
            {
                "split_seed": result["split_seed"],
                "lightgbm_roc_auc": result["roc_auc"],
                "catboost_depth6_roc_auc": baseline["roc_auc"],
                "auc_gain": gain,
                "qualifies": gain >= MINIMUM_AUC_GAIN,
                "lightgbm_log_loss": result["log_loss"],
                "catboost_depth6_log_loss": baseline["log_loss"],
                "best_iteration": result["best_iteration"],
                "actual_iterations": result["actual_iterations"],
            }
        )
    qualifies = len(comparison) == 2 and all(item["qualifies"] for item in comparison)
    return {
        "baseline_candidate": "catboost_native",
        "baseline_depth": 6,
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "comparison": comparison,
        "qualifies": qualifies,
        "decision": "refit_all_training_rows" if qualifies else "retain_catboost",
    }


def _fresh_reload_predictions(
    model_path: Path,
    feature_spec_path: Path,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        features_path = directory_path / "features.csv"
        predictions_path = directory_path / "predictions.npy"
        feature_frame.to_csv(features_path, index=False, float_format="%.17g")
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, lightgbm as lgb, numpy as np, pandas as pd, sys; "
                    "spec=json.loads(open(sys.argv[2], encoding='utf-8').read()); "
                    "frame=pd.read_csv(sys.argv[3]); "
                    "[frame.__setitem__(c, pd.Categorical(frame[c].astype('string'), "
                    "categories=spec['category_levels'][c])) for c in "
                    "spec['categorical_columns']]; "
                    "model=lgb.Booster(model_file=sys.argv[1]); "
                    "np.save(sys.argv[4], model.predict(frame))"
                ),
                str(model_path),
                str(feature_spec_path),
                str(features_path),
                str(predictions_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return np.load(predictions_path)


def _package_versions() -> dict[str, str]:
    packages = package_versions()
    packages["lightgbm"] = version("lightgbm")
    return packages


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _artifact_snapshot(artifacts_dir: Path, excluded_dir: Path) -> dict[str, str]:
    """Hash pre-existing artifact files while excluding this run's directory."""
    excluded_relative = excluded_dir.relative_to(artifacts_dir)
    all_hashes: dict[str, str] = {}
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifacts_dir)
        if relative == excluded_relative or excluded_relative in relative.parents:
            continue
        all_hashes[str(relative)] = sha256_file(path)
    return all_hashes


def _protected_hashes(
    artifacts_dir: Path, keys: set[str], excluded_dir: Path
) -> dict[str, str]:
    all_hashes = _artifact_snapshot(artifacts_dir, excluded_dir)
    return {key: all_hashes[key] for key in sorted(keys) if key in all_hashes}


def _source_hashes() -> dict[str, str]:
    return {
        "lightgbm_experiment.py": sha256_file(Path(__file__)),
        "train.py": sha256_file(Path(__file__).with_name("train.py")),
        "compare.py": sha256_file(Path(__file__).with_name("compare.py")),
    }


def _persist_initialization_failure(
    output_dir: Path, started: float, error: BaseException
) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "error": _error_record(error),
        "selection": None,
        "final": None,
        "total_seconds": round(time.perf_counter() - started, 3),
        "artifacts_dir": str(output_dir),
    }
    try:
        save_json(output_dir / "compare_metadata.json", metadata)
    except BaseException as persistence_error:
        raise error from persistence_error


def _terminal_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": metadata.get("status", "failed"),
        "selection": metadata.get("selection"),
        "final": metadata.get("final"),
        "error": metadata.get("error"),
        "total_seconds": metadata.get("total_seconds"),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly one bounded LightGBM challenger experiment."""
    started = time.perf_counter()
    output_dir = artifacts_dir / "lightgbm"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    protected_before_all = _artifact_snapshot(artifacts_dir, output_dir)
    protected_keys = set(protected_before_all)
    output_dir.mkdir(parents=True)

    try:
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
        input_before = {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        }
        protected_before = {
            key: protected_before_all[key] for key in sorted(protected_keys)
        }
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        spec = validate_inputs(train, test)
        y = target_array(train)
        X = train.loc[:, list(spec.feature_columns)]
        X_test = test.loc[:, list(spec.feature_columns)]

        prior_metadata_path = artifacts_dir / "challengers" / "compare_metadata.json"
        prior_metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
        if prior_metadata["input_sha256"] != input_before:
            raise RuntimeError("input fingerprints do not match CatBoost metadata")
        baseline_results = prior_metadata["validation_results"]["catboost_native"]
        split_definitions = (
            ("outer", 42, "outer_split.npz"),
            ("confirmation", CONFIRMATION_SPLIT_SEED, "confirmation_split.npz"),
        )
        split_arrays: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        split_records: dict[str, Any] = {}
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
            expected_fit, expected_validation = make_catboost_internal_split(
                fit_indices, y, seed=LIGHTGBM_INTERNAL_SEED
            )
            if not np.array_equal(internal_fit, expected_fit) or not np.array_equal(
                internal_validation, expected_validation
            ):
                raise RuntimeError(
                    "recorded CatBoost inner split does not match seed 42"
                )
            if np.intersect1d(internal_validation, holdout_indices).size:
                raise RuntimeError("inner validation overlaps outer holdout")
            split_arrays[split_name] = (
                fit_indices,
                holdout_indices,
                internal_fit,
                internal_validation,
            )
            split_records[split_name] = {
                "seed": split_seed,
                "path": str(split_path),
                "sha256": sha256_file(split_path),
                "fit_indices": fit_indices.tolist(),
                "holdout_indices": holdout_indices.tolist(),
                "inner_fit_indices": internal_fit.tolist(),
                "inner_validation_indices": internal_validation.tolist(),
                "outer_holdout_used_for_eval": False,
            }

        config_hash = hashlib.sha256(
            json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
        ).hexdigest()
        metadata: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "python": sys.version,
            "packages": _package_versions(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "threads": MAX_THREADS,
            "external_timeout_seconds": EXPERIMENT_TIMEOUT_SECONDS,
            "input_sha256_before": input_before,
            "source_sha256": _source_hashes(),
            "config_sha256": config_hash,
            "feature_columns": list(spec.feature_columns),
            "numeric_columns": list(spec.numeric_columns),
            "categorical_columns": list(spec.categorical_columns),
            "predeclared_config": dict(PREDECLARED_CONFIG),
            "baseline": {
                "candidate": "catboost_native",
                "depth": 6,
                "metadata_path": str(prior_metadata_path),
                "metadata_sha256": sha256_file(prior_metadata_path),
                "validation_results": baseline_results,
            },
            "recorded_splits": split_records,
            "metric_policy": {
                "configured_metric": "auc",
                "registered_metrics": ["auc"],
                "early_stopping_metric": "auc",
                "callback_first_metric_only": False,
            },
            "smoke_fit": {"completed": False},
            "validation_results": [],
            "completed_split_names": [],
            "current_fit": None,
            "selection": None,
            "final": None,
            "protected_baseline_sha256_before": protected_before,
            "protected_artifact_keys": sorted(protected_keys),
            "artifacts_dir": str(output_dir),
        }
        metadata_path = output_dir / "compare_metadata.json"
        save_json(metadata_path, metadata)
    except BaseException as error:
        _persist_initialization_failure(output_dir, started, error)
        raise

    caught_error: BaseException | None = None
    integrity_error: RuntimeError | None = None
    previous_term_handler = signal.getsignal(signal.SIGTERM)

    def stop_experiment(_signum: int, _frame: Any) -> None:
        raise LightGBMExperimentTimeout("LightGBM experiment timed out")

    try:
        signal.signal(signal.SIGTERM, stop_experiment)
        smoke_fit = np.asarray(split_arrays["outer"][2][:20000], dtype=np.int64)
        smoke_validation = np.asarray(split_arrays["outer"][3][:2000], dtype=np.int64)
        smoke_model = build_lightgbm_classifier()
        smoke_levels, smoke_seconds, smoke_details = _fit_with_inner_validation(
            smoke_model,
            X,
            y,
            smoke_fit,
            smoke_validation,
            spec,
            callbacks=_fit_callbacks(),
        )
        del smoke_model, smoke_levels
        metadata["smoke_fit"] = {
            "completed": True,
            "rows": len(smoke_fit),
            "validation_rows": len(smoke_validation),
            "fit_seconds": round(smoke_seconds, 3),
            **smoke_details,
        }
        save_json(metadata_path, metadata)

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
            model = build_lightgbm_classifier()
            levels, fit_seconds, fit_details = _fit_with_inner_validation(
                model,
                X,
                y,
                internal_fit,
                internal_validation,
                spec,
                callbacks=_fit_callbacks(),
            )
            predictions, result = _validation_result(
                model,
                X,
                y,
                holdout_indices,
                levels,
                spec,
                split_seed=split_seed,
                fit_seconds=fit_seconds,
                fit_details=fit_details,
                internal_fit=internal_fit,
                internal_validation=internal_validation,
            )
            validation_path = output_dir / f"validation_lightgbm_seed{split_seed}.csv"
            _save_validation_predictions(
                validation_path, train, holdout_indices, predictions
            )
            result["validation_predictions_path"] = str(validation_path)
            metadata["validation_results"].append(result)
            metadata["completed_split_names"].append(split_name)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del model, levels, predictions

        metadata["selection"] = _selection(
            metadata["validation_results"], baseline_results
        )
        if metadata["selection"]["qualifies"]:
            selected_iterations = _selected_iterations(metadata["validation_results"])
            metadata["current_fit"] = {"status": "refit_running"}
            save_json(metadata_path, metadata)
            category_levels = fit_category_levels(X, spec)
            X_all = prepare_lightgbm_frame(X, spec, category_levels)
            X_test_aligned = prepare_lightgbm_frame(X_test, spec, category_levels)
            model = build_lightgbm_classifier(n_estimators=selected_iterations)
            before_resources = _resource_snapshot()
            fit_started = time.perf_counter()
            model.fit(
                X_all,
                y,
                categorical_feature=list(spec.categorical_columns),
                callbacks=[lgb.log_evaluation(0)],
            )
            fit_seconds = time.perf_counter() - fit_started
            predictions = model.predict_proba(X_test_aligned)[:, 1]
            validate_probabilities(predictions, len(X_test_aligned))
            model_path = output_dir / "final_model_lightgbm.txt"
            model.booster_.save_model(model_path)
            feature_spec_path = output_dir / "feature_spec.json"
            save_json(
                feature_spec_path,
                {
                    "feature_columns": list(spec.feature_columns),
                    "numeric_columns": list(spec.numeric_columns),
                    "categorical_columns": list(spec.categorical_columns),
                    "category_levels": category_levels,
                },
            )
            submission_path = output_dir / "submission_lightgbm.csv"
            pd.DataFrame(
                {ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: predictions}
            ).to_csv(submission_path, index=False, float_format="%.17g")
            submission = validate_submission_file(submission_path, test[ID_COLUMN])
            reloaded = _fresh_reload_predictions(model_path, feature_spec_path, X_test)
            reload_difference = float(np.max(np.abs(predictions - reloaded)))
            csv_difference = float(
                np.max(np.abs(predictions - submission[TARGET_COLUMN].to_numpy()))
            )
            if reload_difference > 1e-12 or csv_difference > 1e-12:
                raise RuntimeError("native reload or CSV predictions differ")
            metadata["final"] = {
                "model": "lightgbm_native",
                "model_path": str(model_path),
                "feature_spec_path": str(feature_spec_path),
                "submission_path": str(submission_path),
                "submission_rows": len(submission),
                "selected_iterations": selected_iterations,
                "iteration_policy": (
                    "median LightGBM best_iteration_ from two validation fits"
                ),
                "early_stopping": False,
                "actual_iterations": int(model.booster_.current_iteration()),
                "best_iteration": None,
                "fit_seconds": round(fit_seconds, 3),
                "resources": _resource_delta(before_resources, _resource_snapshot()),
                "schema_validation": {
                    "rows": len(submission),
                    "columns": list(submission.columns),
                    "id_order_exact": True,
                    "probabilities_finite_and_in_range": True,
                },
                "reload_check": {
                    "rows": len(predictions),
                    "max_absolute_difference": reload_difference,
                    "csv_max_absolute_difference": csv_difference,
                },
                "model_sha256": sha256_file(model_path),
                "feature_spec_sha256": sha256_file(feature_spec_path),
            }
            metadata["current_fit"] = None
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = "failed"
        metadata["error"] = _error_record(error)
        current = metadata.get("current_fit")
        if current is not None:
            current["status"] = "failed"
            current["error"] = _error_record(error)
        metadata["current_fit"] = current
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        input_after = {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        }
        protected_after = _protected_hashes(artifacts_dir, protected_keys, output_dir)
        metadata["input_sha256_after"] = input_after
        metadata["protected_baseline_sha256_after"] = protected_after
        metadata["protected_baseline_unchanged"] = protected_before == protected_after
        if input_after != input_before or protected_before != protected_after:
            integrity_error = RuntimeError(
                "input or protected artifact fingerprint changed"
            )
            metadata["status"] = "failed"
            metadata["error"] = _error_record(integrity_error)
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        save_json(output_dir / "protected_baseline_sha256.json", protected_after)
        save_json(metadata_path, metadata)

    if integrity_error is not None:
        if caught_error is not None:
            raise integrity_error from caught_error
        raise integrity_error
    if caught_error is not None:
        raise caught_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one bounded native LightGBM challenger experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        metadata = run_experiment(args.data_dir, args.artifacts_dir)
    except Exception as error:
        metadata = {
            "status": "failed",
            "selection": None,
            "final": None,
            "error": _error_record(error),
            "total_seconds": None,
        }
        print(json.dumps(_terminal_summary(metadata), sort_keys=True))
        raise
    print(json.dumps(_terminal_summary(metadata), sort_keys=True))


if __name__ == "__main__":
    main()
