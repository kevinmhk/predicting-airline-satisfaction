from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Keep BLAS and OpenMP usage bounded before NumPy and scikit-learn are imported.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ID_COLUMN = "id"
TARGET_COLUMN = "satisfaction"
CATEGORICAL_COLUMNS = (
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
)
OUTER_SPLIT_SEED = 42
CONFIRMATION_SPLIT_SEED = 2024
HGB_MAX_ITER = 200
HGB_MAX_LEAF_NODES = 31
HGB_LEARNING_RATE = 0.08
HGB_L2_REGULARIZATION = 0.1
HGB_RANDOM_STATE = 42
HGB_MIN_GAIN = 0.0005
LOGISTIC_MAX_ITER = 500
LOGISTIC_RANDOM_STATE = 42
MAX_THREADS = 2


@dataclass(frozen=True)
class InputSpec:
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def validate_inputs(train: pd.DataFrame, test: pd.DataFrame) -> InputSpec:
    """Validate the input contract and return the stable feature definition."""
    missing_train = {ID_COLUMN, TARGET_COLUMN} - set(train.columns)
    missing_test = {ID_COLUMN} - set(test.columns)
    if missing_train:
        raise ValueError(f"train is missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test is missing required columns: {sorted(missing_test)}")

    train_features = [
        column for column in train.columns if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    test_features = [column for column in test.columns if column != ID_COLUMN]
    if set(train_features) != set(test_features):
        raise ValueError("train and test predictors must match")

    for name, frame in (("train", train), ("test", test)):
        if frame[ID_COLUMN].isna().any():
            raise ValueError(f"{name} IDs must not be missing")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} IDs must be unique")

    overlap = pd.Index(train[ID_COLUMN]).intersection(pd.Index(test[ID_COLUMN]))
    if len(overlap) != 0:
        raise ValueError("train and test IDs must be disjoint")

    target = train[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError("satisfaction must not be missing")
    if pd.api.types.is_bool_dtype(target):
        target_values = target.astype(np.int8)
    else:
        values = set(pd.unique(target))
        if not values.issubset({0, 1}):
            raise ValueError("satisfaction must be a binary 0/1 or boolean target")
        target_values = target.astype(np.int8)
    if set(target_values.unique()) != {0, 1}:
        raise ValueError("satisfaction must contain both binary classes")

    missing_categorical = set(CATEGORICAL_COLUMNS) - set(train_features)
    if missing_categorical:
        raise ValueError(
            f"train is missing required categorical predictors: "
            f"{sorted(missing_categorical)}"
        )

    ordered_features = tuple(train_features)
    categorical = tuple(
        column for column in CATEGORICAL_COLUMNS if column in ordered_features
    )
    numeric = tuple(column for column in ordered_features if column not in categorical)
    return InputSpec(ordered_features, numeric, categorical)


def target_array(train: pd.DataFrame) -> np.ndarray:
    """Convert the validated boolean or binary target to integer labels."""
    return train[TARGET_COLUMN].astype(np.int8).to_numpy(copy=True)


def make_split(
    train: pd.DataFrame, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible stratified fit and holdout row indices."""
    indices = np.arange(len(train), dtype=np.int64)
    fit_indices, holdout_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    if np.intersect1d(fit_indices, holdout_indices).size:
        raise RuntimeError("split contains overlapping rows")
    if np.unique(np.concatenate((fit_indices, holdout_indices))).size != len(train):
        raise RuntimeError("split does not cover every training row exactly once")
    if np.unique(y[fit_indices]).size != 2 or np.unique(y[holdout_indices]).size != 2:
        raise RuntimeError("split does not preserve both target classes")
    return fit_indices, holdout_indices


def build_logistic_pipeline(spec: InputSpec) -> Pipeline:
    """Build the train-fitted impute, scale, and one-hot logistic pipeline."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(spec.numeric_columns)),
            ("categorical", categorical, list(spec.categorical_columns)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", transformer),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=LOGISTIC_RANDOM_STATE,
                ),
            ),
        ]
    )


def build_hgb_pipeline(
    spec: InputSpec, *, max_iter: int = HGB_MAX_ITER, early_stopping: bool = True
) -> Pipeline:
    """Build the bounded histogram-gradient-boosting pipeline."""
    categorical_mask = np.asarray(
        [False] * len(spec.numeric_columns) + [True] * len(spec.categorical_columns),
        dtype=bool,
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(spec.numeric_columns)),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    dtype=np.float64,
                ),
                list(spec.categorical_columns),
            ),
        ],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=HGB_LEARNING_RATE,
        max_iter=max_iter,
        max_leaf_nodes=HGB_MAX_LEAF_NODES,
        l2_regularization=HGB_L2_REGULARIZATION,
        early_stopping=early_stopping,
        validation_fraction=0.1 if early_stopping else None,
        n_iter_no_change=20,
        categorical_features=categorical_mask,
        random_state=HGB_RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocess", transformer),
            ("model", classifier),
        ]
    )


def probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return validated positive-class probabilities."""
    values = model.predict_proba(X)[:, 1]
    validate_probabilities(values, len(X))
    return values


def validate_probabilities(values: np.ndarray, expected_rows: int) -> None:
    """Reject malformed, non-finite, or out-of-range probabilities."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError("probabilities have the wrong shape")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")


def validate_submission_file(path: Path, test_ids: pd.Series) -> pd.DataFrame:
    """Reload and validate the exact prediction-file contract."""
    submitted = pd.read_csv(path)
    if list(submitted.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("submission must have exactly id,satisfaction columns")
    expected_ids = test_ids.reset_index(drop=True)
    actual_ids = submitted[ID_COLUMN].reset_index(drop=True)
    if len(submitted) != len(expected_ids):
        raise ValueError("submission row count does not match test")
    if actual_ids.duplicated().any() or not np.array_equal(
        actual_ids.to_numpy(), expected_ids.to_numpy()
    ):
        raise ValueError("submission IDs do not exactly match test order")
    validate_probabilities(submitted[TARGET_COLUMN].to_numpy(), len(expected_ids))
    return submitted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    return {
        name: version(name)
        for name in ("joblib", "numpy", "pandas", "scikit-learn", "scipy")
    }


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _model_iterations(model: Pipeline, model_name: str) -> int | None:
    estimator = model.named_steps["model"]
    if model_name == "logistic":
        return int(np.asarray(estimator.n_iter_).max())
    if model_name == "hist_gradient_boosting":
        return int(estimator.n_iter_)
    return None


def _model_settings(model: Pipeline, model_name: str) -> dict[str, Any]:
    """Return JSON-compatible settings read from the fitted estimator."""
    estimator = model.named_steps["model"]
    if model_name == "logistic":
        return {
            "solver": estimator.solver,
            "max_iter": int(estimator.max_iter),
            "class_weight": estimator.class_weight,
            "random_state": estimator.random_state,
        }
    if model_name == "hist_gradient_boosting":
        validation_fraction = estimator.validation_fraction
        categorical_features = estimator.categorical_features
        return {
            "max_iter": int(estimator.max_iter),
            "max_leaf_nodes": int(estimator.max_leaf_nodes),
            "learning_rate": float(estimator.learning_rate),
            "l2_regularization": float(estimator.l2_regularization),
            "early_stopping": bool(estimator.early_stopping),
            "validation_fraction": (
                None if validation_fraction is None else float(validation_fraction)
            ),
            "n_iter_no_change": int(estimator.n_iter_no_change),
            "categorical_features": categorical_features.tolist(),
            "random_state": estimator.random_state,
            "threads": MAX_THREADS,
        }
    raise ValueError(f"unsupported model: {model_name}")


def _fit_and_score(
    model: Pipeline,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    convergence = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence:
        raise RuntimeError(f"{model_name} did not converge")
    predictions = probabilities(model, X_holdout)
    return predictions, {
        "roc_auc": float(roc_auc_score(y_holdout, predictions)),
        "log_loss": float(log_loss(y_holdout, predictions)),
        "fit_seconds": round(fit_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "actual_iterations": _model_iterations(model, model_name),
        "convergence_warnings": len(convergence),
    }


def _slice_metrics(
    y_true: np.ndarray, predictions: np.ndarray, X: pd.DataFrame
) -> dict[str, dict[str, dict[str, Any]]]:
    """Record non-causal validation diagnostics for the main categorical slices."""
    slices: dict[str, dict[str, dict[str, Any]]] = {}
    for column in ("Type of Travel", "Customer Type", "Class"):
        column_slices: dict[str, dict[str, Any]] = {}
        values = X[column].astype("string").fillna("<missing>")
        for value in values.drop_duplicates().tolist():
            mask = values == value
            group_y = y_true[mask.to_numpy()]
            group_predictions = predictions[mask.to_numpy()]
            details: dict[str, Any] = {
                "rows": int(mask.sum()),
                "positive_rate": float(group_y.mean()),
            }
            if np.unique(group_y).size == 2:
                details["roc_auc"] = float(roc_auc_score(group_y, group_predictions))
                details["log_loss"] = float(log_loss(group_y, group_predictions))
            else:
                details["roc_auc"] = None
                details["log_loss"] = None
            column_slices[str(value)] = details
        slices[column] = column_slices
    return slices


def _fit_smoke_models(
    spec: InputSpec, X: pd.DataFrame, y: np.ndarray, fit_indices: np.ndarray
) -> None:
    """Exercise both model paths before starting the full fits."""
    class_indices = [fit_indices[y[fit_indices] == label][:2500] for label in (0, 1)]
    smoke_indices = np.concatenate(class_indices)
    X_smoke = X.iloc[smoke_indices]
    y_smoke = y[smoke_indices]
    for model_name, builder in (
        ("logistic", build_logistic_pipeline),
        ("hist_gradient_boosting", build_hgb_pipeline),
    ):
        model = builder(spec)
        _fit_and_score(model, X_smoke, y_smoke, X_smoke, y_smoke, model_name)


def _save_split(
    path: Path, fit_indices: np.ndarray, holdout_indices: np.ndarray
) -> None:
    np.savez_compressed(
        path,
        fit_indices=fit_indices,
        holdout_indices=holdout_indices,
    )


def run_training(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the bounded local baseline, confirmation, refit, and export path."""
    started = time.perf_counter()
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
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    outer_fit, outer_holdout = make_split(train, y, OUTER_SPLIT_SEED)
    _save_split(artifacts_dir / "outer_split.npz", outer_fit, outer_holdout)
    _fit_smoke_models(spec, X, y, outer_fit)

    outer_results: dict[str, dict[str, Any]] = {
        "constant": {
            "roc_auc": float(
                roc_auc_score(
                    y[outer_holdout], np.full(len(outer_holdout), y[outer_fit].mean())
                )
            ),
            "log_loss": float(
                log_loss(
                    y[outer_holdout], np.full(len(outer_holdout), y[outer_fit].mean())
                )
            ),
            "fit_seconds": 0.0,
            "settings": {"probability": "fit-partition prevalence"},
        }
    }
    actual_iterations: dict[str, list[int]] = {
        "logistic": [],
        "hist_gradient_boosting": [],
    }
    builders: dict[str, Callable[[InputSpec], Pipeline]] = {
        "logistic": build_logistic_pipeline,
        "hist_gradient_boosting": build_hgb_pipeline,
    }
    X_fit = X.iloc[outer_fit]
    y_fit = y[outer_fit]
    X_holdout = X.iloc[outer_holdout]
    y_holdout = y[outer_holdout]
    for model_name in ("logistic", "hist_gradient_boosting"):
        model = builders[model_name](spec)
        predictions, result = _fit_and_score(
            model, X_fit, y_fit, X_holdout, y_holdout, model_name
        )
        result["settings"] = _model_settings(model, model_name)
        result["slices"] = _slice_metrics(y_holdout, predictions, X_holdout)
        outer_results[model_name] = result
        if result["actual_iterations"] is not None:
            actual_iterations[model_name].append(result["actual_iterations"])
        pd.DataFrame(
            {
                ID_COLUMN: train.iloc[outer_holdout][ID_COLUMN].to_numpy(),
                "probability": predictions,
            }
        ).to_csv(artifacts_dir / f"validation_{model_name}.csv", index=False)
        del model

    confirm_fit, confirm_holdout = make_split(train, y, CONFIRMATION_SPLIT_SEED)
    _save_split(artifacts_dir / "confirmation_split.npz", confirm_fit, confirm_holdout)
    confirmation_results: dict[str, dict[str, Any]] = {}
    for model_name in ("logistic", "hist_gradient_boosting"):
        model = builders[model_name](spec)
        _, result = _fit_and_score(
            model,
            X.iloc[confirm_fit],
            y[confirm_fit],
            X.iloc[confirm_holdout],
            y[confirm_holdout],
            model_name,
        )
        result["settings"] = _model_settings(model, model_name)
        confirmation_results[model_name] = result
        if result["actual_iterations"] is not None:
            actual_iterations[model_name].append(result["actual_iterations"])
        del model

    logistic_scores = [
        outer_results["logistic"]["roc_auc"],
        confirmation_results["logistic"]["roc_auc"],
    ]
    hgb_scores = [
        outer_results["hist_gradient_boosting"]["roc_auc"],
        confirmation_results["hist_gradient_boosting"]["roc_auc"],
    ]
    hgb_wins = all(
        hgb >= logistic + HGB_MIN_GAIN
        for hgb, logistic in zip(hgb_scores, logistic_scores, strict=True)
    )
    selected_model = "hist_gradient_boosting" if hgb_wins else "logistic"
    selection_reason = (
        "histogram boosting exceeded logistic ROC AUC by the minimum gain "
        "on both fixed splits"
        if hgb_wins
        else "logistic retained because histogram boosting did not repeat "
        "the minimum gain on both fixed splits"
    )

    if selected_model == "hist_gradient_boosting":
        selected_iterations = int(
            round(float(np.median(actual_iterations[selected_model])))
        )
        final_model = build_hgb_pipeline(
            spec, max_iter=max(1, selected_iterations), early_stopping=False
        )
        final_iteration_policy = {
            "method": "median actual iterations from outer and confirmation fits",
            "selected_iterations": selected_iterations,
            "early_stopping": False,
        }
    else:
        selected_iterations = None
        final_model = build_logistic_pipeline(spec)
        final_iteration_policy = {
            "method": "converged lbfgs fit with fixed max_iter",
            "selected_iterations": None,
            "early_stopping": False,
        }
    final_elapsed_started = time.perf_counter()
    final_fit_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model.fit(X, y)
    final_fit_seconds = time.perf_counter() - final_fit_started
    convergence = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence:
        raise RuntimeError("final model did not converge")
    final_predictions = probabilities(final_model, X_test)
    model_path = artifacts_dir / "final_model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    reloaded_model = joblib.load(model_path)
    reload_rows = min(1000, len(X_test))
    reloaded_predictions = probabilities(reloaded_model, X_test.iloc[:reload_rows])
    reload_difference = float(
        np.max(np.abs(final_predictions[:reload_rows] - reloaded_predictions))
    )
    if not np.allclose(
        final_predictions[:reload_rows], reloaded_predictions, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError("reloaded model predictions differ from the fitted model")
    submission_path = artifacts_dir / "submission.csv"
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: final_predictions}
    ).to_csv(submission_path, index=False)
    submission = validate_submission_file(submission_path, test[ID_COLUMN])

    final_hashes = {
        "train.csv": sha256_file(train_path),
        "test.csv": sha256_file(test_path),
    }
    if final_hashes != input_hashes:
        raise RuntimeError("input CSV fingerprint changed during training")

    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "packages": package_versions(),
        "threads": MAX_THREADS,
        "input_sha256": input_hashes,
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "outer_split": {
            "seed": OUTER_SPLIT_SEED,
            "fit_rows": len(outer_fit),
            "holdout_rows": len(outer_holdout),
        },
        "confirmation_split": {
            "seed": CONFIRMATION_SPLIT_SEED,
            "fit_rows": len(confirm_fit),
            "holdout_rows": len(confirm_holdout),
        },
        "smoke_fit": {
            "completed": True,
            "models": ["logistic", "hist_gradient_boosting"],
        },
        "outer_results": outer_results,
        "confirmation_results": confirmation_results,
        "selection": {
            "model": selected_model,
            "reason": selection_reason,
            "minimum_roc_auc_gain": HGB_MIN_GAIN,
        },
        "final": {
            "model": selected_model,
            "settings": _model_settings(final_model, selected_model),
            "iteration_policy": final_iteration_policy,
            "actual_iterations": _model_iterations(final_model, selected_model),
            "fit_seconds": round(final_fit_seconds, 3),
            "elapsed_seconds": round(time.perf_counter() - final_elapsed_started, 3),
            "reload_check": {
                "rows": reload_rows,
                "max_absolute_difference": reload_difference,
            },
            "model_path": str(model_path),
            "submission_path": str(submission_path),
            "submission_rows": len(submission),
        },
        "total_seconds": round(time.perf_counter() - started, 3),
        "optional_skips": {
            "catboost_lightgbm": (
                "Skipped; initial VPS delivery is complete with the two required "
                "candidates."
            ),
            "feature_engineering": (
                "Skipped; no measured need before the bounded initial delivery."
            ),
            "ensembles": (
                "Skipped; no justified repeatable gain and no ensemble "
                "infrastructure requested."
            ),
            "mps_mac": "Skipped; VPS-only scope and no Mac work authorized.",
        },
    }
    save_json(artifacts_dir / "run_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate the airline satisfaction models."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_training(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {
                "selected_model": metadata["selection"]["model"],
                "outer_results": metadata["outer_results"],
                "confirmation_results": metadata["confirmation_results"],
                "submission_path": metadata["final"]["submission_path"],
                "total_seconds": metadata["total_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
