from __future__ import annotations

import argparse
import gc
import json
import os
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

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from compare import (
    _load_recorded_split,
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
)
from lightgbm_experiment import _artifact_snapshot
from train import (
    ID_COLUMN,
    MAX_THREADS,
    InputSpec,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)
from weighted_blend_experiment import _read_predictions, weighted_blend_predictions

SPLIT_SEEDS = (42, 2024)
INTERNAL_SEED = 42
RF_TIMEOUT_SECONDS = 10500
EXTERNAL_TIMEOUT_SECONDS = 10800
RF_CONFIG: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 14,
    "max_leaf_nodes": 1024,
    "min_samples_leaf": 30,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": 0.7,
    "random_state": 42,
    "n_jobs": MAX_THREADS,
    "class_weight": None,
}
RF_BLEND_WEIGHT = 0.10
CURRENT_BLEND_WEIGHT = 0.90
MINIMUM_AUC_GAIN = 0.0002
MAX_NODE_COUNT_BOUND = RF_CONFIG["n_estimators"] * (2 * RF_CONFIG["max_leaf_nodes"] - 1)
OUTPUT_NAME = "random_forest"


class RandomForestExperimentTimeout(TimeoutError):
    """Raised when the fixed experiment reaches its internal deadline."""


def build_random_forest() -> RandomForestClassifier:
    """Build the one predeclared memory-bounded forest configuration."""
    return RandomForestClassifier(**RF_CONFIG)


def build_preprocessor(spec: InputSpec) -> ColumnTransformer:
    """Build fit-only median and dense float32 one-hot preprocessing."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", copy=False))]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(spec.numeric_columns)),
            ("categorical", categorical, list(spec.categorical_columns)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def dense_float32(values: Any) -> np.ndarray:
    """Return a contiguous dense float32 matrix for low-memory forest fitting."""
    result = np.asarray(values, dtype=np.float32, order="C")
    if result.ndim != 2:
        raise ValueError("preprocessing output must be a two-dimensional matrix")
    return result


def random_forest_blend_predictions(
    reference_ids: pd.Index | pd.Series | np.ndarray,
    rf_ids: pd.Index | pd.Series | np.ndarray,
    rf_values: np.ndarray,
    current_values: np.ndarray,
) -> np.ndarray:
    """Add 10% RF probability to the fixed 90% current blend by ID."""
    reference = pd.Index(reference_ids)
    index = pd.Index(rf_ids)
    rf_array = np.asarray(rf_values, dtype=np.float64)
    current_array = np.asarray(current_values, dtype=np.float64)
    if not reference.is_unique or not index.is_unique:
        raise ValueError("blend IDs must be unique")
    if len(reference) != len(current_array) or len(index) != len(rf_array):
        raise ValueError("blend IDs must match prediction lengths")
    if not reference.difference(index).empty or not index.difference(reference).empty:
        raise ValueError("RF IDs do not match current blend IDs")
    validate_probabilities(rf_array, len(index))
    validate_probabilities(current_array, len(reference))
    aligned_rf = pd.Series(rf_array, index=index).reindex(reference).to_numpy()
    validate_probabilities(aligned_rf, len(reference))
    result = CURRENT_BLEND_WEIGHT * current_array + RF_BLEND_WEIGHT * aligned_rf
    validate_probabilities(result, len(reference))
    return result


def random_forest_gate(gains: dict[int, float]) -> dict[str, Any]:
    """Apply the predeclared gain gate on both fixed holdouts."""
    qualifies = all(gains[seed] >= MINIMUM_AUC_GAIN for seed in SPLIT_SEEDS)
    return {
        "qualifies": qualifies,
        "decision": "select_rf_blend" if qualifies else "retain_current_blend",
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _data_hashes(data_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(data_dir)): sha256_file(path)
        for path in sorted(data_dir.rglob("*.csv"))
    }


def _versions() -> dict[str, str]:
    result = package_versions()
    result["scikit-learn"] = version("scikit-learn")
    return result


def _source_hashes() -> dict[str, str]:
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "random_forest_experiment.py",
            "weighted_blend_experiment.py",
            "train.py",
            "compare.py",
        )
    }


def _split_record(
    outer_fit: np.ndarray,
    outer_holdout: np.ndarray,
    internal_fit: np.ndarray,
    internal_validation: np.ndarray,
) -> dict[str, Any]:
    def array_hash(values: np.ndarray) -> str:
        return (
            __import__("hashlib")
            .sha256(np.asarray(values, dtype=np.int64).tobytes())
            .hexdigest()
        )

    return {
        "outer_fit_rows": len(outer_fit),
        "outer_holdout_rows": len(outer_holdout),
        "internal_fit_rows": len(internal_fit),
        "internal_validation_rows": len(internal_validation),
        "internal_fit_sha256": array_hash(internal_fit),
        "internal_validation_sha256": array_hash(internal_validation),
        "internal_fit_disjoint_from_outer_holdout": not bool(
            np.intersect1d(internal_fit, outer_holdout).size
        ),
        "internal_validation_disjoint_from_outer_holdout": not bool(
            np.intersect1d(internal_validation, outer_holdout).size
        ),
    }


def _metrics(y_true: np.ndarray, values: np.ndarray) -> dict[str, float]:
    validate_probabilities(values, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, values)),
        "log_loss": float(log_loss(y_true, values)),
    }


def _fit_and_score(
    train: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    spec: InputSpec,
    internal_fit: np.ndarray,
    internal_validation: np.ndarray,
    outer_holdout: np.ndarray,
    current_blend: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resources_before = _resource_snapshot()
    started = time.perf_counter()
    preprocessor = build_preprocessor(spec)
    preprocess_started = time.perf_counter()
    X_fit = dense_float32(preprocessor.fit_transform(X.iloc[internal_fit]))
    X_holdout = dense_float32(preprocessor.transform(X.iloc[outer_holdout]))
    preprocess_seconds = time.perf_counter() - preprocess_started
    model = build_random_forest()
    forest_started = time.perf_counter()
    model.fit(X_fit, y[internal_fit])
    forest_seconds = time.perf_counter() - forest_started
    rf_predictions = model.predict_proba(X_holdout)[:, 1]
    prediction_seconds = time.perf_counter() - forest_started - forest_seconds
    rf_metrics = _metrics(y[outer_holdout], rf_predictions)
    current_values = current_blend["blend_probability"].to_numpy()
    candidate_values = random_forest_blend_predictions(
        current_blend[ID_COLUMN],
        train.iloc[outer_holdout][ID_COLUMN].to_numpy(),
        rf_predictions,
        current_values,
    )
    candidate_metrics = _metrics(y[outer_holdout], candidate_values)
    trees = [estimator.tree_ for estimator in model.estimators_]
    node_counts = np.asarray([tree.node_count for tree in trees], dtype=np.int64)
    result = {
        "split": _split_record(
            np.concatenate((internal_fit, internal_validation)),
            outer_holdout,
            internal_fit,
            internal_validation,
        ),
        "fit_rows": len(internal_fit),
        "outer_holdout_rows": len(outer_holdout),
        "preprocessed_features": int(X_fit.shape[1]),
        "preprocessing": {
            "fit_partition": "competition inner fit only",
            "numeric": "median imputation",
            "categorical": "most_frequent imputation plus one-hot",
            "handle_unknown": "ignore",
            "output": "dense float32",
            "outer_holdout_used_for_fit": False,
        },
        "standalone_random_forest": {
            **rf_metrics,
            "fit_seconds": round(forest_seconds, 3),
            "prediction_seconds": round(prediction_seconds, 3),
        },
        "current_blend_50_50": _metrics(y[outer_holdout], current_values),
        "rf_blend_90_current_10_rf": {
            **candidate_metrics,
            "gain_vs_current": candidate_metrics["roc_auc"]
            - _metrics(y[outer_holdout], current_values)["roc_auc"],
        },
        "timing": {
            "preprocessing_seconds": round(preprocess_seconds, 3),
            "forest_seconds": round(forest_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "forest_structure": {
            "tree_count": len(model.estimators_),
            "max_node_count_observed": int(node_counts.max()),
            "total_node_count_observed": int(node_counts.sum()),
            "max_node_count_bound": MAX_NODE_COUNT_BOUND,
            "max_features": model.max_features,
        },
        "resources": _resource_delta(resources_before, _resource_snapshot()),
    }
    output = pd.DataFrame(
        {
            "row_index": outer_holdout,
            ID_COLUMN: train.iloc[outer_holdout][ID_COLUMN].to_numpy(),
            "random_forest_probability": rf_predictions,
            "current_blend_probability": current_values,
            "rf_blend_probability": candidate_values,
        }
    )
    del model, preprocessor, X_fit, X_holdout, rf_predictions, candidate_values
    gc.collect()
    return output, result


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the one fixed RF configuration on both recorded holdouts."""
    started = time.perf_counter()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "compare_metadata.json"
    train_path = data_dir / "train.csv"
    input_before = _data_hashes(data_dir)
    protected_before = _artifact_snapshot(artifacts_dir, output_dir)
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "internal_timeout_seconds": RF_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "threads": MAX_THREADS,
        "python": sys.version,
        "dependencies": _versions(),
        "source_sha256": _source_hashes(),
        "rf_config": RF_CONFIG,
        "rf_blend": {
            "current_blend_weight": CURRENT_BLEND_WEIGHT,
            "random_forest_weight": RF_BLEND_WEIGHT,
        },
        "max_node_count_bound": MAX_NODE_COUNT_BOUND,
        "input_sha256_before": input_before,
        "protected_artifact_sha256_before": protected_before,
        "validation_results": [],
        "completed_split_seeds": [],
        "current_fit": None,
        "selection": None,
        "output_dir": str(output_dir),
    }
    save_json(metadata_path, metadata)
    caught_error: BaseException | None = None
    prior_term_handler = signal.getsignal(signal.SIGTERM)
    prior_alarm_handler = signal.getsignal(signal.SIGALRM)

    def stop_experiment(_signum: int, _frame: Any) -> None:
        raise RandomForestExperimentTimeout("RandomForest experiment timed out")

    try:
        signal.signal(signal.SIGTERM, stop_experiment)
        signal.signal(signal.SIGALRM, stop_experiment)
        signal.setitimer(signal.ITIMER_REAL, RF_TIMEOUT_SECONDS)
        train = pd.read_csv(train_path)
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        X = train.loc[:, list(spec.feature_columns)]
        metadata["feature_columns"] = list(spec.feature_columns)
        metadata["numeric_columns"] = list(spec.numeric_columns)
        metadata["categorical_columns"] = list(spec.categorical_columns)
        save_json(metadata_path, metadata)
        gains: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            outer_fit, outer_holdout = _load_recorded_split(
                artifacts_dir
                / ("outer_split.npz" if seed == 42 else "confirmation_split.npz"),
                train,
                y,
                seed,
            )
            internal_fit, internal_validation = make_catboost_internal_split(
                outer_fit, y, seed=INTERNAL_SEED
            )
            slow_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            light_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            slow = _read_predictions(slow_path)
            light = _read_predictions(light_path)
            current_blend = weighted_blend_predictions(
                slow[ID_COLUMN],
                slow[ID_COLUMN],
                slow["probability"].to_numpy(),
                light[ID_COLUMN],
                light["probability"].to_numpy(),
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )
            expected_ids = pd.Index(train.iloc[outer_holdout][ID_COLUMN])
            if not expected_ids.equals(pd.Index(current_blend[ID_COLUMN])):
                raise ValueError(f"saved predictions do not match seed {seed} holdout")
            preflight = _metrics(
                y[outer_holdout], current_blend["blend_probability"].to_numpy()
            )
            metadata["current_fit"] = {
                "split_seed": seed,
                "status": "running",
                "fit_rows": len(internal_fit),
                "outer_holdout_rows": len(outer_holdout),
                "preflight_current_blend_50_50": preflight,
                "split": _split_record(
                    outer_fit, outer_holdout, internal_fit, internal_validation
                ),
            }
            save_json(metadata_path, metadata)
            output, result = _fit_and_score(
                train,
                X,
                y,
                spec,
                internal_fit,
                internal_validation,
                outer_holdout,
                current_blend,
            )
            validation_path = output_dir / f"validation_random_forest_seed{seed}.csv"
            output.to_csv(validation_path, index=False, float_format="%.17g")
            result["split_seed"] = seed
            result["preflight_current_blend_50_50"] = preflight
            result["validation_path"] = str(validation_path)
            result["validation_sha256"] = sha256_file(validation_path)
            metadata["validation_results"].append(result)
            metadata["completed_split_seeds"].append(seed)
            gains[seed] = result["rf_blend_90_current_10_rf"]["gain_vs_current"]
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
        metadata["selection"] = random_forest_gate(gains)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, RandomForestExperimentTimeout) else "failed"
        )
        metadata["selection"] = None
        metadata["error"] = _error_record(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, prior_term_handler)
        signal.signal(signal.SIGALRM, prior_alarm_handler)
        input_after = _data_hashes(data_dir)
        protected_after = _artifact_snapshot(artifacts_dir, output_dir)
        metadata["input_sha256_after"] = input_after
        metadata["protected_artifact_sha256_after"] = protected_after
        metadata["protected_artifacts_unchanged"] = protected_before == protected_after
        integrity_failures = []
        if input_after != input_before:
            integrity_failures.append("input CSV fingerprint changed")
        if not metadata["protected_artifacts_unchanged"]:
            integrity_failures.append("pre-existing artifact changed")
        if integrity_failures:
            integrity_error = RuntimeError("; ".join(integrity_failures))
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["integrity_error"] = _error_record(integrity_error)
            if caught_error is None:
                caught_error = integrity_error
                metadata["error"] = _error_record(integrity_error)
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        save_json(output_dir / "protected_baseline_sha256.json", protected_after)
        save_json(metadata_path, metadata)
    if caught_error is not None:
        raise caught_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed memory-bounded RandomForest experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
