"""Run the fixed two-seed CPU XGBoost diagnostic."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
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
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

from compare import _resource_delta, _resource_snapshot
from lightgbm_capacity import (
    _artifact_snapshot,
    _input_hashes,
    _load_components,
    _load_split_arrays,
)
from lightgbm_experiment import fit_category_levels, prepare_lightgbm_frame
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

SPLIT_SEEDS = (42, 2024)
SPLIT_NAMES = {42: "outer", 2024: "confirmation"}
INTERNAL_SEED = 42
INNER_FIT_ROWS = 503_737
INNER_VALIDATION_ROWS = 55_971
OUTER_HOLDOUT_ROWS = 139_927
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_NAME = "xgboost_cpu"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 250
MAX_PROGRESS_EVENTS = 12
BASELINE_AUC = {42: 0.9577942088132286, 2024: 0.9581967799255812}
PREDECLARED_CONFIG: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cpu",
    "enable_categorical": True,
    "n_estimators": 1500,
    "learning_rate": 0.04,
    "max_depth": 6,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "max_bin": 256,
    "grow_policy": "depthwise",
    "random_state": 42,
    "n_jobs": MAX_THREADS,
    "early_stopping_rounds": 100,
}


class XGBoostExperimentTimeout(TimeoutError):
    """Raised when the fixed diagnostic reaches its internal deadline."""


def build_xgboost_classifier(**overrides: Any) -> xgb.XGBClassifier:
    """Build the fixed classifier; overrides are only for synthetic tests."""
    settings = dict(PREDECLARED_CONFIG)
    settings.update(overrides)
    return xgb.XGBClassifier(**settings)


def xgboost_gate(gains: dict[int, float]) -> dict[str, Any]:
    """Require the fixed AUC gain independently on both recorded holdouts."""
    qualifies = all(
        gains.get(seed, float("-inf")) >= MINIMUM_AUC_GAIN for seed in SPLIT_SEEDS
    )
    return {
        "qualifies": qualifies,
        "decision": "select_xgboost_blend" if qualifies else "retain_current_blend",
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "gains_vs_current": {str(seed): gains.get(seed) for seed in SPLIT_SEEDS},
    }


def fixed_xgboost_blend_predictions(
    reference_ids: Any,
    slowcat_ids: Any,
    slowcat_values: np.ndarray,
    lightgbm_ids: Any,
    lightgbm_values: np.ndarray,
    xgboost_ids: Any,
    xgboost_values: np.ndarray,
) -> pd.DataFrame:
    """Align saved components and calculate .4 CatBoost + .4 LightGBM + .2 XGBoost."""
    current = weighted_blend_predictions(
        reference_ids,
        slowcat_ids,
        slowcat_values,
        lightgbm_ids,
        lightgbm_values,
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    candidate = weighted_blend_predictions(
        current[ID_COLUMN],
        current[ID_COLUMN],
        current["blend_probability"].to_numpy(),
        xgboost_ids,
        xgboost_values,
        slow_catboost_weight=0.8,
        lightgbm_weight=0.2,
    )
    return pd.DataFrame(
        {
            ID_COLUMN: candidate[ID_COLUMN],
            "xgboost_probability": candidate["lightgbm_probability"],
            "baseline_probability": current["blend_probability"],
            "candidate_probability": candidate["blend_probability"],
        }
    )


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    validate_probabilities(probabilities, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def _source_hashes() -> dict[str, str]:
    names = (
        "xgboost_experiment.py",
        "lightgbm_capacity.py",
        "lightgbm_experiment.py",
        "weighted_blend_experiment.py",
        "compare.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _package_versions() -> dict[str, str]:
    packages = package_versions()
    packages.update(
        {"lightgbm": version("lightgbm"), "xgboost-cpu": version("xgboost-cpu")}
    )
    return packages


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


class _ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, metadata: dict[str, Any], path: Path, split_seed: int) -> None:
        self.metadata, self.path, self.split_seed = metadata, path, split_seed

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        del model, evals_log
        tree = epoch + 1
        if tree % PROGRESS_INTERVAL:
            return False
        current = self.metadata.get("current_fit")
        if isinstance(current, dict):
            progress = current.setdefault("progress_log", [])
            if len(progress) < MAX_PROGRESS_EVENTS:
                progress.append(
                    {"utc": datetime.now(timezone.utc).isoformat(), "tree": tree}
                )
            current["last_progress_tree"] = tree
            save_json(self.path, self.metadata)
        print(
            f"{datetime.now(timezone.utc).isoformat()} "
            f"split_seed={self.split_seed} tree={tree}",
            flush=True,
        )
        return False


def _output_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly two sequential CPU XGBoost fits and save no models."""
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
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "predeclared_config": dict(PREDECLARED_CONFIG),
        "config_sha256": hashlib.sha256(
            json.dumps(PREDECLARED_CONFIG, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "validation_results": [],
        "completed_split_seeds": [],
        "current_fit": None,
        "selection": None,
    }
    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    timer_active = False
    previous_alarm, previous_term = (
        signal.getsignal(signal.SIGALRM),
        signal.getsignal(signal.SIGTERM),
    )

    def on_timeout(_signum: int, _frame: Any) -> None:
        raise XGBoostExperimentTimeout("XGBoost diagnostic timed out")

    try:
        save_json(metadata_path, metadata)
        signal.signal(signal.SIGALRM, on_timeout)
        signal.signal(signal.SIGTERM, on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        timer_active = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        source_before = _source_hashes()
        metadata.update(
            {
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
                "input_sha256_before": input_before,
                "source_sha256": source_before,
                "protected_artifact_sha256_before": protected_before,
                "split_seeds": list(SPLIT_SEEDS),
                "model_seed": PREDECLARED_CONFIG["random_state"],
                "internal_validation_seed": INTERNAL_SEED,
                "fold_inner_array_provenance": {},
            }
        )
        save_json(metadata_path, metadata)
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        if len(spec.feature_columns) != 21 or len(spec.categorical_columns) != 4:
            raise RuntimeError(
                "XGBoost diagnostic requires the original 21-feature input"
            )
        y = target_array(train)
        X = train.loc[:, list(spec.feature_columns)]
        prior_path = artifacts_dir / "lightgbm" / "compare_metadata.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") != "completed"
            or prior.get("predeclared_config", {}).get("num_leaves") != 31
        ):
            raise RuntimeError(
                "prior metadata is not the completed original LightGBM-31 run"
            )
        if prior.get("input_sha256_before") != {
            name: input_before[name] for name in ("train.csv", "test.csv")
        }:
            raise RuntimeError(
                "input fingerprints do not match original LightGBM metadata"
            )
        splits: dict[
            int,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                pd.DataFrame,
                pd.DataFrame,
            ],
        ] = {}
        for seed in SPLIT_SEEDS:
            outer_fit, outer_holdout, inner_fit, inner_validation, record = (
                _load_split_arrays(artifacts_dir, train, y, prior, seed)
            )
            if (len(inner_fit), len(inner_validation), len(outer_holdout)) != (
                INNER_FIT_ROWS,
                INNER_VALIDATION_ROWS,
                OUTER_HOLDOUT_ROWS,
            ):
                raise RuntimeError(
                    "recorded split row counts do not match the fixed diagnostic"
                )
            paths = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv",
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv",
            )
            slow, light = (
                _load_components(path, train, outer_holdout) for path in paths
            )
            splits[seed] = (
                outer_fit,
                outer_holdout,
                inner_fit,
                inner_validation,
                slow,
                light,
            )
            metadata["fold_inner_array_provenance"][SPLIT_NAMES[seed]] = record
            metadata.setdefault("reused_artifacts", {}).update(
                {
                    str(path.relative_to(artifacts_dir)): sha256_file(path)
                    for path in paths
                }
            )
        metadata.update(
            {
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
            }
        )
        save_json(metadata_path, metadata)

        gains: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            _outer_fit, outer_holdout, inner_fit, inner_validation, slow, light = (
                splits[seed]
            )
            reference_ids = train.iloc[outer_holdout][ID_COLUMN].to_numpy()
            current = weighted_blend_predictions(
                reference_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                light[ID_COLUMN].to_numpy(),
                light["probability"].to_numpy(),
                slow_catboost_weight=0.5,
                lightgbm_weight=0.5,
            )
            baseline = _metrics(
                y[outer_holdout], current["blend_probability"].to_numpy()
            )
            if abs(baseline["roc_auc"] - BASELINE_AUC[seed]) > 1e-12:
                raise RuntimeError(f"recomputed baseline blend changed for seed {seed}")
            metadata["current_fit"] = {
                "split_seed": seed,
                "status": "running",
                "start_utc": datetime.now(timezone.utc).isoformat(),
                "fit_rows": len(inner_fit),
                "eval_rows": len(inner_validation),
                "score_rows": len(outer_holdout),
                "outer_holdout_used_for_eval": False,
                "category_levels_fit_indices_only": True,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)
            levels = fit_category_levels(X.iloc[inner_fit], spec)
            model = build_xgboost_classifier(
                callbacks=[_ProgressCallback(metadata, metadata_path, seed)]
            )
            before_resources = _resource_snapshot()
            fit_started = time.perf_counter()
            model.fit(
                prepare_lightgbm_frame(X.iloc[inner_fit], spec, levels),
                y[inner_fit],
                eval_set=[
                    (
                        prepare_lightgbm_frame(X.iloc[inner_validation], spec, levels),
                        y[inner_validation],
                    )
                ],
                verbose=False,
            )
            fit_seconds = time.perf_counter() - fit_started
            if set(model.evals_result()["validation_0"]) != {"auc"}:
                raise RuntimeError(
                    "XGBoost diagnostic registered a metric other than auc"
                )
            predictions = model.predict_proba(
                prepare_lightgbm_frame(X.iloc[outer_holdout], spec, levels)
            )[:, 1]
            validate_probabilities(predictions, len(outer_holdout))
            blended = fixed_xgboost_blend_predictions(
                reference_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                light[ID_COLUMN].to_numpy(),
                light["probability"].to_numpy(),
                reference_ids,
                predictions,
            )
            standalone = _metrics(y[outer_holdout], predictions)
            candidate = _metrics(
                y[outer_holdout], blended["candidate_probability"].to_numpy()
            )
            gain = candidate["roc_auc"] - baseline["roc_auc"]
            validation_path = output_dir / f"validation_xgboost_seed{seed}.csv"
            pd.DataFrame(
                {
                    "row_index": outer_holdout,
                    ID_COLUMN: blended[ID_COLUMN],
                    "xgboost_probability": blended["xgboost_probability"],
                    "baseline_probability": blended["baseline_probability"],
                    "candidate_probability": blended["candidate_probability"],
                }
            ).to_csv(validation_path, index=False, float_format="%.17g")
            best_zero = int(model.best_iteration)
            actual_rounds = int(model.get_booster().num_boosted_rounds())
            result = {
                "split_seed": seed,
                "standalone": {**standalone, "fit_seconds": round(fit_seconds, 3)},
                "candidate": {**candidate, "fit_seconds": round(fit_seconds, 3)},
                "current_baseline": baseline,
                "gain_vs_current": gain,
                "fit_seconds": round(fit_seconds, 3),
                "best_iteration_index_zero_based": best_zero,
                "best_iteration_count_one_based": best_zero + 1,
                "actual_boosted_rounds": actual_rounds,
                "params": dict(PREDECLARED_CONFIG),
                "validation_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "inner_validation": {
                    "seed": INTERNAL_SEED,
                    "fit_rows": len(inner_fit),
                    "validation_rows": len(inner_validation),
                    "outer_holdout_used": False,
                    "category_levels_fit_indices_only": True,
                },
            }
            result["resources"] = _resource_delta(
                before_resources, _resource_snapshot()
            )
            metadata["validation_results"].append(result)
            metadata["completed_split_seeds"].append(seed)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            gains[seed] = gain
            del model, levels, predictions, blended
        metadata["selection"] = xgboost_gate(gains)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata.update(
            {
                "status": "timeout"
                if isinstance(error, XGBoostExperimentTimeout)
                else "failed",
                "error": _error_record(error),
                "selection": None,
                "current_fit": None,
            }
        )
    finally:
        if timer_active:
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
        if source_before is not None:
            try:
                source_after = _source_hashes()
                metadata["source_sha256_after"] = source_after
                if source_after != source_before:
                    finalization_errors.append(
                        RuntimeError("source fingerprint changed")
                    )
            except BaseException as error:
                finalization_errors.append(error)
        if finalization_errors:
            finalization_error = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata.update(
                {
                    "status": "failed",
                    "selection": None,
                    "current_fit": None,
                    "integrity_error": _error_record(finalization_error),
                }
            )
            if caught_error is None:
                caught_error = finalization_error
                metadata["error"] = _error_record(finalization_error)
            else:
                metadata["finalization_error"] = _error_record(finalization_error)
        metadata["current_fit"] = None
        if metadata["status"] != "completed":
            metadata["selection"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        for _ in range(6):
            save_json(metadata_path, metadata)
            actual_bytes = _output_bytes(output_dir)
            if actual_bytes > OUTPUT_BUDGET_BYTES:
                budget_error = RuntimeError("output exceeded 100 MiB")
                metadata.update(
                    {
                        "status": "failed",
                        "selection": None,
                        "output_bytes": actual_bytes,
                    }
                )
                if caught_error is None:
                    caught_error = budget_error
                    metadata["error"] = _error_record(budget_error)
                else:
                    metadata["finalization_error"] = _error_record(budget_error)
                continue
            if metadata.get("output_bytes") == actual_bytes:
                break
            metadata["output_bytes"] = actual_bytes
    if caught_error is not None:
        raise caught_error
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
