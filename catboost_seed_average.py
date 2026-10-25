"""Run the fixed two-seed CatBoost average experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
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
from sklearn.metrics import log_loss, roc_auc_score

from blend_experiment import aligned_blend_predictions
from categorical_ratings import _array_fingerprint, _recorded_array_fingerprint
from compare import (
    _load_recorded_split,
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import _artifact_snapshot
from random_forest_experiment import _data_hashes
from slower_learning import build_slow_catboost
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
from weighted_blend_experiment import _read_predictions, weighted_blend_predictions

SPLITS = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
SEEDS = tuple(SPLITS)
INTERNAL_SEED = 42
MODEL_SEED = 2025
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 10_500
EXTERNAL_TIMEOUT_SECONDS = 10_800
OUTPUT_NAME = "catboost_seed_average"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 500
BASELINE_AUC = {42: 0.9577942088132286, 2024: 0.9581967799255812}


class CatBoostSeedAverageTimeout(TimeoutError):
    """Raised when the fixed experiment reaches its bounded runtime."""


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise CatBoostSeedAverageTimeout("CatBoost seed-average experiment timed out")


def build_seed_average_catboost() -> Any:
    """Build slow CatBoost and change only its model random seed."""
    model = build_slow_catboost()
    before = model.get_params()
    model.set_params(random_seed=MODEL_SEED)
    after = model.get_params()
    changed = {
        key: (before.get(key), after.get(key))
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    if changed != {"random_seed": (42, MODEL_SEED)}:
        raise RuntimeError(f"seed-average model changed unexpected settings: {changed}")
    return model


def seed_average_gate(gains: dict[int, float]) -> dict[str, Any]:
    """Apply the predeclared gain gate independently on both fixed holdouts."""
    qualifies = all(gains[seed] >= MINIMUM_AUC_GAIN for seed in SEEDS)
    return {
        "qualifies": qualifies,
        "decision": "select_seed_average" if qualifies else "retain_current_blend",
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "gains_vs_current": {str(seed): gains[seed] for seed in SEEDS},
    }


def fixed_candidate_predictions(
    reference_ids: Any,
    catboost_42_ids: Any,
    catboost_42_values: np.ndarray,
    catboost_2025_ids: Any,
    catboost_2025_values: np.ndarray,
    lightgbm_ids: Any,
    lightgbm_values: np.ndarray,
) -> pd.DataFrame:
    """Align the components and calculate fixed .25/.25/.5 probabilities."""
    averaged = aligned_blend_predictions(
        reference_ids,
        catboost_42_ids,
        catboost_42_values,
        catboost_2025_ids,
        catboost_2025_values,
    )
    candidate = weighted_blend_predictions(
        reference_ids,
        averaged[ID_COLUMN],
        averaged["blend_probability"].to_numpy(),
        lightgbm_ids,
        lightgbm_values,
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    values = candidate["blend_probability"].to_numpy()
    validate_probabilities(values, len(candidate))
    return pd.DataFrame(
        {
            ID_COLUMN: candidate[ID_COLUMN],
            "catboost_seed42_probability": averaged["catboost_probability"],
            "catboost_seed2025_probability": averaged["lightgbm_probability"],
            "lightgbm_probability": candidate["lightgbm_probability"],
            "averaged_catboost_probability": averaged["blend_probability"],
            "candidate_probability": values,
        }
    )


def _dependency_versions() -> dict[str, str]:
    dependencies = package_versions()
    for name in ("catboost", "lightgbm", "threadpoolctl"):
        dependencies[name] = version(name)
    return dependencies


def _source_hashes() -> dict[str, str]:
    names = (
        "catboost_seed_average.py",
        "slower_learning.py",
        "compare.py",
        "blend_experiment.py",
        "weighted_blend_experiment.py",
        "lightgbm_experiment.py",
        "random_forest_experiment.py",
        "categorical_ratings.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _array_record(
    outer_fit: np.ndarray,
    outer_holdout: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
) -> dict[str, Any]:
    arrays = {
        "outer_fit": outer_fit,
        "outer_holdout": outer_holdout,
        "inner_fit": inner_fit,
        "inner_validation": inner_validation,
    }
    return {
        "rows": {name: len(value) for name, value in arrays.items()},
        "sha256": {name: _array_fingerprint(value) for name, value in arrays.items()},
        "combined_sha256": _recorded_array_fingerprint(arrays),
        "inner_covers_outer_fit": np.array_equal(
            np.sort(np.concatenate((inner_fit, inner_validation))),
            np.sort(outer_fit),
        ),
        "inner_disjoint_from_outer_holdout": not bool(
            np.intersect1d(inner_fit, outer_holdout).size
            or np.intersect1d(inner_validation, outer_holdout).size
        ),
    }


def _id_record(
    train: pd.DataFrame, y: np.ndarray, holdout: np.ndarray
) -> dict[str, Any]:
    ids = train.iloc[holdout][ID_COLUMN].to_numpy(dtype=np.int64)
    labels = y[holdout]
    return {
        "rows": len(ids),
        "id_sha256": _array_fingerprint(ids),
        "label_sha256": _array_fingerprint(labels),
        "first_id": int(ids[0]),
        "last_id": int(ids[-1]),
        "positive_labels": int(labels.sum()),
    }


def _settings(model: Any) -> dict[str, Any]:
    params = model.get_params()
    return {
        key: params[key]
        for key in (
            "iterations",
            "depth",
            "learning_rate",
            "loss_function",
            "eval_metric",
            "l2_leaf_reg",
            "random_seed",
            "thread_count",
            "allow_writing_files",
            "early_stopping_rounds",
        )
    }


def _validate_saved_component(
    path: Path, train: pd.DataFrame, holdout: np.ndarray
) -> pd.DataFrame:
    frame = _read_predictions(path)
    expected_rows = np.asarray(holdout, dtype=np.int64)
    expected_ids = train.iloc[holdout][ID_COLUMN].to_numpy()
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError(f"saved prediction rows do not match recorded split: {path}")
    if not np.array_equal(frame[ID_COLUMN].to_numpy(), expected_ids):
        raise ValueError(f"saved prediction IDs do not match recorded split: {path}")
    return frame


def _metrics(y_true: np.ndarray, values: np.ndarray) -> dict[str, float]:
    validate_probabilities(values, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, values)),
        "log_loss": float(log_loss(y_true, values)),
    }


def _save_predictions(
    path: Path, row_index: np.ndarray, ids: np.ndarray, values: np.ndarray
) -> None:
    validate_probabilities(values, len(values))
    pd.DataFrame(
        {"row_index": row_index, ID_COLUMN: ids, "probability": values}
    ).to_csv(path, index=False, float_format="%.17g")


def _progress_logger(
    metadata: dict[str, Any], metadata_path: Path, split_seed: int
) -> Any:
    pattern = re.compile(r"^\s*(\d+):")

    def log(message: str) -> None:
        for line in str(message).splitlines():
            match = pattern.match(line)
            if match is None:
                continue
            tree = int(match.group(1))
            if tree == 0 or tree % PROGRESS_INTERVAL:
                continue
            event = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "tree": tree,
            }
            current = metadata.get("current_fit")
            if isinstance(current, dict):
                current.setdefault("progress_log", []).append(event)
                current["last_progress_tree"] = tree
                save_json(metadata_path, metadata)
            print(
                f"{event['utc']} split_seed={split_seed} catboost_seed={MODEL_SEED} "
                f"tree={tree}",
                flush=True,
            )

    return log


def _fit_seed_2025(
    train: pd.DataFrame,
    y: np.ndarray,
    spec: Any,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    outer_holdout: np.ndarray,
    split_seed: int,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = build_seed_average_catboost()
    X_cat = prepare_catboost_frame(train, spec)
    resources_before = _resource_snapshot()
    fit_started = time.perf_counter()
    current_fit: dict[str, Any] = {
        "split_seed": split_seed,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "outer_holdout_rows": len(outer_holdout),
        "model_seed": MODEL_SEED,
        "progress_log": [],
        "resources_before": resources_before,
    }
    metadata["current_fit"] = current_fit
    save_json(metadata_path, metadata)
    try:
        model.fit(
            X_cat.iloc[inner_fit],
            y[inner_fit],
            eval_set=(X_cat.iloc[inner_validation], y[inner_validation]),
            cat_features=list(spec.categorical_columns),
            verbose=PROGRESS_INTERVAL,
            log_cout=_progress_logger(metadata, metadata_path, split_seed),
        )
        fit_seconds = time.perf_counter() - fit_started
        predictions = model.predict_proba(X_cat.iloc[outer_holdout])[:, 1]
        validate_probabilities(predictions, len(outer_holdout))
        best_zero_based = int(model.get_best_iteration())
        best_one_based = best_zero_based + 1
        resources = _resource_delta(resources_before, _resource_snapshot())
        result = {
            "split_seed": split_seed,
            "candidate": "catboost_seed2025",
            "standalone_catboost": _metrics(y[outer_holdout], predictions),
            "fit_seconds": round(fit_seconds, 3),
            "elapsed_seconds": round(time.perf_counter() - fit_started, 3),
            "best_iteration_zero_based": best_zero_based,
            "best_iteration_1based": best_one_based,
            "actual_tree_count": int(model.tree_count_),
            "settings": _settings(model),
            "resources": resources,
            "progress_log": current_fit["progress_log"],
        }
        current_fit["status"] = "completed"
        current_fit["completed_utc"] = datetime.now(timezone.utc).isoformat()
        current_fit["actual_tree_count"] = int(model.tree_count_)
        return predictions, result
    finally:
        del model, X_cat
        gc.collect()


def _load_prior_records(
    artifacts_dir: Path, input_hashes: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    challenger_path = artifacts_dir / "challengers" / "compare_metadata.json"
    slower_path = artifacts_dir / "slower_learning" / "compare_metadata.json"
    lightgbm_path = artifacts_dir / "lightgbm" / "compare_metadata.json"
    challenger = json.loads(challenger_path.read_text(encoding="utf-8"))
    slower = json.loads(slower_path.read_text(encoding="utf-8"))
    lightgbm = json.loads(lightgbm_path.read_text(encoding="utf-8"))
    competition_hashes = {
        name: input_hashes[name] for name in ("train.csv", "test.csv")
    }
    for name, metadata in (
        ("challengers", challenger),
        ("slower_learning", slower),
        ("lightgbm", lightgbm),
    ):
        recorded = metadata.get("input_sha256") or metadata.get("input_sha256_before")
        if recorded != competition_hashes:
            raise RuntimeError(f"input fingerprints do not match {name} metadata")
    if slower.get("status") != "completed" or lightgbm.get("status") != "completed":
        raise RuntimeError("reused component metadata is not completed")
    return challenger, slower, lightgbm


def _load_inner_arrays(
    prior: dict[str, Any],
    train: pd.DataFrame,
    y: np.ndarray,
    artifacts_dir: Path,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    outer_fit, outer_holdout = _load_recorded_split(
        artifacts_dir / SPLITS[split_seed], train, y, split_seed
    )
    split_name = "outer" if split_seed == 42 else "confirmation"
    recorded = prior["internal_validation_splits"][split_name]
    inner_fit = np.asarray(recorded["fit_indices"], dtype=np.int64)
    inner_validation = np.asarray(recorded["validation_indices"], dtype=np.int64)
    expected_fit, expected_validation = make_catboost_internal_split(
        outer_fit, y, seed=INTERNAL_SEED
    )
    if not np.array_equal(inner_fit, expected_fit) or not np.array_equal(
        inner_validation, expected_validation
    ):
        raise RuntimeError("prior CatBoost inner split does not match seed 42")
    if np.intersect1d(inner_validation, outer_holdout).size:
        raise RuntimeError("inner validation overlaps outer holdout")
    return (
        outer_fit,
        outer_holdout,
        inner_fit,
        inner_validation,
        _array_record(outer_fit, outer_holdout, inner_fit, inner_validation),
    )


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run one fixed seed-2025 CatBoost fit per recorded holdout."""
    started = time.perf_counter()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    protected_before = _artifact_snapshot(artifacts_dir, output_dir)
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "selection": None,
        "validation_results": [],
        "completed_split_seeds": [],
        "current_fit": None,
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
    }
    save_json(metadata_path, metadata)
    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    input_after: dict[str, str] | None = None
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_alarm = signal.getsignal(signal.SIGALRM)
    signals_installed = False
    try:
        input_before = _data_hashes(data_dir)
        metadata.update(
            {
                "python": sys.version,
                "dependencies": _dependency_versions(),
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
                "source_sha256": _source_hashes(),
                "predeclared_config": {
                    "model_seed": MODEL_SEED,
                    "split_seeds": list(SEEDS),
                    "internal_validation_seed": INTERNAL_SEED,
                    "internal_validation_fraction": 0.1,
                    "catboost": _settings(build_slow_catboost()),
                    "blend_weights": {
                        "catboost_seed42": 0.25,
                        "catboost_seed2025": 0.25,
                        "lightgbm": 0.5,
                    },
                    "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
                },
                "reused_artifacts": {},
            }
        )
        save_json(metadata_path, metadata)
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        metadata.update(
            {
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "numeric_nan_policy": "preserve numeric NaN; no imputation",
            }
        )
        prior, _, _ = _load_prior_records(artifacts_dir, input_before)
        prior_paths = {
            "challengers/compare_metadata.json": artifacts_dir
            / "challengers"
            / "compare_metadata.json",
            "slower_learning/compare_metadata.json": artifacts_dir
            / "slower_learning"
            / "compare_metadata.json",
            "lightgbm/compare_metadata.json": artifacts_dir
            / "lightgbm"
            / "compare_metadata.json",
        }
        metadata["prior_metadata_sha256"] = {
            name: sha256_file(path) for name, path in prior_paths.items()
        }
        split_records: dict[str, Any] = {}
        split_arrays: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        components: dict[int, dict[str, pd.DataFrame]] = {}
        for seed in SEEDS:
            outer_fit, outer_holdout, inner_fit, inner_validation, split_record = (
                _load_inner_arrays(prior, train, y, artifacts_dir, seed)
            )
            split_name = "outer" if seed == 42 else "confirmation"
            split_records[split_name] = {
                "seed": seed,
                "path": str(artifacts_dir / SPLITS[seed]),
                "sha256": sha256_file(artifacts_dir / SPLITS[seed]),
                **split_record,
                "holdout_id_provenance": _id_record(train, y, outer_holdout),
            }
            split_arrays[seed] = (outer_fit, outer_holdout, inner_fit, inner_validation)
            cat_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            light_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            cat = _validate_saved_component(cat_path, train, outer_holdout)
            light = _validate_saved_component(light_path, train, outer_holdout)
            components[seed] = {"catboost_seed42": cat, "lightgbm": light}
            metadata["reused_artifacts"].update(
                {
                    str(cat_path.relative_to(artifacts_dir)): sha256_file(cat_path),
                    str(light_path.relative_to(artifacts_dir)): sha256_file(light_path),
                }
            )
        metadata["recorded_splits"] = split_records
        save_json(metadata_path, metadata)

        signal.signal(signal.SIGTERM, _on_timeout)
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        gains: dict[int, float] = {}
        for seed in SEEDS:
            outer_fit, outer_holdout, inner_fit, inner_validation = split_arrays[seed]
            old_cat = components[seed]["catboost_seed42"]
            light = components[seed]["lightgbm"]
            expected_ids = train.iloc[outer_holdout][ID_COLUMN].to_numpy()
            baseline_frame = fixed_candidate_predictions(
                expected_ids,
                old_cat[ID_COLUMN].to_numpy(),
                old_cat["probability"].to_numpy(),
                old_cat[ID_COLUMN].to_numpy(),
                old_cat["probability"].to_numpy(),
                light[ID_COLUMN].to_numpy(),
                light["probability"].to_numpy(),
            )
            baseline_values = (
                0.5 * baseline_frame["averaged_catboost_probability"].to_numpy()
                + 0.5 * baseline_frame["lightgbm_probability"].to_numpy()
            )
            baseline_metrics = _metrics(y[outer_holdout], baseline_values)
            if abs(baseline_metrics["roc_auc"] - BASELINE_AUC[seed]) > 1e-12:
                raise RuntimeError(f"recomputed baseline blend changed for seed {seed}")
            metadata["current_fit"] = {
                "split_seed": seed,
                "status": "preflight_complete",
                "preflight_utc": datetime.now(timezone.utc).isoformat(),
                "fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_holdout_rows": len(outer_holdout),
                "baseline_current_blend_50_50": baseline_metrics,
            }
            save_json(metadata_path, metadata)

            predictions, cat_result = _fit_seed_2025(
                train,
                y,
                spec,
                inner_fit,
                inner_validation,
                outer_holdout,
                seed,
                metadata,
                metadata_path,
            )
            cat_path = output_dir / f"validation_catboost_seed2025_seed{seed}.csv"
            _save_predictions(cat_path, outer_holdout, expected_ids, predictions)
            original_cat_path = (
                output_dir / f"validation_catboost_seed42_seed{seed}.csv"
            )
            light_path = output_dir / f"validation_lightgbm_seed{seed}.csv"
            shutil.copyfile(
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv",
                original_cat_path,
            )
            shutil.copyfile(
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv",
                light_path,
            )
            candidate = fixed_candidate_predictions(
                expected_ids,
                components[seed]["catboost_seed42"][ID_COLUMN].to_numpy(),
                components[seed]["catboost_seed42"]["probability"].to_numpy(),
                expected_ids,
                predictions,
                components[seed]["lightgbm"][ID_COLUMN].to_numpy(),
                components[seed]["lightgbm"]["probability"].to_numpy(),
            )
            averaged_path = output_dir / f"validation_catboost_average_seed{seed}.csv"
            candidate_path = output_dir / f"validation_candidate_blend_seed{seed}.csv"
            _save_predictions(
                averaged_path,
                outer_holdout,
                expected_ids,
                candidate["averaged_catboost_probability"].to_numpy(),
            )
            _save_predictions(
                candidate_path,
                outer_holdout,
                expected_ids,
                candidate["candidate_probability"].to_numpy(),
            )
            averaged_metrics = _metrics(
                y[outer_holdout], candidate["averaged_catboost_probability"].to_numpy()
            )
            candidate_metrics = _metrics(
                y[outer_holdout], candidate["candidate_probability"].to_numpy()
            )
            cat_result["averaged_catboost"] = averaged_metrics
            cat_result["candidate_blend"] = candidate_metrics
            cat_result["current_blend_50_50"] = baseline_metrics
            cat_result["gain_vs_current"] = (
                candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"]
            )
            cat_result["standalone_catboost_seed2025"] = cat_result.pop(
                "standalone_catboost"
            )
            cat_result["prediction_paths"] = {
                "catboost_seed42": str(original_cat_path),
                "catboost_seed2025": str(cat_path),
                "lightgbm": str(light_path),
                "averaged_catboost": str(averaged_path),
                "candidate_blend": str(candidate_path),
            }
            metadata["validation_results"].append(cat_result)
            metadata["completed_split_seeds"].append(seed)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            gains[seed] = cat_result["gain_vs_current"]
            gc.collect()
        metadata["selection"] = seed_average_gate(gains)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, CatBoostSeedAverageTimeout) else "failed"
        )
        metadata["selection"] = None
        metadata["error"] = _error_record(error)
    finally:
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGALRM, previous_alarm)
            except BaseException as error:
                finalization_errors.append(error)
        try:
            input_after = _data_hashes(data_dir)
            metadata["input_sha256_after"] = input_after
            if input_before is not None and input_after != input_before:
                finalization_errors.append(
                    RuntimeError("input CSV fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        try:
            protected_after = _artifact_snapshot(artifacts_dir, output_dir)
            metadata["protected_artifact_sha256_before"] = protected_before
            metadata["protected_artifact_sha256_after"] = protected_after
            metadata["protected_artifacts_unchanged"] = (
                protected_after == protected_before
            )
            if protected_after != protected_before:
                finalization_errors.append(
                    RuntimeError("pre-existing artifact fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        if finalization_errors:
            finalization_error = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata["status"] = "failed"
            metadata["integrity_error"] = _error_record(finalization_error)
            if caught_error is None:
                caught_error = finalization_error
                metadata["error"] = _error_record(finalization_error)
            else:
                metadata["finalization_error"] = _error_record(finalization_error)
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        try:
            save_json(
                output_dir / "protected_baseline_sha256.json",
                metadata.get("protected_artifact_sha256_after", protected_before),
            )
            save_json(metadata_path, metadata)
            metadata["output_bytes"] = sum(
                path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
            )
            if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
                budget_error = RuntimeError("output directory exceeded 100 MiB budget")
                metadata["status"] = "failed"
                metadata["integrity_error"] = _error_record(budget_error)
                if caught_error is None:
                    caught_error = budget_error
                    metadata["error"] = _error_record(budget_error)
                else:
                    metadata["finalization_error"] = _error_record(budget_error)
            save_json(metadata_path, metadata)
        except BaseException as error:
            finalization_errors.append(error)
            if caught_error is None:
                caught_error = error
                metadata["status"] = "failed"
                metadata["error"] = _error_record(error)
    if caught_error is not None:
        raise caught_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed two-seed CatBoost average experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "selection": metadata["selection"],
                "validation_results": metadata["validation_results"],
                "artifacts_dir": metadata["output_dir"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
