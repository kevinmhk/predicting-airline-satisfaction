"""Run the fixed five-fold CatBoost/LightGBM validation comparison."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from functools import partial
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from categorical_ratings import _array_fingerprint
from compare import (
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _artifact_snapshot,
    fit_category_levels,
    prepare_lightgbm_frame,
)
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

FOLD_COUNT = 5
FOLD_SEED = 2026
INTERNAL_SEED = 42
MODEL_SEEDS = (42, 2025)
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 21_300
EXTERNAL_TIMEOUT_SECONDS = 21_600
OUTPUT_NAME = "five_fold_cv"
OUTPUT_BUDGET_BYTES = 150 * 1024 * 1024
PROGRESS_INTERVAL = 500


class FiveFoldCVTimeout(TimeoutError):
    """Raised when the fixed five-fold run reaches its internal deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise FiveFoldCVTimeout("five-fold CV exceeded its six-hour budget")


def make_five_fold_splits(y: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the one frozen stratified five-fold membership definition."""
    indices = np.arange(len(y), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=FOLD_COUNT, shuffle=True, random_state=FOLD_SEED
    )
    folds = tuple(
        (np.asarray(fit, dtype=np.int64), np.asarray(holdout, dtype=np.int64))
        for fit, holdout in splitter.split(indices, y)
    )
    holdout = np.concatenate([fold[1] for fold in folds])
    if (
        len(folds) != FOLD_COUNT
        or np.unique(holdout).size != len(y)
        or not np.array_equal(np.sort(holdout), indices)
    ):
        raise RuntimeError("five-fold holdouts do not cover each row exactly once")
    for fit, validation in folds:
        if np.intersect1d(fit, validation).size:
            raise RuntimeError("fold fit and holdout overlap")
    return folds


def assign_oof(
    row_count: int, records: list[tuple[int, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one validated OOF probability vector and its fold IDs."""
    values = np.full(row_count, np.nan, dtype=np.float64)
    fold_ids = np.full(row_count, -1, dtype=np.int8)
    for fold_id, rows, probabilities in records:
        rows = np.asarray(rows, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if len(rows) != len(probabilities):
            raise ValueError("OOF row and probability lengths do not match")
        if (
            np.any(rows < 0)
            or np.any(rows >= row_count)
            or np.unique(rows).size != len(rows)
        ):
            raise ValueError("OOF component contains an invalid or duplicate row index")
        if np.any(fold_ids[rows] != -1):
            raise ValueError("OOF component assigns a row more than once")
        validate_probabilities(probabilities, len(rows))
        values[rows] = probabilities
        fold_ids[rows] = fold_id
    if np.any(fold_ids == -1):
        raise ValueError("OOF component is missing one or more rows")
    return values, fold_ids


def fixed_blends(
    catboost_42: np.ndarray, catboost_2025: np.ndarray, lightgbm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the frozen current and two-seed candidate blend weights."""
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (catboost_42, catboost_2025, lightgbm)
    )
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("blend components must have equal lengths")
    for value in arrays:
        validate_probabilities(value, len(value))
    baseline = 0.5 * arrays[0] + 0.5 * arrays[2]
    candidate = 0.25 * arrays[0] + 0.25 * arrays[1] + 0.5 * arrays[2]
    validate_probabilities(baseline, len(baseline))
    validate_probabilities(candidate, len(candidate))
    return baseline, candidate


def five_fold_gate(
    mean_paired_auc_gain: float, positive_fold_count: int, pooled_oof_auc_gain: float
) -> dict[str, Any]:
    """Apply the predeclared informational advancement rule."""
    qualifies = (
        mean_paired_auc_gain >= MINIMUM_AUC_GAIN
        and positive_fold_count >= 4
        and pooled_oof_auc_gain > 0
    )
    return {
        "qualifies": qualifies,
        "decision": "advance_candidate" if qualifies else "retain_current_blend",
        "minimum_mean_paired_auc_gain": MINIMUM_AUC_GAIN,
        "minimum_positive_folds": 4,
        "pooled_oof_gain_must_be_positive": True,
    }


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    validate_probabilities(probabilities, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def build_catboost(seed: int) -> Any:
    model = build_slow_catboost()
    if seed == 42:
        return model
    if seed != 2025:
        raise ValueError(f"unexpected CatBoost model seed: {seed}")
    before = model.get_params()
    model.set_params(random_seed=seed)
    after = model.get_params()
    changed = {
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    }
    if changed != {"random_seed"}:
        raise RuntimeError(f"CatBoost candidate changed unexpected settings: {changed}")
    return model


def _cat_settings(model: Any) -> dict[str, Any]:
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


def _progress_logger(
    metadata: dict[str, Any], path: Path, fold_id: int, model_name: str
) -> Any:
    def log(message: str) -> None:
        for line in str(message).splitlines():
            text = line.strip()
            if not text or not text[0].isdigit() or ":" not in text:
                continue
            tree = int(text.split(":", 1)[0])
            if tree == 0 or tree % PROGRESS_INTERVAL:
                continue
            event = {"utc": datetime.now(timezone.utc).isoformat(), "tree": tree}
            current = metadata.get("current_fit")
            if isinstance(current, dict):
                current.setdefault("progress_log", []).append(event)
                current["last_progress_tree"] = tree
                save_json(path, metadata)
            print(
                f"{event['utc']} fold={fold_id} model={model_name} tree={tree}",
                flush=True,
            )

    return log


def _save_component(
    path: Path,
    rows: np.ndarray,
    ids: np.ndarray,
    fold_id: int,
    probabilities: np.ndarray,
) -> None:
    validate_probabilities(probabilities, len(rows))
    np.savez_compressed(
        path,
        row_index=rows,
        id=ids,
        fold_id=np.full(len(rows), fold_id, dtype=np.int8),
        probability=probabilities,
    )


def _cat_fit(
    model: Any,
    X_cat: pd.DataFrame,
    y: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    spec: Any,
    logger: Any,
) -> None:
    model.fit(
        X_cat.iloc[inner_fit],
        y[inner_fit],
        eval_set=(X_cat.iloc[inner_validation], y[inner_validation]),
        cat_features=list(spec.categorical_columns),
        verbose=PROGRESS_INTERVAL,
        log_cout=logger,
    )


def _cat_details(model: Any, seed: int) -> dict[str, Any]:
    best = int(model.get_best_iteration())
    return {
        "model_seed": seed,
        "best_iteration_zero_based": best,
        "best_iteration_1based": best + 1,
        "actual_tree_count": int(model.tree_count_),
        "settings": _cat_settings(model),
    }


def _cat_predict(model: Any, X_cat: pd.DataFrame, holdout: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_cat.iloc[holdout])[:, 1]


def _light_fit(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    spec: Any,
    levels: dict[str, list[str]],
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
) -> None:
    model.fit(
        prepare_lightgbm_frame(X.iloc[inner_fit], spec, levels),
        y[inner_fit],
        eval_set=[
            (
                prepare_lightgbm_frame(X.iloc[inner_validation], spec, levels),
                y[inner_validation],
            )
        ],
        eval_names=["inner_validation"],
        eval_metric="auc",
        categorical_feature=list(spec.categorical_columns),
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )


def _light_details(model: Any, levels: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "best_iteration_1based": int(model.best_iteration_),
        "actual_tree_count": int(model.booster_.current_iteration()),
        "settings": dict(PREDECLARED_CONFIG),
        "early_stopping": {"metric": "auc", "rounds": 100},
        "fit_only_category_level_counts": {
            name: len(values) for name, values in levels.items()
        },
    }


def _light_predict(
    model: Any,
    X: pd.DataFrame,
    spec: Any,
    levels: dict[str, list[str]],
    holdout: np.ndarray,
) -> np.ndarray:
    return model.predict_proba(prepare_lightgbm_frame(X.iloc[holdout], spec, levels))[
        :, 1
    ]


def _fit(
    name: str,
    fold_id: int,
    model: Any,
    fit: Any,
    predict: Any,
    result_details: Any,
    y: np.ndarray,
    holdout: np.ndarray,
    ids: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    metadata["current_fit"] = {
        "fold_id": fold_id,
        "model": name,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "outer_holdout_rows": len(holdout),
    }
    save_json(metadata_path, metadata)
    try:
        before = _resource_snapshot()
        fit()
        probabilities = np.asarray(predict(), dtype=np.float64)
        validate_probabilities(probabilities, len(holdout))
        prediction_path = output_dir / f"fold{fold_id}_{name}.npz"
        _save_component(prediction_path, holdout, ids[holdout], fold_id, probabilities)
        result = {
            "fold_id": fold_id,
            "model": name,
            "metrics": _metrics(y[holdout], probabilities),
            "fit_seconds": round(time.perf_counter() - started, 3),
            "resources": _resource_delta(before, _resource_snapshot()),
            "prediction_path": str(prediction_path),
            **result_details(),
        }
        metadata["current_fit"]["status"] = "completed"
        metadata["current_fit"]["completed_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        metadata["completed_fits"].append(result)
        metadata["current_fit"] = None
        save_json(metadata_path, metadata)
        return probabilities, result
    finally:
        del model
        gc.collect()


def _source_hashes() -> dict[str, str]:
    names = (
        "five_fold_cv.py",
        "slower_learning.py",
        "compare.py",
        "lightgbm_experiment.py",
        "categorical_ratings.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _data_hashes(data_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(data_dir)): sha256_file(path)
        for path in sorted(data_dir.rglob("*.csv"))
    }


def _record_split(
    fold_id: int,
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
        "fold_id": fold_id,
        "rows": {name: len(value) for name, value in arrays.items()},
        "sha256": {name: _array_fingerprint(value) for name, value in arrays.items()},
        "inner_fit_disjoint_from_validation": not bool(
            np.intersect1d(inner_fit, inner_validation).size
        ),
        "inner_covers_outer_fit": np.array_equal(
            np.sort(np.concatenate((inner_fit, inner_validation))), np.sort(outer_fit)
        ),
        "inner_disjoint_from_outer_holdout": not bool(
            np.intersect1d(inner_fit, outer_holdout).size
            or np.intersect1d(inner_validation, outer_holdout).size
        ),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run all 15 fixed fits and persist complete OOF evidence."""
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
        "completed_fits": [],
        "fold_results": [],
        "current_fit": None,
        "selection": None,
    }
    save_json(metadata_path, metadata)
    protected_before: dict[str, str] = {}
    input_before: dict[str, str] | None = None
    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_alarm = signal.getsignal(signal.SIGALRM)
    signals_installed = False
    try:
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _data_hashes(data_dir)
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        X = train.loc[:, list(spec.feature_columns)]
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        if not pd.Index(ids).is_unique:
            raise RuntimeError("training IDs are not unique")
        folds = make_five_fold_splits(y)
        X_cat = prepare_catboost_frame(train, spec)
        metadata.update(
            {
                "python": sys.version,
                "dependencies": {
                    **package_versions(),
                    **{
                        name: version(name)
                        for name in ("catboost", "lightgbm", "threadpoolctl")
                    },
                },
                "threads": MAX_THREADS,
                "input_sha256_before": input_before,
                "source_sha256": _source_hashes(),
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "numeric_nan_policy": "preserve numeric NaN; no imputation",
                "predeclared_config": {
                    "fold_count": FOLD_COUNT,
                    "fold_seed": FOLD_SEED,
                    "inner_validation_seed": INTERNAL_SEED,
                    "catboost_model_seeds": list(MODEL_SEEDS),
                    "catboost": _cat_settings(build_slow_catboost()),
                    "lightgbm": dict(PREDECLARED_CONFIG),
                    "blend_weights": {
                        "baseline": {"catboost_seed42": 0.5, "lightgbm": 0.5},
                        "candidate": {
                            "catboost_seed42": 0.25,
                            "catboost_seed2025": 0.25,
                            "lightgbm": 0.5,
                        },
                    },
                    "advancement_rule": {
                        "minimum_mean_paired_auc_gain": MINIMUM_AUC_GAIN,
                        "minimum_positive_folds": 4,
                        "pooled_oof_auc_gain_positive": True,
                    },
                },
                "protected_artifact_sha256_before": protected_before,
            }
        )
        save_json(metadata_path, metadata)
        records: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {
            name: []
            for name in (
                "catboost_seed42",
                "catboost_seed2025",
                "lightgbm",
                "baseline",
                "candidate",
            )
        }
        for fold_id, (outer_fit, holdout) in enumerate(folds, start=1):
            inner_fit, inner_validation = make_catboost_internal_split(
                outer_fit, y, seed=INTERNAL_SEED
            )
            split = _record_split(
                fold_id, outer_fit, holdout, inner_fit, inner_validation
            )
            if not all(
                (
                    split["inner_fit_disjoint_from_validation"],
                    split["inner_covers_outer_fit"],
                    split["inner_disjoint_from_outer_holdout"],
                )
            ):
                raise RuntimeError(f"invalid inner split membership in fold {fold_id}")
            metadata.setdefault("splits", []).append(split)
            cat42_model = build_catboost(42)
            cat42, cat42_result = _fit(
                "catboost_seed42",
                fold_id,
                cat42_model,
                partial(
                    _cat_fit,
                    cat42_model,
                    X_cat,
                    y,
                    inner_fit,
                    inner_validation,
                    spec,
                    _progress_logger(
                        metadata, metadata_path, fold_id, "catboost_seed42"
                    ),
                ),
                partial(_cat_predict, cat42_model, X_cat, holdout),
                partial(_cat_details, cat42_model, 42),
                y,
                holdout,
                ids,
                output_dir,
                metadata,
                metadata_path,
            )
            cat2025_model = build_catboost(2025)
            cat2025, cat2025_result = _fit(
                "catboost_seed2025",
                fold_id,
                cat2025_model,
                partial(
                    _cat_fit,
                    cat2025_model,
                    X_cat,
                    y,
                    inner_fit,
                    inner_validation,
                    spec,
                    _progress_logger(
                        metadata, metadata_path, fold_id, "catboost_seed2025"
                    ),
                ),
                partial(_cat_predict, cat2025_model, X_cat, holdout),
                partial(_cat_details, cat2025_model, 2025),
                y,
                holdout,
                ids,
                output_dir,
                metadata,
                metadata_path,
            )
            levels = fit_category_levels(X.iloc[inner_fit], spec)
            light_model = lgb.LGBMClassifier(**PREDECLARED_CONFIG)
            light, light_result = _fit(
                "lightgbm",
                fold_id,
                light_model,
                partial(
                    _light_fit,
                    light_model,
                    X,
                    y,
                    spec,
                    levels,
                    inner_fit,
                    inner_validation,
                ),
                partial(_light_predict, light_model, X, spec, levels, holdout),
                partial(_light_details, light_model, levels),
                y,
                holdout,
                ids,
                output_dir,
                metadata,
                metadata_path,
            )
            baseline, candidate = fixed_blends(cat42, cat2025, light)
            for name, values in (
                ("catboost_seed42", cat42),
                ("catboost_seed2025", cat2025),
                ("lightgbm", light),
                ("baseline", baseline),
                ("candidate", candidate),
            ):
                records[name].append((fold_id, holdout, values))
            baseline_metrics, candidate_metrics = (
                _metrics(y[holdout], baseline),
                _metrics(y[holdout], candidate),
            )
            metadata["fold_results"].append(
                {
                    "fold_id": fold_id,
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "paired_auc_gain": candidate_metrics["roc_auc"]
                    - baseline_metrics["roc_auc"],
                    "components": {
                        "catboost_seed42": cat42_result,
                        "catboost_seed2025": cat2025_result,
                        "lightgbm": light_result,
                    },
                }
            )
            save_json(metadata_path, metadata)
            del cat42, cat2025, light, baseline, candidate
            del cat42_model, cat2025_model, light_model, levels
            gc.collect()
        oof: dict[str, np.ndarray] = {}
        fold_ids: np.ndarray | None = None
        for name, component_records in records.items():
            values, assigned = assign_oof(len(train), component_records)
            if fold_ids is None:
                fold_ids = assigned
            elif not np.array_equal(fold_ids, assigned):
                raise RuntimeError(f"OOF fold membership differs for {name}")
            oof[name] = values
        assert fold_ids is not None
        oof_path = output_dir / "oof_predictions.npz"
        np.savez_compressed(
            oof_path,
            row_index=np.arange(len(train), dtype=np.int64),
            id=ids,
            fold_id=fold_ids,
            catboost_seed42_probability=oof["catboost_seed42"],
            catboost_seed2025_probability=oof["catboost_seed2025"],
            lightgbm_probability=oof["lightgbm"],
            baseline_probability=oof["baseline"],
            candidate_probability=oof["candidate"],
        )
        with np.load(oof_path) as saved:
            if (
                not np.array_equal(saved["row_index"], np.arange(len(train)))
                or not np.array_equal(saved["id"], ids)
                or not np.array_equal(saved["fold_id"], fold_ids)
            ):
                raise RuntimeError("saved OOF row, ID, or fold alignment failed")
        baseline_all, candidate_all = (
            _metrics(y, oof["baseline"]),
            _metrics(y, oof["candidate"]),
        )
        gains = [result["paired_auc_gain"] for result in metadata["fold_results"]]
        summary: dict[str, Any] = {
            "fold_count": FOLD_COUNT,
            "oof_rows": len(train),
            "oof_unique_rows": int(len(np.unique(np.arange(len(train))))),
            "mean_fold_baseline": float(
                np.mean(
                    [
                        result["baseline"]["roc_auc"]
                        for result in metadata["fold_results"]
                    ]
                )
            ),
            "mean_fold_candidate": float(
                np.mean(
                    [
                        result["candidate"]["roc_auc"]
                        for result in metadata["fold_results"]
                    ]
                )
            ),
            "mean_paired_auc_gain": float(np.mean(gains)),
            "std_paired_auc_gain": float(np.std(gains)),
            "paired_auc_gain_range": [float(min(gains)), float(max(gains))],
            "positive_fold_count": int(sum(gain > 0 for gain in gains)),
            "pooled_oof": {
                "baseline": baseline_all,
                "candidate": candidate_all,
                "roc_auc_gain": candidate_all["roc_auc"] - baseline_all["roc_auc"],
            },
            "interpretation": (
                "Pooled OOF scores combine fold-specific fits; fold standard "
                "deviation is descriptive, not a formal confidence interval."
            ),
        }
        metadata["summary"] = summary
        metadata["selection"] = five_fold_gate(
            float(summary["mean_paired_auc_gain"]),
            int(summary["positive_fold_count"]),
            float(summary["pooled_oof"]["roc_auc_gain"]),
        )
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, FiveFoldCVTimeout) else "failed"
        )
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
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
            metadata["integrity_error"] = {
                "type": type(finalization_error).__name__,
                "message": str(finalization_error),
            }
            if caught_error is None:
                caught_error = finalization_error
                metadata["error"] = metadata["integrity_error"]
            else:
                metadata["finalization_error"] = metadata["integrity_error"]
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        try:
            save_json(
                output_dir / "protected_baseline_sha256.json",
                metadata.get("protected_artifact_sha256_after", protected_before),
            )
            metadata["output_bytes"] = sum(
                path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
            )
            if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
                budget_error = RuntimeError("output directory exceeded 150 MiB budget")
                metadata["status"] = "failed"
                metadata["error"] = {
                    "type": type(budget_error).__name__,
                    "message": str(budget_error),
                }
                if caught_error is None:
                    caught_error = budget_error
                else:
                    metadata["finalization_error"] = metadata["error"]
            save_json(metadata_path, metadata)
        except BaseException as error:
            if caught_error is None:
                caught_error = error
                metadata["status"] = "failed"
                metadata["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
    if caught_error is not None:
        raise caught_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed five-fold CV comparison."
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
                "summary": metadata["summary"],
                "artifacts_dir": metadata["output_dir"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
