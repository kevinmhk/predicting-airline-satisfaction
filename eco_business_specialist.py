"""Run the one fixed Eco/business/loyal specialist OOF experiment."""

from __future__ import annotations

import argparse
import gc
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
from sklearn.metrics import log_loss, roc_auc_score

from categorical_ratings import _array_fingerprint
from compare import (
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from five_fold_cv import (
    _cat_settings as cv_cat_settings,
)
from five_fold_cv import (
    _data_hashes,
    make_five_fold_splits,
)
from five_fold_cv import (
    _metrics as fold_metrics,
)
from five_fold_cv import (
    _progress_logger as cv_progress_logger,
)
from lightgbm_experiment import PREDECLARED_CONFIG, _artifact_snapshot
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
GROUP_ROW_COUNT = 65_542
SPECIALIST_WEIGHT = 0.25
BASELINE_WEIGHT = 0.75
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 21_300
EXTERNAL_TIMEOUT_SECONDS = 21_600
OUTPUT_NAME = "eco_business_specialist"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 500
GROUP_COLUMNS = ("Class", "Type of Travel", "Customer Type")
GROUP_VALUES = ("Eco", "Business travel", "Loyal Customer")


class EcoBusinessSpecialistTimeout(TimeoutError):
    """Raised when the fixed specialist run reaches its internal deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise EcoBusinessSpecialistTimeout(
        "Eco business specialist experiment exceeded its six-hour budget"
    )


def group_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return the frozen specialist group mask without changing the frame."""
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(GROUP_COLUMNS, GROUP_VALUES, strict=True):
        mask &= frame[column].eq(value).to_numpy(dtype=bool)
    return mask


def specialist_membership(
    outer_fit: np.ndarray,
    outer_holdout: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build and validate the group-only fit, early-stop, and holdout rows."""
    inner_fit, inner_validation = make_catboost_internal_split(
        np.asarray(outer_fit, dtype=np.int64), y, seed=INTERNAL_SEED
    )
    group = np.asarray(group, dtype=bool)
    result = {
        "inner_fit": inner_fit[group[inner_fit]],
        "inner_validation": inner_validation[group[inner_validation]],
        "outer_holdout": np.asarray(outer_holdout, dtype=np.int64)[
            group[outer_holdout]
        ],
    }
    if any(np.unique(y[rows]).size != 2 for rows in result.values()):
        raise RuntimeError("specialist membership does not contain both classes")
    if np.intersect1d(result["inner_fit"], result["inner_validation"]).size:
        raise RuntimeError("specialist inner fit and validation overlap")
    if any(
        np.intersect1d(result[name], result["outer_holdout"]).size
        for name in ("inner_fit", "inner_validation")
    ):
        raise RuntimeError("specialist inner rows overlap outer holdout")
    return result


def route_candidate(
    baseline: np.ndarray, specialist: np.ndarray, group: np.ndarray
) -> np.ndarray:
    """Replace only group rows with the fixed 75/25 baseline/specialist blend."""
    baseline = np.asarray(baseline, dtype=np.float64)
    specialist = np.asarray(specialist, dtype=np.float64)
    group = np.asarray(group, dtype=bool)
    if baseline.ndim != 1 or specialist.shape != baseline.shape:
        raise ValueError("baseline and specialist probabilities must align")
    validate_probabilities(baseline, len(baseline))
    validate_probabilities(specialist[group], int(group.sum()))
    result = baseline.copy()
    result[group] = (
        BASELINE_WEIGHT * baseline[group] + SPECIALIST_WEIGHT * specialist[group]
    )
    validate_probabilities(result, len(result))
    return result


def assign_specialist_oof(
    row_count: int,
    records: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every specialist-group OOF row exactly once."""
    values = np.full(row_count, np.nan, dtype=np.float64)
    fold_ids = np.full(row_count, -1, dtype=np.int8)
    for fold_id, rows, probabilities in records:
        rows = np.asarray(rows, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if len(rows) != len(probabilities):
            raise ValueError("specialist row and probability lengths do not match")
        if (
            np.any(rows < 0)
            or np.any(rows >= row_count)
            or np.unique(rows).size != len(rows)
        ):
            raise ValueError("specialist OOF component has invalid or duplicate rows")
        if np.any(fold_ids[rows] != -1):
            raise ValueError("specialist OOF component assigns a row twice")
        validate_probabilities(probabilities, len(rows))
        values[rows] = probabilities
        fold_ids[rows] = fold_id
    assigned = fold_ids != -1
    if not assigned.any():
        raise ValueError("specialist OOF component is empty")
    if np.any(~np.isfinite(values[assigned])):
        raise ValueError("specialist OOF component contains invalid values")
    return values, fold_ids


def specialist_gate(
    mean_paired_auc_gain: float, positive_fold_count: int, pooled_auc_gain: float
) -> dict[str, Any]:
    """Apply the frozen conservative advancement rule."""
    qualifies = (
        mean_paired_auc_gain >= MINIMUM_AUC_GAIN
        and positive_fold_count >= 4
        and pooled_auc_gain > 0
    )
    return {
        "qualifies": qualifies,
        "decision": "advance_candidate" if qualifies else "retain_current_blend",
        "minimum_mean_paired_auc_gain": MINIMUM_AUC_GAIN,
        "minimum_positive_folds": 4,
        "pooled_auc_gain_must_be_positive": True,
    }


def _calibration_report(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= 0.5
    actual = np.asarray(y_true, dtype=np.int8)
    errors = predicted != actual
    return {
        "mean_prediction": float(np.mean(probabilities)),
        "mean_actual": float(np.mean(actual)),
        "mean_prediction_minus_actual": float(np.mean(probabilities) - np.mean(actual)),
        "threshold": 0.5,
        "error_count": int(errors.sum()),
        "error_rate": float(np.mean(errors)),
        "accuracy": float(np.mean(~errors)),
        "positive_prediction_rate": float(np.mean(predicted)),
    }


def _subgroup_report(
    y_true: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        name: {
            "metrics": fold_metrics(y_true, values),
            "calibration_and_threshold_0_5": _calibration_report(y_true, values),
        }
        for name, values in predictions.items()
    }


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_hashes() -> dict[str, str]:
    names = (
        "eco_business_specialist.py",
        "five_fold_cv.py",
        "slower_learning.py",
        "compare.py",
        "lightgbm_experiment.py",
        "categorical_ratings.py",
        "train.py",
    )
    hashes = {name: sha256_file(Path(__file__).with_name(name)) for name in names}
    hashes["eco_business_specialist.py"] = sha256_file(Path(__file__))
    return hashes


def _baseline_hashes(artifacts_dir: Path) -> dict[str, str]:
    names = (
        "five_fold_cv/compare_metadata.json",
        "five_fold_cv/oof_predictions.npz",
        "five_fold_cv/protected_baseline_sha256.json",
    )
    return {
        name: sha256_file(artifacts_dir / name)
        for name in names
        if (artifacts_dir / name).is_file()
    }


def _fold_membership_hash(folds: tuple[tuple[np.ndarray, np.ndarray], ...]) -> str:
    digest = hashlib.sha256()
    for fold_id, (fit, holdout) in enumerate(folds, start=1):
        digest.update(str(fold_id).encode("ascii"))
        digest.update(_array_fingerprint(fit).encode("ascii"))
        digest.update(_array_fingerprint(holdout).encode("ascii"))
    return digest.hexdigest()


def _fold_record(
    fold_id: int,
    outer_fit: np.ndarray,
    outer_holdout: np.ndarray,
    membership: dict[str, np.ndarray],
) -> dict[str, Any]:
    arrays = {
        "outer_fit": outer_fit,
        "outer_holdout": outer_holdout,
        **membership,
    }
    return {
        "fold_id": fold_id,
        "rows": {name: len(value) for name, value in arrays.items()},
        "sha256": {name: _array_fingerprint(value) for name, value in arrays.items()},
        "inner_fit_disjoint_from_validation": not bool(
            np.intersect1d(membership["inner_fit"], membership["inner_validation"]).size
        ),
        "inner_disjoint_from_outer_holdout": not bool(
            np.intersect1d(membership["inner_fit"], membership["outer_holdout"]).size
            or np.intersect1d(
                membership["inner_validation"], membership["outer_holdout"]
            ).size
        ),
    }


def _frozen_config(spec: Any) -> dict[str, Any]:
    return {
        "fold_count": FOLD_COUNT,
        "fold_seed": FOLD_SEED,
        "inner_validation_seed": INTERNAL_SEED,
        "group_columns": list(GROUP_COLUMNS),
        "group_values": list(GROUP_VALUES),
        "group_row_count": GROUP_ROW_COUNT,
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "numeric_nan_policy": "preserve numeric NaN; no imputation",
        "catboost": cv_cat_settings(build_slow_catboost()),
        "catboost_cat_features": list(spec.categorical_columns),
        "routing": {
            "baseline_weight": BASELINE_WEIGHT,
            "specialist_weight": SPECIALIST_WEIGHT,
            "non_group_action": "bitwise_copy_baseline",
        },
        "advancement_rule": {
            "minimum_mean_paired_auc_gain": MINIMUM_AUC_GAIN,
            "minimum_positive_folds": 4,
            "pooled_auc_gain_positive": True,
        },
        "external_data_used": False,
    }


def _save_component(
    path: Path, rows: np.ndarray, ids: np.ndarray, fold_id: int, values: np.ndarray
) -> None:
    validate_probabilities(values, len(rows))
    np.savez_compressed(
        path,
        row_index=np.asarray(rows, dtype=np.int64),
        id=np.asarray(ids, dtype=np.int64),
        fold_id=np.full(len(rows), fold_id, dtype=np.int8),
        group_probability=np.asarray(values, dtype=np.float64),
    )


def _fit_specialist(
    *,
    fold_id: int,
    X_cat: pd.DataFrame,
    y: np.ndarray,
    spec: Any,
    membership: dict[str, np.ndarray],
    ids: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    metadata["current_fit"] = {
        "fold_id": fold_id,
        "model": "eco_business_specialist",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inner_fit_rows": len(membership["inner_fit"]),
        "inner_validation_rows": len(membership["inner_validation"]),
        "outer_holdout_group_rows": len(membership["outer_holdout"]),
    }
    save_json(metadata_path, metadata)
    model: Any = None
    try:
        model = build_slow_catboost()
        before = _resource_snapshot()
        fit_started = time.perf_counter()
        model.fit(
            X_cat.iloc[membership["inner_fit"]],
            y[membership["inner_fit"]],
            eval_set=(
                X_cat.iloc[membership["inner_validation"]],
                y[membership["inner_validation"]],
            ),
            cat_features=list(spec.categorical_columns),
            verbose=PROGRESS_INTERVAL,
            log_cout=cv_progress_logger(
                metadata, metadata_path, fold_id, "eco_business_specialist"
            ),
        )
        fit_seconds = time.perf_counter() - fit_started
        probabilities = np.asarray(
            model.predict_proba(X_cat.iloc[membership["outer_holdout"]])[:, 1],
            dtype=np.float64,
        )
        validate_probabilities(probabilities, len(membership["outer_holdout"]))
        after = _resource_snapshot()
        prediction_path = output_dir / f"fold{fold_id}_specialist.npz"
        _save_component(
            prediction_path,
            membership["outer_holdout"],
            ids[membership["outer_holdout"]],
            fold_id,
            probabilities,
        )
        result = {
            "fold_id": fold_id,
            "fit_seconds": round(fit_seconds, 3),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "metrics": fold_metrics(y[membership["outer_holdout"]], probabilities),
            "best_iteration_zero_based": int(model.get_best_iteration()),
            "best_iteration_1based": int(model.get_best_iteration()) + 1,
            "actual_tree_count": int(model.tree_count_),
            "settings": cv_cat_settings(build_slow_catboost()),
            "resources": _resource_delta(before, after),
            "prediction_path": str(prediction_path),
        }
        current = metadata["current_fit"]
        if isinstance(current, dict):
            current["status"] = "completed"
            current["completed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_fits"].append(result)
        metadata["current_fit"] = None
        save_json(metadata_path, metadata)
        return probabilities, result
    finally:
        del model
        gc.collect()


def _verify_prior_oof(
    data_dir: Path,
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    current_input_hashes: dict[str, str],
) -> dict[str, Any]:
    """Verify the completed frozen baseline and return its row-aligned arrays."""
    metadata_path = artifacts_dir / "five_fold_cv" / "compare_metadata.json"
    oof_path = artifacts_dir / "five_fold_cv" / "oof_predictions.npz"
    if not metadata_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError(
            "completed five-fold baseline OOF artifacts are required"
        )
    prior = json.loads(metadata_path.read_text(encoding="utf-8"))
    if prior.get("status") != "completed":
        raise RuntimeError("frozen five-fold baseline is not completed")
    if (
        prior.get("completed_fits") is None
        or len(prior["completed_fits"]) != FOLD_COUNT * 3
        or len(prior.get("splits", [])) != FOLD_COUNT
    ):
        raise RuntimeError("frozen five-fold baseline does not contain 15 fits")
    if prior.get("input_sha256_before") != current_input_hashes:
        raise RuntimeError("frozen five-fold input fingerprints changed")
    current_sources = _source_hashes()
    prior_sources = prior.get("source_sha256", {})
    for name, digest in prior_sources.items():
        if name not in current_sources or current_sources[name] != digest:
            raise RuntimeError(f"frozen five-fold source fingerprint changed: {name}")
    prior_config = prior.get("predeclared_config", {})
    if prior_config.get("fold_count") != FOLD_COUNT:
        raise RuntimeError("frozen five-fold count changed")
    if prior_config.get("fold_seed") != FOLD_SEED:
        raise RuntimeError("frozen five-fold seed changed")
    if prior_config.get("inner_validation_seed") != INTERNAL_SEED:
        raise RuntimeError("frozen inner validation seed changed")
    if prior_config.get("catboost_model_seeds") != [42, 2025]:
        raise RuntimeError("frozen CatBoost seeds changed")
    if prior_config.get("catboost") != cv_cat_settings(build_slow_catboost()):
        raise RuntimeError("frozen CatBoost configuration changed")
    if prior_config.get("lightgbm") != PREDECLARED_CONFIG:
        raise RuntimeError("frozen LightGBM configuration changed")
    if prior_config.get("blend_weights", {}).get("baseline") != {
        "catboost_seed42": 0.5,
        "lightgbm": 0.5,
    }:
        raise RuntimeError("frozen baseline blend weights changed")

    with np.load(oof_path) as saved:
        expected_rows = np.arange(len(train), dtype=np.int64)
        row_index = np.asarray(saved["row_index"], dtype=np.int64)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        saved_ids = np.asarray(saved["id"], dtype=np.int64)
        fold_ids = np.asarray(saved["fold_id"], dtype=np.int8)
        cat42 = np.asarray(saved["catboost_seed42_probability"], dtype=np.float64)
        light = np.asarray(saved["lightgbm_probability"], dtype=np.float64)
        baseline = np.asarray(saved["baseline_probability"], dtype=np.float64)
    if not np.array_equal(row_index, expected_rows) or not np.array_equal(
        saved_ids, ids
    ):
        raise RuntimeError("frozen OOF row or ID order changed")
    if not np.array_equal(
        np.sort(fold_ids), np.repeat(np.arange(1, 6), len(train) // 5)
    ):
        raise RuntimeError("frozen OOF fold IDs are incomplete")
    validate_probabilities(cat42, len(train))
    validate_probabilities(light, len(train))
    validate_probabilities(baseline, len(train))
    expected_baseline = 0.5 * cat42 + 0.5 * light
    if not np.array_equal(baseline, expected_baseline):
        raise RuntimeError("frozen baseline is not exactly 50/50 CatBoost/LightGBM")

    folds = make_five_fold_splits(y)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_id, (_fit, holdout) in enumerate(folds, start=1):
        expected_fold_ids[holdout] = fold_id
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise RuntimeError("frozen OOF fold membership differs from regenerated folds")
    if prior.get("summary", {}).get("oof_rows") != len(train) or prior.get(
        "summary", {}
    ).get("oof_unique_rows") != len(train):
        raise RuntimeError("frozen OOF row counts changed")
    return {
        "metadata": prior,
        "metadata_path": metadata_path,
        "oof_path": oof_path,
        "row_index": row_index,
        "ids": ids,
        "fold_ids": fold_ids,
        "catboost_seed42": cat42,
        "lightgbm": light,
        "baseline": baseline,
        "folds": folds,
    }


def _summary(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    specialist: np.ndarray,
    group: np.ndarray,
    fold_results: list[dict[str, Any]],
    assigned_folds: np.ndarray,
) -> dict[str, Any]:
    def mean_metrics(key: str) -> dict[str, float]:
        return {
            metric: float(np.mean([row[key][metric] for row in fold_results]))
            for metric in ("roc_auc", "log_loss")
        }

    auc_gains = [float(row["paired_auc_gain"]) for row in fold_results]
    log_loss_gains = [float(row["paired_log_loss_gain"]) for row in fold_results]
    pooled = _subgroup_report(
        y[group],
        {
            "baseline": baseline[group],
            "specialist": specialist[group],
            "candidate": candidate[group],
        },
    )
    baseline_metrics = fold_metrics(y, baseline)
    candidate_metrics = fold_metrics(y, candidate)
    return {
        "rows": len(y),
        "specialist_oof_rows": int(np.count_nonzero(assigned_folds > 0)),
        "specialist_oof_unique_rows": int(
            np.unique(np.flatnonzero(assigned_folds > 0)).size
        ),
        "global_per_fold": fold_results,
        "mean_fold": {
            "baseline": mean_metrics("global_baseline"),
            "candidate": mean_metrics("global_candidate"),
            "gain": {
                "roc_auc": float(np.mean(auc_gains)),
                "log_loss": float(np.mean(log_loss_gains)),
            },
        },
        "mean_paired_auc_gain": float(np.mean(auc_gains)),
        "positive_fold_count": int(sum(gain > 0 for gain in auc_gains)),
        "pooled_global": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "roc_auc_gain": candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"],
            "log_loss_gain": candidate_metrics["log_loss"]
            - baseline_metrics["log_loss"],
        },
        "subgroup": {
            "per_fold": [row["subgroup"] for row in fold_results],
            "pooled": pooled,
            "threshold_note": "Classification errors use probability >= 0.5.",
        },
        "interpretation": (
            "Global fold metrics include cross-group ordering effects. "
            "Subgroup diagnostics are descriptive calibration/error summaries, "
            "not label-noise or causal claims."
        ),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run five fixed specialist fits and persist row-aligned OOF evidence."""
    started = time.perf_counter()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory: {output_dir}"
        )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "completed_fits": [],
        "current_fit": None,
        "current_fold": None,
        "fold_results": [],
        "summary": None,
        "selection": None,
        "external_data_used": False,
    }
    save_json(metadata_path, metadata)
    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    protected_before: dict[str, str] = {}
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    baseline_before: dict[str, str] = {}
    configuration: dict[str, Any] | None = None
    spec: Any = None
    y: np.ndarray | None = None
    folds: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
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
        source_before = _source_hashes()
        baseline_before = _baseline_hashes(artifacts_dir)
        metadata.update(
            {
                "protected_artifact_sha256_before": protected_before,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "frozen_baseline_sha256_before": baseline_before,
            }
        )
        save_json(metadata_path, metadata)

        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        prior = _verify_prior_oof(data_dir, artifacts_dir, train, y, input_before)
        folds = prior["folds"]
        if folds is None:
            raise RuntimeError("frozen five-fold membership is missing")
        group = group_mask(train)
        if int(group.sum()) != GROUP_ROW_COUNT:
            raise RuntimeError(
                f"specialist group row count changed: {int(group.sum())}"
            )
        if np.unique(y[group]).size != 2:
            raise RuntimeError("specialist group does not contain both target classes")
        configuration = _frozen_config(spec)
        feature_spec = {
            "feature_columns": list(spec.feature_columns),
            "numeric_columns": list(spec.numeric_columns),
            "categorical_columns": list(spec.categorical_columns),
        }
        metadata.update(
            {
                "python": sys.version,
                "dependencies": {
                    **package_versions(),
                    "catboost": version("catboost"),
                    "lightgbm": version("lightgbm"),
                    "threadpoolctl": version("threadpoolctl"),
                },
                "threads": MAX_THREADS,
                "feature_spec": feature_spec,
                "configuration": configuration,
                "configuration_sha256_before": _hash_json(configuration),
                "feature_spec_sha256_before": _hash_json(feature_spec),
                "fold_membership_sha256_before": _fold_membership_hash(folds),
                "folds": [],
                "prior_oof_metadata_path": str(prior["metadata_path"]),
                "prior_oof_path": str(prior["oof_path"]),
                "prior_oof_metadata_sha256": sha256_file(prior["metadata_path"]),
                "prior_oof_sha256": sha256_file(prior["oof_path"]),
                "group": {
                    "columns": list(GROUP_COLUMNS),
                    "values": list(GROUP_VALUES),
                    "rows": int(group.sum()),
                    "label_counts": {
                        str(label): int(count)
                        for label, count in zip(
                            *np.unique(y[group], return_counts=True), strict=True
                        )
                    },
                },
            }
        )
        save_json(metadata_path, metadata)
        X_cat = prepare_catboost_frame(train, spec)
        ids = train[ID_COLUMN].to_numpy(dtype=np.int64)
        specialist_records: list[tuple[int, np.ndarray, np.ndarray]] = []
        specialist_full = np.full(len(train), np.nan, dtype=np.float64)
        specialist_fold_ids = np.full(len(train), -1, dtype=np.int8)
        for fold_id, (outer_fit, outer_holdout) in enumerate(folds, start=1):
            metadata["current_fold"] = {
                "fold_id": fold_id,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            membership = specialist_membership(outer_fit, outer_holdout, y, group)
            metadata["folds"].append(
                _fold_record(fold_id, outer_fit, outer_holdout, membership)
            )
            save_json(metadata_path, metadata)
            probabilities, specialist_result = _fit_specialist(
                fold_id=fold_id,
                X_cat=X_cat,
                y=y,
                spec=spec,
                membership=membership,
                ids=ids,
                output_dir=output_dir,
                metadata=metadata,
                metadata_path=metadata_path,
            )
            rows = membership["outer_holdout"]
            specialist_records.append((fold_id, rows, probabilities))
            specialist_full[rows] = probabilities
            specialist_fold_ids[rows] = fold_id
            baseline_fold = prior["baseline"][outer_holdout]
            candidate_fold = baseline_fold.copy()
            candidate_fold[group[outer_holdout]] = (
                BASELINE_WEIGHT * baseline_fold[group[outer_holdout]]
                + SPECIALIST_WEIGHT * probabilities
            )
            fold_group = group[outer_holdout]
            baseline_group = baseline_fold[fold_group]
            candidate_group = candidate_fold[fold_group]
            metadata["fold_results"].append(
                {
                    "fold_id": fold_id,
                    "global_baseline": fold_metrics(y[outer_holdout], baseline_fold),
                    "global_candidate": fold_metrics(y[outer_holdout], candidate_fold),
                    "paired_auc_gain": float(
                        roc_auc_score(y[outer_holdout], candidate_fold)
                        - roc_auc_score(y[outer_holdout], baseline_fold)
                    ),
                    "paired_log_loss_gain": float(
                        log_loss(y[outer_holdout], candidate_fold)
                        - log_loss(y[outer_holdout], baseline_fold)
                    ),
                    "subgroup": _subgroup_report(
                        y[outer_holdout][fold_group],
                        {
                            "baseline": baseline_group,
                            "specialist": probabilities,
                            "candidate": candidate_group,
                        },
                    ),
                    "specialist_fit": specialist_result,
                }
            )
            metadata["current_fold"]["status"] = "completed"
            metadata["current_fold"]["completed_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            save_json(metadata_path, metadata)
            del probabilities, membership, specialist_result
            gc.collect()
        specialist_oof, assigned_folds = assign_specialist_oof(
            len(train), specialist_records
        )
        if not np.array_equal(specialist_oof[group], specialist_full[group]):
            raise RuntimeError("specialist OOF assignment changed during assembly")
        if not np.array_equal(assigned_folds[group], specialist_fold_ids[group]):
            raise RuntimeError("specialist OOF fold assignment changed during assembly")
        if np.count_nonzero(assigned_folds[group] > 0) != GROUP_ROW_COUNT:
            raise RuntimeError("specialist OOF coverage is incomplete")
        if np.any(assigned_folds[~group] != -1):
            raise RuntimeError("specialist OOF assigned a non-group row")
        candidate = route_candidate(prior["baseline"], specialist_oof, group)
        if not np.array_equal(candidate[~group], prior["baseline"][~group]):
            raise RuntimeError("non-group baseline predictions changed")
        candidate_oof_path = output_dir / "candidate_oof.npz"
        np.savez_compressed(
            candidate_oof_path,
            row_index=prior["row_index"],
            id=prior["ids"],
            fold_id=prior["fold_ids"],
            group_mask=group,
            specialist_fold_id=assigned_folds,
            baseline_probability=prior["baseline"],
            specialist_probability=specialist_oof,
            candidate_probability=candidate,
        )
        with np.load(candidate_oof_path) as saved:
            if not np.array_equal(saved["row_index"], prior["row_index"]):
                raise RuntimeError("candidate OOF row alignment failed")
            if not np.array_equal(saved["id"], prior["ids"]):
                raise RuntimeError("candidate OOF ID alignment failed")
            validate_probabilities(saved["candidate_probability"], len(train))
        metadata["summary"] = _summary(
            y,
            prior["baseline"],
            candidate,
            specialist_oof,
            group,
            metadata["fold_results"],
            assigned_folds,
        )
        metadata["selection"] = specialist_gate(
            metadata["summary"]["mean_paired_auc_gain"],
            metadata["summary"]["positive_fold_count"],
            metadata["summary"]["pooled_global"]["roc_auc_gain"],
        )
        metadata["candidate_oof_path"] = str(candidate_oof_path)
        metadata["status"] = "completed"
        metadata["current_fold"] = None
        del X_cat, train, test
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, EcoBusinessSpecialistTimeout) else "failed"
        )
        if isinstance(metadata.get("current_fit"), dict):
            metadata["current_fit"]["status"] = "interrupted"
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
                protected_after == protected_before
            )
            if protected_after != protected_before:
                finalization_errors.append(
                    RuntimeError("pre-existing artifact fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        metadata["frozen_baseline_sha256_after"] = _baseline_hashes(artifacts_dir)
        metadata["configuration_sha256_after"] = (
            _hash_json(configuration) if configuration is not None else None
        )
        metadata["feature_spec_sha256_after"] = (
            _hash_json(metadata["feature_spec"])
            if metadata.get("feature_spec") is not None
            else None
        )
        metadata["fold_membership_sha256_after"] = (
            _fold_membership_hash(folds) if folds is not None else None
        )
        for before_key, after_key in (
            ("configuration_sha256_before", "configuration_sha256_after"),
            ("feature_spec_sha256_before", "feature_spec_sha256_after"),
            ("fold_membership_sha256_before", "fold_membership_sha256_after"),
        ):
            before_value = metadata.get(before_key)
            after_value = metadata.get(after_key)
            if before_value is not None and after_value != before_value:
                finalization_errors.append(
                    RuntimeError(f"{before_key} changed during the experiment")
                )
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
            save_json(output_dir / "protected_baseline_sha256.json", protected_before)
            metadata["output_bytes"] = sum(
                path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
            )
            if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
                budget_error = RuntimeError("output directory exceeded 100 MiB budget")
                metadata["status"] = "failed"
                metadata["error"] = {
                    "type": type(budget_error).__name__,
                    "message": str(budget_error),
                }
                if caught_error is None:
                    caught_error = budget_error
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
        description="Run the fixed Eco/business/loyal specialist OOF experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    result = run_experiment(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selection": result["selection"],
                "summary": result["summary"],
                "output_dir": result["output_dir"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
