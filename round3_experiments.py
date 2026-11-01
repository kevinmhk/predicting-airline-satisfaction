"""Run the ten fixed synthetic-first round-three experiment modes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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

import final_sprint_experiments as f
import round2_cached_experiments as r
import round2_feature_experiments as q
import route_pair_light70_export as release
from train import (
    ID_COLUMN,
    MAX_THREADS,
    InputSpec,
    save_json,
    sha256_file,
    validate_probabilities,
)

LIGHT80 = "light80"
AGREEMENT_WEIGHT = "agreement_weight"
UNCERTAINTY_WEIGHT = "uncertainty_weight"
JOINT_PASSENGER = q.JOINT_PASSENGER
SLOW_CONTEXT = "slow_context"
SLOW_HIERARCHICAL = "slow_hierarchical"
HIERARCHICAL63 = "hierarchical63"
CONTEXT_L2_5 = "context_l2_5"
TEACHER_CONSENSUS = "teacher_consensus"
FUSION63 = "fusion63"
MODES = (
    LIGHT80,
    AGREEMENT_WEIGHT,
    UNCERTAINTY_WEIGHT,
    JOINT_PASSENGER,
    SLOW_CONTEXT,
    SLOW_HIERARCHICAL,
    HIERARCHICAL63,
    CONTEXT_L2_5,
    TEACHER_CONSENSUS,
    FUSION63,
)
SPLIT_SEEDS = (42, 2024)
INTERNAL_SEED = 42
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_SPLIT_ROWS = (503737, 55971, 139927)
EXPECTED_BASE_FEATURES = (39, 35, 4)
EXPECTED_ORIGINAL_STUDENT_FEATURES = (43, 39, 4)
EXPECTED_EXTRA_STUDENT_FEATURES = (44, 40, 4)
EXPECTED_FUSION_FEATURES = (46, 42, 4)
LATEST_BASELINE_AUC = {42: 0.9598727586443816, 2024: 0.9600868558567431}
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
VALIDATION_COLUMNS = q.VALIDATION_COLUMNS
TEACHER_CONSENSUS_FEATURE = "flight_distance_teacher_consensus_mean"
FUSION_HIERARCHICAL_FEATURES = (
    f.FLIGHT_DISTANCE_CLASS_HIERARCHICAL,
    f.FLIGHT_DISTANCE_TRAVEL_HIERARCHICAL,
    f.FLIGHT_DISTANCE_CUSTOMER_HIERARCHICAL,
)

MODE_DEFINITIONS: dict[str, dict[str, Any]] = {
    LIGHT80: {
        "cached": True,
        "formula": ".20*catboost + .40*context + .40*hierarchical",
    },
    AGREEMENT_WEIGHT: {
        "cached": True,
        "formula": "w*catboost + (1-w)*(.5*context + .5*hierarchical)",
        "w": "clip(.20 + abs(context - hierarchical), .20, .50)",
        "label_free": True,
    },
    UNCERTAINTY_WEIGHT: {
        "cached": True,
        "formula": "w*catboost + (1-w)*m",
        "m": ".5*(context + hierarchical)",
        "w": ".20 + .30*4*m*(1-m)",
        "label_free": True,
    },
    JOINT_PASSENGER: {
        "cached": False,
        "formula": ".30*catboost + .35*new_context + .35*new_hierarchical",
        "feature": q.CONDITIONAL_FEATURE_NAMES[JOINT_PASSENGER],
    },
    SLOW_CONTEXT: {
        "cached": False,
        "formula": ".30*catboost + .35*new_context + .35*old_hierarchical",
        "overrides": {"learning_rate": 0.02, "n_estimators": 3000},
        "early_stopping_rounds": 200,
    },
    SLOW_HIERARCHICAL: {
        "cached": False,
        "formula": ".30*catboost + .35*old_context + .35*new_hierarchical",
        "overrides": {"learning_rate": 0.02, "n_estimators": 3000},
        "early_stopping_rounds": 200,
    },
    HIERARCHICAL63: {
        "cached": False,
        "formula": ".30*catboost + .35*old_context + .35*new_hierarchical",
        "overrides": {"num_leaves": 63},
    },
    CONTEXT_L2_5: {
        "cached": False,
        "formula": ".30*catboost + .35*new_context + .35*old_hierarchical",
        "overrides": {"reg_lambda": 5.0},
    },
    TEACHER_CONSENSUS: {
        "cached": False,
        "formula": ".30*catboost + .35*new_context + .35*new_hierarchical",
        "feature": TEACHER_CONSENSUS_FEATURE,
        "label_free": True,
    },
    FUSION63: {
        "cached": False,
        "formula": ".30*catboost + .175*old_context + .175*old_hierarchical + "
        ".35*new_fusion",
        "overrides": {"num_leaves": 63},
        "feature": list(FUSION_HIERARCHICAL_FEATURES),
    },
}


class Round3Timeout(TimeoutError):
    """Raised when one bounded round-three run reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise Round3Timeout("round-three experiment exceeded its deadline")


def _feature_spec(spec: InputSpec) -> dict[str, list[str]]:
    return {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }


def _schema(spec: InputSpec) -> tuple[int, int, int]:
    return (
        len(spec.feature_columns),
        len(spec.numeric_columns),
        len(spec.categorical_columns),
    )


def _assert_schema(spec: InputSpec, expected: tuple[int, int, int]) -> None:
    if _schema(spec) != expected:
        raise RuntimeError(f"unexpected feature schema: {_schema(spec)} != {expected}")


def _own_source_hashes() -> dict[str, str]:
    hashes = release._current_code_hashes(Path(__file__).resolve().parent)
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _append_numeric_feature(
    features: pd.DataFrame, spec: InputSpec, name: str, values: np.ndarray
) -> tuple[pd.DataFrame, InputSpec]:
    f._validate_feature_frame(features, spec)
    if name in features.columns:
        raise ValueError(f"feature already exists: {name}")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(features):
        raise ValueError("feature values do not match the feature frame")
    validate_probabilities(array, len(features))
    result = features.copy()
    result[name] = array
    return result, InputSpec(
        spec.feature_columns + (name,),
        spec.numeric_columns + (name,),
        spec.categorical_columns,
    )


def blend_light80(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> np.ndarray:
    return _bounded_blend((0.20, 0.40, 0.40), catboost, context, hierarchical)


def blend_agreement_weight(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    catboost, context, hierarchical = _components(catboost, context, hierarchical)
    weight = np.clip(0.20 + np.abs(context - hierarchical), 0.20, 0.50)
    candidate = weight * catboost + (1.0 - weight) * (
        0.5 * context + 0.5 * hierarchical
    )
    return _validate_blend(candidate), weight


def blend_uncertainty_weight(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    catboost, context, hierarchical = _components(catboost, context, hierarchical)
    mean = 0.5 * (context + hierarchical)
    weight = 0.20 + 0.30 * 4.0 * mean * (1.0 - mean)
    candidate = weight * catboost + (1.0 - weight) * mean
    return _validate_blend(candidate), weight


def blend_light70(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> np.ndarray:
    return _bounded_blend((0.30, 0.35, 0.35), catboost, context, hierarchical)


def _components(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (catboost, context, hierarchical)
    )
    if (
        any(value.ndim != 1 for value in values)
        or len({len(value) for value in values}) != 1
    ):
        raise ValueError("blend components must be one-dimensional and equally sized")
    for value in values:
        validate_probabilities(value, len(value))
    return values[0], values[1], values[2]


def _validate_blend(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or ((result < 0.0) | (result > 1.0)).any():
        raise ValueError("blend probabilities must be finite and bounded")
    return result


def _bounded_blend(
    weights: tuple[float, float, float],
    catboost: np.ndarray,
    context: np.ndarray,
    hierarchical: np.ndarray,
) -> np.ndarray:
    values = _components(catboost, context, hierarchical)
    return _validate_blend(
        weights[0] * values[0] + weights[1] * values[1] + weights[2] * values[2]
    )


def _teacher_consensus(
    base_features: pd.DataFrame, inner_fit: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    teacher = base_features[f.SOURCE_TEACHER_FEATURE].to_numpy(dtype=np.float64)
    distances = [
        f.canonical_distance_key(value) for value in base_features["Flight Distance"]
    ]
    fit_rows = np.asarray(inner_fit, dtype=np.int64)
    table = (
        pd.DataFrame(
            {
                "distance": np.asarray(distances, dtype=object)[fit_rows],
                "teacher": teacher[fit_rows],
            }
        )
        .groupby("distance", sort=True)["teacher"]
        .mean()
    )
    fallback = float(np.mean(teacher[fit_rows]))
    values = (
        pd.Series(distances, dtype=object)
        .map(table)
        .fillna(fallback)
        .to_numpy(dtype=np.float64)
    )
    validate_probabilities(values, len(base_features))
    return values, {
        "feature": TEACHER_CONSENSUS_FEATURE,
        "recipe": "mean frozen source_teacher_probability by canonical Flight Distance",
        "source": f.SOURCE_TEACHER_FEATURE,
        "fit_rows": len(fit_rows),
        "fit_row_fingerprint": f._array_fingerprint(fit_rows),
        "group_count": int(len(table)),
        "global_fallback": fallback,
        "predictor_only": True,
        "uses_labels": False,
        "validation_or_holdout_in_group_table": False,
    }


def _common_preflight(
    data_dir: Path,
    artifacts_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    InputSpec,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    train, y, spec, prior, profile_metadata, _, input_hashes, _ = q._common_preflight(
        data_dir, artifacts_dir, metadata, metadata_path
    )
    source_hashes = _own_source_hashes()
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError("training row count does not match the approved screen")
    profile_path = artifacts_dir / "distance_profile_feature" / "compare_metadata.json"
    winner = release._validate_winner(artifacts_dir, input_hashes, data_dir)
    public_csv = artifacts_dir / "route_pair_light70_export" / release.SUBMISSION_NAME
    if (
        sha256_file(public_csv)
        != "9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa"
    ):
        raise RuntimeError("current submitted baseline fingerprint changed")
    metadata.update(
        {
            "python": sys.version,
            "threads": MAX_THREADS,
            "input_sha256_before": input_hashes,
            "source_sha256_before": source_hashes,
            "feature_spec_original": _feature_spec(spec),
            "split_seeds": list(SPLIT_SEEDS),
            "internal_seed": INTERNAL_SEED,
            "early_stopping": {"metric": "auc", "rounds": "mode-specific"},
            "baseline_reference": {
                "artifact": "round2_pair_light70",
                "column": "candidate_probability",
                "weights": {"catboost": 0.30, "context": 0.35, "hierarchical": 0.35},
                "auc_by_seed": LATEST_BASELINE_AUC,
                "qualification": winner["selection"],
                "metadata_sha256": winner["metadata_sha256"],
                "public_submission_path": str(public_csv),
                "public_submission_sha256": sha256_file(public_csv),
            },
            "reused_artifacts": {
                "source_teacher_metadata": metadata["reused_artifacts"][
                    "source_teacher_metadata"
                ],
                "distance_profile_metadata": sha256_file(profile_path),
                "pair_light70_metadata": winner["metadata_sha256"],
            },
        }
    )
    save_json(metadata_path, metadata)
    return (
        train,
        y,
        spec,
        prior,
        profile_metadata,
        winner,
        input_hashes,
        source_hashes,
    )


def _latest_baseline_frame(
    winner: dict[str, Any], seed: int, train: pd.DataFrame, holdout: np.ndarray
) -> pd.DataFrame:
    records = {int(item["split_seed"]): item for item in winner["validation"]}
    record = records[seed]
    path = Path(record["validation_path"])
    frame = pd.read_csv(path, float_precision="round_trip")
    q._assert_prediction_schema(frame, train, holdout)
    if record.get("validation_sha256") != sha256_file(path):
        raise RuntimeError("pair_light70 validation fingerprint changed")
    return frame


def _cached_components(
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    prior: dict[str, Any],
    input_hashes: dict[str, str],
    winner: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], dict[str, Any]]:
    # The loader verifies cached lineage; its old baseline metrics are ignored.
    legacy_refs = r._validate_incumbent(artifacts_dir, input_hashes)
    assembled, holdout, _old_metrics, lineage = r._load_seed_components(
        r.PAIR_LIGHT60,
        artifacts_dir,
        train,
        y,
        prior,
        input_hashes,
        f._source_hashes(),
        legacy_refs,
        seed,
    )
    latest = _latest_baseline_frame(winner, seed, train, holdout)
    ids = assembled[ID_COLUMN].to_numpy(dtype=np.int64)
    latest_ids = latest[ID_COLUMN].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, latest_ids):
        raise RuntimeError("latest baseline IDs do not match cached components")
    catboost = assembled["catboost_probability"].to_numpy(dtype=np.float64)
    context = assembled["context_probability"].to_numpy(dtype=np.float64)
    hierarchical = assembled["hierarchical_probability"].to_numpy(dtype=np.float64)
    baseline = blend_light70(catboost, context, hierarchical)
    if not np.allclose(latest["candidate_probability"], baseline, rtol=0.0, atol=1e-12):
        raise RuntimeError("latest baseline candidate arithmetic changed")
    metrics = f._metrics(y[holdout], baseline)
    if not np.isclose(
        metrics["roc_auc"], LATEST_BASELINE_AUC[seed], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("latest baseline AUC changed")
    lineage["latest_baseline"] = {
        "path": str(
            Path(winner["validation"][0]["validation_path"]).parent
            / f"validation_pair_light70_seed{seed}.csv"
        ),
        "sha256": next(
            item["validation_sha256"]
            for item in winner["validation"]
            if int(item["split_seed"]) == seed
        ),
        "column": "candidate_probability",
    }
    return assembled, holdout, metrics, lineage


def _native_components(mode: str) -> tuple[str, ...]:
    if mode in (JOINT_PASSENGER, SLOW_CONTEXT, CONTEXT_L2_5, TEACHER_CONSENSUS):
        return (
            ("context", "hierarchical")
            if mode in (JOINT_PASSENGER, TEACHER_CONSENSUS)
            else ("context",)
        )
    if mode in (SLOW_HIERARCHICAL, HIERARCHICAL63):
        return ("hierarchical",)
    if mode == FUSION63:
        return ("fusion",)
    raise ValueError(f"mode has no native components: {mode}")


def _route_feature(
    component: str,
    base_features: pd.DataFrame,
    base_spec: InputSpec,
    y: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    holdout: np.ndarray,
    *,
    extra_values: pd.Series | None = None,
    extra_mode: str | None = None,
    consensus_values: np.ndarray | None = None,
) -> tuple[pd.DataFrame, InputSpec, dict[str, Any]]:
    if component == "context":
        route = f.fit_route_target_encoding(
            base_features,
            base_spec,
            y[inner_fit],
            inner_fit,
            inner_validation,
            holdout,
            include_context=True,
        )
    elif component == "hierarchical":
        route = f.fit_hierarchical_route_encoding(
            base_features,
            base_spec,
            y[inner_fit],
            inner_fit,
            inner_validation,
            holdout,
        )
    else:
        raise ValueError(f"unknown route component: {component}")
    features, spec = route.features, route.spec
    recipe: dict[str, Any] = {"route_encoding": route.metadata}
    if extra_values is not None:
        if extra_mode is None:
            raise ValueError("conditional feature mode is missing")
        features, spec = q.append_conditional_feature(
            features, spec, extra_values, extra_mode
        )
        recipe["conditional_feature"] = q.CONDITIONAL_FEATURE_NAMES[extra_mode]
    if consensus_values is not None:
        features, spec = _append_numeric_feature(
            features, spec, TEACHER_CONSENSUS_FEATURE, consensus_values
        )
        recipe["teacher_consensus"] = TEACHER_CONSENSUS_FEATURE
    return features, spec, recipe


def _fusion_feature(
    base_features: pd.DataFrame,
    base_spec: InputSpec,
    y: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    holdout: np.ndarray,
) -> tuple[pd.DataFrame, InputSpec, dict[str, Any]]:
    context = f.fit_route_target_encoding(
        base_features,
        base_spec,
        y[inner_fit],
        inner_fit,
        inner_validation,
        holdout,
        include_context=True,
    )
    hierarchical = f.fit_hierarchical_route_encoding(
        base_features,
        base_spec,
        y[inner_fit],
        inner_fit,
        inner_validation,
        holdout,
    )
    features = context.features.copy()
    for name in FUSION_HIERARCHICAL_FEATURES:
        features[name] = hierarchical.features[name].to_numpy(dtype=np.float64)
    spec = InputSpec(
        context.spec.feature_columns + FUSION_HIERARCHICAL_FEATURES,
        context.spec.numeric_columns + FUSION_HIERARCHICAL_FEATURES,
        context.spec.categorical_columns,
    )
    _assert_schema(spec, EXPECTED_FUSION_FEATURES)
    return (
        features,
        spec,
        {
            "context_route_encoding": context.metadata,
            "hierarchical_columns": list(FUSION_HIERARCHICAL_FEATURES),
            "excluded_hierarchical_column": f.FLIGHT_DISTANCE_HIERARCHICAL,
            "representation": "original43_context_plus_three_hierarchical_pair_columns",
        },
    )


def _model_for(mode: str, component: str) -> tuple[Any, dict[str, Any]]:
    base_mode = (
        f.ROUTE_CONTEXT_TARGET_127
        if component in ("context", "fusion")
        else f.ROUTE_HIERARCHICAL
    )
    model = f.build_trial_classifier(base_mode)
    overrides = dict(MODE_DEFINITIONS[mode].get("overrides", {}))
    model.set_params(**overrides)
    return model, overrides


def _fit_native(
    mode: str,
    component: str,
    features: pd.DataFrame,
    spec: InputSpec,
    y: np.ndarray,
    holdout: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    metadata: dict[str, Any],
    metadata_path: Path,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, overrides = _model_for(mode, component)
    patience = int(MODE_DEFINITIONS[mode].get("early_stopping_rounds", 100))
    metadata["current_fit"] = {
        "split_seed": seed,
        "component": component,
        "mode": mode,
        "status": "running",
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "outer_holdout_rows": len(holdout),
        "outer_holdout_used_for_training_or_early_stopping": False,
        "overrides": overrides,
    }
    save_json(metadata_path, metadata)
    callbacks = f._fit_callbacks()
    if patience != 100:
        callbacks = [
            lgb.early_stopping(patience, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ]
    callbacks.append(f._progress_callback(metadata, metadata_path, seed))
    levels, fit_seconds, details = f._fit_with_inner_validation(
        model,
        features,
        y,
        inner_fit,
        inner_validation,
        spec,
        callbacks=callbacks,
    )
    predictions = np.asarray(
        model.predict_proba(
            f.prepare_lightgbm_frame(features.iloc[holdout], spec, levels)
        )[:, 1],
        dtype=np.float64,
    )
    validate_probabilities(predictions, len(holdout))
    record = {
        "component": component,
        "fit_seconds": round(float(fit_seconds), 3),
        "settings": model.get_params(),
        "best_iteration": details["best_iteration"],
        "actual_iterations": details["actual_iterations"],
        "resources": details["resources"],
        "standalone": f._metrics(y[holdout], predictions),
        "category_levels_from_inner_fit": {
            name: len(values) for name, values in levels.items()
        },
        "early_stopping_rounds": patience,
        "feature_spec": _feature_spec(spec),
        "feature_schema": _schema(spec),
    }
    del model
    gc.collect()
    return predictions, record


def _save_component(
    path: Path, holdout: np.ndarray, ids: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    return q._save_component(path, holdout, ids, predictions)


def _fit_component(
    mode: str,
    component: str,
    output_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    base_features: pd.DataFrame,
    base_spec: InputSpec,
    holdout: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    seed: int,
    extra_values: pd.Series | None,
    extra_mode: str | None,
    consensus_values: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if component == "fusion":
        features, student_spec, recipe = _fusion_feature(
            base_features, base_spec, y, inner_fit, inner_validation, holdout
        )
        expected = EXPECTED_FUSION_FEATURES
    else:
        features, student_spec, recipe = _route_feature(
            component,
            base_features,
            base_spec,
            y,
            inner_fit,
            inner_validation,
            holdout,
            extra_values=extra_values,
            extra_mode=extra_mode,
            consensus_values=consensus_values,
        )
        expected = (
            EXPECTED_EXTRA_STUDENT_FEATURES
            if mode in (JOINT_PASSENGER, TEACHER_CONSENSUS)
            else EXPECTED_ORIGINAL_STUDENT_FEATURES
        )
    _assert_schema(student_spec, expected)
    predictions, fit_record = _fit_native(
        mode,
        component,
        features,
        student_spec,
        y,
        holdout,
        inner_fit,
        inner_validation,
        metadata,
        metadata_path,
        seed,
    )
    fit_record["feature_recipe"] = recipe
    path = output_dir / f"predictions_{component}_seed{seed}.npz"
    fit_record["prediction"] = _save_component(
        path,
        holdout,
        train.iloc[holdout][ID_COLUMN].to_numpy(dtype=np.int64),
        predictions,
    )
    metadata.setdefault("component_fits", {}).setdefault(str(seed), {})[component] = (
        fit_record
    )
    metadata["fits"] = int(metadata.get("fits", 0)) + 1
    metadata["completed_fits"].append(f"{component}_seed_{seed}")
    metadata["training_seconds"] = round(
        float(metadata.get("training_seconds", 0.0)) + float(fit_record["fit_seconds"]),
        3,
    )
    metadata["current_fit"] = None
    save_json(metadata_path, metadata)
    del features, student_spec
    gc.collect()
    return predictions, fit_record, recipe


def _candidate(
    mode: str,
    catboost: np.ndarray,
    old_context: np.ndarray,
    old_hierarchical: np.ndarray,
    predictions: dict[str, np.ndarray],
    recipes: dict[str, Any],
) -> np.ndarray:
    if mode == LIGHT80:
        return blend_light80(catboost, old_context, old_hierarchical)
    if mode == AGREEMENT_WEIGHT:
        result, weights = blend_agreement_weight(
            catboost, old_context, old_hierarchical
        )
        recipes["row_weight"] = {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "label_free": True,
        }
        return result
    if mode == UNCERTAINTY_WEIGHT:
        result, weights = blend_uncertainty_weight(
            catboost, old_context, old_hierarchical
        )
        recipes["row_weight"] = {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "label_free": True,
        }
        return result
    if mode == FUSION63:
        return _validate_blend(
            0.30 * catboost
            + 0.175 * old_context
            + 0.175 * old_hierarchical
            + 0.35 * predictions["fusion"]
        )
    return blend_light70(
        catboost,
        predictions.get("context", old_context),
        predictions.get("hierarchical", old_hierarchical),
    )


def _run_seed(
    mode: str,
    output_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    spec: InputSpec,
    prior: dict[str, Any],
    profile_metadata: dict[str, Any],
    winner: dict[str, Any],
    input_hashes: dict[str, str],
    teacher_cache: dict[str, Any],
    seed: int,
) -> tuple[float, float]:
    _, holdout, inner_fit, inner_validation, split_record = f._load_split_arrays(
        output_dir.parent, train, y, prior, seed
    )
    if (len(inner_fit), len(inner_validation), len(holdout)) != EXPECTED_SPLIT_ROWS:
        raise RuntimeError("recorded split row counts changed")
    assembled, holdout, baseline_metrics, lineage = _cached_components(
        output_dir.parent,
        train,
        y,
        prior,
        input_hashes,
        winner,
        seed,
    )
    catboost = assembled["catboost_probability"].to_numpy(dtype=np.float64)
    old_context = assembled["context_probability"].to_numpy(dtype=np.float64)
    old_hierarchical = assembled["hierarchical_probability"].to_numpy(dtype=np.float64)
    baseline = blend_light70(catboost, old_context, old_hierarchical)
    base_features, base_spec = q._base_features_for_seed(
        train,
        spec,
        teacher_cache,
        profile_metadata,
        output_dir.parent,
        seed,
        inner_fit,
    )
    _assert_schema(base_spec, EXPECTED_BASE_FEATURES)
    component_predictions: dict[str, np.ndarray] = {}
    component_records: dict[str, dict[str, Any]] = {}
    recipes: dict[str, Any] = {}

    if MODE_DEFINITIONS[mode]["cached"]:
        component_predictions = {
            "context": old_context,
            "hierarchical": old_hierarchical,
        }
    else:
        conditional_values: pd.Series | None = None
        conditional_mode: str | None = None
        consensus_values: np.ndarray | None = None
        if mode == JOINT_PASSENGER:
            conditional_values, recipes["conditional_feature"] = (
                q.fit_conditional_feature(
                    base_features,
                    base_spec,
                    y[inner_fit],
                    inner_fit,
                    inner_validation,
                    holdout,
                    q.JOINT_PASSENGER,
                )
            )
            conditional_mode = q.JOINT_PASSENGER
        elif mode == TEACHER_CONSENSUS:
            consensus_values, recipes["teacher_consensus"] = _teacher_consensus(
                base_features, inner_fit
            )
        for component in _native_components(mode):
            predictions, record, recipe = _fit_component(
                mode,
                component,
                output_dir,
                metadata,
                metadata_path,
                train,
                y,
                base_features,
                base_spec,
                holdout,
                inner_fit,
                inner_validation,
                seed,
                conditional_values,
                conditional_mode,
                consensus_values,
            )
            component_predictions[component] = predictions
            component_records[component] = record
            recipes[component] = recipe

    candidate = _candidate(
        mode,
        catboost,
        old_context,
        old_hierarchical,
        component_predictions,
        recipes,
    )

    candidate_metrics = f._metrics(y[holdout], candidate)
    ids = train.iloc[holdout][ID_COLUMN].to_numpy(dtype=np.int64)
    validation = q._validation_frame(
        holdout,
        ids,
        catboost,
        component_predictions.get("context", old_context),
        component_predictions.get("hierarchical", old_hierarchical),
        baseline,
        candidate,
    )
    q._assert_prediction_schema(validation, train, holdout)
    validation_path = output_dir / f"validation_{mode}_seed{seed}.csv"
    validation.to_csv(validation_path, index=False, float_format="%.17g")
    result = {
        "split_seed": seed,
        "split": split_record,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
        - baseline_metrics["roc_auc"],
        "training_seconds": round(
            sum(
                float(item.get("fit_seconds", 0.0))
                for item in component_records.values()
            ),
            3,
        ),
        "fits": len(component_records),
        "validation_predictions_path": str(validation_path),
        "validation_predictions_sha256": sha256_file(validation_path),
        "component_fits": component_records,
        "feature_recipes": recipes,
        "mode_definition": MODE_DEFINITIONS[mode],
        "lineage": lineage,
    }
    metadata["validation_results"].append(result)
    metadata["current_fit"] = None
    save_json(metadata_path, metadata)
    del base_features, validation, assembled
    gc.collect()
    return baseline_metrics["roc_auc"], candidate_metrics["roc_auc"]


def _persist_metadata(path: Path, metadata: dict[str, Any], output_dir: Path) -> None:
    q._persist_metadata(path, metadata, output_dir)


def run_experiment(mode: str, data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run one fixed round-three mode without export or orchestration branches."""
    if mode not in MODES:
        raise ValueError(f"unknown round-three mode: {mode}")
    started = time.perf_counter()
    output_dir = artifacts_dir / f"round3_{mode}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "experiment": mode,
        "predeclared_modes": list(MODES),
        "mode_definition": MODE_DEFINITIONS[mode],
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "completed_fits": [],
        "component_fits": {},
        "fits": 0,
        "training_seconds": 0.0,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "output_dir": str(output_dir),
    }
    save_json(metadata_path, metadata)
    caught: BaseException | None = None
    integrity_errors: list[BaseException] = []
    cleanup_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signals_installed = True
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        protected_before = f._artifact_snapshot(artifacts_dir, output_dir)
        input_before = f._input_hashes(data_dir)
        source_before = _own_source_hashes()
        metadata["protected_artifact_sha256_before"] = protected_before
        save_json(metadata_path, metadata)
        train, y, spec, prior, profile_metadata, winner, checked_inputs, _ = (
            _common_preflight(data_dir, artifacts_dir, metadata, metadata_path)
        )
        if checked_inputs != input_before:
            raise RuntimeError("preflight input fingerprint changed")
        teacher_cache = f._verify_teacher_cache(
            artifacts_dir, train, spec, input_before
        )
        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            baseline, candidate = _run_seed(
                mode,
                output_dir,
                metadata,
                metadata_path,
                train,
                y,
                spec,
                prior,
                profile_metadata,
                winner,
                input_before,
                teacher_cache,
                seed,
            )
            baseline_auc[seed] = baseline
            candidate_auc[seed] = candidate
        metadata["selection"] = f.final_sprint_gate(mode, baseline_auc, candidate_auc)
    except BaseException as error:
        caught = error
        metadata["error"] = _error_record(error)
    finally:
        metadata["current_fit"] = None
        metadata["timed_out"] = isinstance(caught, Round3Timeout)
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                signal.signal(signal.SIGALRM, previous_alarm)
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                cleanup_errors.append(error)
        if input_before is not None:
            try:
                metadata["input_sha256_after"] = f._input_hashes(data_dir)
                if metadata["input_sha256_after"] != input_before:
                    integrity_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if source_before is not None:
            try:
                metadata["source_sha256_after"] = _own_source_hashes()
                if metadata["source_sha256_after"] != source_before:
                    integrity_errors.append(RuntimeError("source fingerprint changed"))
            except BaseException as error:
                integrity_errors.append(error)
        if protected_before is not None:
            try:
                protected_after = f._artifact_snapshot(artifacts_dir, output_dir)
                metadata["protected_artifact_sha256_after"] = protected_after
                metadata["protected_artifacts_unchanged"] = (
                    protected_after == protected_before
                )
                if protected_after != protected_before:
                    integrity_errors.append(
                        RuntimeError("pre-existing artifact changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if cleanup_errors:
            metadata["cleanup_error"] = _error_record(
                RuntimeError("; ".join(str(error) for error in cleanup_errors))
            )
        if integrity_errors:
            integrity_error = RuntimeError(
                "; ".join(str(error) for error in integrity_errors)
            )
            metadata["integrity_error"] = _error_record(integrity_error)
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = integrity_error
                metadata["error"] = _error_record(integrity_error)
        elif cleanup_errors:
            cleanup_error = RuntimeError(
                "; ".join(str(error) for error in cleanup_errors)
            )
            metadata["cleanup_error"] = _error_record(cleanup_error)
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = cleanup_error
                metadata["error"] = _error_record(cleanup_error)
        elif caught is not None:
            metadata["status"] = "failed"
            metadata["selection"] = None
        else:
            metadata["status"] = "completed"
        metadata["elapsed_wall_seconds"] = round(time.perf_counter() - started, 3)
        if metadata["status"] != "completed":
            metadata["selection"] = None
        try:
            _persist_metadata(metadata_path, metadata, output_dir)
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["metadata_persistence_error"] = _error_record(error)
            if caught is None:
                caught = error
                metadata["error"] = _error_record(error)
            try:
                _persist_metadata(metadata_path, metadata, output_dir)
            except BaseException:
                pass
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=MODES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.experiment, args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
