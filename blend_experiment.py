from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Bound numerical libraries before importing NumPy, pandas, or model libraries.
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
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from compare import (
    CATBOOST_DEPTH,
    CATBOOST_EARLY_STOPPING_ROUNDS,
    CATBOOST_INTERNAL_SEED,
    CATBOOST_INTERNAL_VALIDATION_FRACTION,
    CATBOOST_ITERATIONS,
    CATBOOST_L2_REGULARIZATION,
    CATBOOST_LEARNING_RATE,
    CATBOOST_RANDOM_SEED,
    _clean_reload_predictions,
    _fit_catboost,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _artifact_snapshot,
    _fit_callbacks,
    _fit_with_inner_validation,
    _fresh_reload_predictions,
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
)
from lightgbm_experiment import (
    _validation_result as lightgbm_validation_result,
)
from train import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    MAX_THREADS,
    TARGET_COLUMN,
    make_split,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
    validate_submission_file,
)

FRESH_SPLIT_SEED = 31415
BLEND_WEIGHT = 0.5
MINIMUM_AUC_GAIN = 0.0002
OUTPUT_NAME = "blend_50_50"


def aligned_blend_predictions(
    reference_ids: Sequence[Any],
    catboost_ids: Sequence[Any],
    catboost_values: np.ndarray,
    lightgbm_ids: Sequence[Any],
    lightgbm_values: np.ndarray,
) -> pd.DataFrame:
    """Align both components by unique ID and calculate the fixed half blend."""
    reference = pd.Index(reference_ids)
    if not reference.is_unique:
        raise ValueError("reference IDs must be unique")
    frames: list[pd.Series] = []
    for name, ids, values in (
        ("CatBoost", catboost_ids, catboost_values),
        ("LightGBM", lightgbm_ids, lightgbm_values),
    ):
        index = pd.Index(ids)
        values_array = np.asarray(values, dtype=np.float64)
        if len(index) != len(values_array) or not index.is_unique:
            raise ValueError(f"{name} IDs must be unique and match predictions")
        if (
            not reference.difference(index).empty
            or not index.difference(reference).empty
        ):
            raise ValueError(f"{name} IDs do not match reference IDs")
        validate_probabilities(values_array, len(index))
        frames.append(pd.Series(values_array, index=index).reindex(reference))
    if frames[0].isna().any() or frames[1].isna().any():
        raise ValueError("ID alignment produced missing predictions")
    catboost = frames[0].to_numpy()
    lightgbm = frames[1].to_numpy()
    return pd.DataFrame(
        {
            ID_COLUMN: reference.to_numpy(),
            "catboost_probability": catboost,
            "lightgbm_probability": lightgbm,
            "blend_probability": BLEND_WEIGHT * catboost + BLEND_WEIGHT * lightgbm,
        }
    )


def confirmation_gate(
    catboost_auc: float, lightgbm_auc: float, blend_auc: float
) -> bool:
    """Apply the predeclared fresh-split blend gate without tuning weights."""
    return blend_auc - catboost_auc >= MINIMUM_AUC_GAIN and blend_auc > lightgbm_auc


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _dependency_versions() -> dict[str, str]:
    versions = package_versions()
    for name in ("catboost", "lightgbm", "threadpoolctl"):
        versions[name] = version(name)
    return versions


def _settings() -> dict[str, Any]:
    return {
        "catboost": {
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
        },
        "lightgbm": dict(PREDECLARED_CONFIG),
        "blend": {"catboost_weight": BLEND_WEIGHT, "lightgbm_weight": BLEND_WEIGHT},
    }


def _split_record(
    path: Path,
    outer_fit: np.ndarray,
    outer_holdout: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "outer_fit_rows": len(outer_fit),
        "outer_holdout_rows": len(outer_holdout),
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "inner_disjoint_from_outer_holdout": not bool(
            np.intersect1d(inner_fit, outer_holdout).size
            or np.intersect1d(inner_validation, outer_holdout).size
        ),
        "inner_covers_outer_fit": np.array_equal(
            np.sort(np.concatenate((inner_fit, inner_validation))),
            np.sort(outer_fit),
        ),
    }


def _read_saved_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_index", ID_COLUMN, "probability"]:
        raise ValueError(f"saved predictions have unexpected schema: {path}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"saved predictions contain duplicate IDs: {path}")
    validate_probabilities(frame["probability"].to_numpy(), len(frame))
    return frame


def _prior_blend_evidence(
    artifacts_dir: Path, train: pd.DataFrame
) -> list[dict[str, Any]]:
    """Recompute prior two-seed blend scores after explicit ID alignment."""
    evidence = []
    y_by_id = pd.Series(target_array(train), index=train[ID_COLUMN])
    for seed in (42, 2024):
        catboost = _read_saved_predictions(
            artifacts_dir / "challengers" / f"validation_catboost_native_seed{seed}.csv"
        )
        lightgbm = _read_saved_predictions(
            artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
        )
        aligned = aligned_blend_predictions(
            catboost[ID_COLUMN],
            catboost[ID_COLUMN],
            catboost["probability"].to_numpy(),
            lightgbm[ID_COLUMN],
            lightgbm["probability"].to_numpy(),
        )
        y_true = y_by_id.reindex(aligned[ID_COLUMN]).to_numpy()
        if pd.isna(y_true).any():
            raise ValueError(f"saved predictions contain unknown IDs for seed {seed}")
        cat_auc = float(roc_auc_score(y_true, aligned["catboost_probability"]))
        lgb_auc = float(roc_auc_score(y_true, aligned["lightgbm_probability"]))
        blend_auc = float(roc_auc_score(y_true, aligned["blend_probability"]))
        evidence.append(
            {
                "split_seed": seed,
                "catboost_roc_auc": cat_auc,
                "lightgbm_roc_auc": lgb_auc,
                "blend_roc_auc": blend_auc,
                "blend_gain_over_catboost": blend_auc - cat_auc,
                "gain_at_least_0.0002": blend_auc - cat_auc >= MINIMUM_AUC_GAIN,
                "id_alignment": "unique IDs matched and reordered before blending",
            }
        )
    return evidence


def _validate_reused_artifacts(
    artifacts_dir: Path, input_hashes: dict[str, str], spec: Any
) -> list[int]:
    """Validate provenance and configuration before reusing prior artifacts."""
    challengers_dir = artifacts_dir / "challengers"
    cat_metadata = json.loads(
        (challengers_dir / "compare_metadata.json").read_text(encoding="utf-8")
    )
    if cat_metadata.get("input_sha256") != input_hashes:
        raise ValueError(
            "CatBoost metadata input_sha256 does not match current train/test hashes"
        )
    if cat_metadata.get("feature_columns") != list(spec.feature_columns):
        raise ValueError(
            "CatBoost metadata feature_columns do not match current inputs"
        )
    if cat_metadata.get("numeric_columns") != list(spec.numeric_columns):
        raise ValueError(
            "CatBoost metadata numeric_columns do not match current inputs"
        )
    if cat_metadata.get("categorical_columns") != list(spec.categorical_columns):
        raise ValueError(
            "CatBoost metadata categorical_columns do not match current inputs"
        )
    expected_catboost = dict(_settings()["catboost"])
    expected_catboost.update(
        {
            "internal_validation_seed": CATBOOST_INTERNAL_SEED,
            "internal_validation_fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
        }
    )
    if cat_metadata.get("predeclared_configs", {}).get("catboost_native") != (
        expected_catboost
    ):
        raise ValueError(
            "CatBoost metadata catboost_native configuration does not match the "
            "current depth-6 builder"
        )
    if cat_metadata.get("selection", {}).get("selected_candidate") != (
        "catboost_native"
    ):
        raise ValueError("CatBoost metadata selected_candidate must be catboost_native")

    expected_catboost_model = challengers_dir / "final_model_catboost.cbm"
    expected_catboost_spec = challengers_dir / "feature_spec.json"
    cat_final = cat_metadata.get("final")
    if not isinstance(cat_final, dict):
        raise ValueError("CatBoost metadata is missing the reusable native final model")
    if cat_final.get("model") != "catboost_native":
        raise ValueError("CatBoost final artifact is not the native catboost model")
    if (
        Path(str(cat_final.get("model_path", ""))).resolve()
        != expected_catboost_model.resolve()
    ):
        raise ValueError(
            "CatBoost final model path is not the expected native artifact"
        )
    if (
        Path(str(cat_final.get("feature_spec_path", ""))).resolve()
        != expected_catboost_spec.resolve()
    ):
        raise ValueError(
            "CatBoost final feature spec path is not the expected artifact"
        )
    if not expected_catboost_model.is_file() or not expected_catboost_spec.is_file():
        raise ValueError(
            "CatBoost native model and feature spec artifacts are required"
        )
    cat_spec = json.loads(expected_catboost_spec.read_text(encoding="utf-8"))
    for name, expected in (
        ("feature_columns", list(spec.feature_columns)),
        ("numeric_columns", list(spec.numeric_columns)),
        ("categorical_columns", list(spec.categorical_columns)),
    ):
        if cat_spec.get(name) != expected:
            raise ValueError(
                f"CatBoost feature spec {name} does not match current inputs"
            )

    lgb_metadata = json.loads(
        (artifacts_dir / "lightgbm" / "compare_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if lgb_metadata.get("status") != "completed":
        raise ValueError("LightGBM metadata must have completed status for reuse")
    for field in ("input_sha256_before", "input_sha256_after"):
        if lgb_metadata.get(field) != input_hashes:
            raise ValueError(
                f"LightGBM metadata {field} does not match current train/test hashes"
            )
    if lgb_metadata.get("predeclared_config") != PREDECLARED_CONFIG:
        raise ValueError(
            "LightGBM predeclared_config does not match the fixed PREDECLARED_CONFIG"
        )
    metric_policy = lgb_metadata.get("metric_policy", {})
    if (
        metric_policy.get("configured_metric") != "auc"
        or metric_policy.get("early_stopping_metric") != "auc"
        or metric_policy.get("registered_metrics") != ["auc"]
    ):
        raise ValueError("LightGBM metadata must record explicit AUC-only metrics")
    for name, expected in (
        ("feature_columns", list(spec.feature_columns)),
        ("numeric_columns", list(spec.numeric_columns)),
        ("categorical_columns", list(spec.categorical_columns)),
    ):
        if lgb_metadata.get(name) != expected:
            raise ValueError(f"LightGBM metadata {name} does not match current inputs")

    results = lgb_metadata.get("validation_results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError(
            "LightGBM metadata must contain exactly two validation results, "
            "one per seed"
        )
    expected_seeds = {42, 2024}
    seen_seeds: set[int] = set()
    prior_counts: list[int] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("LightGBM validation results must be metadata objects")
        seed = result.get("split_seed")
        if type(seed) is not int or seed in seen_seeds:
            raise ValueError(
                "LightGBM validation results must contain one result for each "
                "seed 42 and 2024"
            )
        seen_seeds.add(seed)
        if result.get("candidate") != "lightgbm_native":
            raise ValueError("LightGBM validation result candidate must be native")
        if result.get("settings") != PREDECLARED_CONFIG:
            raise ValueError(
                "LightGBM validation result settings do not match PREDECLARED_CONFIG"
            )
        if result.get("early_stopping", {}).get("eval_metric") != "auc":
            raise ValueError("LightGBM validation results must use metric auc")
        count = result.get("best_iteration")
        if (
            type(count) is not int
            or not 1 <= count <= PREDECLARED_CONFIG["n_estimators"]
        ):
            raise ValueError(
                "LightGBM best_iteration must be a positive one-based count no greater "
                "than n_estimators"
            )
        prior_counts.append(count)
    if seen_seeds != expected_seeds:
        raise ValueError(
            "LightGBM validation results must contain one result for each seed "
            "42 and 2024"
        )
    return prior_counts


def _terminal_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": metadata.get("status", "failed"),
        "selection": metadata.get("selection"),
        "final": metadata.get("final"),
        "error": metadata.get("error"),
        "total_seconds": metadata.get("total_seconds"),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the fixed fresh-split validation and conditional 50/50 refit."""
    started = time.perf_counter()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    protected_before = _artifact_snapshot(artifacts_dir, output_dir)
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "external_timeout_seconds": 20 * 60,
        "threads": MAX_THREADS,
        "python": sys.version,
        "dependencies": _dependency_versions(),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "blend_experiment.py",
                "train.py",
                "compare.py",
                "lightgbm_experiment.py",
            )
        },
        "settings": _settings(),
        "settings_sha256": hashlib.sha256(
            json.dumps(_settings(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "protected_artifact_sha256_before": protected_before,
        "artifacts_dir": str(output_dir),
        "selection": None,
        "final": None,
    }
    save_json(metadata_path, metadata)
    caught_error: BaseException | None = None
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    input_before: dict[str, str] | None = None
    try:
        input_before = {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        }
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        spec = validate_inputs(train, test)
        y = target_array(train)
        X = train.loc[:, list(spec.feature_columns)]
        X_test = test.loc[:, list(spec.feature_columns)]
        metadata["input_sha256_before"] = input_before
        metadata["feature_columns"] = list(spec.feature_columns)
        metadata["numeric_columns"] = list(spec.numeric_columns)
        metadata["categorical_columns"] = list(spec.categorical_columns)

        outer_fit, outer_holdout = make_split(train, y, FRESH_SPLIT_SEED)
        inner_fit, inner_validation = make_catboost_internal_split(
            outer_fit, y, seed=CATBOOST_INTERNAL_SEED
        )
        split_path = output_dir / "fresh_split_31415.npz"
        np.savez_compressed(
            split_path,
            outer_fit=outer_fit,
            outer_holdout=outer_holdout,
            inner_fit=inner_fit,
            inner_validation=inner_validation,
        )
        metadata["fresh_split"] = {
            "outer_seed": FRESH_SPLIT_SEED,
            "inner_seed": CATBOOST_INTERNAL_SEED,
            "inner_validation_fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
            **_split_record(
                split_path, outer_fit, outer_holdout, inner_fit, inner_validation
            ),
        }
        if not metadata["fresh_split"]["inner_disjoint_from_outer_holdout"]:
            raise RuntimeError("fresh inner split overlaps outer holdout")
        save_json(metadata_path, metadata)

        catboost, cat_predictions, cat_result, cat_internal = _fit_catboost(
            spec,
            X,
            y,
            outer_fit,
            X.iloc[outer_holdout],
            y[outer_holdout],
            split_seed=FRESH_SPLIT_SEED,
            depth=CATBOOST_DEPTH,
            iterations=CATBOOST_ITERATIONS,
            candidate="catboost_depth6",
        )
        if not np.array_equal(
            cat_internal["fit_indices"], inner_fit
        ) or not np.array_equal(cat_internal["validation_indices"], inner_validation):
            raise RuntimeError("CatBoost helper changed the fresh inner split")
        cat_result["best_iteration_index_zero_based"] = cat_result["best_iteration"]
        cat_result["best_iteration_count_one_based"] = (
            int(cat_result["best_iteration"]) + 1
        )
        cat_result["early_stopping_eval_partition"] = (
            "fresh outer-fit inner validation only"
        )
        del catboost
        gc.collect()

        levels = fit_category_levels(X.iloc[inner_fit], spec)
        lgb_model = build_lightgbm_classifier()
        fitted_levels, lgb_fit_seconds, lgb_details = _fit_with_inner_validation(
            lgb_model,
            X,
            y,
            inner_fit,
            inner_validation,
            spec,
            callbacks=_fit_callbacks(),
        )
        registered_metrics = sorted(lgb_model.evals_result_["inner_validation"])
        if registered_metrics != ["auc"]:
            raise RuntimeError(f"unexpected LightGBM metrics: {registered_metrics}")
        lgb_predictions, lgb_result = lightgbm_validation_result(
            lgb_model,
            X,
            y,
            outer_holdout,
            fitted_levels,
            spec,
            split_seed=FRESH_SPLIT_SEED,
            fit_seconds=lgb_fit_seconds,
            fit_details=lgb_details,
            internal_fit=inner_fit,
            internal_validation=inner_validation,
        )
        lgb_result["best_iteration_count_one_based"] = lgb_result["best_iteration"]
        lgb_result["best_iteration_index_zero_based"] = None
        levels = fitted_levels
        validation = aligned_blend_predictions(
            train.iloc[outer_holdout][ID_COLUMN],
            train.iloc[outer_holdout][ID_COLUMN],
            cat_predictions,
            train.iloc[outer_holdout][ID_COLUMN],
            lgb_predictions,
        )
        validation.insert(0, "row_index", outer_holdout)
        validation_path = output_dir / "validation_fresh_seed31415.csv"
        validation.to_csv(validation_path, index=False, float_format="%.17g")
        y_holdout = y[outer_holdout]
        blend_auc = float(roc_auc_score(y_holdout, validation["blend_probability"]))
        blend_log_loss = float(log_loss(y_holdout, validation["blend_probability"]))
        metadata["fresh_validation"] = {
            "prediction_path": str(validation_path),
            "catboost": cat_result,
            "lightgbm": lgb_result,
            "blend": {
                "candidate": "fixed_50_50",
                "roc_auc": blend_auc,
                "log_loss": blend_log_loss,
                "catboost_weight": BLEND_WEIGHT,
                "lightgbm_weight": BLEND_WEIGHT,
            },
        }
        prior_evidence = _prior_blend_evidence(artifacts_dir, train)
        if (
            len(prior_evidence) != 2
            or {item.get("split_seed") for item in prior_evidence} != {42, 2024}
            or any(
                float(item["blend_gain_over_catboost"]) < MINIMUM_AUC_GAIN
                for item in prior_evidence
            )
        ):
            raise ValueError(
                "prior saved-prediction blend evidence does not meet the "
                "0.0002 gain requirement on both seeds"
            )
        metadata["prior_two_split_evidence"] = prior_evidence
        gate_passes = confirmation_gate(
            cat_result["roc_auc"], lgb_result["roc_auc"], blend_auc
        )
        metadata["selection"] = {
            "confirmation_pass": gate_passes,
            "minimum_blend_gain_over_catboost": MINIMUM_AUC_GAIN,
            "strictly_exceeds_lightgbm": blend_auc > lgb_result["roc_auc"],
            "decision": "refit_and_export" if gate_passes else "retain_catboost",
            "policy": (
                "Fresh fixed 50/50 blend must gain at least 0.0002 ROC AUC over "
                "fresh depth-6 CatBoost and strictly exceed fresh LightGBM."
            ),
            "not_statistically_independent": (
                "The fresh seeded holdout overlaps earlier rows and is stability "
                "evidence, not independent confirmation."
            ),
        }
        save_json(metadata_path, metadata)
        del lgb_model, levels, cat_predictions, lgb_predictions, validation
        gc.collect()

        if gate_passes:
            prior_lgb_counts = _validate_reused_artifacts(
                artifacts_dir,
                input_before,
                spec,
            )
            catboost_model_path = (
                artifacts_dir / "challengers" / "final_model_catboost.cbm"
            )
            catboost_feature_spec_path = (
                artifacts_dir / "challengers" / "feature_spec.json"
            )
            catboost_model = CatBoostClassifier()
            catboost_model.load_model(catboost_model_path)
            catboost_test = catboost_model.predict_proba(
                prepare_catboost_frame(X_test, spec)
            )[:, 1]
            prior_submission = validate_submission_file(
                artifacts_dir / "challengers" / "submission.csv", test[ID_COLUMN]
            )
            cat_prior_difference = float(
                np.max(
                    np.abs(
                        catboost_test
                        - prior_submission[TARGET_COLUMN].to_numpy(dtype=np.float64)
                    )
                )
            )
            if cat_prior_difference > 1e-12:
                raise RuntimeError(
                    "reused CatBoost predictions differ from prior submission"
                )

            lgb_counts = prior_lgb_counts + [
                int(lgb_result["best_iteration_count_one_based"])
            ]
            lgb_selected_iterations = int(round(float(np.median(lgb_counts))))
            final_levels = fit_category_levels(X, spec)
            final_lgb = build_lightgbm_classifier(n_estimators=lgb_selected_iterations)
            fit_started = time.perf_counter()
            final_lgb.fit(
                prepare_lightgbm_frame(X, spec, final_levels),
                y,
                categorical_feature=list(spec.categorical_columns),
                callbacks=[lgb.log_evaluation(0)],
            )
            final_lgb_fit_seconds = time.perf_counter() - fit_started
            lgb_test = final_lgb.predict_proba(
                prepare_lightgbm_frame(X_test, spec, final_levels)
            )[:, 1]
            validate_probabilities(lgb_test, len(X_test))
            lgb_model_path = output_dir / "final_model_lightgbm.txt"
            final_lgb.booster_.save_model(lgb_model_path)
            lgb_feature_spec_path = output_dir / "feature_spec.json"
            save_json(
                lgb_feature_spec_path,
                {
                    "feature_columns": list(spec.feature_columns),
                    "numeric_columns": list(spec.numeric_columns),
                    "categorical_columns": list(spec.categorical_columns),
                    "category_levels": final_levels,
                },
            )
            final_aligned = aligned_blend_predictions(
                test[ID_COLUMN],
                test[ID_COLUMN],
                catboost_test,
                test[ID_COLUMN],
                lgb_test,
            )
            final_submission_path = (
                output_dir / "submission_catboost_lightgbm_50_50.csv"
            )
            pd.DataFrame(
                {
                    ID_COLUMN: test[ID_COLUMN],
                    TARGET_COLUMN: final_aligned["blend_probability"],
                }
            ).to_csv(final_submission_path, index=False, float_format="%.17g")
            final_submission = validate_submission_file(
                final_submission_path, test[ID_COLUMN]
            )
            reloaded_cat = _clean_reload_predictions(
                catboost_model_path,
                "catboost",
                prepare_catboost_frame(X_test, spec),
            )
            reloaded_lgb = _fresh_reload_predictions(
                lgb_model_path, lgb_feature_spec_path, X_test
            )
            reloaded_blend = BLEND_WEIGHT * reloaded_cat + BLEND_WEIGHT * reloaded_lgb
            reload_component_difference = max(
                float(np.max(np.abs(catboost_test - reloaded_cat))),
                float(np.max(np.abs(lgb_test - reloaded_lgb))),
            )
            csv_blend_difference = float(
                np.max(
                    np.abs(
                        reloaded_blend
                        - final_submission[TARGET_COLUMN].to_numpy(dtype=np.float64)
                    )
                )
            )
            if reload_component_difference > 1e-12 or csv_blend_difference > 1e-12:
                raise RuntimeError("clean reload or final CSV blend differs")
            metadata["final"] = {
                "model": "fixed_catboost_lightgbm_50_50",
                "catboost_model_path": str(catboost_model_path),
                "catboost_feature_spec_path": str(catboost_feature_spec_path),
                "catboost_prior_submission_path": str(
                    artifacts_dir / "challengers" / "submission.csv"
                ),
                "catboost_prior_submission_max_difference": cat_prior_difference,
                "lightgbm_model_path": str(lgb_model_path),
                "lightgbm_feature_spec_path": str(lgb_feature_spec_path),
                "lightgbm_refit_iteration_policy": (
                    "median one-based LightGBM best_iteration counts from prior seeds "
                    f"{lgb_counts[:-1]} plus fresh seed 31415 count {lgb_counts[-1]}"
                ),
                "lightgbm_selected_iterations": lgb_selected_iterations,
                "lightgbm_early_stopping": False,
                "lightgbm_eval_set": None,
                "lightgbm_fit_seconds": round(final_lgb_fit_seconds, 3),
                "submission_path": str(final_submission_path),
                "submission_rows": len(final_submission),
                "weights": {"catboost": BLEND_WEIGHT, "lightgbm": BLEND_WEIGHT},
                "clean_interpreter_reload": {
                    "rows": len(final_submission),
                    "max_component_difference": reload_component_difference,
                    "max_blend_csv_difference": csv_blend_difference,
                },
                "model_sha256": sha256_file(lgb_model_path),
                "feature_spec_sha256": sha256_file(lgb_feature_spec_path),
            }
            del catboost_model, final_lgb, catboost_test, lgb_test
            gc.collect()
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = "failed"
        metadata["error"] = _error_record(error)
    finally:
        try:
            metadata["input_sha256_after"] = {
                "train.csv": sha256_file(train_path),
                "test.csv": sha256_file(test_path),
            }
            metadata["protected_artifact_sha256_after"] = _artifact_snapshot(
                artifacts_dir, output_dir
            )
            metadata["protected_artifacts_unchanged"] = (
                metadata["protected_artifact_sha256_before"]
                == metadata["protected_artifact_sha256_after"]
            )
            integrity_failures: list[str] = []
            if (
                input_before is not None
                and metadata["input_sha256_after"] != input_before
            ):
                integrity_failures.append("input CSV fingerprint changed")
            if not metadata["protected_artifacts_unchanged"]:
                integrity_failures.append("pre-existing artifact changed")
            if integrity_failures:
                integrity_error = RuntimeError(
                    "; ".join(integrity_failures) + " during experiment"
                )
                metadata["status"] = "failed"
                metadata["selection"] = None
                metadata["final"] = None
                metadata["integrity_error"] = _error_record(integrity_error)
                if caught_error is None:
                    metadata["error"] = _error_record(integrity_error)
                    caught_error = integrity_error
            metadata["total_seconds"] = round(time.perf_counter() - started, 3)
            save_json(
                output_dir / "protected_baseline_sha256.json",
                metadata["protected_artifact_sha256_after"],
            )
            save_json(metadata_path, metadata)
        except BaseException as finalization_error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["final"] = None
            metadata["integrity_error"] = _error_record(finalization_error)
            if caught_error is None:
                metadata["error"] = _error_record(finalization_error)
            metadata["total_seconds"] = round(time.perf_counter() - started, 3)
            if caught_error is None:
                caught_error = finalization_error
            try:
                save_json(metadata_path, metadata)
            except Exception as persistence_error:
                print(
                    f"Could not persist failure metadata: {persistence_error}",
                    file=sys.stderr,
                )
    if caught_error is not None:
        raise caught_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed fresh-split CatBoost/LightGBM 50/50 experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        metadata = run_experiment(args.data_dir, args.artifacts_dir)
    except BaseException as error:
        print(
            json.dumps(
                _terminal_summary(
                    {
                        "status": "failed",
                        "selection": None,
                        "final": None,
                        "error": _error_record(error),
                        "total_seconds": None,
                    }
                ),
                sort_keys=True,
            )
        )
        raise
    print(json.dumps(_terminal_summary(metadata), sort_keys=True))


if __name__ == "__main__":
    main()
