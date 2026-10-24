"""Compare slower-learning LightGBM and CatBoost components on fixed 50/50 blends."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from blend_experiment import aligned_blend_predictions
from compare import (
    _clean_reload_predictions,
    _load_recorded_split,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import (
    _artifact_snapshot,
    _fresh_reload_predictions,
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
)
from train import (
    ID_COLUMN,
    TARGET_COLUMN,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
    validate_submission_file,
)

MINIMUM_AUC_GAIN = 0.0002
SEEDS = (42, 2024)
OUTPUT_NAME = "slower_learning"
OUTPUT_BUDGET_BYTES = 300 * 1024 * 1024
EXPERIMENT_TIMEOUT_SECONDS = 90 * 60
SLOW_LIGHTGBM: dict[str, Any] = {
    "learning_rate": 0.02,
    "n_estimators": 3000,
    "patience": 200,
}
SLOW_CATBOOST: dict[str, Any] = {
    "learning_rate": 0.03,
    "iterations": 6000,
    "patience": 200,
}
BASELINE_AUC = {42: 0.957713923393243, 2024: 0.9580214315945633}
CANDIDATES = (
    ("original_cat_slow_light", "catboost_native", "slow_lightgbm"),
    ("slow_cat_original_light", "slow_catboost", "lightgbm_native"),
    ("slow_cat_slow_light", "slow_catboost", "slow_lightgbm"),
)
SPLITS = {42: "outer_split.npz", 2024: "confirmation_split.npz"}


class SlowerLearningTimeout(TimeoutError):
    """Raised when the bounded slower-learning run reaches its deadline."""


def _on_timeout(signum: int, frame: Any) -> None:
    raise SlowerLearningTimeout("slower-learning experiment exceeded its deadline")


def build_slow_catboost() -> CatBoostClassifier:
    """Build CatBoost with only the learning rate, cap, and patience changed."""
    return CatBoostClassifier(
        iterations=SLOW_CATBOOST["iterations"],
        depth=6,
        learning_rate=SLOW_CATBOOST["learning_rate"],
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=3,
        random_seed=42,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
        early_stopping_rounds=SLOW_CATBOOST["patience"],
    )


def build_slow_lightgbm() -> lgb.LGBMClassifier:
    """Build LightGBM with only the learning rate and tree cap changed."""
    return build_lightgbm_classifier(
        learning_rate=SLOW_LIGHTGBM["learning_rate"],
        n_estimators=SLOW_LIGHTGBM["n_estimators"],
    )


def select_candidate(
    per_seed: dict[int, dict[str, float]], baseline: dict[int, float]
) -> dict[str, Any]:
    """Require the gain on both seeds, then keep the best worst-seed gain."""
    comparison: list[dict[str, Any]] = []
    for name, _cat, _light in CANDIDATES:
        gains = {seed: per_seed[seed][name] - baseline[seed] for seed in SEEDS}
        comparison.append(
            {
                "candidate": name,
                "gains": {str(seed): gain for seed, gain in gains.items()},
                "worst_seed_gain": min(gains.values()),
                "qualifies": all(gain >= MINIMUM_AUC_GAIN for gain in gains.values()),
            }
        )
    passing = [row for row in comparison if row["qualifies"]]
    winner = (
        max(passing, key=lambda row: row["worst_seed_gain"])["candidate"]
        if passing
        else None
    )
    return {
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "comparison": comparison,
        "selected_candidate": winner,
        "decision": "confirm_fresh_split" if winner else "retain_current_blend",
    }


def _read_saved(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_index", ID_COLUMN, "probability"]:
        raise ValueError(f"unexpected saved prediction schema: {path}")
    validate_probabilities(frame["probability"].to_numpy(), len(frame))
    return frame


def _save_predictions(
    path: Path, row_index: np.ndarray, ids: np.ndarray, values: np.ndarray
) -> None:
    validate_probabilities(values, len(values))
    pd.DataFrame(
        {"row_index": row_index, ID_COLUMN: ids, "probability": values}
    ).to_csv(path, index=False)


def _fit_components(
    X: pd.DataFrame,
    y: np.ndarray,
    spec: Any,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, Any]:
    """Fit both slow components on one split and return holdout predictions."""
    levels = fit_category_levels(X.iloc[inner_fit], spec)
    light = build_slow_lightgbm()
    started = time.perf_counter()
    light.fit(
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
        callbacks=[
            lgb.early_stopping(SLOW_LIGHTGBM["patience"], verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    light_seconds = time.perf_counter() - started
    light_values = light.predict_proba(
        prepare_lightgbm_frame(X.iloc[holdout], spec, levels)
    )[:, 1]
    light_result = {
        "best_iteration": int(light.best_iteration_),
        "actual_iterations": int(light.booster_.current_iteration()),
        "fit_seconds": round(light_seconds, 3),
    }
    del light, levels
    gc.collect()

    cat = build_slow_catboost()
    X_cat = prepare_catboost_frame(X, spec)
    started = time.perf_counter()
    cat.fit(
        X_cat.iloc[inner_fit],
        y[inner_fit],
        eval_set=(X_cat.iloc[inner_validation], y[inner_validation]),
        cat_features=list(spec.categorical_columns),
        verbose=False,
    )
    cat_seconds = time.perf_counter() - started
    cat_values = cat.predict_proba(X_cat.iloc[holdout])[:, 1]
    cat_result = {
        "best_iteration": int(cat.get_best_iteration()),
        "actual_iterations": int(cat.tree_count_),
        "fit_seconds": round(cat_seconds, 3),
    }
    del cat, X_cat
    gc.collect()
    return {
        "slow_lightgbm": (light_values, light_result),
        "slow_catboost": (cat_values, cat_result),
    }


def run(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    output_dir = artifacts_dir / OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = _artifact_snapshot(artifacts_dir, output_dir)
    metadata: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": [name for name, _c, _l in CANDIDATES],
        "slow_lightgbm": SLOW_LIGHTGBM,
        "slow_catboost": SLOW_CATBOOST,
        "selection": None,
        "validation_results": [],
        "protected_artifact_sha256_before": protected_before,
    }
    metadata_path = output_dir / "compare_metadata.json"
    save_json(metadata_path, metadata)
    previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(EXPERIMENT_TIMEOUT_SECONDS)
    caught: BaseException | None = None
    try:
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        ids = train[ID_COLUMN].to_numpy()
        metadata["input_sha256_before"] = {
            "train.csv": sha256_file(data_dir / "train.csv"),
            "test.csv": sha256_file(data_dir / "test.csv"),
        }
        metadata["source_sha256"] = {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "slower_learning.py",
                "train.py",
                "compare.py",
                "lightgbm_experiment.py",
                "blend_experiment.py",
            )
        }
        metadata["dependencies"] = package_versions()

        per_seed: dict[int, dict[str, float]] = {}
        for seed in SEEDS:
            outer_fit, outer_holdout = _load_recorded_split(
                artifacts_dir / SPLITS[seed], train, y, seed
            )
            inner_fit, inner_validation = make_catboost_internal_split(
                outer_fit, y, seed=42
            )
            if np.intersect1d(inner_validation, outer_holdout).size:
                raise RuntimeError("inner validation overlaps the outer holdout")
            holdout_ids = ids[outer_holdout]
            fitted = _fit_components(
                train, y, spec, inner_fit, inner_validation, outer_holdout
            )
            components = {
                "catboost_native": _read_saved(
                    artifacts_dir
                    / "challengers"
                    / f"validation_catboost_native_seed{seed}.csv"
                ),
                "lightgbm_native": _read_saved(
                    artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
                ),
            }
            for name, (values, result) in fitted.items():
                _save_predictions(
                    output_dir / f"validation_{name}_seed{seed}.csv",
                    outer_holdout,
                    holdout_ids,
                    values,
                )
                components[name] = pd.DataFrame(
                    {
                        "row_index": outer_holdout,
                        ID_COLUMN: holdout_ids,
                        "probability": values,
                    }
                )
                result["roc_auc"] = float(roc_auc_score(y[outer_holdout], values))
                result["log_loss"] = float(log_loss(y[outer_holdout], values))
            scores: dict[str, float] = {}
            candidates: dict[str, Any] = {}
            for name, cat_name, light_name in CANDIDATES:
                aligned = aligned_blend_predictions(
                    holdout_ids,
                    components[cat_name][ID_COLUMN],
                    components[cat_name]["probability"].to_numpy(),
                    components[light_name][ID_COLUMN],
                    components[light_name]["probability"].to_numpy(),
                )
                values = aligned["blend_probability"].to_numpy()
                scores[name] = float(roc_auc_score(y[outer_holdout], values))
                candidates[name] = {
                    "roc_auc": scores[name],
                    "log_loss": float(log_loss(y[outer_holdout], values)),
                }
            baseline_blend = aligned_blend_predictions(
                holdout_ids,
                components["catboost_native"][ID_COLUMN],
                components["catboost_native"]["probability"].to_numpy(),
                components["lightgbm_native"][ID_COLUMN],
                components["lightgbm_native"]["probability"].to_numpy(),
            )
            baseline = float(
                roc_auc_score(y[outer_holdout], baseline_blend["blend_probability"])
            )
            if abs(baseline - BASELINE_AUC[seed]) > 1e-9:
                raise RuntimeError(f"recomputed baseline blend changed for seed {seed}")
            per_seed[seed] = scores
            metadata["validation_results"].append(
                {
                    "split_seed": seed,
                    "rows": int(len(outer_holdout)),
                    "baseline_blend_roc_auc": baseline,
                    "slow_lightgbm": fitted["slow_lightgbm"][1],
                    "slow_catboost": fitted["slow_catboost"][1],
                    "candidates": candidates,
                }
            )
            save_json(metadata_path, metadata)
            del fitted, components, baseline_blend
            gc.collect()

        metadata["selection"] = select_candidate(per_seed, BASELINE_AUC)
        save_json(metadata_path, metadata)
    except BaseException as error:
        caught = error
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        metadata["status"] = "failed" if caught else "completed"
        if caught:
            metadata["error"] = {"type": type(caught).__name__, "message": str(caught)}
        metadata["protected_artifact_sha256_after"] = _artifact_snapshot(
            artifacts_dir, output_dir
        )
        metadata["protected_artifacts_unchanged"] = (
            metadata["protected_artifact_sha256_after"]
            == metadata["protected_artifact_sha256_before"]
        )
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        metadata["output_budget_bytes"] = OUTPUT_BUDGET_BYTES
        save_json(metadata_path, metadata)
        if not metadata["protected_artifacts_unchanged"]:
            raise RuntimeError("protected artifacts changed during the experiment")
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            raise RuntimeError("output directory exceeded the predeclared budget")
    return metadata


def export_submission(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Refit the slow CatBoost winner over all rows and export the fixed blend."""
    output_dir = artifacts_dir / OUTPUT_NAME
    metadata_path = output_dir / "compare_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "completed":
        raise RuntimeError("only a completed slower-learning run can be exported")
    counts = [
        int(result["slow_catboost"]["best_iteration"]) + 1
        for result in metadata["validation_results"]
    ]
    iterations = int(round(float(np.median(counts))))
    cat_model_path = output_dir / "final_model_catboost_slow.cbm"
    prior_fit = metadata.get("export_fit") or {}
    reuse_fit = (
        prior_fit.get("selected_iterations") == iterations
        and cat_model_path.is_file()
        and sha256_file(cat_model_path) == prior_fit.get("catboost_model_sha256")
    )

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    spec = validate_inputs(train, test)
    y = target_array(train)
    X_cat = prepare_catboost_frame(train, spec)

    if reuse_fit:
        cat = CatBoostClassifier()
        cat.load_model(cat_model_path)
        fit_seconds = float(prior_fit.get("catboost_fit_seconds") or 0.0)
    else:
        cat = CatBoostClassifier(
            iterations=iterations,
            depth=6,
            learning_rate=SLOW_CATBOOST["learning_rate"],
            loss_function="Logloss",
            l2_leaf_reg=3,
            random_seed=42,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
        )
        started = time.perf_counter()
        cat.fit(X_cat, y, cat_features=list(spec.categorical_columns), verbose=False)
        fit_seconds = time.perf_counter() - started
        cat.save_model(cat_model_path)
        metadata["export_fit"] = {
            "catboost_iteration_counts": counts,
            "selected_iterations": iterations,
            "catboost_fit_seconds": round(fit_seconds, 3) if not reuse_fit else None,
            "catboost_model_sha256": sha256_file(cat_model_path),
        }
        save_json(metadata_path, metadata)
    cat_test = cat.predict_proba(prepare_catboost_frame(test, spec))[:, 1]
    validate_probabilities(cat_test, len(test))
    del cat, X_cat
    gc.collect()

    lgb_dir = artifacts_dir / "blend_50_50"
    lgb_model_path = lgb_dir / "final_model_lightgbm.txt"
    lgb_spec_path = lgb_dir / "feature_spec.json"
    lgb_spec = json.loads(lgb_spec_path.read_text(encoding="utf-8"))
    if list(lgb_spec["feature_columns"]) != list(spec.feature_columns):
        raise ValueError("reused LightGBM feature specification changed")
    lgb_model = lgb.Booster(model_file=str(lgb_model_path))
    light_test = np.asarray(
        lgb_model.predict(
            prepare_lightgbm_frame(test, spec, lgb_spec["category_levels"])
        ),
        dtype=np.float64,
    )
    validate_probabilities(light_test, len(test))

    aligned = aligned_blend_predictions(
        test[ID_COLUMN], test[ID_COLUMN], cat_test, test[ID_COLUMN], light_test
    )
    submission_path = output_dir / "submission_slowcat_lightgbm_50_50.csv"
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN], TARGET_COLUMN: aligned["blend_probability"]}
    ).to_csv(submission_path, index=False, float_format="%.17g")
    submitted = validate_submission_file(submission_path, test[ID_COLUMN])

    reloaded_cat = _clean_reload_predictions(
        cat_model_path, "catboost", prepare_catboost_frame(test, spec)
    )
    reloaded_light = _fresh_reload_predictions(
        lgb_model_path, lgb_spec_path, test.loc[:, list(spec.feature_columns)]
    )
    reloaded_blend = 0.5 * reloaded_cat + 0.5 * reloaded_light
    differences = {
        "catboost_max_abs": float(np.max(np.abs(cat_test - reloaded_cat))),
        "lightgbm_max_abs": float(np.max(np.abs(light_test - reloaded_light))),
        "blend_csv_max_abs": float(
            np.max(np.abs(submitted[TARGET_COLUMN].to_numpy() - reloaded_blend))
        ),
    }
    if max(differences.values()) > 1e-12:
        raise RuntimeError(f"reload verification failed: {differences}")

    metadata["final"] = {
        "candidate": "slow_cat_original_light",
        "gate_decision": metadata["selection"]["decision"],
        "export_authorized_by_user": True,
        "catboost_iteration_counts": counts,
        "selected_iterations": iterations,
        "catboost_fit_seconds": round(fit_seconds, 3) if not reuse_fit else None,
        "catboost_model_path": str(cat_model_path),
        "catboost_model_sha256": sha256_file(cat_model_path),
        "lightgbm_model_path": str(lgb_model_path),
        "lightgbm_model_sha256": sha256_file(lgb_model_path),
        "submission_path": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "submission_rows": int(len(submitted)),
        "reload_differences": differences,
    }
    metadata["output_bytes"] = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    save_json(metadata_path, metadata)
    return metadata["final"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.export:
        print(
            json.dumps(export_submission(args.data_dir, args.artifacts_dir), indent=2)
        )
        return
    result = run(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {"status": result["status"], "selection": result["selection"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
