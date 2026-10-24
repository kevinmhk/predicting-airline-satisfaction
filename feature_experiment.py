from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import blend_experiment
from compare import (
    CATBOOST_DEPTH,
    CATBOOST_INTERNAL_SEED,
    CATBOOST_ITERATIONS,
    _fit_catboost,
    _load_recorded_split,
    _resource_delta,
    _resource_snapshot,
    build_catboost_classifier,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import (
    _artifact_snapshot,
    _fit_callbacks,
    _fit_with_inner_validation,
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
)
from lightgbm_experiment import _validation_result as lightgbm_validation_result
from train import (
    ID_COLUMN,
    MAX_THREADS,
    TARGET_COLUMN,
    InputSpec,
    make_split,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
    validate_submission_file,
)

RATING_COLUMNS = (
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
)
DEPARTURE_DELAY_COLUMN = "Departure Delay in Minutes"
ARRIVAL_DELAY_COLUMN = "Arrival Delay in Minutes"
ENGINEERED_COLUMNS = (
    "rating_mean",
    "low_rating_count",
    "departure_delayed",
    "arrival_delayed",
    "arrival_delay_missing",
)
FRESH_SPLIT_SEED = 31415
MINIMUM_AUC_GAIN = 0.0002
BLEND_WEIGHT = 0.5
OUTPUT_NAME = "feature_pass"
EXPERIMENT_TIMEOUT_SECONDS = 25 * 60
SCRATCH_DIR = Path("/home/kevinmhk/.hermes/profiles/coder/cache/scratch")
EXPECTED_PRIOR_BLEND_AUC = {
    42: 0.957713923393243,
    2024: 0.9580214315945633,
}
EXPECTED_FRESH_BLEND_AUC = 0.9588362995285812


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the five fixed row-wise numeric predictors appended."""
    required = set(RATING_COLUMNS) | {DEPARTURE_DELAY_COLUMN, ARRIVAL_DELAY_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"feature engineering missing required columns: {sorted(missing)}"
        )
    collisions = set(ENGINEERED_COLUMNS) & set(frame.columns)
    if collisions:
        raise ValueError(
            "feature engineering columns collide with input columns: "
            f"{sorted(collisions)}"
        )

    engineered = frame.copy()
    ratings = frame.loc[:, list(RATING_COLUMNS)]
    engineered["rating_mean"] = ratings.mean(axis=1)
    engineered["low_rating_count"] = ratings.le(2).sum(axis=1).astype(np.int64)
    for source, target in (
        (DEPARTURE_DELAY_COLUMN, "departure_delayed"),
        (ARRIVAL_DELAY_COLUMN, "arrival_delayed"),
    ):
        values = frame[source]
        engineered[target] = values.gt(0).astype(np.float64).where(values.notna())
    engineered["arrival_delay_missing"] = (
        frame[ARRIVAL_DELAY_COLUMN].isna().astype(np.int8)
    )
    return engineered


def engineered_gate(
    baseline_auc_by_seed: dict[int, float], engineered_auc_by_seed: dict[int, float]
) -> dict[str, Any]:
    """Require the fixed minimum AUC gain on both recorded seeds."""
    comparison = []
    for seed in (42, 2024):
        if seed not in baseline_auc_by_seed or seed not in engineered_auc_by_seed:
            continue
        gain = engineered_auc_by_seed[seed] - baseline_auc_by_seed[seed]
        comparison.append(
            {
                "split_seed": seed,
                "baseline_roc_auc": baseline_auc_by_seed[seed],
                "engineered_roc_auc": engineered_auc_by_seed[seed],
                "auc_gain": gain,
                "qualifies": gain >= MINIMUM_AUC_GAIN,
            }
        )
    qualifies = len(comparison) == 2 and all(item["qualifies"] for item in comparison)
    return {
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "comparison": comparison,
        "qualifies": qualifies,
        "decision": "continue_fresh_confirmation"
        if qualifies
        else "retain_unengineered_blend",
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _dependency_versions() -> dict[str, str]:
    dependencies = package_versions()
    for name in ("catboost", "lightgbm", "threadpoolctl"):
        dependencies[name] = version(name)
    return dependencies


def _source_hashes() -> dict[str, str]:
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "feature_experiment.py",
            "train.py",
            "compare.py",
            "lightgbm_experiment.py",
            "blend_experiment.py",
        )
    }


def _split_reference(
    path: Path,
    split_seed: int,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    internal_fit: np.ndarray,
    internal_validation: np.ndarray,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "outer_seed": split_seed,
        "inner_seed": CATBOOST_INTERNAL_SEED,
        "outer_fit_rows": len(fit_indices),
        "outer_holdout_rows": len(holdout_indices),
        "inner_fit_rows": len(internal_fit),
        "inner_validation_rows": len(internal_validation),
        "inner_covers_outer_fit": np.array_equal(
            np.sort(np.concatenate((internal_fit, internal_validation))),
            np.sort(fit_indices),
        ),
        "inner_disjoint_from_outer_holdout": not bool(
            np.intersect1d(internal_fit, holdout_indices).size
            or np.intersect1d(internal_validation, holdout_indices).size
        ),
    }


def _load_fresh_split(
    path: Path, train: pd.DataFrame, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing recorded fresh split: {path}")
    with np.load(path) as recorded:
        outer_fit = np.asarray(recorded["outer_fit"], dtype=np.int64)
        outer_holdout = np.asarray(recorded["outer_holdout"], dtype=np.int64)
        inner_fit = np.asarray(recorded["inner_fit"], dtype=np.int64)
        inner_validation = np.asarray(recorded["inner_validation"], dtype=np.int64)
    expected_outer_fit, expected_outer_holdout = make_split(train, y, FRESH_SPLIT_SEED)
    expected_inner_fit, expected_inner_validation = make_catboost_internal_split(
        expected_outer_fit, y, seed=CATBOOST_INTERNAL_SEED
    )
    for actual, expected, name in (
        (outer_fit, expected_outer_fit, "fresh outer fit"),
        (outer_holdout, expected_outer_holdout, "fresh outer holdout"),
        (inner_fit, expected_inner_fit, "fresh inner fit"),
        (inner_validation, expected_inner_validation, "fresh inner validation"),
    ):
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"recorded split does not match {name}")
    return outer_fit, outer_holdout, inner_fit, inner_validation


def _validate_saved_holdout_ids(
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    split_definitions: tuple[tuple[str, int, str], ...],
) -> None:
    for _split_name, seed, split_filename in split_definitions:
        _fit_indices, holdout_indices = _load_recorded_split(
            artifacts_dir / split_filename, train, y, seed
        )
        expected_ids = set(train.iloc[holdout_indices][ID_COLUMN])
        for path in (
            artifacts_dir
            / "challengers"
            / f"validation_catboost_native_seed{seed}.csv",
            artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv",
        ):
            saved = blend_experiment._read_saved_predictions(path)
            if set(saved[ID_COLUMN]) != expected_ids:
                raise ValueError(f"saved prediction IDs do not match split: {path}")


def _validate_fresh_reference(
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    original_spec: InputSpec,
    input_hashes: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    blend_dir = artifacts_dir / "blend_50_50"
    metadata = json.loads(
        (blend_dir / "compare_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("status") != "completed":
        raise ValueError("existing blend metadata is not completed")
    if (
        metadata.get("input_sha256_before") != input_hashes
        or metadata.get("input_sha256_after") != input_hashes
    ):
        raise ValueError("existing blend input hashes do not match current inputs")
    for name, expected in (
        ("feature_columns", list(original_spec.feature_columns)),
        ("numeric_columns", list(original_spec.numeric_columns)),
        ("categorical_columns", list(original_spec.categorical_columns)),
    ):
        if metadata.get(name) != expected:
            raise ValueError(f"existing blend {name} does not match original inputs")
    fresh_path = blend_dir / "fresh_split_31415.npz"
    arrays = _load_fresh_split(fresh_path, train, y)
    validation = pd.read_csv(blend_dir / "validation_fresh_seed31415.csv")
    expected_columns = [
        "row_index",
        ID_COLUMN,
        "catboost_probability",
        "lightgbm_probability",
        "blend_probability",
    ]
    if list(validation.columns) != expected_columns:
        raise ValueError("existing fresh blend validation schema changed")
    validate_probabilities(validation["blend_probability"].to_numpy(), len(validation))
    expected_ids = train.iloc[arrays[1]][ID_COLUMN].to_numpy()
    if not np.array_equal(np.sort(validation[ID_COLUMN]), np.sort(expected_ids)):
        raise ValueError("existing fresh blend IDs do not match recorded holdout")
    return (*arrays, validation)


def _read_baseline_evidence(
    artifacts_dir: Path, train: pd.DataFrame, expected: dict[int, float]
) -> list[dict[str, Any]]:
    evidence = blend_experiment._prior_blend_evidence(artifacts_dir, train)
    if {int(item["split_seed"]) for item in evidence} != {42, 2024}:
        raise ValueError("prior blend evidence must contain seeds 42 and 2024")
    for item in evidence:
        seed = int(item["split_seed"])
        if not np.isclose(
            float(item["blend_roc_auc"]), expected[seed], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"saved blend AUC changed for seed {seed}")
    return evidence


def _validation_row(
    train: pd.DataFrame,
    holdout_indices: np.ndarray,
    catboost_values: np.ndarray,
    lightgbm_values: np.ndarray,
) -> pd.DataFrame:
    aligned = blend_experiment.aligned_blend_predictions(
        train.iloc[holdout_indices][ID_COLUMN],
        train.iloc[holdout_indices][ID_COLUMN],
        catboost_values,
        train.iloc[holdout_indices][ID_COLUMN],
        lightgbm_values,
    )
    aligned.insert(0, "row_index", holdout_indices)
    return aligned


def _feature_validation_result(
    train: pd.DataFrame,
    y: np.ndarray,
    holdout_indices: np.ndarray,
    catboost_result: dict[str, Any],
    lightgbm_result: dict[str, Any],
    catboost_values: np.ndarray,
    lightgbm_values: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = _validation_row(
        train, holdout_indices, catboost_values, lightgbm_values
    )
    blend_values = validation["blend_probability"].to_numpy()
    return validation, {
        "split_seed": int(catboost_result["split_seed"]),
        "catboost": catboost_result,
        "lightgbm": lightgbm_result,
        "blend": {
            "candidate": "fixed_50_50_features",
            "roc_auc": float(roc_auc_score(y[holdout_indices], blend_values)),
            "log_loss": float(log_loss(y[holdout_indices], blend_values)),
            "catboost_weight": BLEND_WEIGHT,
            "lightgbm_weight": BLEND_WEIGHT,
        },
    }


def _fresh_process_reload(
    raw_test_path: Path,
    catboost_model_path: Path,
    lightgbm_model_path: Path,
    feature_spec_path: Path,
    category_spec_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as directory:
        predictions_path = Path(directory) / "predictions.npy"
        code = (
            "import json, sys; "
            "import lightgbm as lgb, numpy as np, pandas as pd; "
            "from catboost import CatBoostClassifier; "
            "from feature_experiment import add_engineered_features; "
            "raw = pd.read_csv(sys.argv[1]); "
            "features = add_engineered_features(raw); "
            "feature_spec = json.loads(open(sys.argv[4], encoding='utf-8').read()); "
            "category_spec = json.loads(open(sys.argv[5], encoding='utf-8').read()); "
            "features = features.loc[:, feature_spec['feature_columns']]; "
            "cat_features = features.copy(); "
            "[cat_features.__setitem__(c, cat_features[c].astype('string').fillna("
            "'<missing>').astype(object)) "
            " for c in feature_spec['categorical_columns']]; "
            "cat_model = CatBoostClassifier(); cat_model.load_model(sys.argv[2]); "
            "cat = cat_model.predict_proba(cat_features)[:, 1]; "
            "lgb_features = features.copy(); "
            "[lgb_features.__setitem__(c, pd.Categorical("
            "lgb_features[c].astype('string'), "
            "categories=category_spec['category_levels'][c])) "
            " for c in feature_spec['categorical_columns']]; "
            "lgb_model = lgb.Booster(model_file=sys.argv[3]); "
            "lgb_values = lgb_model.predict(lgb_features); "
            "np.save(sys.argv[6], np.column_stack((cat, lgb_values)))"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(raw_test_path.resolve()),
                str(catboost_model_path.resolve()),
                str(lightgbm_model_path.resolve()),
                str(feature_spec_path.resolve()),
                str(category_spec_path.resolve()),
                str(predictions_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        values = np.load(predictions_path)
    return values[:, 0], values[:, 1]


def _finalize_metadata(
    metadata: dict[str, Any],
    metadata_path: Path,
    output_dir: Path,
    artifacts_dir: Path,
    data_dir: Path,
    input_before: dict[str, str] | None,
    protected_before: dict[str, str],
    caught_error: BaseException | None,
    started: float,
) -> tuple[dict[str, Any], BaseException | None]:
    finalization_errors: list[BaseException] = []
    try:
        input_after = {
            "train.csv": sha256_file(data_dir / "train.csv"),
            "test.csv": sha256_file(data_dir / "test.csv"),
        }
        metadata["input_sha256_after"] = input_after
        if input_before is not None and input_after != input_before:
            finalization_errors.append(
                RuntimeError("input CSV fingerprint changed during feature experiment")
            )
    except BaseException as error:
        finalization_errors.append(error)
    try:
        protected_after = _artifact_snapshot(artifacts_dir, output_dir)
        metadata["protected_artifact_sha256_after"] = protected_after
        metadata["protected_artifacts_unchanged"] = protected_before == protected_after
        if protected_before != protected_after:
            finalization_errors.append(
                RuntimeError("pre-existing artifact changed during feature experiment")
            )
    except BaseException as error:
        finalization_errors.append(error)
        protected_after = None

    if finalization_errors:
        finalization_error = RuntimeError(
            "; ".join(str(error) for error in finalization_errors)
        )
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["integrity_error"] = _error_record(finalization_error)
        if caught_error is None:
            metadata["error"] = _error_record(finalization_error)
            caught_error = finalization_error
        else:
            metadata["error"] = _error_record(caught_error)

    metadata["total_seconds"] = round(time.perf_counter() - started, 3)
    try:
        if protected_after is not None:
            save_json(output_dir / "protected_baseline_sha256.json", protected_after)
        save_json(metadata_path, metadata)
    except BaseException as persistence_error:
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["finalization_error"] = _error_record(persistence_error)
        if caught_error is None:
            metadata["error"] = _error_record(persistence_error)
            caught_error = persistence_error
        try:
            save_json(metadata_path, metadata)
        except BaseException as second_error:
            print(
                "Could not persist feature experiment failure metadata: "
                f"{second_error}",
                file=sys.stderr,
            )
    return metadata, caught_error


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the one fixed five-feature CatBoost/LightGBM experiment."""
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
        "external_timeout_seconds": EXPERIMENT_TIMEOUT_SECONDS,
        "threads": MAX_THREADS,
        "python": sys.version,
        "protected_artifact_sha256_before": protected_before,
        "artifacts_dir": str(output_dir),
        "selection": None,
        "final": None,
    }
    save_json(metadata_path, metadata)
    input_before: dict[str, str] | None = None
    caught_error: BaseException | None = None
    try:
        metadata["source_sha256"] = _source_hashes()
        metadata["dependencies"] = _dependency_versions()
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
        input_before = {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        }
        original_train = pd.read_csv(train_path)
        original_test = pd.read_csv(test_path)
        original_spec = validate_inputs(original_train, original_test)
        baseline_evidence = _read_baseline_evidence(
            artifacts_dir, original_train, EXPECTED_PRIOR_BLEND_AUC
        )
        _validate_saved_holdout_ids(
            artifacts_dir,
            original_train,
            target_array(original_train),
            (
                ("outer", 42, "outer_split.npz"),
                ("confirmation", 2024, "confirmation_split.npz"),
            ),
        )
        blend_experiment._validate_reused_artifacts(
            artifacts_dir, input_before, original_spec
        )
        y = target_array(original_train)
        fresh_arrays = _validate_fresh_reference(
            artifacts_dir, original_train, y, original_spec, input_before
        )
        train = add_engineered_features(original_train)
        test = add_engineered_features(original_test)
        spec = validate_inputs(train, test)
        if spec.categorical_columns != original_spec.categorical_columns:
            raise RuntimeError("feature engineering changed nominal predictors")
        if spec.feature_columns[: len(original_spec.feature_columns)] != (
            original_spec.feature_columns
        ):
            raise RuntimeError("engineered predictors were not appended")
        metadata.update(
            {
                "input_sha256_before": input_before,
                "original_feature_columns": list(original_spec.feature_columns),
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "engineered_columns": list(ENGINEERED_COLUMNS),
                "rating_columns": list(RATING_COLUMNS),
                "prior_two_split_evidence": baseline_evidence,
                "recorded_splits": {},
                "validation_results": [],
            }
        )
        save_json(metadata_path, metadata)

        X = train.loc[:, list(spec.feature_columns)]
        X_test = test.loc[:, list(spec.feature_columns)]
        split_definitions = (
            ("outer", 42, "outer_split.npz"),
            ("confirmation", 2024, "confirmation_split.npz"),
        )
        split_arrays: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for split_name, split_seed, split_filename in split_definitions:
            fit_indices, holdout_indices = _load_recorded_split(
                artifacts_dir / split_filename, original_train, y, split_seed
            )
            internal_fit, internal_validation = make_catboost_internal_split(
                fit_indices, y, seed=CATBOOST_INTERNAL_SEED
            )
            split_arrays[split_name] = (
                fit_indices,
                holdout_indices,
                internal_fit,
                internal_validation,
            )
            metadata["recorded_splits"][split_name] = _split_reference(
                artifacts_dir / split_filename,
                split_seed,
                fit_indices,
                holdout_indices,
                internal_fit,
                internal_validation,
            )
        metadata["fresh_split_reference"] = _split_reference(
            artifacts_dir / "blend_50_50" / "fresh_split_31415.npz",
            FRESH_SPLIT_SEED,
            fresh_arrays[0],
            fresh_arrays[1],
            fresh_arrays[2],
            fresh_arrays[3],
        )
        save_json(metadata_path, metadata)

        for split_name, split_seed, _split_filename in split_definitions:
            fit_indices, holdout_indices, inner_fit, inner_validation = split_arrays[
                split_name
            ]
            metadata["current_fit"] = {
                "split_name": split_name,
                "split_seed": split_seed,
                "status": "running",
            }
            save_json(metadata_path, metadata)
            cat_model, cat_predictions, cat_result, cat_internal = _fit_catboost(
                spec,
                X,
                y,
                fit_indices,
                X.iloc[holdout_indices],
                y[holdout_indices],
                split_seed=split_seed,
                depth=CATBOOST_DEPTH,
                iterations=CATBOOST_ITERATIONS,
                candidate="catboost_features",
            )
            if not np.array_equal(
                cat_internal["fit_indices"], inner_fit
            ) or not np.array_equal(
                cat_internal["validation_indices"], inner_validation
            ):
                raise RuntimeError("CatBoost helper changed the recorded inner split")
            cat_result["best_iteration_count_one_based"] = (
                int(cat_result["best_iteration"]) + 1
            )
            del cat_model
            gc.collect()

            lgb_model = build_lightgbm_classifier()
            levels, lgb_fit_seconds, lgb_details = _fit_with_inner_validation(
                lgb_model,
                X,
                y,
                inner_fit,
                inner_validation,
                spec,
                callbacks=_fit_callbacks(),
            )
            if sorted(lgb_model.evals_result_["inner_validation"]) != ["auc"]:
                raise RuntimeError("feature LightGBM fit registered a non-AUC metric")
            lgb_predictions, lgb_result = lightgbm_validation_result(
                lgb_model,
                X,
                y,
                holdout_indices,
                levels,
                spec,
                split_seed=split_seed,
                fit_seconds=lgb_fit_seconds,
                fit_details=lgb_details,
                internal_fit=inner_fit,
                internal_validation=inner_validation,
            )
            lgb_result["best_iteration_count_one_based"] = lgb_result["best_iteration"]
            validation, result = _feature_validation_result(
                train,
                y,
                holdout_indices,
                cat_result,
                lgb_result,
                cat_predictions,
                lgb_predictions,
            )
            validation_path = output_dir / f"validation_features_seed{split_seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            result["validation_predictions_path"] = str(validation_path)
            metadata["validation_results"].append(result)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del lgb_model, levels, cat_predictions, lgb_predictions, validation
            gc.collect()

        baseline_by_seed = {
            int(item["split_seed"]): float(item["blend_roc_auc"])
            for item in baseline_evidence
        }
        engineered_by_seed = {
            int(item["split_seed"]): float(item["blend"]["roc_auc"])
            for item in metadata["validation_results"]
        }
        metadata["selection"] = {
            "recorded_gate": engineered_gate(baseline_by_seed, engineered_by_seed),
            "baseline_candidate": "unengineered_fixed_50_50_blend",
            "engineered_candidate": "fixed_50_50_features",
            "decision": "continue_fresh_confirmation",
        }
        if not metadata["selection"]["recorded_gate"]["qualifies"]:
            metadata["selection"]["decision"] = "retain_unengineered_blend"
            metadata["status"] = "completed"
            save_json(metadata_path, metadata)
        else:
            (
                fresh_outer_fit,
                fresh_holdout,
                fresh_inner_fit,
                fresh_inner_validation,
                _fresh_reference_validation,
            ) = fresh_arrays
            metadata["current_fit"] = {
                "split_name": "fresh",
                "split_seed": FRESH_SPLIT_SEED,
                "status": "running",
            }
            save_json(metadata_path, metadata)
            cat_model, cat_predictions, cat_result, cat_internal = _fit_catboost(
                spec,
                X,
                y,
                fresh_outer_fit,
                X.iloc[fresh_holdout],
                y[fresh_holdout],
                split_seed=FRESH_SPLIT_SEED,
                depth=CATBOOST_DEPTH,
                iterations=CATBOOST_ITERATIONS,
                candidate="catboost_features",
            )
            if not np.array_equal(
                cat_internal["fit_indices"], fresh_inner_fit
            ) or not np.array_equal(
                cat_internal["validation_indices"], fresh_inner_validation
            ):
                raise RuntimeError("CatBoost helper changed the fresh inner split")
            cat_result["best_iteration_count_one_based"] = (
                int(cat_result["best_iteration"]) + 1
            )
            del cat_model
            gc.collect()

            lgb_model = build_lightgbm_classifier()
            levels, lgb_fit_seconds, lgb_details = _fit_with_inner_validation(
                lgb_model,
                X,
                y,
                fresh_inner_fit,
                fresh_inner_validation,
                spec,
                callbacks=_fit_callbacks(),
            )
            if sorted(lgb_model.evals_result_["inner_validation"]) != ["auc"]:
                raise RuntimeError(
                    "feature LightGBM fresh fit registered a non-AUC metric"
                )
            lgb_predictions, lgb_result = lightgbm_validation_result(
                lgb_model,
                X,
                y,
                fresh_holdout,
                levels,
                spec,
                split_seed=FRESH_SPLIT_SEED,
                fit_seconds=lgb_fit_seconds,
                fit_details=lgb_details,
                internal_fit=fresh_inner_fit,
                internal_validation=fresh_inner_validation,
            )
            lgb_result["best_iteration_count_one_based"] = lgb_result["best_iteration"]
            fresh_validation, fresh_result = _feature_validation_result(
                train,
                y,
                fresh_holdout,
                cat_result,
                lgb_result,
                cat_predictions,
                lgb_predictions,
            )
            fresh_validation_path = (
                output_dir / "validation_features_fresh_seed31415.csv"
            )
            fresh_validation.to_csv(
                fresh_validation_path, index=False, float_format="%.17g"
            )
            fresh_result["validation_predictions_path"] = str(fresh_validation_path)
            metadata["fresh_validation"] = fresh_result
            reference_values = _validate_fresh_reference(
                artifacts_dir, original_train, y, original_spec, input_before
            )[-1]
            reference_auc = float(
                roc_auc_score(
                    y[fresh_holdout],
                    reference_values.set_index(ID_COLUMN).reindex(
                        train.iloc[fresh_holdout][ID_COLUMN]
                    )["blend_probability"],
                )
            )
            if not np.isclose(
                reference_auc, EXPECTED_FRESH_BLEND_AUC, rtol=0.0, atol=1e-12
            ):
                raise ValueError("saved fresh blend AUC changed")
            fresh_gate = engineered_gate(
                {42: reference_auc, 2024: reference_auc},
                {
                    42: fresh_result["blend"]["roc_auc"],
                    2024: fresh_result["blend"]["roc_auc"],
                },
            )
            fresh_gain = fresh_result["blend"]["roc_auc"] - reference_auc
            fresh_gate["comparison"] = [
                {
                    "split_seed": FRESH_SPLIT_SEED,
                    "baseline_roc_auc": reference_auc,
                    "engineered_roc_auc": fresh_result["blend"]["roc_auc"],
                    "auc_gain": fresh_gain,
                    "qualifies": fresh_gain >= MINIMUM_AUC_GAIN,
                }
            ]
            fresh_gate["qualifies"] = fresh_gain >= MINIMUM_AUC_GAIN
            fresh_gate["decision"] = (
                "refit_and_export"
                if fresh_gate["qualifies"]
                else "retain_unengineered_blend"
            )
            metadata["selection"]["fresh_gate"] = fresh_gate
            metadata["selection"]["decision"] = fresh_gate["decision"]
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del lgb_model, levels, cat_predictions, lgb_predictions, fresh_validation
            gc.collect()

            if fresh_gate["qualifies"]:
                metadata["current_fit"] = {"status": "refit_running"}
                save_json(metadata_path, metadata)
                cat_counts = [
                    int(item["catboost"]["best_iteration_count_one_based"])
                    for item in metadata["validation_results"]
                ] + [int(cat_result["best_iteration_count_one_based"])]
                lgb_counts = [
                    int(item["lightgbm"]["best_iteration_count_one_based"])
                    for item in metadata["validation_results"]
                ] + [int(lgb_result["best_iteration_count_one_based"])]
                cat_iterations = int(round(float(np.median(cat_counts))))
                lgb_iterations = int(round(float(np.median(lgb_counts))))
                cat_model = build_catboost_classifier(
                    iterations=cat_iterations,
                    early_stopping_rounds=None,
                    depth=CATBOOST_DEPTH,
                )
                cat_started = time.perf_counter()
                cat_model.fit(
                    prepare_catboost_frame(X, spec),
                    y,
                    cat_features=list(spec.categorical_columns),
                    verbose=False,
                )
                cat_fit_seconds = time.perf_counter() - cat_started
                cat_test = cat_model.predict_proba(
                    prepare_catboost_frame(X_test, spec)
                )[:, 1]
                validate_probabilities(cat_test, len(X_test))
                cat_path = output_dir / "final_model_catboost_features.cbm"
                cat_model.save_model(cat_path)
                del cat_model

                category_levels = fit_category_levels(X, spec)
                lgb_model = build_lightgbm_classifier(n_estimators=lgb_iterations)
                lgb_before = _resource_snapshot()
                lgb_started = time.perf_counter()
                lgb_model.fit(
                    prepare_lightgbm_frame(X, spec, category_levels),
                    y,
                    categorical_feature=list(spec.categorical_columns),
                    callbacks=[lgb.log_evaluation(0)],
                )
                lgb_fit_seconds = time.perf_counter() - lgb_started
                lgb_test = lgb_model.predict_proba(
                    prepare_lightgbm_frame(X_test, spec, category_levels)
                )[:, 1]
                validate_probabilities(lgb_test, len(X_test))
                lgb_path = output_dir / "final_model_lightgbm_features.txt"
                lgb_model.booster_.save_model(lgb_path)
                del lgb_model

                feature_spec_path = output_dir / "feature_spec.json"
                category_spec_path = output_dir / "category_spec.json"
                bundle_spec_path = output_dir / "bundle_spec.json"
                save_json(
                    feature_spec_path,
                    {
                        "original_feature_columns": list(original_spec.feature_columns),
                        "feature_columns": list(spec.feature_columns),
                        "numeric_columns": list(spec.numeric_columns),
                        "categorical_columns": list(spec.categorical_columns),
                        "engineered_columns": list(ENGINEERED_COLUMNS),
                        "rating_columns": list(RATING_COLUMNS),
                        "departure_delay_column": DEPARTURE_DELAY_COLUMN,
                        "arrival_delay_column": ARRIVAL_DELAY_COLUMN,
                    },
                )
                save_json(
                    category_spec_path,
                    {
                        "categorical_columns": list(spec.categorical_columns),
                        "category_levels": category_levels,
                    },
                )
                save_json(
                    bundle_spec_path,
                    {
                        "catboost_model_path": str(cat_path),
                        "lightgbm_model_path": str(lgb_path),
                        "feature_spec_path": str(feature_spec_path),
                        "category_spec_path": str(category_spec_path),
                        "weights": {"catboost": BLEND_WEIGHT, "lightgbm": BLEND_WEIGHT},
                        "catboost_iterations": cat_iterations,
                        "lightgbm_iterations": lgb_iterations,
                        "early_stopping": False,
                        "eval_set": None,
                    },
                )
                final_blend = BLEND_WEIGHT * cat_test + BLEND_WEIGHT * lgb_test
                final_submission_path = (
                    output_dir / "submission_catboost_lightgbm_features_50_50.csv"
                )
                pd.DataFrame(
                    {ID_COLUMN: original_test[ID_COLUMN], TARGET_COLUMN: final_blend}
                ).to_csv(final_submission_path, index=False, float_format="%.17g")
                submission = validate_submission_file(
                    final_submission_path, original_test[ID_COLUMN]
                )
                reloaded_cat, reloaded_lgb = _fresh_process_reload(
                    test_path,
                    cat_path,
                    lgb_path,
                    feature_spec_path,
                    category_spec_path,
                )
                max_component_difference = max(
                    float(np.max(np.abs(cat_test - reloaded_cat))),
                    float(np.max(np.abs(lgb_test - reloaded_lgb))),
                )
                csv_difference = float(
                    np.max(
                        np.abs(
                            final_blend
                            - submission[TARGET_COLUMN].to_numpy(dtype=np.float64)
                        )
                    )
                )
                if max_component_difference > 1e-12 or csv_difference > 1e-12:
                    raise RuntimeError("fresh-process reload or final CSV differs")
                metadata["final"] = {
                    "model": "fixed_catboost_lightgbm_features_50_50",
                    "catboost_model_path": str(cat_path),
                    "lightgbm_model_path": str(lgb_path),
                    "feature_spec_path": str(feature_spec_path),
                    "category_spec_path": str(category_spec_path),
                    "bundle_spec_path": str(bundle_spec_path),
                    "catboost_iteration_counts": cat_counts,
                    "lightgbm_iteration_counts": lgb_counts,
                    "catboost_selected_iterations": cat_iterations,
                    "lightgbm_selected_iterations": lgb_iterations,
                    "catboost_fit_seconds": round(cat_fit_seconds, 3),
                    "lightgbm_fit_seconds": round(lgb_fit_seconds, 3),
                    "lightgbm_resources": _resource_delta(
                        lgb_before, _resource_snapshot()
                    ),
                    "submission_path": str(final_submission_path),
                    "submission_rows": len(submission),
                    "weights": {"catboost": BLEND_WEIGHT, "lightgbm": BLEND_WEIGHT},
                    "clean_process_reload": {
                        "rows": len(submission),
                        "max_component_difference": max_component_difference,
                        "max_csv_difference": csv_difference,
                    },
                    "model_sha256": {
                        "catboost": sha256_file(cat_path),
                        "lightgbm": sha256_file(lgb_path),
                    },
                }
                metadata["current_fit"] = None
            metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["error"] = _error_record(error)
        current = metadata.get("current_fit")
        if current is not None:
            current["status"] = "failed"
            current["error"] = _error_record(error)
            metadata["current_fit"] = current

    metadata, finalization_error = _finalize_metadata(
        metadata,
        metadata_path,
        output_dir,
        artifacts_dir,
        data_dir,
        input_before,
        protected_before,
        caught_error,
        started,
    )
    if finalization_error is not None:
        raise finalization_error
    return metadata


def _terminal_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": metadata.get("status", "failed"),
        "selection": metadata.get("selection"),
        "final": metadata.get("final"),
        "error": metadata.get("error"),
        "total_seconds": metadata.get("total_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed five-feature CatBoost/LightGBM blend experiment."
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
