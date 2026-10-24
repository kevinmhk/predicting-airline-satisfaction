from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
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
    CATBOOST_EARLY_STOPPING_ROUNDS,
    CATBOOST_INTERNAL_SEED,
    CATBOOST_INTERNAL_VALIDATION_FRACTION,
    CATBOOST_L2_REGULARIZATION,
    CATBOOST_LEARNING_RATE,
    CATBOOST_RANDOM_SEED,
    _fit_catboost,
    _load_recorded_split,
    build_catboost_classifier,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import _artifact_snapshot, prepare_lightgbm_frame
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

LONGER_CATBOOST_ITERATIONS = 3000
MINIMUM_AUC_GAIN = 0.0002
FRESH_SPLIT_SEED = 31415
EXPERIMENT_TIMEOUT_SECONDS = 40 * 60
OUTPUT_NAME = "longer_catboost"
OUTPUT_BUDGET_BYTES = 200 * 1024 * 1024
SCRATCH_DIR = Path("/home/kevinmhk/.hermes/profiles/coder/cache/scratch")
CURRENT_BLEND_AUC = {42: 0.957713923393243, 2024: 0.9580214315945633}
CURRENT_FRESH_BLEND_AUC = 0.9588362995285812


class LongerCatBoostTimeout(TimeoutError):
    """Raised when the bounded longer-CatBoost run reaches its deadline."""


def build_longer_catboost_classifier() -> Any:
    """Build CatBoost with the prior settings and only the longer cap changed."""
    return build_catboost_classifier(iterations=LONGER_CATBOOST_ITERATIONS)


def selected_iterations(results: list[dict[str, Any]]) -> int:
    """Use the median one-based count from completed CatBoost validation fits."""
    counts = [int(result["best_iteration"]) + 1 for result in results]
    if len(counts) != 3 or any(count < 1 for count in counts):
        raise ValueError("three completed CatBoost validation counts are required")
    return int(round(float(np.median(counts))))


def blend_gate(
    baseline_auc_by_seed: dict[int, float], new_auc_by_seed: dict[int, float]
) -> dict[str, Any]:
    """Require the longer CatBoost blend gain on both initial seeds."""
    comparison = []
    for seed in (42, 2024):
        if seed not in baseline_auc_by_seed or seed not in new_auc_by_seed:
            continue
        gain = float(new_auc_by_seed[seed]) - float(baseline_auc_by_seed[seed])
        comparison.append(
            {
                "split_seed": seed,
                "baseline_blend_roc_auc": baseline_auc_by_seed[seed],
                "new_blend_roc_auc": new_auc_by_seed[seed],
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
        else "retain_current_blend",
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _dependencies() -> dict[str, str]:
    values = package_versions()
    for name in ("catboost", "lightgbm", "threadpoolctl"):
        values[name] = version(name)
    return values


def _source_hashes() -> dict[str, str]:
    names = (
        "longer_catboost.py",
        "train.py",
        "compare.py",
        "blend_experiment.py",
        "lightgbm_experiment.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _config() -> dict[str, Any]:
    return {
        "depth": CATBOOST_DEPTH,
        "iterations": LONGER_CATBOOST_ITERATIONS,
        "learning_rate": CATBOOST_LEARNING_RATE,
        "l2_leaf_reg": CATBOOST_L2_REGULARIZATION,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": CATBOOST_RANDOM_SEED,
        "thread_count": MAX_THREADS,
        "allow_writing_files": False,
        "early_stopping_rounds": CATBOOST_EARLY_STOPPING_ROUNDS,
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "internal_validation_seed": CATBOOST_INTERNAL_SEED,
        "internal_validation_fraction": CATBOOST_INTERNAL_VALIDATION_FRACTION,
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
        "inner_covers_outer_fit": np.array_equal(
            np.sort(np.concatenate((inner_fit, inner_validation))),
            np.sort(outer_fit),
        ),
        "inner_disjoint_from_outer_holdout": not bool(
            np.intersect1d(inner_fit, outer_holdout).size
            or np.intersect1d(inner_validation, outer_holdout).size
        ),
    }


def _load_fresh_split(
    path: Path, train: pd.DataFrame, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as recorded:
        outer_fit = np.asarray(recorded["outer_fit"], dtype=np.int64)
        outer_holdout = np.asarray(recorded["outer_holdout"], dtype=np.int64)
        inner_fit = np.asarray(recorded["inner_fit"], dtype=np.int64)
        inner_validation = np.asarray(recorded["inner_validation"], dtype=np.int64)
    arrays = (outer_fit, outer_holdout, inner_fit, inner_validation)
    expected_outer_fit, expected_outer_holdout = make_split(train, y, FRESH_SPLIT_SEED)
    expected_inner_fit, expected_inner_validation = make_catboost_internal_split(
        expected_outer_fit, y, seed=CATBOOST_INTERNAL_SEED
    )
    for actual, expected, name in zip(
        arrays,
        (
            expected_outer_fit,
            expected_outer_holdout,
            expected_inner_fit,
            expected_inner_validation,
        ),
        ("outer fit", "outer holdout", "inner fit", "inner validation"),
        strict=True,
    ):
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"fresh split does not match expected {name}")
    return arrays


def _read_fresh_validation(path: Path, expected_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = [
        "row_index",
        ID_COLUMN,
        "catboost_probability",
        "lightgbm_probability",
        "blend_probability",
    ]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"fresh validation schema changed: {path}")
    if frame[ID_COLUMN].duplicated().any() or set(frame[ID_COLUMN]) != set(
        expected_ids
    ):
        raise ValueError(f"fresh validation IDs do not match recorded holdout: {path}")
    for column in expected_columns[2:]:
        validate_probabilities(frame[column].to_numpy(), len(frame))
    return frame


def _validation_frame(
    train: pd.DataFrame,
    holdout: np.ndarray,
    catboost_values: np.ndarray,
    lightgbm_ids: Any,
    lightgbm_values: np.ndarray,
    catboost_ids: Any | None = None,
) -> pd.DataFrame:
    holdout_ids = train.iloc[holdout][ID_COLUMN].to_numpy()
    aligned = blend_experiment.aligned_blend_predictions(
        holdout_ids,
        holdout_ids if catboost_ids is None else catboost_ids,
        catboost_values,
        lightgbm_ids,
        lightgbm_values,
    )
    aligned.insert(0, "row_index", holdout)
    return aligned


def _validation_result(
    frame: pd.DataFrame, y: np.ndarray, holdout: np.ndarray, cat_result: dict[str, Any]
) -> dict[str, Any]:
    blend_values = frame["blend_probability"].to_numpy()
    return {
        "split_seed": int(cat_result["split_seed"]),
        "catboost": cat_result,
        "lightgbm": {"source": "reused saved corrected validation predictions"},
        "blend": {
            "candidate": "fixed_50_50_longer_catboost_lightgbm",
            "catboost_weight": 0.5,
            "lightgbm_weight": 0.5,
            "roc_auc": float(roc_auc_score(y[holdout], blend_values)),
            "log_loss": float(log_loss(y[holdout], blend_values)),
        },
    }


def _resolve_path(value: str, expected: Path, artifacts_dir: Path) -> None:
    raw = Path(value)
    candidates = [raw, artifacts_dir.parent / raw, artifacts_dir / raw]
    if not any(candidate.resolve() == expected.resolve() for candidate in candidates):
        raise ValueError(f"reused artifact path is not {expected}")


def _validate_reused_state(
    artifacts_dir: Path,
    input_hashes: dict[str, str],
    spec: Any,
) -> dict[str, Path]:
    prior_counts = blend_experiment._validate_reused_artifacts(
        artifacts_dir, input_hashes, spec
    )
    if prior_counts != [879, 783]:
        raise ValueError(f"corrected LightGBM counts changed: {prior_counts}")
    blend_dir = artifacts_dir / "blend_50_50"
    metadata = json.loads((blend_dir / "compare_metadata.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError("existing blend metadata is not completed")
    if (
        metadata.get("input_sha256_before") != input_hashes
        or metadata.get("input_sha256_after") != input_hashes
    ):
        raise ValueError("existing blend input hashes do not match current inputs")
    if metadata.get("selection", {}).get("confirmation_pass") is not True:
        raise ValueError("existing blend did not pass its confirmation gate")
    final = metadata.get("final")
    if not isinstance(final, dict) or final.get("weights") != {
        "catboost": 0.5,
        "lightgbm": 0.5,
    }:
        raise ValueError("existing blend final settings are not fixed 50/50")
    cat_model = artifacts_dir / "challengers" / "final_model_catboost.cbm"
    cat_spec = artifacts_dir / "challengers" / "feature_spec.json"
    lgb_model = blend_dir / "final_model_lightgbm.txt"
    lgb_spec = blend_dir / "feature_spec.json"
    _resolve_path(str(final.get("catboost_model_path")), cat_model, artifacts_dir)
    _resolve_path(str(final.get("catboost_feature_spec_path")), cat_spec, artifacts_dir)
    _resolve_path(str(final.get("lightgbm_model_path")), lgb_model, artifacts_dir)
    _resolve_path(str(final.get("lightgbm_feature_spec_path")), lgb_spec, artifacts_dir)
    if (
        final.get("lightgbm_selected_iterations") != 783
        or final.get("lightgbm_early_stopping") is not False
        or final.get("lightgbm_eval_set") is not None
    ):
        raise ValueError("existing full-train LightGBM settings changed")
    if final.get("model_sha256") != sha256_file(lgb_model):
        raise ValueError("existing LightGBM model hash changed")
    if final.get("feature_spec_sha256") != sha256_file(lgb_spec):
        raise ValueError("existing LightGBM feature spec hash changed")
    cat_spec_data = json.loads(cat_spec.read_text())
    lgb_spec_data = json.loads(lgb_spec.read_text())
    for data, name in ((cat_spec_data, "CatBoost"), (lgb_spec_data, "LightGBM")):
        for field in ("feature_columns", "numeric_columns", "categorical_columns"):
            if data.get(field) != list(getattr(spec, field)):
                raise ValueError(f"reused {name} feature spec changed: {field}")
    return {
        "cat_model": cat_model,
        "cat_spec": cat_spec,
        "lgb_model": lgb_model,
        "lgb_spec": lgb_spec,
    }


def _fresh_process_reload(
    test_path: Path, cat_model: Path, cat_spec: Path, lgb_model: Path, lgb_spec: Path
) -> np.ndarray:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as directory:
        output = Path(directory) / "predictions.npy"
        code = (
            "import json, lightgbm as lgb, numpy as np, pandas as pd, sys; "
            "from catboost import CatBoostClassifier; "
            "raw=pd.read_csv(sys.argv[1]); cs=json.loads(open(sys.argv[3]).read()); "
            "ls=json.loads(open(sys.argv[5]).read()); "
            "features=raw.loc[:, cs['feature_columns']].copy(); "
            "[features.__setitem__(c, features[c].astype('string').fillna("
            "'<missing>').astype(object)) "
            "for c in cs['categorical_columns']]; "
            "cat=CatBoostClassifier(); cat.load_model(sys.argv[2]); "
            "cat_values=cat.predict_proba(features)[:,1]; "
            "lgb_features=features.copy(); "
            "[lgb_features.__setitem__(c, pd.Categorical("
            "lgb_features[c].astype('string'), categories=ls['category_levels'][c])) "
            "for c in ls['categorical_columns']]; "
            "booster=lgb.Booster(model_file=sys.argv[4]); "
            "lgb_values=booster.predict(lgb_features); "
            "np.save(sys.argv[6], np.column_stack((cat_values,lgb_values)))"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(test_path.resolve()),
                str(cat_model.resolve()),
                str(cat_spec.resolve()),
                str(lgb_model.resolve()),
                str(lgb_spec.resolve()),
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        return np.load(output)


def _output_bytes(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())


def _postrun_budget_validation(
    metadata: dict[str, Any], metadata_path: Path, output_dir: Path
) -> tuple[int, bool]:
    """Record stable post-write size and validate the final output budget."""
    for _ in range(3):
        measured = _output_bytes(output_dir)
        metadata["output_bytes"] = measured
        save_json(metadata_path, metadata)
        postwrite = _output_bytes(output_dir)
        if postwrite == measured:
            return postwrite, postwrite > OUTPUT_BUDGET_BYTES
    raise RuntimeError("longer CatBoost output size did not stabilize")


def _finalize(
    metadata: dict[str, Any],
    metadata_path: Path,
    output_dir: Path,
    artifacts_dir: Path,
    data_dir: Path,
    input_before: dict[str, str] | None,
    protected_before: dict[str, str],
    caught: BaseException | None,
    started: float,
) -> tuple[dict[str, Any], BaseException | None]:
    errors: list[BaseException] = []
    try:
        input_after = {
            "train.csv": sha256_file(data_dir / "train.csv"),
            "test.csv": sha256_file(data_dir / "test.csv"),
        }
        metadata["input_sha256_after"] = input_after
        if input_before is not None and input_after != input_before:
            errors.append(
                RuntimeError(
                    "input CSV fingerprint changed during longer CatBoost experiment"
                )
            )
    except BaseException as error:
        errors.append(error)
    try:
        protected_after = _artifact_snapshot(artifacts_dir, output_dir)
        metadata["protected_artifact_sha256_after"] = protected_after
        metadata["protected_artifacts_unchanged"] = protected_before == protected_after
        if protected_before != protected_after:
            errors.append(
                RuntimeError(
                    "pre-existing artifact changed during longer CatBoost experiment"
                )
            )
    except BaseException as error:
        protected_after = None
        errors.append(error)
    if errors:
        finalization_error = RuntimeError("; ".join(str(error) for error in errors))
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["integrity_error"] = _error_record(finalization_error)
        if caught is None:
            caught = finalization_error
            metadata["error"] = _error_record(finalization_error)
        else:
            metadata["finalization_error"] = _error_record(finalization_error)
    metadata["total_seconds"] = round(time.perf_counter() - started, 3)
    metadata["output_budget_bytes"] = OUTPUT_BUDGET_BYTES
    try:
        if protected_after is not None:
            save_json(output_dir / "protected_baseline_sha256.json", protected_after)
        _, budget_exceeded = _postrun_budget_validation(
            metadata, metadata_path, output_dir
        )
        if budget_exceeded:
            finalization_error = RuntimeError(
                "longer CatBoost artifact budget exceeded"
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["final"] = None
            metadata["integrity_error"] = _error_record(finalization_error)
            if caught is None:
                caught = finalization_error
                metadata["error"] = _error_record(finalization_error)
            else:
                metadata["finalization_error"] = _error_record(finalization_error)
            _postrun_budget_validation(metadata, metadata_path, output_dir)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["finalization_error"] = _error_record(error)
        if caught is None:
            caught = error
            metadata["error"] = _error_record(error)
        try:
            save_json(metadata_path, metadata)
        except BaseException as second_error:
            print(
                f"Could not persist longer CatBoost metadata: {second_error}",
                file=sys.stderr,
            )
    return metadata, caught


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the one longer-CatBoost validation and conditional fixed blend refit."""
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
        "timeout_seconds": EXPERIMENT_TIMEOUT_SECONDS,
        "threads": MAX_THREADS,
        "artifacts_dir": str(output_dir),
        "selection": None,
        "final": None,
        "protected_artifact_sha256_before": protected_before,
    }
    save_json(metadata_path, metadata)
    input_before: dict[str, str] | None = None
    caught: BaseException | None = None
    previous_alarm = signal.getitimer(signal.ITIMER_REAL)
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    previous_term_handler = signal.getsignal(signal.SIGTERM)

    def stop_experiment(_signum: int, _frame: Any) -> None:
        raise LongerCatBoostTimeout("longer CatBoost experiment timed out")

    try:
        signal.signal(signal.SIGALRM, stop_experiment)
        signal.signal(signal.SIGTERM, stop_experiment)
        signal.setitimer(signal.ITIMER_REAL, EXPERIMENT_TIMEOUT_SECONDS)
        metadata["python"] = sys.version
        metadata["dependencies"] = _dependencies()
        metadata["source_sha256"] = _source_hashes()
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
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
        reused = _validate_reused_state(artifacts_dir, input_before, spec)
        metadata.update(
            {
                "input_sha256_before": input_before,
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "config": _config(),
                "config_sha256": hashlib.sha256(
                    json.dumps(_config(), sort_keys=True).encode()
                ).hexdigest(),
                "reused_artifacts": {name: str(path) for name, path in reused.items()},
                "validation_results": [],
                "fresh_validation": None,
                "current_fit": None,
            }
        )
        save_json(metadata_path, metadata)

        split_definitions = (
            ("outer", 42, "outer_split.npz"),
            ("confirmation", 2024, "confirmation_split.npz"),
        )
        split_arrays: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        prior_metadata = json.loads(
            (artifacts_dir / "challengers" / "compare_metadata.json").read_text()
        )
        for split_name, seed, filename in split_definitions:
            fit, holdout = _load_recorded_split(
                artifacts_dir / filename, train, y, seed
            )
            prior = prior_metadata["internal_validation_splits"][split_name]
            inner_fit = np.asarray(prior["fit_indices"], dtype=np.int64)
            inner_validation = np.asarray(prior["validation_indices"], dtype=np.int64)
            expected_fit, expected_validation = make_catboost_internal_split(
                fit, y, seed=CATBOOST_INTERNAL_SEED
            )
            if not np.array_equal(inner_fit, expected_fit) or not np.array_equal(
                inner_validation, expected_validation
            ):
                raise RuntimeError("prior CatBoost internal split changed")
            if np.intersect1d(inner_validation, holdout).size:
                raise RuntimeError("inner validation overlaps outer holdout")
            split_arrays[split_name] = (fit, holdout, inner_fit, inner_validation)
            metadata.setdefault("recorded_splits", {})[split_name] = _split_record(
                artifacts_dir / filename, fit, holdout, inner_fit, inner_validation
            )
        fresh_path = artifacts_dir / "blend_50_50" / "fresh_split_31415.npz"
        fresh_arrays = _load_fresh_split(fresh_path, train, y)
        metadata["fresh_split"] = _split_record(fresh_path, *fresh_arrays)
        save_json(metadata_path, metadata)

        baseline_auc: dict[int, float] = {}
        new_auc: dict[int, float] = {}
        for split_name, seed, _filename in split_definitions:
            fit, holdout, inner_fit, inner_validation = split_arrays[split_name]
            cat_path = (
                artifacts_dir
                / "challengers"
                / f"validation_catboost_native_seed{seed}.csv"
            )
            lgb_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            saved_lgb = blend_experiment._read_saved_predictions(lgb_path)
            expected_ids = train.iloc[holdout][ID_COLUMN].to_numpy()
            if set(saved_lgb[ID_COLUMN]) != set(expected_ids):
                raise ValueError(
                    f"saved LightGBM IDs do not match holdout for seed {seed}"
                )
            saved_cat = blend_experiment._read_saved_predictions(cat_path)
            baseline_frame = _validation_frame(
                train,
                holdout,
                saved_cat["probability"].to_numpy(),
                saved_lgb[ID_COLUMN],
                saved_lgb["probability"].to_numpy(),
                catboost_ids=saved_cat[ID_COLUMN],
            )
            baseline_auc[seed] = float(
                roc_auc_score(y[holdout], baseline_frame["blend_probability"])
            )
            if not np.isclose(
                baseline_auc[seed], CURRENT_BLEND_AUC[seed], rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"current blend baseline changed for seed {seed}")
            metadata["current_fit"] = {
                "split_name": split_name,
                "split_seed": seed,
                "status": "running",
            }
            save_json(metadata_path, metadata)
            model, cat_predictions, cat_result, internal = _fit_catboost(
                spec,
                X,
                y,
                fit,
                X.iloc[holdout],
                y[holdout],
                split_seed=seed,
                depth=CATBOOST_DEPTH,
                iterations=LONGER_CATBOOST_ITERATIONS,
                candidate="catboost_iterations3000",
            )
            if not np.array_equal(
                internal["fit_indices"], inner_fit
            ) or not np.array_equal(internal["validation_indices"], inner_validation):
                raise RuntimeError(
                    "longer CatBoost helper changed the recorded inner split"
                )
            cat_result["best_iteration_index_zero_based"] = cat_result["best_iteration"]
            cat_result["best_iteration_count_one_based"] = (
                int(cat_result["best_iteration"]) + 1
            )
            cat_result["maximum_configured_iterations"] = LONGER_CATBOOST_ITERATIONS
            frame = _validation_frame(
                train,
                holdout,
                cat_predictions,
                saved_lgb[ID_COLUMN],
                saved_lgb["probability"].to_numpy(),
            )
            result = _validation_result(frame, y, holdout, cat_result)
            validation_path = output_dir / f"validation_longer_catboost_seed{seed}.csv"
            frame.to_csv(validation_path, index=False, float_format="%.17g")
            result["validation_predictions_path"] = str(validation_path)
            metadata["validation_results"].append(result)
            new_auc[seed] = float(result["blend"]["roc_auc"])
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del model, cat_predictions, frame
            gc.collect()

        metadata["selection"] = {
            "initial_gate": blend_gate(baseline_auc, new_auc),
            "baseline_candidate": "current_unengineered_fixed_50_50_blend",
            "new_candidate": "longer_catboost_fixed_50_50_blend",
            "decision": "continue_fresh_confirmation",
        }
        if not metadata["selection"]["initial_gate"]["qualifies"]:
            metadata["selection"]["decision"] = "retain_current_blend"
            metadata["status"] = "completed"
            save_json(metadata_path, metadata)
        else:
            fresh_fit, fresh_holdout, fresh_inner_fit, fresh_inner_validation = (
                fresh_arrays
            )
            fresh_saved = _read_fresh_validation(
                artifacts_dir / "blend_50_50" / "validation_fresh_seed31415.csv",
                train.iloc[fresh_holdout][ID_COLUMN].to_numpy(),
            )
            fresh_baseline = float(
                roc_auc_score(
                    y[fresh_holdout],
                    fresh_saved.set_index(ID_COLUMN).reindex(
                        train.iloc[fresh_holdout][ID_COLUMN]
                    )["blend_probability"],
                )
            )
            if not np.isclose(
                fresh_baseline, CURRENT_FRESH_BLEND_AUC, rtol=0.0, atol=1e-12
            ):
                raise ValueError("current fresh blend baseline changed")
            metadata["current_fit"] = {
                "split_name": "fresh",
                "split_seed": FRESH_SPLIT_SEED,
                "status": "running",
            }
            save_json(metadata_path, metadata)
            model, cat_predictions, cat_result, internal = _fit_catboost(
                spec,
                X,
                y,
                fresh_fit,
                X.iloc[fresh_holdout],
                y[fresh_holdout],
                split_seed=FRESH_SPLIT_SEED,
                depth=CATBOOST_DEPTH,
                iterations=LONGER_CATBOOST_ITERATIONS,
                candidate="catboost_iterations3000",
            )
            if not np.array_equal(
                internal["fit_indices"], fresh_inner_fit
            ) or not np.array_equal(
                internal["validation_indices"], fresh_inner_validation
            ):
                raise RuntimeError(
                    "longer CatBoost helper changed the fresh inner split"
                )
            cat_result["best_iteration_index_zero_based"] = cat_result["best_iteration"]
            cat_result["best_iteration_count_one_based"] = (
                int(cat_result["best_iteration"]) + 1
            )
            cat_result["maximum_configured_iterations"] = LONGER_CATBOOST_ITERATIONS
            frame = _validation_frame(
                train,
                fresh_holdout,
                cat_predictions,
                fresh_saved[ID_COLUMN],
                fresh_saved["lightgbm_probability"].to_numpy(),
            )
            fresh_result = _validation_result(frame, y, fresh_holdout, cat_result)
            fresh_validation_path = (
                output_dir / "validation_longer_catboost_fresh_seed31415.csv"
            )
            frame.to_csv(fresh_validation_path, index=False, float_format="%.17g")
            fresh_result["validation_predictions_path"] = str(fresh_validation_path)
            metadata["fresh_validation"] = fresh_result
            fresh_gain = float(fresh_result["blend"]["roc_auc"]) - fresh_baseline
            metadata["selection"]["fresh_gate"] = {
                "split_seed": FRESH_SPLIT_SEED,
                "baseline_blend_roc_auc": fresh_baseline,
                "new_blend_roc_auc": fresh_result["blend"]["roc_auc"],
                "auc_gain": fresh_gain,
                "minimum_auc_gain": MINIMUM_AUC_GAIN,
                "qualifies": fresh_gain >= MINIMUM_AUC_GAIN,
                "decision": "refit_and_export"
                if fresh_gain >= MINIMUM_AUC_GAIN
                else "retain_current_blend",
            }
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del model, cat_predictions, frame
            gc.collect()
            if not metadata["selection"]["fresh_gate"]["qualifies"]:
                metadata["selection"]["decision"] = "retain_current_blend"
                metadata["status"] = "completed"
            else:
                refit_results = [
                    item["catboost"] for item in metadata["validation_results"]
                ] + [fresh_result["catboost"]]
                counts = [int(result["best_iteration"]) + 1 for result in refit_results]
                iterations = selected_iterations(refit_results)
                metadata["current_fit"] = {
                    "status": "refit_running",
                    "iterations": iterations,
                }
                save_json(metadata_path, metadata)
                model = build_catboost_classifier(
                    iterations=iterations,
                    early_stopping_rounds=None,
                    depth=CATBOOST_DEPTH,
                )
                model.fit(
                    prepare_catboost_frame(X, spec),
                    y,
                    cat_features=list(spec.categorical_columns),
                    verbose=False,
                )
                cat_actual_iterations = int(model.tree_count_)
                cat_test = model.predict_proba(prepare_catboost_frame(X_test, spec))[
                    :, 1
                ]
                validate_probabilities(cat_test, len(X_test))
                cat_model_path = output_dir / "final_model_catboost3000.cbm"
                model.save_model(cat_model_path)
                del model
                lgb_spec_data = json.loads(reused["lgb_spec"].read_text())
                lgb_levels = lgb_spec_data["category_levels"]
                lgb_features = prepare_lightgbm_frame(X_test, spec, lgb_levels)
                lgb_values = np.asarray(
                    lgb.Booster(model_file=str(reused["lgb_model"])).predict(
                        lgb_features
                    ),
                    dtype=np.float64,
                )
                validate_probabilities(lgb_values, len(X_test))
                final_values = 0.5 * cat_test + 0.5 * lgb_values
                final_path = output_dir / "submission_catboost3000_lightgbm_50_50.csv"
                pd.DataFrame(
                    {ID_COLUMN: test[ID_COLUMN], TARGET_COLUMN: final_values}
                ).to_csv(final_path, index=False, float_format="%.17g")
                submission = validate_submission_file(final_path, test[ID_COLUMN])
                cat_spec_path = output_dir / "feature_spec.json"
                save_json(
                    cat_spec_path,
                    {
                        "feature_columns": list(spec.feature_columns),
                        "numeric_columns": list(spec.numeric_columns),
                        "categorical_columns": list(spec.categorical_columns),
                    },
                )
                bundle_spec_path = output_dir / "bundle_spec.json"
                save_json(
                    bundle_spec_path,
                    {
                        "catboost_model_path": str(cat_model_path),
                        "catboost_feature_spec_path": str(cat_spec_path),
                        "lightgbm_model_path": str(reused["lgb_model"]),
                        "lightgbm_feature_spec_path": str(reused["lgb_spec"]),
                        "catboost_iterations": iterations,
                        "lightgbm_iterations": 783,
                        "weights": {"catboost": 0.5, "lightgbm": 0.5},
                        "early_stopping": False,
                        "eval_set": None,
                    },
                )
                reloaded = _fresh_process_reload(
                    test_path,
                    cat_model_path,
                    cat_spec_path,
                    reused["lgb_model"],
                    reused["lgb_spec"],
                )
                reload_difference = float(np.max(np.abs(cat_test - reloaded[:, 0])))
                lgb_reload_difference = float(
                    np.max(np.abs(lgb_values - reloaded[:, 1]))
                )
                csv_difference = float(
                    np.max(np.abs(final_values - submission[TARGET_COLUMN].to_numpy()))
                )
                if (
                    max(reload_difference, lgb_reload_difference, csv_difference)
                    > 1e-12
                ):
                    raise RuntimeError("native model reload or final CSV differs")
                metadata["final"] = {
                    "model": "longer_catboost3000_lightgbm_50_50",
                    "catboost_model_path": str(cat_model_path),
                    "catboost_feature_spec_path": str(cat_spec_path),
                    "lightgbm_model_path": str(reused["lgb_model"]),
                    "lightgbm_feature_spec_path": str(reused["lgb_spec"]),
                    "bundle_spec_path": str(bundle_spec_path),
                    "catboost_best_iteration_counts": counts,
                    "catboost_selected_iterations": iterations,
                    "catboost_actual_iterations": cat_actual_iterations,
                    "lightgbm_selected_iterations": 783,
                    "submission_path": str(final_path),
                    "submission_rows": len(submission),
                    "weights": {"catboost": 0.5, "lightgbm": 0.5},
                    "clean_process_reload": {
                        "rows": len(submission),
                        "max_catboost_difference": reload_difference,
                        "max_lightgbm_difference": lgb_reload_difference,
                        "max_csv_difference": csv_difference,
                    },
                    "model_sha256": {
                        "catboost": sha256_file(cat_model_path),
                        "lightgbm_reused": sha256_file(reused["lgb_model"]),
                    },
                }
                metadata["current_fit"] = None
                metadata["selection"]["decision"] = "refit_and_export"
                metadata["status"] = "completed"
                save_json(metadata_path, metadata)
    except LongerCatBoostTimeout as error:
        caught = error
        metadata["status"] = "timeout"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["error"] = _error_record(error)
    except BaseException as error:
        caught = error
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["final"] = None
        metadata["error"] = _error_record(error)
        if metadata.get("current_fit") is not None:
            metadata["current_fit"]["status"] = "failed"
            metadata["current_fit"]["error"] = _error_record(error)
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, *previous_alarm)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
            signal.signal(signal.SIGTERM, previous_term_handler)
        except BaseException as error:
            if caught is None:
                caught = error
                metadata["error"] = _error_record(error)
            else:
                metadata["finalization_error"] = _error_record(error)
    metadata, caught = _finalize(
        metadata,
        metadata_path,
        output_dir,
        artifacts_dir,
        data_dir,
        input_before,
        protected_before,
        caught,
        started,
    )
    if caught is not None:
        raise caught
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
        description="Run the one longer-CatBoost experiment."
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
                    }
                ),
                sort_keys=True,
            )
        )
        raise
    print(json.dumps(_terminal_summary(metadata), sort_keys=True))


if __name__ == "__main__":
    main()
