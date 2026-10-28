"""Run the fixed two-seed LightGBM num_leaves=63 capacity diagnostic."""

from __future__ import annotations

# Scientific imports intentionally follow the thread-limit environment setup.
# ruff: noqa: E402
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from compare import (
    _load_recorded_split,
    make_catboost_internal_split,
)
from lightgbm_experiment import (
    PREDECLARED_CONFIG,
    _artifact_snapshot,
    _fit_callbacks,
    _fit_with_inner_validation,
    build_lightgbm_classifier,
    prepare_lightgbm_frame,
)
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

SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
SPLIT_SEEDS = tuple(SPLIT_FILES)
INTERNAL_SEED = 42
CAPACITY_NUM_LEAVES = 63
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_NAME = "lightgbm_capacity63"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 500
BASELINE_AUC = {42: 0.9577942088132286, 2024: 0.9581967799255812}
EXPECTED_INPUT_FILES = {
    "sample_submission.csv",
    "source_airline/data.csv",
    "test.csv",
    "train.csv",
}

CAPACITY_CONFIG = dict(PREDECLARED_CONFIG)
CAPACITY_CONFIG["num_leaves"] = CAPACITY_NUM_LEAVES


class LightGBMCapacityTimeout(TimeoutError):
    """Raised when the fixed capacity diagnostic reaches its deadline."""


def build_lightgbm_capacity_classifier() -> lgb.LGBMClassifier:
    """Build the approved classifier with only num_leaves changed to 63."""
    return build_lightgbm_classifier(num_leaves=CAPACITY_NUM_LEAVES)


def capacity_gate(gains: dict[int, float]) -> dict[str, Any]:
    """Require the fixed AUC gain independently on both recorded holdouts."""
    qualifies = all(
        gains.get(seed, float("-inf")) >= MINIMUM_AUC_GAIN for seed in SPLIT_SEEDS
    )
    return {
        "qualifies": qualifies,
        "decision": (
            "select_lightgbm63_blend" if qualifies else "retain_current_blend"
        ),
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "gains_vs_current": {str(seed): gains.get(seed) for seed in SPLIT_SEEDS},
    }


def fixed_capacity_blend_predictions(
    reference_ids: Any,
    slowcat_ids: Any,
    slowcat_values: np.ndarray,
    original_lightgbm_ids: Any,
    original_lightgbm_values: np.ndarray,
    lightgbm63_ids: Any,
    lightgbm63_values: np.ndarray,
) -> pd.DataFrame:
    """Align the saved components and calculate the two fixed 50/50 blends."""
    current = weighted_blend_predictions(
        reference_ids,
        slowcat_ids,
        slowcat_values,
        original_lightgbm_ids,
        original_lightgbm_values,
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    candidate = weighted_blend_predictions(
        reference_ids,
        slowcat_ids,
        slowcat_values,
        lightgbm63_ids,
        lightgbm63_values,
        slow_catboost_weight=0.5,
        lightgbm_weight=0.5,
    )
    if not np.array_equal(current[ID_COLUMN], candidate[ID_COLUMN]):
        raise ValueError("current and candidate blend ID orders differ")
    result = pd.DataFrame(
        {
            ID_COLUMN: candidate[ID_COLUMN],
            "lightgbm63_probability": candidate["lightgbm_probability"],
            "slowcat_probability": candidate["slow_catboost_probability"],
            "baseline_probability": current["blend_probability"],
            "candidate_probability": candidate["blend_probability"],
        }
    )
    validate_probabilities(result["baseline_probability"].to_numpy(), len(result))
    validate_probabilities(result["candidate_probability"].to_numpy(), len(result))
    return result


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    validate_probabilities(probabilities, len(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _input_hashes(data_dir: Path) -> dict[str, str]:
    hashes = {
        str(path.relative_to(data_dir)): sha256_file(path)
        for path in sorted(data_dir.rglob("*.csv"))
    }
    missing = EXPECTED_INPUT_FILES - set(hashes)
    if missing:
        raise FileNotFoundError(f"required input CSVs are missing: {sorted(missing)}")
    return hashes


def _array_fingerprint(array: np.ndarray) -> str:
    value = np.asarray(array, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _array_record(*arrays: tuple[str, np.ndarray]) -> dict[str, Any]:
    return {
        "rows": {name: len(value) for name, value in arrays},
        "sha256": {name: _array_fingerprint(value) for name, value in arrays},
    }


def _source_hashes() -> dict[str, str]:
    names = (
        "lightgbm_capacity.py",
        "lightgbm_experiment.py",
        "weighted_blend_experiment.py",
        "compare.py",
        "train.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _package_versions() -> dict[str, str]:
    packages = package_versions()
    packages["lightgbm"] = version("lightgbm")
    return packages


def _config_hash() -> str:
    encoded = json.dumps(CAPACITY_CONFIG, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_components(
    path: Path, train: pd.DataFrame, holdout_indices: np.ndarray
) -> pd.DataFrame:
    frame = _read_predictions(path)
    expected_rows = np.asarray(holdout_indices, dtype=np.int64)
    expected_ids = train.iloc[holdout_indices][ID_COLUMN].to_numpy()
    if not np.array_equal(frame["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError(f"saved prediction rows do not match recorded split: {path}")
    if not np.array_equal(frame[ID_COLUMN].to_numpy(), expected_ids):
        raise ValueError(f"saved prediction IDs do not match recorded split: {path}")
    return frame


def _load_split_arrays(
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    prior: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    split_name = "outer" if seed == 42 else "confirmation"
    split_path = artifacts_dir / SPLIT_FILES[seed]
    outer_fit, outer_holdout = _load_recorded_split(split_path, train, y, seed)
    recorded = prior["recorded_splits"][split_name]
    recorded_fit = np.asarray(recorded["fit_indices"], dtype=np.int64)
    recorded_holdout = np.asarray(recorded["holdout_indices"], dtype=np.int64)
    inner_fit = np.asarray(recorded["inner_fit_indices"], dtype=np.int64)
    inner_validation = np.asarray(recorded["inner_validation_indices"], dtype=np.int64)
    if not np.array_equal(outer_fit, recorded_fit) or not np.array_equal(
        outer_holdout, recorded_holdout
    ):
        raise RuntimeError(
            f"recorded {split_name} split disagrees with LightGBM metadata"
        )
    expected_inner_fit, expected_inner_validation = make_catboost_internal_split(
        outer_fit, y, seed=INTERNAL_SEED
    )
    if not np.array_equal(inner_fit, expected_inner_fit) or not np.array_equal(
        inner_validation, expected_inner_validation
    ):
        raise RuntimeError(f"recorded {split_name} inner split disagrees with seed 42")
    if (
        np.intersect1d(inner_fit, outer_holdout).size
        or np.intersect1d(inner_validation, outer_holdout).size
    ):
        raise RuntimeError("inner split overlaps outer holdout")
    return (
        outer_fit,
        outer_holdout,
        inner_fit,
        inner_validation,
        {
            "seed": seed,
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "outer_holdout_used_for_eval": False,
            "arrays": _array_record(
                ("outer_fit", outer_fit),
                ("outer_holdout", outer_holdout),
                ("inner_fit", inner_fit),
                ("inner_validation", inner_validation),
            ),
        },
    )


def _progress_callback(
    metadata: dict[str, Any], metadata_path: Path, split_seed: int
) -> Any:
    def callback(environment: Any) -> None:
        tree = int(environment.iteration) + 1
        if tree == 0 or tree % PROGRESS_INTERVAL:
            return
        current = metadata.get("current_fit")
        if isinstance(current, dict):
            event = {"utc": datetime.now(timezone.utc).isoformat(), "tree": tree}
            current.setdefault("progress_log", []).append(event)
            current["last_progress_tree"] = tree
            save_json(metadata_path, metadata)
        print(
            f"{datetime.now(timezone.utc).isoformat()} split_seed={split_seed} "
            f"tree={tree}",
            flush=True,
        )

    callback.order = 20  # type: ignore[attr-defined]
    return callback


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run exactly two sequential LightGBM fits and save no model files."""
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
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "completed_split_seeds": [],
        "output_dir": str(output_dir),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "predeclared_settings": dict(CAPACITY_CONFIG),
    }
    save_json(metadata_path, metadata)

    caught_error: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    timer_active = False
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)

    def on_timeout(_signum: int, _frame: Any) -> None:
        raise LightGBMCapacityTimeout("LightGBM capacity diagnostic timed out")

    try:
        signal.signal(signal.SIGALRM, on_timeout)
        signal.signal(signal.SIGTERM, on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        timer_active = True
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
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
                "source_sha256": _source_hashes(),
                "config_sha256": _config_hash(),
                "protected_artifact_sha256_before": protected_before,
                "feature_change": {"only": "num_leaves", "from": 31, "to": 63},
                "early_stopping": {
                    "rounds": 100,
                    "metric": "auc",
                    "callbacks": ["early_stopping", "log_evaluation", "progress"],
                },
                "split_seeds": list(SPLIT_SEEDS),
                "model_seed": CAPACITY_CONFIG["random_state"],
                "internal_validation_seed": INTERNAL_SEED,
                "reused_artifacts": {},
            }
        )
        save_json(metadata_path, metadata)

        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        X = train.loc[:, list(spec.feature_columns)]
        prior_path = artifacts_dir / "lightgbm" / "compare_metadata.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("status") != "completed":
            raise RuntimeError("prior corrected LightGBM metadata is not completed")
        prior_config = prior.get("predeclared_config", {})
        expected_prior_config = dict(CAPACITY_CONFIG)
        expected_prior_config["num_leaves"] = 31
        if prior_config != expected_prior_config:
            raise RuntimeError(
                "prior LightGBM metadata is not the corrected num_leaves=31 run"
            )
        if {
            name: input_before[name] for name in ("train.csv", "test.csv")
        } != prior.get("input_sha256_before"):
            raise RuntimeError(
                "input fingerprints do not match prior LightGBM metadata"
            )
        metadata["reused_artifacts"]["lightgbm/compare_metadata.json"] = sha256_file(
            prior_path
        )
        metadata["feature_columns"] = list(spec.feature_columns)
        metadata["numeric_columns"] = list(spec.numeric_columns)
        metadata["categorical_columns"] = list(spec.categorical_columns)

        split_arrays: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        components: dict[int, dict[str, pd.DataFrame]] = {}
        split_records: dict[str, Any] = {}
        for seed in SPLIT_SEEDS:
            outer_fit, outer_holdout, inner_fit, inner_validation, record = (
                _load_split_arrays(artifacts_dir, train, y, prior, seed)
            )
            split_name = "outer" if seed == 42 else "confirmation"
            split_records[split_name] = record
            split_arrays[seed] = (outer_fit, outer_holdout, inner_fit, inner_validation)
            slow_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            original_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            slow = _load_components(slow_path, train, outer_holdout)
            original = _load_components(original_path, train, outer_holdout)
            components[seed] = {"slowcat": slow, "original_lightgbm": original}
            for path in (slow_path, original_path):
                metadata["reused_artifacts"][str(path.relative_to(artifacts_dir))] = (
                    sha256_file(path)
                )
        metadata["recorded_splits"] = split_records
        save_json(metadata_path, metadata)

        gains: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            outer_fit, outer_holdout, inner_fit, inner_validation = split_arrays[seed]
            slow = components[seed]["slowcat"]
            original = components[seed]["original_lightgbm"]
            expected_ids = train.iloc[outer_holdout][ID_COLUMN].to_numpy()
            preflight = fixed_capacity_blend_predictions(
                expected_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                original[ID_COLUMN].to_numpy(),
                original["probability"].to_numpy(),
                original[ID_COLUMN].to_numpy(),
                original["probability"].to_numpy(),
            )
            baseline_metrics = _metrics(
                y[outer_holdout], preflight["baseline_probability"].to_numpy()
            )
            if abs(baseline_metrics["roc_auc"] - BASELINE_AUC[seed]) > 1e-12:
                raise RuntimeError(f"recomputed baseline blend changed for seed {seed}")
            metadata["current_fit"] = {
                "split_seed": seed,
                "status": "running",
                "fit_rows": len(inner_fit),
                "inner_validation_rows": len(inner_validation),
                "outer_holdout_rows": len(outer_holdout),
                "baseline_current_blend_50_50": baseline_metrics,
                "progress_log": [],
            }
            save_json(metadata_path, metadata)

            model = build_lightgbm_capacity_classifier()
            levels, fit_seconds, fit_details = _fit_with_inner_validation(
                model,
                X,
                y,
                inner_fit,
                inner_validation,
                spec,
                callbacks=_fit_callbacks()
                + [_progress_callback(metadata, metadata_path, seed)],
            )
            predictions = model.predict_proba(
                prepare_lightgbm_frame(X.iloc[outer_holdout], spec, levels)
            )[:, 1]
            validate_probabilities(predictions, len(outer_holdout))
            blended = fixed_capacity_blend_predictions(
                expected_ids,
                slow[ID_COLUMN].to_numpy(),
                slow["probability"].to_numpy(),
                original[ID_COLUMN].to_numpy(),
                original["probability"].to_numpy(),
                expected_ids,
                predictions,
            )
            standalone_metrics = _metrics(y[outer_holdout], predictions)
            candidate_metrics = _metrics(
                y[outer_holdout], blended["candidate_probability"].to_numpy()
            )
            validation_path = output_dir / f"validation_lightgbm63_seed{seed}.csv"
            output = pd.DataFrame(
                {
                    "row_index": outer_holdout,
                    ID_COLUMN: blended[ID_COLUMN],
                    "lightgbm63_probability": blended["lightgbm63_probability"],
                    "slowcat_probability": blended["slowcat_probability"],
                    "baseline_probability": blended["baseline_probability"],
                    "candidate_probability": blended["candidate_probability"],
                }
            )
            output.to_csv(validation_path, index=False, float_format="%.17g")
            result = {
                "split_seed": seed,
                "standalone_lightgbm63": standalone_metrics,
                "current_blend_50_50": baseline_metrics,
                "candidate_blend_50_50": candidate_metrics,
                "settings": dict(CAPACITY_CONFIG),
                "gain_vs_current": (
                    candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"]
                ),
                "fit_seconds": round(fit_seconds, 3),
                "best_iteration": fit_details["best_iteration"],
                "actual_iterations": fit_details["actual_iterations"],
                "resources": fit_details["resources"],
                "validation_predictions_path": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
                "internal_validation": {
                    "seed": INTERNAL_SEED,
                    "fit_rows": len(inner_fit),
                    "validation_rows": len(inner_validation),
                    "outer_holdout_used": False,
                },
            }
            metadata["validation_results"].append(result)
            metadata["completed_split_seeds"].append(seed)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            gains[seed] = result["gain_vs_current"]
            del model, levels, predictions
            gc.collect()
        metadata["selection"] = capacity_gate(gains)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = (
            "timeout" if isinstance(error, LightGBMCapacityTimeout) else "failed"
        )
        metadata["error"] = _error_record(error)
        metadata["selection"] = None
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
        try:
            source_after = _source_hashes()
            metadata["source_sha256_after"] = source_after
            if (
                metadata.get("source_sha256")
                and source_after != metadata["source_sha256"]
            ):
                finalization_errors.append(RuntimeError("source fingerprint changed"))
        except BaseException as error:
            finalization_errors.append(error)
        metadata["current_fit"] = None
        if finalization_errors:
            finalization_error = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata["status"] = "failed"
            metadata["integrity_error"] = _error_record(finalization_error)
            if caught_error is None:
                caught_error = finalization_error
                metadata["error"] = _error_record(finalization_error)
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("output exceeded 100 MiB")
            metadata["status"] = "failed"
            metadata["error"] = _error_record(budget_error)
            if caught_error is None:
                caught_error = budget_error
        if metadata["status"] != "completed":
            metadata["selection"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        try:
            save_json(
                output_dir / "protected_baseline_sha256.json",
                metadata.get("protected_artifact_sha256_after", protected_before or {}),
            )
            # Include final bookkeeping files and the size field's own digits.
            for _ in range(4):
                save_json(metadata_path, metadata)
                actual_bytes = sum(
                    path.stat().st_size
                    for path in output_dir.rglob("*")
                    if path.is_file()
                )
                if actual_bytes > OUTPUT_BUDGET_BYTES:
                    raise RuntimeError("output exceeded 100 MiB after finalization")
                if actual_bytes == metadata["output_bytes"]:
                    break
                metadata["output_bytes"] = actual_bytes
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught_error is None:
                caught_error = error
                metadata["error"] = _error_record(error)
            else:
                metadata["finalization_error"] = _error_record(error)
            save_json(metadata_path, metadata)
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
