from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
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

from lightgbm_experiment import _artifact_snapshot
from train import (
    ID_COLUMN,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)

SPLIT_SEEDS = (42, 2024)
SLOW_CATBOOST_WEIGHT = 0.75
ORIGINAL_LIGHTGBM_WEIGHT = 0.25
CURRENT_WEIGHT = 0.5
MINIMUM_AUC_GAIN = 0.0002
OUTPUT_NAME = "blend_75_25"


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_index", ID_COLUMN, "probability"]:
        raise ValueError(f"saved predictions have unexpected schema: {path}")
    if frame["row_index"].duplicated().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"saved predictions contain duplicate keys: {path}")
    validate_probabilities(frame["probability"].to_numpy(), len(frame))
    return frame


def weighted_blend_predictions(
    reference_ids: Sequence[Any],
    slow_catboost_ids: Sequence[Any],
    slow_catboost_values: np.ndarray,
    lightgbm_ids: Sequence[Any],
    lightgbm_values: np.ndarray,
    *,
    slow_catboost_weight: float = SLOW_CATBOOST_WEIGHT,
    lightgbm_weight: float = ORIGINAL_LIGHTGBM_WEIGHT,
) -> pd.DataFrame:
    """Align fixed component probabilities by unique ID and blend them."""
    if not np.isclose(slow_catboost_weight + lightgbm_weight, 1.0):
        raise ValueError("blend weights must sum to one")
    reference = pd.Index(reference_ids)
    if not reference.is_unique:
        raise ValueError("reference IDs must be unique")
    frames: list[pd.Series] = []
    for name, ids, values in (
        ("slow CatBoost", slow_catboost_ids, slow_catboost_values),
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
    if any(frame.isna().any() for frame in frames):
        raise ValueError("ID alignment produced missing predictions")
    slow_catboost = frames[0].to_numpy()
    lightgbm = frames[1].to_numpy()
    return pd.DataFrame(
        {
            ID_COLUMN: reference.to_numpy(),
            "slow_catboost_probability": slow_catboost,
            "lightgbm_probability": lightgbm,
            "blend_probability": (
                slow_catboost_weight * slow_catboost + lightgbm_weight * lightgbm
            ),
        }
    )


def blend_gate(gains: dict[int, float]) -> dict[str, Any]:
    """Apply the predeclared minimum AUC gain on both fixed seeds."""
    qualifies = all(gains[seed] >= MINIMUM_AUC_GAIN for seed in SPLIT_SEEDS)
    return {
        "qualifies": qualifies,
        "decision": "select_75_25" if qualifies else "retain_current_blend",
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _versions() -> dict[str, str]:
    result = package_versions()
    result["scikit-learn"] = version("scikit-learn")
    return result


def _source_hashes() -> dict[str, str]:
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("weighted_blend_experiment.py", "train.py")
    }


def _score_candidate(
    train: pd.DataFrame,
    slow_catboost_path: Path,
    lightgbm_path: Path,
    *,
    slow_catboost_weight: float,
    lightgbm_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    slow_catboost = _read_predictions(slow_catboost_path)
    lightgbm = _read_predictions(lightgbm_path)
    if not np.array_equal(slow_catboost["row_index"], lightgbm["row_index"]):
        raise ValueError("saved prediction row_index values do not match")
    y_by_id = pd.Series(target_array(train), index=train[ID_COLUMN])
    aligned = weighted_blend_predictions(
        slow_catboost[ID_COLUMN],
        slow_catboost[ID_COLUMN],
        slow_catboost["probability"].to_numpy(),
        lightgbm[ID_COLUMN],
        lightgbm["probability"].to_numpy(),
        slow_catboost_weight=slow_catboost_weight,
        lightgbm_weight=lightgbm_weight,
    )
    y_true = y_by_id.reindex(aligned[ID_COLUMN]).to_numpy()
    if pd.isna(y_true).any():
        raise ValueError("saved predictions contain unknown IDs")
    probabilities = aligned["blend_probability"].to_numpy()
    return aligned, {
        "rows": len(aligned),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Score the one fixed 75/25 blend without fitting any model."""
    started = time.perf_counter()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / "compare_metadata.json"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    input_before = {
        "train.csv": sha256_file(train_path),
        "test.csv": sha256_file(test_path),
    }
    protected_before = _artifact_snapshot(artifacts_dir, output_dir)
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "threads": 2,
        "python": sys.version,
        "dependencies": _versions(),
        "source_sha256": _source_hashes(),
        "input_sha256_before": input_before,
        "protected_artifact_sha256_before": protected_before,
        "weights": {
            "slow_catboost": SLOW_CATBOOST_WEIGHT,
            "original_lightgbm": ORIGINAL_LIGHTGBM_WEIGHT,
        },
        "output_dir": str(output_dir),
        "validation_results": [],
        "selection": None,
    }
    save_json(metadata_path, metadata)
    caught_error: BaseException | None = None
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        spec = validate_inputs(train, test)
        metadata["feature_columns"] = list(spec.feature_columns)
        metadata["numeric_columns"] = list(spec.numeric_columns)
        metadata["categorical_columns"] = list(spec.categorical_columns)
        current_results: dict[int, dict[str, Any]] = {}
        new_results: dict[int, dict[str, Any]] = {}
        for seed in SPLIT_SEEDS:
            slow_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            light_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            current_frame, current = _score_candidate(
                train,
                slow_path,
                light_path,
                slow_catboost_weight=CURRENT_WEIGHT,
                lightgbm_weight=CURRENT_WEIGHT,
            )
            new_frame, candidate = _score_candidate(
                train,
                slow_path,
                light_path,
                slow_catboost_weight=SLOW_CATBOOST_WEIGHT,
                lightgbm_weight=ORIGINAL_LIGHTGBM_WEIGHT,
            )
            if not np.array_equal(current_frame[ID_COLUMN], new_frame[ID_COLUMN]):
                raise RuntimeError("current and candidate ID orders differ")
            new_path = output_dir / f"validation_blend_75_25_seed{seed}.csv"
            pd.DataFrame(
                {
                    "row_index": _read_predictions(slow_path)["row_index"].to_numpy(),
                    ID_COLUMN: new_frame[ID_COLUMN],
                    "probability": new_frame["blend_probability"],
                }
            ).to_csv(new_path, index=False, float_format="%.17g")
            current_results[seed] = current
            new_results[seed] = candidate
            metadata["validation_results"].append(
                {
                    "split_seed": seed,
                    "rows": candidate["rows"],
                    "current_50_50": current,
                    "candidate_75_25": candidate,
                    "gain_vs_current": candidate["roc_auc"] - current["roc_auc"],
                    "validation_path": str(new_path),
                    "validation_sha256": sha256_file(new_path),
                }
            )
            save_json(metadata_path, metadata)
        gains = {
            int(result["split_seed"]): float(result["gain_vs_current"])
            for result in metadata["validation_results"]
        }
        metadata["selection"] = blend_gate(gains)
        metadata["status"] = "completed"
    except BaseException as error:
        caught_error = error
        metadata["status"] = "failed"
        metadata["selection"] = None
        metadata["error"] = _error_record(error)
    finally:
        metadata["input_sha256_after"] = {
            "train.csv": sha256_file(train_path),
            "test.csv": sha256_file(test_path),
        }
        protected_after = _artifact_snapshot(artifacts_dir, output_dir)
        metadata["protected_artifact_sha256_after"] = protected_after
        metadata["protected_artifacts_unchanged"] = protected_before == protected_after
        if metadata["input_sha256_after"] != input_before:
            metadata["status"] = "failed"
            metadata["integrity_error"] = "input CSV fingerprint changed"
            if caught_error is None:
                caught_error = RuntimeError("input CSV fingerprint changed")
                metadata["error"] = _error_record(caught_error)
        if not metadata["protected_artifacts_unchanged"]:
            metadata["status"] = "failed"
            metadata["integrity_error"] = "pre-existing artifact changed"
            if caught_error is None:
                caught_error = RuntimeError("pre-existing artifact changed")
                metadata["error"] = _error_record(caught_error)
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        save_json(output_dir / "protected_baseline_sha256.json", protected_after)
        save_json(metadata_path, metadata)
    if caught_error is not None:
        raise caught_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the fixed 75/25 blend.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
