"""Run one fixed external-source augmentation experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
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
from compare import (
    _load_recorded_split,
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
    prepare_catboost_frame,
)
from lightgbm_experiment import _artifact_snapshot
from slower_learning import build_slow_catboost
from train import (
    ID_COLUMN,
    TARGET_COLUMN,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)

SEEDS = (42, 2024)
SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
BASELINE_AUC = {42: 0.957794208813, 2024: 0.958196779926}
MINIMUM_AUC_GAIN = 0.0002
INTERNAL_TIMEOUT_SECONDS = 10_500
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
PROGRESS_INTERVAL = 1000


class SourceAugmentationTimeout(TimeoutError):
    """Raised when the authorized experiment budget expires."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise SourceAugmentationTimeout("source augmentation experiment timed out")


def _canonical(
    frame: pd.DataFrame, features: tuple[str, ...], categoricals: tuple[str, ...]
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in features:
        if column in categoricals:
            result[column] = (
                frame[column].astype("string").fillna("<missing>").astype(object)
            )
        else:
            result[column] = pd.to_numeric(frame[column], errors="raise").astype(
                "float64"
            )
    return result.loc[:, list(features)]


def clean_source_rows(
    source: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, spec: Any
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Dedupe source rows and exclude exact feature overlaps without labels."""
    if set(source.columns) != set(spec.feature_columns) | {TARGET_COLUMN}:
        raise ValueError("source columns must equal the 21 predictors plus target")
    features = _canonical(source, spec.feature_columns, spec.categorical_columns)
    labels = source[TARGET_COLUMN].astype(np.int8).to_numpy(copy=True)
    if set(labels) != {0, 1}:
        raise ValueError("source target must contain both binary classes")
    grouped = features.assign(_label=labels).groupby(
        list(spec.feature_columns), dropna=False, sort=False, observed=True
    )
    conflicts = int(grouped["_label"].nunique().gt(1).sum())
    if conflicts:
        raise ValueError("source duplicate feature rows have conflicting labels")
    keep = ~features.duplicated(keep="first")
    features, labels = (
        features.loc[keep].reset_index(drop=True),
        labels[keep.to_numpy()],
    )

    hashes = pd.util.hash_pandas_object(features, index=False).to_numpy(np.uint64)
    candidates: dict[int, list[int]] = {}
    for row, value in enumerate(hashes):
        candidates.setdefault(int(value), []).append(row)
    blocked = np.zeros(len(features), dtype=bool)
    overlaps: dict[str, dict[str, int]] = {}
    for name, other in (("train", train), ("test", test)):
        other_features = _canonical(
            other, spec.feature_columns, spec.categorical_columns
        )
        other_hashes = pd.util.hash_pandas_object(other_features, index=False).to_numpy(
            np.uint64
        )
        matched_source: set[int] = set()
        matched_other: set[int] = set()
        for other_row, value in enumerate(other_hashes):
            for source_row in candidates.get(int(value), []):
                if features.iloc[source_row].equals(other_features.iloc[other_row]):
                    matched_source.add(source_row)
                    matched_other.add(other_row)
        blocked[list(matched_source)] = True
        overlaps[name] = {
            "source_rows": len(matched_source),
            "competition_rows": len(matched_other),
        }
    clean = features.loc[~blocked].reset_index(drop=True)
    return (
        clean,
        labels[~blocked],
        {
            "source_rows": len(source),
            "source_unique_feature_rows": len(features),
            "duplicate_feature_rows_removed": len(source) - len(features),
            "conflicting_label_feature_groups": conflicts,
            "overlap": overlaps,
            "clean_rows": len(clean),
            "clean_unique_feature_rows": len(clean.drop_duplicates()),
            "method": (
                "hash candidates then exact equality on all 21 features; "
                "IDs and labels excluded"
            ),
        },
    )


def augmented_fit_indices(
    inner_fit: np.ndarray, competition_rows: int, source_rows: int
) -> np.ndarray:
    """Return competition inner-fit rows followed by every clean source row."""
    source = np.arange(competition_rows, competition_rows + source_rows, dtype=np.int64)
    result = np.concatenate((np.asarray(inner_fit, dtype=np.int64), source))
    if len(np.unique(result)) != len(result):
        raise RuntimeError("augmented fit rows are not unique")
    return result


def source_gate(
    baseline: dict[int, float], candidate: dict[int, float]
) -> dict[str, Any]:
    """Require at least the fixed AUC gain on both overlapping holdouts."""
    comparison = [
        {
            "split_seed": seed,
            "baseline_roc_auc": baseline[seed],
            "candidate_roc_auc": candidate[seed],
            "auc_gain": candidate[seed] - baseline[seed],
            "qualifies": candidate[seed] - baseline[seed] >= MINIMUM_AUC_GAIN,
        }
        for seed in SEEDS
    ]
    qualifies = all(row["qualifies"] for row in comparison)
    return {
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "comparison": comparison,
        "qualifies": qualifies,
        "decision": "candidate_passed_gate" if qualifies else "retain_current_blend",
    }


def _saved(path: Path, holdout: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_index", ID_COLUMN, "probability"]:
        raise ValueError(f"unexpected saved prediction schema: {path}")
    if frame[ID_COLUMN].duplicated().any() or frame["row_index"].duplicated().any():
        raise ValueError(f"saved prediction IDs or rows are not unique: {path}")
    rows = frame["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(rows), np.sort(holdout)):
        raise ValueError(
            f"saved prediction rows do not match the recorded split: {path}"
        )
    validate_probabilities(frame["probability"].to_numpy(), len(frame))
    return frame


def _input_hashes(data_dir: Path) -> dict[str, str]:
    paths = sorted(data_dir.rglob("*.csv"))
    if len(paths) != 4:
        raise RuntimeError(f"expected exactly four data CSV inputs, found {len(paths)}")
    return {str(path.relative_to(data_dir)): sha256_file(path) for path in paths}


def _fit(
    train: pd.DataFrame,
    y: np.ndarray,
    source: pd.DataFrame,
    source_y: np.ndarray,
    spec: Any,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    holdout: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    competition = prepare_catboost_frame(train, spec)
    source_frame = prepare_catboost_frame(source, spec)
    fit_frame = pd.concat(
        [competition.iloc[inner_fit], source_frame], ignore_index=True
    )
    fit_y = np.concatenate((y[inner_fit], source_y))
    model = build_slow_catboost()
    before = _resource_snapshot()
    started = time.perf_counter()
    model.fit(
        fit_frame,
        fit_y,
        eval_set=(competition.iloc[inner_validation], y[inner_validation]),
        cat_features=list(spec.categorical_columns),
        verbose=PROGRESS_INTERVAL,
    )
    fit_seconds = time.perf_counter() - started
    predictions = model.predict_proba(competition.iloc[holdout])[:, 1]
    validate_probabilities(predictions, len(holdout))
    result = {
        "candidate": "augmented_slow_catboost",
        "fit_seconds": round(fit_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "actual_iterations": int(model.tree_count_),
        "best_iteration": int(model.get_best_iteration()),
        "settings": model.get_params(),
        "resources": _resource_delta(before, _resource_snapshot()),
        "competition_inner_fit_rows": len(inner_fit),
        "source_fit_rows": len(source),
        "augmented_fit_rows": len(fit_frame),
        "early_stopping_rows": len(inner_validation),
        "outer_holdout_rows": len(holdout),
    }
    del model, competition, source_frame, fit_frame
    gc.collect()
    return predictions, result


def _baseline(
    artifacts: Path, train: pd.DataFrame, y: np.ndarray, holdout: np.ndarray, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    ids = train.iloc[holdout][ID_COLUMN].to_numpy()
    slow = _saved(
        artifacts / "slower_learning" / f"validation_slow_catboost_seed{seed}.csv",
        holdout,
    )
    light = _saved(
        artifacts / "lightgbm" / f"validation_lightgbm_seed{seed}.csv", holdout
    )
    aligned = aligned_blend_predictions(
        ids,
        slow[ID_COLUMN],
        slow["probability"].to_numpy(),
        light[ID_COLUMN],
        light["probability"].to_numpy(),
    )
    values = aligned["blend_probability"].to_numpy()
    auc = float(roc_auc_score(y[holdout], values))
    if abs(auc - BASELINE_AUC[seed]) > 1e-9:
        raise RuntimeError(f"current slow-CatBoost baseline changed for seed {seed}")
    return light, {"roc_auc": auc, "log_loss": float(log_loss(y[holdout], values))}


def run(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run the one authorized source augmentation experiment."""
    started = time.perf_counter()
    output = artifacts_dir / "source_augmentation"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output}")
    output.mkdir(parents=True)
    metadata: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": 10_800,
        "progress_interval": PROGRESS_INTERVAL,
        "selection": None,
        "validation_results": [],
        "current_fit": None,
        "license": "CC0-1.0",
        "source_provenance": (
            "Kaggle arseniyshutko; metadata cites educational AI/ML course material"
        ),
        "source_uncertainty": (
            "Source and competition marginals differ materially; this is not an "
            "independent generator confirmation."
        ),
        "rules_note": (
            "Saved competition rules permit reasonably accessible external data "
            "subject to rights and attribution; no publication or submission is "
            "performed."
        ),
    }
    metadata_path = output / "compare_metadata.json"
    save_json(metadata_path, metadata)
    caught: BaseException | None = None
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    old_alarm = signal.getitimer(signal.ITIMER_REAL)
    old_alarm_handler = signal.getsignal(signal.SIGALRM)
    old_term_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        protected_before = _artifact_snapshot(artifacts_dir, output)
        input_before = _input_hashes(data_dir)
        metadata.update(
            {
                "protected_artifact_sha256_before": protected_before,
                "input_sha256_before": input_before,
                "source_sha256": {
                    name: sha256_file(Path(__file__).with_name(name))
                    for name in (
                        "source_augmentation.py",
                        "train.py",
                        "compare.py",
                        "slower_learning.py",
                        "blend_experiment.py",
                    )
                },
                "dependencies": {**package_versions(), "catboost": version("catboost")},
            }
        )
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        source_raw = pd.read_csv(data_dir / "source_airline" / "data.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        source, source_y, source_record = clean_source_rows(
            source_raw, train, test, spec
        )
        metadata.update(
            {
                "source_rows": source_record,
                "feature_columns": list(spec.feature_columns),
                "categorical_columns": list(spec.categorical_columns),
                "model_policy": {
                    "builder": "slower_learning.build_slow_catboost",
                    "ratings_numeric": True,
                    "rating_zero_valid": True,
                    "arrival_delay_nan_preserved": True,
                    "id_and_target_excluded": True,
                    "origin_indicator": False,
                    "sample_weights": False,
                },
                "catboost_settings": build_slow_catboost().get_params(),
            }
        )
        save_json(metadata_path, metadata)
        prior = json.loads(
            (artifacts_dir / "challengers" / "compare_metadata.json").read_text()
        )
        if prior.get("input_sha256") != {
            "train.csv": input_before["train.csv"],
            "test.csv": input_before["test.csv"],
        }:
            raise RuntimeError(
                "prior CatBoost metadata does not match train/test inputs"
            )
        baselines: dict[int, float] = {}
        candidates: dict[int, float] = {}
        for fit_number, seed in enumerate(SEEDS, 1):
            outer_fit, holdout = _load_recorded_split(
                artifacts_dir / SPLIT_FILES[seed], train, y, seed
            )
            prior_inner = prior["internal_validation_splits"][
                "outer" if seed == 42 else "confirmation"
            ]
            inner_fit = np.asarray(prior_inner["fit_indices"], dtype=np.int64)
            inner_validation = np.asarray(
                prior_inner["validation_indices"], dtype=np.int64
            )
            expected_fit, expected_validation = make_catboost_internal_split(
                outer_fit, y, seed=42
            )
            if not np.array_equal(inner_fit, expected_fit) or not np.array_equal(
                inner_validation, expected_validation
            ):
                raise RuntimeError("saved CatBoost inner split does not match seed 42")
            if np.intersect1d(inner_validation, holdout).size:
                raise RuntimeError(
                    "competition inner validation overlaps outer holdout"
                )
            light, baseline = _baseline(artifacts_dir, train, y, holdout, seed)
            baselines[seed] = baseline["roc_auc"]
            augmented = augmented_fit_indices(inner_fit, len(train), len(source))
            metadata["current_fit"] = {
                "fit_number": fit_number,
                "split_seed": seed,
                "status": "running",
                "fit_started_utc": datetime.now(timezone.utc).isoformat(),
                "competition_inner_fit_rows": len(inner_fit),
                "competition_inner_validation_rows": len(inner_validation),
                "source_fit_rows": len(source),
                "augmented_fit_rows": len(augmented),
                "outer_holdout_rows": len(holdout),
            }
            save_json(metadata_path, metadata)
            predictions, result = _fit(
                train, y, source, source_y, spec, inner_fit, inner_validation, holdout
            )
            aligned = aligned_blend_predictions(
                train.iloc[holdout][ID_COLUMN].to_numpy(),
                train.iloc[holdout][ID_COLUMN].to_numpy(),
                predictions,
                light[ID_COLUMN],
                light["probability"].to_numpy(),
            )
            blend = aligned["blend_probability"].to_numpy()
            candidates[seed] = float(roc_auc_score(y[holdout], blend))
            path = output / f"validation_augmented_catboost_seed{seed}.csv"
            aligned.insert(0, "row_index", holdout)
            aligned.rename(
                columns={
                    "catboost_probability": "augmented_catboost_probability",
                    "lightgbm_probability": "original_lightgbm_probability",
                }
            ).to_csv(path, index=False, float_format="%.17g")
            result.update(
                {
                    "split_seed": seed,
                    "baseline_blend_roc_auc": baseline["roc_auc"],
                    "baseline_blend_log_loss": baseline["log_loss"],
                    "candidate_catboost_roc_auc": float(
                        roc_auc_score(y[holdout], predictions)
                    ),
                    "candidate_catboost_log_loss": float(
                        log_loss(y[holdout], predictions)
                    ),
                    "blend_roc_auc": candidates[seed],
                    "blend_log_loss": float(log_loss(y[holdout], blend)),
                    "auc_gain_vs_current_blend": candidates[seed] - baseline["roc_auc"],
                    "validation_predictions_path": str(path),
                }
            )
            metadata["validation_results"].append(result)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del light, predictions, aligned
            gc.collect()
        metadata["selection"] = source_gate(baselines, candidates)
    except BaseException as error:
        caught = error
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm_handler)
        signal.signal(signal.SIGTERM, old_term_handler)
        if old_alarm[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_alarm[0], old_alarm[1])
        metadata["status"] = (
            "timeout"
            if isinstance(caught, SourceAugmentationTimeout)
            else "failed"
            if caught is not None
            else "completed"
        )
        if caught is not None:
            metadata["error"] = {"type": type(caught).__name__, "message": str(caught)}
        try:
            metadata["input_sha256_after"] = _input_hashes(data_dir)
        except BaseException as error:
            metadata["input_sha256_after_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        try:
            protected_after = _artifact_snapshot(artifacts_dir, output)
            metadata["protected_artifact_sha256_after"] = protected_after
            metadata["protected_artifacts_unchanged"] = (
                protected_before == protected_after
            )
        except BaseException as error:
            metadata["protected_artifact_snapshot_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        metadata["input_artifacts_unchanged"] = (
            input_before is not None
            and metadata.get("input_sha256_after") == input_before
        )
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        save_json(metadata_path, metadata)
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        )
        save_json(metadata_path, metadata)
    integrity: RuntimeError | None = None
    if not metadata.get("input_artifacts_unchanged", False):
        integrity = RuntimeError(
            "input CSV fingerprint changed during source augmentation"
        )
    elif not metadata.get("protected_artifacts_unchanged", False):
        integrity = RuntimeError("existing artifact changed during source augmentation")
    elif metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
        integrity = RuntimeError("source augmentation output exceeded 100 MiB")
    if integrity is not None:
        metadata["status"] = "failed"
        metadata["integrity_error"] = {
            "type": type(integrity).__name__,
            "message": str(integrity),
        }
        save_json(metadata_path, metadata)
    if caught is not None:
        if integrity is not None:
            raise caught from integrity
        raise caught
    if integrity is not None:
        raise integrity
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    result = run(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selection": result["selection"],
                "validation_results": result["validation_results"],
                "output_bytes": result["output_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
