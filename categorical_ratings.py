"""Run the authorized native-categorical service-rating experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from blend_experiment import aligned_blend_predictions
from compare import (
    _load_recorded_split,
    _resource_delta,
    _resource_snapshot,
    make_catboost_internal_split,
)
from feature_experiment import RATING_COLUMNS
from slower_learning import build_slow_catboost
from train import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    InputSpec,
    package_versions,
    save_json,
    sha256_file,
    target_array,
    validate_inputs,
    validate_probabilities,
)

MINIMUM_AUC_GAIN = 0.0002
SEEDS = (42, 2024)
OUTPUT_NAME = "categorical_ratings"
SECOND_SPLIT_OUTPUT_NAME = "categorical_ratings_seed2024"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
INTERNAL_TIMEOUT_SECONDS = 55 * 60
SECOND_SPLIT_INTERNAL_TIMEOUT_SECONDS = 85 * 60
MISSING_CATEGORY = "<missing>"
SPLIT_FILES = {42: "outer_split.npz", 2024: "confirmation_split.npz"}
BASELINE_AUC = {42: 0.9577942088132286, 2024: 0.9581967799255812}


class CategoricalRatingsTimeout(TimeoutError):
    """Raised when the bounded categorical-ratings run reaches its deadline."""


def _on_timeout(_signum: int, _frame: Any) -> None:
    raise CategoricalRatingsTimeout("categorical-ratings experiment timed out")


def categorical_columns(spec: InputSpec) -> tuple[str, ...]:
    """Return the four nominal and thirteen rating categorical columns."""
    if tuple(spec.categorical_columns) != CATEGORICAL_COLUMNS:
        raise ValueError(
            "categorical-ratings experiment requires the original four nominal columns"
        )
    return tuple(spec.categorical_columns) + tuple(RATING_COLUMNS)


def _rating_category_values(series: pd.Series, column: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"missing rating values are not allowed in {column}")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"non-numeric rating values are not allowed in {column}")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"rating values must be integral in {column}")
    if ((values < 0) | (values > 5)).any():
        raise ValueError(f"rating values in {column} must be in the allowed range")
    return pd.Series(
        values.astype(np.int64).astype(str), index=series.index, dtype=object
    )


def prepare_rating_categorical_frame(
    frame: pd.DataFrame, spec: InputSpec
) -> pd.DataFrame:
    """Copy features and convert only ratings to stable native-category tokens."""
    prepared = frame.loc[:, list(spec.feature_columns)].copy()
    for column in spec.categorical_columns:
        prepared[column] = (
            prepared[column].astype("string").fillna(MISSING_CATEGORY).astype(object)
        )
    for column in RATING_COLUMNS:
        prepared[column] = _rating_category_values(frame[column], column)
    return prepared


def build_categorical_ratings_catboost() -> CatBoostClassifier:
    """Build the unchanged slower-learning CatBoost configuration."""
    return build_slow_catboost()


def categorical_gate(
    baseline_auc_by_seed: dict[int, float], candidate_auc_by_seed: dict[int, float]
) -> dict[str, Any]:
    """Require the predeclared AUC gain on both recorded seeds."""
    comparison = []
    for seed in SEEDS:
        if seed not in baseline_auc_by_seed or seed not in candidate_auc_by_seed:
            continue
        gain = candidate_auc_by_seed[seed] - baseline_auc_by_seed[seed]
        comparison.append(
            {
                "split_seed": seed,
                "baseline_roc_auc": baseline_auc_by_seed[seed],
                "candidate_roc_auc": candidate_auc_by_seed[seed],
                "auc_gain": gain,
                "qualifies": gain >= MINIMUM_AUC_GAIN,
            }
        )
    qualifies = len(comparison) == len(SEEDS) and all(
        item["qualifies"] for item in comparison
    )
    return {
        "minimum_auc_gain_each_seed": MINIMUM_AUC_GAIN,
        "comparison": comparison,
        "qualifies": qualifies,
        "decision": (
            "request_approval_for_fresh_confirmation"
            if qualifies
            else "retain_current_blend"
        ),
    }


def _source_hashes() -> dict[str, str]:
    names = (
        "categorical_ratings.py",
        "train.py",
        "compare.py",
        "slower_learning.py",
        "feature_experiment.py",
        "blend_experiment.py",
        "lightgbm_experiment.py",
    )
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _dependency_versions() -> dict[str, str]:
    dependencies = package_versions()
    for name in ("catboost", "lightgbm", "threadpoolctl"):
        dependencies[name] = version(name)
    return dependencies


def _artifact_snapshot(artifacts_dir: Path, output_dir: Path) -> dict[str, str]:
    """Hash every artifact except this run's isolated output directory."""
    output_relative = output_dir.relative_to(artifacts_dir)
    hashes: dict[str, str] = {}
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(artifacts_dir)
        if relative == output_relative or output_relative in relative.parents:
            continue
        hashes[str(relative)] = sha256_file(path)
    return hashes


def _array_fingerprint(array: np.ndarray) -> str:
    value = np.asarray(array, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _recorded_array_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(_array_fingerprint(arrays[name]).encode("ascii"))
    return digest.hexdigest()


def _input_hashes(data_dir: Path) -> dict[str, str]:
    return {
        "train.csv": sha256_file(data_dir / "train.csv"),
        "test.csv": sha256_file(data_dir / "test.csv"),
    }


def _read_saved_predictions(path: Path, holdout_indices: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_index", ID_COLUMN, "probability"]:
        raise ValueError(f"unexpected saved prediction schema: {path}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"saved prediction IDs are not unique: {path}")
    if not np.array_equal(
        np.sort(frame["row_index"].to_numpy(dtype=np.int64)), np.sort(holdout_indices)
    ):
        raise ValueError(
            f"saved prediction rows do not match the recorded split: {path}"
        )
    validate_probabilities(frame["probability"].to_numpy(), len(frame))
    return frame


def _run_configuration(complete_second_split: bool) -> dict[str, Any]:
    """Return the bounded run plan without changing the default experiment."""
    if complete_second_split:
        return {
            "seeds": (2024,),
            "output_name": SECOND_SPLIT_OUTPUT_NAME,
            "internal_timeout_seconds": SECOND_SPLIT_INTERNAL_TIMEOUT_SECONDS,
        }
    return {
        "seeds": SEEDS,
        "output_name": OUTPUT_NAME,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
    }


def _load_reused_seed42_summary(
    artifacts_dir: Path,
    data_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    input_hashes: dict[str, str],
    cat_columns: tuple[str, ...],
) -> dict[str, Any]:
    """Verify and summarize the prior seed-42 result for the second split."""
    metadata_path = artifacts_dir / OUTPUT_NAME / "compare_metadata.json"
    prior = json.loads(metadata_path.read_text(encoding="utf-8"))
    if prior.get("input_sha256_before") != input_hashes:
        raise RuntimeError("prior categorical-ratings input provenance changed")
    if (
        prior.get("catboost_settings")
        != build_categorical_ratings_catboost().get_params()
    ):
        raise RuntimeError("prior categorical-ratings CatBoost configuration changed")
    if prior.get("catboost_cat_features") != list(cat_columns):
        raise RuntimeError("prior categorical-ratings categorical columns changed")
    prior_results = [
        result
        for result in prior.get("validation_results", [])
        if result.get("split_seed") == 42
    ]
    if len(prior_results) != 1:
        raise RuntimeError("prior categorical-ratings seed-42 result is not unique")

    result = prior_results[0]
    split_path = artifacts_dir / SPLIT_FILES[42]
    _, holdout = _load_recorded_split(split_path, train, y, 42)
    validation_path = Path(result["validation_predictions_path"])
    if not validation_path.is_absolute():
        validation_path = data_dir.parent / validation_path
    validation = pd.read_csv(validation_path)
    expected_columns = [
        "row_index",
        ID_COLUMN,
        "categorical_catboost_probability",
        "original_lightgbm_probability",
        "blend_probability",
    ]
    if list(validation.columns) != expected_columns:
        raise ValueError("prior categorical-ratings seed-42 CSV schema changed")
    if not np.array_equal(
        np.sort(validation["row_index"].to_numpy(dtype=np.int64)), np.sort(holdout)
    ):
        raise ValueError("prior categorical-ratings seed-42 CSV rows changed")
    for column in expected_columns[2:]:
        validate_probabilities(validation[column].to_numpy(), len(validation))

    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "status": prior.get("status"),
        "error": prior.get("error"),
        "validation_predictions_path": str(validation_path),
        "validation_predictions_sha256": sha256_file(validation_path),
        "validation_result": result,
        "source_sha256": prior.get("source_sha256"),
        "dependencies": prior.get("dependencies"),
        "catboost_settings": prior["catboost_settings"],
        "catboost_cat_features": prior["catboost_cat_features"],
        "input_sha256_before": prior["input_sha256_before"],
        "recorded_split": prior["recorded_splits"]["42"],
    }


def _save_validation(
    path: Path,
    row_index: np.ndarray,
    ids: np.ndarray,
    candidate: np.ndarray,
    lightgbm: np.ndarray,
    blend: np.ndarray,
) -> None:
    for values in (candidate, lightgbm, blend):
        validate_probabilities(values, len(row_index))
    pd.DataFrame(
        {
            "row_index": row_index,
            ID_COLUMN: ids,
            "categorical_catboost_probability": candidate,
            "original_lightgbm_probability": lightgbm,
            "blend_probability": blend,
        }
    ).to_csv(path, index=False, float_format="%.17g")


def _fit_candidate(
    X_categorical: pd.DataFrame,
    y: np.ndarray,
    inner_fit: np.ndarray,
    inner_validation: np.ndarray,
    holdout: np.ndarray,
    cat_columns: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    model = build_categorical_ratings_catboost()
    started = time.perf_counter()
    before_resources = _resource_snapshot()
    model.fit(
        X_categorical.iloc[inner_fit],
        y[inner_fit],
        eval_set=(X_categorical.iloc[inner_validation], y[inner_validation]),
        cat_features=list(cat_columns),
        verbose=False,
    )
    fit_seconds = time.perf_counter() - started
    predictions = model.predict_proba(X_categorical.iloc[holdout])[:, 1]
    validate_probabilities(predictions, len(holdout))
    result = {
        "candidate": "categorical_ratings_catboost",
        "fit_seconds": round(fit_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "actual_tree_count": int(model.tree_count_),
        "best_iteration": int(model.get_best_iteration()),
        "best_iteration_count_one_based": int(model.get_best_iteration()) + 1,
        "settings": model.get_params(),
        "resources": _resource_delta(before_resources, _resource_snapshot()),
    }
    del model
    gc.collect()
    return predictions, result


def _load_recorded_arrays(
    artifacts_dir: Path,
    prior_metadata: dict[str, Any],
    train: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    split_path = artifacts_dir / SPLIT_FILES[seed]
    outer_fit, outer_holdout = _load_recorded_split(split_path, train, y, seed)
    split_name = "outer" if seed == 42 else "confirmation"
    prior = prior_metadata["internal_validation_splits"][split_name]
    inner_fit = np.asarray(prior["fit_indices"], dtype=np.int64)
    inner_validation = np.asarray(prior["validation_indices"], dtype=np.int64)
    expected_fit, expected_validation = make_catboost_internal_split(
        outer_fit, y, seed=42
    )
    if not np.array_equal(inner_fit, expected_fit) or not np.array_equal(
        inner_validation, expected_validation
    ):
        raise RuntimeError("recorded CatBoost internal arrays do not match seed 42")
    if np.intersect1d(inner_validation, outer_holdout).size:
        raise RuntimeError("recorded internal validation overlaps outer holdout")
    arrays = {
        "outer_fit": outer_fit,
        "outer_holdout": outer_holdout,
        "inner_fit": inner_fit,
        "inner_validation": inner_validation,
    }
    record = {
        "outer_split_path": str(split_path),
        "outer_split_sha256": sha256_file(split_path),
        "outer_fit_rows": len(outer_fit),
        "outer_holdout_rows": len(outer_holdout),
        "inner_fit_rows": len(inner_fit),
        "inner_validation_rows": len(inner_validation),
        "array_sha256": {
            name: _array_fingerprint(value) for name, value in arrays.items()
        },
        "recorded_arrays_fingerprint": _recorded_array_fingerprint(arrays),
        "inner_seed": 42,
        "inner_validation_fraction": 0.1,
    }
    return outer_fit, outer_holdout, inner_fit, inner_validation, record


def run(
    data_dir: Path, artifacts_dir: Path, *, complete_second_split: bool = False
) -> dict[str, Any]:
    """Run the default experiment or only its authorized second split."""
    configuration = _run_configuration(complete_second_split)
    seeds = configuration["seeds"]
    output_name = configuration["output_name"]
    internal_timeout_seconds = configuration["internal_timeout_seconds"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = artifacts_dir / output_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "internal_timeout_seconds": internal_timeout_seconds,
        "retrained_split_seeds": list(seeds),
        "skipped_retraining_split_seeds": [seed for seed in SEEDS if seed not in seeds],
        "selection": None,
        "validation_results": [],
        "current_fit": None,
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    protected_before: dict[str, str] | None = None
    protected_after: dict[str, str] | None = None
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    source_after: dict[str, str] | None = None
    previous_alarm = signal.getitimer(signal.ITIMER_REAL)
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.signal(signal.SIGTERM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, internal_timeout_seconds)
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _input_hashes(data_dir)
        metadata["protected_artifact_sha256_before"] = protected_before
        metadata["input_sha256_before"] = input_before
        source_before = _source_hashes()
        metadata["source_sha256"] = source_before
        metadata["dependencies"] = _dependency_versions()
        save_json(metadata_path, metadata)

        prior_metadata_path = artifacts_dir / "challengers" / "compare_metadata.json"
        prior_metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
        if prior_metadata.get("input_sha256") != input_before:
            raise RuntimeError("current inputs do not match prior CatBoost metadata")
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        cat_columns = categorical_columns(spec)
        y = target_array(train)
        X_categorical = prepare_rating_categorical_frame(train, spec)
        prepare_rating_categorical_frame(test, spec)
        if complete_second_split:
            reused_seed42 = _load_reused_seed42_summary(
                artifacts_dir, data_dir, train, y, input_before, cat_columns
            )
            metadata["lineage"] = {
                "reused_seed42": reused_seed42,
                "gate_baseline": "current slow CatBoost + original LightGBM",
                "information_only": True,
            }
        metadata.update(
            {
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": [
                    column
                    for column in spec.numeric_columns
                    if column not in RATING_COLUMNS
                ],
                "categorical_columns": list(cat_columns),
                "rating_columns": list(RATING_COLUMNS),
                "rating_representation": {
                    "observed_zero_is_valid": True,
                    "missing_nominal_sentinel": MISSING_CATEGORY,
                    "missing_ratings": "rejected",
                    "non_integral_or_out_of_domain_ratings": "rejected",
                },
                "catboost_settings": build_categorical_ratings_catboost().get_params(),
                "catboost_cat_features": list(cat_columns),
                "prior_metadata_path": str(prior_metadata_path),
                "prior_metadata_sha256": sha256_file(prior_metadata_path),
            }
        )
        save_json(metadata_path, metadata)

        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        if complete_second_split:
            prior_result = metadata["lineage"]["reused_seed42"]["validation_result"]
            baseline_auc[42] = prior_result["baseline_blend_roc_auc"]
            candidate_auc[42] = prior_result["blend_roc_auc"]
        for fit_number, seed in enumerate(seeds, start=1):
            (
                outer_fit,
                outer_holdout,
                inner_fit,
                inner_validation,
                split_record,
            ) = _load_recorded_arrays(artifacts_dir, prior_metadata, train, y, seed)
            metadata.setdefault("recorded_splits", {})[str(seed)] = split_record
            slow_path = (
                artifacts_dir
                / "slower_learning"
                / f"validation_slow_catboost_seed{seed}.csv"
            )
            light_path = (
                artifacts_dir / "lightgbm" / f"validation_lightgbm_seed{seed}.csv"
            )
            slow = _read_saved_predictions(slow_path, outer_holdout)
            light = _read_saved_predictions(light_path, outer_holdout)
            holdout_ids = train.iloc[outer_holdout][ID_COLUMN].to_numpy()
            if set(slow[ID_COLUMN]) != set(holdout_ids) or set(light[ID_COLUMN]) != set(
                holdout_ids
            ):
                raise ValueError(f"saved prediction IDs do not match seed {seed}")
            baseline_aligned = aligned_blend_predictions(
                holdout_ids,
                slow[ID_COLUMN],
                slow["probability"].to_numpy(),
                light[ID_COLUMN],
                light["probability"].to_numpy(),
            )
            baseline_values = baseline_aligned["blend_probability"].to_numpy()
            baseline_auc[seed] = float(roc_auc_score(y[outer_holdout], baseline_values))
            if not np.isclose(
                baseline_auc[seed], BASELINE_AUC[seed], rtol=0.0, atol=1e-9
            ):
                raise RuntimeError(
                    f"current slow-CatBoost baseline changed for seed {seed}"
                )

            metadata["current_fit"] = {
                "fit_number": fit_number,
                "split_seed": seed,
                "status": "running",
            }
            save_json(metadata_path, metadata)
            candidate_values, result = _fit_candidate(
                X_categorical,
                y,
                inner_fit,
                inner_validation,
                outer_holdout,
                cat_columns,
            )
            candidate_aligned = aligned_blend_predictions(
                holdout_ids,
                holdout_ids,
                candidate_values,
                light[ID_COLUMN],
                light["probability"].to_numpy(),
            )
            blend_values = candidate_aligned["blend_probability"].to_numpy()
            candidate_auc[seed] = float(roc_auc_score(y[outer_holdout], blend_values))
            validation_path = (
                output_dir / f"validation_categorical_ratings_seed{seed}.csv"
            )
            _save_validation(
                validation_path,
                outer_holdout,
                candidate_aligned[ID_COLUMN].to_numpy(),
                candidate_aligned["catboost_probability"].to_numpy(),
                candidate_aligned["lightgbm_probability"].to_numpy(),
                blend_values,
            )
            result.update(
                {
                    "split_seed": seed,
                    "rows": len(outer_holdout),
                    "roc_auc": float(roc_auc_score(y[outer_holdout], candidate_values)),
                    "log_loss": float(log_loss(y[outer_holdout], candidate_values)),
                    "blend_roc_auc": candidate_auc[seed],
                    "blend_log_loss": float(log_loss(y[outer_holdout], blend_values)),
                    "baseline_blend_roc_auc": baseline_auc[seed],
                    "baseline_blend_log_loss": float(
                        log_loss(y[outer_holdout], baseline_values)
                    ),
                    "auc_gain_vs_current_blend": candidate_auc[seed]
                    - baseline_auc[seed],
                    "validation_predictions_path": str(validation_path),
                    "original_lightgbm_predictions_path": str(light_path),
                    "original_lightgbm_predictions_sha256": sha256_file(light_path),
                    "current_slow_catboost_predictions_path": str(slow_path),
                    "current_slow_catboost_predictions_sha256": sha256_file(slow_path),
                }
            )
            metadata["validation_results"].append(result)
            metadata["current_fit"] = None
            save_json(metadata_path, metadata)
            del outer_fit, inner_fit, inner_validation, slow, light, candidate_values
            gc.collect()

        metadata["selection"] = categorical_gate(baseline_auc, candidate_auc)
        metadata["next_approval_needed"] = (
            "fresh seed 31415 confirmation and any refit/export "
            "require separate approval"
            if metadata["selection"]["qualifies"]
            else "none; current blend remains the selected candidate"
        )
    except BaseException as error:
        caught = error
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        signal.signal(signal.SIGTERM, previous_term_handler)
        if previous_alarm[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_alarm[0], previous_alarm[1])
        metadata["status"] = "failed" if caught else "completed"
        if caught is not None:
            metadata["error"] = {"type": type(caught).__name__, "message": str(caught)}
        try:
            metadata["input_sha256_after"] = _input_hashes(data_dir)
        except BaseException as error:
            metadata["input_sha256_after_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        if source_before is not None:
            try:
                source_after = _source_hashes()
                metadata["source_sha256_after"] = source_after
            except BaseException as error:
                metadata["source_sha256_after_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        try:
            protected_after = _artifact_snapshot(artifacts_dir, output_dir)
            metadata["protected_artifact_sha256_after"] = protected_after
        except BaseException as error:
            metadata["protected_artifact_sha256_after_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        metadata["input_artifacts_unchanged"] = (
            input_before is not None
            and metadata.get("input_sha256_after") == input_before
        )
        metadata["protected_artifacts_unchanged"] = (
            protected_before is not None and protected_after == protected_before
        )
        metadata["source_files_unchanged"] = (
            source_before is not None and source_after == source_before
        )
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        save_json(metadata_path, metadata)

    integrity_error: RuntimeError | None = None
    if not metadata["input_artifacts_unchanged"]:
        integrity_error = RuntimeError("input files changed during the experiment")
    elif not metadata["protected_artifacts_unchanged"]:
        integrity_error = RuntimeError(
            "protected artifacts changed during the experiment"
        )
    elif not metadata["source_files_unchanged"]:
        integrity_error = RuntimeError("source files changed during the experiment")
    elif metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
        integrity_error = RuntimeError(
            "output directory exceeded the predeclared budget"
        )
    if caught is not None:
        raise caught
    if integrity_error is not None:
        raise integrity_error
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--complete-second-split",
        action="store_true",
        help="fit only split seed 2024 with the authorized 85-minute bound",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.signal(signal.SIGTERM, _on_timeout)
    result = run(
        args.data_dir,
        args.artifacts_dir,
        complete_second_split=args.complete_second_split,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selection": result["selection"],
                "validation_results": result["validation_results"],
                "output_dir": result["output_dir"],
                "output_bytes": result["output_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
