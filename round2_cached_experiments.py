"""Score three fixed blends from the latest cached route-pair incumbent."""

from __future__ import annotations

import argparse
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
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import cached_route_blends as cached
import final_sprint_experiments as f
from train import ID_COLUMN, save_json, sha256_file, target_array, validate_inputs
from weighted_blend_experiment import weighted_blend_predictions

PAIR_LIGHT60 = "pair_light60"
PAIR_RANK_AVERAGE = "pair_rank_average"
PAIR_LOGIT_AVERAGE = "pair_logit_average"
MODES = (PAIR_LIGHT60, PAIR_RANK_AVERAGE, PAIR_LOGIT_AVERAGE)
SPLIT_SEEDS = cached.SPLIT_SEEDS
LATEST_BASELINE_AUC = {
    42: 0.9595596911655389,
    2024: 0.959864554491251,
}
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
VALIDATION_COLUMNS = (
    "row_index",
    ID_COLUMN,
    "catboost_probability",
    "context_probability",
    "hierarchical_probability",
    "baseline_probability",
    "candidate_probability",
)
MODE_DEFINITIONS: dict[str, dict[str, Any]] = {
    PAIR_LIGHT60: {
        "weights": {"catboost": 0.4, "context": 0.3, "hierarchical": 0.3},
        "transform": "identity",
    },
    PAIR_RANK_AVERAGE: {
        "weights": {"catboost": 0.5, "context": 0.25, "hierarchical": 0.25},
        "transform": "rankdata(method='average') / (n + 1)",
        "rank_semantics": (
            "transductive unlabelled-batch rank transform; not probability calibration"
        ),
    },
    PAIR_LOGIT_AVERAGE: {
        "weights": {"catboost": 0.5, "context": 0.25, "hierarchical": 0.25},
        "transform": "sigmoid(weighted clipped component logits)",
        "clip": [1e-6, 0.999999],
    },
}


def _align_component(
    reference_ids: Any,
    component_ids: Any,
    values: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_ids)
    ids = np.asarray(component_ids)
    values_array = np.asarray(values, dtype=np.float64)
    if reference.ndim != 1 or len(reference) == 0:
        raise ValueError("reference IDs must be a non-empty one-dimensional array")
    if ids.ndim != 1 or values_array.ndim != 1 or len(ids) != len(values_array):
        raise ValueError("component IDs and probabilities have the wrong shape")
    reference_values: list[Any] = reference.tolist()
    component_values: list[Any] = ids.tolist()
    aligned = weighted_blend_predictions(
        reference_values,
        component_values,
        values_array,
        component_values,
        values_array,
        slow_catboost_weight=1.0,
        lightgbm_weight=0.0,
    )
    return aligned["blend_probability"].to_numpy(dtype=np.float64)


def assemble_probabilities(
    mode: str,
    reference_ids: Any,
    catboost_ids: Any,
    catboost_values: np.ndarray,
    context_ids: Any,
    context_values: np.ndarray,
    hierarchical_ids: Any,
    hierarchical_values: np.ndarray,
) -> pd.DataFrame:
    """Align three cached components and apply one declared fixed formula."""
    if mode not in MODES:
        raise ValueError(f"unknown round2 mode: {mode}")
    catboost = _align_component(reference_ids, catboost_ids, catboost_values)
    context = _align_component(reference_ids, context_ids, context_values)
    hierarchical = _align_component(
        reference_ids, hierarchical_ids, hierarchical_values
    )
    baseline = 0.5 * catboost + 0.25 * context + 0.25 * hierarchical
    if mode == PAIR_LIGHT60:
        candidate = 0.4 * catboost + 0.3 * context + 0.3 * hierarchical
    elif mode == PAIR_RANK_AVERAGE:
        denominator = len(catboost) + 1.0
        candidate = (
            0.5 * rankdata(catboost, method="average")
            + 0.25 * rankdata(context, method="average")
            + 0.25 * rankdata(hierarchical, method="average")
        ) / denominator
    else:
        clipped = np.clip(
            np.column_stack((catboost, context, hierarchical)), 1e-6, 1.0 - 1e-6
        )
        logits = np.log(clipped / (1.0 - clipped))
        weighted_logit = 0.5 * logits[:, 0] + 0.25 * logits[:, 1] + 0.25 * logits[:, 2]
        candidate = 1.0 / (1.0 + np.exp(-weighted_logit))
    if not np.isfinite(candidate).all() or ((candidate < 0) | (candidate > 1)).any():
        raise RuntimeError("assembled probabilities are not finite and bounded")
    return pd.DataFrame(
        {
            ID_COLUMN: np.asarray(reference_ids),
            "catboost_probability": catboost,
            "context_probability": context,
            "hierarchical_probability": hierarchical,
            "baseline_probability": baseline,
            "candidate_probability": candidate,
        }
    )


def _own_source_hashes() -> dict[str, str]:
    hashes = cached._own_source_hashes()
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _incumbent_prediction_path(
    directory: Path, record: dict[str, Any], seed: int
) -> Path:
    expected = directory / f"validation_route_pair_average_seed{seed}.csv"
    recorded = record.get("validation_predictions_path")
    if not isinstance(recorded, str) or Path(recorded).resolve() != expected.resolve():
        raise RuntimeError("latest incumbent CSV path changed")
    return expected


def _validate_incumbent(
    artifacts_dir: Path, input_hashes: dict[str, str]
) -> dict[int, dict[str, Any]]:
    incumbent_dir = artifacts_dir / "cached_route_blends_route_pair_average"
    metadata_path = incumbent_dir / "compare_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("latest route-pair incumbent is not completed")
    if metadata.get("choice") != cached.ROUTE_PAIR_AVERAGE:
        raise RuntimeError("latest route-pair incumbent has the wrong choice")
    if metadata.get("selection", {}).get("qualifies") is not True:
        raise RuntimeError("latest route-pair incumbent is not selected")
    if metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError("latest route-pair incumbent inputs changed")
    if metadata.get("input_sha256_after") != input_hashes:
        raise RuntimeError(
            "latest route-pair incumbent input after-fingerprint changed"
        )
    source_hashes = cached._own_source_hashes()
    if metadata.get("source_sha256_before") != source_hashes:
        raise RuntimeError("latest route-pair incumbent source fingerprints changed")
    if metadata.get("source_sha256_after") != source_hashes:
        raise RuntimeError(
            "latest route-pair incumbent source after-fingerprint changed"
        )
    references: dict[int, dict[str, Any]] = {}
    for seed in SPLIT_SEEDS:
        record = cached._record_for_seed(metadata, seed)
        expected_auc = LATEST_BASELINE_AUC[seed]
        if not np.isclose(
            float(record.get("candidate", {}).get("roc_auc")),
            expected_auc,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("latest route-pair incumbent AUC changed")
        path = _incumbent_prediction_path(incumbent_dir, record, seed)
        expected_hash = record.get(
            "validation_predictions_sha256", record.get("vpred_sha256")
        )
        if expected_hash != sha256_file(path):
            raise RuntimeError("latest route-pair incumbent CSV fingerprint changed")
        references[seed] = {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "validation_predictions_path": str(path),
            "validation_predictions_sha256": sha256_file(path),
        }
    return references


def _load_seed_components(
    mode: str,
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    prior: dict[str, Any],
    input_hashes: dict[str, str],
    source_hashes: dict[str, str],
    incumbent_refs: dict[int, dict[str, Any]],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], dict[str, Any]]:
    source_frame, checked_result, _ = cached._source_components(
        cached.ROUTE_PAIR_AVERAGE,
        artifacts_dir,
        train,
        y,
        prior,
        input_hashes,
        source_hashes,
        seed,
    )
    holdout = source_frame["row_index"].to_numpy(dtype=np.int64)
    accepted = f._read_saved_distance_validation(
        Path(incumbent_refs[seed]["validation_predictions_path"]), train, holdout
    )
    if not np.allclose(
        accepted["candidate_probability"],
        source_frame["candidate_probability"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "accepted incumbent does not match verified source components"
        )

    components: list[pd.DataFrame] = []
    component_refs: list[dict[str, Any]] = []
    for source_name in cached.SOURCE_DIRS[cached.ROUTE_PAIR_AVERAGE]:
        source_dir = artifacts_dir / source_name
        path = source_dir / (
            f"validation_{source_name.removeprefix('final_sprint_')}" f"_seed{seed}.csv"
        )
        frame = f._read_saved_distance_validation(path, train, holdout)
        components.append(frame)
        component_refs.append(
            {
                "metadata_path": str(source_dir / "compare_metadata.json"),
                "metadata_sha256": sha256_file(source_dir / "compare_metadata.json"),
                "validation_predictions_path": str(path),
                "validation_predictions_sha256": sha256_file(path),
            }
        )
    context, hierarchical = components
    assembled = assemble_probabilities(
        mode,
        source_frame[ID_COLUMN].to_numpy(dtype=np.int64),
        source_frame[ID_COLUMN].to_numpy(dtype=np.int64),
        source_frame["catboost_probability"].to_numpy(dtype=np.float64),
        context[ID_COLUMN].to_numpy(dtype=np.int64),
        context["student_probability"].to_numpy(dtype=np.float64),
        hierarchical[ID_COLUMN].to_numpy(dtype=np.int64),
        hierarchical["student_probability"].to_numpy(dtype=np.float64),
    )
    if not np.allclose(
        assembled["baseline_probability"],
        source_frame["candidate_probability"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("latest incumbent arithmetic is not .5/.25/.25")
    baseline_metrics = f._metrics(
        y[holdout], assembled["baseline_probability"].to_numpy(dtype=np.float64)
    )
    if not np.isclose(
        baseline_metrics["roc_auc"],
        LATEST_BASELINE_AUC[seed],
        rtol=0.0,
        atol=1e-12,
    ) or not cached._metric_match(baseline_metrics, checked_result["candidate"]):
        raise RuntimeError("recomputed latest route-pair baseline changed")
    lineage = {
        "accepted_incumbent": incumbent_refs[seed],
        "source_components": component_refs,
        "source_code_sha256": source_hashes,
    }
    return assembled, holdout, baseline_metrics, lineage


def _persist_metadata(path: Path, metadata: dict[str, Any], output_dir: Path) -> None:
    metadata["output_bytes"] = f._output_size(output_dir)
    if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
        raise RuntimeError("output exceeded 100 MiB")
    cached._save_sized_metadata(path, metadata)
    if metadata["output_bytes"] != f._output_size(output_dir):
        raise RuntimeError("final metadata byte count did not stabilize")


def run_experiment(mode: str, data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run one fixed cached mode without fitting, refitting, or exporting."""
    if mode not in MODES:
        raise ValueError(f"unknown round2 mode: {mode}")
    started = time.perf_counter()
    output_dir = artifacts_dir / f"round2_{mode}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "experiment": mode,
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "training_seconds": 0.0,
        "fits": 0,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "output_dir": str(output_dir),
        "baseline_reference": {
            "artifact": "cached_route_blends_route_pair_average",
            "column": "candidate_probability",
            "auc_by_seed": LATEST_BASELINE_AUC,
        },
        "mode_definition": MODE_DEFINITIONS[mode],
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    alarm_installed = False
    term_installed = False
    timer_started = False
    try:
        signal.signal(signal.SIGALRM, f._on_timeout)
        alarm_installed = True
        signal.signal(signal.SIGTERM, f._on_timeout)
        term_installed = True
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        timer_started = True
        protected_before = f._artifact_snapshot(artifacts_dir, output_dir)
        input_before = f._input_hashes(data_dir)
        source_before = _own_source_hashes()
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        prior = f._load_prior_lightgbm(artifacts_dir, input_before)
        source_hashes = f._source_hashes()
        incumbent_refs = _validate_incumbent(artifacts_dir, input_before)
        metadata.update(
            {
                "python": sys.version,
                "threads": 2,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "feature_spec": {
                    "feature_columns": list(spec.feature_columns),
                    "numeric_columns": list(spec.numeric_columns),
                    "categorical_columns": list(spec.categorical_columns),
                },
                "split_seeds": list(SPLIT_SEEDS),
            }
        )
        save_json(metadata_path, metadata)
        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            scoring_started = time.perf_counter()
            assembled, holdout, baseline_metrics, lineage = _load_seed_components(
                mode,
                artifacts_dir,
                train,
                y,
                prior,
                input_before,
                source_hashes,
                incumbent_refs,
                seed,
            )
            candidate_metrics = f._metrics(
                y[holdout],
                assembled["candidate_probability"].to_numpy(dtype=np.float64),
            )
            validation = assembled.copy()
            validation.insert(0, "row_index", holdout)
            if list(validation.columns) != list(VALIDATION_COLUMNS):
                raise RuntimeError("round2 output schema changed")
            validation_path = output_dir / f"validation_{mode}_seed{seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            baseline_auc[seed] = baseline_metrics["roc_auc"]
            candidate_auc[seed] = candidate_metrics["roc_auc"]
            metadata["validation_results"].append(
                {
                    "split_seed": seed,
                    "rows": len(validation),
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "auc_gain_vs_baseline": candidate_metrics["roc_auc"]
                    - baseline_metrics["roc_auc"],
                    "elapsed_scoring_seconds": round(
                        time.perf_counter() - scoring_started, 3
                    ),
                    "training_seconds": 0.0,
                    "fits": 0,
                    "validation_predictions_path": str(validation_path),
                    "validation_predictions_sha256": sha256_file(validation_path),
                    "lineage": lineage,
                }
            )
            save_json(metadata_path, metadata)
        metadata["selection"] = f.final_sprint_gate(mode, baseline_auc, candidate_auc)
    except BaseException as error:
        caught = error
        metadata["error"] = _error_record(error)
    finally:
        metadata["current_fit"] = None
        if timer_started:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException as error:
                finalization_errors.append(error)
        if alarm_installed:
            try:
                signal.signal(signal.SIGALRM, previous_alarm)
            except BaseException as error:
                finalization_errors.append(error)
        if term_installed:
            try:
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                finalization_errors.append(error)
        if input_before is not None:
            try:
                metadata["input_sha256_after"] = f._input_hashes(data_dir)
                if metadata["input_sha256_after"] != input_before:
                    finalization_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
            except BaseException as error:
                finalization_errors.append(error)
        if source_before is not None:
            try:
                metadata["source_sha256_after"] = _own_source_hashes()
                if metadata["source_sha256_after"] != source_before:
                    finalization_errors.append(
                        RuntimeError("source fingerprint changed")
                    )
            except BaseException as error:
                finalization_errors.append(error)
        try:
            protected_after = f._artifact_snapshot(artifacts_dir, output_dir)
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
        if finalization_errors:
            integrity_error = RuntimeError(
                "; ".join(str(error) for error in finalization_errors)
            )
            metadata["integrity_error"] = _error_record(integrity_error)
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = integrity_error
                metadata["error"] = _error_record(integrity_error)
        elif caught is not None:
            metadata["status"] = (
                "timeout" if isinstance(caught, f.FinalSprintTimeout) else "failed"
            )
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
