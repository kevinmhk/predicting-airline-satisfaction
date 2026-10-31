"""Run the fixed cached 30/35/35 round-two blend."""

from __future__ import annotations

import argparse
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

# ruff: noqa: E402
import numpy as np

import final_sprint_experiments as f
import round2_cached_experiments as r
import round2_feature_experiments as q
from train import save_json, sha256_file

PAIR_LIGHT70 = "pair_light70"
SPLIT_SEEDS = r.SPLIT_SEEDS
INTERNAL_TIMEOUT_SECONDS = q.INTERNAL_TIMEOUT_SECONDS
EXTERNAL_TIMEOUT_SECONDS = q.EXTERNAL_TIMEOUT_SECONDS
OUTPUT_BUDGET_BYTES = q.OUTPUT_BUDGET_BYTES
VALIDATION_COLUMNS = r.VALIDATION_COLUMNS


def light70_blend(
    catboost: np.ndarray, context: np.ndarray, hierarchical: np.ndarray
) -> np.ndarray:
    """Apply the one approved cached 30/35/35 probability blend."""
    values = [
        np.asarray(value, dtype=np.float64)
        for value in (catboost, context, hierarchical)
    ]
    if any(value.ndim != 1 or len(value) != len(values[0]) for value in values):
        raise ValueError("blend components must be one-dimensional and equally sized")
    candidate = 0.30 * values[0] + 0.35 * values[1] + 0.35 * values[2]
    if (
        not np.isfinite(candidate).all()
        or ((candidate < 0.0) | (candidate > 1.0)).any()
    ):
        raise ValueError("blend probabilities must be finite and bounded")
    return candidate


def _light70_frame(assembled: Any) -> Any:
    result = assembled.copy()
    result["candidate_probability"] = light70_blend(
        result["catboost_probability"].to_numpy(dtype=np.float64),
        result["context_probability"].to_numpy(dtype=np.float64),
        result["hierarchical_probability"].to_numpy(dtype=np.float64),
    )
    return result


def _own_source_hashes() -> dict[str, str]:
    hashes = q._own_source_hashes()
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _validation_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"validation_{PAIR_LIGHT70}_seed{seed}.csv"


def _initial_metadata(output_dir: Path) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "experiment": PAIR_LIGHT70,
        "mode_definition": {
            "cached_only": True,
            "components": {
                "catboost": "original accepted slow CatBoost",
                "context": "original accepted context127 student",
                "hierarchical": "original accepted hierarchical31 student",
            },
            "weights": {"catboost": 0.30, "context": 0.35, "hierarchical": 0.35},
            "transform": "identity",
            "conditional_features": [],
        },
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "completed_fits": [],
        "fits": 0,
        "training_seconds": 0.0,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "output_dir": str(output_dir),
    }


def run_experiment(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run pair_light70 from verified caches without fitting or exporting."""
    started = time.perf_counter()
    output_dir = artifacts_dir / f"round2_{PAIR_LIGHT70}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata = _initial_metadata(output_dir)
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    integrity_errors: list[BaseException] = []
    cleanup_errors: list[BaseException] = []
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
        metadata["protected_artifact_sha256_before"] = protected_before
        (
            train,
            y,
            _spec,
            prior,
            _profile_metadata,
            incumbent_refs,
            checked_inputs,
            _checked_sources,
        ) = q._common_preflight(data_dir, artifacts_dir, metadata, metadata_path)
        if checked_inputs != input_before:
            raise RuntimeError("preflight input fingerprint changed")
        metadata["source_sha256_before"] = source_before
        metadata["cached_only"] = True
        save_json(metadata_path, metadata)

        source_hashes = f._source_hashes()
        baseline_auc: dict[int, float] = {}
        candidate_auc: dict[int, float] = {}
        for seed in SPLIT_SEEDS:
            scoring_started = time.perf_counter()
            assembled, holdout, baseline_metrics, lineage = r._load_seed_components(
                r.PAIR_LIGHT60,
                artifacts_dir,
                train,
                y,
                prior,
                input_before,
                source_hashes,
                incumbent_refs,
                seed,
            )
            validation = _light70_frame(assembled)
            validation.insert(0, "row_index", holdout)
            if list(validation.columns) != list(VALIDATION_COLUMNS):
                raise RuntimeError("round-two output schema changed")
            q._assert_prediction_schema(validation, train, holdout)
            candidate_metrics = f._metrics(
                y[holdout],
                validation["candidate_probability"].to_numpy(dtype=np.float64),
            )
            validation_path = _validation_path(output_dir, seed)
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
        metadata["selection"] = f.final_sprint_gate(
            PAIR_LIGHT70, baseline_auc, candidate_auc
        )
    except BaseException as error:
        caught = error
        metadata["error"] = _error_record(error)
    finally:
        metadata["current_fit"] = None
        if timer_started:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException as error:
                cleanup_errors.append(error)
        if alarm_installed:
            try:
                signal.signal(signal.SIGALRM, previous_alarm)
            except BaseException as error:
                cleanup_errors.append(error)
        if term_installed:
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
                        RuntimeError("pre-existing artifact fingerprint changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if integrity_errors:
            metadata["integrity_error"] = _error_record(
                RuntimeError("; ".join(str(error) for error in integrity_errors))
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = RuntimeError("integrity checks failed")
                metadata["error"] = _error_record(caught)
        if cleanup_errors:
            metadata["cleanup_error"] = _error_record(
                RuntimeError("; ".join(str(error) for error in cleanup_errors))
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = RuntimeError("cleanup checks failed")
                metadata["error"] = _error_record(caught)
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
            q._persist_metadata(metadata_path, metadata, output_dir)
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["metadata_persistence_error"] = _error_record(error)
            if caught is None:
                caught = error
                metadata["error"] = _error_record(error)
            try:
                q._persist_metadata(metadata_path, metadata, output_dir)
            except BaseException:
                pass
    if caught is not None:
        raise caught
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
