"""Score two fixed blends from completed final-sprint validation caches."""

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

import final_sprint_experiments as f
from train import ID_COLUMN, save_json, sha256_file, target_array, validate_inputs
from weighted_blend_experiment import weighted_blend_predictions

ROUTE_PAIR_AVERAGE = "route_pair_average"
HIERARCHICAL_LIGHT60 = "hierarchical_light60"
CHOICES = (ROUTE_PAIR_AVERAGE, HIERARCHICAL_LIGHT60)
SPLIT_SEEDS = (42, 2024)
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
VALIDATION_COLUMNS = (
    "row_index",
    ID_COLUMN,
    "student_probability",
    "catboost_probability",
    "baseline_probability",
    "candidate_probability",
)
SOURCE_DIRS = {
    ROUTE_PAIR_AVERAGE: (
        "final_sprint_route_context_target_encoding127",
        "final_sprint_route_hierarchical_encoding",
    ),
    HIERARCHICAL_LIGHT60: ("final_sprint_route_hierarchical_encoding",),
}


def assemble_cached_probabilities(
    choice: str,
    reference_ids: Any,
    catboost_ids: Any,
    catboost_values: np.ndarray,
    primary_student_ids: Any,
    primary_student_values: np.ndarray,
    secondary_student_ids: Any = None,
    secondary_student_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Align cached components and apply one of the two fixed blend choices."""
    if choice not in CHOICES:
        raise ValueError(f"unknown cached choice: {choice}")
    if choice == ROUTE_PAIR_AVERAGE:
        if secondary_student_ids is None or secondary_student_values is None:
            raise ValueError("route_pair_average requires two student components")
        student = weighted_blend_predictions(
            reference_ids,
            primary_student_ids,
            primary_student_values,
            secondary_student_ids,
            secondary_student_values,
            slow_catboost_weight=0.5,
            lightgbm_weight=0.5,
        )
        catboost_weight, student_weight = 0.5, 0.5
    else:
        student = weighted_blend_predictions(
            reference_ids,
            primary_student_ids,
            primary_student_values,
            primary_student_ids,
            primary_student_values,
            slow_catboost_weight=0.0,
            lightgbm_weight=1.0,
        )
        catboost_weight, student_weight = 0.4, 0.6
    candidate = weighted_blend_predictions(
        reference_ids,
        catboost_ids,
        catboost_values,
        student[ID_COLUMN],
        student["blend_probability"].to_numpy(dtype=np.float64),
        slow_catboost_weight=catboost_weight,
        lightgbm_weight=student_weight,
    )
    return pd.DataFrame(
        {
            ID_COLUMN: candidate[ID_COLUMN],
            "student_probability": student["blend_probability"].to_numpy(
                dtype=np.float64
            ),
            "catboost_probability": candidate["slow_catboost_probability"].to_numpy(
                dtype=np.float64
            ),
            "candidate_probability": candidate["blend_probability"].to_numpy(
                dtype=np.float64
            ),
        }
    )


def _own_source_hashes() -> dict[str, str]:
    hashes = f._source_hashes()
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _metric_match(actual: dict[str, float], recorded: dict[str, Any]) -> bool:
    return all(
        np.isclose(float(actual[name]), float(recorded[name]), rtol=0.0, atol=1e-12)
        for name in ("roc_auc", "log_loss")
    )


def _record_for_seed(metadata: dict[str, Any], seed: int) -> dict[str, Any]:
    records = [
        record
        for record in metadata.get("validation_results", [])
        if record.get("split_seed") == seed
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"cached metadata must contain one validation result for {seed}"
        )
    return records[0]


def _resolve_cached_path(source_dir: Path, record: dict[str, Any]) -> Path:
    recorded = record.get("validation_predictions_path")
    if not isinstance(recorded, str):
        raise RuntimeError("cached validation path is missing")
    path = Path(recorded)
    actual = path if path.is_absolute() else Path.cwd() / path
    name = (
        f"validation_{source_dir.name.removeprefix('final_sprint_')}"
        f"_seed{record['split_seed']}.csv"
    )
    expected = source_dir / name
    if actual.resolve() != expected.resolve():
        raise RuntimeError("cached validation path is outside its source artifact")
    return expected


def _validate_cached_metadata(
    source_dir: Path,
    input_hashes: dict[str, str],
    source_hashes: dict[str, str],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    metadata_path = source_dir / "compare_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError(f"cached metadata is not completed: {metadata_path}")
    if metadata.get("input_sha256_before") != input_hashes:
        raise RuntimeError(f"cached inputs changed: {metadata_path}")
    if metadata.get("input_sha256_after") != input_hashes:
        raise RuntimeError(f"cached input after-fingerprint changed: {metadata_path}")
    recorded_sources = metadata.get("source_sha256_before")
    if recorded_sources != metadata.get("source_sha256_after"):
        raise RuntimeError(f"cached source fingerprints changed: {metadata_path}")
    if recorded_sources != source_hashes:
        raise RuntimeError(f"cached source fingerprints changed: {metadata_path}")
    record = _record_for_seed(metadata, seed)
    path = _resolve_cached_path(source_dir, record)
    expected_hash = record.get(
        "vpred_sha256", record.get("validation_predictions_sha256")
    )
    if expected_hash != sha256_file(path):
        raise RuntimeError(f"cached validation CSV fingerprint changed: {path}")
    return metadata, record, path


def _validate_source_contract(
    record: dict[str, Any],
    source_name: str,
    split: dict[str, Any],
    inner_fit: np.ndarray,
) -> None:
    mode = (
        f.ROUTE_CONTEXT_TARGET_127
        if source_name == "final_sprint_route_context_target_encoding127"
        else f.ROUTE_HIERARCHICAL
    )
    if record.get("settings") != f.build_trial_classifier(mode).get_params():
        raise RuntimeError("cached model settings changed")
    if record.get("split") != split or record.get("profile_fit_membership") != {
        "rows": len(inner_fit),
        "sha256": f._array_fingerprint(inner_fit),
    }:
        raise RuntimeError("cached split or inner-fit membership changed")
    suffix = (
        [
            f.FLIGHT_DISTANCE_TARGET,
            f.FLIGHT_DISTANCE_CLASS_TARGET,
            f.FLIGHT_DISTANCE_TRAVEL_TARGET,
            f.FLIGHT_DISTANCE_CUSTOMER_TARGET,
        ]
        if mode == f.ROUTE_CONTEXT_TARGET_127
        else [
            f.FLIGHT_DISTANCE_HIERARCHICAL,
            f.FLIGHT_DISTANCE_CLASS_HIERARCHICAL,
            f.FLIGHT_DISTANCE_TRAVEL_HIERARCHICAL,
            f.FLIGHT_DISTANCE_CUSTOMER_HIERARCHICAL,
        ]
    )
    spec = record.get("feature_spec", {})
    if (
        len(spec.get("feature_columns", [])) != 43
        or len(spec.get("numeric_columns", [])) != 39
        or spec.get("categorical_columns")
        != ["Gender", "Customer Type", "Type of Travel", "Class"]
        or spec["feature_columns"][-4:] != suffix
        or spec["numeric_columns"][-4:] != suffix
    ):
        raise RuntimeError("cached student feature schema changed")


def _source_components(
    choice: str,
    artifacts_dir: Path,
    train: pd.DataFrame,
    y: np.ndarray,
    prior: dict[str, Any],
    input_hashes: dict[str, str],
    source_hashes: dict[str, str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    scoring_started = time.perf_counter()
    _, holdout, inner_fit, _, split = f._load_split_arrays(
        artifacts_dir, train, y, prior, seed
    )
    source_records: list[tuple[dict[str, Any], dict[str, Any], pd.DataFrame]] = []
    for source_name in SOURCE_DIRS[choice]:
        source_dir = artifacts_dir / source_name
        source_metadata, source_record, path = _validate_cached_metadata(
            source_dir, input_hashes, source_hashes, seed
        )
        _validate_source_contract(source_record, source_name, split, inner_fit)
        frame = f._read_saved_distance_validation(path, train, holdout)
        source_records.append((source_metadata, source_record, frame))

    slow_path = (
        artifacts_dir / "slower_learning" / f"validation_slow_catboost_seed{seed}.csv"
    )
    slow = f._load_components(slow_path, train, holdout)
    profile_path = (
        artifacts_dir
        / "distance_profile_feature"
        / f"validation_distance_profile_seed{seed}.csv"
    )
    profile = f._read_saved_distance_validation(profile_path, train, holdout)
    baseline, baseline_metrics = f._metrics_from_saved_baseline(
        y, holdout, slow, profile, seed
    )
    expected_ids = profile[ID_COLUMN].to_numpy(dtype=np.int64)
    expected_catboost = slow["probability"].to_numpy(dtype=np.float64)
    for _, record, frame in source_records:
        if not np.allclose(
            frame[ID_COLUMN].to_numpy(dtype=np.int64), expected_ids, rtol=0.0, atol=0.0
        ):
            raise RuntimeError("cached source IDs do not match the accepted baseline")
        if not np.allclose(
            frame["catboost_probability"].to_numpy(dtype=np.float64),
            expected_catboost,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("cached CatBoost probability changed")
        if not np.allclose(
            frame["baseline_probability"].to_numpy(dtype=np.float64),
            baseline,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("cached baseline probability changed")
        source_candidate = 0.5 * expected_catboost + 0.5 * frame[
            "student_probability"
        ].to_numpy(dtype=np.float64)
        if not np.allclose(
            source_candidate,
            frame["candidate_probability"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("cached candidate arithmetic changed")
        source_metrics = f._metrics(y[holdout], source_candidate)
        if not _metric_match(source_metrics, record.get("candidate", {})):
            raise RuntimeError("cached candidate metrics changed")
        if not _metric_match(baseline_metrics, record.get("baseline", {})):
            raise RuntimeError("cached baseline metrics changed")

    primary = source_records[0][2]
    secondary = source_records[1][2] if len(source_records) == 2 else None
    assembled = assemble_cached_probabilities(
        choice,
        expected_ids,
        primary[ID_COLUMN],
        primary["catboost_probability"].to_numpy(dtype=np.float64),
        primary[ID_COLUMN],
        primary["student_probability"].to_numpy(dtype=np.float64),
        None if secondary is None else secondary[ID_COLUMN],
        None
        if secondary is None
        else secondary["student_probability"].to_numpy(dtype=np.float64),
    )
    result = pd.DataFrame(
        {
            "row_index": holdout,
            ID_COLUMN: assembled[ID_COLUMN],
            "student_probability": assembled["student_probability"],
            "catboost_probability": assembled["catboost_probability"],
            "baseline_probability": baseline,
            "candidate_probability": assembled["candidate_probability"],
        }
    )
    if list(result.columns) != list(VALIDATION_COLUMNS):
        raise RuntimeError("cached output schema changed")
    candidate_metrics = f._metrics(
        y[holdout], result["candidate_probability"].to_numpy(dtype=np.float64)
    )
    student_metrics = f._metrics(
        y[holdout], result["student_probability"].to_numpy(dtype=np.float64)
    )
    lineage = {
        "split": split,
        "profile_fit_membership": {
            "rows": len(inner_fit),
            "sha256": f._array_fingerprint(inner_fit),
        },
        "components": [
            {
                "metadata_path": str(
                    artifacts_dir / source_name / "compare_metadata.json"
                ),
                "metadata_sha256": sha256_file(
                    artifacts_dir / source_name / "compare_metadata.json"
                ),
                "validation_predictions_path": str(
                    artifacts_dir
                    / source_name
                    / (
                        f"validation_{source_name.removeprefix('final_sprint_')}"
                        f"_seed{seed}.csv"
                    )
                ),
                "validation_predictions_sha256": sha256_file(
                    artifacts_dir
                    / source_name
                    / (
                        f"validation_{source_name.removeprefix('final_sprint_')}"
                        f"_seed{seed}.csv"
                    )
                ),
                "settings": record.get("settings"),
                "feature_spec": record.get("feature_spec"),
                "counts": {
                    "rows": len(source_records[index][2]),
                    "holdout_rows": len(holdout),
                    "inner_fit_rows": len(inner_fit),
                },
            }
            for index, (source_name, (_, record, _)) in enumerate(
                zip(SOURCE_DIRS[choice], source_records, strict=True)
            )
        ],
        "accepted_baseline": {
            "profile_path": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "slow_catboost_path": str(slow_path),
            "slow_catboost_sha256": sha256_file(slow_path),
        },
    }
    return (
        result,
        {
            "split_seed": seed,
            "baseline": baseline_metrics,
            "student": student_metrics,
            "candidate": candidate_metrics,
            "elapsed_scoring_seconds": round(time.perf_counter() - scoring_started, 3),
            "training_seconds": 0.0,
            "fits": 0,
            "weights": (
                {"catboost": 0.5, "student": 0.5, "student_components": [0.5, 0.5]}
                if choice == ROUTE_PAIR_AVERAGE
                else {"catboost": 0.4, "student": 0.6}
            ),
            "lineage": lineage,
        },
        {
            "baseline": baseline_metrics,
            "student": student_metrics,
            "candidate": candidate_metrics,
        },
    )


def _save_sized_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    other_bytes = f._output_size(path.parent) - (
        path.stat().st_size if path.exists() else 0
    )
    if temporary.exists():
        other_bytes -= temporary.stat().st_size
    for _ in range(4):
        payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        size = other_bytes + len(payload.encode("utf-8"))
        if size == metadata.get("output_bytes"):
            break
        metadata["output_bytes"] = size
    else:
        raise RuntimeError("metadata byte count did not stabilize")
    if size > OUTPUT_BUDGET_BYTES:
        raise RuntimeError("output exceeded 100 MiB")
    try:
        save_json(temporary, metadata)
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def run_experiment(choice: str, data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Run one fixed cached blend without fitting or refitting any model."""
    if choice not in CHOICES:
        raise ValueError(f"unknown cached choice: {choice}")
    started = time.perf_counter()
    output_dir = artifacts_dir / f"cached_route_blends_{choice}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    metadata_path = output_dir / "compare_metadata.json"
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "choice": choice,
        "current_fit": None,
        "selection": None,
        "validation_results": [],
        "training_seconds": 0.0,
        "fits": 0,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
        "output_dir": str(output_dir),
    }
    save_json(metadata_path, metadata)

    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    try:
        signal.signal(signal.SIGALRM, f._on_timeout)
        signal.signal(signal.SIGTERM, f._on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = f._artifact_snapshot(artifacts_dir, output_dir)
        input_before = f._input_hashes(data_dir)
        source_before = _own_source_hashes()
        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        prior = f._load_prior_lightgbm(artifacts_dir, input_before)
        metadata.update(
            {
                "python": sys.version,
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
            validation, result, metrics = _source_components(
                choice,
                artifacts_dir,
                train,
                y,
                prior,
                input_before,
                f._source_hashes(),
                seed,
            )
            validation_path = output_dir / f"validation_{choice}_seed{seed}.csv"
            validation.to_csv(validation_path, index=False, float_format="%.17g")
            result["validation_predictions_path"] = str(validation_path)
            result["validation_predictions_sha256"] = sha256_file(validation_path)
            metadata["validation_results"].append(result)
            baseline_auc[seed] = metrics["baseline"]["roc_auc"]
            candidate_auc[seed] = metrics["candidate"]["roc_auc"]
            save_json(metadata_path, metadata)
        metadata["selection"] = f.final_sprint_gate(choice, baseline_auc, candidate_auc)
    except BaseException as error:
        caught = error
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        metadata["current_fit"] = None
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
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
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["integrity_error"] = {
                "type": type(integrity_error).__name__,
                "message": str(integrity_error),
            }
            if caught is None:
                caught = integrity_error
                metadata["error"] = metadata["integrity_error"]
        elif caught is not None:
            metadata["status"] = (
                "timeout" if isinstance(caught, f.FinalSprintTimeout) else "failed"
            )
            metadata["selection"] = None
        else:
            metadata["status"] = "completed"
        metadata["elapsed_scoring_seconds"] = round(time.perf_counter() - started, 3)
        metadata["output_bytes"] = f._output_size(output_dir)
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("output exceeded 100 MiB")
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = budget_error
                metadata["error"] = {
                    "type": type(budget_error).__name__,
                    "message": str(budget_error),
                }
        if metadata["status"] != "completed":
            metadata["selection"] = None
        try:
            _save_sized_metadata(metadata_path, metadata)
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["metadata_persistence_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            if caught is None:
                caught = error
            try:
                _save_sized_metadata(metadata_path, metadata)
            except BaseException:
                pass  # Atomic writes leave the earlier non-completed record intact.
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=CHOICES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = run_experiment(args.experiment, args.data_dir, args.artifacts_dir)
    print(json.dumps(metadata["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
