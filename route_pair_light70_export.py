"""Export the approved cached route-pair blend with 30/35/35 weights."""

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
import pandas as pd

import round2_feature_experiments as q
import round2_light70_experiment as f
import route_pair_export as e
from train import ID_COLUMN, sha256_file, validate_probabilities

OUTPUT_NAME = "route_pair_light70_export"
METADATA_NAME = "export_metadata.json"
SOURCE_NAME = "route_pair_export"
SOURCE_COMPONENTS = "component_test_predictions.npz"
SUBMISSION_NAME = "submission_slowcat_route_context_hierarchical_30_35_35.csv"
RELOAD_NAME = "fresh_reload_predictions.npy"
WEIGHTS = {"catboost": 0.30, "context": 0.35, "hierarchical": 0.35}
SPLIT_SEEDS = (42, 2024)
EXPECTED_ROWS = 299844
EXPECTED_TRAIN_ROWS = 699635
EXPECTED_CONTEXT_ITERATIONS = 219
EXPECTED_HIERARCHICAL_ITERATIONS = 612
EXPECTED_TEACHER_ITERATIONS = 531
EXPECTED_CATBOOST_ITERATIONS = 4740
INTERNAL_TIMEOUT_SECONDS = 1680
EXTERNAL_TIMEOUT_SECONDS = 1800
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024


def _error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _current_code_hashes(project_dir: Path) -> dict[str, str]:
    hashes = e._code_hashes(project_dir)
    hashes.update(f._own_source_hashes())
    hashes[Path(e.__file__).name] = sha256_file(Path(e.__file__))
    hashes[Path(__file__).name] = sha256_file(Path(__file__))
    return hashes


def _recorded_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} fingerprint changed")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _verify_validation_metrics(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    y: np.ndarray,
    holdout: np.ndarray,
    record: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    q._assert_prediction_schema(frame, train, holdout)
    scores = []
    for name in ("baseline", "candidate"):
        actual = q.f._metrics(
            y[holdout], frame[name + "_probability"].to_numpy(dtype=np.float64)
        )
        if any(
            not np.isclose(value, record[name].get(metric, np.nan), rtol=0, atol=1e-14)
            for metric, value in actual.items()
        ):
            raise RuntimeError("pair_light70 validation metrics changed")
        scores.append(actual)
    return scores[0], scores[1]


def _validate_winner(
    artifacts_dir: Path, input_hashes: dict[str, str], data_dir: Path
) -> dict[str, Any]:
    path = artifacts_dir / "round2_pair_light70" / "compare_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("pair_light70 metadata is not completed")
    if (
        metadata.get("input_sha256_before") != input_hashes
        or metadata.get("input_sha256_after") != input_hashes
    ):
        raise RuntimeError("pair_light70 input lineage changed")
    current_sources = f._own_source_hashes()
    if (
        metadata.get("source_sha256_before") != current_sources
        or metadata.get("source_sha256_after") != current_sources
    ):
        raise RuntimeError("pair_light70 source lineage changed")
    definition = metadata.get("mode_definition", {})
    if (
        definition.get("weights") != WEIGHTS
        or definition.get("transform") != "identity"
    ):
        raise RuntimeError("pair_light70 weights or transform changed")

    if len(metadata.get("validation_results", [])) != len(SPLIT_SEEDS):
        raise RuntimeError(
            "pair_light70 validation records are duplicated or incomplete"
        )
    records = {
        int(item.get("split_seed", -1)): item
        for item in metadata.get("validation_results", [])
    }
    if set(records) != set(SPLIT_SEEDS) or len(records) != len(SPLIT_SEEDS):
        raise RuntimeError("pair_light70 validation records are incomplete")
    baseline: dict[int, float] = {}
    candidate: dict[int, float] = {}
    checks: list[dict[str, Any]] = []
    train = pd.read_csv(data_dir / "train.csv")
    y = q.target_array(train)
    prior = q.f._load_prior_lightgbm(artifacts_dir, input_hashes)
    for seed in SPLIT_SEEDS:
        record = records[seed]
        expected_path = (
            artifacts_dir
            / "round2_pair_light70"
            / (f"validation_pair_light70_seed{seed}.csv")
        )
        recorded_path = Path(record.get("validation_predictions_path", ""))
        actual_path = (
            recorded_path if recorded_path.is_absolute() else Path.cwd() / recorded_path
        )
        if actual_path.resolve() != expected_path.resolve():
            raise RuntimeError("pair_light70 validation path changed")
        if record.get("validation_predictions_sha256") != sha256_file(expected_path):
            raise RuntimeError("pair_light70 validation CSV fingerprint changed")
        frame = pd.read_csv(expected_path, float_precision="round_trip")
        _, holdout, _, _, _ = q.f._load_split_arrays(
            artifacts_dir, train, y, prior, seed
        )
        baseline_metrics, candidate_metrics = _verify_validation_metrics(
            frame, train, y, holdout, record
        )
        if list(frame.columns) != list(f.r.VALIDATION_COLUMNS):
            raise RuntimeError("pair_light70 validation schema changed")
        if frame[ID_COLUMN].duplicated().any() or frame["row_index"].duplicated().any():
            raise RuntimeError("pair_light70 validation IDs or rows are duplicated")
        cat = frame["catboost_probability"].to_numpy(dtype=np.float64)
        context = frame["context_probability"].to_numpy(dtype=np.float64)
        hierarchical = frame["hierarchical_probability"].to_numpy(dtype=np.float64)
        for values in (cat, context, hierarchical):
            validate_probabilities(values, len(frame))
        old = 0.50 * cat + 0.25 * context + 0.25 * hierarchical
        new = f.light70_blend(cat, context, hierarchical)
        if not np.allclose(frame["baseline_probability"], old, rtol=0.0, atol=1e-12):
            raise RuntimeError("pair_light70 saved baseline formula changed")
        if not np.allclose(frame["candidate_probability"], new, rtol=0.0, atol=1e-12):
            raise RuntimeError("pair_light70 saved candidate formula changed")
        baseline[seed] = baseline_metrics["roc_auc"]
        candidate[seed] = candidate_metrics["roc_auc"]
        if not np.isclose(
            baseline[seed], f.r.LATEST_BASELINE_AUC[seed], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("pair_light70 baseline AUC changed")
        checks.append(
            {
                "split_seed": seed,
                "rows": len(frame),
                "validation_path": str(expected_path),
                "validation_sha256": sha256_file(expected_path),
                "baseline": record["baseline"],
                "candidate": record["candidate"],
            }
        )
    gate = f.f.final_sprint_gate(f.PAIR_LIGHT70, baseline, candidate)
    if gate != metadata.get("selection") or gate.get("qualifies") is not True:
        raise RuntimeError("pair_light70 saved selection gate changed")
    return {
        "metadata_path": str(path),
        "metadata_sha256": sha256_file(path),
        "selection": gate,
        "validation": checks,
    }


def _validate_source_export(
    artifacts_dir: Path, input_hashes: dict[str, str], project_dir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_dir = artifacts_dir / SOURCE_NAME
    metadata_path = source_dir / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "completed"
        or metadata.get("selection", {}).get("qualifies") is not True
    ):
        raise RuntimeError("source route-pair export is not accepted")
    if (
        metadata.get("input_sha256_before") != input_hashes
        or metadata.get("input_sha256_after") != input_hashes
    ):
        raise RuntimeError("source export input lineage changed")
    recorded_sources = metadata.get("source_sha256_before")
    if recorded_sources != metadata.get("source_sha256_after"):
        raise RuntimeError("source export source fingerprints are inconsistent")
    for name, digest in recorded_sources.items():
        if sha256_file(project_dir / name) != digest:
            raise RuntimeError(f"source export source fingerprint changed: {name}")
    expected_files = {
        SOURCE_COMPONENTS,
        e.CONTEXT_MODEL_NAME,
        e.HIERARCHICAL_MODEL_NAME,
        e.PROFILE_TABLE_NAME,
        e.FEATURE_SPEC_NAME,
        e.CONTEXT_MAPS_NAME,
        e.HIERARCHICAL_MAPS_NAME,
    }
    if set(metadata.get("files", {})) != expected_files:
        raise RuntimeError("source export file manifest changed")
    for name, digest in metadata["files"].items():
        _recorded_file(source_dir / name, digest, f"source export {name}")
    prior_submission = metadata["submission"]
    if (
        Path(prior_submission["path"]).resolve()
        != (source_dir / e.SUBMISSION_NAME).resolve()
    ):
        raise RuntimeError("previous submission path changed")
    _recorded_file(
        source_dir / e.SUBMISSION_NAME,
        prior_submission["sha256"],
        "previous submission",
    )
    if metadata.get("refit_iterations") != {
        "context": EXPECTED_CONTEXT_ITERATIONS,
        "hierarchical": EXPECTED_HIERARCHICAL_ITERATIONS,
    }:
        raise RuntimeError("source student iteration counts changed")
    if metadata.get("teacher_iterations") != EXPECTED_TEACHER_ITERATIONS:
        raise RuntimeError("source teacher iteration count changed")
    for name, expected, num_leaves in (
        ("context_student_fit", EXPECTED_CONTEXT_ITERATIONS, 127),
        ("hierarchical_student_fit", EXPECTED_HIERARCHICAL_ITERATIONS, 31),
    ):
        fit = metadata.get(name, {})
        if (
            fit.get("fit_rows") != EXPECTED_TRAIN_ROWS
            or fit.get("actual_iterations") != expected
            or fit.get("settings", {}).get("num_leaves") != num_leaves
            or fit.get("early_stopping") is not False
            or fit.get("eval_set") is not None
            or fit.get("callbacks") != []
        ):
            raise RuntimeError(f"source {name} lineage changed")
    teacher_path = Path(metadata["teacher_model_path"])
    if teacher_path.resolve() != (
        artifacts_dir / "source_teacher_export" / "teacher_model.txt"
    ).resolve() or metadata.get("teacher_model_sha256") != sha256_file(teacher_path):
        raise RuntimeError("accepted teacher model lineage changed")
    catboost_path = Path(metadata["catboost_model_path"])
    if (
        catboost_path.resolve()
        != (
            artifacts_dir / "slower_learning" / "final_model_catboost_slow.cbm"
        ).resolve()
    ):
        raise RuntimeError("accepted CatBoost model lineage changed")
    slow_metadata = json.loads(
        (artifacts_dir / "slower_learning" / "compare_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        slow_metadata.get("export_fit", {}).get("selected_iterations")
        != EXPECTED_CATBOOST_ITERATIONS
    ):
        raise RuntimeError("accepted CatBoost iteration count changed")
    if metadata.get("catboost_model_sha256") != sha256_file(catboost_path):
        raise RuntimeError("accepted CatBoost model fingerprint changed")
    if slow_metadata.get("export_fit", {}).get("catboost_model_sha256") != metadata.get(
        "catboost_model_sha256"
    ):
        raise RuntimeError("accepted CatBoost lineage fingerprint changed")
    if metadata.get("completed_fits") != ["context_student", "hierarchical_student"]:
        raise RuntimeError("source export contains an unexpected fit lineage")
    component_path = source_dir / SOURCE_COMPONENTS
    with np.load(component_path) as saved:
        if set(saved.files) != {
            "id",
            "catboost",
            "context",
            "hierarchical",
            "candidate",
        }:
            raise RuntimeError("source component fields changed")
        components = {name: np.asarray(saved[name]) for name in saved.files}
    rows = len(components["id"])
    if rows != EXPECTED_ROWS or any(
        value.ndim != 1 or len(value) != rows for value in components.values()
    ):
        raise RuntimeError("source component row contract changed")
    for name in ("catboost", "context", "hierarchical", "candidate"):
        validate_probabilities(components[name].astype(np.float64), rows)
    return (
        {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "files": {
                name: _recorded_file(source_dir / name, digest, "source export file")
                for name, digest in metadata["files"].items()
            },
            "catboost_model_sha256": metadata["catboost_model_sha256"],
        },
        components,
    )


def _validate_fresh_components(
    source: dict[str, np.ndarray], fresh: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    fresh = np.asarray(fresh, dtype=np.float64)
    if fresh.ndim != 2 or fresh.shape[1] != 4 or fresh.shape[0] != len(source["id"]):
        raise RuntimeError("fresh reload component shape changed")
    for column in fresh.T:
        validate_probabilities(column, len(column))
    for index, name in enumerate(("catboost", "context", "hierarchical")):
        difference = float(np.max(np.abs(fresh[:, index] - source[name])))
        if difference > 1e-12:
            raise RuntimeError(f"fresh {name} component changed: {difference}")
    old_difference = float(np.max(np.abs(fresh[:, 3] - source["candidate"])))
    if old_difference > 1e-12:
        raise RuntimeError(f"fresh old blend changed: {old_difference}")
    new_values = f.light70_blend(fresh[:, 0], fresh[:, 1], fresh[:, 2])
    return new_values, {
        "rows": len(fresh),
        "max_abs_difference": {
            "catboost": float(np.max(np.abs(fresh[:, 0] - source["catboost"]))),
            "context": float(np.max(np.abs(fresh[:, 1] - source["context"]))),
            "hierarchical": float(np.max(np.abs(fresh[:, 2] - source["hierarchical"]))),
            "old_blend": old_difference,
        },
        "old_blend_max_abs": old_difference,
    }


def _write_new_submission(
    path: Path, source: dict[str, np.ndarray], fresh: np.ndarray
) -> dict[str, Any]:
    values, _ = _validate_fresh_components(source, fresh)
    return e._write_submission(path, source["id"].astype(np.int64), values)


_invalidate_export = e._invalidate_export


def export_submission(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Export 30/35/35 from existing full-data native models and cached components."""
    project_dir = Path(__file__).resolve().parent
    data_dir = data_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / METADATA_NAME
    submission_path = output_dir / SUBMISSION_NAME
    reload_path = output_dir / RELOAD_NAME
    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "selection": None,
        "qualification": None,
        "current_fit": None,
        "fit_policy": "reuse already verified full-data fits; no refit or new training",
        "fits": 0,
        "new_training_seconds": 0.0,
        "weights": WEIGHTS,
        "internal_timeout_seconds": INTERNAL_TIMEOUT_SECONDS,
        "external_timeout_seconds": EXTERNAL_TIMEOUT_SECONDS,
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
    }
    e._atomic_json(metadata_path, metadata)
    started = time.perf_counter()
    caught: BaseException | None = None
    integrity_errors: list[BaseException] = []
    cleanup_errors: list[BaseException] = []
    input_before: dict[str, str] | None = None
    protected_before: dict[str, str] | None = None
    source_before: dict[str, str] | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    signals_installed = False
    try:
        signal.signal(signal.SIGALRM, e._on_timeout)
        signal.signal(signal.SIGTERM, e._on_timeout)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_TIMEOUT_SECONDS)
        signals_installed = True
        protected_before = e.cached.f._artifact_snapshot(artifacts_dir, output_dir)
        input_before = e.cached.f._input_hashes(data_dir)
        source_before = _current_code_hashes(project_dir)
        winner = _validate_winner(artifacts_dir, input_before, data_dir)
        source_refs, source = _validate_source_export(
            artifacts_dir, input_before, project_dir
        )
        test_ids = pd.read_csv(data_dir / "test.csv")[ID_COLUMN].to_numpy(
            dtype=np.int64
        )
        if not np.array_equal(test_ids, source["id"].astype(np.int64)):
            raise RuntimeError("source component IDs do not match test order")
        metadata.update(
            {
                "input_sha256_before": input_before,
                "winner_source_sha256": f._own_source_hashes(),
                "source_sha256_before": source_before,
                "protected_artifact_sha256_before": protected_before,
                "qualification": winner,
                "source_export": source_refs,
                "code_sha256": source_before,
                "reused_fits": {
                    "context": {
                        "fit_rows": EXPECTED_TRAIN_ROWS,
                        "iterations": EXPECTED_CONTEXT_ITERATIONS,
                        "num_leaves": 127,
                    },
                    "hierarchical": {
                        "fit_rows": EXPECTED_TRAIN_ROWS,
                        "iterations": EXPECTED_HIERARCHICAL_ITERATIONS,
                        "num_leaves": 31,
                    },
                    "teacher_iterations": EXPECTED_TEACHER_ITERATIONS,
                    "catboost_iterations": EXPECTED_CATBOOST_ITERATIONS,
                },
            }
        )
        e._atomic_json(metadata_path, metadata)
        reload_started = time.perf_counter()
        e._fresh_reload_predictions(
            artifacts_dir / SOURCE_NAME,
            data_dir / "test.csv",
            reload_path,
            timeout_seconds=max(
                0.1, INTERNAL_TIMEOUT_SECONDS - (time.perf_counter() - started)
            ),
        )
        fresh = np.asarray(np.load(reload_path), dtype=np.float64)
        reload_path.unlink()
        new_values, fresh_check = _validate_fresh_components(source, fresh)
        submission = e._write_submission(submission_path, source["id"], new_values)
        metadata.update(
            {
                "status": "completed",
                "selection": {
                    "qualifies": True,
                    "decision": "select_final_sprint_pair_light70_export",
                    "weights": WEIGHTS,
                },
                "submission": submission,
                "reload_check": {
                    "fresh_interpreter": True,
                    "seconds": round(time.perf_counter() - reload_started, 3),
                    "temporary_path_removed": not reload_path.exists(),
                    **fresh_check,
                    "max_abs_difference": {
                        **fresh_check["max_abs_difference"],
                        "new_blend": submission["max_abs_difference"],
                    },
                },
            }
        )
    except BaseException as error:
        caught = error
        metadata["error"] = _error(error)
    finally:
        if signals_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
            except BaseException as error:
                cleanup_errors.append(error)
        if input_before is not None:
            try:
                metadata["input_sha256_after"] = e.cached.f._input_hashes(data_dir)
                if metadata["input_sha256_after"] != input_before:
                    integrity_errors.append(
                        RuntimeError("input CSV fingerprint changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if source_before is not None:
            try:
                metadata["source_sha256_after"] = _current_code_hashes(project_dir)
                if metadata["source_sha256_after"] != source_before:
                    integrity_errors.append(RuntimeError("source fingerprint changed"))
            except BaseException as error:
                integrity_errors.append(error)
        if protected_before is not None:
            try:
                protected_after = e.cached.f._artifact_snapshot(
                    artifacts_dir, output_dir
                )
                metadata["protected_artifact_sha256_after"] = protected_after
                metadata["protected_artifacts_unchanged"] = (
                    protected_after == protected_before
                )
                if protected_after != protected_before:
                    integrity_errors.append(
                        RuntimeError("pre-existing artifact changed")
                    )
            except BaseException as error:
                integrity_errors.append(error)
        if integrity_errors:
            metadata["integrity_error"] = _error(
                RuntimeError("; ".join(map(str, integrity_errors)))
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = RuntimeError("integrity checks failed")
                metadata["error"] = _error(caught)
        if caught is not None or integrity_errors or cleanup_errors:
            try:
                _invalidate_export(metadata, submission_path, reload_path)
            except BaseException as error:
                cleanup_errors.append(error)
            metadata["status"] = (
                "failed"
                if not isinstance(caught, e.RoutePairExportTimeout)
                else "timeout"
            )
            metadata["selection"] = None
        if cleanup_errors:
            metadata["cleanup_error"] = _error(
                RuntimeError("; ".join(map(str, cleanup_errors)))
            )
            metadata["status"] = "failed"
            metadata["selection"] = None
            if caught is None:
                caught = RuntimeError("cleanup checks failed")
                metadata["error"] = _error(caught)
        metadata["current_fit"] = None
        metadata["total_seconds"] = round(time.perf_counter() - started, 3)
        try:
            q._persist_metadata(metadata_path, metadata, output_dir)
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["selection"] = None
            metadata["metadata_persistence_error"] = _error(error)
            try:
                _invalidate_export(metadata, submission_path, reload_path)
                q._persist_metadata(metadata_path, metadata, output_dir)
            except BaseException:
                pass
            if caught is None:
                caught = error
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    result = export_submission(args.data_dir, args.artifacts_dir)
    print(
        json.dumps(
            {"status": result["status"], "submission": result["submission"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
