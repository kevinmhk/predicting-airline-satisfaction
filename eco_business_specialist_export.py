"""Export the user-authorized below-gate Eco/business/loyal specialist."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from compare import _resource_delta, _resource_snapshot, prepare_catboost_frame
from eco_business_specialist import (
    BASELINE_WEIGHT,
    GROUP_COLUMNS,
    GROUP_VALUES,
    SPECIALIST_WEIGHT,
    _data_hashes,
    _fold_membership_hash,
    _frozen_config,
    _hash_json,
    _source_hashes,
    _verify_prior_oof,
    group_mask,
    route_candidate,
)
from five_fold_cv import make_five_fold_splits
from lightgbm_experiment import _artifact_snapshot
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

OUTPUT_NAME = "eco_business_specialist_export"
SUBMISSION_NAME = "submission_eco_business_specialist_25pct.csv"
MODEL_NAME = "specialist_model_catboost.cbm"
FEATURE_SPEC_NAME = "feature_spec.json"
GROUP_SPEC_NAME = "group_routing_spec.json"
METADATA_NAME = "export_metadata.json"
OUTPUT_BUDGET_BYTES = 100 * 1024 * 1024
EXPECTED_GROUP_ROWS = 65_542
EXPECTED_BASELINE_SHA256 = (
    "5a5f985fd98a78bce5cf93e0a2dde6f228b36d25cc4d60c54b859be6d7262059"
)


def derive_export_iterations(counts: list[int] | tuple[int, ...]) -> int:
    """Return the median of exactly five positive one-based CV counts."""
    values = [int(count) for count in counts]
    if len(values) != 5 or any(count <= 0 for count in values):
        raise ValueError("export iteration derivation requires five positive counts")
    selected = int(median(values))
    if selected <= 0:
        raise ValueError("derived export iteration count is invalid")
    return selected


def route_csv_records(
    records: list[tuple[str, str]], group: np.ndarray, candidate: np.ndarray
) -> list[tuple[str, str]]:
    """Replace only group probability strings while preserving other CSV text."""
    group = np.asarray(group, dtype=bool)
    candidate = np.asarray(candidate, dtype=np.float64)
    if len(records) != len(group) or len(candidate) != len(records):
        raise ValueError("CSV records, group mask, and candidate must align")
    validate_probabilities(candidate, len(candidate))
    return [
        (record[0], format(float(value), ".17g") if in_group else record[1])
        for record, in_group, value in zip(records, group, candidate, strict=True)
    ]


def _resolve_recorded_path(value: str, project_dir: Path, artifacts_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (project_dir / path, artifacts_dir / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (project_dir / path).resolve()


def _source_fingerprints() -> dict[str, str]:
    return {
        **_source_hashes(),
        "eco_business_specialist_export.py": sha256_file(Path(__file__).resolve()),
    }


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _read_baseline_csv(
    path: Path, test_ids: np.ndarray
) -> tuple[list[tuple[str, str]], np.ndarray]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != [ID_COLUMN, TARGET_COLUMN]:
            raise ValueError(
                "frozen baseline must have exactly id,satisfaction columns"
            )
        for row in reader:
            if len(row) != 2:
                raise ValueError("frozen baseline contains a malformed row")
            records.append((row[0], row[1]))
    expected_ids = [str(int(value)) for value in test_ids]
    actual_ids = [record[0] for record in records]
    if len(records) != len(expected_ids) or actual_ids != expected_ids:
        raise ValueError("frozen baseline IDs do not match exact test order")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("frozen baseline IDs are not unique")
    values = np.asarray([float(record[1]) for record in records], dtype=np.float64)
    validate_probabilities(values, len(test_ids))
    return records, values


def _write_submission(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ID_COLUMN, TARGET_COLUMN])
        writer.writerows(records)


def _validate_submission(
    path: Path,
    expected_records: list[tuple[str, str]],
    group: np.ndarray,
    expected_values: np.ndarray,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("exported submission schema changed")
    actual_records = [tuple(row) for row in rows[1:]]
    if len(actual_records) != len(expected_records) or any(
        len(row) != 2 for row in actual_records
    ):
        raise ValueError("exported submission contains a malformed row")
    if [row[0] for row in actual_records] != [row[0] for row in expected_records]:
        raise ValueError("exported submission IDs changed order")
    group = np.asarray(group, dtype=bool)
    actual_values = np.asarray([float(row[1]) for row in actual_records])
    validate_probabilities(actual_values, len(expected_records))
    if np.any(~group) and any(
        actual[1] != expected[1]
        for actual, expected, in_group in zip(
            actual_records, expected_records, group, strict=True
        )
        if not in_group
    ):
        raise ValueError("non-group baseline probability text changed")
    differences = np.abs(actual_values - expected_values)
    return {
        "rows": len(actual_records),
        "max_abs_difference": float(np.max(differences)),
        "outside_group_text_preserved": True,
        "probability_min": float(np.min(actual_values)),
        "probability_max": float(np.max(actual_values)),
    }


def _clean_reload_candidate(
    model_path: Path,
    feature_spec_path: Path,
    group_spec_path: Path,
    test_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> None:
    code = r"""
import csv
import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

model_path, feature_path, group_path, test_path, baseline_path, output_path = (
    sys.argv[1:]
)
feature_spec = json.loads(open(feature_path, encoding="utf-8").read())
group_spec = json.loads(open(group_path, encoding="utf-8").read())
test = pd.read_csv(test_path)
features = test.loc[:, feature_spec["feature_columns"]].copy()
for column in feature_spec["categorical_columns"]:
    features[column] = (
        features[column].astype("string").fillna("<missing>").astype(object)
    )
with open(baseline_path, "r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    next(reader)
    baseline = np.asarray([float(row[1]) for row in reader], dtype=np.float64)
group = np.ones(len(test), dtype=bool)
for column, value in zip(
    group_spec["group_columns"], group_spec["group_values"], strict=True
):
    group &= test[column].eq(value).to_numpy(dtype=bool)
model = CatBoostClassifier()
model.load_model(model_path)
specialist = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
candidate = baseline.copy()
candidate[group] = (
    group_spec["routing"]["baseline_weight"] * baseline[group]
    + group_spec["routing"]["specialist_weight"] * specialist[group]
)
np.save(output_path, candidate)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(model_path),
            str(feature_spec_path),
            str(group_spec_path),
            str(test_path),
            str(baseline_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _verify_specialist_metadata(
    metadata: dict[str, Any], spec: Any, data_hashes: dict[str, str], folds: Any
) -> list[int]:
    if metadata.get("status") != "completed":
        raise RuntimeError("current specialist CV metadata is not completed")
    completed = metadata.get("completed_fits")
    if not isinstance(completed, list) or len(completed) != 5:
        raise RuntimeError("current specialist CV must contain five completed fits")
    fold_ids = [int(row.get("fold_id", -1)) for row in completed]
    if sorted(fold_ids) != [1, 2, 3, 4, 5]:
        raise RuntimeError("current specialist CV folds are not five distinct folds")
    counts: list[int] = []
    for row in sorted(completed, key=lambda value: int(value["fold_id"])):
        count = int(row["best_iteration_1based"])
        if count != int(row["actual_tree_count"]) or count <= 0:
            raise RuntimeError("specialist CV iteration provenance is invalid")
        counts.append(count)
    expected_feature_spec = {
        "feature_columns": list(spec.feature_columns),
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
    }
    if metadata.get("feature_spec") != expected_feature_spec:
        raise RuntimeError("specialist feature specification changed")
    if metadata.get("configuration") != _frozen_config(spec):
        raise RuntimeError("specialist CV configuration changed")
    if metadata.get("group", {}).get("columns") != list(GROUP_COLUMNS):
        raise RuntimeError("specialist group columns changed")
    if metadata.get("group", {}).get("values") != list(GROUP_VALUES):
        raise RuntimeError("specialist group values changed")
    if metadata.get("group", {}).get("rows") != EXPECTED_GROUP_ROWS:
        raise RuntimeError("specialist group row count changed")
    if (
        metadata.get("input_sha256_before") != data_hashes
        or metadata.get("input_sha256_after") != data_hashes
    ):
        raise RuntimeError("specialist CV input fingerprints changed")
    current_sources = _source_hashes()
    if (
        metadata.get("source_sha256_before") != current_sources
        or metadata.get("source_sha256_after") != current_sources
    ):
        raise RuntimeError("specialist CV source fingerprints changed")
    if metadata.get("configuration_sha256_before") != _hash_json(
        metadata["configuration"]
    ) or metadata.get("configuration_sha256_after") != metadata.get(
        "configuration_sha256_before"
    ):
        raise RuntimeError("specialist configuration fingerprints changed")
    if metadata.get("feature_spec_sha256_before") != _hash_json(
        expected_feature_spec
    ) or metadata.get("feature_spec_sha256_after") != metadata.get(
        "feature_spec_sha256_before"
    ):
        raise RuntimeError("specialist feature fingerprints changed")
    if metadata.get("fold_membership_sha256_before") != _fold_membership_hash(
        folds
    ) or metadata.get("fold_membership_sha256_after") != metadata.get(
        "fold_membership_sha256_before"
    ):
        raise RuntimeError("specialist fold fingerprints changed")
    if metadata.get("selection", {}).get("qualifies") is not False:
        raise RuntimeError(
            "specialist gate provenance is not the recorded below-gate result"
        )
    return counts


def export_submission(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Fit the group-only specialist and write the routed descriptive export."""
    project_dir = Path(__file__).resolve().parent
    data_dir = data_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    output_dir = artifacts_dir / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    metadata_path = output_dir / METADATA_NAME
    metadata: dict[str, Any] = {
        "status": "initializing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "export_authorized_by_user": True,
        "designation": "below_gate_user_authorized_not_winner",
        "output_budget_bytes": OUTPUT_BUDGET_BYTES,
    }
    save_json(metadata_path, metadata)
    started = time.perf_counter()
    protected_before: dict[str, str] = {}
    input_before: dict[str, str] = {}
    source_before: dict[str, str] = {}
    caught: BaseException | None = None
    finalization_errors: list[BaseException] = []
    submission_path = output_dir / SUBMISSION_NAME
    try:
        protected_before = _artifact_snapshot(artifacts_dir, output_dir)
        input_before = _data_hashes(data_dir)
        source_before = _source_fingerprints()
        metadata.update(
            {
                "prior_artifact_sha256_before": protected_before,
                "input_sha256_before": input_before,
                "source_sha256_before": source_before,
            }
        )
        save_json(metadata_path, metadata)

        train = pd.read_csv(data_dir / "train.csv")
        test = pd.read_csv(data_dir / "test.csv")
        spec = validate_inputs(train, test)
        y = target_array(train)
        folds = make_five_fold_splits(y)
        current_cv_path = (
            artifacts_dir / "eco_business_specialist" / "compare_metadata.json"
        )
        current_cv = json.loads(current_cv_path.read_text(encoding="utf-8"))
        counts = _verify_specialist_metadata(current_cv, spec, input_before, folds)
        prior = _verify_prior_oof(data_dir, artifacts_dir, train, y, input_before)

        candidate_oof_path = _resolve_recorded_path(
            current_cv["candidate_oof_path"], project_dir, artifacts_dir
        )
        if not candidate_oof_path.is_file():
            raise FileNotFoundError("specialist candidate OOF provenance is missing")
        with np.load(candidate_oof_path) as saved:
            if not np.array_equal(saved["row_index"], np.arange(len(train))):
                raise RuntimeError("specialist OOF row provenance changed")
            if not np.array_equal(
                saved["id"], train[ID_COLUMN].to_numpy(dtype=np.int64)
            ):
                raise RuntimeError("specialist OOF ID provenance changed")
            if not np.array_equal(saved["fold_id"], prior["fold_ids"]):
                raise RuntimeError("specialist OOF fold provenance changed")
            group = group_mask(train)
            if not np.array_equal(saved["group_mask"], group):
                raise RuntimeError("specialist OOF group mask changed")
            validate_probabilities(saved["baseline_probability"], len(train))
            validate_probabilities(saved["candidate_probability"], len(train))
            if not np.array_equal(
                saved["candidate_probability"][~group],
                saved["baseline_probability"][~group],
            ):
                raise RuntimeError("specialist OOF non-group routing changed")

        if int(group.sum()) != EXPECTED_GROUP_ROWS:
            raise RuntimeError("training specialist group is not exactly 65,542 rows")
        iterations = derive_export_iterations(counts)
        model = CatBoostClassifier(
            iterations=iterations,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=3,
            random_seed=42,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
        )
        model_params = model.get_params()
        if model_params.get("early_stopping_rounds") not in (None,):
            raise RuntimeError("export model unexpectedly enables early stopping")
        if model_params.get("eval_set") is not None:
            raise RuntimeError("export model unexpectedly has an evaluation set")
        X_cat = prepare_catboost_frame(train, spec)
        before_resources = _resource_snapshot()
        fit_started = time.perf_counter()
        model.fit(
            X_cat.iloc[group],
            y[group],
            cat_features=list(spec.categorical_columns),
            verbose=False,
        )
        fit_seconds = time.perf_counter() - fit_started
        after_resources = _resource_snapshot()
        if int(model.tree_count_) != iterations:
            raise RuntimeError(
                "full specialist fit did not retain the derived tree count"
            )
        model_path = output_dir / MODEL_NAME
        model.save_model(model_path)
        model_test = np.asarray(
            model.predict_proba(prepare_catboost_frame(test, spec))[:, 1],
            dtype=np.float64,
        )
        test_group = group_mask(test)
        baseline_metadata_path = (
            artifacts_dir / "slower_learning" / "compare_metadata.json"
        )
        baseline_metadata = json.loads(
            baseline_metadata_path.read_text(encoding="utf-8")
        )
        if baseline_metadata.get("status") != "completed":
            raise RuntimeError("frozen baseline metadata is not completed")
        baseline_path = _resolve_recorded_path(
            baseline_metadata["final"]["submission_path"], project_dir, artifacts_dir
        )
        baseline_hash = sha256_file(baseline_path)
        if (
            baseline_hash != EXPECTED_BASELINE_SHA256
            or baseline_hash != baseline_metadata["final"]["submission_sha256"]
        ):
            raise RuntimeError(
                "frozen baseline submission fingerprint is not authorized"
            )
        test_ids = test[ID_COLUMN].to_numpy(dtype=np.int64)
        baseline_records, baseline_values = _read_baseline_csv(baseline_path, test_ids)
        candidate_values = route_candidate(baseline_values, model_test, test_group)
        output_records = route_csv_records(
            baseline_records, test_group, candidate_values
        )
        _write_submission(submission_path, output_records)

        feature_spec = {
            "feature_columns": list(spec.feature_columns),
            "numeric_columns": list(spec.numeric_columns),
            "categorical_columns": list(spec.categorical_columns),
            "numeric_nan_policy": "preserve numeric NaN; no imputation",
            "categorical_missing_policy": "string <missing>",
            "id_column_excluded": ID_COLUMN,
            "target_column_excluded": TARGET_COLUMN,
        }
        feature_spec_path = output_dir / FEATURE_SPEC_NAME
        save_json(feature_spec_path, feature_spec)
        group_spec = {
            "group_columns": list(GROUP_COLUMNS),
            "group_values": list(GROUP_VALUES),
            "train_group_rows": int(group.sum()),
            "test_group_rows": int(test_group.sum()),
            "routing": {
                "baseline_weight": BASELINE_WEIGHT,
                "specialist_weight": SPECIALIST_WEIGHT,
                "non_group_action": "preserve_baseline_probability_text",
            },
        }
        group_spec_path = output_dir / GROUP_SPEC_NAME
        save_json(group_spec_path, group_spec)
        csv_check = _validate_submission(
            submission_path, baseline_records, test_group, candidate_values
        )
        reload_path = output_dir / "reload_candidate.npy"
        _clean_reload_candidate(
            model_path,
            feature_spec_path,
            group_spec_path,
            data_dir / "test.csv",
            baseline_path,
            reload_path,
        )
        reloaded_candidate = np.asarray(np.load(reload_path), dtype=np.float64)
        reload_difference = float(np.max(np.abs(candidate_values - reloaded_candidate)))
        if reload_difference > 1e-12:
            raise RuntimeError(
                f"clean subprocess reload differs by {reload_difference}"
            )
        reload_csv_difference = float(
            np.max(
                np.abs(
                    candidate_values
                    - np.asarray([float(row[1]) for row in output_records])
                )
            )
        )
        if reload_csv_difference > 1e-12:
            raise RuntimeError(
                "exported CSV probabilities differ from candidate values"
            )
        reload_path.unlink()
        metadata.update(
            {
                "status": "completed",
                "model_path": str(model_path.resolve()),
                "model_sha256": sha256_file(model_path),
                "feature_spec_path": str(feature_spec_path.resolve()),
                "feature_spec_sha256": sha256_file(feature_spec_path),
                "group_routing_spec_path": str(group_spec_path.resolve()),
                "group_routing_spec_sha256": sha256_file(group_spec_path),
                "submission_path": str(submission_path.resolve()),
                "submission_sha256": sha256_file(submission_path),
                "submission_rows": len(output_records),
                "cv_gate_qualifies": False,
                "cv_gate_decision": current_cv["selection"]["decision"],
                "one_based_cv_iteration_counts": counts,
                "selected_iterations": iterations,
                "catboost_configuration": {
                    "iterations": iterations,
                    "depth": 6,
                    "learning_rate": 0.03,
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "l2_leaf_reg": 3,
                    "random_seed": 42,
                    "thread_count": 2,
                    "allow_writing_files": False,
                    "early_stopping_rounds": None,
                    "eval_set": None,
                },
                "fit_seconds": fit_seconds,
                "resources": _resource_delta(before_resources, after_resources),
                "group": {
                    **group_spec,
                    "train_group_ids_sha256": _hash_json(
                        train[ID_COLUMN][group].tolist()
                    ),
                    "test_group_ids_sha256": _hash_json(
                        test[ID_COLUMN][test_group].tolist()
                    ),
                    "test_ids_sha256": _hash_json(test[ID_COLUMN].tolist()),
                    "group_mask_and_ids_aligned": True,
                },
                "baseline": {
                    "metadata_path": str(baseline_metadata_path.resolve()),
                    "submission_path": str(baseline_path.resolve()),
                    "submission_sha256": baseline_hash,
                    "input_frozen": True,
                },
                "csv_validation": csv_check,
                "reload_max_abs_difference": reload_difference,
                "reload_csv_max_abs_difference": reload_csv_difference,
                "runtime_seconds": time.perf_counter() - started,
                "dependencies": {
                    **package_versions(),
                    "catboost": version("catboost"),
                },
                "prior_cv_metadata_path": str(current_cv_path.resolve()),
                "prior_cv_metadata_sha256": sha256_file(current_cv_path),
                "prior_oof_metadata_path": str(prior["metadata_path"].resolve()),
                "prior_oof_metadata_sha256": sha256_file(prior["metadata_path"]),
                "prior_oof_path": str(prior["oof_path"].resolve()),
                "prior_oof_sha256": sha256_file(prior["oof_path"]),
            }
        )
        del model, X_cat
        gc.collect()
    except BaseException as error:
        caught = error
        metadata["status"] = "failed"
        metadata["error"] = _error_record(error)
    finally:
        try:
            input_after = _data_hashes(data_dir)
            metadata["input_sha256_after"] = input_after
            if input_before and input_after != input_before:
                finalization_errors.append(
                    RuntimeError("input CSV fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        try:
            source_after = _source_fingerprints()
            metadata["source_sha256_after"] = source_after
            if source_before and source_after != source_before:
                finalization_errors.append(RuntimeError("source fingerprint changed"))
        except BaseException as error:
            finalization_errors.append(error)
        try:
            protected_after = _artifact_snapshot(artifacts_dir, output_dir)
            metadata["prior_artifact_sha256_after"] = protected_after
            if protected_after != protected_before:
                finalization_errors.append(
                    RuntimeError("pre-existing artifact fingerprint changed")
                )
        except BaseException as error:
            finalization_errors.append(error)
        if finalization_errors:
            metadata["status"] = "failed"
            metadata["integrity_error"] = {
                "type": "RuntimeError",
                "message": "; ".join(str(error) for error in finalization_errors),
            }
            if caught is None:
                caught = RuntimeError(metadata["integrity_error"]["message"])
                metadata["error"] = metadata["integrity_error"]
        if metadata.get("status") != "completed" and submission_path.exists():
            submission_path.unlink()
        metadata["output_bytes"] = sum(
            path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
        )
        if metadata["output_bytes"] > OUTPUT_BUDGET_BYTES:
            budget_error = RuntimeError("export directory exceeded 100 MiB budget")
            metadata["status"] = "failed"
            metadata["error"] = _error_record(budget_error)
            if caught is None:
                caught = budget_error
        save_json(metadata_path, metadata)
    if caught is not None:
        raise caught
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(export_submission(args.data_dir, args.artifacts_dir), indent=2))


if __name__ == "__main__":
    main()
