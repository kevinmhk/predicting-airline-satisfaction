import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import distance_profile_export as export
from distance_profile_experiment import (
    DISTANCE_COLUMN,
    append_distance_profile_features,
    apply_distance_profile,
    fit_distance_profile_table,
)
from train import InputSpec


def _small_spec() -> InputSpec:
    numeric = ("Age", DISTANCE_COLUMN, "Rating", "Delay")
    return InputSpec(numeric, numeric, ())


def _small_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [20.0, 30.0, 40.0, np.nan, 60.0],
            DISTANCE_COLUMN: [100.0, 100.0, 200.0, np.nan, 0.0],
            "Rating": [0.0, 2.0, 4.0, np.nan, 0.0],
            "Delay": [10.0, np.nan, 30.0, 40.0, np.nan],
        }
    )


def _full_spec() -> InputSpec:
    numeric = ("Age", DISTANCE_COLUMN, *[f"numeric_{i}" for i in range(15)])
    categorical = ("Gender", "Customer Type", "Type of Travel", "Class")
    return InputSpec(numeric + categorical, numeric, categorical)


def _full_features() -> pd.DataFrame:
    spec = _full_spec()
    values: dict[str, object] = {
        DISTANCE_COLUMN: [0.0, 0.0, 100.0, np.nan],
        "Age": [20.0, 30.0, 40.0, 50.0],
        "Gender": ["Female", "Male", "Female", "Male"],
        "Customer Type": [
            "Loyal Customer",
            "Loyal Customer",
            "disloyal Customer",
            "Loyal Customer",
        ],
        "Type of Travel": [
            "Business travel",
            "Personal Travel",
            "Business travel",
            "Personal Travel",
        ],
        "Class": ["Eco", "Business", "Eco", "Eco Plus"],
    }
    for index, column in enumerate(spec.numeric_columns):
        if column not in values:
            values[column] = [float(index + row) for row in range(4)]
    values["numeric_0"] = [np.nan] * 4
    return pd.DataFrame(values).loc[:, list(spec.feature_columns)]


class DistanceProfileExportTests(unittest.TestCase):
    def test_reload_subprocess_timeout_terminates_child(self) -> None:
        actual_run = export.subprocess.run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path, marker = root / "pid", root / "late-write"
            code = (
                "import os,sys,time;from pathlib import Path;"
                "Path(sys.argv[1]).write_text(str(os.getpid()));"
                "time.sleep(30);Path(sys.argv[2]).write_text('unexpected')"
            )

            def run_short_timeout(args: Any, **kwargs: Any) -> Any:
                self.assertEqual(kwargs["timeout"], export.INTERNAL_TIMEOUT_SECONDS)
                kwargs["timeout"] = 0.5
                return actual_run(
                    [export.sys.executable, "-c", code, str(pid_path), str(marker)],
                    **kwargs,
                )

            with patch.object(export.subprocess, "run", side_effect=run_short_timeout):
                with self.assertRaises(export.subprocess.TimeoutExpired):
                    export._fresh_reload_predictions(
                        model_path=root,
                        teacher_path=root,
                        catboost_path=root,
                        feature_spec_path=root,
                        profile_table_path=root,
                        test_path=root,
                        output_path=root,
                    )
            self.assertTrue(pid_path.exists())
            with self.assertRaises(ProcessLookupError):
                export.os.kill(int(pid_path.read_text()), 0)
            self.assertFalse(marker.exists())

    def test_frozen_native_teacher_accepts_backend_space_normalization(self) -> None:
        spec = _full_spec()
        features = _full_features()
        levels = export.fit_category_levels(features, spec)
        model = export.build_lightgbm_classifier(
            n_estimators=1, min_child_samples=1, num_leaves=2
        )
        model.fit(
            export.prepare_lightgbm_frame(features, spec, levels),
            np.array([0, 1, 0, 1]),
            categorical_feature=list(spec.categorical_columns),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "source_teacher_export"
            folder.mkdir()
            path = folder / "teacher_model.txt"
            model.booster_.save_model(str(path))
            levels_path = folder / "teacher_category_levels.json"
            levels_path.write_text(json.dumps(levels))
            digest = export.sha256_file(path)
            metadata = {
                "status": "completed",
                "feature_spec": {
                    "original": export._feature_spec(spec),
                    "category_levels": {
                        "source_full": levels,
                        "competition_full": levels,
                    },
                },
                "teacher_model": {
                    "actual_iterations": 1,
                    "model_path": str(path),
                    "model_sha256": digest,
                },
                "files": {
                    "teacher_model.txt": digest,
                    "teacher_category_levels.json": export.sha256_file(levels_path),
                },
            }
            (folder / "export_metadata.json").write_text(json.dumps(metadata))
            with patch.object(export, "EXPECTED_TEACHER_ITERATIONS", 1):
                loaded, *_ = export._load_frozen_teacher(root, root, spec)
            self.assertEqual(
                loaded.feature_name(),
                [name.replace(" ", "_") for name in spec.feature_columns],
            )
        self.assertIn("Flight Distance", spec.feature_columns)

    def test_profile_cv_oof_contract_uses_source_teacher_candidate_baseline(
        self,
    ) -> None:
        ids = np.array([10, 11, 12, 13], dtype=np.int64)
        folds = np.array([1, 1, 2, 2], dtype=np.int8)
        teacher = np.full(4, 0.4)
        catboost = np.array([0.2, 0.4, 0.6, 0.8])
        student = np.array([0.3, 0.5, 0.7, 0.9])
        baseline = np.array([0.31, 0.51, 0.71, 0.91])
        arrays: dict[str, np.ndarray] = {
            "row_index": np.arange(4, dtype=np.int64),
            "id": ids,
            "fold_id": folds,
            "teacher_probability": teacher,
            "catboost_probability": catboost,
            "student_probability": student,
            "baseline_probability": baseline,
            "candidate_probability": 0.5 * catboost + 0.5 * student,
        }
        export.validate_profile_cv_oof(arrays, ids, folds, teacher, catboost, baseline)
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            export.validate_profile_cv_oof(
                {
                    **arrays,
                    "baseline_probability": arrays["baseline_probability"][::-1],
                },
                ids,
                folds,
                teacher,
                catboost,
                baseline,
            )

    def test_iteration_lineage_derives_542_and_rejects_bad_folds(self) -> None:
        metadata: dict[str, Any] = {
            "status": "completed",
            "selection": {"qualifies": True},
            "fixed_hypothesis": {
                "fold_count": 5,
                "student_configuration": dict(export.PREDECLARED_CONFIG),
            },
            "fold_results": [
                {
                    "fold_id": fold_id,
                    "student": {
                        "best_iteration_1based": count,
                        "actual_iterations": count,
                    },
                }
                for fold_id, count in enumerate([542, 541, 543, 542, 542], 1)
            ],
        }
        self.assertEqual(export.derive_profile_student_iterations(metadata), 542)
        with self.assertRaisesRegex(RuntimeError, "five fold"):
            export.derive_profile_student_iterations(
                {**metadata, "fold_results": metadata["fold_results"][:-1]}
            )
        invalid = json.loads(json.dumps(metadata))
        invalid["fold_results"][0]["student"]["actual_iterations"] = 541
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            export.derive_profile_student_iterations(invalid)

    def test_profile_table_persistence_preserves_fallback_nan_and_zero(self) -> None:
        spec = _full_spec()
        features = _full_features()
        table = fit_distance_profile_table(features.iloc[:2], spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.npz"
            export._save_export_profile_table(path, table)
            loaded = export._load_export_profile_table(path, spec)
        probe = features.iloc[[0, 2, 3]].copy()
        expected = apply_distance_profile(probe, table)
        actual = apply_distance_profile(probe, loaded)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.iloc[0]["distance_group_count"], 2.0)
        self.assertEqual(actual.iloc[1]["distance_group_count"], 0.0)
        self.assertTrue(
            actual.iloc[1]["distance_group_mean__numeric_0"]
            != actual.iloc[1]["distance_group_mean__numeric_0"]
        )

    def test_profile_student_contract_is_39_columns_and_35_numeric(self) -> None:
        spec = _full_spec()
        features = _full_features()
        teacher_features, teacher_spec = export.append_teacher_feature(
            features, np.full(len(features), 0.5), spec
        )
        table = fit_distance_profile_table(features, spec)
        student_features, student_spec = append_distance_profile_features(
            teacher_features, teacher_spec, table
        )
        export._validate_profile_student_schema(student_spec)
        self.assertEqual(student_features.shape[1], 39)
        self.assertEqual(len(student_spec.numeric_columns), 35)

    def test_failure_preserves_primary_error_and_removes_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    export,
                    "_artifact_snapshot",
                    side_effect=[{}, {"old": "changed"}],
                ),
                patch.object(export, "_input_hashes", return_value={}),
                patch.object(export, "_code_hashes", return_value={}),
                patch.object(
                    export.pd,
                    "read_csv",
                    side_effect=ValueError("original failure"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "original failure"):
                    export.export_submission(root / "data", root / "artifacts")
            output_dir = root / "artifacts/distance_profile_export"
            metadata = json.loads(
                (output_dir / "export_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["message"], "original failure")
            self.assertIn(
                "pre-existing artifact", metadata["integrity_error"]["message"]
            )
            self.assertFalse((output_dir / export.SUBMISSION_NAME).exists())


if __name__ == "__main__":
    unittest.main()
