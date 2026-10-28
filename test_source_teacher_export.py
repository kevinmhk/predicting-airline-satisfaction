import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import source_teacher_export
from train import InputSpec


class SourceTeacherExportTests(unittest.TestCase):
    def test_eight_field_cv_contract_uses_frozen_original_lightgbm(self) -> None:
        ids = np.array([10, 11])
        cat = np.array([0.2, 0.4])
        light = np.array([0.6, 0.8])
        student = np.array([0.4, 0.6])
        teacher = np.array([0.1, 0.9])
        folds = np.array([1, 2])
        arrays = {
            "row_index": np.arange(2),
            "id": ids,
            "fold_id": folds,
            "teacher_probability": teacher,
            "catboost_probability": cat,
            "student_probability": student,
            "baseline_probability": 0.5 * cat + 0.5 * light,
            "candidate_probability": 0.5 * cat + 0.5 * student,
        }
        prior = {"fold_ids": folds, "catboost_seed42": cat, "lightgbm": light}
        source_teacher_export.validate_cv_oof(arrays, ids, prior, teacher)
        with self.assertRaisesRegex(RuntimeError, "CatBoost component"):
            source_teacher_export.validate_cv_oof(
                {**arrays, "catboost_probability": cat[::-1]}, ids, prior, teacher
            )

    def test_original_failure_is_not_overwritten_by_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    source_teacher_export,
                    "_artifact_snapshot",
                    side_effect=[{}, {"old": "changed"}],
                ),
                patch.object(source_teacher_export, "_input_hashes", return_value={}),
                patch.object(source_teacher_export, "_code_hashes", return_value={}),
                patch.object(source_teacher_export, "_versions", return_value={}),
                patch.object(
                    source_teacher_export.pd,
                    "read_csv",
                    side_effect=ValueError("original failure"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "original failure"):
                    source_teacher_export.export_submission(
                        root / "data", root / "artifacts"
                    )
            metadata = json.loads(
                (
                    root / "artifacts/source_teacher_export/export_metadata.json"
                ).read_text()
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["message"], "original failure")
            self.assertIn(
                "pre-existing artifact", metadata["integrity_error"]["message"]
            )

    def test_iteration_lineage_uses_five_distinct_positive_counts(self) -> None:
        metadata: dict[str, Any] = {
            "status": "completed",
            "selection": {"qualifies": True},
            "fixed_hypothesis": {"fold_count": 5},
            "fold_results": [
                {
                    "fold_id": fold_id,
                    "student": {
                        "best_iteration_1based": count,
                        "actual_iterations": count,
                    },
                }
                for fold_id, count in enumerate([795, 299, 417, 777, 533], 1)
            ],
        }
        self.assertEqual(source_teacher_export.derive_student_iterations(metadata), 533)

    def test_iteration_lineage_rejects_bad_gate_or_fold_provenance(self) -> None:
        metadata: dict[str, Any] = {
            "status": "completed",
            "selection": {"qualifies": False},
            "fixed_hypothesis": {"fold_count": 5},
            "fold_results": [],
        }
        with self.assertRaisesRegex(RuntimeError, "not qualified"):
            source_teacher_export.derive_student_iterations(metadata)
        metadata["selection"]["qualifies"] = True
        metadata["fold_results"] = [
            {
                "fold_id": 1,
                "student": {"best_iteration_1based": 3, "actual_iterations": 2},
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "five fold results"):
            source_teacher_export.derive_student_iterations(metadata)

    def test_teacher_lineage_rejects_new_search_or_early_stopping(self) -> None:
        metadata: dict[str, Any] = {
            "teacher": {
                "source_rows": 129859,
                "probe": {"selected_best_iteration": 531},
                "full_fit": {
                    "actual_rounds": 531,
                    "fit_rows": 129859,
                    "eval_set": None,
                    "callbacks": [],
                },
            }
        }
        self.assertEqual(source_teacher_export.derive_teacher_iterations(metadata), 531)
        metadata["teacher"]["full_fit"]["callbacks"] = ["early_stopping"]
        with self.assertRaisesRegex(RuntimeError, "evaluation or callbacks"):
            source_teacher_export.derive_teacher_iterations(metadata)

    def test_export_config_changes_only_iteration_count(self) -> None:
        teacher = source_teacher_export.build_export_config(531)
        student = source_teacher_export.build_export_config(533)
        self.assertEqual(
            set(teacher) - {"n_estimators"}, set(student) - {"n_estimators"}
        )
        self.assertEqual(
            {name: teacher[name] for name in teacher if name != "n_estimators"},
            {name: student[name] for name in student if name != "n_estimators"},
        )
        self.assertEqual(teacher["n_estimators"], 531)
        self.assertEqual(student["n_estimators"], 533)

    def test_full_fit_policy_passes_only_categories_and_no_eval_data(self) -> None:
        model = Mock()
        spec = InputSpec(("Age", "Class"), ("Age",), ("Class",))
        features = pd.DataFrame({"Age": [1.0, 2.0], "Class": ["Eco", "Business"]})
        source_teacher_export.fit_full_lightgbm(
            model, features, np.array([0, 1], dtype=np.int8), spec
        )
        kwargs = model.fit.call_args.kwargs
        self.assertEqual(kwargs, {"categorical_feature": ["Class"]})
        self.assertNotIn("eval_set", kwargs)
        self.assertNotIn("callbacks", kwargs)

    def test_feature_order_blend_and_id_contract(self) -> None:
        spec = InputSpec(("Age", "Class"), ("Age",), ("Class",))
        feature_spec = source_teacher_export.build_feature_spec(
            spec,
            {"Class": ["Eco", "Business"]},
            {"Class": ["Eco", "Business"]},
        )
        self.assertEqual(feature_spec["original"]["feature_columns"], ["Age", "Class"])
        self.assertEqual(
            feature_spec["extended"]["feature_columns"],
            ["Age", "Class", "source_teacher_probability"],
        )
        self.assertNotIn("id", feature_spec["original"]["feature_columns"])
        self.assertNotIn("satisfaction", feature_spec["extended"]["feature_columns"])
        np.testing.assert_allclose(
            source_teacher_export.blend_probabilities(
                np.array([0.2, 0.4]), np.array([0.6, 0.8])
            ),
            [0.4, 0.6],
        )
        source_teacher_export.validate_test_ids(
            np.array([10, 11], dtype=np.int64), np.array([10, 11], dtype=np.int64)
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            source_teacher_export.validate_test_ids(
                np.array([11, 10], dtype=np.int64), np.array([10, 11], dtype=np.int64)
            )

    def test_submission_validation_rejects_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            path.write_text("id,satisfaction\n10,nan\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                source_teacher_export.validate_submission_file(
                    path, pd.Series([10], name="id")
                )


if __name__ == "__main__":
    unittest.main()
