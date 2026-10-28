import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import source_teacher_cv
from five_fold_cv import make_five_fold_splits
from train import InputSpec


class FakeStudent:
    def __init__(self, rows: int) -> None:
        self.rows = rows

    def get_params(self) -> dict[str, bool]:
        return {"synthetic": True}

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = 0.1 + 0.8 * frame.index.to_numpy(dtype=np.float64) / (self.rows - 1)
        return np.column_stack((1.0 - values, values))


class SourceTeacherCVTests(unittest.TestCase):
    def test_fixed_blends_oof_alignment_and_gate_contract(self) -> None:
        catboost = np.array([0.2, 0.4, 0.6, 0.8])
        lightgbm = np.array([0.4, 0.6, 0.8, 0.2])
        baseline = source_teacher_cv._candidate_blend(catboost, lightgbm)
        np.testing.assert_allclose(baseline, [0.3, 0.5, 0.7, 0.5])
        student = np.array([0.6, 0.8, 0.2, 0.4])
        np.testing.assert_allclose(
            source_teacher_cv._candidate_blend(catboost, student),
            [0.4, 0.6, 0.4, 0.6],
        )
        ids = np.arange(10, dtype=np.int64)
        fold_ids = np.repeat(np.arange(1, 6, dtype=np.int8), 2)
        source_teacher_cv._validate_oof_alignment(
            np.arange(10, dtype=np.int64), ids, fold_ids, ids, fold_ids
        )
        with tempfile.TemporaryDirectory() as directory:
            oof = source_teacher_cv._save_final_oof(
                Path(directory) / "oof_predictions.npz",
                np.arange(4, dtype=np.int64),
                np.array([1, 1, 2, 2], dtype=np.int8),
                np.array([0.3, 0.4, 0.5, 0.6]),
                catboost,
                lightgbm,
                student,
            )
            self.assertEqual(set(oof), {"baseline", "candidate", "student"})
            with np.load(Path(directory) / "oof_predictions.npz") as saved:
                self.assertEqual(len(saved["row_index"]), 4)
                np.testing.assert_allclose(saved["baseline_probability"], baseline)
                np.testing.assert_allclose(
                    saved["candidate_probability"], [0.4, 0.6, 0.4, 0.6]
                )
        gate = source_teacher_cv.five_fold_gate(0.0002, 4, 0.00001)
        self.assertTrue(gate["qualifies"])

    def test_alignment_and_feature_contracts_reject_changes(self) -> None:
        ids = np.arange(4, dtype=np.int64)
        probabilities = np.full(4, 0.5)
        with self.assertRaisesRegex(RuntimeError, "ID order"):
            source_teacher_cv._validate_teacher_cache_arrays(
                np.arange(4), ids[::-1], probabilities, ids
            )
        with self.assertRaisesRegex(RuntimeError, "fold membership"):
            source_teacher_cv._validate_oof_alignment(
                np.arange(4),
                ids,
                np.array([1, 1, 2, 2], dtype=np.int8),
                ids,
                np.array([1, 2, 1, 2], dtype=np.int8),
            )
        spec = InputSpec(
            feature_columns=("Age", "Class"),
            numeric_columns=("Age",),
            categorical_columns=("Class",),
        )
        with self.assertRaisesRegex(RuntimeError, "feature definition"):
            source_teacher_cv._validate_feature_spec(
                {
                    "feature_columns": ["Age"],
                    "numeric_columns": ["Age"],
                    "categorical_columns": ["Class"],
                },
                spec,
                include_teacher=False,
            )

    def test_synthetic_runner_fits_only_five_students(self) -> None:
        rows = 25
        features = pd.DataFrame(
            {
                "Age": np.arange(rows, dtype=np.float64),
                "Class": ["Eco", "Business"] * 12 + ["Eco"],
            }
        )
        student_spec = InputSpec(
            feature_columns=("Age", "Class", source_teacher_cv.SOURCE_TEACHER_FEATURE),
            numeric_columns=("Age", source_teacher_cv.SOURCE_TEACHER_FEATURE),
            categorical_columns=("Class",),
        )
        y = np.array([0, 1] * 12 + [0], dtype=np.int8)
        ids = np.arange(100, 100 + rows, dtype=np.int64)
        folds = make_five_fold_splits(y)
        inner_splits = [
            source_teacher_cv.make_catboost_internal_split(outer_fit, y, seed=42)
            for outer_fit, _holdout in folds
        ]
        split_records = [
            source_teacher_cv._record_split(
                fold_id, outer_fit, holdout, inner[0], inner[1]
            )
            for fold_id, ((outer_fit, holdout), inner) in enumerate(
                zip(folds, inner_splits, strict=True), start=1
            )
        ]
        catboost = np.linspace(0.2, 0.8, rows)
        lightgbm = np.linspace(0.3, 0.7, rows)
        fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
        build_calls = 0

        def build_fake() -> FakeStudent:
            nonlocal build_calls
            build_calls += 1
            return FakeStudent(rows)

        def fit_fake(
            model: FakeStudent,
            _features: pd.DataFrame,
            _labels: np.ndarray,
            inner_fit: np.ndarray,
            inner_validation: np.ndarray,
            _spec: InputSpec,
            *,
            callbacks: list[object],
        ) -> tuple[dict[str, list[str]], float, dict[str, object]]:
            fit_calls.append((inner_fit.copy(), inner_validation.copy()))
            return (
                {},
                0.001,
                {
                    "best_iteration": 3,
                    "actual_iterations": 3,
                    "resources": {"synthetic": True},
                },
            )

        metadata: dict[str, Any] = {
            "current_fit": None,
            "completed_fits": [],
            "fold_results": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            metadata_path = output_dir / "metadata.json"
            with (
                patch.object(
                    source_teacher_cv, "build_lightgbm_classifier", build_fake
                ),
                patch.object(source_teacher_cv, "_fit_with_inner_validation", fit_fake),
                patch.object(
                    source_teacher_cv,
                    "prepare_lightgbm_frame",
                    lambda frame, _spec, _levels: frame,
                ),
            ):
                results, student_oof = source_teacher_cv._run_student_folds(
                    student_features=features.assign(
                        source_teacher_probability=np.linspace(0.2, 0.8, rows)
                    ),
                    student_spec=student_spec,
                    y=y,
                    ids=ids,
                    folds=folds,
                    inner_splits=inner_splits,
                    split_records=split_records,
                    catboost_probability=catboost,
                    lightgbm_probability=lightgbm,
                    metadata=metadata,
                    metadata_path=metadata_path,
                    output_dir=output_dir,
                )
            component_count = len(list(output_dir.glob("fold*_student.npz")))
        self.assertEqual(build_calls, 5)
        self.assertEqual(len(fit_calls), 5)
        self.assertEqual(len(results), 5)
        self.assertTrue(np.isfinite(student_oof).all())
        for inner_fit, inner_validation in fit_calls:
            self.assertFalse(np.intersect1d(inner_fit, inner_validation).size)
        self.assertEqual(component_count, 5)


if __name__ == "__main__":
    unittest.main()
