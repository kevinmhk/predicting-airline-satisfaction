import unittest

import numpy as np
import pandas as pd

from source_teacher_experiment import (
    SOURCE_TEACHER_FEATURE,
    append_teacher_feature,
    fit_source_teacher,
    predict_source_teacher,
    teacher_feature_gate,
)
from train import InputSpec


def teacher_frames() -> tuple[pd.DataFrame, np.ndarray, InputSpec]:
    features = pd.DataFrame(
        {
            "Age": np.arange(20, 44, dtype=np.float64),
            "Class": ["Eco", "Business"] * 12,
        }
    )
    labels = np.array([0, 1] * 12, dtype=np.int8)
    spec = InputSpec(
        feature_columns=("Age", "Class"),
        numeric_columns=("Age",),
        categorical_columns=("Class",),
    )
    return features, labels, spec


class SourceTeacherTests(unittest.TestCase):
    def test_teacher_uses_source_only_disjoint_split_and_full_refit(self) -> None:
        features, labels, spec = teacher_frames()
        teacher = fit_source_teacher(features, labels, spec, iteration_cap=8)
        split = teacher.record["split"]
        self.assertTrue(split["disjoint"])
        self.assertEqual(split["fit_rows"] + split["validation_rows"], len(features))
        self.assertEqual(teacher.record["full_fit"]["fit_rows"], len(features))
        self.assertIsNone(teacher.record["full_fit"]["eval_set"])
        self.assertEqual(teacher.record["full_fit"]["callbacks"], [])
        self.assertIn("resources", teacher.record["full_fit"])
        self.assertEqual(
            teacher.record["full_fit"]["settings"]["n_estimators"],
            teacher.record["probe"]["selected_best_iteration"],
        )
        predictions = predict_source_teacher(teacher, features, spec)
        self.assertEqual(len(predictions), len(features))
        self.assertTrue(np.isfinite(predictions).all())

    def test_teacher_feature_is_appended_without_mutation_or_leakage_columns(
        self,
    ) -> None:
        features, _, spec = teacher_frames()
        original = features.copy(deep=True)
        probabilities = np.linspace(0.1, 0.9, len(features))
        extended, extended_spec = append_teacher_feature(features, probabilities, spec)
        pd.testing.assert_frame_equal(features, original)
        self.assertEqual(
            extended[SOURCE_TEACHER_FEATURE].tolist(), probabilities.tolist()
        )
        self.assertIn(SOURCE_TEACHER_FEATURE, extended_spec.numeric_columns)
        self.assertEqual(extended_spec.categorical_columns, spec.categorical_columns)
        self.assertNotIn("id", extended.columns)
        self.assertNotIn("satisfaction", extended.columns)

    def test_fixed_candidate_alignment_math_and_gate(self) -> None:
        from weighted_blend_experiment import weighted_blend_predictions

        baseline = weighted_blend_predictions(
            [10, 11, 12],
            [12, 10, 11],
            np.array([0.6, 0.2, 0.4]),
            [11, 12, 10],
            np.array([0.5, 0.7, 0.1]),
            slow_catboost_weight=0.5,
            lightgbm_weight=0.5,
        )
        candidate = weighted_blend_predictions(
            [10, 11, 12],
            [12, 10, 11],
            np.array([0.6, 0.2, 0.4]),
            [12, 10, 11],
            np.array([0.5, 0.3, 0.9]),
            slow_catboost_weight=0.5,
            lightgbm_weight=0.5,
        )
        np.testing.assert_allclose(baseline["blend_probability"], [0.15, 0.45, 0.65])
        np.testing.assert_allclose(candidate["blend_probability"], [0.25, 0.65, 0.55])
        accepted = teacher_feature_gate(
            {42: 0.9, 2024: 0.9}, {42: 0.90021, 2024: 0.90022}
        )
        self.assertTrue(accepted["qualifies"])
        rejected = teacher_feature_gate(
            {42: 0.9, 2024: 0.9}, {42: 0.9003, 2024: 0.9001999}
        )
        self.assertFalse(rejected["qualifies"])
        self.assertEqual(rejected["decision"], "retain_current_blend")


if __name__ == "__main__":
    unittest.main()
