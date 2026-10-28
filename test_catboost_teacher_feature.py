import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from catboost_teacher_feature import (
    BASELINE_AUC,
    MINIMUM_AUC_GAIN,
    _load_teacher_student_predictions,
    build_catboost_teacher_classifier,
    teacher_feature_blend_predictions,
    teacher_feature_gate,
)
from compare import prepare_catboost_frame
from slower_learning import build_slow_catboost
from source_teacher_experiment import append_teacher_feature
from train import ID_COLUMN, InputSpec


def _teacher_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": [2, 0, 1],
            ID_COLUMN: [12, 10, 11],
            "teacher_probability": [0.8, 0.2, 0.4],
            "student_probability": [0.5, 0.3, 0.7],
            "baseline_probability": [0.91, 0.92, 0.93],
            "candidate_probability": [0.94, 0.95, 0.96],
        }
    )


class CatBoostTeacherFeatureTests(unittest.TestCase):
    def test_model_is_the_original_slow_builder(self) -> None:
        self.assertEqual(
            build_catboost_teacher_classifier().get_params(),
            build_slow_catboost().get_params(),
        )

    def test_prepare_adds_only_numeric_teacher_feature_without_mutation(self) -> None:
        features = pd.DataFrame(
            {
                "Age": [20.0, 30.0],
                "Class": ["Eco", "Business"],
            }
        )
        original = features.copy(deep=True)
        spec = InputSpec(
            feature_columns=("Age", "Class"),
            numeric_columns=("Age",),
            categorical_columns=("Class",),
        )
        extended, extended_spec = append_teacher_feature(
            features, np.array([0.2, 0.8]), spec
        )
        prepared = prepare_catboost_frame(extended, extended_spec)
        pd.testing.assert_frame_equal(features, original)
        self.assertEqual(
            list(prepared.columns), ["Age", "Class", "source_teacher_probability"]
        )
        self.assertEqual(
            extended_spec.numeric_columns, ("Age", "source_teacher_probability")
        )
        self.assertEqual(extended_spec.categorical_columns, ("Class",))
        self.assertNotIn(ID_COLUMN, prepared)
        self.assertNotIn("satisfaction", prepared)

    def test_shuffled_six_column_cache_uses_student_probability_only(self) -> None:
        with TemporaryDirectory() as directory:
            frame = _load_teacher_student_predictions(
                _write_frame(_teacher_frame(), Path(directory)),
                np.array([0, 1, 2]),
                np.array([10, 11, 12]),
            )
        result = teacher_feature_blend_predictions(
            np.array([10, 11, 12]),
            np.array([12, 10, 11]),
            np.array([0.6, 0.2, 0.4]),
            frame,
            np.array([10, 11, 12]),
            np.array([0.2, 0.4, 0.8]),
        )
        np.testing.assert_allclose(
            result["lightgbm_teacher_probability"], [0.3, 0.7, 0.5]
        )
        np.testing.assert_allclose(result["baseline_probability"], [0.25, 0.55, 0.55])
        np.testing.assert_allclose(result["candidate_probability"], [0.25, 0.55, 0.65])
        self.assertEqual(
            list(result.columns),
            [
                ID_COLUMN,
                "lightgbm_teacher_probability",
                "baseline_probability",
                "candidate_probability",
            ],
        )

    def test_teacher_cache_rejects_duplicate_or_wrong_ids(self) -> None:
        duplicate = _teacher_frame()
        duplicate.loc[1, "row_index"] = 2
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _load_teacher_student_predictions(
                    _write_frame(duplicate, Path(directory)),
                    np.array([0, 1, 2]),
                    np.array([10, 11, 12]),
                )
        wrong_id = _teacher_frame()
        wrong_id.loc[1, ID_COLUMN] = 99
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "IDs"):
                _load_teacher_student_predictions(
                    _write_frame(wrong_id, Path(directory)),
                    np.array([0, 1, 2]),
                    np.array([10, 11, 12]),
                )

    def test_fixed_gate_requires_both_holdouts(self) -> None:
        passed = teacher_feature_gate(
            BASELINE_AUC,
            {
                seed: BASELINE_AUC[seed] + MINIMUM_AUC_GAIN + 1e-9
                for seed in BASELINE_AUC
            },
        )
        self.assertTrue(passed["qualifies"])
        failed = teacher_feature_gate(
            BASELINE_AUC,
            {
                42: BASELINE_AUC[42] + MINIMUM_AUC_GAIN + 1e-9,
                2024: BASELINE_AUC[2024],
            },
        )
        self.assertFalse(failed["qualifies"])
        self.assertEqual(failed["decision"], "retain_current_blend")

    def test_runner_preserves_original_failure_and_clears_selection(self) -> None:
        from catboost_teacher_feature import run_experiment

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "catboost_teacher_feature._input_hashes",
                side_effect=ValueError("original failure"),
            ):
                with self.assertRaisesRegex(ValueError, "original failure"):
                    run_experiment(root / "data", root / "artifacts")
            metadata = json.loads(
                (
                    root / "artifacts/catboost_teacher_feature/compare_metadata.json"
                ).read_text()
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["current_fit"])
            self.assertEqual(metadata["error"]["message"], "original failure")


def _write_frame(frame: pd.DataFrame, directory: Path) -> Path:
    path = directory / "teacher.csv"
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    unittest.main()
