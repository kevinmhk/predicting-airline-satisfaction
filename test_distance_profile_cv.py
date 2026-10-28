import unittest

import numpy as np

import distance_profile_cv


def _arrays() -> dict[str, np.ndarray]:
    catboost = np.array([0.2, 0.4, 0.6, 0.8])
    teacher_student = np.array([0.3, 0.5, 0.7, 0.9])
    return {
        "row_index": np.arange(4, dtype=np.int64),
        "id": np.arange(10, 14, dtype=np.int64),
        "fold_id": np.array([1, 1, 2, 2], dtype=np.int8),
        "teacher_probability": np.full(4, 0.4),
        "catboost_probability": catboost,
        "student_probability": teacher_student,
        "baseline_probability": 0.5 * catboost + 0.5 * 0.6,
        "candidate_probability": 0.5 * catboost + 0.5 * teacher_student,
    }


def _prior(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "fold_ids": arrays["fold_id"],
        "catboost_seed42": arrays["catboost_probability"],
        "lightgbm": np.full(4, 0.6),
    }


class DistanceProfileCVTests(unittest.TestCase):
    def test_current_baseline_uses_prior_teacher_candidate(self) -> None:
        arrays = _arrays()
        np.testing.assert_array_equal(
            distance_profile_cv._current_baseline_probability(arrays),
            arrays["candidate_probability"],
        )
        self.assertFalse(
            np.array_equal(
                arrays["candidate_probability"], arrays["baseline_probability"]
            )
        )

    def test_eight_field_contract_rejects_id_and_fold_drift(self) -> None:
        arrays = _arrays()
        with self.assertRaisesRegex(ValueError, "IDs"):
            drifted = dict(arrays)
            drifted["id"] = arrays["id"][::-1]
            distance_profile_cv._validate_teacher_cv_oof(
                drifted, arrays["id"], _prior(arrays), arrays["teacher_probability"]
            )
        with self.assertRaisesRegex(RuntimeError, "fold"):
            drifted = dict(arrays)
            drifted["fold_id"] = np.array([1, 2, 1, 2], dtype=np.int8)
            distance_profile_cv._validate_teacher_cv_oof(
                drifted, arrays["id"], _prior(arrays), arrays["teacher_probability"]
            )


if __name__ == "__main__":
    unittest.main()
