import unittest

import numpy as np

from round2_cached_experiments import (
    PAIR_LIGHT60,
    PAIR_LOGIT_AVERAGE,
    PAIR_RANK_AVERAGE,
    assemble_probabilities,
)


class Round2CachedExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_ids = np.array([10, 20, 30])
        self.catboost_ids = np.array([30, 10, 20])
        self.catboost = np.array([0.8, 0.2, 0.5])
        self.context_ids = np.array([20, 30, 10])
        self.context = np.array([0.6, 0.9, 0.4])
        self.hierarchical_ids = np.array([10, 30, 20])
        self.hierarchical = np.array([0.2, 0.7, 0.8])

    def test_latest_incumbent_path_is_aggregate_not_student_filename(self) -> None:
        from pathlib import Path

        import round2_cached_experiments as r

        directory = Path("artifacts/cached_route_blends_route_pair_average")
        expected = directory / "validation_route_pair_average_seed42.csv"
        self.assertEqual(
            r._incumbent_prediction_path(
                directory, {"validation_predictions_path": str(expected)}, 42
            ),
            expected,
        )
        with self.assertRaises(RuntimeError):
            r._incumbent_prediction_path(
                directory,
                {"validation_predictions_path": str(directory / "wrong.csv")},
                42,
            )

    def test_pair_light60_uses_fixed_three_component_weights(self) -> None:
        result = assemble_probabilities(
            PAIR_LIGHT60,
            self.reference_ids,
            self.catboost_ids,
            self.catboost,
            self.context_ids,
            self.context,
            self.hierarchical_ids,
            self.hierarchical,
        )

        np.testing.assert_array_equal(result["id"], self.reference_ids)
        np.testing.assert_allclose(result["catboost_probability"], [0.2, 0.5, 0.8])
        np.testing.assert_allclose(result["context_probability"], [0.4, 0.6, 0.9])
        np.testing.assert_allclose(result["hierarchical_probability"], [0.2, 0.8, 0.7])
        np.testing.assert_allclose(result["baseline_probability"], [0.25, 0.6, 0.8])
        np.testing.assert_allclose(result["candidate_probability"], [0.26, 0.62, 0.8])

    def test_pair_rank_average_uses_average_ties_and_is_id_invariant(self) -> None:
        result = assemble_probabilities(
            PAIR_RANK_AVERAGE,
            self.reference_ids,
            self.catboost_ids,
            self.catboost,
            self.context_ids,
            self.context,
            self.hierarchical_ids,
            self.hierarchical,
        )
        tied = assemble_probabilities(
            PAIR_RANK_AVERAGE,
            self.reference_ids,
            np.array([30, 10, 20]),
            np.array([0.8, 0.2, 0.2]),
            self.context_ids,
            self.context,
            self.hierarchical_ids,
            self.hierarchical,
        )
        permuted = assemble_probabilities(
            PAIR_RANK_AVERAGE,
            self.reference_ids,
            np.array([20, 30, 10]),
            np.array([0.5, 0.8, 0.2]),
            np.array([10, 30, 20]),
            np.array([0.4, 0.9, 0.6]),
            np.array([30, 20, 10]),
            np.array([0.7, 0.8, 0.2]),
        )

        np.testing.assert_allclose(
            result["candidate_probability"], [0.25, 0.5625, 0.6875]
        )
        np.testing.assert_allclose(tied["candidate_probability"], [0.3125, 0.5, 0.6875])
        np.testing.assert_allclose(
            permuted["candidate_probability"], result["candidate_probability"]
        )

    def test_pair_logit_average_clips_endpoints_and_is_finite(self) -> None:
        result = assemble_probabilities(
            PAIR_LOGIT_AVERAGE,
            self.reference_ids,
            self.reference_ids,
            np.array([0.0, 0.5, 1.0]),
            self.reference_ids,
            np.array([0.0, 0.5, 1.0]),
            self.reference_ids,
            np.array([0.0, 0.5, 1.0]),
        )

        expected = np.array([1e-6, 0.5, 1 - 1e-6])
        np.testing.assert_allclose(result["candidate_probability"], expected)
        self.assertTrue(np.isfinite(result["candidate_probability"]).all())
        self.assertTrue(
            (
                (result["candidate_probability"] >= 0.0)
                & (result["candidate_probability"] <= 1.0)
            ).all()
        )

    def test_malformed_shapes_ids_and_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assemble_probabilities(
                PAIR_LIGHT60,
                np.array([], dtype=np.int64),
                self.catboost_ids,
                self.catboost,
                self.context_ids,
                self.context,
                self.hierarchical_ids,
                self.hierarchical,
            )
        with self.assertRaises(ValueError):
            assemble_probabilities(
                PAIR_LIGHT60,
                self.reference_ids,
                np.array([10, 20, 99]),
                self.catboost,
                self.context_ids,
                self.context,
                self.hierarchical_ids,
                self.hierarchical,
            )
        with self.assertRaises(ValueError):
            assemble_probabilities(
                PAIR_LIGHT60,
                self.reference_ids,
                np.array([10, 10, 30]),
                self.catboost,
                self.context_ids,
                self.context,
                self.hierarchical_ids,
                self.hierarchical,
            )
        for bad in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                assemble_probabilities(
                    PAIR_LIGHT60,
                    self.reference_ids,
                    self.catboost_ids,
                    np.array([bad, 0.5, 0.8]),
                    self.context_ids,
                    self.context,
                    self.hierarchical_ids,
                    self.hierarchical,
                )


if __name__ == "__main__":
    unittest.main()
