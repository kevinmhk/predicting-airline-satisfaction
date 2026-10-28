import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from lightgbm_capacity import (
    BASELINE_AUC,
    CAPACITY_NUM_LEAVES,
    build_lightgbm_capacity_classifier,
    capacity_gate,
    fixed_capacity_blend_predictions,
)
from lightgbm_experiment import build_lightgbm_classifier


class LightGBMCapacityTests(unittest.TestCase):
    def test_capacity_changes_only_num_leaves(self) -> None:
        original = build_lightgbm_classifier().get_params()
        capacity = build_lightgbm_capacity_classifier().get_params()
        changed = {
            key: (original.get(key), capacity.get(key))
            for key in set(original) | set(capacity)
            if original.get(key) != capacity.get(key)
        }
        self.assertEqual(changed, {"num_leaves": (31, CAPACITY_NUM_LEAVES)})

    def test_fixed_blends_align_ids_and_gate_both_seeds(self) -> None:
        result = fixed_capacity_blend_predictions(
            np.array([10, 11, 12]),
            np.array([12, 10, 11]),
            np.array([0.6, 0.2, 0.4]),
            np.array([11, 12, 10]),
            np.array([0.5, 0.7, 0.1]),
            np.array([10, 11, 12]),
            np.array([0.3, 0.9, 0.5]),
        )
        np.testing.assert_allclose(result["baseline_probability"], [0.15, 0.45, 0.65])
        np.testing.assert_allclose(result["candidate_probability"], [0.25, 0.65, 0.55])
        passed = capacity_gate({seed: 0.0002 for seed in BASELINE_AUC})
        self.assertFalse(capacity_gate({42: 0.0003})["qualifies"])
        self.assertFalse(capacity_gate({42: 0.0003, 2024: 0.0001999})["qualifies"])
        self.assertTrue(passed["qualifies"])
        self.assertEqual(passed["decision"], "select_lightgbm63_blend")
        failed = capacity_gate({42: 0.0003, 2024: 0.0001})
        self.assertFalse(failed["qualifies"])
        self.assertEqual(failed["decision"], "retain_current_blend")

    def test_failure_preserves_original_error_and_clears_selection(self) -> None:
        from lightgbm_capacity import run_experiment

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("lightgbm_capacity._input_hashes", return_value={}),
                patch("lightgbm_capacity._package_versions", return_value={}),
                patch("lightgbm_capacity._source_hashes", return_value={}),
                patch(
                    "lightgbm_capacity._artifact_snapshot",
                    side_effect=[{}, {"old": "changed"}],
                ),
                patch(
                    "lightgbm_capacity.pd.read_csv",
                    side_effect=ValueError("original failure"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "original failure"):
                    run_experiment(root / "data", root / "artifacts")
            path = root / "artifacts/lightgbm_capacity63/compare_metadata.json"
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["current_fit"])
            self.assertEqual(metadata["error"]["message"], "original failure")
            self.assertIn(
                "pre-existing artifact", metadata["integrity_error"]["message"]
            )

    def test_fixed_blend_rejects_id_mismatch_and_duplicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match"):
            fixed_capacity_blend_predictions(
                np.array([10, 11]),
                np.array([10, 11]),
                np.array([0.2, 0.4]),
                np.array([10, 12]),
                np.array([0.3, 0.5]),
                np.array([10, 11]),
                np.array([0.1, 0.9]),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            fixed_capacity_blend_predictions(
                np.array([10, 11]),
                np.array([10, 10]),
                np.array([0.2, 0.4]),
                np.array([10, 11]),
                np.array([0.3, 0.5]),
                np.array([10, 11]),
                np.array([0.1, 0.9]),
            )


if __name__ == "__main__":
    unittest.main()
