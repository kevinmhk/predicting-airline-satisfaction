import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import round2_light70_experiment as experiment


class Round2Light70Tests(unittest.TestCase):
    def test_fixed_blend_matches_independent_arithmetic_including_endpoints(
        self,
    ) -> None:
        catboost = np.array([0.0, 0.25, 1.0])
        context = np.array([1.0, 0.5, 0.0])
        hierarchical = np.array([0.0, 1.0, 1.0])
        expected = 0.30 * catboost + 0.35 * context + 0.35 * hierarchical
        np.testing.assert_allclose(
            experiment.light70_blend(catboost, context, hierarchical), expected
        )
        self.assertEqual(
            experiment.light70_blend(catboost, context, hierarchical)[0], 0.35
        )

    def test_fixed_mode_filename_and_schema_are_current(self) -> None:
        output_dir = Path("artifacts/round2_pair_light70")
        self.assertEqual(
            experiment._validation_path(output_dir, 42),
            output_dir / "validation_pair_light70_seed42.csv",
        )
        self.assertEqual(experiment.PAIR_LIGHT70, "pair_light70")
        self.assertEqual(
            experiment.VALIDATION_COLUMNS,
            (
                "row_index",
                "id",
                "catboost_probability",
                "context_probability",
                "hierarchical_probability",
                "baseline_probability",
                "candidate_probability",
            ),
        )

    def test_preflight_failure_persists_fail_closed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            artifacts_dir = root / "artifacts"
            with (
                patch.object(
                    experiment.f, "_input_hashes", return_value={"train.csv": "x"}
                ),
                patch.object(
                    experiment.q,
                    "_common_preflight",
                    side_effect=RuntimeError("synthetic preflight failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "synthetic preflight failure"
                ):
                    experiment.run_experiment(data_dir, artifacts_dir)
            metadata_path = (
                artifacts_dir / "round2_pair_light70" / "compare_metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["current_fit"])
            self.assertEqual(metadata["fits"], 0)
            self.assertEqual(metadata["completed_fits"], [])
            self.assertEqual(
                metadata["error"]["message"], "synthetic preflight failure"
            )


if __name__ == "__main__":
    unittest.main()
