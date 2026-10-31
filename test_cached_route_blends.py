import unittest

import numpy as np

from cached_route_blends import (
    HIERARCHICAL_LIGHT60,
    ROUTE_PAIR_AVERAGE,
    assemble_cached_probabilities,
)


class CachedRouteBlendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_ids = np.array([10, 20, 30])
        self.catboost_ids = np.array([30, 10, 20])
        self.catboost = np.array([0.8, 0.2, 0.5])
        self.context_ids = np.array([20, 30, 10])
        self.context = np.array([0.6, 0.9, 0.4])
        self.hierarchical_ids = np.array([10, 30, 20])
        self.hierarchical = np.array([0.2, 0.7, 0.8])

    def test_route_pair_average_uses_aligned_complementary_students(self) -> None:
        result = assemble_cached_probabilities(
            ROUTE_PAIR_AVERAGE,
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
        np.testing.assert_allclose(result["student_probability"], [0.3, 0.7, 0.8])
        np.testing.assert_allclose(result["candidate_probability"], [0.25, 0.6, 0.8])

    def test_hierarchical_light60_uses_only_one_student(self) -> None:
        result = assemble_cached_probabilities(
            HIERARCHICAL_LIGHT60,
            self.reference_ids,
            self.catboost_ids,
            self.catboost,
            self.hierarchical_ids,
            self.hierarchical,
        )

        np.testing.assert_allclose(result["student_probability"], [0.2, 0.8, 0.7])
        np.testing.assert_allclose(result["candidate_probability"], [0.2, 0.68, 0.74])

    def test_final_metadata_size_counts_its_own_bytes(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        import cached_route_blends as c

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"status": "completed", "selection": {"qualifies": False}}
            c._save_sized_metadata(root / "compare_metadata.json", metadata)
            self.assertEqual(metadata["output_bytes"], c.f._output_size(root))
            self.assertEqual(
                json.loads((root / "compare_metadata.json").read_text()), metadata
            )

    def test_cached_contract_rejects_settings_or_membership_changes(self) -> None:
        import copy

        import cached_route_blends as c

        inner = np.array([0, 2, 4])
        names = [
            "flight_distance_hierarchical_target_mean",
            "flight_distance_class_hierarchical_target_mean",
            "flight_distance_travel_hierarchical_target_mean",
            "flight_distance_customer_hierarchical_target_mean",
        ]
        cats = ["Gender", "Customer Type", "Type of Travel", "Class"]
        record = {
            "settings": c.f.build_trial_classifier(c.f.ROUTE_HIERARCHICAL).get_params(),
            "split": {"sha256": "test"},
            "profile_fit_membership": {
                "rows": 3,
                "sha256": c.f._array_fingerprint(inner),
            },
            "feature_spec": {
                "feature_columns": [f"base_{i}" for i in range(39)] + names,
                "numeric_columns": [f"n_{i}" for i in range(35)] + names,
                "categorical_columns": cats,
            },
        }
        c._validate_source_contract(
            record, "final_sprint_route_hierarchical_encoding", record["split"], inner
        )
        for field in ("settings", "profile_fit_membership"):
            changed = copy.deepcopy(record)
            changed[field] = {}
            with self.assertRaises(RuntimeError):
                c._validate_source_contract(
                    changed,
                    "final_sprint_route_hierarchical_encoding",
                    record["split"],
                    inner,
                )

    def test_alternate_source_filename_is_rejected(self) -> None:
        from pathlib import Path

        import cached_route_blends as c

        directory = Path("artifacts/final_sprint_route_hierarchical_encoding")
        record = {
            "split_seed": 42,
            "validation_predictions_path": str(directory / "alternate.csv"),
        }
        with self.assertRaises(RuntimeError):
            c._resolve_cached_path(directory, record)

    def test_failed_final_write_cannot_publish_completed_metadata(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import cached_route_blends as c

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compare_metadata.json"
            path.write_text(json.dumps({"status": "initializing"}))
            with patch.object(Path, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    c._save_sized_metadata(
                        path, {"status": "completed", "selection": {"qualifies": True}}
                    )
            self.assertEqual(json.loads(path.read_text())["status"], "initializing")

    def test_unknown_choice_and_missing_pair_component_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown cached choice"):
            assemble_cached_probabilities(
                "unknown",
                self.reference_ids,
                self.catboost_ids,
                self.catboost,
                self.hierarchical_ids,
                self.hierarchical,
            )
        with self.assertRaisesRegex(ValueError, "requires two student"):
            assemble_cached_probabilities(
                ROUTE_PAIR_AVERAGE,
                self.reference_ids,
                self.catboost_ids,
                self.catboost,
                self.context_ids,
                self.context,
            )

    def test_mismatched_and_duplicate_ids_fail_through_alignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match reference IDs"):
            assemble_cached_probabilities(
                HIERARCHICAL_LIGHT60,
                self.reference_ids,
                self.catboost_ids,
                self.catboost,
                [10, 20, 99],
                self.hierarchical,
            )
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            assemble_cached_probabilities(
                HIERARCHICAL_LIGHT60,
                self.reference_ids,
                [30, 10, 10],
                self.catboost,
                self.hierarchical_ids,
                self.hierarchical,
            )


if __name__ == "__main__":
    unittest.main()
