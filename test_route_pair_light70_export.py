import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import route_pair_light70_export as export


class RoutePairLight70ExportTests(unittest.TestCase):
    def test_forged_winner_metrics_are_rejected(self) -> None:
        frame = pd.DataFrame(
            {name: [0.5, 0.5] for name in export.f.r.VALIDATION_COLUMNS}
        )
        frame["id"] = [11, 12]
        frame["row_index"] = [0, 1]
        train = pd.DataFrame({"id": [11, 12]})
        y = np.array([0, 1])
        hold = np.array([0, 1])
        forged = {
            "baseline": {"roc_auc": 1.0, "log_loss": 0.0},
            "candidate": {"roc_auc": 1.0, "log_loss": 0.0},
        }
        with self.assertRaisesRegex(RuntimeError, "metrics changed"):
            export._verify_validation_metrics(frame, train, y, hold, forged)

    def test_fresh_components_recompute_new_blend_and_check_old_column(self) -> None:
        ids = np.array([101, 102, 103], dtype=np.int64)
        source = {
            "id": ids,
            "catboost": np.array([0.2, 0.4, 0.6]),
            "context": np.array([0.4, 0.6, 0.8]),
            "hierarchical": np.array([0.8, 0.6, 0.4]),
            "candidate": np.array([0.4, 0.5, 0.6]),
        }
        fresh = np.column_stack(
            (
                source["catboost"],
                source["context"],
                source["hierarchical"],
                source["candidate"],
            )
        )

        new_values, checks = export._validate_fresh_components(source, fresh)

        np.testing.assert_allclose(new_values, [0.48, 0.54, 0.6])
        self.assertEqual(checks["rows"], 3)
        self.assertEqual(checks["old_blend_max_abs"], 0.0)

    def test_failed_export_cleanup_clears_only_new_publishable_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "new.csv"
            reload_path = root / "fresh.npy"
            csv_path.write_text("candidate", encoding="utf-8")
            reload_path.write_bytes(b"temporary")
            metadata = {
                "selection": {"qualifies": True},
                "qualification": {},
                "current_fit": {"status": "running"},
                "submission": {"path": str(csv_path)},
                "reload_check": {"fresh_interpreter": True},
            }

            export._invalidate_export(metadata, csv_path, reload_path)

            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["qualification"])
            self.assertIsNone(metadata["current_fit"])
            self.assertIsNone(metadata["submission"])
            self.assertIsNone(metadata["reload_check"])
            self.assertFalse(csv_path.exists())
            self.assertFalse(reload_path.exists())

    def test_complete_export_reads_plain_npy_from_fresh_child(self) -> None:
        source = {
            "id": np.array([201, 202]),
            "catboost": np.array([0.1, 0.9]),
            "context": np.array([0.2, 0.8]),
            "hierarchical": np.array([0.3, 0.7]),
        }
        source["candidate"] = (
            0.5 * source["catboost"]
            + 0.25 * source["context"]
            + 0.25 * source["hierarchical"]
        )
        fresh = np.column_stack(
            [source[k] for k in ("catboost", "context", "hierarchical", "candidate")]
        )

        def save_fresh(_source, _test, destination, *, timeout_seconds):
            np.save(destination, fresh)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "test.csv").write_text("id\n201\n202\n")
            with (
                patch.object(
                    export.e.cached.f,
                    "_input_hashes",
                    return_value={"test.csv": "fixture"},
                ),
                patch.object(
                    export, "_current_code_hashes", return_value={"fixture": "hash"}
                ),
                patch.object(
                    export,
                    "_validate_winner",
                    return_value={"selection": {"qualifies": True}},
                ),
                patch.object(
                    export, "_validate_source_export", return_value=({}, source)
                ),
                patch.object(
                    export.e, "_fresh_reload_predictions", side_effect=save_fresh
                ),
            ):
                result = export.export_submission(data, root / "artifacts")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["submission"]["rows"], 2)
            self.assertTrue(result["reload_check"]["fresh_interpreter"])
            self.assertTrue(result["reload_check"]["temporary_path_removed"])
            self.assertEqual(result["fits"], 0)

    def test_mocked_submission_boundary_uses_existing_writer(self) -> None:
        ids = np.array([201, 202], dtype=np.int64)
        source = {
            "id": ids,
            "catboost": np.array([0.1, 0.9]),
            "context": np.array([0.2, 0.8]),
            "hierarchical": np.array([0.3, 0.7]),
            "candidate": np.array([0.2, 0.8]),
        }
        fresh = np.column_stack(
            (
                source["catboost"],
                source["context"],
                source["hierarchical"],
                source["candidate"],
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            with patch.object(
                export.e, "_write_submission", wraps=export.e._write_submission
            ) as writer:
                result = export._write_new_submission(path, source, fresh)

            writer.assert_called_once()
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["columns"], ["id", "satisfaction"])
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
