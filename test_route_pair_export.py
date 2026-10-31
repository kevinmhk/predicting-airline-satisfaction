import json
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import route_pair_export as export
from final_sprint_experiments import fit_route_target_encoding
from train import InputSpec


def _route_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flight Distance": [0.0, 0.0, 100.0, 100.0, np.nan, 200.0],
            "Class": ["Eco", "Business", "Eco", "Business", "Eco", "Eco"],
            "Type of Travel": [
                "Business travel",
                "Business travel",
                "Personal Travel",
                "Personal Travel",
                "Business travel",
                "Personal Travel",
            ],
            "Customer Type": [
                "Loyal Customer",
                "Loyal Customer",
                "disloyal Customer",
                "disloyal Customer",
                "Loyal Customer",
                "disloyal Customer",
            ],
        }
    )


class RoutePairExportTests(unittest.TestCase):
    def test_refit_counts_are_ceiling_medians_for_both_required_seeds(self) -> None:
        metadata = {
            "status": "completed",
            "validation_results": [
                {"split_seed": 42, "best_iteration": 189, "actual_iterations": 189},
                {
                    "split_seed": 2024,
                    "best_iteration": 249,
                    "actual_iterations": 249,
                },
            ],
        }
        self.assertEqual(export.derive_refit_iterations(metadata), 219)
        hierarchical = json.loads(json.dumps(metadata))
        for record, count in zip(
            hierarchical["validation_results"], (609, 614), strict=True
        ):
            record["best_iteration"] = record["actual_iterations"] = count
        self.assertEqual(export.derive_refit_iterations(hierarchical), 612)
        with self.assertRaisesRegex(RuntimeError, "two seeds"):
            export.derive_refit_iterations(
                {**metadata, "validation_results": metadata["validation_results"][:1]}
            )
        invalid = json.loads(json.dumps(metadata))
        invalid["validation_results"][1]["actual_iterations"] = 248
        with self.assertRaisesRegex(RuntimeError, "iteration provenance"):
            export.derive_refit_iterations(invalid)

    def test_fixed_student_config_changes_only_rounds_and_leaves(self) -> None:
        baseline = export.build_student_config(1500, 31)
        context = export.build_student_config(219, 127)
        hierarchical = export.build_student_config(612, 31)
        self.assertEqual(
            set(baseline) - {"n_estimators", "num_leaves"},
            set(context) - {"n_estimators", "num_leaves"},
        )
        self.assertEqual(
            {
                key: baseline[key]
                for key in baseline
                if key not in {"n_estimators", "num_leaves"}
            },
            {
                key: context[key]
                for key in context
                if key not in {"n_estimators", "num_leaves"}
            },
        )
        self.assertEqual(context["n_estimators"], 219)
        self.assertEqual(context["num_leaves"], 127)
        self.assertEqual(hierarchical["n_estimators"], 612)
        self.assertEqual(hierarchical["num_leaves"], 31)

    def test_compact_maps_reconstruct_hierarchical_unseen_pair_prior(self) -> None:
        features = _route_frame().iloc[:4].reset_index(drop=True)
        labels = np.array([0, 1, 0, 1], dtype=np.int8)
        maps = export.fit_route_maps(features, labels, hierarchical=True)
        probe = pd.DataFrame(
            {
                "Flight Distance": [999.0, 0.0, np.nan],
                "Class": ["Eco", "Business", "Eco"],
                "Type of Travel": [
                    "Business travel",
                    "Business travel",
                    "Business travel",
                ],
                "Customer Type": [
                    "Loyal Customer",
                    "Loyal Customer",
                    "Loyal Customer",
                ],
            }
        )
        actual = export.apply_route_maps(probe, maps)
        expected = 0.0  # Eco parent mean is zero, unlike global prevalence 0.5.
        self.assertAlmostEqual(actual.iloc[0][export.ROUTE_MAP_NAMES[1]], expected)
        self.assertTrue(np.isfinite(actual.to_numpy()).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.json"
            export.save_route_maps(path, maps)
            reloaded = export.load_route_maps(path)
        pd.testing.assert_frame_equal(actual, export.apply_route_maps(probe, reloaded))

    def test_training_encoding_is_the_existing_crossfit_transform(self) -> None:
        combined_fixture = pd.concat(
            [_route_frame(), _route_frame()], ignore_index=True
        )
        train = combined_fixture.iloc[:10].reset_index(drop=True)
        test = combined_fixture.iloc[10:].reset_index(drop=True)
        combined = pd.concat([train, test], ignore_index=True)
        spec = InputSpec(tuple(combined.columns), tuple(), tuple(combined.columns[1:]))
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        actual = export.fit_route_features(combined, spec, labels, "context")
        expected = fit_route_target_encoding(
            combined,
            spec,
            labels,
            np.arange(len(train)),
            np.array([], dtype=np.int64),
            np.arange(len(train), len(combined)),
            include_context=True,
        )
        np.testing.assert_allclose(
            actual.features.loc[
                : len(train) - 1, list(actual.spec.feature_columns[-4:])
            ],
            expected.features.loc[
                : len(train) - 1, list(expected.spec.feature_columns[-4:])
            ],
            atol=0.0,
            rtol=0.0,
        )

    def test_failed_export_removes_only_new_publishable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "new.csv"
            reload_path = root / "reload.npy"
            csv_path.write_text("candidate")
            reload_path.write_text("temporary")
            metadata = {
                "selection": {"qualifies": True},
                "qualification": {},
                "current_fit": {"running": True},
                "submission": {"path": str(csv_path)},
                "reload_check": {"fresh_interpreter": True},
            }
            export._invalidate_export(metadata, csv_path, reload_path)
            self.assertIsNone(metadata["submission"])
            self.assertIsNone(metadata["reload_check"])
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["qualification"])
            self.assertIsNone(metadata["current_fit"])
            self.assertFalse(csv_path.exists())
            self.assertFalse(reload_path.exists())

    def test_full_maps_match_original_test_transforms_for_both_students(self) -> None:
        combined = pd.concat([_route_frame(), _route_frame()], ignore_index=True)
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.int8)
        spec = InputSpec(
            tuple(combined.columns), ("Flight Distance",), tuple(combined.columns[1:])
        )
        for kind in ("context", "hierarchical"):
            encoded = export.fit_route_features(combined, spec, labels, kind)
            maps = export.fit_route_maps(
                combined.iloc[: len(labels)],
                labels,
                hierarchical=kind == "hierarchical",
                feature_names=tuple(encoded.spec.feature_columns[-4:]),
            )
            actual = export.apply_route_maps(combined.iloc[len(labels) :], maps)
            np.testing.assert_allclose(
                actual.to_numpy(),
                encoded.features.iloc[len(labels) :, -4:].to_numpy(),
                atol=1e-15,
                rtol=0,
            )

    def test_reload_subprocess_uses_remaining_timeout(self) -> None:
        from unittest.mock import patch

        with patch.object(export.subprocess, "run") as run:
            export._fresh_reload_predictions(
                Path("out"), Path("test.csv"), Path("reload.npy"), timeout_seconds=0.5
            )
        self.assertEqual(run.call_args.kwargs["timeout"], 0.5)

    def test_tiny_native_model_save_reload_and_blend_formula(self) -> None:
        spec = InputSpec(("Age", "Class"), ("Age",), ("Class",))
        frame = pd.DataFrame(
            {"Age": [1.0, 2.0, 3.0, 4.0], "Class": ["A", "B", "A", "B"]}
        )
        levels = {"Class": ["A", "B"]}
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2,
            num_leaves=2,
            min_child_samples=1,
            n_jobs=2,
            verbosity=-1,
        )
        prepared = export.prepare_lightgbm_frame(frame, spec, levels)
        model.fit(prepared, np.array([0, 1, 0, 1]), categorical_feature=["Class"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.txt"
            model.booster_.save_model(str(path))
            reloaded = lgb.Booster(model_file=str(path))
            expected = model.predict_proba(prepared)[:, 1]
            actual = reloaded.predict(prepared)
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            export.route_pair_blend(
                np.array([0.2, 0.4]), np.array([0.6, 0.8]), np.array([0.8, 0.4])
            ),
            [0.45, 0.5],
        )


if __name__ == "__main__":
    unittest.main()
