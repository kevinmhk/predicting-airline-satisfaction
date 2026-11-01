import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import final_sprint_experiments as f
import round2_feature_experiments as q
import round3_experiments as r3
from train import ID_COLUMN, InputSpec


def _base_features(rows: int = 40) -> tuple[pd.DataFrame, InputSpec]:
    numeric = ["Flight Distance", r3.f.SOURCE_TEACHER_FEATURE] + [
        f"numeric_{index}" for index in range(33)
    ]
    categorical = ["Gender", "Customer Type", "Type of Travel", "Class"]
    frame = pd.DataFrame(index=np.arange(rows))
    frame["Flight Distance"] = np.tile([0.0, 1.0, 2.0, 3.0], rows // 4)
    frame[r3.f.SOURCE_TEACHER_FEATURE] = np.tile([0.1, 0.2, 0.7, 0.9], rows // 4)
    for index, name in enumerate(numeric[2:]):
        frame[name] = (np.arange(rows) + index) / (rows + index + 1)
    frame["Gender"] = np.tile(["Female", "Male"], rows // 2)
    frame["Customer Type"] = np.tile(["Loyal", "New"], rows // 2)
    frame["Type of Travel"] = np.tile(["Personal", "Business"], rows // 2)
    frame["Class"] = np.tile(["Eco", "Business"], rows // 2)
    return frame, InputSpec(
        tuple(numeric + categorical), tuple(numeric), tuple(categorical)
    )


class Round3SyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features, self.spec = _base_features()
        self.labels = np.tile([0, 1], 20).astype(np.int8)
        self.fit = np.arange(30, dtype=np.int64)
        self.validation = np.arange(30, 35, dtype=np.int64)
        self.holdout = np.arange(35, 40, dtype=np.int64)

    def test_exact_fixed_modes_and_latest_comparator(self) -> None:
        self.assertEqual(len(r3.MODES), 10)
        self.assertEqual(len(set(r3.MODES)), 10)
        self.assertEqual(
            r3.LATEST_BASELINE_AUC,
            {42: 0.9598727586443816, 2024: 0.9600868558567431},
        )
        self.assertNotIn("0.959559", json.dumps(r3.MODE_DEFINITIONS))
        self.assertEqual(
            r3.MODE_DEFINITIONS[r3.LIGHT80]["formula"],
            ".20*catboost + .40*context + .40*hierarchical",
        )

    def test_cached_formulas_have_fixed_endpoints_and_no_label_input(self) -> None:
        catboost = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        context = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
        hierarchical = np.array([1.0, 0.5, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(
            r3.blend_light80(catboost, context, hierarchical),
            0.20 * catboost + 0.40 * context + 0.40 * hierarchical,
        )
        agreement, agreement_weight = r3.blend_agreement_weight(
            catboost, context, hierarchical
        )
        np.testing.assert_allclose(agreement_weight, [0.5, 0.2, 0.2, 0.5, 0.5])
        np.testing.assert_allclose(
            agreement,
            agreement_weight * catboost
            + (1.0 - agreement_weight) * (context + hierarchical) / 2.0,
        )
        uncertainty, uncertainty_weight = r3.blend_uncertainty_weight(
            catboost, context, hierarchical
        )
        mean = (context + hierarchical) / 2.0
        expected_weight = 0.20 + 0.30 * 4.0 * mean * (1.0 - mean)
        np.testing.assert_allclose(uncertainty_weight, expected_weight)
        np.testing.assert_allclose(
            uncertainty,
            uncertainty_weight * catboost + (1.0 - uncertainty_weight) * mean,
        )

    def test_model_overrides_change_only_declared_parameters(self) -> None:
        pairs = (
            (r3.SLOW_CONTEXT, "context", f.ROUTE_CONTEXT_TARGET_127),
            (r3.SLOW_HIERARCHICAL, "hierarchical", f.ROUTE_HIERARCHICAL),
            (r3.HIERARCHICAL63, "hierarchical", f.ROUTE_HIERARCHICAL),
            (r3.CONTEXT_L2_5, "context", f.ROUTE_CONTEXT_TARGET_127),
            (r3.FUSION63, "fusion", f.ROUTE_CONTEXT_TARGET_127),
        )
        for mode, component, base_mode in pairs:
            with self.subTest(mode=mode):
                actual, overrides = r3._model_for(mode, component)
                expected = f.build_trial_classifier(base_mode).get_params()
                for key, value in overrides.items():
                    expected[key] = value
                self.assertEqual(actual.get_params(), expected)
        self.assertEqual(
            r3.MODE_DEFINITIONS[r3.SLOW_CONTEXT]["early_stopping_rounds"], 200
        )
        self.assertEqual(
            r3.MODE_DEFINITIONS[r3.SLOW_HIERARCHICAL]["early_stopping_rounds"], 200
        )

    def test_teacher_consensus_uses_inner_fit_predictors_and_fallback(self) -> None:
        features = self.features.copy()
        features.loc[35:, "Flight Distance"] = 99.0
        values, metadata = r3._teacher_consensus(features, self.fit)
        fallback = float(features.loc[self.fit, f.SOURCE_TEACHER_FEATURE].mean())
        self.assertEqual(metadata["fit_rows"], len(self.fit))
        self.assertEqual(metadata["group_count"], 4)
        self.assertFalse(metadata["uses_labels"])
        self.assertTrue(np.all(values[self.holdout] == fallback))
        changed = features.copy()
        changed.loc[self.validation, f.SOURCE_TEACHER_FEATURE] = 0.0
        changed.loc[self.holdout, f.SOURCE_TEACHER_FEATURE] = 1.0
        changed_values, _ = r3._teacher_consensus(changed, self.fit)
        np.testing.assert_array_equal(values, changed_values)

    def test_joint_feature_preserves_base_columns_and_has_44_40_4_schema(self) -> None:
        extra, _ = q.fit_conditional_feature(
            self.features,
            self.spec,
            self.labels[self.fit],
            self.fit,
            self.validation,
            self.holdout,
            r3.JOINT_PASSENGER,
        )
        for component in ("context", "hierarchical"):
            with self.subTest(component=component):
                result, result_spec, _ = r3._route_feature(
                    component,
                    self.features,
                    self.spec,
                    self.labels,
                    self.fit,
                    self.validation,
                    self.holdout,
                    extra_values=extra,
                    extra_mode=r3.JOINT_PASSENGER,
                )
                self.assertEqual(r3._schema(result_spec), (44, 40, 4))
                pd.testing.assert_frame_equal(
                    result.iloc[:, : self.features.shape[1]], self.features
                )

    def test_fusion_preserves_original_context_representation_and_schema(self) -> None:
        result, result_spec, recipe = r3._fusion_feature(
            self.features,
            self.spec,
            self.labels,
            self.fit,
            self.validation,
            self.holdout,
        )
        self.assertEqual(r3._schema(result_spec), (46, 42, 4))
        pd.testing.assert_frame_equal(
            result.iloc[:, : self.features.shape[1]], self.features
        )
        context = f.fit_route_target_encoding(
            self.features,
            self.spec,
            self.labels[self.fit],
            self.fit,
            self.validation,
            self.holdout,
            include_context=True,
        )
        pd.testing.assert_frame_equal(
            result.iloc[:, :43], context.features, check_dtype=True
        )
        self.assertEqual(
            recipe["excluded_hierarchical_column"], f.FLIGHT_DISTANCE_HIERARCHICAL
        )
        self.assertEqual(
            list(result_spec.feature_columns[-3:]),
            list(r3.FUSION_HIERARCHICAL_FEATURES),
        )

    def test_saved_validation_and_component_outputs_keep_own_rows_and_ids(self) -> None:
        train = pd.DataFrame({ID_COLUMN: np.arange(40, dtype=np.int64)})
        probabilities = np.linspace(0.1, 0.9, len(self.holdout))
        frame = q._validation_frame(
            self.holdout,
            train.iloc[self.holdout][ID_COLUMN].to_numpy(),
            probabilities,
            probabilities,
            probabilities,
            probabilities,
            probabilities,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.csv"
            frame.to_csv(path, index=False)
            loaded = pd.read_csv(path)
            q._assert_prediction_schema(loaded, train, self.holdout)
            record = r3._save_component(
                Path(directory) / "component.npz",
                self.holdout,
                train.iloc[self.holdout][ID_COLUMN].to_numpy(),
                probabilities,
            )
            with np.load(record["path"]) as saved:
                np.testing.assert_array_equal(saved["row_index"], self.holdout)
                np.testing.assert_array_equal(
                    saved["id"], train.iloc[self.holdout][ID_COLUMN].to_numpy()
                )
                np.testing.assert_allclose(saved["probability"], probabilities)

    def test_primary_integrity_and_cleanup_failures_are_recorded_separately(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    r3.f, "_input_hashes", return_value={"data": "fixed"}
                ),
                mock.patch.object(
                    r3, "_own_source_hashes", return_value={"runner": "fixed"}
                ),
                mock.patch.object(
                    r3.f,
                    "_artifact_snapshot",
                    side_effect=[{}, RuntimeError("integrity failure")],
                ),
                mock.patch.object(
                    r3,
                    "_common_preflight",
                    side_effect=r3.Round3Timeout("primary failure"),
                ),
                mock.patch.object(r3.signal, "signal") as handlers,
                mock.patch.object(
                    r3.signal,
                    "setitimer",
                    side_effect=[None, RuntimeError("cleanup failure")],
                ),
            ):
                with self.assertRaisesRegex(r3.Round3Timeout, "primary failure"):
                    r3.run_experiment(r3.LIGHT80, root / "data", root / "artifacts")
            metadata = json.loads(
                (root / "artifacts/round3_light80/compare_metadata.json").read_text()
            )
            self.assertTrue(metadata["timed_out"])
            self.assertEqual(metadata["error"]["message"], "primary failure")
            self.assertIn("integrity_error", metadata)
            self.assertIn("cleanup_error", metadata)
            self.assertEqual(handlers.call_count, 4)
            self.assertIsNone(metadata["selection"])

    def test_final_persistence_failure_retries_failed_state(self) -> None:
        persist = r3.q._persist_metadata
        calls = []

        def fail_once(path, metadata, output):
            calls.append(metadata["status"])
            if len(calls) == 1:
                raise OSError("synthetic disk full")
            persist(path, metadata, output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(r3.f, "_input_hashes", return_value={}),
                mock.patch.object(r3, "_own_source_hashes", return_value={}),
                mock.patch.object(r3.f, "_artifact_snapshot", return_value={}),
                mock.patch.object(
                    r3, "_common_preflight", side_effect=RuntimeError("primary failure")
                ),
                mock.patch.object(r3, "_persist_metadata", side_effect=fail_once),
            ):
                with self.assertRaisesRegex(RuntimeError, "primary failure"):
                    r3.run_experiment(r3.LIGHT80, root / "data", root / "artifacts")
            result = json.loads(
                (root / "artifacts/round3_light80/compare_metadata.json").read_text()
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["message"], "primary failure")
            self.assertEqual(
                result["metadata_persistence_error"]["message"], "synthetic disk full"
            )
            self.assertIsNone(result["selection"])

    def test_timeout_is_countable_failed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(r3.f, "_input_hashes", return_value={}),
                mock.patch.object(r3, "_own_source_hashes", return_value={}),
                mock.patch.object(r3.f, "_artifact_snapshot", return_value={}),
                mock.patch.object(
                    r3, "_common_preflight", side_effect=r3.Round3Timeout("deadline")
                ),
            ):
                with self.assertRaises(r3.Round3Timeout):
                    r3.run_experiment(r3.LIGHT80, root / "data", root / "artifacts")
            result = json.loads(
                (root / "artifacts/round3_light80/compare_metadata.json").read_text()
            )
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["timed_out"])
            self.assertIsNone(result["selection"])

    def test_primary_failure_writes_failed_metadata_without_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory) / "artifacts"
            data = Path(directory) / "data"
            with (
                mock.patch.object(
                    r3.f, "_input_hashes", return_value={"data": "fixed"}
                ),
                mock.patch.object(
                    r3, "_own_source_hashes", return_value={"runner": "fixed"}
                ),
                mock.patch.object(r3.f, "_artifact_snapshot", return_value={}),
                mock.patch.object(
                    r3,
                    "_common_preflight",
                    side_effect=RuntimeError("synthetic primary failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic primary failure"):
                    r3.run_experiment(r3.LIGHT80, data, artifacts)
            metadata = json.loads(
                (artifacts / "round3_light80" / "compare_metadata.json").read_text()
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["current_fit"])
            self.assertEqual(metadata["error"]["message"], "synthetic primary failure")


if __name__ == "__main__":
    unittest.main()
