import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import blend_experiment as blend_module
import categorical_ratings as categorical_module
import compare as compare_module
import feature_experiment as feature_module
import lightgbm_experiment as lightgbm_module
import longer_catboost as longer_catboost_module
import source_augmentation as source_module
import weighted_blend_experiment as weighted_blend_module
from blend_experiment import aligned_blend_predictions, confirmation_gate
from compare import (
    _fit_catboost,
    _refit_winner,
    build_catboost_classifier,
    make_catboost_internal_split,
    prepare_catboost_frame,
    run_deeper_catboost,
)
from lightgbm_experiment import (
    _artifact_snapshot,
    _fit_callbacks,
    _fit_with_inner_validation,
    _fresh_reload_predictions,
    _selected_iterations,
    _selection,
    _terminal_summary,
    build_lightgbm_classifier,
    fit_category_levels,
    prepare_lightgbm_frame,
    run_experiment,
)
from train import (
    CATEGORICAL_COLUMNS,
    build_hgb_pipeline,
    build_logistic_pipeline,
    make_split,
    validate_inputs,
    validate_probabilities,
    validate_submission_file,
)


def feature_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = synthetic_frames()
    rating_values = {
        column: [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0]
        for column in feature_module.RATING_COLUMNS
    }
    for column, values in rating_values.items():
        train[column] = values[: len(train)]
        test[column] = values[: len(test)]
    for column in feature_module.RATING_COLUMNS:
        train[column] = np.resize(rating_values[column], len(train))
        test[column] = np.resize(rating_values[column], len(test))
    train[feature_module.DEPARTURE_DELAY_COLUMN] = np.resize(
        [0.0, 1.0, np.nan, 3.0], len(train)
    )
    test[feature_module.DEPARTURE_DELAY_COLUMN] = np.resize(
        [0.0, 1.0, np.nan, 3.0], len(test)
    )
    train.loc[1, feature_module.RATING_COLUMNS[0]] = 5
    test.loc[1, feature_module.RATING_COLUMNS[0]] = 5
    return train, test


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "id": np.arange(12),
            "Age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            "Arrival Delay in Minutes": [1.0, np.nan, 3.0, 4.0] * 3,
            "Gender": ["A", "B"] * 6,
            "Customer Type": ["Loyal", "New"] * 6,
            "Type of Travel": ["Business", "Personal"] * 6,
            "Class": ["Eco", "Business", "Eco"] * 4,
            "satisfaction": [True, False] * 6,
        }
    )
    test = train.drop(columns="satisfaction").iloc[:4].copy()
    test["id"] += 100
    test.loc[0, "Gender"] = "Unseen"
    test.loc[1, "Arrival Delay in Minutes"] = np.nan
    return train, test


class TrainContractTests(unittest.TestCase):
    def test_source_overlap_exclusion_ignores_ids_and_targets_and_preserves_nan(
        self,
    ) -> None:
        train, test = feature_frames()
        source = train.drop(columns="id").copy()
        source["Age"] = source["Age"].astype("float64")
        source[feature_module.RATING_COLUMNS[0]] = source[
            feature_module.RATING_COLUMNS[0]
        ].astype("float64")
        source.loc[0, "satisfaction"] = not bool(source.loc[0, "satisfaction"])
        source_original = source.copy(deep=True)
        train_original = train.copy(deep=True)
        test_original = test.copy(deep=True)
        spec = validate_inputs(train, test)
        clean, labels, record = source_module.clean_source_rows(
            source, train, test, spec
        )
        self.assertEqual(record["clean_rows"], 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(clean), 0)
        self.assertTrue(pd.isna(train.loc[1, "Arrival Delay in Minutes"]))
        self.assertTrue(pd.isna(test.loc[1, "Arrival Delay in Minutes"]))
        pd.testing.assert_frame_equal(source, source_original)
        pd.testing.assert_frame_equal(train, train_original)
        pd.testing.assert_frame_equal(test, test_original)

    def test_source_conflicting_duplicate_labels_are_rejected(self) -> None:
        train, test = synthetic_frames()
        source = pd.concat(
            [train.drop(columns="id").iloc[[0]], train.drop(columns="id").iloc[[0]]],
            ignore_index=True,
        )
        source.loc[1, "satisfaction"] = not bool(source.loc[0, "satisfaction"])
        spec = validate_inputs(train, test)
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            source_module.clean_source_rows(source, train, test, spec)

    def test_augmented_fit_membership_is_competition_inner_fit_plus_source(
        self,
    ) -> None:
        result = source_module.augmented_fit_indices(np.array([2, 5, 7]), 10, 4)
        np.testing.assert_array_equal(result, [2, 5, 7, 10, 11, 12, 13])

    def test_source_gate_requires_both_seed_gains(self) -> None:
        baseline = {42: 0.957794208813, 2024: 0.958196779926}
        passed = source_module.source_gate(
            baseline, {42: baseline[42] + 0.0002001, 2024: baseline[2024] + 0.0003}
        )
        self.assertTrue(passed["qualifies"])
        failed = source_module.source_gate(
            baseline, {42: baseline[42] + 0.0002, 2024: baseline[2024] + 0.0001999}
        )
        self.assertFalse(failed["qualifies"])
        self.assertEqual(failed["decision"], "retain_current_blend")

    def test_engineered_features_have_exact_values_and_preserve_input(self) -> None:
        frame = pd.DataFrame(
            {column: [0, 5, 2] for column in feature_module.RATING_COLUMNS}
        )
        first_row = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0]
        for column, value in zip(feature_module.RATING_COLUMNS, first_row, strict=True):
            frame.loc[0, column] = value
        frame["Departure Delay in Minutes"] = [0.0, np.nan, 3.0]
        frame["Arrival Delay in Minutes"] = [np.nan, -1.0, 0.0]
        frame["id"] = [10, 11, 12]
        frame["satisfaction"] = [True, False, True]
        original = frame.copy(deep=True)

        engineered = feature_module.add_engineered_features(frame)

        self.assertEqual(
            list(engineered.columns[-5:]), list(feature_module.ENGINEERED_COLUMNS)
        )
        self.assertAlmostEqual(engineered.loc[0, "rating_mean"], 30 / 13)
        self.assertEqual(engineered.loc[0, "low_rating_count"], 7)
        self.assertEqual(engineered.loc[0, "departure_delayed"], 0.0)
        self.assertTrue(pd.isna(engineered.loc[0, "arrival_delayed"]))
        self.assertEqual(engineered.loc[0, "arrival_delay_missing"], 1)
        self.assertEqual(engineered.loc[1, "low_rating_count"], 0)
        self.assertTrue(pd.isna(engineered.loc[1, "departure_delayed"]))
        self.assertEqual(engineered.loc[1, "arrival_delayed"], 0.0)
        self.assertEqual(engineered.loc[2, "low_rating_count"], 13)
        self.assertEqual(engineered.loc[2, "departure_delayed"], 1.0)
        self.assertEqual(engineered.loc[2, "arrival_delayed"], 0.0)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(list(engineered["id"]), [10, 11, 12])
        self.assertEqual(list(engineered["satisfaction"]), [True, False, True])

    def test_engineered_features_reject_missing_and_colliding_columns(self) -> None:
        frame = pd.DataFrame({feature_module.RATING_COLUMNS[0]: [1]})
        with self.assertRaisesRegex(ValueError, "missing required"):
            feature_module.add_engineered_features(frame)

        train, _ = feature_frames()
        train["rating_mean"] = 0.0
        with self.assertRaisesRegex(ValueError, "collide"):
            feature_module.add_engineered_features(train)

    def test_engineered_spec_adds_only_numeric_predictors(self) -> None:
        train, test = feature_frames()
        engineered_train = feature_module.add_engineered_features(train)
        engineered_test = feature_module.add_engineered_features(test)
        spec = validate_inputs(engineered_train, engineered_test)

        self.assertEqual(spec.categorical_columns, CATEGORICAL_COLUMNS)
        self.assertEqual(spec.numeric_columns[-5:], feature_module.ENGINEERED_COLUMNS)
        self.assertEqual(
            len(spec.numeric_columns),
            len(validate_inputs(train, test).numeric_columns) + 5,
        )

    def test_engineered_gate_requires_the_minimum_gain_on_both_seeds(self) -> None:
        baseline = {42: 0.9577, 2024: 0.9580}
        self.assertTrue(
            feature_module.engineered_gate(baseline, {42: 0.95791, 2024: 0.95821})[
                "qualifies"
            ]
        )
        rejected = feature_module.engineered_gate(baseline, {42: 0.95791, 2024: 0.9581})
        self.assertFalse(rejected["qualifies"])
        self.assertEqual(rejected["decision"], "retain_unengineered_blend")

    def test_feature_source_discovery_failure_persists_failed_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            for name in ("train.csv", "test.csv"):
                (data_dir / name).write_text("synthetic placeholder\n")
            artifacts_dir = root / "artifacts"
            original = RuntimeError("source discovery failed")
            with mock.patch.object(
                feature_module, "_source_hashes", side_effect=original
            ):
                with self.assertRaises(RuntimeError) as raised:
                    feature_module.run_experiment(data_dir, artifacts_dir)
            self.assertIs(raised.exception, original)
            path = artifacts_dir / "feature_pass" / "compare_metadata.json"
            self.assertTrue(path.exists())
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["message"], "source discovery failed")

    def test_feature_reload_resolves_relative_paths_before_changing_cwd(self) -> None:
        paths = [
            Path(name)
            for name in (
                "data/test.csv",
                "models/cat.cbm",
                "models/lgb.txt",
                "models/spec.json",
                "models/categories.json",
            )
        ]

        def reload_stub(command: list[str], **kwargs: object) -> None:
            for supplied, expected in zip(command[3:8], paths, strict=True):
                self.assertEqual(Path(supplied), expected.resolve())
            np.save(command[-1], np.array([[0.1, 0.3], [0.2, 0.4]]))

        with mock.patch.object(
            feature_module.subprocess, "run", side_effect=reload_stub
        ):
            cat, light = feature_module._fresh_process_reload(*paths)
        np.testing.assert_array_equal(cat, [0.1, 0.2])
        np.testing.assert_array_equal(light, [0.3, 0.4])

    def test_fixed_blend_aligns_by_id_and_uses_half_weights(self) -> None:
        aligned = aligned_blend_predictions(
            [1, 2, 3],
            [3, 1, 2],
            np.array([0.3, 0.1, 0.2]),
            [2, 3, 1],
            np.array([0.5, 0.6, 0.4]),
        )
        np.testing.assert_allclose(aligned["catboost_probability"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(aligned["lightgbm_probability"], [0.4, 0.5, 0.6])
        np.testing.assert_allclose(aligned["blend_probability"], [0.25, 0.35, 0.45])

    def test_fixed_blend_rejects_duplicate_or_mismatched_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            aligned_blend_predictions(
                [1, 1], [1, 1], np.array([0.1, 0.2]), [1, 1], np.array([0.3, 0.4])
            )
        with self.assertRaisesRegex(ValueError, "do not match"):
            aligned_blend_predictions(
                [1, 2], [1, 3], np.array([0.1, 0.2]), [1, 2], np.array([0.3, 0.4])
            )

    def test_confirmation_gate_requires_both_fixed_criteria(self) -> None:
        self.assertTrue(confirmation_gate(0.95, 0.9501, 0.9502))
        self.assertFalse(confirmation_gate(0.95, 0.949, 0.9501))
        self.assertFalse(confirmation_gate(0.95, 0.9502, 0.9502))

    def test_weighted_blend_aligns_by_id_and_uses_fixed_75_25_weights(self) -> None:
        aligned = weighted_blend_module.weighted_blend_predictions(
            [1, 2, 3],
            [3, 1, 2],
            np.array([0.3, 0.1, 0.2]),
            [2, 3, 1],
            np.array([0.5, 0.6, 0.4]),
        )
        np.testing.assert_allclose(aligned["blend_probability"], [0.175, 0.275, 0.375])

    def test_weighted_blend_gate_requires_both_seed_gains(self) -> None:
        self.assertTrue(
            weighted_blend_module.blend_gate({42: 0.0002, 2024: 0.00021})["qualifies"]
        )
        rejected = weighted_blend_module.blend_gate({42: 0.0002, 2024: 0.0001999})
        self.assertFalse(rejected["qualifies"])
        self.assertEqual(rejected["decision"], "retain_current_blend")

    def test_longer_baseline_aligns_reordered_saved_catboost_ids(self) -> None:
        train, _ = synthetic_frames()
        frame = longer_catboost_module._validation_frame(
            train,
            np.array([0, 1, 2]),
            np.array([0.3, 0.1, 0.2]),
            [0, 1, 2],
            np.array([0.4, 0.5, 0.6]),
            catboost_ids=[2, 0, 1],
        )
        np.testing.assert_allclose(frame["catboost_probability"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(frame["blend_probability"], [0.25, 0.35, 0.45])

    def test_longer_catboost_changes_only_iteration_cap(self) -> None:
        model = longer_catboost_module.build_longer_catboost_classifier()
        settings = model.get_params()
        self.assertEqual(settings["iterations"], 3000)
        self.assertEqual(settings["depth"], 6)
        self.assertEqual(settings["learning_rate"], 0.06)
        self.assertEqual(settings["l2_leaf_reg"], 3.0)
        self.assertEqual(settings["loss_function"], "Logloss")
        self.assertEqual(settings["eval_metric"], "AUC")
        self.assertEqual(settings["random_seed"], 42)
        self.assertEqual(settings["thread_count"], 2)
        self.assertFalse(settings["allow_writing_files"])
        self.assertEqual(settings["early_stopping_rounds"], 100)

    def test_longer_blend_gate_requires_both_seed_gains(self) -> None:
        accepted = longer_catboost_module.blend_gate(
            {42: 0.957713923393243, 2024: 0.9580214315945633},
            {42: 0.957923923393243, 2024: 0.9582314315945633},
        )
        self.assertTrue(accepted["qualifies"])
        rejected = longer_catboost_module.blend_gate(
            {42: 0.957713923393243, 2024: 0.9580214315945633},
            {42: 0.957923923393243, 2024: 0.9582204315945633},
        )
        self.assertFalse(rejected["qualifies"])
        self.assertEqual(rejected["decision"], "retain_current_blend")

    def test_longer_refit_count_uses_median_one_based_best_counts(self) -> None:
        results = [
            {"best_iteration": 1170},
            {"best_iteration": 1199},
            {"best_iteration": 1420},
        ]
        self.assertEqual(longer_catboost_module.selected_iterations(results), 1200)

    def test_blend_cli_reports_new_schema_failure_without_stale_success(self) -> None:
        error = ValueError("new schema failure")
        with mock.patch.object(blend_module, "run_experiment", side_effect=error):
            with mock.patch.object(
                sys, "argv", ["blend_experiment.py", "--data-dir", "missing"]
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    with self.assertRaisesRegex(ValueError, str(error)):
                        blend_module.main()
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["selection"], None)
        self.assertEqual(summary["final"], None)
        self.assertEqual(
            summary["error"], {"type": "ValueError", "message": str(error)}
        )

    def _write_deeper_fixture(self, directory: Path) -> tuple[Path, Path]:
        base_train, base_test = synthetic_frames()
        train = pd.concat(
            [
                base_train.assign(id=base_train["id"] + offset)
                for offset in range(0, 120, 12)
            ],
            ignore_index=True,
        )
        test = base_test.assign(id=base_test["id"] + 10000)
        data_dir = directory / "data"
        artifacts_dir = directory / "artifacts"
        challengers_dir = artifacts_dir / "challengers"
        data_dir.mkdir()
        challengers_dir.mkdir(parents=True)
        train.to_csv(data_dir / "train.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)

        y = train["satisfaction"].astype(int).to_numpy()
        internal_splits = {}
        for split_name, seed, filename in (
            ("outer", 42, "outer_split.npz"),
            ("confirmation", 2024, "confirmation_split.npz"),
        ):
            fit_indices, holdout_indices = make_split(train, y, seed)
            np.savez(
                artifacts_dir / filename,
                fit_indices=fit_indices,
                holdout_indices=holdout_indices,
            )
            internal_fit, internal_validation = make_catboost_internal_split(
                fit_indices, y, seed=42
            )
            internal_splits[split_name] = {
                "fit_indices": internal_fit.tolist(),
                "validation_indices": internal_validation.tolist(),
            }

        prior_metadata = {
            "input_sha256": {
                "train.csv": compare_module.sha256_file(data_dir / "train.csv"),
                "test.csv": compare_module.sha256_file(data_dir / "test.csv"),
            },
            "internal_validation_splits": internal_splits,
            "validation_results": {
                "catboost_native": [
                    {"split_seed": 42, "roc_auc": 0.5, "log_loss": 0.7},
                    {"split_seed": 2024, "roc_auc": 0.5, "log_loss": 0.7},
                ]
            },
        }
        compare_module.save_json(
            challengers_dir / "compare_metadata.json", prior_metadata
        )
        return data_dir, artifacts_dir

    def _write_blend_reuse_fixture(
        self, directory: Path
    ) -> tuple[Path, Path, dict[str, str]]:
        train, test = synthetic_frames()
        data_dir = directory / "data"
        artifacts_dir = directory / "artifacts"
        challengers_dir = artifacts_dir / "challengers"
        lightgbm_dir = artifacts_dir / "lightgbm"
        data_dir.mkdir()
        challengers_dir.mkdir(parents=True)
        lightgbm_dir.mkdir()
        train.to_csv(data_dir / "train.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)
        spec = validate_inputs(train, test)
        input_hashes = {
            "train.csv": compare_module.sha256_file(data_dir / "train.csv"),
            "test.csv": compare_module.sha256_file(data_dir / "test.csv"),
        }
        catboost_config = dict(blend_module._settings()["catboost"])
        catboost_config.update(
            {
                "internal_validation_seed": 42,
                "internal_validation_fraction": 0.1,
            }
        )
        catboost_model_path = challengers_dir / "final_model_catboost.cbm"
        catboost_spec_path = challengers_dir / "feature_spec.json"
        catboost_model_path.write_text("synthetic model", encoding="utf-8")
        compare_module.save_json(
            catboost_spec_path,
            {
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
            },
        )
        compare_module.save_json(
            challengers_dir / "compare_metadata.json",
            {
                "input_sha256": input_hashes,
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "predeclared_configs": {"catboost_native": catboost_config},
                "selection": {"selected_candidate": "catboost_native"},
                "final": {
                    "model": "catboost_native",
                    "model_path": str(catboost_model_path),
                    "feature_spec_path": str(catboost_spec_path),
                },
            },
        )
        lightgbm_results = [
            {
                "candidate": "lightgbm_native",
                "split_seed": seed,
                "settings": dict(blend_module.PREDECLARED_CONFIG),
                "early_stopping": {"eval_metric": "auc"},
                "best_iteration": count,
            }
            for seed, count in ((42, 1), (2024, 2))
        ]
        compare_module.save_json(
            lightgbm_dir / "compare_metadata.json",
            {
                "status": "completed",
                "input_sha256_before": input_hashes,
                "input_sha256_after": input_hashes,
                "feature_columns": list(spec.feature_columns),
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "predeclared_config": dict(blend_module.PREDECLARED_CONFIG),
                "metric_policy": {
                    "configured_metric": "auc",
                    "early_stopping_metric": "auc",
                    "registered_metrics": ["auc"],
                },
                "validation_results": lightgbm_results,
            },
        )
        return data_dir, artifacts_dir, input_hashes

    def _run_blend_integrity_case(
        self,
        directory: Path,
        mutate: Callable[[], None],
        original_error: Exception | None = None,
    ) -> dict[str, object]:
        data_dir = directory / "data"
        artifacts_dir = directory / "artifacts"
        protected_path = artifacts_dir / "protected.txt"
        protected_path.write_text("original", encoding="utf-8")

        def fake_catboost(
            *args: object, **kwargs: object
        ) -> tuple[object, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
            outer_holdout = np.asarray(args[4]).shape[0]
            outer_fit = np.asarray(args[3], dtype=np.int64)
            inner_fit, inner_validation = outer_fit[:-2], outer_fit[-2:]
            mutate()
            if original_error is not None:
                raise original_error
            return (
                object(),
                np.full(outer_holdout, 0.5),
                {"roc_auc": 0.5, "best_iteration": 0},
                {"fit_indices": inner_fit, "validation_indices": inner_validation},
            )

        fake_lgb_model = mock.Mock(evals_result_={"inner_validation": {"auc": []}})

        with (
            mock.patch.object(blend_module, "_fit_catboost", side_effect=fake_catboost),
            mock.patch.object(
                blend_module,
                "make_catboost_internal_split",
                side_effect=lambda outer_fit, y, seed: (
                    np.asarray(outer_fit[:-2]),
                    np.asarray(outer_fit[-2:]),
                ),
            ),
            mock.patch.object(
                blend_module,
                "_fit_with_inner_validation",
                return_value=(
                    {},
                    0.0,
                    {"best_iteration": 1, "actual_iterations": 1, "resources": {}},
                ),
            ),
            mock.patch.object(
                blend_module, "build_lightgbm_classifier", return_value=fake_lgb_model
            ),
            mock.patch.object(
                blend_module,
                "lightgbm_validation_result",
                return_value=(
                    np.full(3, 0.5),
                    {
                        "best_iteration": 1,
                        "actual_iterations": 1,
                        "roc_auc": 0.5,
                        "log_loss": 0.7,
                    },
                ),
            ),
            mock.patch.object(
                blend_module,
                "_prior_blend_evidence",
                return_value=[
                    {"split_seed": 42, "blend_gain_over_catboost": 0.0002},
                    {"split_seed": 2024, "blend_gain_over_catboost": 0.0002},
                ],
            ),
            mock.patch.object(blend_module, "confirmation_gate", return_value=False),
        ):
            blend_module.run_experiment(data_dir, artifacts_dir)

        return json.loads(
            (artifacts_dir / "blend_50_50" / "compare_metadata.json").read_text(
                encoding="utf-8"
            )
        )

    def test_blend_reuse_guards_accept_valid_and_reject_changed_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir, input_hashes = self._write_blend_reuse_fixture(
                Path(directory)
            )
            spec = validate_inputs(
                pd.read_csv(data_dir / "train.csv"),
                pd.read_csv(data_dir / "test.csv"),
            )
            counts = blend_module._validate_reused_artifacts(
                artifacts_dir, input_hashes, spec
            )
            self.assertEqual(counts, [1, 2])

            cat_path = artifacts_dir / "challengers" / "compare_metadata.json"
            lgb_path = artifacts_dir / "lightgbm" / "compare_metadata.json"
            valid_cat = json.loads(cat_path.read_text(encoding="utf-8"))
            valid_lgb = json.loads(lgb_path.read_text(encoding="utf-8"))
            cases = (
                ("CatBoost metadata input_sha256", "cat_hash"),
                ("CatBoost metadata catboost_native configuration", "cat_config"),
                ("LightGBM validation results", "lgb_seeds"),
                ("LightGBM best_iteration", "lgb_count"),
            )
            for message, case in cases:
                with self.subTest(case=case):
                    cat_metadata = copy.deepcopy(valid_cat)
                    lgb_metadata = copy.deepcopy(valid_lgb)
                    if case == "cat_hash":
                        cat_metadata["input_sha256"]["train.csv"] = "changed"
                    elif case == "cat_config":
                        cat_metadata["predeclared_configs"]["catboost_native"][
                            "depth"
                        ] = 8
                    elif case == "lgb_seeds":
                        lgb_metadata["validation_results"][1]["split_seed"] = 42
                    else:
                        lgb_metadata["validation_results"][0]["best_iteration"] = 0
                    compare_module.save_json(cat_path, cat_metadata)
                    compare_module.save_json(lgb_path, lgb_metadata)
                    with self.assertRaisesRegex(ValueError, message):
                        blend_module._validate_reused_artifacts(
                            artifacts_dir, input_hashes, spec
                        )

    def test_blend_input_integrity_failure_is_persisted_and_raised(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir, _ = self._write_blend_reuse_fixture(
                Path(directory)
            )
            train_path = data_dir / "train.csv"

            def mutate() -> None:
                train_path.write_text(
                    train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
                )

            with self.assertRaisesRegex(RuntimeError, "input CSV fingerprint changed"):
                self._run_blend_integrity_case(Path(directory), mutate)
            metadata = json.loads(
                (artifacts_dir / "blend_50_50" / "compare_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["final"])
            self.assertIn("input CSV fingerprint changed", metadata["error"]["message"])

    def test_blend_protected_integrity_preserves_original_exception(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir, _ = self._write_blend_reuse_fixture(
                Path(directory)
            )
            protected_path = artifacts_dir / "protected.txt"

            def mutate() -> None:
                protected_path.write_text("changed", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                self._run_blend_integrity_case(
                    Path(directory), mutate, RuntimeError("synthetic fit failure")
                )
            metadata = json.loads(
                (artifacts_dir / "blend_50_50" / "compare_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metadata["error"],
                {"type": "RuntimeError", "message": "synthetic fit failure"},
            )
            self.assertIn(
                "pre-existing artifact changed", metadata["integrity_error"]["message"]
            )
            self.assertIsNone(metadata["selection"])
            self.assertIsNone(metadata["final"])

    def test_blend_reuse_accepts_equivalent_relative_artifact_paths(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir, hashes = self._write_blend_reuse_fixture(
                Path(directory)
            )
            spec = validate_inputs(
                pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")
            )
            path = artifacts_dir / "challengers" / "compare_metadata.json"
            metadata = json.loads(path.read_text())
            for field in ("model_path", "feature_spec_path"):
                metadata["final"][field] = os.path.relpath(metadata["final"][field])
            compare_module.save_json(path, metadata)
            self.assertEqual(
                blend_module._validate_reused_artifacts(
                    artifacts_dir.resolve(), hashes, spec
                ),
                [1, 2],
            )

    def test_blend_reuse_resolves_project_relative_paths_outside_cwd(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, artifacts_dir, hashes = self._write_blend_reuse_fixture(root)
            spec = validate_inputs(
                pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")
            )
            path = artifacts_dir / "challengers" / "compare_metadata.json"
            metadata = json.loads(path.read_text())
            metadata["final"]["model_path"] = (
                "artifacts/challengers/final_model_catboost.cbm"
            )
            metadata["final"]["feature_spec_path"] = (
                "artifacts/challengers/feature_spec.json"
            )
            compare_module.save_json(path, metadata)
            outside_project = root / "outside-project"
            outside_project.mkdir()
            with mock.patch.object(Path, "cwd", return_value=outside_project):
                self.assertEqual(
                    blend_module._validate_reused_artifacts(
                        artifacts_dir.resolve(), hashes, spec
                    ),
                    [1, 2],
                )

    def test_blend_repeated_metadata_write_failure_preserves_fit_error(self) -> None:
        with TemporaryDirectory() as directory:
            self._write_blend_reuse_fixture(Path(directory))
            original = RuntimeError("original fit failure")
            real_save = blend_module.save_json
            writes = 0

            def fail_after_setup(path: Path, value: dict[str, object]) -> None:
                nonlocal writes
                writes += 1
                if writes > 2:
                    raise OSError("metadata storage unavailable")
                real_save(path, value)

            with (
                mock.patch.object(
                    blend_module, "save_json", side_effect=fail_after_setup
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    self._run_blend_integrity_case(
                        Path(directory), lambda: None, original
                    )
            self.assertIs(raised.exception, original)

    def test_longer_timeout_installs_and_restores_signal_state_on_failure(self) -> None:
        original_alarm = object()
        original_term = object()
        original_timer = (12.0, 3.0)
        failure = RuntimeError("synthetic initialization failure")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts_dir = root / "artifacts"
            with (
                mock.patch.object(
                    longer_catboost_module.signal,
                    "getitimer",
                    return_value=original_timer,
                ),
                mock.patch.object(
                    longer_catboost_module.signal,
                    "getsignal",
                    side_effect=[original_alarm, original_term],
                ),
                mock.patch.object(
                    longer_catboost_module.signal, "signal"
                ) as signal_setter,
                mock.patch.object(
                    longer_catboost_module.signal, "setitimer"
                ) as timer_setter,
                mock.patch.object(
                    longer_catboost_module, "_dependencies", side_effect=failure
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, str(failure)):
                    longer_catboost_module.run_experiment(
                        root / "missing-data", artifacts_dir
                    )

        signal_calls = signal_setter.call_args_list
        self.assertEqual(signal_calls[0][0][0], longer_catboost_module.signal.SIGALRM)
        self.assertEqual(signal_calls[1][0][0], longer_catboost_module.signal.SIGTERM)
        self.assertIn(
            mock.call(longer_catboost_module.signal.SIGALRM, original_alarm),
            signal_calls,
        )
        self.assertIn(
            mock.call(longer_catboost_module.signal.SIGTERM, original_term),
            signal_calls,
        )
        timer_setter.assert_any_call(
            longer_catboost_module.signal.ITIMER_REAL,
            longer_catboost_module.EXPERIMENT_TIMEOUT_SECONDS,
        )
        timer_setter.assert_any_call(
            longer_catboost_module.signal.ITIMER_REAL, *original_timer
        )

    def test_longer_final_metadata_records_postwrite_output_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            artifacts_dir = root / "artifacts"
            output_dir = artifacts_dir / "longer_catboost"
            data_dir.mkdir()
            artifacts_dir.mkdir()
            output_dir.mkdir()
            (data_dir / "train.csv").write_text("train\n", encoding="utf-8")
            (data_dir / "test.csv").write_text("test\n", encoding="utf-8")
            (artifacts_dir / "protected.txt").write_text("protected", encoding="utf-8")
            (output_dir / "validation.csv").write_text("synthetic\n", encoding="utf-8")
            metadata_path = output_dir / "compare_metadata.json"
            metadata, caught = longer_catboost_module._finalize(
                {"status": "completed", "selection": None, "final": None},
                metadata_path,
                output_dir,
                artifacts_dir,
                data_dir,
                {
                    "train.csv": compare_module.sha256_file(data_dir / "train.csv"),
                    "test.csv": compare_module.sha256_file(data_dir / "test.csv"),
                },
                longer_catboost_module._artifact_snapshot(artifacts_dir, output_dir),
                None,
                time.perf_counter(),
            )

            self.assertIsNone(caught)
            persisted = json.loads(metadata_path.read_text())
            measured = sum(
                path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
            )
            self.assertEqual(metadata["output_bytes"], measured)
            self.assertEqual(persisted["output_bytes"], measured)
            self.assertIn(
                "protected_baseline_sha256.json",
                {path.name for path in output_dir.iterdir()},
            )

    def test_longer_budget_rejects_postwrite_size(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            artifacts_dir = root / "artifacts"
            output_dir = artifacts_dir / "longer_catboost"
            data_dir.mkdir()
            artifacts_dir.mkdir()
            output_dir.mkdir()
            (data_dir / "train.csv").write_text("train\n", encoding="utf-8")
            (data_dir / "test.csv").write_text("test\n", encoding="utf-8")
            (output_dir / "validation.csv").write_text("synthetic\n", encoding="utf-8")
            input_hashes = {
                "train.csv": compare_module.sha256_file(data_dir / "train.csv"),
                "test.csv": compare_module.sha256_file(data_dir / "test.csv"),
            }
            with mock.patch.object(longer_catboost_module, "OUTPUT_BUDGET_BYTES", 1):
                metadata, caught = longer_catboost_module._finalize(
                    {"status": "completed", "selection": None, "final": None},
                    output_dir / "compare_metadata.json",
                    output_dir,
                    artifacts_dir,
                    data_dir,
                    input_hashes,
                    longer_catboost_module._artifact_snapshot(
                        artifacts_dir, output_dir
                    ),
                    None,
                    time.perf_counter(),
                )

            self.assertIsNotNone(caught)
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("budget exceeded", metadata["error"]["message"])

    def test_catboost_default_and_requested_depths(self) -> None:
        self.assertEqual(
            build_catboost_classifier(
                iterations=1, early_stopping_rounds=None
            ).get_params()["depth"],
            6,
        )
        self.assertEqual(
            build_catboost_classifier(
                iterations=1,
                early_stopping_rounds=None,
                depth=8,
            ).get_params()["depth"],
            8,
        )

    def test_lightgbm_native_frame_aligns_categories_and_preserves_nan(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        features = train[list(spec.feature_columns)]
        levels = fit_category_levels(features, spec)
        train_frame = prepare_lightgbm_frame(features, spec, levels)
        test_frame = prepare_lightgbm_frame(test, spec, levels)

        self.assertEqual(list(train_frame.columns), list(spec.feature_columns))
        self.assertNotIn("id", train_frame.columns)
        self.assertNotIn("satisfaction", train_frame.columns)
        self.assertIsInstance(train_frame["Gender"].dtype, pd.CategoricalDtype)
        self.assertTrue(pd.isna(test_frame.loc[0, "Gender"]))
        self.assertTrue(pd.isna(test_frame.loc[1, "Arrival Delay in Minutes"]))

    def test_lightgbm_native_fit_reload_and_class_one_probabilities(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        features = train[list(spec.feature_columns)]
        y = train["satisfaction"].astype(int).to_numpy()
        model = build_lightgbm_classifier(
            n_estimators=20,
            min_child_samples=2,
            num_leaves=7,
        )
        levels, _, details = _fit_with_inner_validation(
            model,
            features,
            y,
            np.arange(10, dtype=np.int64),
            np.arange(10, 12, dtype=np.int64),
            spec,
            callbacks=_fit_callbacks(),
        )
        predictions = model.predict_proba(prepare_lightgbm_frame(test, spec, levels))[
            :, 1
        ]
        validate_probabilities(predictions, len(test))
        self.assertGreaterEqual(details["best_iteration"], 1)

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.txt"
            spec_path = directory_path / "feature_spec.json"
            model.booster_.save_model(model_path)
            compare_module.save_json(
                spec_path,
                {
                    "categorical_columns": list(spec.categorical_columns),
                    "category_levels": levels,
                },
            )
            reloaded = _fresh_reload_predictions(
                model_path,
                spec_path,
                test[list(spec.feature_columns)],
            )

        np.testing.assert_allclose(predictions, reloaded, rtol=0, atol=1e-12)

    def test_lightgbm_registers_only_auc_for_binary_training(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        features = train[list(spec.feature_columns)]
        y = train["satisfaction"].astype(int).to_numpy()
        model = build_lightgbm_classifier(
            n_estimators=20,
            min_child_samples=2,
            num_leaves=7,
        )

        _fit_with_inner_validation(
            model,
            features,
            y,
            np.arange(10, dtype=np.int64),
            np.arange(10, 12, dtype=np.int64),
            spec,
            callbacks=_fit_callbacks(),
        )

        self.assertEqual(model.get_params()["metric"], "auc")
        self.assertEqual(
            set(model.evals_result_["inner_validation"]),
            {"auc"},
        )

    def test_lightgbm_initialization_failure_persists_fresh_error(self) -> None:
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            data_dir = directory_path / "missing-data"
            artifacts_dir = directory_path / "artifacts"

            with self.assertRaisesRegex(FileNotFoundError, "train.csv"):
                run_experiment(data_dir, artifacts_dir)

            metadata = json.loads(
                (artifacts_dir / "lightgbm" / "compare_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["type"], "FileNotFoundError")
            self.assertIn("train.csv", metadata["error"]["message"])

    def test_lightgbm_cli_failure_summary_uses_caught_error(self) -> None:
        error = RuntimeError("fresh initialization failure")
        with TemporaryDirectory() as directory:
            artifacts_dir = Path(directory) / "artifacts"
            stale_path = artifacts_dir / "lightgbm" / "compare_metadata.json"
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "selection": {"decision": "stale"},
                        "final": {"submission_path": "stale.csv"},
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(lightgbm_module, "run_experiment", side_effect=error),
                mock.patch.object(
                    sys,
                    "argv",
                    ["lightgbm_experiment.py", "--artifacts-dir", str(artifacts_dir)],
                ),
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    with self.assertRaisesRegex(RuntimeError, str(error)):
                        lightgbm_module.main()

        summary = json.loads(output.getvalue())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["error"], {"type": "RuntimeError", "message": str(error)}
        )

    def test_lightgbm_integrity_failure_is_raised_after_metadata_persisted(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir = self._write_deeper_fixture(Path(directory))

            def fake_fit(
                *args: object, **kwargs: object
            ) -> tuple[dict[str, list[str]], float, dict[str, object]]:
                return (
                    {},
                    0.0,
                    {
                        "best_iteration": 1,
                        "actual_iterations": 1,
                        "resources": {},
                    },
                )

            validation_count = 0

            def fake_validation(
                *args: object, **kwargs: object
            ) -> tuple[np.ndarray, dict[str, object]]:
                nonlocal validation_count
                validation_count += 1
                holdout_indices = np.asarray(args[3], dtype=np.int64)
                return np.full(len(holdout_indices), 0.5), {
                    "split_seed": 42 if validation_count == 1 else 2024,
                    "roc_auc": 0.5,
                    "log_loss": 0.7,
                    "best_iteration": 1,
                    "actual_iterations": 1,
                }

            with (
                mock.patch.object(
                    lightgbm_module, "_fit_with_inner_validation", side_effect=fake_fit
                ),
                mock.patch.object(
                    lightgbm_module,
                    "_validation_result",
                    side_effect=fake_validation,
                ),
                mock.patch.object(
                    lightgbm_module,
                    "_protected_hashes",
                    return_value={"changed.txt": "changed"},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
                    run_experiment(data_dir, artifacts_dir)

            metadata = json.loads(
                (artifacts_dir / "lightgbm" / "compare_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIn("fingerprint changed", metadata["error"]["message"])

    def test_lightgbm_protects_every_existing_artifact_file(self) -> None:
        with TemporaryDirectory() as directory:
            artifacts_dir = Path(directory) / "artifacts"
            output_dir = artifacts_dir / "lightgbm"
            for relative_path in (
                Path("root.txt"),
                Path("challengers") / "model.txt",
                Path("deeper_catboost") / "metadata.json",
                Path("lightgbm_dual_metric_review") / "compare_metadata.json",
            ):
                path = artifacts_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path.as_posix(), encoding="utf-8")
            output_dir.mkdir(parents=True)
            (output_dir / "new-run.json").write_text("new", encoding="utf-8")

            before = _artifact_snapshot(artifacts_dir, output_dir)
            after = lightgbm_module._protected_hashes(
                artifacts_dir, set(before), output_dir
            )

            self.assertEqual(after, before)
            self.assertIn("deeper_catboost/metadata.json", before)
            self.assertIn("lightgbm_dual_metric_review/compare_metadata.json", before)
            self.assertNotIn("lightgbm/new-run.json", before)

    def test_lightgbm_smoke_uses_disjoint_inner_subsets(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir = self._write_deeper_fixture(Path(directory))
            fit_calls: list[tuple[np.ndarray, np.ndarray]] = []

            def fake_fit(
                *args: object, **kwargs: object
            ) -> tuple[dict[str, list[str]], float, dict[str, object]]:
                fit_indices = np.asarray(args[3], dtype=np.int64)
                validation_indices = np.asarray(args[4], dtype=np.int64)
                fit_calls.append((fit_indices.copy(), validation_indices.copy()))
                return (
                    {},
                    0.0,
                    {
                        "best_iteration": 1,
                        "actual_iterations": 1,
                        "resources": {},
                    },
                )

            validation_count = 0

            def fake_validation(
                *args: object, **kwargs: object
            ) -> tuple[np.ndarray, dict[str, object]]:
                nonlocal validation_count
                validation_count += 1
                holdout_indices = np.asarray(args[3], dtype=np.int64)
                return np.full(len(holdout_indices), 0.5), {
                    "split_seed": 42 if validation_count == 1 else 2024,
                    "roc_auc": 0.5,
                    "log_loss": 0.7,
                    "best_iteration": 1,
                    "actual_iterations": 1,
                }

            with (
                mock.patch.object(
                    lightgbm_module, "_fit_with_inner_validation", side_effect=fake_fit
                ),
                mock.patch.object(
                    lightgbm_module,
                    "_validation_result",
                    side_effect=fake_validation,
                ),
            ):
                metadata = run_experiment(data_dir, artifacts_dir)

            smoke_fit, smoke_validation = fit_calls[0]
            recorded = metadata["recorded_splits"]["outer"]
            inner_fit = set(recorded["inner_fit_indices"])
            inner_validation = set(recorded["inner_validation_indices"])
            self.assertTrue(set(smoke_fit).issubset(inner_fit))
            self.assertTrue(set(smoke_validation).issubset(inner_validation))
            self.assertFalse(set(smoke_fit) & set(smoke_validation))

    def test_lightgbm_refit_iterations_use_one_based_best_iteration(self) -> None:
        results = [{"best_iteration": 4}, {"best_iteration": 8}]
        self.assertEqual(_selected_iterations(results), 6)
        self.assertEqual(
            build_lightgbm_classifier(
                n_estimators=_selected_iterations(results)
            ).get_params()["n_estimators"],
            6,
        )

    def test_lightgbm_rejection_and_terminal_summary_are_branch_safe(self) -> None:
        validation = [
            {
                "split_seed": 42,
                "roc_auc": 0.9574,
                "log_loss": 0.23,
                "best_iteration": 100,
                "actual_iterations": 100,
            },
            {
                "split_seed": 2024,
                "roc_auc": 0.9578,
                "log_loss": 0.23,
                "best_iteration": 101,
                "actual_iterations": 101,
            },
        ]
        baseline = [
            {"split_seed": 42, "roc_auc": 0.9573498661, "log_loss": 0.23},
            {"split_seed": 2024, "roc_auc": 0.9577455323, "log_loss": 0.23},
        ]
        selection = _selection(validation, baseline)
        self.assertFalse(selection["qualifies"])
        self.assertEqual(selection["decision"], "retain_catboost")
        summary = _terminal_summary(
            {"status": "failed", "selection": None, "final": None}
        )
        self.assertEqual(
            set(summary), {"status", "selection", "final", "error", "total_seconds"}
        )

    def test_catboost_native_frame_handles_unknown_category_and_numeric_nan(
        self,
    ) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)

        train_frame = prepare_catboost_frame(train, spec)
        test_frame = prepare_catboost_frame(test, spec)

        self.assertEqual(list(train_frame.columns), list(spec.feature_columns))
        self.assertNotIn("id", train_frame.columns)
        self.assertNotIn("satisfaction", train_frame.columns)
        self.assertTrue(pd.isna(train_frame.loc[1, "Arrival Delay in Minutes"]))
        self.assertEqual(test_frame.loc[0, "Gender"], "Unseen")

    def test_catboost_smoke_fit_handles_missing_numeric_and_reloads_cleanly(
        self,
    ) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        train_frame = prepare_catboost_frame(train, spec)
        test_frame = prepare_catboost_frame(test, spec)
        model = build_catboost_classifier(
            iterations=20,
            early_stopping_rounds=None,
        )
        model.fit(
            train_frame,
            train["satisfaction"].astype(int).to_numpy(),
            cat_features=list(spec.categorical_columns),
            verbose=False,
        )
        expected = model.predict_proba(test_frame)[:, 1]
        validate_probabilities(expected, len(test))

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.cbm"
            features_path = directory_path / "features.csv"
            output_path = directory_path / "probabilities.npy"
            model.save_model(model_path)
            test_frame.to_csv(features_path, index=False)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import numpy as np, pandas as pd, sys; "
                        "from catboost import CatBoostClassifier; "
                        "model = CatBoostClassifier(); "
                        "model.load_model(sys.argv[1]); "
                        "features = pd.read_csv(sys.argv[2]); "
                        "np.save(sys.argv[3], model.predict_proba(features)[:, 1])"
                    ),
                    str(model_path),
                    str(features_path),
                    str(output_path),
                ],
                check=True,
                cwd=directory,
                capture_output=True,
                text=True,
            )
            reloaded = np.load(output_path)

        np.testing.assert_array_equal(expected, reloaded)

    def test_catboost_internal_validation_is_disjoint_from_outer_holdout(self) -> None:
        train_parts = []
        for offset in range(0, 40, 12):
            part, _ = synthetic_frames()
            part["id"] += offset
            train_parts.append(part)
        train = pd.concat(train_parts, ignore_index=True).iloc[:40]
        y = train["satisfaction"].astype(int).to_numpy()
        outer_fit = np.arange(32, dtype=np.int64)
        outer_holdout = np.arange(32, 40, dtype=np.int64)
        internal_fit, internal_validation = make_catboost_internal_split(
            outer_fit,
            y,
            seed=42,
        )

        self.assertEqual(set(internal_fit) | set(internal_validation), set(outer_fit))
        self.assertFalse(set(internal_fit) & set(internal_validation))
        self.assertFalse(set(internal_fit) & set(outer_holdout))
        self.assertFalse(set(internal_validation) & set(outer_holdout))

    def test_depth8_propagates_through_validation_and_final_refit(self) -> None:
        train_parts = []
        for offset in range(0, 40, 12):
            part, _ = synthetic_frames()
            part["id"] += offset
            train_parts.append(part)
        train = pd.concat(train_parts, ignore_index=True).iloc[:40]
        test = train.drop(columns="satisfaction").iloc[:4].copy()
        test["id"] += 1000
        spec = validate_inputs(train, test)
        X = train[list(spec.feature_columns)]
        y = train["satisfaction"].astype(int).to_numpy()

        model, _, validation, _ = _fit_catboost(
            spec,
            X,
            y,
            np.arange(32, dtype=np.int64),
            X.iloc[32:],
            y[32:],
            split_seed=42,
            depth=8,
            iterations=8,
            candidate="catboost_depth8",
        )
        self.assertEqual(model.get_params()["depth"], 8)
        self.assertEqual(validation["settings"]["depth"], 8)

        with TemporaryDirectory() as directory:
            final = _refit_winner(
                "catboost_depth8",
                spec,
                X,
                y,
                test[list(spec.feature_columns)],
                test["id"],
                {"catboost_depth8": [{"best_iteration": 1}]},
                Path(directory),
                depth=8,
                output_stem="catboost_depth8",
            )
            reloaded = CatBoostClassifier()
            reloaded.load_model(final["model_path"])
            self.assertEqual(reloaded.get_params()["depth"], 8)
            self.assertEqual(final["settings"]["depth"], 8)

    def test_deeper_fit_failure_persists_error_and_protected_hashes(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir, artifacts_dir = self._write_deeper_fixture(Path(directory))
            with mock.patch.object(
                compare_module,
                "_fit_catboost",
                side_effect=ValueError("synthetic fit failure"),
            ):
                with self.assertRaisesRegex(ValueError, "synthetic fit failure"):
                    run_deeper_catboost(data_dir, artifacts_dir)

            deeper_dir = artifacts_dir / "deeper_catboost"
            metadata = json.loads(
                (deeper_dir / "compare_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(
                metadata["error"],
                {"type": "ValueError", "message": "synthetic fit failure"},
            )
            self.assertEqual(
                metadata["input_sha256_before"], metadata["input_sha256_after"]
            )
            self.assertEqual(
                metadata["protected_baseline_sha256_before"],
                metadata["protected_baseline_sha256_after"],
            )
            self.assertTrue(metadata["protected_baseline_unchanged"])
            self.assertEqual(
                json.loads(
                    (deeper_dir / "protected_baseline_sha256.json").read_text(
                        encoding="utf-8"
                    )
                ),
                metadata["protected_baseline_sha256_after"],
            )

    def test_deeper_cli_summary_for_completed_rejection(self) -> None:
        result = {
            "status": "completed",
            "selection": {
                "qualifies": False,
                "decision": "retain_depth6_candidate",
            },
            "final": None,
            "total_seconds": 0.1,
        }
        with mock.patch.object(
            compare_module, "run_deeper_catboost", return_value=result
        ):
            with mock.patch.object(sys, "argv", ["compare.py", "--deeper-catboost"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    compare_module.main()

        self.assertEqual(json.loads(output.getvalue()), result)

    def test_deeper_cli_summary_for_timeout_without_selection(self) -> None:
        result = {
            "status": "timeout",
            "selection": None,
            "final": None,
            "total_seconds": 1.0,
        }
        with mock.patch.object(
            compare_module, "run_deeper_catboost", return_value=result
        ):
            with mock.patch.object(sys, "argv", ["compare.py", "--deeper-catboost"]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    compare_module.main()

        self.assertEqual(json.loads(output.getvalue()), result)

    def test_rejects_overlapping_train_and_test_ids(self) -> None:
        train, test = synthetic_frames()
        test.loc[0, "id"] = 2

        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_inputs(train, test)

    def test_validates_feature_contract_and_excludes_id_and_target(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)

        self.assertNotIn("id", spec.feature_columns)
        self.assertNotIn("satisfaction", spec.feature_columns)
        self.assertEqual(spec.categorical_columns, CATEGORICAL_COLUMNS)
        self.assertEqual(spec.numeric_columns, ("Age", "Arrival Delay in Minutes"))

    def test_rejects_nonbinary_target(self) -> None:
        train, test = synthetic_frames()
        train["satisfaction"] = train["satisfaction"].astype(object)
        train.loc[0, "satisfaction"] = 2

        with self.assertRaisesRegex(ValueError, "binary"):
            validate_inputs(train, test)

    def test_hgb_transformer_preserves_nan_and_encodes_unknown_as_nan(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        transformer = build_hgb_pipeline(spec).named_steps["preprocess"]
        transformed = transformer.fit_transform(train[list(spec.feature_columns)])

        self.assertTrue(np.isnan(transformed[1, 1]))
        self.assertTrue(
            np.isnan(transformer.transform(test[list(spec.feature_columns)])[0, 2])
        )
        self.assertEqual(transformed.shape, (len(train), 6))

    def test_hgb_joblib_reload_in_separate_interpreter_preserves_probabilities(
        self,
    ) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        features = list(spec.feature_columns)
        model = build_hgb_pipeline(spec, max_iter=20, early_stopping=False)
        model.fit(train[features], train["satisfaction"].astype(int).to_numpy())
        expected = model.predict_proba(test[features])[:, 1]

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.joblib"
            input_path = directory_path / "features.csv"
            output_path = directory_path / "probabilities.npy"
            joblib.dump(model, model_path)
            test[features].to_csv(input_path, index=False)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import joblib, numpy as np, pandas as pd, sys; "
                        "model = joblib.load(sys.argv[1]); "
                        "features = pd.read_csv(sys.argv[2]); "
                        "np.save(sys.argv[3], model.predict_proba(features)[:, 1])"
                    ),
                    str(model_path),
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                cwd=directory,
                capture_output=True,
                text=True,
            )
            reloaded = np.load(output_path)

        np.testing.assert_array_equal(expected, reloaded)

    def test_both_models_fit_and_return_class_one_probabilities(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        X = train[list(spec.feature_columns)]
        y = train["satisfaction"].astype(int).to_numpy()
        for model in (
            build_logistic_pipeline(spec),
            build_hgb_pipeline(spec, max_iter=20),
            build_hgb_pipeline(spec, max_iter=20, early_stopping=False),
        ):
            model.fit(X, y)
            predictions = model.predict_proba(test[list(spec.feature_columns)])[:, 1]
            validate_probabilities(predictions, len(test))

    def test_split_is_reproducible_and_stratified(self) -> None:
        train, _ = synthetic_frames()
        y = train["satisfaction"].astype(int).to_numpy()
        fit_a, holdout_a = make_split(train, y, 42)
        fit_b, holdout_b = make_split(train, y, 42)

        self.assertTrue(np.array_equal(fit_a, fit_b))
        self.assertTrue(np.array_equal(holdout_a, holdout_b))
        self.assertEqual(set(y[fit_a]), {0, 1})
        self.assertEqual(set(y[holdout_a]), {0, 1})

    def test_submission_validation_rejects_bad_schema_duplicate_and_bad_order(
        self,
    ) -> None:
        test_ids = pd.Series([10, 11, 12])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            pd.DataFrame({"id": [10, 11, 12], "wrong": [0.1, 0.2, 0.3]}).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "exactly id,satisfaction"):
                validate_submission_file(path, test_ids)

            pd.DataFrame({"id": [10, 12, 11], "satisfaction": [0.1, 0.2, 0.3]}).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                validate_submission_file(path, test_ids)

            pd.DataFrame({"id": [10, 10, 12], "satisfaction": [0.1, 0.2, 0.3]}).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                validate_submission_file(path, test_ids)

            pd.DataFrame({"id": [10, 11, 12], "satisfaction": [0.1, 0.2, 0.3]}).to_csv(
                path, index=False
            )
            result = validate_submission_file(path, test_ids)
            self.assertEqual(list(result.columns), ["id", "satisfaction"])

    def test_submission_validation_rejects_nan_inf_and_out_of_range_probabilities(
        self,
    ) -> None:
        test_ids = pd.Series([10, 11, 12])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            for probability in (np.nan, np.inf, -0.1, 1.1):
                pd.DataFrame(
                    {
                        "id": [10, 11, 12],
                        "satisfaction": [0.1, probability, 0.3],
                    }
                ).to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_submission_file(path, test_ids)

    def test_slower_learning_changes_only_learning_rate_cap_and_patience(self) -> None:
        import slower_learning as slower

        cat = slower.build_slow_catboost().get_params()
        self.assertEqual(cat["iterations"], 6000)
        self.assertAlmostEqual(cat["learning_rate"], 0.03)
        self.assertEqual(cat["depth"], 6)
        self.assertEqual(cat["l2_leaf_reg"], 3)
        self.assertEqual(cat["random_seed"], 42)

        light = slower.build_slow_lightgbm().get_params()
        self.assertEqual(light["n_estimators"], 3000)
        self.assertAlmostEqual(light["learning_rate"], 0.02)
        self.assertEqual(light["num_leaves"], 31)
        self.assertEqual(light["random_state"], 42)

    def test_slower_learning_gate_requires_both_seeds_and_picks_worst_seed(
        self,
    ) -> None:
        import slower_learning as slower

        baseline = {42: 0.90, 2024: 0.90}
        per_seed = {
            42: {
                "original_cat_slow_light": 0.9001,
                "slow_cat_original_light": 0.9020,
                "slow_cat_slow_light": 0.9030,
            },
            2024: {
                "original_cat_slow_light": 0.9025,
                "slow_cat_original_light": 0.9025,
                "slow_cat_slow_light": 0.9030,
            },
        }
        result = slower.select_candidate(per_seed, baseline)
        self.assertEqual(result["decision"], "confirm_fresh_split")
        self.assertEqual(result["selected_candidate"], "slow_cat_slow_light")

        for scores in per_seed.values():
            for name in scores:
                scores[name] = 0.9001
        per_seed[42]["original_cat_slow_light"] = 0.91
        rejected = slower.select_candidate(per_seed, baseline)
        self.assertEqual(rejected["decision"], "retain_current_blend")
        self.assertIsNone(rejected["selected_candidate"])

    def test_categorical_ratings_are_canonical_and_input_is_unchanged(self) -> None:
        train, test = feature_frames()
        train_original = train.copy(deep=True)
        test_original = test.copy(deep=True)
        spec = validate_inputs(train, test)

        prepared_train = categorical_module.prepare_rating_categorical_frame(
            train, spec
        )
        prepared_test = categorical_module.prepare_rating_categorical_frame(test, spec)

        self.assertEqual(list(prepared_train.columns), list(spec.feature_columns))
        self.assertEqual(
            categorical_module.categorical_columns(spec),
            tuple(spec.categorical_columns) + feature_module.RATING_COLUMNS,
        )
        self.assertEqual(prepared_train.loc[0, feature_module.RATING_COLUMNS[0]], "0")
        self.assertEqual(prepared_train.loc[1, feature_module.RATING_COLUMNS[0]], "5")
        self.assertEqual(prepared_test.loc[0, "Gender"], "Unseen")
        self.assertTrue(pd.isna(prepared_train.loc[1, "Arrival Delay in Minutes"]))
        pd.testing.assert_frame_equal(train, train_original)
        pd.testing.assert_frame_equal(test, test_original)

    def test_categorical_ratings_reject_missing_fractional_and_out_of_domain_values(
        self,
    ) -> None:
        train, test = feature_frames()
        spec = validate_inputs(train, test)

        missing = train.copy()
        missing.loc[0, feature_module.RATING_COLUMNS[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "missing rating"):
            categorical_module.prepare_rating_categorical_frame(missing, spec)

        fractional = train.copy()
        fractional[feature_module.RATING_COLUMNS[0]] = fractional[
            feature_module.RATING_COLUMNS[0]
        ].astype(float)
        fractional.loc[0, feature_module.RATING_COLUMNS[0]] = 1.5
        with self.assertRaisesRegex(ValueError, "integral"):
            categorical_module.prepare_rating_categorical_frame(fractional, spec)

        unexpected = train.copy()
        unexpected.loc[0, feature_module.RATING_COLUMNS[0]] = 6
        with self.assertRaisesRegex(ValueError, "allowed"):
            categorical_module.prepare_rating_categorical_frame(unexpected, spec)

    def test_categorical_ratings_preserve_slow_catboost_settings_and_gate(self) -> None:
        model = categorical_module.build_categorical_ratings_catboost()
        expected = categorical_module.build_slow_catboost().get_params()
        self.assertEqual(model.get_params(), expected)
        self.assertEqual(
            len(
                categorical_module.categorical_columns(
                    validate_inputs(*feature_frames())
                )
            ),
            17,
        )

        baseline = {42: 0.9577, 2024: 0.9580}
        self.assertTrue(
            categorical_module.categorical_gate(baseline, {42: 0.95791, 2024: 0.9582})[
                "qualifies"
            ]
        )
        rejected = categorical_module.categorical_gate(
            baseline, {42: 0.95791, 2024: 0.95819}
        )
        self.assertFalse(rejected["qualifies"])
        self.assertEqual(rejected["decision"], "retain_current_blend")

    def test_categorical_second_split_mode_only_fits_seed2024_and_uses_85_minutes(
        self,
    ) -> None:
        configuration = categorical_module._run_configuration(True)

        self.assertEqual(configuration["seeds"], (2024,))
        self.assertNotIn(42, configuration["seeds"])
        self.assertEqual(configuration["output_name"], "categorical_ratings_seed2024")
        self.assertEqual(configuration["internal_timeout_seconds"], 85 * 60)

    def test_categorical_ratings_source_failure_persists_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "train.csv").write_text("placeholder\n")
            (data_dir / "test.csv").write_text("placeholder\n")
            artifacts_dir = root / "artifacts"
            original = RuntimeError("source discovery failed")
            with mock.patch.object(
                categorical_module, "_source_hashes", side_effect=original
            ):
                with self.assertRaisesRegex(RuntimeError, "source discovery failed"):
                    categorical_module.run(data_dir, artifacts_dir)

            metadata = json.loads(
                (
                    artifacts_dir / "categorical_ratings" / "compare_metadata.json"
                ).read_text()
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["message"], "source discovery failed")


if __name__ == "__main__":
    unittest.main()
