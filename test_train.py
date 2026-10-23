import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import compare as compare_module
from compare import (
    _fit_catboost,
    _refit_winner,
    build_catboost_classifier,
    make_catboost_internal_split,
    prepare_catboost_frame,
    run_deeper_catboost,
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


if __name__ == "__main__":
    unittest.main()
