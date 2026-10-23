import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

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
