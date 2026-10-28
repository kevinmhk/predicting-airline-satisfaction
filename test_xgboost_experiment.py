import unittest

import numpy as np
import pandas as pd

from lightgbm_experiment import fit_category_levels, prepare_lightgbm_frame
from train import InputSpec
from xgboost_experiment import (
    MINIMUM_AUC_GAIN,
    PREDECLARED_CONFIG,
    build_xgboost_classifier,
    fixed_xgboost_blend_predictions,
    xgboost_gate,
)


class XGBoostExperimentTests(unittest.TestCase):
    def test_cpu_distribution_metadata_is_available(self) -> None:
        from importlib.metadata import version

        from xgboost_experiment import _package_versions

        self.assertEqual(_package_versions()["xgboost-cpu"], version("xgboost-cpu"))

    def test_cpu_fixed_configuration_and_two_seed_gate(self) -> None:
        params = build_xgboost_classifier().get_params()
        for key, value in PREDECLARED_CONFIG.items():
            self.assertEqual(params[key], value)
        self.assertEqual(params["device"], "cpu")
        self.assertTrue(params["enable_categorical"])
        self.assertTrue(
            xgboost_gate({42: MINIMUM_AUC_GAIN, 2024: MINIMUM_AUC_GAIN})["qualifies"]
        )
        self.assertFalse(
            xgboost_gate({42: MINIMUM_AUC_GAIN, 2024: MINIMUM_AUC_GAIN - 1e-7})[
                "qualifies"
            ]
        )

    def test_fixed_blend_aligns_unique_ids(self) -> None:
        result = fixed_xgboost_blend_predictions(
            np.array([10, 11, 12]),
            np.array([12, 10, 11]),
            np.array([0.6, 0.2, 0.4]),
            np.array([11, 12, 10]),
            np.array([0.5, 0.7, 0.1]),
            np.array([10, 12, 11]),
            np.array([0.3, 0.9, 0.5]),
        )
        np.testing.assert_array_equal(result["id"], [10, 11, 12])
        np.testing.assert_allclose(result["xgboost_probability"], [0.3, 0.5, 0.9])
        np.testing.assert_allclose(result["baseline_probability"], [0.15, 0.45, 0.65])
        np.testing.assert_allclose(result["candidate_probability"], [0.18, 0.46, 0.7])

    def test_tiny_actual_categorical_fit_uses_best_iteration_and_unknowns(self) -> None:
        spec = InputSpec(("category", "value"), ("value",), ("category",))
        fit = pd.DataFrame(
            {
                "category": ["a", "b", "a", "b", "a", "b"],
                "value": [0.0, 1.0, np.nan, 1.0, 0.0, 1.0],
            }
        )
        validation = pd.DataFrame(
            {"category": ["a", "unknown", "b"], "value": [np.nan, 0.5, 1.0]}
        )
        levels = fit_category_levels(fit, spec)
        prepared_fit = prepare_lightgbm_frame(fit, spec, levels)
        prepared_validation = prepare_lightgbm_frame(validation, spec, levels)
        self.assertTrue(pd.isna(prepared_validation.loc[1, "category"]))
        self.assertTrue(pd.isna(prepared_fit.loc[2, "value"]))
        model = build_xgboost_classifier(
            n_estimators=20,
            early_stopping_rounds=3,
            max_depth=2,
            min_child_weight=1,
        )
        model.fit(
            prepared_fit,
            np.array([0, 1, 0, 1, 0, 1]),
            eval_set=[(prepared_validation, np.array([0, 1, 1]))],
            verbose=False,
        )
        self.assertGreaterEqual(model.best_iteration, 0)
        best_iteration_count = model.best_iteration + 1
        self.assertGreater(best_iteration_count, 0)
        predictions = model.predict_proba(prepared_validation)[:, 1]
        explicit = model.predict_proba(
            prepared_validation,
            iteration_range=(0, model.best_iteration + 1),
        )[:, 1]
        np.testing.assert_allclose(predictions, explicit)
        self.assertGreaterEqual(
            model.get_booster().num_boosted_rounds(), model.best_iteration + 1
        )


if __name__ == "__main__":
    unittest.main()
