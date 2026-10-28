import unittest

import numpy as np
import pandas as pd

from distance_profile_experiment import (
    DISTANCE_COLUMN,
    PROFILE_COUNT_COLUMN,
    PROFILE_MEAN_PREFIX,
    append_distance_profile_features,
    apply_distance_profile,
    distance_profile_feature_names,
    distance_profile_gate,
    fit_distance_profile_table,
)
from source_teacher_experiment import append_teacher_feature
from train import InputSpec


def _spec() -> InputSpec:
    return InputSpec(
        feature_columns=("Age", DISTANCE_COLUMN, "Rating", "Delay"),
        numeric_columns=("Age", DISTANCE_COLUMN, "Rating", "Delay"),
        categorical_columns=(),
    )


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [20.0, 30.0, 40.0, 50.0, np.nan],
            DISTANCE_COLUMN: [100.0, 100.0, 200.0, np.nan, 0.0],
            "Rating": [0.0, 2.0, 4.0, np.nan, 0.0],
            "Delay": [10.0, np.nan, 30.0, 40.0, np.nan],
        }
    )


class DistanceProfileTests(unittest.TestCase):
    def test_profiles_exclude_the_frozen_teacher_feature(self) -> None:
        features, spec = _features(), _spec()
        teacher = np.full(len(features), 0.4)
        with_teacher, teacher_spec = append_teacher_feature(features, teacher, spec)
        table = fit_distance_profile_table(features.iloc[:3], spec)
        result, result_spec = append_distance_profile_features(
            with_teacher, teacher_spec, table
        )
        self.assertEqual(table.mean_columns, ("Age", "Rating", "Delay"))
        np.testing.assert_array_equal(result["source_teacher_probability"], teacher)
        self.assertEqual(result_spec.categorical_columns, spec.categorical_columns)

    def test_gate_requires_both_recorded_seeds(self) -> None:
        baseline = {42: 0.9, 2024: 0.9}
        passed = distance_profile_gate(baseline, {42: 0.9003, 2024: 0.9003})
        self.assertTrue(passed["qualifies"])
        self.assertEqual(passed["decision"], "select_distance_profile_feature")
        failed = distance_profile_gate(baseline, {42: 0.9003, 2024: 0.9001})
        self.assertFalse(failed["qualifies"])
        self.assertEqual(failed["decision"], "retain_current_blend")

    def test_fit_uses_only_fit_predictors_and_is_label_independent(self) -> None:
        features = _features()
        fit_rows = features.iloc[:3]
        table = fit_distance_profile_table(fit_rows, _spec())
        changed_validation = features.iloc[3:].copy()
        changed_validation.loc[:, :] = 999.0
        repeated = fit_distance_profile_table(fit_rows, _spec())
        np.testing.assert_array_equal(table.group_counts, repeated.group_counts)
        pd.testing.assert_frame_equal(table.group_means, repeated.group_means)
        pd.testing.assert_series_equal(table.global_means, repeated.global_means)
        with self.assertRaisesRegex(ValueError, "schema exactly"):
            fit_distance_profile_table(fit_rows.assign(satisfaction=[1, 0, 1]), _spec())
        self.assertFalse(changed_validation.equals(features.iloc[3:]))
        self.assertNotIn("satisfaction", table.mean_columns)
        self.assertNotIn("id", table.mean_columns)

    def test_group_count_and_means_preserve_zero_and_skip_nan(self) -> None:
        table = fit_distance_profile_table(_features().iloc[:3], _spec())
        mapped = apply_distance_profile(_features(), table)
        self.assertEqual(mapped.loc[0, PROFILE_COUNT_COLUMN], 2.0)
        self.assertEqual(mapped.loc[4, PROFILE_COUNT_COLUMN], 0.0)
        self.assertEqual(mapped.loc[0, f"{PROFILE_MEAN_PREFIX}Rating"], 1.0)
        self.assertEqual(mapped.loc[0, f"{PROFILE_MEAN_PREFIX}Delay"], 10.0)
        self.assertEqual(mapped.loc[4, f"{PROFILE_MEAN_PREFIX}Rating"], 2.0)
        self.assertEqual(mapped.loc[4, f"{PROFILE_MEAN_PREFIX}Delay"], 20.0)

    def test_unseen_and_missing_distance_use_global_fallbacks(self) -> None:
        table = fit_distance_profile_table(_features().iloc[:3], _spec())
        probe = pd.DataFrame(
            {
                "Age": [1.0, 2.0],
                DISTANCE_COLUMN: [300.0, np.nan],
                "Rating": [5.0, 5.0],
                "Delay": [5.0, 5.0],
            }
        )
        mapped = apply_distance_profile(probe, table)
        self.assertTrue((mapped[PROFILE_COUNT_COLUMN] == 0.0).all())
        np.testing.assert_allclose(mapped[f"{PROFILE_MEAN_PREFIX}Rating"], [2.0, 2.0])
        np.testing.assert_allclose(mapped[f"{PROFILE_MEAN_PREFIX}Delay"], [20.0, 20.0])

    def test_all_nan_global_column_remains_nan_for_unknown_keys(self) -> None:
        features = _features().drop(columns="Delay")
        features["Delay"] = np.nan
        table = fit_distance_profile_table(features, _spec())
        mapped = apply_distance_profile(features, table)
        self.assertTrue(mapped[f"{PROFILE_MEAN_PREFIX}Delay"].isna().all())

    def test_append_preserves_original_frame_and_declares_numeric_features(
        self,
    ) -> None:
        features = _features()
        original = features.copy(deep=True)
        spec = _spec()
        table = fit_distance_profile_table(features.iloc[:3], spec)
        extended, extended_spec = append_distance_profile_features(
            features, spec, table
        )
        pd.testing.assert_frame_equal(features, original)
        self.assertEqual(
            list(extended.columns),
            list(spec.feature_columns)
            + list(distance_profile_feature_names(table.mean_columns)),
        )
        self.assertEqual(
            extended_spec.numeric_columns,
            spec.numeric_columns + distance_profile_feature_names(table.mean_columns),
        )
        self.assertEqual(extended_spec.categorical_columns, ())

    def test_schema_and_name_collisions_are_rejected(self) -> None:
        spec = _spec()
        features = _features()
        table = fit_distance_profile_table(features.iloc[:3], spec)
        with self.assertRaisesRegex(ValueError, "schema exactly"):
            fit_distance_profile_table(features.assign(id=1), spec)
        with self.assertRaisesRegex(ValueError, "already exist"):
            collision_features = features.assign(**{PROFILE_COUNT_COLUMN: 1.0})
            collision_spec = InputSpec(
                feature_columns=spec.feature_columns + (PROFILE_COUNT_COLUMN,),
                numeric_columns=spec.numeric_columns + (PROFILE_COUNT_COLUMN,),
                categorical_columns=(),
            )
            append_distance_profile_features(collision_features, collision_spec, table)
        with self.assertRaisesRegex(ValueError, "unique"):
            distance_profile_feature_names(("Rating", "Rating"))


if __name__ == "__main__":
    unittest.main()
