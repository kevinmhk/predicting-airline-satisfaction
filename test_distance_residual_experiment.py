import unittest
from typing import Any

import numpy as np
import pandas as pd

from distance_profile_experiment import (
    DISTANCE_COLUMN,
    append_distance_profile_features,
    fit_distance_profile_table,
)
from distance_residual_experiment import (
    RESIDUAL_PREFIX,
    append_distance_residual_features,
    distance_residual_feature_names,
)
from source_teacher_experiment import append_teacher_feature
from train import InputSpec

NUMERIC = (
    DISTANCE_COLUMN,
    "Age",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
)
CATEGORICAL = ("Gender", "Customer Type", "Type of Travel", "Class")


def _spec() -> InputSpec:
    return InputSpec(NUMERIC + CATEGORICAL, NUMERIC, CATEGORICAL)


def _features() -> pd.DataFrame:
    values: dict[str, Any] = {
        column: np.array([10.0, 20.0, np.nan, 0.0], dtype=np.float64)
        for column in NUMERIC
    }
    values[DISTANCE_COLUMN] = np.array([100.0, 100.0, 300.0, 0.0])
    values["Age"] = np.array([10.0, 30.0, 50.0, np.nan])
    values["Inflight wifi service"] = np.array([0.0, 2.0, np.nan, 0.0])
    values.update(
        {
            "Gender": ["Male", "Female", "Male", "Female"],
            "Customer Type": ["Loyal", "Loyal", "Disloyal", "Loyal"],
            "Type of Travel": ["Business", "Business", "Personal", "Business"],
            "Class": ["Business", "Eco", "Eco", "Business"],
        }
    )
    return pd.DataFrame(values).loc[:, list(_spec().feature_columns)]


class DistanceResidualTests(unittest.TestCase):
    def test_declares_exactly_sixteen_original_numeric_residuals(self) -> None:
        names = distance_residual_feature_names(
            tuple(column for column in NUMERIC if column != DISTANCE_COLUMN)
        )
        self.assertEqual(len(names), 16)
        self.assertEqual(names[0], f"{RESIDUAL_PREFIX}Age")
        self.assertNotIn(f"{RESIDUAL_PREFIX}{DISTANCE_COLUMN}", names)

    def test_fallback_missing_and_zero_arithmetic(self) -> None:
        spec = _spec()
        features = _features()
        original = features.copy(deep=True)
        table = fit_distance_profile_table(features.iloc[:2], spec)
        with_teacher, teacher_spec = append_teacher_feature(
            features, np.full(len(features), 0.4), spec
        )
        with_profile, profile_spec = append_distance_profile_features(
            with_teacher, teacher_spec, table
        )
        residual, residual_spec = append_distance_residual_features(
            with_profile, profile_spec, table
        )

        self.assertEqual(len(residual_spec.feature_columns), 55)
        self.assertEqual(len(residual_spec.numeric_columns), 51)
        self.assertEqual(residual_spec.categorical_columns, CATEGORICAL)
        np.testing.assert_allclose(
            residual.loc[2, f"{RESIDUAL_PREFIX}Age"], 50.0 - 20.0
        )
        self.assertTrue(
            pd.isna(residual.loc[2, f"{RESIDUAL_PREFIX}Inflight wifi service"])
        )
        self.assertEqual(
            residual.loc[3, f"{RESIDUAL_PREFIX}Inflight wifi service"], -1.0
        )
        pd.testing.assert_frame_equal(features, original)

    def test_schema_id_target_and_name_collisions_are_rejected(self) -> None:
        spec = _spec()
        features = _features()
        table = fit_distance_profile_table(features.iloc[:2], spec)
        with_teacher, teacher_spec = append_teacher_feature(
            features, np.full(len(features), 0.4), spec
        )
        with_profile, profile_spec = append_distance_profile_features(
            with_teacher, teacher_spec, table
        )
        collision = f"{RESIDUAL_PREFIX}Age"
        collision_features = with_profile.assign(**{collision: 0.0})
        collision_spec = InputSpec(
            profile_spec.feature_columns + (collision,),
            profile_spec.numeric_columns + (collision,),
            profile_spec.categorical_columns,
        )
        with self.assertRaisesRegex(ValueError, "already exist"):
            append_distance_residual_features(collision_features, collision_spec, table)
        with self.assertRaisesRegex(ValueError, "schema exactly"):
            append_distance_residual_features(
                with_profile.assign(id=1), profile_spec, table
            )
        with self.assertRaisesRegex(ValueError, "schema exactly"):
            append_distance_residual_features(
                with_profile.assign(satisfaction=1), profile_spec, table
            )


if __name__ == "__main__":
    unittest.main()
