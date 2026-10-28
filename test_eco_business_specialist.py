import unittest

import numpy as np
import pandas as pd

from eco_business_specialist import (
    assign_specialist_oof,
    group_mask,
    route_candidate,
    specialist_membership,
)
from eco_business_specialist_export import derive_export_iterations, route_csv_records
from five_fold_cv import make_five_fold_splits
from test_train import synthetic_frames
from train import validate_inputs


class EcoBusinessSpecialistTests(unittest.TestCase):
    def test_group_mask_is_exact(self) -> None:
        frame = pd.DataFrame(
            {
                "Class": ["Eco", "Eco", "Business", "Eco"],
                "Type of Travel": [
                    "Business travel",
                    "Personal Travel",
                    "Business travel",
                    "Business travel",
                ],
                "Customer Type": [
                    "Loyal Customer",
                    "Loyal Customer",
                    "Loyal Customer",
                    "disloyal Customer",
                ],
            }
        )
        np.testing.assert_array_equal(group_mask(frame), [True, False, False, False])

    def test_membership_is_inner_group_only_and_disjoint(self) -> None:
        y = np.array([0, 1] * 10, dtype=np.int8)
        group = np.ones(len(y), dtype=bool)
        folds = make_five_fold_splits(y)
        outer_fit, outer_holdout = folds[0]
        membership = specialist_membership(outer_fit, outer_holdout, y, group)
        self.assertTrue(set(membership["inner_fit"]).issubset(set(outer_fit)))
        self.assertTrue(set(membership["inner_validation"]).issubset(set(outer_fit)))
        self.assertTrue(set(membership["outer_holdout"]).issubset(set(outer_holdout)))
        self.assertFalse(
            set(membership["inner_fit"]) & set(membership["inner_validation"])
        )
        self.assertFalse(
            set(membership["inner_fit"]) & set(membership["outer_holdout"])
        )

    def test_route_uses_fixed_weights_and_preserves_non_group_bits(self) -> None:
        baseline = np.array([0.1, 0.2, 0.3, 0.4])
        specialist = np.array([0.9, 0.8, 0.7, 0.6])
        group = np.array([False, True, False, True])
        candidate = route_candidate(baseline, specialist, group)
        self.assertAlmostEqual(candidate[1], 0.35)
        self.assertAlmostEqual(candidate[3], 0.45)
        self.assertEqual(candidate[0], baseline[0])
        self.assertEqual(candidate[2], baseline[2])
        self.assertTrue(np.array_equal(candidate[~group], baseline[~group]))

    def test_specialist_oof_requires_complete_group_coverage(self) -> None:
        values, folds = assign_specialist_oof(
            6,
            [
                (1, np.array([0, 2]), np.array([0.1, 0.2])),
                (2, np.array([4]), np.array([0.3])),
            ],
        )
        self.assertEqual(int(np.count_nonzero(folds > 0)), 3)
        self.assertTrue(np.isnan(values[1]))
        with self.assertRaisesRegex(ValueError, "twice"):
            assign_specialist_oof(
                3,
                [
                    (1, np.array([0]), np.array([0.1])),
                    (2, np.array([0]), np.array([0.2])),
                ],
            )

    def test_export_iterations_use_five_one_based_counts(self) -> None:
        self.assertEqual(derive_export_iterations([791, 954, 764, 1182, 1038]), 954)

    def test_export_routing_preserves_outside_text_and_excludes_raw_id(self) -> None:
        train, test = synthetic_frames()
        spec = validate_inputs(train, test)
        self.assertNotIn("id", spec.feature_columns)
        self.assertNotIn("satisfaction", spec.feature_columns)
        records = [("100", "0.10000000000000001"), ("101", "0.2")]
        routed = route_csv_records(
            records,
            np.array([False, True]),
            np.array([0.1, 0.35]),
        )
        self.assertEqual(routed[0][1], records[0][1])
        self.assertEqual(routed[1][1], "0.34999999999999998")


if __name__ == "__main__":
    unittest.main()
