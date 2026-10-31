import json
import unittest

import numpy as np
import pandas as pd

import final_sprint_experiments as f
from round2_feature_experiments import (
    BOARDING_ROUTE,
    CONDITIONAL_FEATURE_NAMES,
    JOINT_PASSENGER,
    TEACHER_ROUTE,
    _teacher_bin,
    append_conditional_feature,
    fit_conditional_feature,
)
from train import InputSpec


def _features() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "Flight Distance": [0.0, 100, 200, np.nan] * 10,
            "Online boarding": ([0, 1, 2, np.nan, 3, 4, 5, 99] * 5),
            "source_teacher_probability": np.tile([0.0, 0.2, 0.4, 0.8, 1.0], 8),
            "Class": (["Eco", "Business", "Eco", None] * 10),
            "Type of Travel": (["Personal", "Business", None, "Personal"] * 10),
            "Customer Type": (["Loyal", "New", "Loyal", None] * 10),
            "Gender": (["Female", "Male", "Female", None] * 10),
        },
        index=np.arange(rows),
    )


def _spec(features: pd.DataFrame) -> InputSpec:
    categorical = ("Gender", "Class", "Type of Travel", "Customer Type")
    return InputSpec(
        tuple(features.columns),
        tuple(column for column in features if column not in categorical),
        categorical,
    )


def _json_key(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _oracle(
    mode: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    holdout: np.ndarray,
) -> np.ndarray:
    distances = [
        f.canonical_distance_key(value) for value in features["Flight Distance"]
    ]
    if mode == BOARDING_ROUTE:
        parents = [
            f.canonical_distance_key(value) for value in features["Online boarding"]
        ]
        keys = [
            _json_key((distance, parent))
            for distance, parent in zip(distances, parents, strict=True)
        ]
    elif mode == TEACHER_ROUTE:
        parents = [
            str(_teacher_bin(value)) for value in features["source_teacher_probability"]
        ]
        keys = [
            _json_key((distance, parent))
            for distance, parent in zip(distances, parents, strict=True)
        ]
    else:
        class_values = [f._canonical_class_key(value) for value in features["Class"]]
        travel_values = [
            f._canonical_class_key(value) for value in features["Type of Travel"]
        ]
        customer_values = [
            f._canonical_class_key(value) for value in features["Customer Type"]
        ]
        parents = [
            _json_key((category, travel, customer))
            for category, travel, customer in zip(
                class_values, travel_values, customer_values, strict=True
            )
        ]
        keys = [
            _json_key((distance, category, travel, customer))
            for distance, category, travel, customer in zip(
                distances, class_values, travel_values, customer_values, strict=True
            )
        ]

    result = np.full(len(features), np.nan, dtype=np.float64)

    def map_rows(statistics: np.ndarray, queries: np.ndarray) -> None:
        prior = float(np.mean(labels[statistics]))
        for row in queries:
            members = [index for index in statistics if keys[index] == keys[row]]
            parent_members = [
                index for index in statistics if parents[index] == parents[row]
            ]
            parent_prior = (
                float(np.mean(labels[parent_members])) if parent_members else prior
            )
            result[row] = (float(np.sum(labels[members])) + 20.0 * parent_prior) / (
                len(members) + 20.0
            )

    for statistics, queries in f._route_fold_queries(fit, labels[fit]):
        map_rows(statistics, queries)
    map_rows(fit, np.concatenate((validation, holdout)))
    return result


class Round2FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = _features()
        self.spec = _spec(self.features)
        self.fit = np.arange(30, dtype=np.int64)
        self.validation = np.arange(30, 35, dtype=np.int64)
        self.holdout = np.arange(35, 40, dtype=np.int64)
        self.labels = np.tile([0, 1], 20).astype(np.int8)

    def test_validation_frame_keeps_both_student_columns(self) -> None:
        import round2_feature_experiments as q

        rows = np.array([0, 1])
        ids = np.array([10, 20])
        p = np.array([0.2, 0.8])
        frame = q._validation_frame(rows, ids, p, p, p, p, p)
        q._assert_prediction_schema(frame, pd.DataFrame({"id": ids}), rows)
        self.assertEqual(len(q.VALIDATION_COLUMNS), 7)

    def test_teacher_bins_preserve_boundaries(self) -> None:
        self.assertEqual(_teacher_bin(np.nextafter(0.2, 0.0)), 0)
        self.assertEqual(_teacher_bin(np.nextafter(0.4, 0.0)), 1)
        self.assertEqual(_teacher_bin(np.nextafter(0.8, 0.0)), 3)
        self.assertEqual(
            [_teacher_bin(value) for value in [0.0, 0.2, 0.4, 0.8, 1.0]],
            [0, 1, 2, 4, 4],
        )

    def test_all_three_encodings_match_the_count_oracle(self) -> None:
        for mode in (BOARDING_ROUTE, TEACHER_ROUTE, JOINT_PASSENGER):
            with self.subTest(mode=mode):
                values, metadata = fit_conditional_feature(
                    self.features,
                    self.spec,
                    self.labels[self.fit],
                    self.fit,
                    self.validation,
                    self.holdout,
                    mode,
                )
                np.testing.assert_allclose(
                    values.to_numpy(),
                    _oracle(
                        mode,
                        self.features,
                        self.labels,
                        self.fit,
                        self.validation,
                        self.holdout,
                    ),
                    atol=1e-15,
                )
                self.assertEqual(len(metadata["fold_membership"]), 5)
                self.assertEqual(metadata["key_policy"], mode)
                self.assertEqual(
                    metadata["prior_policy"], "fit-label parent prevalence"
                )

    def test_heldout_labels_cannot_change_the_encoding(self) -> None:
        original, _ = fit_conditional_feature(
            self.features,
            self.spec,
            self.labels[self.fit],
            self.fit,
            self.validation,
            self.holdout,
            JOINT_PASSENGER,
        )
        changed = self.labels.copy()
        changed[30:] = 1 - changed[30:]
        replacement, _ = fit_conditional_feature(
            self.features,
            self.spec,
            changed[self.fit],
            self.fit,
            self.validation,
            self.holdout,
            JOINT_PASSENGER,
        )
        pd.testing.assert_series_equal(original, replacement)

    def test_append_preserves_route_features_and_gives_both_models_44_40_4(
        self,
    ) -> None:
        base = self.features.copy()
        for number in range(32):
            base[f"base_{number}"] = float(number)
        base_spec = _spec(base)
        values, _ = fit_conditional_feature(
            base,
            base_spec,
            self.labels[self.fit],
            self.fit,
            self.validation,
            self.holdout,
            BOARDING_ROUTE,
        )
        context = f.fit_route_target_encoding(
            base,
            base_spec,
            self.labels[self.fit],
            self.fit,
            self.validation,
            self.holdout,
            include_context=True,
        )
        hierarchical = f.fit_hierarchical_route_encoding(
            base,
            base_spec,
            self.labels[self.fit],
            self.fit,
            self.validation,
            self.holdout,
        )
        context_features, context_spec = append_conditional_feature(
            context.features, context.spec, values, BOARDING_ROUTE
        )
        hierarchical_features, hierarchical_spec = append_conditional_feature(
            hierarchical.features, hierarchical.spec, values, BOARDING_ROUTE
        )
        for result, result_spec, route in (
            (context_features, context_spec, context.features),
            (hierarchical_features, hierarchical_spec, hierarchical.features),
        ):
            self.assertEqual(len(result_spec.feature_columns), 44)
            self.assertEqual(len(result_spec.numeric_columns), 40)
            self.assertEqual(len(result_spec.categorical_columns), 4)
            self.assertEqual(
                result_spec.feature_columns[-1],
                CONDITIONAL_FEATURE_NAMES[BOARDING_ROUTE],
            )
            pd.testing.assert_frame_equal(result.iloc[:, :-1], route)

    def test_fixed_student_modes_keep_the_declared_leaf_settings(self) -> None:
        context = f.build_trial_classifier(f.ROUTE_CONTEXT_TARGET_127).get_params()
        hierarchical = f.build_trial_classifier(f.ROUTE_HIERARCHICAL).get_params()
        self.assertEqual(context["num_leaves"], 127)
        self.assertEqual(hierarchical["num_leaves"], 31)
        self.assertEqual(context["learning_rate"], 0.04)
        self.assertEqual(hierarchical["min_child_samples"], 40)


if __name__ == "__main__":
    unittest.main()
