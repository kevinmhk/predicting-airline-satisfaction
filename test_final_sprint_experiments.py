import json
import unittest
from typing import Any

import numpy as np
import pandas as pd

from final_sprint_experiments import (
    DISTANCE_CATEGORY,
    EXPERIMENTS,
    FLIGHT_DISTANCE_CATEGORY,
    ROUTE_CONTEXT_TARGET,
    ROUTE_CONTEXT_TARGET_127,
    ROUTE_DISTANCE_TARGET,
    ROUTE_DISTANCE_TARGET_127,
    ROUTE_HIERARCHICAL,
    ROUTE_TEACHER_RESIDUAL,
    append_distance_category,
    build_trial_classifier,
    canonical_distance_key,
    fit_hierarchical_route_encoding,
    fit_route_target_encoding,
    fit_route_teacher_residual_encoding,
)
from lightgbm_experiment import prepare_lightgbm_frame
from train import InputSpec

SPEC = InputSpec(
    feature_columns=("Flight Distance", "Class", "Age"),
    numeric_columns=("Flight Distance", "Age"),
    categorical_columns=("Class",),
)

CONTEXT_SPEC = InputSpec(
    feature_columns=(
        "Flight Distance",
        "Class",
        "Age",
        "Type of Travel",
        "Customer Type",
    ),
    numeric_columns=("Flight Distance", "Age"),
    categorical_columns=("Class", "Type of Travel", "Customer Type"),
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flight Distance": [
                0.0,
                1.0,
                1.0,
                np.nan,
                2.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                99.0,
                1.0,
            ],
            "Class": [
                "Eco",
                "Business",
                "Eco",
                "Eco",
                "Business",
                "Eco",
                "Business",
                "Eco",
                "Business",
                "Eco",
                "Eco",
                None,
            ],
            "Age": list(range(20, 32)),
        }
    )


def _context_features() -> pd.DataFrame:
    features = _features().copy()
    features["Type of Travel"] = [
        "Personal",
        "Business",
        "Business",
        "Personal",
        "Business",
        "Personal",
        "Business",
        "Personal",
        "Business",
        "Personal",
        "Business",
        None,
    ]
    features["Customer Type"] = [
        "Loyal Customer",
        "disloyal Customer",
        "Loyal Customer",
        "disloyal Customer",
        "Loyal Customer",
        "disloyal Customer",
        "Loyal Customer",
        "disloyal Customer",
        "Loyal Customer",
        "disloyal Customer",
        "Loyal Customer",
        None,
    ]
    return features


class FinalSprintContracts(unittest.TestCase):
    def test_only_eight_predeclared_experiments_exist(self) -> None:
        self.assertEqual(
            tuple(EXPERIMENTS),
            (
                "distance_category",
                "leaves127",
                "route_target_encoding",
                "route_target_encoding127",
                "route_context_target_encoding",
                "route_context_target_encoding127",
                "route_hierarchical_encoding",
                "route_teacher_residual_encoding",
            ),
        )
        self.assertEqual(DISTANCE_CATEGORY, "distance_category")
        self.assertEqual(FLIGHT_DISTANCE_CATEGORY, "flight_distance_category")
        self.assertEqual(ROUTE_DISTANCE_TARGET, "route_target_encoding")
        self.assertEqual(ROUTE_DISTANCE_TARGET_127, "route_target_encoding127")
        self.assertEqual(ROUTE_CONTEXT_TARGET, "route_context_target_encoding")
        self.assertEqual(ROUTE_CONTEXT_TARGET_127, "route_context_target_encoding127")
        self.assertEqual(ROUTE_HIERARCHICAL, "route_hierarchical_encoding")
        self.assertEqual(ROUTE_TEACHER_RESIDUAL, "route_teacher_residual_encoding")

    def test_leaves127_changes_only_num_leaves(self) -> None:
        original = build_trial_classifier("distance_category").get_params()
        leaves127 = build_trial_classifier("leaves127").get_params()
        changed = {
            key: (original.get(key), leaves127.get(key))
            for key in set(original) | set(leaves127)
            if original.get(key) != leaves127.get(key)
        }
        self.assertEqual(changed, {"num_leaves": (31, 127)})
        self.assertEqual(leaves127["min_child_samples"], 40)
        self.assertEqual(leaves127["learning_rate"], 0.04)
        self.assertEqual(leaves127["reg_lambda"], 1.0)

    def test_route_target_encoding127_changes_only_num_leaves(self) -> None:
        original = build_trial_classifier(ROUTE_DISTANCE_TARGET).get_params()
        target_encoding127 = build_trial_classifier(
            ROUTE_DISTANCE_TARGET_127
        ).get_params()
        changed = {
            key: (original.get(key), target_encoding127.get(key))
            for key in set(original) | set(target_encoding127)
            if original.get(key) != target_encoding127.get(key)
        }
        self.assertEqual(changed, {"num_leaves": (31, 127)})

    def test_route_context_target_encoding127_changes_only_num_leaves(self) -> None:
        original = build_trial_classifier(ROUTE_CONTEXT_TARGET).get_params()
        context127 = build_trial_classifier(ROUTE_CONTEXT_TARGET_127).get_params()
        changed = {
            key: (original.get(key), context127.get(key))
            for key in set(original) | set(context127)
            if original.get(key) != context127.get(key)
        }
        self.assertEqual(changed, {"num_leaves": (31, 127)})

    def test_distance_category_is_canonical_and_fit_levels_are_native(self) -> None:
        features = _features()
        extended, extended_spec = append_distance_category(features, SPEC)
        self.assertEqual(
            extended_spec.categorical_columns, ("Class", FLIGHT_DISTANCE_CATEGORY)
        )
        self.assertEqual(extended_spec.numeric_columns, SPEC.numeric_columns)
        self.assertEqual(
            extended[FLIGHT_DISTANCE_CATEGORY].tolist()[:5],
            ["0", "1", "1", "<missing>", "2"],
        )
        self.assertEqual(canonical_distance_key(0.0), "0")
        self.assertEqual(canonical_distance_key(np.nan), "<missing>")
        self.assertEqual(canonical_distance_key(1.0), canonical_distance_key(1))
        prepared = prepare_lightgbm_frame(
            extended.iloc[:5],
            extended_spec,
            {"Class": ["Eco", "Business"], FLIGHT_DISTANCE_CATEGORY: ["0", "1"]},
        )
        self.assertEqual(str(prepared[FLIGHT_DISTANCE_CATEGORY].dtype), "category")
        self.assertTrue(pd.isna(prepared.iloc[3][FLIGHT_DISTANCE_CATEGORY]))

    def test_route_encoding_is_crossfit_and_heldout_labels_do_not_matter(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        features = _features()
        fit_indices = np.arange(10, dtype=np.int64)
        validation_indices = np.array([10], dtype=np.int64)
        holdout_indices = np.array([11], dtype=np.int64)
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
        encoded = fit_route_target_encoding(
            features,
            SPEC,
            labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        labels[10:] = 1 - labels[10:]
        changed_heldout = fit_route_target_encoding(
            features,
            SPEC,
            labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        pd.testing.assert_frame_equal(encoded.features, changed_heldout.features)
        pd.testing.assert_frame_equal(encoded.features.iloc[:, :3], features)
        self.assertEqual(encoded.spec.categorical_columns, SPEC.categorical_columns)
        self.assertEqual(
            encoded.spec.numeric_columns,
            SPEC.numeric_columns
            + (
                "flight_distance_target_mean",
                "flight_distance_class_target_mean",
            ),
        )

        # Independent smoothed-count oracle: each model-fit row uses only
        # its four complementary inner folds, never its own target.
        distances = [
            None if pd.isna(x) else float(x) for x in features["Flight Distance"]
        ]
        classes = [None if pd.isna(x) else str(x) for x in features["Class"]]
        keys: list[list[object]] = [
            list(distances),
            list(zip(distances, classes, strict=True)),
        ]
        expected = np.zeros((len(features), 2))

        def map_rows(fit: np.ndarray, query: np.ndarray) -> None:
            prior = float(np.mean(labels[fit]))
            for column, values in enumerate(keys):
                for row in query:
                    members = [index for index in fit if values[index] == values[row]]
                    expected[row, column] = (
                        float(np.sum(labels[members])) + 20.0 * prior
                    ) / (len(members) + 20.0)

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fit, query in folds.split(fit_indices, labels[fit_indices]):
            map_rows(fit_indices[fit], fit_indices[query])
        map_rows(fit_indices, np.concatenate((validation_indices, holdout_indices)))
        np.testing.assert_allclose(encoded.features.iloc[:, -2:], expected, atol=1e-15)
        self.assertEqual(encoded.features.iloc[10, -2], np.mean(labels[fit_indices]))
        self.assertEqual(encoded.features.iloc[11, -1], np.mean(labels[fit_indices]))
        self.assertEqual(encoded.metadata["fit_transform_rows"], len(fit_indices))
        self.assertEqual(encoded.metadata["cv"], 5)
        self.assertEqual(encoded.metadata["smooth"], 20.0)
        self.assertEqual(encoded.metadata["source"], "inner-fit satisfaction labels")
        self.assertEqual(
            encoded.metadata["prior_policy"], "encoding-fit global prevalence"
        )
        membership = encoded.metadata["fold_membership"]
        self.assertEqual(len(membership), 5)
        from lightgbm_capacity import _array_fingerprint

        for recorded, (fit, query) in zip(
            membership, folds.split(fit_indices, labels[fit_indices]), strict=True
        ):
            self.assertEqual(
                recorded["fit_sha256"], _array_fingerprint(fit_indices[fit])
            )
            self.assertEqual(
                recorded["query_sha256"], _array_fingerprint(fit_indices[query])
            )

    def test_context_route_encoding_has_four_oracle_checked_features(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        features = _context_features()
        fit_indices = np.arange(10, dtype=np.int64)
        validation_indices = np.array([10], dtype=np.int64)
        holdout_indices = np.array([11], dtype=np.int64)
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)
        encoded = fit_route_target_encoding(
            features,
            CONTEXT_SPEC,
            labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
            include_context=True,
        )

        original = fit_route_target_encoding(
            features.loc[:, list(SPEC.feature_columns)],
            SPEC,
            labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        np.testing.assert_array_equal(
            encoded.features.iloc[:, -4:-2].to_numpy(),
            original.features.iloc[:, -2:].to_numpy(),
        )
        self.assertEqual(
            encoded.spec.numeric_columns,
            CONTEXT_SPEC.numeric_columns
            + (
                "flight_distance_target_mean",
                "flight_distance_class_target_mean",
                "flight_distance_travel_target_mean",
                "flight_distance_customer_target_mean",
            ),
        )

        def distance_key(value: Any) -> str:
            if pd.isna(value):
                return "<missing>"
            numeric = float(value)
            return "0" if numeric == 0.0 else format(numeric, ".17g")

        def category_key(value: Any) -> str:
            return "<missing>" if pd.isna(value) else str(value)

        distances = [distance_key(value) for value in features["Flight Distance"]]
        classes = [category_key(value) for value in features["Class"]]
        travels = [category_key(value) for value in features["Type of Travel"]]
        customers = [category_key(value) for value in features["Customer Type"]]

        def pair_keys(values: list[str]) -> list[str]:
            return [
                json.dumps((distance, value), ensure_ascii=True, separators=(",", ":"))
                for distance, value in zip(distances, values, strict=True)
            ]

        keys: list[list[Any]] = [
            distances,
            pair_keys(classes),
            pair_keys(travels),
            pair_keys(customers),
        ]
        expected = np.zeros((len(features), 4))

        def map_rows(fit: np.ndarray, query: np.ndarray) -> None:
            prior = float(np.mean(labels[fit]))
            for column, values in enumerate(keys):
                for row in query:
                    members = [index for index in fit if values[index] == values[row]]
                    expected[row, column] = (
                        float(np.sum(labels[members])) + 20.0 * prior
                    ) / (len(members) + 20.0)

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fit, query in folds.split(fit_indices, labels[fit_indices]):
            map_rows(fit_indices[fit], fit_indices[query])
        map_rows(fit_indices, np.concatenate((validation_indices, holdout_indices)))
        np.testing.assert_allclose(encoded.features.iloc[:, -4:], expected, atol=1e-15)
        prior = float(np.mean(labels[fit_indices]))
        self.assertEqual(encoded.features.iloc[10, -1], prior)
        self.assertEqual(encoded.features.iloc[11, -2], prior)
        self.assertEqual(encoded.features.iloc[11, -1], prior)

        changed_labels = labels.copy()
        changed_labels[10:] = 1 - changed_labels[10:]
        changed_heldout = fit_route_target_encoding(
            features,
            CONTEXT_SPEC,
            changed_labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
            include_context=True,
        )
        pd.testing.assert_frame_equal(encoded.features, changed_heldout.features)

    def test_hierarchical_route_encoding_uses_conditional_priors(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        features = _context_features().copy()
        spec = InputSpec(
            tuple(features.columns),
            ("Flight Distance", "Age"),
            ("Class", "Type of Travel", "Customer Type"),
        )
        fit_indices = np.arange(10, dtype=np.int64)
        validation_indices = np.array([10], dtype=np.int64)
        holdout_indices = np.array([11], dtype=np.int64)
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int8)
        encoded = fit_hierarchical_route_encoding(
            features,
            spec,
            labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        pd.testing.assert_frame_equal(
            encoded.features.iloc[:, : len(spec.feature_columns)], features
        )
        self.assertEqual(
            encoded.spec.feature_columns[: len(spec.feature_columns)],
            spec.feature_columns,
        )

        distances = [
            canonical_distance_key(value) for value in features["Flight Distance"]
        ]
        categories = {
            column: [
                "<missing>" if pd.isna(value) else str(value)
                for value in features[column]
            ]
            for column in ("Class", "Type of Travel", "Customer Type")
        }
        keys: list[list[Any]] = [
            distances,
            list(zip(distances, categories["Class"], strict=True)),
            list(zip(distances, categories["Type of Travel"], strict=True)),
            list(zip(distances, categories["Customer Type"], strict=True)),
        ]
        expected = np.zeros((len(features), 4))

        def map_rows(fit: np.ndarray, query: np.ndarray) -> None:
            global_prior = float(np.mean(labels[fit]))
            for column, values in enumerate(keys):
                for row in query:
                    category_prior = global_prior
                    if column:
                        category = values[row][1]
                        category_members = [
                            index
                            for index in fit
                            if categories[
                                ("Class", "Type of Travel", "Customer Type")[column - 1]
                            ][index]
                            == category
                        ]
                        if category_members:
                            category_prior = float(np.mean(labels[category_members]))
                    members = [index for index in fit if values[index] == values[row]]
                    expected[row, column] = (
                        float(np.sum(labels[members])) + 20.0 * category_prior
                    ) / (len(members) + 20.0)

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fit, query in folds.split(fit_indices, labels[fit_indices]):
            map_rows(fit_indices[fit], fit_indices[query])
        map_rows(fit_indices, np.concatenate((validation_indices, holdout_indices)))

        np.testing.assert_allclose(encoded.features.iloc[:, -4:], expected, atol=1e-15)
        self.assertEqual(
            encoded.spec.numeric_columns,
            spec.numeric_columns
            + (
                "flight_distance_hierarchical_target_mean",
                "flight_distance_class_hierarchical_target_mean",
                "flight_distance_travel_hierarchical_target_mean",
                "flight_distance_customer_hierarchical_target_mean",
            ),
        )
        self.assertEqual(encoded.metadata["smooth"], 20.0)
        self.assertEqual(
            encoded.metadata["prior_policy"], "global; pair category conditional"
        )
        self.assertTrue(np.isfinite(encoded.features.iloc[:, -4:].to_numpy()).all())

        changed_labels = labels.copy()
        changed_labels[10:] = 1 - changed_labels[10:]
        changed_heldout = fit_hierarchical_route_encoding(
            features,
            spec,
            changed_labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        pd.testing.assert_frame_equal(encoded.features, changed_heldout.features)

    def test_teacher_residual_route_encoding_uses_frozen_teacher_residuals(
        self,
    ) -> None:
        from sklearn.model_selection import StratifiedKFold

        features = _context_features().copy()
        features["source_teacher_probability"] = np.array(
            [0.9, 0.1, 0.2, 0.8, 0.1, 0.7, 0.2, 0.8, 0.1, 0.6, 0.3, 0.4]
        )
        spec = InputSpec(
            tuple(features.columns),
            ("Flight Distance", "Age", "source_teacher_probability"),
            ("Class", "Type of Travel", "Customer Type"),
        )
        fit_indices = np.arange(10, dtype=np.int64)
        validation_indices = np.array([10], dtype=np.int64)
        holdout_indices = np.array([11], dtype=np.int64)
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int8)
        encoded = fit_route_teacher_residual_encoding(
            features,
            spec,
            labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        pd.testing.assert_frame_equal(
            encoded.features.iloc[:, : len(spec.feature_columns)], features
        )
        self.assertEqual(
            encoded.spec.feature_columns[: len(spec.feature_columns)],
            spec.feature_columns,
        )

        distances = [
            canonical_distance_key(value) for value in features["Flight Distance"]
        ]
        categories = {
            column: [
                "<missing>" if pd.isna(value) else str(value)
                for value in features[column]
            ]
            for column in ("Class", "Type of Travel", "Customer Type")
        }
        keys: list[list[Any]] = [
            distances,
            list(zip(distances, categories["Class"], strict=True)),
            list(zip(distances, categories["Type of Travel"], strict=True)),
            list(zip(distances, categories["Customer Type"], strict=True)),
        ]
        residual_targets = labels - features["source_teacher_probability"].to_numpy()
        expected = np.zeros((len(features), 4))

        def map_rows(fit: np.ndarray, query: np.ndarray) -> None:
            prior = float(np.mean(residual_targets[fit]))
            for column, values in enumerate(keys):
                for row in query:
                    members = [index for index in fit if values[index] == values[row]]
                    expected[row, column] = (
                        float(np.sum(residual_targets[members])) + 20.0 * prior
                    ) / (len(members) + 20.0)

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fit, query in folds.split(fit_indices, labels[fit_indices]):
            map_rows(fit_indices[fit], fit_indices[query])
        map_rows(fit_indices, np.concatenate((validation_indices, holdout_indices)))

        np.testing.assert_allclose(encoded.features.iloc[:, -4:], expected, atol=1e-15)
        self.assertEqual(
            encoded.spec.numeric_columns[-4:],
            (
                "flight_distance_teacher_residual_mean",
                "flight_distance_class_teacher_residual_mean",
                "flight_distance_travel_teacher_residual_mean",
                "flight_distance_customer_teacher_residual_mean",
            ),
        )
        values = encoded.features.iloc[:, -4:].to_numpy()
        self.assertTrue((values < 0).any())
        self.assertTrue((values > 0).any())
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(encoded.metadata["smooth"], 20.0)
        self.assertEqual(
            encoded.metadata["target_strategy"],
            "label minus frozen teacher probability",
        )

        changed_labels = labels.copy()
        changed_labels[10:] = 1 - changed_labels[10:]
        changed_heldout = fit_route_teacher_residual_encoding(
            features,
            spec,
            changed_labels[fit_indices],
            fit_indices,
            validation_indices,
            holdout_indices,
        )
        pd.testing.assert_frame_equal(encoded.features, changed_heldout.features)


if __name__ == "__main__":
    unittest.main()
