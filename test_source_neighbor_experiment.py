import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import source_neighbor_experiment as neighbor
from distance_profile_experiment import (
    _profile_record,
    _save_profile_table,
    fit_distance_profile_table,
)
from distance_profile_experiment import (
    _source_hashes as profile_source_hashes,
)
from source_neighbor_experiment import (
    SOURCE_NEIGHBOR_FEATURE,
    append_source_neighbor_feature,
    compute_source_neighbor_probabilities,
)
from train import InputSpec


def _spec() -> InputSpec:
    return InputSpec(
        feature_columns=("Flight Distance", "Age", "Class"),
        numeric_columns=("Flight Distance", "Age"),
        categorical_columns=("Class",),
    )


def _source() -> tuple[pd.DataFrame, np.ndarray]:
    return (
        pd.DataFrame(
            {
                "Flight Distance": [100.0, 100.0, 100.0, 200.0, 300.0],
                "Age": [10.0, 10.0, np.nan, 0.0, 0.0],
                "Class": ["Eco", "Business", pd.NA, "Eco", "Business"],
            }
        ),
        np.array([0, 1, 1, 0, 1], dtype=np.int8),
    )


def _oracle(
    query: pd.DataFrame, source: pd.DataFrame, labels: np.ndarray
) -> np.ndarray:
    result = np.full(len(query), float(np.mean(labels)), dtype=np.float64)
    for query_index, query_row in query.iterrows():
        distances = source["Flight Distance"]
        if pd.isna(query_row["Flight Distance"]):
            continue
        same_group = distances.notna() & distances.eq(query_row["Flight Distance"])
        source_rows = source.loc[same_group]
        if source_rows.empty:
            continue
        values: list[float] = []
        weights: list[float] = []
        for source_index, source_row in source_rows.iterrows():
            mismatches = 0
            for column in ("Age", "Class"):
                left, right = query_row[column], source_row[column]
                if pd.isna(left) and pd.isna(right):
                    continue
                if pd.isna(left) or pd.isna(right) or left != right:
                    mismatches += 1
            weight = np.exp(-mismatches)
            weights.append(weight)
            values.append(weight * labels[source_index])
        result[query_index] = sum(values) / sum(weights)
    return result


class SourceNeighborTests(unittest.TestCase):
    def test_all_positive_groups_remain_bounded_despite_roundoff(self) -> None:
        rng = np.random.default_rng(1)
        columns = ("Flight Distance", *[f"x{i}" for i in range(20)])
        spec = InputSpec(columns, columns, ())
        source = pd.DataFrame(rng.integers(0, 2, (7, 21)), columns=columns)
        query = pd.DataFrame(rng.integers(0, 2, (100, 21)), columns=columns)
        source["Flight Distance"] = query["Flight Distance"] = 100
        probabilities = compute_source_neighbor_probabilities(
            query, source, np.ones(len(source)), spec
        )
        np.testing.assert_allclose(probabilities, np.ones(len(query)), atol=1e-15)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_pair_limit_smaller_than_query_block_does_not_skip_rows(self) -> None:
        source, labels = _source()
        query = source.iloc[:3].reset_index(drop=True)
        actual = compute_source_neighbor_probabilities(
            query, source, labels, _spec(), query_block_size=4, max_pair_elements=1
        )
        np.testing.assert_allclose(actual, _oracle(query, source, labels), atol=1e-15)

    def test_reused_profile_with_nan_means_is_valid(self) -> None:
        source, _ = _source()
        source["Age"] = np.nan
        table = fit_distance_profile_table(source, _spec())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.npz"
            _save_profile_table(path, table)
            record = _profile_record(table, path)
            loaded = neighbor._load_verified_profile_table(
                path, record, source, _spec(), np.arange(len(source))
            )
        self.assertTrue(loaded.group_means["Age"].isna().all())

    def test_fingerprints_cover_every_reused_profile_dependency(self) -> None:
        expected, actual = profile_source_hashes(), neighbor._source_hashes()
        self.assertLessEqual(expected.keys(), actual.keys())
        self.assertTrue(all(actual[name] == value for name, value in expected.items()))

    def test_original_failure_survives_a_later_integrity_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    neighbor, "_artifact_snapshot", side_effect=[{}, {"old": "changed"}]
                ),
                patch.object(neighbor, "_input_hashes", return_value={}),
                patch.object(neighbor, "_source_hashes", return_value={}),
                patch.object(
                    neighbor.pd, "read_csv", side_effect=ValueError("original")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "original"):
                    neighbor.run_experiment(root / "data", root / "artifacts")
            metadata = json.loads(
                (
                    root / "artifacts/source_neighbor_feature/compare_metadata.json"
                ).read_text()
            )
            self.assertEqual(metadata["error"]["message"], "original")
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["selection"])

    def test_matches_bruteforce_oracle_with_missing_and_zero_values(self) -> None:
        source, labels = _source()
        query = pd.DataFrame(
            {
                "Flight Distance": [100.0, 100.0, 200.0, 999.0, np.nan],
                "Age": [10.0, np.nan, 0.0, 0.0, 0.0],
                "Class": ["Eco", pd.NA, "Eco", "Eco", "Eco"],
            }
        )

        actual = compute_source_neighbor_probabilities(
            query, source, labels, _spec(), query_block_size=2, max_pair_elements=3
        )

        np.testing.assert_allclose(actual, _oracle(query, source, labels))

    def test_permuted_source_and_query_order_preserves_row_values(self) -> None:
        source, labels = _source()
        query = pd.DataFrame(
            {
                "Flight Distance": [100.0, 200.0, 300.0],
                "Age": [10.0, 0.0, 0.0],
                "Class": ["Eco", "Eco", "Business"],
            }
        )
        expected = compute_source_neighbor_probabilities(
            query, source, labels, _spec(), query_block_size=1, max_pair_elements=2
        )
        source_order = [4, 2, 0, 3, 1]
        query_order = [2, 0, 1]
        permuted = compute_source_neighbor_probabilities(
            query.iloc[query_order].reset_index(drop=True),
            source.iloc[source_order].reset_index(drop=True),
            labels[source_order],
            _spec(),
            query_block_size=3,
            max_pair_elements=10,
        )
        np.testing.assert_allclose(permuted, expected[query_order])

    def test_chunk_sizes_do_not_change_exact_all_source_result(self) -> None:
        source, labels = _source()
        query = pd.concat([source.iloc[:3], source.iloc[3:]], ignore_index=True)
        expected = compute_source_neighbor_probabilities(
            query, source, labels, _spec(), query_block_size=1, max_pair_elements=1
        )
        actual = compute_source_neighbor_probabilities(
            query, source, labels, _spec(), query_block_size=4, max_pair_elements=20
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_source_labels_change_probability_and_competition_target_is_rejected(
        self,
    ) -> None:
        source, labels = _source()
        query = source.iloc[[0]].copy()
        first = compute_source_neighbor_probabilities(query, source, labels, _spec())
        second = compute_source_neighbor_probabilities(
            query, source, 1 - labels, _spec()
        )
        self.assertNotEqual(first[0], second[0])
        with self.assertRaisesRegex(ValueError, "predictor schema exactly"):
            compute_source_neighbor_probabilities(
                query.assign(satisfaction=1), source, labels, _spec()
            )
        with self.assertRaisesRegex(ValueError, "predictor schema exactly"):
            compute_source_neighbor_probabilities(
                query, source.assign(id=1), labels, _spec()
            )

    def test_feature_append_is_numeric_and_does_not_mutate_inputs(self) -> None:
        source, labels = _source()
        query = source.iloc[:2].copy()
        original = query.copy(deep=True)
        probabilities = compute_source_neighbor_probabilities(
            query, source, labels, _spec()
        )
        extended, extended_spec = append_source_neighbor_feature(
            query, probabilities, _spec()
        )
        pd.testing.assert_frame_equal(query, original)
        self.assertEqual(
            list(extended.columns),
            list(_spec().feature_columns) + [SOURCE_NEIGHBOR_FEATURE],
        )
        self.assertIn(SOURCE_NEIGHBOR_FEATURE, extended_spec.numeric_columns)
        self.assertNotIn("id", extended.columns)
        self.assertNotIn("satisfaction", extended.columns)


if __name__ == "__main__":
    unittest.main()
