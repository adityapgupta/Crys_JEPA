"""Controls for attributing reference-pool and weight effects separately."""

import unittest

import numpy as np

from reference_baselines.baselines.mean_ablation import aggregate, gate_ranks, pool_ids


class MeanAblationTests(unittest.TestCase):
    def test_plain_mean_is_mean_distance_not_distance_to_centroid(self):
        query = np.array([0.0])
        references = np.array([[-2.0], [2.0]])
        distance = np.square(references - query).sum(1)[None, :]
        self.assertEqual(aggregate(distance)[0], 4.0)
        self.assertEqual(np.square(references.mean(0) - query).sum(), 0.0)

    def test_weighting_uses_normalized_distance_weights(self):
        d = np.array([[1.0, 11.0]])
        expected = (1.0 + 11.0 * np.exp(-1)) / (1 + np.exp(-1))
        self.assertAlmostEqual(aggregate(d, 10)[0], expected)
        self.assertLess(aggregate(d, 10)[0], aggregate(d)[0])

    def test_equal_distances_and_large_temperature_recover_uniform_mean(self):
        np.testing.assert_allclose(aggregate([[4, 4]], 10), [4])
        np.testing.assert_allclose(
            aggregate([[1, 3, 10]], 1e12), aggregate([[1, 3, 10]])
        )

    def test_union_does_not_double_weight_an_existing_neighbour(self):
        np.testing.assert_array_equal(pool_ids([1, 2, 7], [2, 7, 9]), [1, 2, 7, 9])

    def test_gate_keeps_mean_order_within_each_group(self):
        rows = np.array([4, 2, 9, 7])
        scores = np.array([1.0, 2.0, 0.5, 2.0])
        gated = np.array([False, True, False, True])
        rank = gate_ranks(rows, scores, gated)
        np.testing.assert_array_equal(rows[np.argsort(rank)], [2, 7, 9, 4])

    def test_invalid_pool_or_temperature_is_rejected(self):
        for d in [[[]], [[np.nan]], [[-1.0]]]:
            with self.assertRaises(ValueError):
                aggregate(d)
        for t in [0, -1, np.inf]:
            with self.assertRaises(ValueError):
                aggregate([[1, 2]], t)


if __name__ == "__main__":
    unittest.main()
