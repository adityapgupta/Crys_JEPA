"""Numerical controls for label-free reference retrieval and spectral filters."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.linalg import expm

from reference_baselines.baselines.angular import masked_mean
from reference_baselines.baselines.conductance import prefix_conductance
from reference_baselines.core.angular import unit_vectors, within_angle
from reference_baselines.core.retrieval import exact_knn
from reference_baselines.graphs.diffusion import (
    heat_features,
    heat_seed_distribution,
    resolvent_features,
    score_moments,
    transition_matrix,
)


class NonparametricTests(unittest.TestCase):
    def setUp(self):
        self.graph = sparse.csr_matrix(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.p = transition_matrix(self.graph)
        self.refs = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]])
        self.features = np.column_stack([self.refs, np.square(self.refs).sum(1)])

    def test_cosine_scale_invariance_and_ties(self):
        refs = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        q = unit_vectors(np.array([[5.0, 0.0]]))
        r = unit_vectors(refs)
        ids, d = exact_knn(q, r, 2, batch_size=1)
        self.assertEqual(ids.tolist(), [[0, 1]])
        np.testing.assert_allclose(d, 0.0, atol=1e-14)
        with self.assertRaises(ValueError):
            unit_vectors([[0.0, 0.0]])

    def test_theta_boundary_and_empty_pool(self):
        cosdist = np.array([[0.0, 1.0, 2.0]])
        np.testing.assert_array_equal(within_angle(cosdist, 90), [[True, True, False]])
        self.assertTrue(
            np.isnan(masked_mean(cosdist, np.zeros_like(cosdist, dtype=bool)))[0]
        )

    def test_resolvent_matches_dense_reference_distribution(self):
        alpha = 0.85
        kernel = np.linalg.solve(
            np.eye(3) - alpha * self.p.toarray(), (1 - alpha) * np.eye(3)
        )
        actual, diagnostic = resolvent_features(
            self.p, self.features, alpha, 1e-10, 500
        )
        np.testing.assert_allclose(actual, kernel @ self.features, atol=1e-10)
        self.assertTrue(diagnostic["converged"])
        query = np.array([[0.5, 0.1]])
        expected = np.sum(kernel[0] * np.square(self.refs - query).sum(1))
        np.testing.assert_allclose(
            score_moments(query, actual[:1]), expected, atol=1e-10
        )
        # Known-node diffusion produces nonnegative reference weights summing to one.
        self.assertGreaterEqual(kernel.min(), 0.0)
        np.testing.assert_allclose(kernel.sum(1), 1.0)

    def test_heat_matches_matrix_exponential(self):
        kernel = expm(3 * (self.p.toarray() - np.eye(3)))
        actual, diagnostic = heat_features(self.p, self.features, 3, 1e-10, 100)
        np.testing.assert_allclose(actual, kernel @ self.features, atol=1e-10)
        self.assertTrue(diagnostic["converged"])
        seed = np.array([1.0, 0.0, 0.0])
        weights, bound = heat_seed_distribution(self.p, seed, 3, tolerance=1e-10)
        np.testing.assert_allclose(weights, kernel.T @ seed, atol=1e-10)
        self.assertLessEqual(bound["probability_l1_error_bound"], 1e-10)

    def test_disconnected_node_retains_its_features(self):
        p = transition_matrix(sparse.csr_matrix((3, 3)))
        np.testing.assert_array_equal(p.toarray(), np.eye(3))
        for operator, parameter in [(resolvent_features, 0.5), (heat_features, 3)]:
            actual, _ = operator(p, self.features, parameter)
            np.testing.assert_allclose(actual, self.features, atol=1e-10)

    def test_zero_diffusion_recovers_seed_distance(self):
        actual, _ = resolvent_features(self.p, self.features, 0.0)
        np.testing.assert_array_equal(actual, self.features)

    def test_conductance_matches_explicit_cut(self):
        degree = np.array([1.0, 2.0, 1.0])
        actual = prefix_conductance(self.graph, degree, np.array([0, 1, 2]), [1, 2])
        # Both sets have cut weight 1 and smaller-side volume 1.
        np.testing.assert_allclose(actual, [1.0, 1.0])

    def test_eigenmode_cluster_solvers_on_small_graph(self):
        from reference_baselines.graphs import eigenmodes

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            graph = sparse.diags(
                [np.ones(19), np.ones(19)], [-1, 1], shape=(20, 20)
            ).tocsr()
            sparse.save_npz(path / "reference_graph_k10.npz", graph)
            with (
                patch.object(eigenmodes, "RESULTS_DIR", path),
                patch.object(eigenmodes, "BASELINE_RESULTS_DIR", path),
            ):
                for solver in ["shift-invert", "iterative-shift"]:
                    eigenmodes.solve("euclidean", 3, solver)
                    report = json.loads(
                        (path / "euclidean_eigenmodes.json").read_text()
                    )
                    self.assertEqual(report["status"], "passed", report)
                    self.assertLess(max(report["residual_norms"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
