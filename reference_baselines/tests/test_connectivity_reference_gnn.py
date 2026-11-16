"""Unit tests for the connectivity experiment (no large data needed for most)."""

import unittest

import numpy as np
from scipy import sparse
import torch

from reference_baselines.datasets.reference_graph import GRAPH_NAMES
from reference_baselines.graphs import connectivity_adjacency as conn
from reference_baselines.evaluation.connectivity_reference_gnn import topologies
from reference_baselines.models.reference_gnn import KINDS, ReferenceGNN


def _synthetic_batch(pool_sizes=(2, 2), dim=612, edges=None):
    """A minimal batch matching collate()'s keys, for a forward-pass smoke test."""
    total = sum(pool_sizes)
    owner = torch.repeat_interleave(torch.arange(len(pool_sizes)), torch.tensor(pool_sizes))
    if edges is None:
        edge = torch.zeros(2, 0, dtype=torch.long)
        weight = torch.zeros(0)
    else:
        edge = torch.tensor(edges, dtype=torch.long)
        weight = torch.ones(edge.shape[1])
    torch.manual_seed(0)
    return {
        "query": torch.randn(len(pool_sizes), dim),
        "x": torch.randn(total, dim),
        "pair": torch.randn(total, 3),
        "edge": edge,
        "weight": weight,
        "owner": owner,
        "target": torch.randint(0, 2, (total,)).float(),
        "has_target": torch.ones(len(pool_sizes), dtype=torch.bool),
        "log_prior": torch.zeros(total),
        "distance": torch.rand(total),
    }


class GraphRegistration(unittest.TestCase):
    def test_new_topologies_registered(self):
        self.assertIn("soap", GRAPH_NAMES)
        self.assertIn("phase", GRAPH_NAMES)

    def test_topologies_order_and_dedup(self):
        cfg = {"graph_primary": "physical", "graph_ablations": ["cosine", "physical", "soap", "none"]}
        # physical appears first and only once, order otherwise preserved.
        self.assertEqual(topologies(cfg), ["physical", "cosine", "soap", "none"])

    def test_readout_kind_available(self):
        self.assertIn("no_graph", KINDS)


class AdjacencyHelpers(unittest.TestCase):
    def test_stats_counts_self_loops_separately(self):
        graph = sparse.csr_matrix(
            ([1.0, 1.0, 1.0], ([0, 0, 1], [0, 1, 1])), shape=(3, 3)
        )  # two self-loops (0,0),(1,1) and one off-diagonal (0,1)
        stats = conn.stats(graph, "x")
        self.assertEqual(stats["nodes"], 3)
        self.assertEqual(stats["edges_including_self"], 3)
        self.assertEqual(stats["off_diagonal_edges"], 1)


class ForwardPass(unittest.TestCase):
    def test_no_graph_readout_forward(self):
        model = ReferenceGNN("no_graph", hidden=16, layers=2).eval()
        with torch.inference_mode():
            logits, weights, score = model(_synthetic_batch())
        self.assertEqual(score.shape, (2,))
        self.assertTrue(torch.isfinite(score).all())

    def test_gnn_with_empty_edges_is_finite(self):
        # The 'none' topology yields empty edges; message passing must not crash.
        for kind in ("gcn", "gat", "sage"):
            model = ReferenceGNN(kind, hidden=16, layers=2).eval()
            with torch.inference_mode():
                _, _, score = model(_synthetic_batch(edges=None))
            self.assertTrue(torch.isfinite(score).all(), kind)

    def test_gnn_with_edges_is_finite(self):
        batch = _synthetic_batch(edges=[[0, 1, 2], [1, 0, 3]])
        for kind in ("gcn", "gat", "sage"):
            model = ReferenceGNN(kind, hidden=16, layers=2).eval()
            with torch.inference_mode():
                _, _, score = model(batch)
            self.assertTrue(torch.isfinite(score).all(), kind)


class RealGraphsIfPresent(unittest.TestCase):
    @unittest.skipUnless(conn.SOAP_PATH.exists(), "SOAP matrix not present")
    def test_soap_graph_symmetric_with_self_loops(self):
        graph = conn.soap_knn_graph()
        self.assertEqual(graph.shape, (conn.REFERENCE_COUNT, conn.REFERENCE_COUNT))
        self.assertEqual((graph != graph.T).nnz, 0)
        self.assertTrue((graph.diagonal() > 0).all())

    @unittest.skipUnless(conn.HULL_PATH.exists(), "Hull adjacency not present")
    def test_phase_graph_symmetric_binary_with_self_loops(self):
        graph = conn.phase_graph()
        self.assertEqual(graph.shape, (conn.REFERENCE_COUNT, conn.REFERENCE_COUNT))
        self.assertEqual((graph != graph.T).nnz, 0)
        self.assertTrue((graph.diagonal() > 0).all())
        self.assertLessEqual(float(graph.data.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
