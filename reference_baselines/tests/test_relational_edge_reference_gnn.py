"""Unit tests for the relational edge-feature experiment (no large data needed)."""

import unittest

import numpy as np
from scipy import sparse
import torch

from reference_baselines.datasets.relational_edge_reference_graph import (
    EDGE_FEATURES,
    cooccurrence_pmi,
    hull_compatibility,
    soap_distance,
    undirected_pair_plan,
)
from reference_baselines.models.edge_attribute_reference_gnn import (
    EdgeAttributeReferenceGNN,
    EdgeReadoutReferenceModel,
)
from reference_baselines.models.reference_gnn import ReferenceGNN
from reference_baselines.models.relational_edge_reference_gnn import (
    EDGE_MODE,
    FEATURE_MASK,
    VARIANTS,
    apply_mask,
    build_model,
)


class FeatureMath(unittest.TestCase):
    def test_soap_distance_range(self):
        soap = np.eye(3, dtype=np.float64)  # orthonormal rows
        distance = soap_distance(soap, np.array([0, 0]), np.array([0, 1]))
        np.testing.assert_allclose(distance, [0.0, 1.0], atol=1e-9)

    def test_cooccurrence_pmi_scaling(self):
        frequency = np.array([10.0, 10.0, 10.0])
        value = cooccurrence_pmi(np.array([5.0]), frequency, 100.0, np.array([0]), np.array([1]))  # c_ij = 5
        self.assertAlmostEqual(float(value[0]), np.log(5.0) / 5.0, places=6)
        self.assertLessEqual(abs(float(value[0])), 1.0)

    def test_hull_lookup(self):
        hull = sparse.csr_matrix(([1.0], ([0], [1])), shape=(3, 3))
        hull = hull.maximum(hull.T)
        value = hull_compatibility(hull.tocsr(), np.array([0, 0]), np.array([1, 2]))
        np.testing.assert_array_equal(value, [1.0, 0.0])

    def test_empty_edges(self):
        self.assertEqual(
            hull_compatibility(sparse.csr_matrix((3, 3)), np.array([], int), np.array([], int)).size, 0
        )


class ShuffledControl(unittest.TestCase):
    def test_reverse_edges_share_a_pair(self):
        # (0,1) and (1,0) are the same undirected pair; (2,2) is a self-loop pair.
        edge_i = np.array([0, 1, 2, 3, 4])
        edge_j = np.array([1, 0, 2, 4, 3])
        pair_id, pair_rep, pair_perm = undirected_pair_plan(edge_i, edge_j, seed=0)
        self.assertEqual(pair_id[0], pair_id[1])          # (0,1) == (1,0)
        self.assertEqual(pair_id[3], pair_id[4])          # (3,4) == (4,3)
        self.assertNotEqual(pair_id[0], pair_id[2])       # distinct pairs stay distinct
        self.assertEqual(len(pair_rep), 3)                # {0-1, 2-2, 3-4}
        self.assertEqual(sorted(pair_perm.tolist()), [0, 1, 2])

    def test_shuffle_keeps_directions_symmetric(self):
        # Reassigning both directions through the pair plan lands on the same endpoints.
        edge_i = np.array([0, 1, 2, 3, 4])
        edge_j = np.array([1, 0, 2, 4, 3])
        pair_id, pair_rep, pair_perm = undirected_pair_plan(edge_i, edge_j, seed=7)
        rep_fwd = pair_rep[pair_perm[pair_id[0]]]  # (0,1)
        rep_rev = pair_rep[pair_perm[pair_id[1]]]  # (1,0)
        self.assertEqual(rep_fwd, rep_rev)

    def test_standardize_centers_and_scales(self):
        feats = np.array([[0.0, 2.0], [2.0, 4.0]])
        mean = feats.mean(axis=0)
        std = np.where(feats.std(axis=0) < 1e-6, 1.0, feats.std(axis=0))
        standardized = (feats - mean) / std
        np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(standardized.std(axis=0), [1.0, 1.0], atol=1e-9)


class VariantWiring(unittest.TestCase):
    def test_variant_tables_are_consistent(self):
        self.assertEqual(set(VARIANTS), set(FEATURE_MASK))
        self.assertEqual(set(VARIANTS), set(EDGE_MODE))
        self.assertEqual(len(FEATURE_MASK["rich"]), len(EDGE_FEATURES))

    def test_apply_mask_zeros_unused_slots(self):
        batch = {"edge_attr": torch.ones(4, 3)}
        apply_mask(batch, "geom")
        np.testing.assert_array_equal(batch["edge_attr"].numpy(), np.tile([1.0, 0.0, 0.0], (4, 1)))

    def test_build_model_routing(self):
        self.assertIsInstance(build_model("scalar", "gcn"), ReferenceGNN)
        self.assertIsInstance(build_model("readout", "readout"), EdgeReadoutReferenceModel)
        self.assertIsInstance(build_model("rich", "gat"), EdgeAttributeReferenceGNN)
        with self.assertRaises(ValueError):
            build_model("readout", "gcn")
        with self.assertRaises(ValueError):
            build_model("not_a_variant", "gcn")


if __name__ == "__main__":
    unittest.main()
