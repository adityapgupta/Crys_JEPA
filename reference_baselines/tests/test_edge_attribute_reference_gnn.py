"""Synthetic checks for the fixed-topology edge-attribute experiment."""

import unittest
from pathlib import Path

import numpy as np
import torch

from reference_baselines.datasets.edge_attribute_reference_graph import (
    edge_attributes,
    electronegativity_profiles,
)
from reference_baselines.models.reference_gnn import ReferenceGNN, reference_loss
from reference_baselines.models.edge_attribute_reference_gnn import build_model


class EdgeFeatureTests(unittest.TestCase):
    def test_feature_and_training_code_do_not_open_candidate_outcomes(self):
        package = Path(__file__).resolve().parents[1]
        for name in (
            "datasets/edge_attribute_reference_graph.py",
            "models/edge_attribute_reference_gnn.py",
            "training/train_edge_attribute_reference_gnn.py",
        ):
            with self.subTest(name=name):
                text = (package / name).read_text()
                self.assertNotIn("candidate_labels.csv", text)
                self.assertNotIn("e_above_hull_proxy", text)

    def test_electronegativity_profile_tracks_mean_spread_and_missing_mass(self):
        composition = np.array(
            [
                [0.0, 0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        en = np.array([np.nan, 1.0, 3.0, np.nan])
        profile = electronegativity_profiles(composition, en)
        self.assertEqual(profile.shape, (3, 3))
        self.assertTrue(np.isfinite(profile).all())
        self.assertGreater(profile[0, 1], profile[1, 1])
        self.assertGreater(profile[2, 2], profile[0, 2])

    def test_edge_features_have_expected_self_and_cross_values(self):
        nodes = np.array([[1.0, 0.0], [0.0, 1.0]])
        edge = np.array([[0, 0, 1], [0, 1, 1]])
        attrs = edge_attributes(
            nodes,
            edge,
            np.array([225, 62]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(attrs[[0, 2]], [[1, 1, 0], [1, 1, 0]])
        self.assertEqual(attrs[1, 0], 0)
        self.assertEqual(attrs[1, 1], 0)
        self.assertGreater(attrs[1, 2], 0)


class RichModelTests(unittest.TestCase):
    @staticmethod
    def batch():
        rng = np.random.default_rng(4)
        references = 5
        return {
            "x": torch.tensor(rng.normal(size=(references, 612)), dtype=torch.float32),
            "query": torch.tensor(rng.normal(size=(2, 612)), dtype=torch.float32),
            "pair": torch.tensor(rng.normal(size=(references, 3)), dtype=torch.float32),
            "edge": torch.tensor([[0, 1, 2, 3, 4, 0, 2], [0, 1, 2, 3, 4, 2, 0]]),
            "weight": torch.ones(7),
            "edge_attr": torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float32),
            "target": torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
            "has_target": torch.tensor([True, True]),
            "log_prior": torch.zeros(references),
            "distance": torch.arange(references, dtype=torch.float32),
            "owner": torch.tensor([0, 1, 0, 1, 0]),
        }

    def test_all_variants_return_finite_gradients(self):
        specs = [("scalar", kind) for kind in ("gcn", "gat", "sage")]
        specs += [("rich", kind) for kind in ("gcn", "gat", "sage")]
        specs += [("shuffled", kind) for kind in ("gcn", "gat", "sage")]
        specs += [("readout", "readout")]
        for variant, kind in specs:
            with self.subTest(variant=variant, kind=kind):
                model = build_model(variant, kind, hidden=8, layers=1)
                batch = self.batch()
                logits, weights, score = model(batch)
                self.assertEqual(logits.shape, (5,))
                self.assertTrue(torch.isfinite(weights).all())
                self.assertTrue(torch.isfinite(score).all())
                reference_loss(logits, batch).backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_scalar_variant_is_the_frozen_model(self):
        self.assertIsInstance(build_model("scalar", "gcn", 8, 1), ReferenceGNN)

    def test_rich_model_is_reference_order_invariant(self):
        batch = self.batch()
        order = torch.tensor([2, 0, 3, 1, 4])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order))
        other = dict(batch)
        for key in ("x", "pair", "target", "log_prior", "distance", "owner"):
            other[key] = batch[key][order]
        other["edge"] = inverse[batch["edge"]]
        model = build_model("rich", "gat", hidden=8, layers=1).eval()
        with torch.inference_mode():
            _, _, first = model(batch)
            _, _, second = model(other)
        np.testing.assert_allclose(first.numpy(), second.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
