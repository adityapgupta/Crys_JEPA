"""Graph isolation, readout normalization, gradients and test-set guards."""

import unittest
import numpy as np
import torch

from reference_baselines.datasets.gnn_comparison import (
    CosineReferenceDataset,
    reserve_system,
)
from reference_baselines.models.message_passing import ReferenceGNN
from reference_baselines.training.compare_gnns import evaluate_scores


def fixture():
    torch.manual_seed(4)
    return {
        "x": torch.randn(4, 612),
        "query": torch.randn(2, 612),
        "pair": torch.randn(4, 3),
        "log_prior": torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5])),
        "batch": torch.tensor([0, 0, 1, 1]),
        "edge": torch.tensor([[0, 1, 0, 1, 2, 3, 2, 3], [0, 1, 1, 0, 2, 3, 3, 2]]),
        "weight": torch.ones(8),
        "target": torch.tensor([1.0, 0.0]),
    }


class GNNComparisonTests(unittest.TestCase):
    def test_reserved_and_old_test_cannot_be_opened_by_training_loader(self):
        for split in ["reserved", "test"]:
            with self.assertRaises(ValueError):
                CosineReferenceDataset(split)

    def test_hash_allocation_is_deterministic(self):
        self.assertEqual(reserve_system("Cl-Na"), reserve_system("Cl-Na"))

    def test_all_parameters_receive_finite_gradients(self):
        for kind in ["gcn", "mpnn", "no_graph"]:
            batch = fixture()
            model = ReferenceGNN(kind, hidden=16)
            logits, weights = model(batch)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["target"]
            ).backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
            )
            torch.testing.assert_close(weights[:2].sum(), torch.tensor(1.0))
            torch.testing.assert_close(weights[2:].sum(), torch.tensor(1.0))

    def test_targets_do_not_affect_predictions_and_graphs_are_isolated(self):
        for kind in ["gcn", "mpnn"]:
            batch = fixture()
            model = ReferenceGNN(kind, hidden=16).eval()
            before, _ = model(batch)
            changed = dict(batch, target=1 - batch["target"])
            after, _ = model(changed)
            torch.testing.assert_close(before, after)
            changed = dict(batch, x=batch["x"].clone())
            changed["x"][2:] += 10
            after, _ = model(changed)
            torch.testing.assert_close(before[:1], after[:1])

    def test_reference_node_permutation_preserves_predictions(self):
        batch = fixture()
        perm = torch.tensor([3, 0, 2, 1])
        inverse = torch.argsort(perm)
        changed = dict(batch)
        for key in ["x", "pair", "batch", "log_prior"]:
            changed[key] = batch[key][perm]
        changed["edge"] = inverse[batch["edge"]]
        for kind in ["gcn", "mpnn"]:
            model = ReferenceGNN(kind, hidden=16).eval()
            a, _ = model(batch)
            b, _ = model(changed)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_fixed_budget_counts_and_greedy_duplicates(self):
        rows = np.arange(100)
        targets = np.zeros(100)
        targets[:5] = 1
        scores = np.arange(100, 0, -1)
        metric = evaluate_scores(rows, targets, scores, {0: {1}, 1: {0}})
        self.assertEqual(metric["passes_5"], 4)
        self.assertEqual(metric["yield_5"], 0.8)


if __name__ == "__main__":
    unittest.main()
