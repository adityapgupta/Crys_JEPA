"""Unit checks for the clean reference-utility experiment.

These tests use small synthetic inputs. They do not read reserved outcomes.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from reference_baselines.datasets import decomposition
from reference_baselines.datasets.reference_graph import (
    assert_split_isolation,
    collate,
)
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.graphs.cooccurrence import (
    cooccurrence_graph,
    graph_stats,
    permuted_graph,
)
from reference_baselines.models.reference_gnn import (
    GATLayer,
    GCNLayer,
    ReferenceGNN,
    reference_loss,
)


class DecompositionTests(unittest.TestCase):
    def test_entry_suffixes_are_removed(self):
        for value in ("mp-123-GGA", "mp-123-GGA+U", "mp-123-R2SCAN"):
            with self.subTest(value=value):
                self.assertEqual(decomposition.strip_entry_id(value), "mp-123")
        self.assertEqual(decomposition.strip_entry_id("mp-123"), "mp-123")

    def test_targets_map_suffixes_and_report_missing_phases(self):
        table = pd.DataFrame(
            {
                "candidate_row": [0, 1],
                "decomposition": [
                    json.dumps(
                        [
                            {"entry_id": "mp-1-GGA+U", "formula": "Li"},
                            {"entry_id": "mp-2-GGA", "formula": "O2"},
                            {"entry_id": "mp-9-GGA", "formula": "Na"},
                        ]
                    ),
                    "[]",
                ],
            }
        )
        index = {"mp-1": 3, "mp-2": 8}
        positives, unmapped, counts = decomposition.build_targets(
            [0, 1], table=table, index=index
        )
        np.testing.assert_array_equal(positives[0], [3, 8])
        self.assertEqual(unmapped[0], ["mp-9"])
        self.assertEqual(counts[0], 3)
        self.assertFalse(positives[1].size)
        stats = decomposition.target_stats(
            [0, 1], positives, unmapped, counts
        )
        self.assertEqual(stats["candidates_with_any_mapped_phase"], 1)
        self.assertEqual(stats["unmapped_phase_count"], 1)
        self.assertFalse(stats["candidate_stability_labels_read"])

    def test_material_index_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "embeddings/data").mkdir(parents=True)
            pd.DataFrame({"material_id": ["mp-1", "mp-1"]}).to_csv(
                root / "embeddings/data/metadata.csv", index=False
            )
            with mock.patch.object(decomposition, "ROOT", root):
                with self.assertRaises(ValueError):
                    decomposition.material_index()

    def test_decomposition_table_reads_labels_free_file(self):
        table = pd.DataFrame({"candidate_row": [0], "decomposition": ["[]"]})
        with mock.patch.object(
            decomposition.pd, "read_csv", return_value=table
        ) as read_csv:
            decomposition.decomposition_table()
        path = Path(read_csv.call_args.args[0])
        self.assertEqual(path.name, "candidate_decompositions.csv")

    def test_training_and_graph_code_do_not_name_candidate_labels(self):
        package = Path(decomposition.__file__).resolve().parents[1]
        for name in (
            "datasets/decomposition.py",
            "datasets/reference_graph.py",
            "graphs/build_reference_graph.py",
            "training/train_reference_gnn.py",
        ):
            with self.subTest(name=name):
                text = (package / name).read_text()
                self.assertNotIn("candidate_labels.csv", text)
        evaluator = (package / "evaluation/reference_gnn.py").read_text()
        self.assertIn("candidate_labels.csv", evaluator)


class SplitTests(unittest.TestCase):
    def test_legacy_test_rows_are_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            from reference_baselines.datasets import reference_graph

            results = Path(folder) / "results"
            split_dir = results / "gnn_comparison"
            split_dir.mkdir(parents=True)
            np.save(results / "test_candidate_rows.npy", np.array([2, 3]))
            np.save(split_dir / "reserved_rows.npy", np.array([9]))
            np.save(split_dir / "validation_rows.npy", np.array([4]))
            with (
                mock.patch.object(reference_graph, "RESULTS_DIR", results),
                mock.patch.object(reference_graph, "OUT", split_dir),
            ):
                with self.assertRaises(ValueError):
                    assert_split_isolation("train", np.array([1, 3]))

    def test_reserved_rows_are_blocked_from_train(self):
        with tempfile.TemporaryDirectory() as folder:
            from reference_baselines.datasets import reference_graph

            results = Path(folder) / "results"
            split_dir = results / "gnn_comparison"
            split_dir.mkdir(parents=True)
            np.save(split_dir / "reserved_rows.npy", np.array([9, 10]))
            np.save(split_dir / "validation_rows.npy", np.array([4]))
            with (
                mock.patch.object(reference_graph, "RESULTS_DIR", results),
                mock.patch.object(reference_graph, "OUT", split_dir),
            ):
                with self.assertRaises(ValueError):
                    assert_split_isolation("train", np.array([1, 9]))


class GraphTests(unittest.TestCase):
    def test_cooccurrence_is_symmetric_and_has_self_loops(self):
        graph, counts = cooccurrence_graph({0: np.array([1, 2]), 1: np.array([1, 2])}, 4)
        self.assertIsInstance(graph, sparse.csr_matrix)
        np.testing.assert_array_equal(graph.toarray(), graph.toarray().T)
        np.testing.assert_array_equal(graph.diagonal(), np.ones(4))
        self.assertEqual(counts[(1, 2)], 2)
        self.assertAlmostEqual(graph[1, 2], np.log1p(2))
        stats = graph_stats(graph, "fixture")
        self.assertEqual(stats["undirected_nonself_edges"], 1)
        self.assertEqual(stats["nodes_with_nonself_edge"], 2)

    def test_permutation_preserves_degree_sequence(self):
        graph, _ = cooccurrence_graph(
            {0: np.array([0, 1, 2]), 1: np.array([1, 2, 3])}, 4
        )
        other, permutation = permuted_graph(graph, 7)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation))
        np.testing.assert_array_equal(other.toarray(), graph.toarray()[inverse][:, inverse])
        np.testing.assert_array_equal(
            np.sort(np.asarray(other.sum(1)).ravel()),
            np.sort(np.asarray(graph.sum(1)).ravel()),
        )


class ModelTests(unittest.TestCase):
    def batch(self, references=5):
        rng = np.random.default_rng(2)
        x = torch.tensor(rng.normal(size=(references, 612)), dtype=torch.float32)
        edge = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        return {
            "x": x,
            "query": torch.tensor(rng.normal(size=(2, 612)), dtype=torch.float32),
            "pair": torch.tensor(rng.normal(size=(references, 3)), dtype=torch.float32),
            "edge": edge,
            "weight": torch.ones(references),
            "target": torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
            "has_target": torch.tensor([True, True]),
            "log_prior": torch.zeros(references),
            "distance": torch.arange(references, dtype=torch.float32),
            "owner": torch.tensor([0, 1, 0, 1, 0]),
        }

    def test_models_return_finite_gradients(self):
        for kind in ("gcn", "gat", "sage", "no_graph"):
            with self.subTest(kind=kind):
                model = ReferenceGNN(kind, hidden=8, layers=1)
                batch = self.batch()
                logits, weights, score = model(batch)
                self.assertEqual(logits.shape, (5,))
                self.assertEqual(weights.shape, (5,))
                self.assertEqual(score.shape, (2,))
                self.assertTrue(torch.isfinite(score).all())
                loss = reference_loss(logits, batch)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_uniform_weights_ignore_reference_logits(self):
        batch = self.batch()
        model = ReferenceGNN("no_graph", hidden=8, layers=1).eval()
        with torch.inference_mode():
            _, weights, score = model(batch, uniform_weights=True)
        expected = torch.tensor([(0 + 2 + 4) / 3, (1 + 3) / 2])
        np.testing.assert_allclose(score.numpy(), expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            weights.numpy(), [1 / 3, 1 / 2, 1 / 3, 1 / 2, 1 / 3], rtol=1e-6
        )

    def test_reference_order_is_invariant(self):
        batch = self.batch()
        order = torch.tensor([2, 0, 3, 1, 4])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order))
        other = dict(batch)
        for key in ("x", "pair", "target", "log_prior", "distance", "owner"):
            other[key] = batch[key][order]
        other["edge"] = inverse[batch["edge"]]
        model = ReferenceGNN("gcn", hidden=8, layers=1).eval()
        with torch.inference_mode():
            _, _, score = model(batch)
            _, _, shuffled = model(other)
        np.testing.assert_allclose(score.numpy(), shuffled.numpy(), rtol=1e-5, atol=1e-6)

    def test_self_only_keeps_only_self_edges(self):
        batch = self.batch()
        batch["edge"] = torch.tensor([[0, 1, 2], [0, 1, 3]])
        batch["weight"] = torch.ones(3)
        model = ReferenceGNN("gcn", hidden=8, layers=1).eval()
        with torch.inference_mode():
            _, _, score = model(batch, self_only=True)
        self.assertTrue(torch.isfinite(score).all())

    def test_empty_pool_is_rejected(self):
        from reference_baselines.datasets.reference_graph import build_table

        with (
            mock.patch(
                "reference_baselines.datasets.reference_graph.split_rows",
                return_value=np.array([0]),
            ),
            mock.patch(
                "reference_baselines.datasets.reference_graph.natural_pool",
                return_value=np.array([], dtype=np.int64),
            ),
            mock.patch(
                "reference_baselines.datasets.reference_graph.decomposition_targets",
                return_value={0: np.array([], dtype=np.int64)},
            ),
            mock.patch(
                "reference_baselines.datasets.reference_graph.load_graph",
                return_value={},
            ),
        ):
            with self.assertRaises(ValueError):
                build_table("train")


class EvaluationTests(unittest.TestCase):
    def test_unique_order_removes_duplicate_structures(self):
        adjacency = {1: {2}, 2: {1}}
        self.assertEqual(unique_order([1, 2, 1, 3], adjacency).tolist(), [1, 3])


def _toy_dataset(rows, distances):
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.rows = np.asarray(rows, dtype=np.int64)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = int(self.rows[index])
            values = distances[row]
            return {
                "row": row,
                "reference_ids": np.arange(len(values)),
                "x": torch.zeros(len(values), 612),
                "query": torch.zeros(612),
                "pair": torch.zeros(len(values), 3),
                "edge": torch.zeros(2, 0, dtype=torch.long),
                "weight": torch.zeros(0),
                "target": torch.zeros(len(values)),
                "has_target": False,
                "log_prior": torch.zeros(len(values)),
                "distance": torch.tensor(values, dtype=torch.float32),
            }

    return ToyDataset()


class ControlTests(unittest.TestCase):
    def test_uniform_pool_scores_returns_equal_weight_mean(self):
        from reference_baselines.evaluation.reference_gnn import uniform_pool_scores

        dataset = _toy_dataset([0, 1], {0: [0.0, 2.0, 4.0], 1: [1.0, 3.0]})
        rows, scores = uniform_pool_scores(dataset, {"batch_size": 2})
        np.testing.assert_array_equal(rows, [0, 1])
        np.testing.assert_allclose(scores, [2.0, 2.0], rtol=1e-6)

    def test_prior_weights_ignore_learned_logits(self):
        batch = ModelTests().batch()
        batch["log_prior"] = torch.zeros(5)
        first = ReferenceGNN("no_graph", hidden=8, layers=1).eval()
        second = ReferenceGNN("no_graph", hidden=8, layers=1).eval()
        with torch.no_grad():
            second.readout[-1].weight.normal_(0, 100)
            second.readout[-1].bias.normal_(0, 100)
        with torch.inference_mode():
            logits_a, _, score_a = first(batch, prior_weights=True)
            logits_b, _, score_b = second(batch, prior_weights=True)
        self.assertFalse(torch.allclose(logits_a, logits_b))
        np.testing.assert_allclose(score_a.numpy(), score_b.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(score_a.numpy(), [2.0, 2.0], rtol=1e-6)


class SmokeModeTests(unittest.TestCase):
    def run_main(self, argv):
        from reference_baselines.evaluation import reference_gnn

        with mock.patch.object(sys, "argv", ["reference_gnn", *argv]):
            reference_gnn.main()

    def test_smoke_refuses_official_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "reserved"
            with (
                mock.patch(
                    "reference_baselines.evaluation.reference_gnn.DEFAULT_OUT", out
                ),
                mock.patch(
                    "reference_baselines.evaluation.reference_gnn.prepare_table"
                ) as prepare,
            ):
                with self.assertRaises(SystemExit):
                    self.run_main(["--smoke", "--prepare", "--out-dir", str(out)])
            prepare.assert_not_called()

    def test_reuse_duplicates_skips_structure_matching(self):
        from reference_baselines.evaluation import reference_gnn

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "reserved"
            table = Path(folder) / "reserved_duplicate_pairs.csv"
            table.write_text("candidate_row_a,candidate_row_b\n")
            pool = reference_gnn.RESULTS_DIR / "reference_gnn/pools/fixture.npz"
            with (
                mock.patch.object(reference_gnn, "EXPERIMENT", Path(folder)),
                mock.patch.object(
                    reference_gnn, "prepare_table", return_value=pool
                ),
                mock.patch.object(reference_gnn, "build_duplicates") as build,
            ):
                self.run_main(
                    [
                        "--prepare",
                        "--reuse-duplicates",
                        "--out-dir",
                        str(out),
                    ]
                )
                record = json.loads((out / "prepare.json").read_text())
            build.assert_not_called()
            self.assertEqual(record["reserved_duplicates"]["status"], "reused")

    def test_labels_are_not_opened_before_checkpoints_freeze(self):
        from reference_baselines.evaluation import reference_gnn

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "reserved"
            out.mkdir()
            (out / "prepare.json").write_text("{}")
            with (
                mock.patch.object(
                    reference_gnn,
                    "frozen_checkpoints",
                    side_effect=FileNotFoundError("Incomplete run"),
                ),
                mock.patch.object(reference_gnn.pd, "read_csv") as read_csv,
                mock.patch.object(reference_gnn, "split_rows") as split_rows,
            ):
                with self.assertRaises(FileNotFoundError):
                    self.run_main(["--evaluate", "--out-dir", str(out)])
            read_csv.assert_not_called()
            split_rows.assert_not_called()

    def test_smoke_evaluation_uses_validation_and_emits_controls(self):
        from reference_baselines.evaluation import reference_gnn

        rows = np.arange(200, dtype=np.int64)
        labels = pd.DataFrame(
            {
                "candidate_row": rows,
                "hull_status": "ok",
                "e_above_hull_proxy": 0.0,
            }
        )
        manifest = pd.DataFrame(
            {"candidate_row": rows, "chemical_system": "Li-O"}
        )

        def read_csv(path, *args, **kwargs):
            name = Path(path).name
            if name == "candidate_labels.csv":
                return labels.copy()
            if name == "split_manifest.csv":
                return manifest.copy()
            raise AssertionError(f"Unexpected CSV read: {path}")

        def fake_learned_scores(loaded, dataset, cfg, **kwargs):
            return np.asarray(dataset.rows), {
                key: np.zeros(len(dataset.rows)) for key in loaded
            }

        def fake_checkpoints(cfg, graph, root, models, seeds):
            return {
                ("gcn", seeds[0]): {
                    "model": object(),
                    "sha256": "fixture",
                    "epoch": 1,
                    "validation_bce": 0.5,
                }
            }

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "smoke"
            out.mkdir()
            (out / "prepare.json").write_text("{}")
            with (
                mock.patch.object(reference_gnn.pd, "read_csv", side_effect=read_csv),
                mock.patch.object(
                    reference_gnn, "split_rows", return_value=rows
                ) as split_rows,
                mock.patch.object(
                    reference_gnn, "validation_adjacency", return_value={}
                ),
                mock.patch.object(
                    reference_gnn,
                    "fixed_score_tables",
                    return_value={"original_crys_jepa": np.zeros(len(rows))},
                ),
                mock.patch.object(
                    reference_gnn, "frozen_checkpoints", side_effect=fake_checkpoints
                ),
                mock.patch.object(
                    reference_gnn, "learned_scores", side_effect=fake_learned_scores
                ),
                mock.patch.object(
                    reference_gnn,
                    "uniform_pool_scores",
                    side_effect=lambda dataset, cfg: (
                        np.asarray(dataset.rows),
                        np.zeros(len(dataset.rows)),
                    ),
                ),
                mock.patch.object(
                    reference_gnn,
                    "PoolDataset",
                    side_effect=lambda *args, **kwargs: _toy_dataset(
                        rows, {int(row): [0.0] for row in rows}
                    ),
                ),
                mock.patch.object(reference_gnn, "sha256", return_value="fixture"),
            ):
                self.run_main(
                    ["--smoke", "--evaluate", "--out-dir", str(out)]
                )
                record = json.loads((out / "evaluation.json").read_text())
            metrics = pd.read_csv(out / "metrics.csv")
            split_rows.assert_called_with("validation")
            self.assertEqual(record["split"], "validation")
            self.assertIn("random_references", record["methods"])
            self.assertIn("uniform_natural_pool", record["methods"])
            self.assertIn("formula_prior_natural_pool", record["methods"])
            self.assertIn("random_references", set(metrics.method))


if __name__ == "__main__":
    unittest.main()
