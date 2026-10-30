"""Offline physics, retrieval, failure-policy and staged integration checks."""

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

from reference_baselines.core.thermodynamics import FixedHull
from reference_baselines.reference_set_comparison import generate_embeddings as export
from reference_baselines.reference_set_comparison.common import database, fingerprint, read_json, receipt, write_json
from reference_baselines.reference_set_comparison.hulls import label_candidates
from reference_baselines.reference_set_comparison.indexing import ExactIndex, score_reference_pools
from reference_baselines.reference_set_comparison.pipeline import execute
from reference_baselines.reference_set_comparison.proxies import load_megnet_weights, predict_with_failures
from reference_baselines.reference_set_comparison.phase_bank import lmdb_entries
from reference_baselines.reference_set_comparison.release import binary_mask
from reference_baselines.reference_set_comparison.reporting import success_metrics
from reference_baselines.reference_set_comparison.selection import make_selections


class ScientificChecks(unittest.TestCase):
    def test_megnet_normalizer_is_allowlisted_only_while_loading(self):
        """Exercise real torch serialization without installing MatGL or downloading weights."""
        import pickle
        normalizer = type("Normalizer", (), {"__module__": "matgl.data.transformer"})
        unexpected = type("Unexpected", (), {"__module__": "matgl.data.transformer"})
        matgl = ModuleType("matgl")
        data = ModuleType("matgl.data")
        transformer = ModuleType("matgl.data.transformer")
        matgl.data, data.transformer = data, transformer
        transformer.Normalizer, transformer.Unexpected = normalizer, unexpected
        modules = {"matgl": matgl, "matgl.data": data, "matgl.data.transformer": transformer}
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
            path = Path(directory) / "model.pt"
            instance = normalizer()
            instance.mean, instance.std = -1.65459073, 1.07266128
            torch.save({"target_transformer": instance}, path)
            def restricted_loader(folder):
                return torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=True)
            matgl.load_model = restricted_loader
            before = set(torch.serialization.get_safe_globals())
            with self.assertRaises(pickle.UnpicklingError):
                restricted_loader(directory)
            loaded = load_megnet_weights(directory)["target_transformer"]
            self.assertIsInstance(loaded, normalizer)
            self.assertEqual((loaded.mean, loaded.std), (instance.mean, instance.std))
            self.assertEqual(set(torch.serialization.get_safe_globals()), before)
            with self.assertRaises(pickle.UnpicklingError):
                restricted_loader(directory)
            torch.save({"unexpected": unexpected()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_megnet_weights(directory)
            self.assertEqual(set(torch.serialization.get_safe_globals()), before)

    def test_lmdb_reader_keeps_all_polymorphs_without_energy_pruning(self):
        import pickle
        records = {b"chemical_systems": pickle.dumps(["Li", "Li-O", "C"]),
                   b"Li.reduced_formulas": pickle.dumps(["Li"]), b"Li.Li.length": pickle.dumps(1),
                   b"Li.Li.0": pickle.dumps({"structure": {"tag": "unary"}}),
                   b"Li-O.reduced_formulas": pickle.dumps(["Li2O"]), b"Li-O.Li2O.length": pickle.dumps(2),
                   b"Li-O.Li2O.0": pickle.dumps({"structure": {"tag": "lower_dft"}, "energy": -10}),
                   b"Li-O.Li2O.1": pickle.dumps({"structure": {"tag": "higher_dft"}, "energy": 10})}
        class Transaction:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def get(self, key): return records.get(key)
        class Environment:
            def begin(self): return Transaction()
            def close(self): pass
        with patch.dict("sys.modules", {"lmdb": SimpleNamespace(open=lambda *a, **k: Environment())}):
            entries = list(lmdb_entries(Path("fixture"), {"Li", "Li-O"}))
        self.assertEqual(len(entries), 3)
        self.assertEqual({entry[3]["tag"] for entry in entries}, {"unary", "lower_dft", "higher_dft"})

    def test_legacy_mattersim_hull_preserves_threshold_and_energy(self):
        from reference_baselines.datasets import labels
        previous = labels.HULL
        try:
            labels.HULL = {}
            labels.init_hull([{"chemical_system": "Li", "formula": "Li", "entry_id": "Li", "energy_per_atom": 0.}])
            result = labels.hull_group(("Li", [{"candidate_row": 7, "formula": "Li", "corrected_energy_per_atom": .1}]))[0]
            self.assertAlmostEqual(result["e_above_hull_proxy"], .1)
            self.assertTrue(result["near_stable_proxy"])
        finally:
            labels.HULL = previous

    def test_binary_masks_reject_nonbinary_or_bad_shape(self):
        np.testing.assert_array_equal(binary_mask(np.array([1, 0]), 2, "vu"), [True, False])
        for value in ([1, 2], [[1, 0]], [True]):
            with self.assertRaises(ValueError):
                binary_mask(value, 2, "vu")

    def test_exact_ties_and_score_union_match_direct_reference_ids(self):
        from reference_baselines.core.retrieval import exact_knn
        refs = np.array([[0., 0.], [1., 0.], [-1., 0.], [2., 0.]])
        queries = np.array([[0., 0.], [0.5, 0.]])
        ids, distances = exact_knn(queries, refs, 2, batch_size=1)
        np.testing.assert_array_equal(ids[0], [0, 1])
        meta = pd.DataFrame({"chemical_system": ["Li", "O", "Na", "Li-O"]})
        candidates = pd.DataFrame({"chemical_system": ["Li-O", "Cl"]})
        scores, pools = score_reference_pools(queries, refs, meta, candidates, ids, distances, [1, 2])
        for i in range(2):
            for k in (1, 2):
                union = np.union1d(pools[candidates.iloc[i].chemical_system], ids[i, :k])
                direct = np.square(refs[union]-queries[i]).sum(1).mean()
                self.assertAlmostEqual(scores.iloc[i][f"union_{k}"], direct)
        self.assertTrue(np.isnan(scores.iloc[1].chemical))
        self.assertEqual(pools["Li-O"].tolist(), [0, 1, 3])

    def test_hull_balance_negative_distances_and_element_offsets_cancel(self):
        refs = [{"formula": "Li", "energy_per_atom": 0., "entry_id": "Li"},
                {"formula": "O2", "energy_per_atom": 0., "entry_id": "O"},
                {"formula": "Li2O", "energy_per_atom": -1.4, "entry_id": "Li2O"}]
        for use_pd in (True, False):
            energy, products, _ = FixedHull(["Li", "O"], refs, use_pd).minimum("LiO")
            self.assertAlmostEqual(energy, -1.05)
            self.assertAlmostEqual(-1.0-energy, .05)
            self.assertLess(-1.1-energy, 0.)
            weights = {p["formula"]: p["fraction"] for p in products}
            self.assertAlmostEqual(weights["Li2O"], .75)
        from pymatgen.core import Composition
        shifted = [{**r, "energy_per_atom": r["energy_per_atom"] + sum(Composition(r["formula"]).get_atomic_fraction(e)*v for e,v in {"Li": -3., "O": -5.}.items())} for r in refs]
        hull = FixedHull(["Li", "O"], shifted).minimum("LiO")[0]
        self.assertAlmostEqual((-1.-4.)-hull, .05)

    def test_incomplete_reference_hull_is_unknown_and_threshold_is_strict(self):
        candidates = pd.DataFrame([{"candidate_row": 0, "id": "c", "eval_folder": 20, "idx": "19", "turn": 2,
                                    "chemical_system": "Li-O", "formula": "LiO", "structure_hash": "h"}])
        phases = pd.DataFrame([{"id": "p1", "chemical_system": "Li", "formula": "Li"},
                               {"id": "p2", "chemical_system": "O", "formula": "O2"}])
        cp = pd.DataFrame([{"id": "c", "status": "ok", "energy_per_atom": -.2}])
        pp = pd.DataFrame([{"id": "p1", "status": "ok", "energy_per_atom": 0.},
                           {"id": "p2", "status": "prediction_error", "energy_per_atom": None}])
        labels, audits = label_candidates(candidates, phases, cp, pp, "formation_energy_per_atom")
        self.assertEqual(labels.iloc[0].hull_status, "incomplete_reference_hull")
        self.assertFalse(audits.iloc[0].complete)
        table = pd.DataFrame({"hull_status": ["ok", "ok", "error"], "e_above_hull_proxy": [.099, .1, np.nan], "chemical_system": ["Li", "O", "Li"]})
        stats = success_metrics(table, 4)
        self.assertEqual(stats["stable_count"], 1)
        self.assertEqual(stats["precision_evaluated"], .5)
        self.assertEqual(stats["stable_yield_lower_bound"], .25)
        self.assertEqual(stats["stable_yield_upper_bound"], .5)
        self.assertEqual(stats["exact_threshold_count"], 1)

    def test_unique_refill_shortfall_determinism_and_zero_budget(self):
        scores = pd.DataFrame({"candidate_row": range(4), "turn": [2]*4, "eval_folder": [20]*4, "chemical": [1., 0., np.nan, 2.]})
        adjacency = {0: {1, 3}, 1: {0, 3}, 3: {0, 1}}
        a, requests = make_selections(scores, ["chemical"], [.01, .5, 1.], adjacency)
        b, _ = make_selections(scores, ["chemical"], [.01, .5, 1.], adjacency)
        pd.testing.assert_frame_equal(a, b)
        unique = requests[(requests.method == "chemical") & (requests.policy == "unique_refill") & (requests.fraction == .5)]
        self.assertTrue((unique.selected_count == 1).all())
        self.assertTrue((requests.loc[requests.fraction.eq(.01), "budget"] == 0).all())

    def test_prediction_failure_isolated(self):
        class Predictor:
            def predict(self, structures):
                if "bad" in structures:
                    raise ValueError("unsupported element")
                return np.array([len(s) for s in structures])
        result = predict_with_failures(Predictor(), ["ok", "bad", "yes"])
        self.assertEqual([r[0] for r in result], ["ok", "prediction_error", "ok"])
        self.assertEqual([result[0][1], result[2][1]], [2, 3])


class PipelineIntegration(unittest.TestCase):
    def test_real_tiny_jepa_to_two_proxy_reports_resume_and_stale_rejection(self):
        """Real untrained JEPA, real structures/hulls, fake external energies; no network."""
        from components.jepa.frame.jepa import JEPA
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            structures = [Structure(Lattice.cubic(4+i*.2), ["Li", "Na"], [[0,0,0],[.2+i*.055,.4,.5]]) for i in range(8)]
            unary = [Structure(Lattice.cubic(3), [e], [[0,0,0]]) for e in ("Li", "Na")]
            references = unary + structures[:3]
            train = root / "train.csv"
            pd.DataFrame({"material_id": [f"mp-{i}" for i in range(5)], "cif": [s.to(fmt="cif") for s in references]}).to_csv(train, index=False)
            model_config = {"hidden_dim": 16, "layers": 1, "attn_head": 4, "dropout": 0.}
            config_path = root / "jepa.yml"
            config_path.write_text(json.dumps({"model": model_config}))
            torch.manual_seed(0)
            model = JEPA(SimpleNamespace(model=SimpleNamespace(**model_config), device="cpu"))
            checkpoint = root / "jepa.pt"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint)
            scaler = root / "scaler.pt"
            torch.save(torch.stack([torch.zeros(6), torch.ones(6)]), scaler)
            reference_export = root / "reference"
            export.run(argparse.Namespace(out_dir=reference_export, train_csv=train, artifact_dir=root,
                limit=None, scaler=scaler, checkpoint=checkpoint, config=config_path, device="cpu", batch_size=2))
            release = root / "release"
            for batch in (20, 21):
                folder = release / str(batch)
                folder.mkdir(parents=True)
                frame = pd.DataFrame({"e_relax": [-1.]*8, "num_sites": [2]*8,
                                      "structure": [s.as_dict() for s in structures], "exception": [False]*7+[True]},
                                     index=[9, 2, 17, 1, 18, 10, 15, 20])
                frame.to_json(folder / "relaxed_mattersim.json")
                torch.save({"vu": np.ones(8, dtype=bool), "vn": np.array([1]*6+[0, 1], dtype=bool)}, folder / "indicators_mattersim_mp_20.pt")
            hull_source = root / "fixture.lmdb"
            hull_source.write_bytes(b"phase input fixture")
            config = {"release_root": str(release), "reference_export": str(reference_export),
                      "checkpoint": str(checkpoint), "scaler": str(scaler), "jepa_config": str(config_path),
                      "output": str(root / "out"), "cache": str(root / "cache"), "hull_database": str(hull_source),
                      "start_batch": 20, "end_batch": 21, "neighbours": [1, 2], "fractions": [.01, .5],
                      "seed": 2026, "device": "cpu", "embedding_batch_size": 2, "prediction_batch_size": 2,
                      "retrieval_batch_size": 2, "bootstrap_repeats": 20,
                      "models": {name: {"backend": name, "directory": str(root/name), "repository": name} for name in ("megnet", "esen")}}
            for stage in ("prepare", "embed", "score", "select"):
                execute(config, stage)
            entries = [(f"{s.composition.chemical_system}.{i}", s.composition.chemical_system, s.composition.reduced_formula, s.as_dict()) for i,s in enumerate(references)]
            with patch("reference_baselines.reference_set_comparison.phase_bank.lmdb_entries", return_value=iter(entries)):
                execute(config, "phases")
            self.assertEqual(read_json(root / "out/phases.json")["phase_count"], 5)
            # Count all polymorphs, including the three same-formula binary references.
            self.assertEqual(read_json(root / "out/phases.json")["systems"]["Li-Na"], 3)
            class FakeAdapter:
                def __init__(self, spec, device):
                    pass
                def predict(self, items):
                    return np.array([-.2 if len(s) > 1 else 0. for s in items])
            with patch("reference_baselines.reference_set_comparison.proxies.model_identity", side_effect=lambda c,n: {"model": n}), \
                 patch("reference_baselines.reference_set_comparison.proxies.MEGNetAdapter", FakeAdapter), \
                 patch("reference_baselines.reference_set_comparison.proxies.ESENAdapter", FakeAdapter):
                for name in ("megnet", "esen"):
                    write_json(root / f"out/{name}_smoke.json", {"passed": True, "fingerprint": fingerprint({"model": name})})
                    self.assertFalse(execute(config, "predict", name, max_items=2))
                    self.assertTrue(execute(config, "predict", name))
                    with patch.object(FakeAdapter, "predict", side_effect=AssertionError("cache miss")):
                        self.assertTrue(execute(config, "predict", name))
                    execute(config, "hull", name)
                    # The completion marker must detect interrupted or modified CSV exports.
                    prediction_path = root / f"out/{name}_phase_predictions.csv"
                    with prediction_path.open("a") as stream:
                        stream.write("corrupt")
                    with self.assertRaisesRegex(ValueError, "Interrupted or changed prediction export"):
                        execute(config, "hull", name)
                    execute(config, "predict", name)
                    execute(config, "hull", name)
                    if name == "megnet":
                        # A completed MEGNet report must not wait for eSEN, which has no predictions/hull yet.
                        execute(config, "report", model="megnet")
                        report_dir = root / "out/reports/megnet"
                        metrics = pd.read_csv(report_dir / "metrics.csv")
                        self.assertEqual(set(metrics.proxy), {"megnet"})
                        self.assertEqual(set(read_json(report_dir / "provenance.json")["models"]), {"megnet"})
                        self.assertFalse((report_dir / "proxy_consensus.csv").exists())
                        self.assertTrue((report_dir / "REPORT.md").is_file())
                        with self.assertRaisesRegex(FileNotFoundError, "esen_hull"):
                            execute(config, "report")
            execute(config, "report")
            out = root / "out"
            self.assertEqual(pd.read_csv(out / "candidates.csv").eligible.sum(), 12)
            self.assertEqual(np.load(out / "candidate_embeddings.npy").shape, (12, 16))
            metrics = pd.read_csv(out / "reports/metrics.csv")
            self.assertEqual(set(metrics.scope), {"turn", "batch", "pooled"})
            self.assertTrue((metrics.valid_label_coverage.dropna() == 1).all())
            self.assertTrue((out / "reports/REPORT.md").is_file())
            self.assertTrue((out / "reports/proxy_consensus.csv").is_file())
            self.assertTrue((out / "reports/megnet/REPORT.md").is_file())
            # Stage resumption must not rewrite completed score/selection outputs.
            execute(config, "score")
            execute(config, "select")
            execute(config, "report")
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                execute({**config, "seed": 0}, "select")
            with (out / "scores.csv").open("a") as stream:
                stream.write("corrupt")
            with self.assertRaisesRegex(ValueError, "Modified or incomplete"):
                execute(config, "select")


if __name__ == "__main__":
    unittest.main()
