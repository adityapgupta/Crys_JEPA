"""Regression checks for the scientific-code corrections (no external data)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from reference_baselines.embedding_structure.build_graph import build_knn_graph, load_embeddings
from reference_baselines.embedding_structure.capture_confound import signature
from reference_baselines.embedding_structure.spacegroup_capture import concentration, bh_adjust
from reference_baselines.embedding_structure.common import aligned_communities
from reference_baselines.embedding_structure.spectral_community import spectral_gap


class ExperimentChecks(unittest.TestCase):
    def test_package_layout(self):
        import importlib
        from reference_baselines.paths import RESULTS_DIR
        from reference_baselines.embedding_structure.common import arguments
        self.assertEqual(arguments('test').parse_args([]).out, RESULTS_DIR / 'embedding_structure')
        for name in ('spectral_community', 'community_lists_and_map', 'energy_analysis',
                     'probe', 'spacegroup_capture', 'capture_confound'):
            importlib.import_module(f'reference_baselines.embedding_structure.{name}')

    def test_formula_parser(self):
        self.assertEqual(signature('K2Ca(CO3)2'), (1, 2, 2, 6))
        self.assertEqual(signature('Ca3(PO4)2'), (2, 3, 8))
        self.assertEqual(signature('CH3COOH'), (1, 1, 2))
        self.assertEqual(signature('NaN'), (1, 1))
        self.assertEqual(signature('Fe0.5Mn0.5O'), (1, 1, 2))

    def test_duplicate_self_exclusion(self):
        W, _ = build_knn_graph(np.zeros((8, 3)), k=3, symmetrize='none')
        self.assertTrue(np.all(np.diff(W.indptr) == 3))
        self.assertTrue(np.all(W.diagonal() == 0))

    def test_graph_validation(self):
        for kwargs in ({'k': 0}, {'k': 2, 'sigma': -1}, {'k': 2, 'normalize': True}):
            with self.assertRaises(ValueError):
                build_knn_graph(np.zeros((5, 2)), **kwargs)

    def test_sample_size_matched_null(self):
        # Many equal-size communities: selecting the best five sampled counts
        # has a much higher null expectation for a small group than for a large one.
        community = np.repeat(np.arange(100), 10)
        labels = np.zeros(1000, dtype=int)
        labels[:20] = 1
        result = concentration(labels, community, draws=100, minimum=1).set_index('space_group')
        self.assertGreater(result.loc[1, 'null_top5_mean'], result.loc[0, 'null_top5_mean'])
        self.assertGreater(result.loc[1, 'null_top5_mean'], .05)
        self.assertTrue(np.all(result.permutation_p > 0))
        again = concentration(labels, community, draws=100, minimum=1).set_index('space_group')
        pd.testing.assert_frame_equal(result, again)

    def test_bh(self):
        np.testing.assert_allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])

    def test_material_id_alignment(self):
        meta = pd.DataFrame({'material_id': ['b', 'a']})
        saved = pd.DataFrame({'material_id': ['a', 'b'], 'community_fine': [1, 2]})
        self.assertEqual(aligned_communities(meta, saved).community_fine.tolist(), [2, 1])
        with self.assertRaises(ValueError):
            aligned_communities(meta, saved.iloc[:1])

    def test_valid_nan_formula(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d)/'embeddings.npy', np.ones((2, 2)))
            (Path(d)/'metadata.csv').write_text('pretty_formula\nNaN\nNaCl\n')
            _, metadata = load_embeddings(d)
            self.assertEqual(metadata.pretty_formula.iloc[0], 'NaN')

    def test_spectrum_against_dense(self):
        W, _ = build_knn_graph(np.arange(12).reshape(6, 2), k=2)
        vals, diagnostics, note = spectral_gap(W, n_eig=3)
        degree = np.asarray(W.sum(1)).ravel()
        L = np.eye(6) - W.toarray() / np.sqrt(degree[:, None]*degree[None, :])
        self.assertIsNotNone(vals, note)
        np.testing.assert_allclose(vals, np.linalg.eigvalsh(L)[:3], atol=1e-9)
        self.assertLess(max(diagnostics['residual_norms']), 1e-9)


if __name__ == '__main__':
    unittest.main()
