"""Checks for conditional permutations and energy-only concentration."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

from reference_baselines.embedding_structure.energy_capture import energy_strata, mixing_diagnostics
from reference_baselines.embedding_structure.spacegroup_capture import concentration, permutation_indices


class EnergyCaptureTests(unittest.TestCase):
    def test_command_aligns_rows_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = pd.DataFrame({'material_id': [f'm{i}' for i in range(40)],
                                 'spacegroup.number': np.repeat([1, 2], 20),
                                 'formation_energy_per_atom': np.tile([-.015, -.005], 20)})
            communities = pd.DataFrame({'material_id': meta.material_id,
                                        'spacegroup_number': meta['spacegroup.number'],
                                        'community_fine': np.repeat(np.arange(20), 2)})
            meta.to_csv(root / 'metadata.csv', index=False)
            communities.iloc[::-1].to_csv(root / 'crystals_by_community.csv', index=False)
            run = subprocess.run([sys.executable, '-m', 'reference_baselines.embedding_structure.energy_capture',
                                  '--data', str(root), '--out', str(root), '--draws', '9',
                                  '--minimum', '1', '--widths', '.01'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            out = root / 'energy_capture'
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['analyses'][0]['groups_tested'], 2)
            result = pd.read_csv(out / 'width_0.01.csv')
            np.testing.assert_allclose(result.top5_share, .5)
            self.assertTrue((out / 'width_0.01_bins.csv').is_file())

    def test_bin_validation(self):
        np.testing.assert_array_equal(energy_strata([-.015, -.005, .005, .015], .01), [-2, -1, 0, 1])
        for energy, width in [([np.nan], .01), ([0], 0), ([0], -1), ([0], np.inf)]:
            with self.assertRaises(ValueError):
                energy_strata(energy, width)

    def test_permutations_stay_inside_bins(self):
        bins = np.repeat(np.arange(4), 20)
        blocks = [np.flatnonzero(bins == b) for b in np.unique(bins)]
        order = permutation_indices(len(bins), np.random.default_rng(4), blocks)
        np.testing.assert_array_equal(bins[order], bins)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(bins)))
        self.assertFalse(np.array_equal(order, np.arange(len(bins))))

    def test_energy_only_concentration_disappears(self):
        community = np.repeat(np.arange(20), 20)
        sg = np.repeat([1, 2], 200)
        raw = concentration(sg, community, draws=99, minimum=1)
        controlled = concentration(sg, community, draws=99, minimum=1, strata=sg)
        self.assertTrue((raw.enrichment > 1).all())
        np.testing.assert_allclose(controlled.enrichment, 1)
        np.testing.assert_allclose(controlled.permutation_p, 1)

    def test_signal_within_energy_bins_survives(self):
        community = np.repeat(np.arange(20), 20)
        sg = np.repeat([1, 2], 200)
        bins = np.tile([0, 1], 200)
        result = concentration(sg, community, draws=99, minimum=1, strata=bins)
        self.assertTrue((result.bh_q <= .05).all())
        _, movable = mixing_diagnostics(bins.astype(float), sg, community, bins)
        self.assertTrue(movable.all())

    def test_singletons_have_no_permutation_evidence(self):
        sg = np.repeat([1, 2], 20)
        result = concentration(sg, np.arange(40), draws=9, minimum=1, strata=np.arange(40))
        np.testing.assert_allclose(result.permutation_p, 1)
        with self.assertRaises(ValueError):
            concentration(sg, np.arange(40), strata=[0])


if __name__ == '__main__':
    unittest.main()
