"""Controls for formula weighting, exact quotas and representation checks."""

import unittest
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice

from reference_baselines.baselines.formula_balance import formula_weights
from reference_baselines.evaluation.pre_gnn import quota_select
from reference_baselines.evaluation.matched_pairs import sample_pairs
from reference_baselines.checks.encoder_invariance import variants, lattice_vector


class PreGNNTests(unittest.TestCase):
    def test_each_formula_gets_equal_total_weight(self):
        f = np.array(["Na", "Na", "Na", "Cl2", "NaCl"])
        w = formula_weights(f)
        for name in np.unique(f):
            self.assertAlmostEqual(w[f == name].sum(), 1 / 3)
        self.assertAlmostEqual(w.sum(), 1)

    def test_repeating_an_identical_polymorph_does_not_change_formula_mean(self):
        a = np.array(["Na", "Cl2", "NaCl"])
        b = np.array(["Na", "Na", "Cl2", "NaCl"])
        self.assertAlmostEqual(
            formula_weights(a) @ [1, 5, 9], formula_weights(b) @ [1, 1, 5, 9]
        )
        self.assertNotEqual(np.mean([1, 5, 9]), np.mean([1, 1, 5, 9]))

    def test_quota_selection_preserves_order_and_exact_counts(self):
        groups = pd.Series({1: "A", 2: "A", 3: "B", 4: "B", 5: "C"})
        chosen = quota_select(np.array([5, 4, 2, 3, 1]), groups, {"A": 1, "B": 2})
        np.testing.assert_array_equal(chosen, [4, 2, 3])
        self.assertEqual(groups.loc[chosen].value_counts().to_dict(), {"B": 2, "A": 1})

    def test_unfillable_quota_is_rejected_not_silently_reduced(self):
        with self.assertRaises(ValueError):
            quota_select(np.array([1]), pd.Series({1: "A"}), {"A": 2})

    def test_pair_sample_is_seeded_bounded_and_input_order_independent(self):
        a = sample_pairs(np.arange(100), np.random.default_rng(6))
        b = sample_pairs(np.arange(100)[::-1], np.random.default_rng(6))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertTrue(all(x < y for x, y in a))

    def test_rigid_rotation_preserves_original_lattice_features(self):
        s = Structure(
            Lattice.from_parameters(4, 5, 6, 80, 90, 100),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        forms = variants(s)
        np.testing.assert_allclose(
            lattice_vector(s), lattice_vector(forms["rigid_rotation"]), atol=1e-5
        )
        self.assertEqual(
            forms["supercell"].composition.reduced_formula,
            s.composition.reduced_formula,
        )
        self.assertEqual(len(forms["supercell"]), 2 * len(s))


if __name__ == "__main__":
    unittest.main()
