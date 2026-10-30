"""Numerical and sampling controls for explanatory audits."""

import unittest
import importlib.util
from io import StringIO

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from reference_baselines.evaluation.selection_audit import membership, score_parts
from reference_baselines.evaluation.mattersim_audit import pilot_rows
from reference_baselines.graphs.embedding_audit import pca_geometry
from reference_baselines.structure.validity import geometry_checks


class InterpretationTests(unittest.TestCase):
    def test_formula_NaN_is_preserved_and_empty_number_is_missing(self):
        from reference_baselines.core.audit_io import read_csv

        table = read_csv(StringIO("formula,score\nNaN,\nNaCl,1\n"))
        self.assertEqual(table.formula.iloc[0], "NaN")
        self.assertTrue(np.isnan(table.score.iloc[0]))

    @unittest.skipUnless(
        importlib.util.find_spec("mattersim"),
        "MatterSim is in the separate audit environment",
    )
    def test_fresh_inference_matches_public_calculator_and_input_order(self):
        from ase import Atoms
        from mattersim.forcefield import MatterSimCalculator
        from reference_baselines.evaluation.mattersim_audit import infer, load_potential
        from reference_baselines.paths import DATA_DIR

        if not (DATA_DIR / "mattersim/mattersim-v1.0.0-1M.pth").exists():
            self.skipTest("Local official checkpoint is needed for integration test")
        potential, _ = load_potential()
        atoms = [
            Atoms(
                "NaCl",
                scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                cell=[4, 4, 4],
                pbc=True,
            ),
            Atoms(
                "Si2",
                scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                cell=[3.5, 3.5, 3.5],
                pbc=True,
            ),
        ]
        energy, forces, _ = infer(potential, atoms)
        reverse, _, _ = infer(potential, atoms[::-1])
        np.testing.assert_allclose(energy, reverse[::-1], atol=1e-5)
        calculator = MatterSimCalculator(potential=potential, device="cpu")
        for i, atom in enumerate(atoms):
            atom.calc = calculator
            self.assertAlmostEqual(energy[i], atom.get_potential_energy(), places=5)
            np.testing.assert_allclose(forces[i], atom.get_forces(), atol=1e-5)

    def test_mean_distance_identity_for_each_query(self):
        refs = np.array([[1.0, 2.0], [3.0, 1.0], [9.0, -4.0]])
        query = np.array([[2.0, 5.0], [-1.0, 0.0]])
        centroid, spread = score_parts(query, refs)
        expected = np.square(query[:, None, :] - refs).sum(2).mean(1)
        np.testing.assert_allclose(centroid + spread, expected)
        # A common translation changes neither term nor the total.
        translated, translated_spread = score_parts(query + 100, refs + 100)
        np.testing.assert_allclose(centroid, translated)
        self.assertAlmostEqual(spread, translated_spread)

    def test_selection_swap_partition(self):
        rows = np.arange(5)
        np.testing.assert_array_equal(
            membership(rows, [0, 1], [1, 2]),
            ["removed", "shared", "added", "neither", "neither"],
        )

    def test_pca_reconstructs_centered_data(self):
        data = np.array([[1.0, 2.0], [3.0, 5.0], [5.0, 1.0], [9.0, 4.0]])
        mean, values, axes = pca_geometry(data)
        np.testing.assert_allclose(axes.T @ axes, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(((data - mean) @ axes) @ axes.T + mean, data)
        self.assertAlmostEqual(values.sum(), np.square(data - mean).sum() / len(data))

    def test_periodic_self_overlap_is_not_missed(self):
        structure = Structure(Lattice.orthorhombic(0.2, 5, 5), ["Li"], [[0, 0, 0]])
        checks = geometry_checks(structure)
        self.assertTrue(checks["legacy_geometry_valid"])
        self.assertFalse(checks["periodic_geometry_valid"])
        self.assertAlmostEqual(checks["minimum_periodic_distance_within4A"], 0.2)
        normal = Structure(Lattice.cubic(3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.assertTrue(geometry_checks(normal)["periodic_geometry_valid"])

    def test_pilot_is_balanced_reproducible_and_not_outcome_ranked(self):
        frame = pd.DataFrame(
            [
                {"candidate_row": i, "selection_group": group, "passing": passing}
                for group in ["added", "removed"]
                for passing in [False, True]
                for i in range(
                    (0 if group == "added" else 20) + (0 if not passing else 10),
                    (0 if group == "added" else 20) + (0 if not passing else 10) + 8,
                )
            ]
        )
        a, b = pilot_rows(frame), pilot_rows(frame.sample(frac=1, random_state=4))
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 12)
        counts = (
            frame[frame.candidate_row.isin(a)]
            .groupby(["selection_group", "passing"])
            .size()
        )
        self.assertTrue(counts.eq(3).all())

    @unittest.skipUnless(
        importlib.util.find_spec("smact"), "SMACT is in the separate audit environment"
    )
    def test_composition_rules_and_alloy_exception(self):
        from reference_baselines.structure.validity import composition_checks

        self.assertTrue(composition_checks("NaCl")["composition_valid"])
        self.assertTrue(composition_checks("FeAl")["composition_valid"])
        self.assertFalse(composition_checks("CsO8")["composition_valid"])


if __name__ == "__main__":
    unittest.main()
