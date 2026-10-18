"""Controls for held-out uniqueness, novelty, and chemistry bias checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

from reference_baselines.evaluation import heldout
from reference_baselines.structure.heldout import novelty_group, selections


class FakeTransaction:
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, key):
        return self.records.get(key)


class HeldoutTests(unittest.TestCase):
    def test_unique_budget_refills_and_breaks_ties_by_row(self):
        rows = np.array([3, 1, 2, 0])
        scores = np.array([[1.0], [0.0], [0.0], [np.nan]])
        selected = selections(rows, ["a"], scores, {1: {2}, 2: {1}}, 2)
        np.testing.assert_array_equal(selected["a"], [1, 3])
        with self.assertRaises(ValueError):
            selections(rows, ["a"], scores, {1: {2}, 2: {1}}, 3)

    def test_novelty_checks_every_polymorph_and_handles_missing_formula(self):
        import pickle
        from unittest.mock import Mock
        from reference_baselines.structure import heldout as structures

        known = Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]])
        other = Structure(Lattice.cubic(3), ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        translated = known.copy()
        translated.translate_sites([0], [0.2, 0.3, 0.4])
        db = Mock()
        db.begin.side_effect = lambda: FakeTransaction(
            {
                b"Li.Li.length": pickle.dumps(2),
                b"Li.Li.0": pickle.dumps(
                    {"entry_id": "other", "structure": other.as_dict()}
                ),
                b"Li.Li.1": pickle.dumps(
                    {"entry_id": "known", "structure": known.as_dict()}
                ),
            }
        )
        with patch.object(structures, "DATABASE", db):
            result = novelty_group(("Li", [(9, translated)]))[0]
            self.assertFalse(result["novel_against_mp2023"])
            self.assertEqual(result["matched_entry_id"], "known")
            missing = novelty_group(
                ("Na", [(10, Structure(Lattice.cubic(3), ["Na"], [[0, 0, 0]]))])
            )[0]
            self.assertTrue(missing["novel_against_mp2023"])
            self.assertEqual(missing["same_formula_references"], 0)

    def test_fixed_group_weights_do_not_confuse_chemistry_mix_with_gain(self):
        records = []
        for kind in ["chemical_system", "composition"]:
            for method, counts in [
                ("original", [(90, 81), (10, 1)]),
                ("new", [(10, 9), (90, 9)]),
            ]:
                for key, (count, passing) in zip(["A", "B"], counts):
                    records.append(
                        {
                            "method": method,
                            "group_type": kind,
                            "group": key,
                            "selected": count,
                            "passing": passing,
                            "pool_count": 100,
                        }
                    )
        result = heldout.shared_group_summary(
            records,
            {
                "comparator": "original",
                "methods": ["original", "new"],
                "minimum_selected_per_shared_group": 5,
            },
        )
        np.testing.assert_allclose(result.population_weighted_within_group_gain, 0)

    def test_frozen_input_change_is_rejected(self):
        from reference_baselines.core.io import sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.write_text("fixed")
            (root / "protocol.json").write_text(
                json.dumps({"sha256": {"input": sha256(source)}})
            )
            with (
                patch.object(heldout, "ROOT", root),
                patch.object(heldout, "OUT", root),
            ):
                heldout.protocol()
                source.write_text("changed")
                with self.assertRaises(ValueError):
                    heldout.protocol()


if __name__ == "__main__":
    unittest.main()
