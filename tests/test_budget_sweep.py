"""Controls for equal-budget selection and incomplete fresh checks."""

import unittest

import numpy as np
import pandas as pd

from reference_baselines.evaluation.budget_sweep import ranked_unique, fresh_result


class BudgetSweepTests(unittest.TestCase):
    def test_rank_prefixes_use_finite_scores_ties_and_structural_uniqueness(self):
        rows = np.array([9, 2, 7, 4, 6])
        scores = np.array([0.0, 0.0, np.nan, 1.0, 2.0])
        order = ranked_unique(rows, scores, {2: {4}, 4: {2}})
        np.testing.assert_array_equal(order, [2, 9, 6])
        np.testing.assert_array_equal(order[:2], [2, 9])

    def test_incomplete_fresh_coverage_has_no_full_set_count(self):
        fresh = pd.DataFrame({"usable": [True], "rerelaxed_passing": [True]}, index=[2])
        self.assertEqual(fresh_result(np.array([2, 9]), fresh, {}), (1, None))

    def test_fresh_invalid_and_duplicate_slots_are_not_refilled(self):
        fresh = pd.DataFrame(
            {"usable": [True, True, False], "rerelaxed_passing": [True, True, True]},
            index=[2, 9, 6],
        )
        self.assertEqual(
            fresh_result(np.array([2, 9, 6]), fresh, {2: {9}, 9: {2}}), (3, 1)
        )


if __name__ == "__main__":
    unittest.main()
