"""Batch quotas, weighted denominators, and scope separation are scientific invariants."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from reference_baselines.reference_set_comparison.aggregation import (
    aggregate_batch_metrics, aggregate_batch_consensus, paired_batch_interval)
from reference_baselines.reference_set_comparison.reporting import pooled_requests
from reference_baselines.reference_set_comparison.reporting import generate_reports
from reference_baselines.reference_set_comparison.common import write_json, read_json
from reference_baselines.reference_set_comparison.analysis import load_report, audit_report, primary_table, plot_primary


def counts():
    return pd.DataFrame([
        dict(scope="batch", group=20, method="chemical", policy="unique_refill", fraction=.01,
             proxy="esen", threshold_eV_per_atom=.1, population=150, budget=1,
             selected_count=1, evaluated_count=1, unknown_count=0, stable_count=1),
        dict(scope="batch", group=21, method="chemical", policy="unique_refill", fraction=.01,
             proxy="esen", threshold_eV_per_atom=.1, population=350, budget=3,
             selected_count=3, evaluated_count=2, unknown_count=1, stable_count=1)])


class BatchwiseChecks(unittest.TestCase):
    def test_floor_each_batch_before_adding_and_recompute_precision(self):
        source = counts()
        result = aggregate_batch_metrics(source).iloc[0]
        self.assertEqual(result.budget, 4)
        self.assertNotEqual(result.budget, int(np.floor(.01*source.population.sum())))
        self.assertAlmostEqual(result.precision_evaluated, 2/3)
        self.assertNotAlmostEqual(result.precision_evaluated, np.mean([1., .5]))
        self.assertAlmostEqual(result.stable_yield_lower_bound, .5)
        self.assertAlmostEqual(result.stable_yield_upper_bound, .75)
        self.assertEqual(result.batch_count, 2)

    def test_never_aggregate_turn_scores_or_duplicate_batches(self):
        source = counts()
        with self.assertRaises(ValueError):
            aggregate_batch_metrics(source.assign(scope="pooled"))
        with self.assertRaises(ValueError):
            aggregate_batch_metrics(pd.concat([source, source.iloc[:1]]))
        with self.assertRaises(AssertionError):
            aggregate_batch_metrics(source.assign(budget=5))

    def test_medians_and_distinct_system_counts_are_not_averaged_or_summed(self):
        source = counts().assign(median_hull_distance=[0., 10.], chemical_systems=[1, 1],
                                 mean_hull_distance=[0., .3])
        result = aggregate_batch_metrics(source)
        self.assertNotIn("median_hull_distance", result)
        self.assertNotIn("chemical_systems", result)
        self.assertAlmostEqual(result.iloc[0].mean_hull_distance, .2)

    def test_consensus_rates_use_summed_counts(self):
        source = counts().drop(columns=["proxy", "threshold_eV_per_atom", "evaluated_count", "unknown_count", "stable_count"])
        source = source.assign(both_evaluated=[1, 2], unknown_either=[0, 1], both_pass=[1, 0],
                               neither_pass=[0, 1], megnet_only_pass=[0, 1], esen_only_pass=[0, 0])
        result = aggregate_batch_consensus(source).iloc[0]
        self.assertEqual(result.budget, 4)
        self.assertEqual(result.both_pass, 1)
        self.assertAlmostEqual(result.agreement_fraction, 2/3)

    def test_paired_batch_intervals_and_missing_batch_rejection(self):
        baseline = pd.DataFrame({"group": [20, 21], "stable_count": [2, 2], "evaluated_count": [10, 10]})
        alternative = baseline.assign(stable_count=4)
        low, high = paired_batch_interval(baseline, alternative, repeats=100)
        self.assertAlmostEqual(low, .2)
        self.assertAlmostEqual(high, .2)
        self.assertEqual((low, high), paired_batch_interval(baseline, alternative, repeats=100))
        with self.assertRaises(ValueError):
            paired_batch_interval(baseline, alternative.iloc[:1])

    def test_report_aggregate_uses_batch_membership_without_reranking(self):
        requests = counts().drop(columns=["proxy", "threshold_eV_per_atom", "evaluated_count", "unknown_count", "stable_count"])
        requests = requests.assign(scoreable_count=requests.population, raw_unique_without_refill=requests.budget,
                                   raw_selected_count=requests.budget)
        turn = requests.iloc[:1].assign(scope="turn", group=2, population=500, budget=5,
                                       selected_count=5, scoreable_count=500, raw_unique_without_refill=5, raw_selected_count=5)
        source = pd.concat([requests, turn], ignore_index=True)
        selection = pd.DataFrame({"scope": ["batch"]*4+["turn"]*5,
                                  "candidate_row": [1, 11, 12, 13, 2, 3, 4, 5, 6]})
        _, historical = pooled_requests(source, selection)
        augmented, corrected = pooled_requests(source, historical, "batch", "batch_aggregate")
        aggregate = augmented[augmented.scope.eq("batch_aggregate")].iloc[0]
        self.assertEqual(aggregate.budget, 4)
        self.assertEqual(set(corrected.loc[corrected.scope.eq("batch_aggregate"), "candidate_row"]), {1, 11, 12, 13})
        self.assertEqual(set(corrected.loc[corrected.scope.eq("pooled"), "candidate_row"]), {2, 3, 4, 5, 6})

    def test_generated_report_and_count_only_plot_use_batch_quotas(self):
        """Real reporting/receipt/analysis path with fixed synthetic labels, no ML dependencies."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohorts = {
                ("batch", 20, "chemical"): [0], ("batch", 21, "chemical"): [10, 11, 12],
                ("batch", 20, "knn_5"): [1], ("batch", 21, "knn_5"): [10, 13, 14],
                ("turn", 2, "chemical"): [0, 1, 2, 3, 4], ("turn", 2, "knn_5"): [2, 3, 4, 5, 6]}
            requests, selections = [], []
            for (scope, group, method), ids in cohorts.items():
                population = {20: 150, 21: 350, 2: 500}[group]
                row = dict(scope=scope, group=group, method=method, policy="unique_refill", fraction=.01,
                           population=population, budget=len(ids), selected_count=len(ids), scoreable_count=population,
                           raw_selected_count=len(ids), raw_unique_without_refill=len(ids))
                requests.append(row)
                selections.extend({**row, "candidate_row": i} for i in ids)
            pd.DataFrame(requests).to_csv(root/"selection_requests.csv", index=False)
            pd.DataFrame(selections).to_csv(root/"selections.csv", index=False)
            ids = sorted({i for group in cohorts.values() for i in group})
            for model in ("megnet", "esen"):
                pd.DataFrame([dict(candidate_row=i, chemical_system="Li", eval_folder=20 if i<10 else 21,
                    hull_status="candidate_prediction_error" if model=="esen" and i==14 else "ok",
                    e_above_hull_proxy=np.nan if model=="esen" and i==14 else (.05 if i%2==0 else .2))
                    for i in ids]).to_csv(root/f"{model}_labels.csv", index=False)
                write_json(root/f"{model}_hull.json", {"fixture": model})
                write_json(root/f"{model}_model.json", {"fingerprint": model})
            write_json(root/"run.json", {"config": {"start_batch":20, "end_batch":21}})
            write_json(root/"select.json", {"evaluation_union_count": len(ids)})
            write_json(root/"phases.json", {"phase_count": 1})
            generate_reports({"output": str(root), "models": {"megnet": {}, "esen": {}},
                              "bootstrap_repeats": 20, "seed": 2026})
            bundle = load_report(root/"reports")
            audit_report(bundle)
            self.assertIn("batch_aggregate", set(bundle.tables["metrics"].scope))
            self.assertTrue((root/"reports/batch_aggregate_metrics.csv").is_file())
            self.assertEqual(read_json(root/"reports/provenance.json")["primary_scope"], "batch_aggregate")
            self.assertIn("separately per batch", (root/"reports/REPORT.md").read_text())
            self.assertTrue((primary_table(bundle, "megnet", scope="batch_aggregate").budget == 4).all())
            self.assertTrue((primary_table(bundle, "megnet", scope="pooled").budget == 5).all())
            fig = plot_primary(bundle, "megnet", root/"figures", scope="batch_aggregate", counts_only=True)
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(len(fig.axes[0].patches), 2)
            self.assertTrue((root/"figures/megnet_primary_counts.png").is_file())
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
