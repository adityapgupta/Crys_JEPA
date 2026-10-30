"""Report analysis must preserve proxy-specific missing-label denominators."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.reference_set_comparison.analysis import (
    ReportBundle, audit_report, budget_winners, consensus_table, load_report, primary_table)
from reference_baselines.reference_set_comparison.common import write_json


def fixture():
    metrics, members, overlaps, paired, consensus = [], [], [], [], []
    for scope, group in (("turn", 2), ("pooled", -1)):
        for method in ("chemical", "knn_5"):
            key = dict(scope=scope, group=group, method=method, policy="unique_refill", fraction=.01)
            base, chosen = set(range(10)), set(range(10) if method == "chemical" else range(5, 15))
            for row in sorted(base | chosen):
                members.append({**key, "candidate_row": row, "selected_baseline": row in base, "selected_alternative": row in chosen})
            shared = len(base & chosen)
            overlaps.append({**key, "shared": shared, "baseline_only": 10-shared, "alternative_only": 10-shared,
                             "selection_jaccard": shared/len(base | chosen)})
            for proxy in ("megnet", "esen"):
                evaluated, passing = {("megnet", "chemical"): (10, 7), ("megnet", "knn_5"): (10, 9),
                                      ("esen", "chemical"): (10, 6), ("esen", "knn_5"): (8, 5)}[(proxy, method)]
                metrics.append({**key, "proxy": proxy, "threshold_eV_per_atom": .1, "population": 1000,
                    "budget": 10, "selected_count": 10, "evaluated_count": evaluated,
                    "unknown_count": 10-evaluated, "stable_count": passing,
                    "precision_evaluated": passing/evaluated, "valid_label_coverage": evaluated/10,
                    "stable_yield_lower_bound": passing/10, "stable_yield_upper_bound": (passing+10-evaluated)/10})
                if method != "chemical":
                    paired.append({**key, "proxy": proxy, "paired_chemical_system_ci_low": -.2 if proxy == "esen" else 0.,
                                   "paired_chemical_system_ci_high": .3 if proxy == "esen" else .4})
            known, both, mg, es, neither = (10, 4, 3, 2, 1) if method == "chemical" else (8, 4, 3, 1, 0)
            consensus.append({**key, "budget": 10, "selected_count": 10, "both_evaluated": known,
                              "unknown_either": 10-known, "both_pass": both, "megnet_only_pass": mg,
                              "esen_only_pass": es, "neither_pass": neither, "agreement_fraction": (both+neither)/known})
    m = pd.DataFrame(metrics)
    return ReportBundle(Path("unused"), {"metrics": m, "pooled_metrics": m[m.scope.eq("pooled")],
        "turn_metrics": m[m.scope.eq("turn")], "batch_metrics": m.iloc[:0],
        "paired_comparisons": pd.DataFrame(paired), "selection_membership": pd.DataFrame(members),
        "overlap": pd.DataFrame(overlaps), "proxy_consensus": pd.DataFrame(consensus),
        "runtime": pd.DataFrame({"stage": ["fixture"], "seconds": [0.]})}, {})


class ReportAnalysisChecks(unittest.TestCase):
    def test_unknown_labels_are_valid_and_precision_winner_can_differ_from_yield(self):
        bundle = fixture()
        audit_report(bundle)
        winners = budget_winners(bundle).query("proxy == 'esen'").set_index("criterion")
        self.assertEqual(winners.loc["precision_evaluated", "method"], "knn_5")
        self.assertEqual(winners.loc["stable_yield_lower_bound", "method"], "chemical")
        row = primary_table(bundle, "esen").set_index("method").loc["knn_5"]
        self.assertAlmostEqual(row.precision_evaluated, 5/8)
        self.assertAlmostEqual(row.stable_yield_lower_bound, 5/10)
        self.assertAlmostEqual(row.stable_yield_upper_bound, 7/10)
        self.assertAlmostEqual(row.paired_chemical_system_ci_low, -.2)

    def test_consensus_keeps_common_label_precision_separate_from_slot_yield(self):
        rows = consensus_table(fixture()).set_index("method")
        self.assertAlmostEqual(rows.loc["knn_5", "both_pass_precision_common"], .5)
        self.assertAlmostEqual(rows.loc["knn_5", "both_pass_yield_lower"], .4)
        self.assertAlmostEqual(rows.loc["knn_5", "both_pass_yield_upper"], .6)
        self.assertAlmostEqual(rows.loc["chemical", "both_pass_yield_lower"], .4)

    def test_audit_rejects_misleading_denominator_and_membership_overlap(self):
        bundle = fixture()
        bad = deepcopy(bundle)
        bad.tables["metrics"].loc[0, "precision_evaluated"] = .99
        with self.assertRaises(AssertionError):
            audit_report(bad)
        bad = deepcopy(bundle)
        bad.tables["overlap"].loc[0, "shared"] -= 1
        with self.assertRaises(AssertionError):
            audit_report(bad)
        bad = deepcopy(bundle)
        bad.tables["metrics"].loc[0, "unknown_count"] = 2
        with self.assertRaises(ValueError):
            audit_report(bad)

    def test_receipt_rejects_changed_transfers_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = fixture()
            products = []
            for name, table in bundle.tables.items():
                path = root / (name + ".csv")
                table.to_csv(path, index=False)
                products.append({"name": path.name, "sha256": sha256(path)})
            write_json(root / "provenance.json", {})
            products.append({"name": "provenance.json", "sha256": sha256(root / "provenance.json")})
            write_json(root / "complete.json", {"outputs": products})
            loaded = load_report(root)
            audit_report(loaded)
            self.assertEqual(loaded.proxies, ("esen", "megnet"))
            (root / "pooled_metrics.csv").write_text("changed\n1\n")
            with self.assertRaisesRegex(ValueError, "transfer"):
                load_report(root)
            products[0]["name"] = "../outside.csv"
            write_json(root / "complete.json", {"outputs": products})
            with self.assertRaises(ValueError):
                load_report(root)


if __name__ == "__main__":
    unittest.main()
