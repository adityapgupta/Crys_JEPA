"""Chemistry-matched selection, formula controls and same-label energy ceiling."""

import json
import numpy as np
import pandas as pd

from reference_baselines.baselines.formula_balance import OUT
from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.budget_sweep import (
    BUDGETS,
    adjacency_from_pairs,
    ranked_unique,
)
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.paths import RESULTS_DIR

METHODS = [
    "chemical_mean",
    "chemical_weighted",
    "intersection_cosine5_fallback_mean",
    "cosine5_mean",
    "theta1_top10_fallback_mean",
    "gate_then_chemical_mean",
    "formula_balanced_mean",
    "gate_then_formula_balanced_mean",
    "direct_hull_oracle",
]


def quota_select(order, groups, quotas):
    frame = pd.DataFrame({"row": order, "group": groups.reindex(order).to_numpy()})
    position = frame.groupby("group", sort=False).cumcount()
    keep = position < frame.group.map(quotas).fillna(0)
    selected = frame.loc[keep, "row"].to_numpy(dtype=int)
    counts = groups.loc[selected].value_counts().to_dict()
    if counts != {k: int(v) for k, v in quotas.items() if v}:
        raise ValueError("Cannot fill every fixed group quota")
    return selected


def main():
    labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index("candidate_row")
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    groups = {"chemical_system": manifest.chemical_system, "formula": labels.formula}
    targets = labels.e_above_hull_proxy.le(0.1).to_numpy(int)
    np.testing.assert_array_equal(labels.index, np.arange(len(labels)))
    metrics, selections, contrasts, inputs = [], [], [], []
    for split in ["validation", "test"]:
        score_path = RESULTS_DIR / "mean_ablation" / f"{split}_scores.npz"
        balanced_path = OUT / f"{split}_formula_scores.npz"
        source, balanced = np.load(score_path), np.load(balanced_path)
        rows = source["candidate_rows"]
        np.testing.assert_array_equal(rows, balanced["candidate_rows"])
        if not labels.loc[rows, "hull_status"].eq("ok").all():
            raise ValueError("Missing hull labels")
        values = dict(zip(source["methods"].tolist(), source["scores"].T))
        values.update(zip(balanced["methods"].tolist(), balanced["scores"].T))
        values["direct_hull_oracle"] = labels.loc[rows, "e_above_hull_proxy"].to_numpy()
        pairs_path = RESULTS_DIR / (
            "heldout/duplicate_pairs.csv"
            if split == "test"
            else "validation_duplicate_pairs.csv"
        )
        adjacency = adjacency_from_pairs(read_csv(pairs_path))
        orders = {m: ranked_unique(rows, values[m], adjacency) for m in METHODS}
        # Greedy StructureMatcher deduplication can retain different numbers of
        # structures under different score orders (matching is not transitive).
        # Fix an energy-blind representative set before matched-quota comparisons.
        canonical = orders["chemical_mean"]
        common_orders = {}
        for m in METHODS:
            v = pd.Series(values[m], index=rows).loc[canonical].to_numpy()
            if not np.isfinite(v).all():
                raise ValueError("Missing scores in common unique candidate universe")
            common_orders[m] = canonical[np.lexsort((canonical, v))]
        inputs.extend([score_path, balanced_path, pairs_path])
        for percent in BUDGETS:
            budget = len(rows) * percent // 100
            global_sets = {m: order[:budget] for m, order in orders.items()}
            settings = [("global", "none", "none", global_sets)]
            common_sets = {m: order[:budget] for m, order in common_orders.items()}
            settings.append(
                (
                    "common_unique_global",
                    "none",
                    "original_representatives",
                    common_sets,
                )
            )
            # Mirror both anchors: neither ranking gets to choose every comparison's chemistry mix.
            for group_name, series in groups.items():
                for anchor in ["chemical_mean", "gate_then_chemical_mean"]:
                    quotas = series.loc[common_sets[anchor]].value_counts().to_dict()
                    sets = {
                        m: quota_select(order, series, quotas)
                        for m, order in common_orders.items()
                    }
                    settings.append(("matched", group_name, anchor, sets))
            for mode, group_name, anchor, sets in settings:
                for method, chosen in sets.items():
                    if len(chosen) != budget:
                        raise ValueError("Unequal budgets")
                    count = int(targets[chosen].sum())
                    metrics.append(
                        {
                            "split": split,
                            "budget_percent": percent,
                            "mode": mode,
                            "matched_group": group_name,
                            "quota_anchor": anchor,
                            "method": method,
                            "selected": budget,
                            "passing": count,
                            "pass_rate": count / budget,
                            "chemical_systems": manifest.loc[
                                chosen, "chemical_system"
                            ].nunique(),
                        }
                    )
                    selections.extend(
                        {
                            "split": split,
                            "budget_percent": percent,
                            "mode": mode,
                            "matched_group": group_name,
                            "quota_anchor": anchor,
                            "method": method,
                            "rank": rank,
                            "candidate_row": int(row),
                        }
                        for rank, row in enumerate(chosen, 1)
                    )
                for treatment in [
                    "chemical_weighted",
                    "formula_balanced_mean",
                    "gate_then_formula_balanced_mean",
                    "theta1_top10_fallback_mean",
                    "gate_then_chemical_mean",
                ]:
                    a, b = sets[treatment], sets["chemical_mean"]
                    low, high = paired_interval(
                        a, b, manifest, targets, seed=20261022, draws=1000
                    )
                    contrasts.append(
                        {
                            "split": split,
                            "budget_percent": percent,
                            "mode": mode,
                            "matched_group": group_name,
                            "quota_anchor": anchor,
                            "method": treatment,
                            "extra_passes": int(targets[a].sum() - targets[b].sum()),
                            "gain_pp": 100 * (targets[a].mean() - targets[b].mean()),
                            "ci_low_pp": 100 * low,
                            "ci_high_pp": 100 * high,
                        }
                    )
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    inputs.extend(
        [RESULTS_DIR / "candidate_labels.csv", RESULTS_DIR / "split_manifest.csv"]
    )
    protocol = {
        "date": "2026-10-22",
        "budgets": BUDGETS,
        "threshold_eV_per_atom": 0.1,
        "direct_hull_role": "Oracle ceiling using the exact evaluation labels, NOT an independent predictor",
        "quotas": "Exact allocations from original or gate on common unique representatives; no energy labels used",
        "matching": "Global retains historical per-method greedy uniqueness. Matched controls fix representatives using original-score greedy order before ranking, to avoid unequal capacities from nontransitive matching. This anchor choice is a limitation.",
        "uncertainty": "1000 chemical-system bootstrap draws; descriptive and unadjusted",
        "test_status": "Exploratory; outcomes already open",
        "new_training": False,
        "sha256": {str(p.relative_to(RESULTS_DIR)): sha256(p) for p in inputs},
    }
    (OUT / "evaluation.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        pd.DataFrame(metrics)
        .query("split == 'test' and budget_percent == 5")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
