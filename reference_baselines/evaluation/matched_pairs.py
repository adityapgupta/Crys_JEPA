"""Outcome-independent same-formula pair sample; retain both wins and losses."""

from itertools import combinations
import json
import numpy as np
import pandas as pd

from reference_baselines.baselines.formula_balance import OUT
from reference_baselines.core.audit_io import read_csv
from reference_baselines.evaluation.budget_sweep import adjacency_from_pairs
from reference_baselines.paths import RESULTS_DIR


def sample_pairs(rows, rng, cap=20):
    rows = np.sort(np.asarray(rows, dtype=int))
    if len(rows) < 2:
        return []
    if len(rows) <= 32:
        pairs = list(combinations(rows, 2))
        return [
            pairs[i]
            for i in np.sort(
                rng.choice(len(pairs), min(cap, len(pairs)), replace=False)
            )
        ]
    chosen = set()
    while len(chosen) < cap:
        chosen.add(tuple(sorted(rng.choice(rows, 2, replace=False).tolist())))
    return sorted(chosen)


def main():
    frame = read_csv(RESULTS_DIR / "interpretation/candidate_audit.csv").set_index(
        "candidate_row"
    )
    results, summaries = [], []
    for split in ["validation", "test"]:
        table = frame[frame.split.eq(split)]
        pairs_path = RESULTS_DIR / (
            "heldout/duplicate_pairs.csv"
            if split == "test"
            else "validation_duplicate_pairs.csv"
        )
        adjacency = adjacency_from_pairs(read_csv(pairs_path))
        source = np.load(RESULTS_DIR / "mean_ablation" / f"{split}_scores.npz")
        scoring = pd.DataFrame(
            source["scores"], index=source["candidate_rows"], columns=source["methods"]
        )
        balanced = np.load(OUT / f"{split}_formula_scores.npz")
        scoring["formula_balanced_mean"] = pd.Series(
            balanced["scores"][:, 0], index=balanced["candidate_rows"]
        )
        scoring["nearest_angle"] = table.nearest_angle_degrees
        methods = [
            "chemical_mean",
            "chemical_weighted",
            "formula_balanced_mean",
            "nearest_angle",
            "gate_then_chemical_mean",
        ]
        rng = np.random.default_rng(20261022)
        for formula, group in table.groupby("formula", sort=True):
            for a, b in sample_pairs(group.index, rng):
                if b in adjacency.get(a, set()):
                    continue
                ra, rb = table.loc[a], table.loc[b]
                record = {
                    "split": split,
                    "formula": formula,
                    "candidate_a": a,
                    "candidate_b": b,
                    "hull_a": ra.e_above_hull_proxy,
                    "hull_b": rb.e_above_hull_proxy,
                    "different_stability_labels": bool(ra.passing != rb.passing),
                    "energy_gap_over_0.01": bool(
                        abs(ra.e_above_hull_proxy - rb.e_above_hull_proxy) > 0.01
                    ),
                }
                for method in methods:
                    delta = scoring.loc[a, method] - scoring.loc[b, method]
                    record[method + "_agreement"] = (
                        float(
                            delta * (ra.e_above_hull_proxy - rb.e_above_hull_proxy) > 0
                        )
                        if np.isfinite(delta) and delta != 0
                        else np.nan
                    )
                results.append(record)
        current = pd.DataFrame([r for r in results if r["split"] == split])
        for criterion in ["different_stability_labels", "energy_gap_over_0.01"]:
            f = current[current[criterion]]
            for method in methods:
                s = f[method + "_agreement"]
                group_means = (
                    f.groupby("formula")[method + "_agreement"].mean().dropna()
                )
                summaries.append(
                    {
                        "split": split,
                        "criterion": criterion,
                        "method": method,
                        "pairs": int(s.notna().sum()),
                        "formulas": len(group_means),
                        "pair_accuracy": float(s.mean()),
                        "formula_macro_accuracy": float(group_means.mean()),
                    }
                )
    pd.DataFrame(results).to_csv(OUT / "same_formula_pairs.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "same_formula_summary.csv", index=False)
    (OUT / "pair_protocol.json").write_text(
        json.dumps(
            {
                "seed": 20261022,
                "cap_per_formula": 20,
                "sampling_uses_energy": False,
                "duplicates": "Exclude structure-matched pairs from existing full split audit",
                "tie_policy": "Nonfinite/tied score pairs excluded per method; report coverage",
                "structure_review": "Earlier exported CIFs, SOAP comparisons and win/loss cases retained",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        pd.DataFrame(summaries)
        .query("split == 'test' and criterion == 'different_stability_labels'")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
