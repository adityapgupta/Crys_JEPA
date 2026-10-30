"""Evaluate pool and mean controls at equal budgets using saved MatterSim labels."""

import json

import numpy as np
import pandas as pd

from reference_baselines.baselines.mean_ablation import OUT, POOLS
from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.budget_sweep import (
    BUDGETS,
    adjacency_from_pairs,
    ranked_unique,
)
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.paths import RESULTS_DIR


def main():
    labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index("candidate_row")
    np.testing.assert_array_equal(labels.index, np.arange(len(labels)))
    targets = labels.e_above_hull_proxy.le(0.1).to_numpy(int)
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    metrics, selections, contrasts = [], [], []
    paths = [RESULTS_DIR / "candidate_labels.csv"]
    for split in ["validation", "test"]:
        path = OUT / f"{split}_scores.npz"
        data = np.load(path)
        rows, names = data["candidate_rows"], data["methods"].tolist()
        if not labels.loc[rows, "hull_status"].eq("ok").all():
            raise ValueError("Missing labels")
        pairs = RESULTS_DIR / (
            "heldout/duplicate_pairs.csv"
            if split == "test"
            else "validation_duplicate_pairs.csv"
        )
        adjacency = adjacency_from_pairs(read_csv(pairs))
        paths += [path, pairs]
        orders = {
            name: ranked_unique(rows, data["scores"][:, j], adjacency)
            for j, name in enumerate(names)
        }
        # Exact identity with the prior 5% original and gate selections.
        if split == "test":
            frozen = read_csv(RESULTS_DIR / "heldout/selections.csv")
            for current, prior in [
                ("chemical_mean", "crys_jepa"),
                ("gate_then_chemical_mean", "theta1_gate_then_original_score"),
            ]:
                expected = (
                    frozen[frozen.method.eq(prior)].sort_values("rank").candidate_row
                )
                np.testing.assert_array_equal(
                    orders[current][: len(rows) * 5 // 100], expected
                )
        for percent in BUDGETS:
            budget = len(rows) * percent // 100
            chosen = {name: order[:budget] for name, order in orders.items()}
            if any(len(value) != budget for value in chosen.values()):
                raise ValueError("Cannot fill equal unique budgets")
            counts = {name: int(targets[value].sum()) for name, value in chosen.items()}
            for name, selected in chosen.items():
                metrics.append(
                    {
                        "split": split,
                        "budget_percent": percent,
                        "method": name,
                        "selected": budget,
                        "passing": counts[name],
                        "pass_rate": counts[name] / budget,
                        "extra_vs_original": counts[name] - counts["chemical_mean"],
                        "gain_pp_vs_original": 100
                        * (counts[name] - counts["chemical_mean"])
                        / budget,
                        "chemical_systems": manifest.loc[
                            selected, "chemical_system"
                        ].nunique(),
                    }
                )
                selections.extend(
                    {
                        "split": split,
                        "budget_percent": percent,
                        "method": name,
                        "rank": rank,
                        "candidate_row": int(row),
                    }
                    for rank, row in enumerate(selected, 1)
                )
            pairs_to_compare = [
                ("chemical_weighted", "chemical_mean", "weighting_only")
            ]
            for pool in POOLS[1:]:
                pairs_to_compare += [
                    (f"{pool}_mean", "chemical_mean", f"{pool}_pool_only"),
                    (
                        f"{pool}_weighted",
                        "chemical_weighted",
                        f"{pool}_pool_with_same_weights",
                    ),
                    (
                        f"{pool}_weighted",
                        f"{pool}_mean",
                        f"{pool}_weighting_with_same_pool",
                    ),
                ]
            pairs_to_compare += [
                (
                    "theta1_top10_fallback_mean",
                    "gate_then_chemical_mean",
                    "angle_pool_vs_candidate_gate_plain",
                ),
                (
                    "theta1_top10_fallback_weighted",
                    "gate_then_chemical_weighted",
                    "angle_pool_vs_candidate_gate_weighted",
                ),
                ("gate_then_chemical_mean", "chemical_mean", "gate_only"),
                (
                    "gate_then_chemical_weighted",
                    "chemical_weighted",
                    "gate_with_same_weights",
                ),
                (
                    "gate_then_chemical_weighted",
                    "gate_then_chemical_mean",
                    "weighting_with_same_gate",
                ),
            ]
            for treatment, comparator, effect in pairs_to_compare:
                low, high = paired_interval(
                    chosen[treatment],
                    chosen[comparator],
                    manifest,
                    targets,
                    seed=20261019,
                    draws=1000,
                )
                contrasts.append(
                    {
                        "split": split,
                        "budget_percent": percent,
                        "effect": effect,
                        "treatment": treatment,
                        "comparator": comparator,
                        "extra_passes": counts[treatment] - counts[comparator],
                        "gain_pp": 100
                        * (counts[treatment] - counts[comparator])
                        / budget,
                        "ci_low_pp": 100 * low,
                        "ci_high_pp": 100 * high,
                    }
                )
            for pool in POOLS[1:]:
                interaction = (
                    counts[f"{pool}_weighted"]
                    - counts[f"{pool}_mean"]
                    - counts["chemical_weighted"]
                    + counts["chemical_mean"]
                )
                contrasts.append(
                    {
                        "split": split,
                        "budget_percent": percent,
                        "effect": f"{pool}_interaction",
                        "treatment": f"{pool}_weighted",
                        "comparator": "two-factor difference of differences",
                        "extra_passes": interaction,
                        "gain_pp": 100 * interaction / budget,
                    }
                )
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    report = {
        "stage": "Same saved MatterSim-relaxed energies for all methods; no mixing with selective fresh checks",
        "cutoff_eV_per_atom": 0.1,
        "budgets_percent": BUDGETS,
        "uniqueness": "same greedy structural uniqueness rule, refill from ranking to equal requested budgets",
        "uncertainty": "1000 paired chemical-system bootstrap draws per contrast; descriptive, unadjusted for multiple comparisons",
        "not_generation": True,
        "test_already_open": True,
        "sha256": {str(p.relative_to(RESULTS_DIR)): sha256(p) for p in paths},
    }
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        pd.DataFrame(metrics)
        .query("split == 'test' and budget_percent == 5")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
