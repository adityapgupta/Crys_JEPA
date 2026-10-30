"""Compare fixed rankings at user-specified budgets, without tuning either rule.

Run: python -m reference_baselines.evaluation.budget_sweep
"""

import json

import numpy as np
import pandas as pd

from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.evaluation.selection_audit import ORIGINAL, ANGLE
from reference_baselines.paths import RESULTS_DIR, DOCS_DIR

BUDGETS = (1, 2, 5, 7)
OUT = RESULTS_DIR / "budget_sweep"


def adjacency_from_pairs(pairs):
    adjacency = {}
    for a, b in pairs.itertuples(index=False, name=None):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def ranked_unique(rows, scores, adjacency):
    available = np.isfinite(scores)
    ranked = rows[available][np.lexsort((rows[available], scores[available]))]
    return unique_order(ranked, adjacency)


def fresh_result(chosen, fresh, adjacency):
    """Never estimate a full-set rate from its selectively checked subset."""
    covered = np.isin(chosen, fresh.index)
    if not covered.all():
        return int(covered.sum()), None
    table = fresh.loc[chosen]
    usable = chosen[table.usable.to_numpy(bool)]
    unique = unique_order(usable, adjacency)
    return len(chosen), int(fresh.loc[unique, "rerelaxed_passing"].sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = RESULTS_DIR / "heldout"
    data = np.load(source / "scores.npz")
    rows = data["candidate_rows"]
    names = data["methods"].tolist()
    adjacency = adjacency_from_pairs(read_csv(source / "duplicate_pairs.csv"))
    orders = {
        method: ranked_unique(rows, data["scores"][:, names.index(method)], adjacency)
        for method in [ORIGINAL, ANGLE]
    }
    # Reproduce the frozen 5% selection before opening labels for new budgets.
    frozen = read_csv(source / "selections.csv")
    for method, order in orders.items():
        expected = frozen[frozen.method.eq(method)].sort_values("rank").candidate_row
        np.testing.assert_array_equal(order[: int(len(rows) * 0.05)], expected)
    labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index("candidate_row")
    if not labels.loc[rows, "hull_status"].eq("ok").all():
        raise ValueError("Missing hull labels")
    targets = labels.e_above_hull_proxy.le(0.1).to_numpy(int)
    np.testing.assert_array_equal(labels.index, np.arange(len(labels)))
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    audit = RESULTS_DIR / "interpretation"
    fresh = read_csv(audit / "rerelaxed_checked.csv").set_index("candidate_row")
    fresh_adjacency = adjacency_from_pairs(
        read_csv(audit / "rerelaxed_duplicate_pairs.csv")
    )
    metrics, selections, comparisons = [], [], []
    for percent in BUDGETS:
        budget = len(rows) * percent // 100
        selected = {method: order[:budget] for method, order in orders.items()}
        if any(len(chosen) != budget for chosen in selected.values()):
            raise ValueError("Cannot fill budget")
        for method, chosen in selected.items():
            count = int(targets[chosen].sum())
            coverage, fresh_count = fresh_result(chosen, fresh, fresh_adjacency)
            metrics.append(
                {
                    "budget_percent": percent,
                    "method": method,
                    "selected": budget,
                    "saved_mattersim_passes": count,
                    "saved_pass_rate": count / budget,
                    "fresh_checked": coverage,
                    "fresh_complete": coverage == budget,
                    "fresh_unique_passes_no_refill": fresh_count,
                    "fresh_yield_per_slot": fresh_count / budget
                    if fresh_count is not None
                    else None,
                    "chemical_systems": manifest.loc[
                        chosen, "chemical_system"
                    ].nunique(),
                }
            )
            selections.extend(
                {
                    "budget_percent": percent,
                    "method": method,
                    "rank": rank,
                    "candidate_row": int(row),
                }
                for rank, row in enumerate(chosen, 1)
            )
        base, gate = (int(targets[selected[m]].sum()) for m in [ORIGINAL, ANGLE])
        interval = paired_interval(
            selected[ANGLE],
            selected[ORIGINAL],
            manifest,
            targets,
            seed=20261019,
            draws=1000,
        )
        comparisons.append(
            {
                "budget_percent": percent,
                "selected_each": budget,
                "original_passes": base,
                "angle_passes": gate,
                "extra_passes": gate - base,
                "gain_percentage_points": 100 * (gate - base) / budget,
                "relative_gain_percent": 100 * (gate - base) / base if base else None,
                "gain_ci_low_pp": interval[0] * 100,
                "gain_ci_high_pp": interval[1] * 100,
            }
        )
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "comparisons.csv", index=False)
    protocol = {
        "candidate_count": len(rows),
        "budget_percentages": BUDGETS,
        "methods_unchanged": True,
        "threshold_eV_per_atom": 0.1,
        "stage": "Saved MatterSim-relaxed structures and energies for all four budgets",
        "selection": "Frozen rankings; greedy structural uniqueness; floor(N * percentage / 100)",
        "fresh_stage": "Separate additional-relaxation results only when selection coverage is complete; no refill",
        "uncertainty": "1000 paired chemical-system bootstrap draws; descriptive, no multiple-comparison adjustment",
        "test_status": "Post-hoc budget sensitivity on already opened test outcomes; no budget winner selection",
        "not_generation": True,
        "sha256": {
            str(p.relative_to(RESULTS_DIR)): sha256(p)
            for p in [
                source / "scores.npz",
                source / "duplicate_pairs.csv",
                source / "selections.csv",
                RESULTS_DIR / "candidate_labels.csv",
                RESULTS_DIR / "split_manifest.csv",
                audit / "rerelaxed_checked.csv",
                audit / "rerelaxed_duplicate_pairs.csv",
            ]
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_report(metrics, comparisons, len(rows))
    print(pd.DataFrame(comparisons).to_string(index=False))


def write_report(metrics, comparisons, total):
    lines = [
        "# Budget study",
        "",
        f"Both rules rank the same {total:,} generated candidates. Neither rule generates crystals.",
        "The rule and the 1° angle stay fixed. Only the number selected changes.",
        "",
        "## Same saved MatterSim check for every budget",
        "",
        "All numbers in this table use the same saved MatterSim-relaxed structures and energies.",
        "A pass means corrected energy above the fixed MP2023 hull is at most 0.1 eV/atom.",
        "Selections remove structural duplicates before taking the budget. This table counts stability passes;",
        "it does not add the extra convergence, composition, novelty, or post-relaxation checks below.",
        "",
        "| Budget | Selected each | Crys-JEPA passes | Angle-rule passes | Extra passes | Gain in percentage points | Relative gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparisons:
        n = r["selected_each"]
        lines.append(
            f"| {r['budget_percent']}% | {n:,} | {r['original_passes']:,} ({100 * r['original_passes'] / n:.1f}%) | "
            f"{r['angle_passes']:,} ({100 * r['angle_passes'] / n:.1f}%) | {r['extra_passes']:+,} | "
            f"{r['gain_percentage_points']:+.2f} | {r['relative_gain_percent']:+.1f}% |"
        )
    lines += [
        "",
        "Descriptive 95% uncertainty ranges for the gain, from 1,000 paired chemical-system bootstrap draws:",
        "",
        "| Budget | Gain range in percentage points |",
        "|---|---:|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['budget_percent']}% | {r['gain_ci_low_pp']:+.2f} to {r['gain_ci_high_pp']:+.2f} |"
        )
    lines += [
        "",
        "At 1%, the range includes zero: this small gain is uncertain. The other three ranges are positive.",
        "These four ranges are descriptive and do not adjust for multiple comparisons.",
        "From 5% to 7%, each rule receives 740 more slots. Crys-JEPA adds 261 passes; the angle rule adds 260.",
        "Thus the extra pass count stays nearly the same (+138 to +137), while the gain in pass rate falls.",
    ]
    lines += [
        "",
        "The 5% row must reproduce the frozen 644 versus 782 result. It is a different stage",
        "from the 662 versus 808 result after extra relaxation and checks.",
        "",
        "## Extra MatterSim relaxation: separate check",
        "",
        "These counts also require convergence, geometry and composition checks, and remove duplicates again.",
        "Failed or duplicate slots are not replaced. Missing checks are not treated as passes or failures.",
        "",
        "| Budget | Crys-JEPA fresh passes | Angle-rule fresh passes | Coverage |",
        "|---|---:|---:|---|",
    ]
    for percent in BUDGETS:
        a, b = [
            next(
                r
                for r in metrics
                if r["budget_percent"] == percent and r["method"] == m
            )
            for m in [ORIGINAL, ANGLE]
        ]
        complete = a["fresh_complete"] and b["fresh_complete"]
        lines.append(
            f"| {percent}% | {a['fresh_unique_passes_no_refill'] if complete else 'Pending'} | "
            f"{b['fresh_unique_passes_no_refill'] if complete else 'Pending'} | "
            f"{a['fresh_checked']:,}/{a['selected']:,} original; {b['fresh_checked']:,}/{b['selected']:,} angle |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "A larger budget can add more passes while lowering the pass rate. Compare both counts and rates.",
        "Both rules get the same budget, so a difference is not due to one rule receiving more slots.",
        "This tests screening quality. It does not show that either rule generates more crystals.",
        "The test outcomes were already open. This is a budget study, not a new unseen test.",
        "Do not choose a budget from this table and call its result independent validation.",
        "The two rules still use different reference eligibility: the angle gate can use cross-chemistry neighbours.",
        "This study does not separate that effect from the angle rule itself.",
        "",
        "## Reproduce",
        "",
        "```sh",
        "reference_baselines/environment/audit/bin/python -m reference_baselines.evaluation.budget_sweep",
        "```",
        "",
        "Full selections, counts, bootstrap intervals and input hashes: `reference_baselines/results/budget_sweep/`.",
        "No new model training or new relaxation runs are needed for the saved-energy table.",
        "",
    ]
    (DOCS_DIR / "budget_study.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
