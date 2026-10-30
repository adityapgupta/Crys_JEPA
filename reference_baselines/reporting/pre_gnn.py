"""Write clear findings and a tracker from completed pre-training controls."""

import json

from reference_baselines.baselines.formula_balance import OUT
from reference_baselines.core.audit_io import read_csv
from reference_baselines.paths import DOCS_DIR


def plot_controls(metrics, pairs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = {
        "chemical_mean": "#4169a1",
        "formula_balanced_mean": "#ad7423",
        "gate_then_chemical_mean": "#147d64",
    }
    labels = {
        "chemical_mean": "Original mean",
        "formula_balanced_mean": "Formula-balanced",
        "gate_then_chemical_mean": "Angle gate",
    }
    test = metrics[metrics.split.eq("test")]
    for method in colors:
        f = test[test["mode"].eq("global") & test.method.eq(method)].sort_values(
            "budget_percent"
        )
        axes[0, 0].plot(
            f.budget_percent,
            f.pass_rate * 100,
            "o-",
            color=colors[method],
            label=labels[method],
        )
    axes[0, 0].set(
        title="A. Global selection",
        xlabel="Selection budget (%)",
        ylabel="Stability passes (%)",
        ylim=(0, 60),
        xticks=[1, 2, 5, 7],
    )
    axes[0, 0].legend(fontsize=9)
    for anchor, style, label in [
        ("chemical_mean", "o-", "Original chemistry counts"),
        ("gate_then_chemical_mean", "s--", "Angle-rule chemistry counts"),
    ]:
        f = test[
            test["mode"].eq("matched")
            & test.matched_group.eq("chemical_system")
            & test.quota_anchor.eq(anchor)
        ]
        original = f[f.method.eq("chemical_mean")].set_index("budget_percent")
        gate = f[f.method.eq("gate_then_chemical_mean")].set_index("budget_percent")
        axes[0, 1].plot(
            original.index,
            100 * (gate.pass_rate - original.pass_rate),
            style,
            label=label,
        )
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set(
        title="B. Same chemistry counts and unique structures",
        xlabel="Selection budget (%)",
        ylabel="Angle gain (percentage points)",
        xticks=[1, 2, 5, 7],
    )
    axes[0, 1].legend(fontsize=8)
    f = pairs[
        pairs.split.eq("test") & pairs.criterion.eq("different_stability_labels")
    ].set_index("method")
    methods = ["chemical_mean", "formula_balanced_mean", "gate_then_chemical_mean"]
    values = 100 * f.loc[methods, "pair_accuracy"]
    axes[1, 0].bar(
        [labels[m] for m in methods], values, color=[colors[m] for m in methods]
    )
    axes[1, 0].axhline(50, color="black", linestyle="--", linewidth=0.8)
    for i, value in enumerate(values):
        axes[1, 0].text(i, value + 2, f"{value:.1f}%", ha="center")
    axes[1, 0].set(
        title="C. Passing vs failing structures: same formula",
        ylabel="Correct pair order (%)",
        ylim=(0, 100),
    )
    x = np.arange(3)
    axes[1, 1].bar(
        x - 0.18,
        [62.5, 25, 12.5],
        width=0.36,
        label="Original mean",
        color=colors["chemical_mean"],
    )
    axes[1, 1].bar(
        x + 0.18,
        [100 / 3] * 3,
        width=0.36,
        label="Formula-balanced",
        color=colors["formula_balanced_mean"],
    )
    axes[1, 1].set(
        title="D. NaCl reference pool",
        ylabel="Total formula weight (%)",
        xticks=x,
        xticklabels=["Na (5 forms)", "Cl₂ (2 forms)", "NaCl (1 form)"],
        ylim=(0, 75),
    )
    axes[1, 1].legend(fontsize=9)
    path = DOCS_DIR / "figures/pre_gnn_controls.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    metrics = read_csv(OUT / "metrics.csv")
    pairs = read_csv(OUT / "same_formula_summary.csv")
    plot_controls(metrics, pairs)
    encoder = json.loads((OUT / "encoder_check.json").read_text())
    test = metrics[metrics.split.eq("test") & metrics.budget_percent.eq(5)]
    lines = [
        "# Checks before GNN design — 22 October 2026",
        "",
        "## Main finding",
        "",
        "The angular screening gain is largely removed when both rules select the same chemistry mix.",
        "This changes our interpretation: the existing evidence is strongest for choosing promising chemical systems,",
        "rather than choosing better structures within each chemical system. This is a controlled diagnostic, not a causal proof.",
        "No model training, new generation, DFT or new MatterSim relaxation was run in this study.",
        "",
        "![Global gain and matched controls](figures/pre_gnn_controls.png)",
        "",
        "## 1. Balance repeated formulas",
        "",
        "Every reduced reference formula gets equal total weight. If it has five known structures, each gets one fifth",
        "of that formula's weight. The five structures remain separate. The reference pool and squared distance do not change.",
        "For NaCl the known pool has five Na, two Cl2, and one NaCl structures. Na's weight falls from 5/8 to 1/3.",
        "",
        "At test 5%, original plain mean gives 644 passes; formula-balanced mean gives 696; the angle gate gives 782.",
        "Balancing formulas and applying the same angle gate gives 786. Thus simple formula balance helps globally,",
        "but it does not account for the full angular-screening gain. At validation 5%, counts are 727, 803, and 857.",
        "",
        "## 2. Hold the chemistry mix fixed",
        "",
        "Both rules get exactly the same per-system counts and the same total budget. We repeat with allocations",
        "from each rule, so the conclusion does not rely on only the original rule's preferred chemistry mix.",
        "We also repeat with exact-formula quotas. Quotas use rankings and composition, never stability labels.",
        "",
        "Greedy duplicate removal depends on ranking because structure matches need not be transitive. For matched",
        "controls we therefore first fix a shared set of unique representatives in original-score order. All rules",
        "then rank those same structures. This representative anchor is a limitation; global historical counts are",
        "reported separately. Comparing matched and global counts without that distinction would be misleading.",
        "",
        "### Test at 5%: 1,849 selections each",
        "",
        "| Control | Original mean | Formula-balanced mean | Angle gate |",
        "|---|---:|---:|---:|",
    ]
    settings = [
        ("global", "none", "none", "Original global comparison"),
        (
            "common_unique_global",
            "none",
            "original_representatives",
            "Shared unique structures, global ranking",
        ),
        (
            "matched",
            "chemical_system",
            "chemical_mean",
            "Same chemistry counts: original allocation",
        ),
        (
            "matched",
            "chemical_system",
            "gate_then_chemical_mean",
            "Same chemistry counts: angle allocation",
        ),
        (
            "matched",
            "formula",
            "chemical_mean",
            "Same formula counts: original allocation",
        ),
        (
            "matched",
            "formula",
            "gate_then_chemical_mean",
            "Same formula counts: angle allocation",
        ),
    ]
    for mode, group, anchor, label in settings:
        f = test[
            test["mode"].eq(mode)
            & test.matched_group.eq(group)
            & test.quota_anchor.eq(anchor)
        ].set_index("method")
        lines.append(
            f"| {label} | {f.loc['chemical_mean', 'passing']} | {f.loc['formula_balanced_mean', 'passing']} | {f.loc['gate_then_chemical_mean', 'passing']} |"
        )
    lines += [
        "",
        "Under original chemistry quotas, the test gate change is -4 passes; under angle chemistry quotas it is +2.",
        "Both descriptive 95% uncertainty ranges include zero (-0.77 to +0.35 and -2.12 to +2.36 percentage points).",
        "The validation result is not identical: its same-chemistry gains are +7 and +36 passes under the two anchors.",
        "So we report both splits and all four budgets; we do not claim an exact universal decomposition of the gain.",
        "",
        "## 3. Compare same-formula structures",
        "",
        "Pairs were sampled with a fixed seed and at most 20 per formula, before using their energy labels.",
        "Known structure-matched pairs were excluded. Tied or missing scores are excluded per method and coverage is reported.",
        "For test pairs with different pass/fail labels, 4,622 pairs across 1,262 formulas are available.",
        "",
        "| Ranking rule | Correct pair order | Mean accuracy across formulas |",
        "|---|---:|---:|",
    ]
    for row in pairs[
        pairs.split.eq("test") & pairs.criterion.eq("different_stability_labels")
    ].itertuples():
        lines.append(
            f"| {row.method} | {100 * row.pair_accuracy:.1f}% | {100 * row.formula_macro_accuracy:.1f}% |"
        )
    lines += [
        "",
        "These descriptive accuracies are close to 50%. The current scores do not show useful discrimination",
        "between passing and failing structures of the same formula in this sample. This is not proof that no structural",
        "signal exists. The earlier selected case studies include both wins and losses, and remain illustrative.",
        "See [structures and examples](selection_interpretation.md#exact-composition-examples), including CIF and SOAP checks.",
        "",
        "## 4. Rank directly by energy above hull",
        "",
        "Direct ranking by the exact saved hull labels gives 1,849/1,849 passes at 5%. It gives 100% at all four",
        "global budgets in this dataset. This is an oracle benchmark using the same labels that define a pass, not an",
        "independent learned predictor. If these energies are already available, neither JEPA screening rule improves",
        "the same-label stability yield over this direct ranking. Any practical advantage must be shown in another",
        "setting, such as independent energy checks, missing hull references, or total pipeline cost.",
        "",
        "## 5. Check the original encoder",
        "",
        "The official model_1972 checkpoint was downloaded from the pinned release and verified with archive CRC and SHA256.",
        "The public data/jepa listing contains mp.csv.gz and mptrj.csv.gz, but no mean_std_scaler.pt. No local scaler",
        "or original embedding-extraction script was found. The scaler is not stored in the checkpoint.",
        "Six equivalent representations were constructed for 16 seeded known crystals: identity, atom order, origin shift,",
        "rigid rotation, lattice-basis permutation, and a doubled cell. Geometry equivalence and raw lattice-feature changes",
        "were checked. These are feature checks, not evidence of encoder invariance.",
        "All 96 variants match their original crystal under the structure matcher. Rigid rotation changes the six raw",
        "lattice features by at most about 5e-6; basis permutations and doubled cells can change them substantially.",
        f"**Encoder status: {encoder['status']}.** Exact saved-embedding reproduction and encoder invariance remain pending.",
        "The check accepts --scaler PATH and first verifies reproduction of the saved vectors before treating invariance as relevant.",
        "A scaler estimated from MP20 would be a different preprocessing pipeline and has not been substituted.",
        "",
        "## 6. What this means for later model design",
        "",
        "- Do not describe the current gain as proof of better physical reference phases or better polymorph selection.",
        "- Keep chemistry allocation and within-formula ranking as separate evaluation tasks.",
        "- Keep separate nodes for different structures of the same formula; control their total readout weight.",
        "- Distinguish chemical eligibility from cross-chemistry structural or energy context.",
        "- Require a graph model to beat plain means, formula balance, weighting, the fixed angle gate, and chemistry-matched controls.",
        "- Do not use candidate energies as input features if the goal is to predict those same stability labels.",
        "- Check the original embedding provenance before treating graph training as ready.",
        "",
        "The original JEPA loss uses formation-energy differences and a predictor conditioned on augmentation parameters.",
        "Neither feature alone guarantees that raw encoder vectors are invariant or that smaller distance means lower energy above hull.",
        "See [Crys-JEPA sections 4.1–4.2](https://arxiv.org/html/2605.14759v2#S4).",
        "The separation between energy prediction, stability classification and discovery yield is also emphasized in",
        "[Matbench Discovery](https://www.nature.com/articles/s42256-025-01055-1). We retain a fixed DFT reference hull",
        "and separate same-label oracle screening from independent physical validation.",
        "",
        "## 7. New evaluation set",
        "",
        "The collection protocol is in configs/fresh_evaluation.toml: 40,000 new generations in 40 batches with a new",
        "seed range. Model weights and all selection settings must be sealed first. No new set has been collected yet,",
        "and no unseen-test gain is claimed. Final collection is a later experiment after model design and validation.",
        "",
        "## Limits and reproduction",
        "",
        "All numbers use the same saved MatterSim-relaxed structures and corrected energies against the fixed MP2023 hull.",
        "The threshold is <=0.1 eV/atom. Current validation and test outcomes are already known; this is exploratory analysis.",
        "Uncertainty ranges use 1,000 paired chemical-system bootstrap draws and are unadjusted for multiple comparisons.",
        "",
        "Use the audit environment Python:",
        "",
        "```sh",
        "python -m reference_baselines.baselines.formula_balance",
        "python -m reference_baselines.evaluation.pre_gnn",
        "python -m reference_baselines.evaluation.matched_pairs",
        "python -m reference_baselines.datasets.fetch_jepa_audit",
        "python -m reference_baselines.checks.encoder_invariance  # add --scaler PATH when available",
        "python -m reference_baselines.reporting.pre_gnn",
        "```",
        "",
        "All selection IDs, pool shares, comparisons and input hashes are local in results/pre_gnn/.",
        "",
    ]
    (DOCS_DIR / "pre_gnn_report.md").write_text("\n".join(lines))
    tracker = [
        "# Pre-GNN progress tracker — 22 October 2026",
        "",
        "| Step | Status | Evidence / remaining work |",
        "|---|---|---|",
        "| Formula-balanced mean | Done | Validation/test, all four budgets; report section 1 |",
        "| Same chemistry and same formula quotas | Done | Both allocation anchors; shared unique representatives; section 2 |",
        "| Matched wins/losses and structure review | Done | Existing CIF/SOAP cases plus seeded pair study; section 3 |",
        "| Direct energy-above-hull comparison | Done | Same-label oracle, explicitly labelled; section 4 |",
        "| Original encoder invariance | Blocked on exact preprocessing | Checkpoint obtained; exact scaler and extraction script missing; section 5 |",
        "| Evidence for GNN design | Findings ready for review | Chemistry mix explains much of the global gain; no clear same-formula ranking gain |",
        "| Fresh evaluation protocol | Ready | configs/fresh_evaluation.toml |",
        "| Collect new final evaluation set | Later, after model freeze | No new candidates collected; no unseen-test claim |",
        "| Full GNN training | Not started | Design review and embedding provenance first |",
        "",
        "[Full report](pre_gnn_report.md). Do not mark all checks complete while encoder reproduction is unresolved.",
        "",
    ]
    (DOCS_DIR / "PRE_GNN_PROGRESS.md").write_text("\n".join(tracker))
    print("Wrote pre-GNN report and tracker")


if __name__ == "__main__":
    main()
