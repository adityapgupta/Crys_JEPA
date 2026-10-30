"""Write the reference-pool versus weighted-mean attribution report."""

from reference_baselines.baselines.mean_ablation import OUT
from reference_baselines.core.audit_io import read_csv
from reference_baselines.paths import DOCS_DIR


LABELS = {
    "chemical_mean": "Original references, plain mean",
    "chemical_weighted": "Original references, weighted mean",
    "union_cosine5_mean": "Original + five cosine neighbours, plain mean",
    "union_cosine5_weighted": "Original + five cosine neighbours, weighted mean",
    "intersection_cosine5_fallback_mean": "Chemical intersection, plain mean",
    "intersection_cosine5_fallback_weighted": "Chemical intersection, weighted mean",
    "cosine5_mean": "Five cosine neighbours only, plain mean",
    "cosine5_weighted": "Five cosine neighbours only, weighted mean",
    "theta1_top10_fallback_mean": "1° reference pool, plain mean",
    "theta1_top10_fallback_weighted": "1° reference pool, weighted mean",
    "gate_then_chemical_mean": "1° candidate gate + original plain mean (reported rule)",
    "gate_then_chemical_weighted": "1° candidate gate + original weighted mean",
}


def main():
    metrics = read_csv(OUT / "metrics.csv")
    contrasts = read_csv(OUT / "contrasts.csv")
    selections = read_csv(OUT / "selections.csv")
    test5 = metrics[metrics.split.eq("test") & metrics.budget_percent.eq(5)].set_index(
        "method"
    )
    chosen5 = selections[selections.split.eq("test") & selections.budget_percent.eq(5)]
    angle_rows = set(
        chosen5.loc[chosen5.method.eq("theta1_top10_fallback_mean"), "candidate_row"]
    )
    gate_rows = set(
        chosen5.loc[chosen5.method.eq("gate_then_chemical_mean"), "candidate_row"]
    )
    lines = [
        "# Reference pools and mean weighting",
        "",
        "## What was actually tested before",
        "",
        "The reported 1° rule did not use a weighted mean. It first prioritizes candidates with any known",
        "reference within 1°, then ranks them by the original plain chemical-reference mean.",
        "It changes candidate ranking. It does not replace that mean with new reference weights.",
        "",
        "The paper's screening score is a uniform mean of squared embedding distances (Eq. 11, section 4.2).",
        "Its energy-weighted training loss is separate from this screening score.",
        "Source: [Crys-JEPA paper](https://arxiv.org/html/2605.14759v2#S4.SS2).",
        "",
        "## Fair controls",
        "",
        "We change the reference pool and averaging rule separately. All scores average the same raw squared",
        "Euclidean embedding distance. Lower is better. We do not average embeddings before computing distance.",
        "",
        "- Original pool: all known MP20 references within the candidate chemical system and its subsystems.",
        "- Enhanced pool: original pool plus five global cosine neighbours, with duplicate reference IDs removed.",
        "- Intersection control: only those five neighbours allowed by chemistry; if empty, use the original pool.",
        "- Neighbours-only control: five global cosine neighbours, without the original pool.",
        "- Angle-pool control: references within 1° among the nearest ten; if empty, use the original pool.",
        "- Candidate-gate control: keep the original pool and change candidate priority using the 1° gate.",
        "",
        "Each pool gets both a plain mean and a weighted mean. The gate also gets both means.",
        "Weighted mean uses w = exp(-(d - smallest d in the pool) / 10), then score = sum(w*d)/sum(w).",
        "Here d is squared embedding distance. Nearby references get more weight; no candidate energies enter the weights.",
        "The scale 10 is the existing chemical_softmax_t10 setting. It was fixed for this study, not tuned on these outcomes.",
        "This tests one stated weighting rule. It does not cover all possible weights or physical phase fractions.",
        "",
        "All rules receive the same candidates, MP20 bank, budget, duplicate checks, fixed MP2023 hull and saved MatterSim",
        "energies. A stability pass requires energy above hull <= 0.1 eV/atom. Extra re-relaxation counts are not mixed in.",
        "",
    ]
    lines += [
        "## Main result at 5%",
        "",
        "| Reference pool | Plain mean passes | Weighted mean passes |",
        "|---|---:|---:|",
    ]
    for pool, label in [
        ("chemical", "Original chemical references"),
        ("theta1_top10_fallback", "1° references, with chemical fallback"),
        ("union_cosine5", "Original plus five cosine neighbours"),
    ]:
        lines.append(
            f"| {label} | {test5.loc[pool + '_mean', 'passing']:,} | {test5.loc[pool + '_weighted', 'passing']:,} |"
        )
    lines += [
        "",
        "Weighting alone adds 48 passes (644 to 692). The angle-reference pool with a plain mean adds 141",
        "(644 to 785). Holding weighting fixed, the angle pool adds 101 (692 to 793).",
        "Thus this weighting rule does not explain the full angular-screening gain.",
        "",
        "However, adding five cosine neighbours to the original pool gives only 38 extra passes with a plain mean",
        "and loses 44 passes when compared with the weighted original pool. Its uncertainty ranges include zero.",
        "We cannot claim that simply enlarging the reference set reliably improves screening.",
        "",
        f"The angle-reference plain mean and the reported candidate gate share {len(angle_rows & gate_rows):,} of {len(angle_rows):,}",
        f"selections ({100 * len(angle_rows & gate_rows) / len(angle_rows):.1f}%). Their passing counts differ by only three (785 versus 782).",
        "This still leaves angular candidate screening as a strong explanation. These results alone do not prove",
        "that better reference identities, rather than the gate and smaller pools, cause the gain.",
        "",
    ]
    for split in ["test", "validation"]:
        frame = metrics[metrics.split.eq(split)]
        lines += [
            f"## {split.title()} results: passing unique candidates",
            "",
            "| Reference pool and mean | 1% | 2% | 5% | 7% |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, label in LABELS.items():
            cells = []
            for percent in [1, 2, 5, 7]:
                row = frame[
                    frame.method.eq(name) & frame.budget_percent.eq(percent)
                ].iloc[0]
                cells.append(
                    f"{row.passing:,}/{row.selected:,} ({100 * row.pass_rate:.1f}%)"
                )
            lines.append("| " + label + " | " + " | ".join(cells) + " |")
        lines += [""]
    lines += [
        "## What each change adds: test at 5%",
        "",
        "These are changes in passing counts at the same 1,849-candidate budget.",
        "",
        "| One change at a time | Extra passes | Gain in percentage points | Descriptive 95% range |",
        "|---|---:|---:|---:|",
    ]
    selected_effects = {
        "weighting_only": "Weight original references; same pool",
        "union_cosine5_pool_only": "Add five cosine neighbours; keep plain mean",
        "union_cosine5_pool_with_same_weights": "Add five cosine neighbours; keep weighted mean",
        "union_cosine5_weighting_with_same_pool": "Weight enhanced pool; same enhanced pool",
        "theta1_top10_fallback_pool_only": "Use 1° reference pool; keep plain mean",
        "theta1_top10_fallback_pool_with_same_weights": "Use 1° reference pool; keep weighted mean",
        "gate_only": "Add 1° candidate gate; keep plain chemical mean",
        "gate_with_same_weights": "Add 1° candidate gate; keep weighted chemical mean",
        "angle_pool_vs_candidate_gate_plain": "Angle-reference pool versus candidate gate; plain means",
        "angle_pool_vs_candidate_gate_weighted": "Angle-reference pool versus candidate gate; weighted means",
    }
    for name, label in selected_effects.items():
        row = contrasts[
            contrasts.split.eq("test")
            & contrasts.budget_percent.eq(5)
            & contrasts.effect.eq(name)
        ].iloc[0]
        lines.append(
            f"| {label} | {row.extra_passes:+,} | {row.gain_pp:+.2f} | {row.ci_low_pp:+.2f} to {row.ci_high_pp:+.2f} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "The original weighted mean tests whether weighting alone explains the gain.",
        "The enhanced plain mean tests whether adding neighbours helps without weighting.",
        "The enhanced weighted mean versus original weighted mean tests the added pool under the same weighting rule.",
        "The interaction in contrasts.csv shows whether the two changes add independently; do not add their gains blindly.",
        "The angle candidate gate is a different operation from changing a reference pool. Its result must be named separately.",
        "Global cosine neighbours may have different elements. The intersection control keeps chemistry fixed.",
        "This is a screening attribution study. It does not prove better physical decomposition references or better generation.",
        "Both validation and test outcomes were already inspected. All these comparisons are exploratory, not a new unseen test.",
        "Bootstrap ranges are descriptive and unadjusted for multiple comparisons. No winner or temperature is chosen here.",
        "",
        "## Reproduce",
        "",
        "Use reference_baselines/environment/audit/bin/python for these commands:",
        "",
        "```sh",
        "python -m reference_baselines.baselines.mean_ablation",
        "python -m reference_baselines.evaluation.mean_ablation",
        "python -m reference_baselines.reporting.mean_ablation",
        "```",
        "",
        "Scores, exact selection IDs, pool sizes, fallbacks, contrasts, and input hashes are saved under",
        "reference_baselines/results/mean_ablation/ (local, Git-ignored). No new training or relaxation is run.",
        "",
    ]
    (DOCS_DIR / "mean_ablation.md").write_text("\n".join(lines))
    print("Wrote", DOCS_DIR / "mean_ablation.md")


if __name__ == "__main__":
    main()
