"""Report accepted fresh MatterSim checks and complete fixed-set re-relaxation."""

import json

from reference_baselines.core.audit_io import read_csv
from reference_baselines.evaluation.selection_audit import OUT
from reference_baselines.paths import DOCS_DIR


def main():
    fresh = json.loads((OUT / "fresh_mattersim.json").read_text())
    full = json.loads((OUT / "rerelaxed_evaluation.json").read_text())
    groups = {row["selection_group"]: row for row in full["group_metrics"]}
    methods = read_csv(OUT / "rerelaxed_method_metrics.csv")
    checked = read_csv(OUT / "rerelaxed_checked.csv")
    added = groups["added"]
    added_final = checked[checked.selection_group.eq("added")]
    novel_added_passes = int((added_final.usable & added_final.rerelaxed_passing & added_final.novel_after_relaxation).sum())
    cases = read_csv(OUT / "cases.csv").set_index("candidate_row")
    case_results = checked[checked.candidate_row.isin(cases.index)]
    lines = [
        "# Fresh MatterSim validation",
        "",
        "## Answer",
        "",
        f"Every one of the {added['count']:,} additional test candidates was run through fresh MatterSim energy/force calculations and atom-plus-cell relaxation.",
        f"All {added['converged']:,} converged. All {added['valid_geometry']:,} pass the periodic geometry checks, and all {added['composition_valid']:,} pass the fixed SMACT14 composition rules.",
        f"After re-relaxation, {added['passing_and_usable_before_dedup']} meet the energy-above-hull cutoff of 0.1 eV/atom. The other {added['count'] - added['passing_and_usable_before_dedup']} do not. Therefore, passing the angle filter does not mean that every candidate is stable.",
        f"Of the added stability passes, {novel_added_passes} are also unmatched against MP2023 after relaxation under the fixed structure-matching rule. This is novelty against that bank, not proof of a new synthesizable material.",
        "These counts are before removing any structures that merged during re-relaxation. The fixed-budget method comparison below includes that check.",
        "",
        "## What was run",
        "",
        "1. Fixed both existing selection lists. No candidates were generated, reranked, or added to fill gaps.",
        f"2. Recomputed energy and forces for all {fresh['candidate_count']:,} crystals selected by either method, including all added candidates and the original-method controls.",
        f"3. Verified saved energies: maximum difference {fresh['energy_absolute_difference_per_atom_max']:.2e} eV/atom; {fresh['pass_label_disagreements']} pass/fail changes at the saved structures.",
        "4. Re-relaxed atoms and unit cells with the same settings for both methods. Recomputed MP2020 corrections on each final structure and compared energies against the same fixed MP2023 hull.",
        "5. Checked geometry, composition, duplicates, and MP2023 novelty again. No refill was used after the new checks.",
        "",
        "## Complete re-relaxation counts",
        "",
        "| Selection group | Candidates | Converged | Valid geometry | Composition passes | Converged, valid stability passes before dedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["added", "removed", "shared"]:
        row = groups[name]
        lines.append(
            f"| {name} | {row['count']} | {row['converged']} | {row['valid_geometry']} | {row['composition_valid']} | {row['passing_and_usable_before_dedup']} |"
        )
    lines += [
        "",
        f"Across the full union, {full['full_union_count']:,} calculations were attempted; {full['converged_count']:,} converged. There were {full['errors']} calculation errors. Unconverged cases are unresolved and excluded from confirmed usable yield.",
        "Composition-rule failures are review flags. Single-state rules can reject real mixed-valence compounds; a passing assignment is not proof of actual oxidation states.",
        "",
        "## Fixed-budget comparison after all checks",
        "",
        "Each method keeps its original budget of 1,849 slots. Failed, unconverged, or duplicate slots are not refilled. Uniqueness is applied in the original selected order. Novelty is checked against all same-composition MP2023 polymorphs.",
        "",
        "| Method | Unique usable | Unique stability passes | Passes per 1,849 slots | Unique, novel stability passes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in methods.itertuples():
        name = (
            "Original Crys-JEPA"
            if row.method == "crys_jepa"
            else "1-degree gate then original score"
        )
        lines.append(
            f"| {name} | {row.unique_usable_no_refill} | {row.unique_passing} | {100 * row.yield_per_original_slot:.2f}% | {row.unique_novel_passing} |"
        )
    a = methods[methods.method.eq("crys_jepa")].iloc[0]
    b = methods[methods.method.eq("theta1_gate_then_original_score")].iloc[0]
    lines += [
        "",
        f"After these fresh checks, the angle rule has {int(b.unique_passing - a.unique_passing):+d} more unique stability passes, a change of {100 * (b.yield_per_original_slot - a.yield_per_original_slot):+.2f} percentage points per original slot.",
        "This is a separate, more complete validation stage. It does not replace or silently alter the original frozen 782-versus-644 result, which used the released relaxed structures and the earlier refill policy.",
        "",
        "## Examples after fresh relaxation",
        "",
        "These examples were chosen to explain wins and failures, not to estimate the population pass rate.",
        "",
        "| Row | Formula | Selection | Final hull distance (eV/atom) | Converged | Stability pass |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in case_results.itertuples():
        lines.append(
            f"| {row.candidate_row} | {row.formula} | {row.selection_group} | {row.rerelaxed_hull:.4f} | {row.converged} | {row.rerelaxed_passing} |"
        )
    lines += [
        "",
        "## Calculation settings and checks",
        "",
        "- Official MatterSim-v1.0.0-1M weights; MatterSim package 1.2.5; CPU inference. No model training.",
        f"- Checkpoint SHA256: `{fresh['checkpoint']['sha256']}`.",
        "- ASE FIRE optimizer; FrechetCellFilter allows atoms and cell to move; at most 300 additional steps; force/filter cutoff 0.05 eV/angstrom.",
        "- Same settings, fixed hull, corrections, and failure policy for both selection methods.",
        "- Accepted inference uses one graph per batch and matches the official ASE calculator in energy, forces, and input-order checks.",
        "- The first multi-structure legacy batching run was rejected because it disagreed with the individual calculator. Those outputs are marked rejected and are not used.",
        "- A fixed analytical basis buffer added by recent MatterSim versions is initialized from the package. No learned checkpoint parameter is missing or ignored.",
        "- MatterSim and basic validity checks support this model-based stability conclusion. Phonon stability, synthesis, and independent DFT remain untested; DFT is reserved for the final publication stage.",
        "",
        "## Files",
        "",
        "Accepted local outputs are in `../results/interpretation/`: `fresh_mattersim.csv`, `rerelaxed_checked.csv`, `rerelaxed_method_metrics.csv`, `rerelaxed_novelty.csv`, and `rerelaxed_all/` (final CIFs).",
        "The checkpoint and its pinned source are in `../data/mattersim/`. These generated data and model files stay out of Git.",
        "For the mechanism, structures, references, and embedding plots, see [the interpretation report](selection_interpretation.md).",
        "",
    ]
    (DOCS_DIR / "mattersim_validation.md").write_text("\n".join(lines))
    print("Wrote fresh MatterSim validation report.")


if __name__ == "__main__":
    main()
