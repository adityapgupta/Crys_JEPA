"""Create readable case studies and scientific plots from the fixed-result audit."""

from reference_baselines.core.audit_io import read_csv

from itertools import product
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Structure

from reference_baselines.evaluation.selection_audit import OUT
from reference_baselines.paths import DOCS_DIR


COLORS = {
    "added": "#147d64",
    "removed": "#b54d3e",
    "shared": "#4169a1",
    "neither": "#b3bac4",
}


def figures(frame, figure_dir):
    test = frame[frame.split.eq("test")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    groups = ["shared", "added", "removed"]
    rates = [test[test.selection_group.eq(g)].passing.mean() * 100 for g in groups]
    axes[0, 0].bar(groups, rates, color=[COLORS[g] for g in groups])
    axes[0, 0].set(
        ylabel="Saved MatterSim pass rate (%)",
        ylim=(0, 55),
        title="A. What changed in the selected set?",
    )
    for i, (g, rate) in enumerate(zip(groups, rates)):
        selected = test[test.selection_group.eq(g)]
        axes[0, 0].text(
            i,
            rate + 1,
            f"{selected.passing.sum()}/{len(selected)}",
            ha="center",
            fontsize=9,
        )
    for g in ["added", "removed"]:
        values = test[test.selection_group.eq(g)]
        axes[0, 1].hist(
            values.nearest_angle_degrees,
            bins=np.linspace(0, 10, 41),
            density=False,
            histtype="step",
            linewidth=2,
            label=g,
            color=COLORS[g],
        )
        axes[1, 0].scatter(
            values.original_score,
            values.nearest_angle_degrees,
            s=6,
            alpha=0.3,
            color=COLORS[g],
            label=g,
            rasterized=True,
        )
    axes[0, 1].set(
        xlabel="Nearest known-reference angle (degrees)",
        ylabel="Selected candidates",
        title="B. Close neighbours (0–10 degrees shown)",
    )
    axes[0, 1].axvline(1, ls="--", color="black", lw=1)
    axes[0, 1].legend()
    axes[1, 0].set(
        xlabel="Original mean-distance score (smaller ranks first)",
        ylabel="Nearest angle (degrees)",
        ylim=(0, 10),
        title="C. Close neighbour does not imply low pool mean",
    )
    axes[1, 0].axhline(1, ls="--", color="black", lw=1)
    for g in ["added", "removed"]:
        values = test[test.selection_group.eq(g)]
        med = values[["centroid_squared_distance", "chemical_pool_spread"]].median()
        x = 0 if g == "added" else 1
        axes[1, 1].bar(
            x,
            med.iloc[0],
            color=COLORS[g],
            label="Distance to pool centre" if x == 0 else None,
        )
        axes[1, 1].bar(
            x,
            med.iloc[1],
            bottom=med.iloc[0],
            color="#babfc7",
            label="Pool spread" if x == 0 else None,
        )
    axes[1, 1].set(
        xticks=[0, 1],
        xticklabels=["added", "removed"],
        ylabel="Squared-distance components",
        title="D. Pool centre and spread: group medians",
    )
    axes[1, 1].legend(fontsize=8)
    fig.savefig(figure_dir / "selection_explanation.png", dpi=160)
    plt.close(fig)

    known = read_csv(OUT / "known_embedding_projection.csv")
    spectrum = read_csv(OUT / "pca_spectrum.csv")
    pairs = read_csv(OUT / "known_pair_diagnostics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    points = axes[0, 0].scatter(
        known.PC1,
        known.PC2,
        c=known.formation_energy_per_atom,
        s=2,
        alpha=0.5,
        cmap="viridis",
        rasterized=True,
    )
    fig.colorbar(points, ax=axes[0, 0], label="Known DFT formation energy (eV/atom)")
    axes[0, 0].set(
        xlabel="PC1",
        ylabel="PC2",
        title="A. Known crystals: only 21.7% of variance shown",
    )
    axes[0, 1].plot(
        spectrum.component.head(60), 100 * spectrum.cumulative_variance.head(60)
    )
    axes[0, 1].axhline(90, color="black", lw=1, ls="--")
    axes[0, 1].set(
        xlabel="Number of components",
        ylabel="Cumulative variance (%)",
        title="B. 512 columns; 28 components explain 90%",
    )
    axes[1, 0].hist(
        known.embedding_norm,
        bins=60,
        histtype="step",
        density=True,
        lw=2,
        label="Known MP20",
    )
    validation = frame[frame.split.eq("validation")]
    axes[1, 0].hist(
        validation.embedding_norm,
        bins=60,
        histtype="step",
        density=True,
        lw=2,
        label="Generated validation",
    )
    axes[1, 0].set(
        xlabel="Embedding vector length",
        ylabel="Density",
        title="C. Vector lengths occupy a narrow range",
    )
    axes[1, 0].legend()
    density = axes[1, 1].hexbin(
        pairs.angle_degrees,
        pairs.formation_energy_gap,
        gridsize=45,
        mincnt=1,
        bins="log",
        cmap="magma",
    )
    fig.colorbar(density, ax=axes[1, 1], label="Known-crystal pairs per bin")
    axes[1, 1].set(
        xlabel="Known-known embedding angle (degrees)",
        ylabel="DFT formation-energy gap (eV/atom)",
        title="D. Energy signal, with substantial overlap",
    )
    fig.savefig(figure_dir / "embedding_diagnostics.png", dpi=160)
    plt.close(fig)


def draw_cell(ax, structure, title):
    vertices = (
        np.array(list(product([0, 1], repeat=3)), float) @ structure.lattice.matrix
    )
    for i, a in enumerate(product([0, 1], repeat=3)):
        for j, b in enumerate(product([0, 1], repeat=3)):
            if j > i and sum(x != y for x, y in zip(a, b)) == 1:
                ax.plot(
                    *np.vstack([vertices[i], vertices[j]]).T, color="#7c8490", lw=0.7
                )
    points = (structure.frac_coords % 1) @ structure.lattice.matrix
    species = [site.specie.symbol for site in structure]
    for i, symbol in enumerate(sorted(set(species))):
        selected = points[np.array(species) == symbol]
        ax.scatter(
            *selected.T,
            s=35,
            label=symbol,
            edgecolors="black",
            linewidths=0.3,
            color=plt.get_cmap("tab10")(i),
        )
    extent = np.ptp(vertices, axis=0)
    ax.set_box_aspect(np.maximum(extent, 0.1))
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def crystal_figure(figure_dir):
    pairs = read_csv(OUT / "case_pairs.csv")
    cases = read_csv(OUT / "cases.csv").set_index("candidate_row")
    chosen = pd.concat(
        [
            pairs[pairs.kind.eq("angle_win_same_formula")].head(2),
            pairs[pairs.kind.eq("angle_loss_same_formula")].head(2),
        ]
    )
    fig = plt.figure(figsize=(10, 12), layout="constrained")
    for i, pair in enumerate(chosen.itertuples()):
        for j, row in enumerate([pair.added_row, pair.removed_row]):
            entry = cases.loc[row]
            structure = Structure.from_file(OUT / f"structures/candidate_{row}.cif")
            ax = fig.add_subplot(len(chosen), 2, i * 2 + j + 1, projection="3d")
            title = f"{entry.formula}, row {row}, {entry.selection_group}\nHull {entry.e_above_hull_proxy:.3f} eV/atom; angle {entry.nearest_angle_degrees:.2f} degrees"
            draw_cell(ax, structure, title)
    fig.savefig(figure_dir / "crystal_examples.png", dpi=160)
    plt.close(fig)


def main():
    figure_dir = DOCS_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    frame = read_csv(OUT / "candidate_projection.csv")
    figures(frame, figure_dir)
    crystal_figure(figure_dir)
    stats = json.loads((OUT / "embedding_audit.json").read_text())
    fresh = read_csv(OUT / "fresh_mattersim.csv")
    cases = read_csv(OUT / "cases.csv")
    fresh_counts = fresh.groupby("selection_group").fresh_passing.sum()
    pilot = read_csv(OUT / "rerelaxed_pilot.csv")
    lines = [
        "# Why the angle rule helped, and where it fails",
        "",
        "This is an explanation of the fixed results. It does not tune another method on the opened test set.",
        "",
        "## The gain is a swap, not new crystal generation",
        "",
        "Both methods select 1,849 existing generated crystals. They share 455. The angle rule adds 1,394 and drops 1,394.",
        "The added group has 613 saved MatterSim stability passes; the dropped group has 475. The shared group has 169. Thus 613 - 475 = 138 additional passes.",
        "",
        "A crystal can pass geometry and composition checks while failing the stability cutoff. These are separate checks.",
        "",
        "![Selection explanation](figures/selection_explanation.png)",
        "",
        "## Why the two rules choose different crystals",
        "",
        "Original Crys-JEPA uses the mean squared distance to every eligible chemical reference. A close individual neighbour can be hidden by distant members of that pool.",
        "The mean score equals distance to the pool centre plus the pool's internal spread. This is an exact identity, not a fitted explanation. It means the score depends on both the candidate and the reference pool's distribution.",
        "Within an exact-composition pair, the reference pool and its spread are identical. The original ranking difference within that pair comes from distance to the same pool centre. Pool-spread differences matter when comparing candidates across different chemical systems.",
        "For the added test group, the median original score is 145.7 and median pool spread is 62.9. For the dropped group, these are 89.6 and 52.5. Centre distance also differs; pool spread alone does not explain the gain.",
        "The gate instead asks whether at least one known embedding is very close. Added candidates have median nearest angle 0.91 degrees; dropped candidates have 2.74 degrees.",
        "Only 5.8% of added candidates have their nearest cosine reference in the original chemical pool. Most close references have different elements. They may provide structural or energy context, but cannot be physical decomposition phases with missing elements.",
        "",
        "## Exact-composition examples",
        "",
        "These examples are selected after seeing outcomes. They illustrate mechanisms; they do not estimate the frequency of each mechanism. Each pair has the same reduced formula. Pass means saved MatterSim hull distance <=0.1 eV/atom.",
        "",
        "| Formula | Row | Selection | Original score | Nearest angle | Hull distance | Pass |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in cases.itertuples():
        lines.append(
            f"| {row.formula} | {row.candidate_row} | {row.selection_group} | {row.original_score:.2f} | {row.nearest_angle_degrees:.2f} degrees | {row.e_above_hull_proxy:.3f} | {'yes' if row.passing else 'no'} |"
        )
    lines += [
        "",
        "Li3Ni2O5 is a clear win: row 310813 passes at 0.028 eV/atom but has worse original score than removed row 22551, which fails at 0.234. Its close global reference is Li4Fe3(NiO5)2, outside the Li-Ni-O pool. A LiNiO2 reference in that pool has even closer geometry-only SOAP, despite a larger embedding angle. Embedding angle is therefore not identical to structural distance.",
        "YbAl3 shows a reference-pool issue: five of its six eligible references are pure-element Al or Yb polymorphs, and only one is a compound (YbAl2). Equal reference weights therefore give five-sixths of the mean to elemental structures. The passing added candidate has original score 161.85, while the failing removed candidate has score 98.02. Its nearest global reference is UGaNi, which supplies cross-chemistry context, not a physical decomposition phase. This illustrates why chemical eligibility alone does not make a uniform mean informative.",
        "CeSn2 is a counterexample: added row 59917 is only 0.85 degrees from a known reference, yet fails at 0.190 eV/atom. Removed row 66383 passes at -0.022 despite a nearest angle of 21.70 degrees. Even a same-element close reference does not guarantee stability at the candidate's composition.",
        "![Single-cell views of matched examples](figures/crystal_examples.png)",
        "Atoms and unit-cell edges are shown; bonds are not inferred. Different cell choices can make equivalent structures look different. Use CIFs and quantitative checks alongside these pictures.",
        "",
        "## What the embeddings contain",
        "",
        "The encoder returns one 512-number vector for a whole crystal. Its training loss uses cosine similarity and formation-energy differences. Its original screening score uses squared Euclidean distance. Neither is trained directly to give energy above hull.",
        f"Known vector lengths have a standard deviation of {100 * stats['norm_coefficient_of_variation']:.2f}% of their mean. Direction is therefore a major source of distance variation. This helps explain why Euclidean and cosine nearest-neighbour results are similar; the main change here is local gating versus a whole-pool mean.",
        f"The first two PCA axes show only {100 * stats['top2_variance']:.1f}% of variance. The first ten show {100 * stats['top10_variance']:.1f}%; {stats['components_for_90_percent']} components show 90%. This is structure in the representation, not proof that 28 physical variables explain stability.",
        f"Across 30,000 random known-crystal pairs, angle and absolute formation-energy gap have Spearman correlation {stats['random_pair_cosine_vs_formation_gap_spearman']:.2f}. This supports an energy signal with substantial overlap. Similar formation energies across different chemistries do not imply similar energy above hull.",
        f"The most retrieved 1% of known references receive {100 * stats['top1_percent_known_references_share_of_validation_top5_links']:.1f}% of validation top-5 links. Some references are more popular, but a tiny group does not account for all retrievals.",
        "![Embedding diagnostics](figures/embedding_diagnostics.png)",
        "",
        "## Are the additional crystals valid?",
        "",
        "All 1,394 added test crystals pass finite-cell, periodic atom-distance, volume, and SMACT14 composition checks. These include charge-balance and electronegativity rules, with the same allowance for metal-only alloys. These are plausibility checks, not synthesis or phonon stability.",
        "The dropped group has two composition-rule failures. Single-oxidation-state rules can reject real mixed-valence compounds, so these are review flags rather than proven invalid materials.",
        "Passing the rule is also not a chemical proof: for Li3Ni2O5, one allowed SMACT14 assignment is Li(+1), Ni(+1), O(-1). This is only a charge-balance witness, not an identified oxidation state or bonding pattern. Chemistry review remains useful even for a passing composition.",
        f"Fresh MatterSim energy and force calculations were run for all {len(fresh):,} crystals in the union of the two selections, including every added crystal. No training was done.",
        f"Fresh single-structure stability counts: added {int(fresh_counts['added'])}, dropped {int(fresh_counts['removed'])}, shared {int(fresh_counts['shared'])}. Thus original selects {int(fresh_counts['removed'] + fresh_counts['shared'])} fresh passes and the angle rule selects {int(fresh_counts['added'] + fresh_counts['shared'])}.",
        f"A seeded and illustrative pilot re-relaxed {len(pilot)} crystals with atoms and cell free to move. {int(pilot.converged.sum())} met the force/cell convergence cutoff. This pilot is not a full-population re-relaxation result.",
        "The later complete check re-relaxed all 3,243 selected crystals. All 1,394 added candidates converged; 635 pass stability after the extra relaxation. After validity, uniqueness, and convergence checks, the fixed angle selection has 808 unique stability passes versus 662 for the original selection, with no refill. See [full fresh validation](mattersim_validation.md) for the changed evaluation stage and all failure rules. The earlier 782-versus-644 result remains the frozen released-structure comparison.",
        "The first multi-structure batched inference attempt was rejected: MatterSim 1.2.5's legacy graph batching produced results inconsistent with its individual ASE calculator. Accepted outputs use one graph per inference batch and are tested against the calculator. The rejected outputs are retained locally with a failure note and do not support any reported gain.",
        "",
        "## What this suggests for a future model",
        "",
        "The evidence supports learning how to combine a close local reference with chemical-pool context. It does not yet show that a GNN or transformer is required.",
        "A useful model should distinguish reference roles: same-composition, same-system/subsystem, and cross-system context. It should see composition differences and local distances. Uniform averaging can hide a useful neighbour; a graph that averages indiscriminately could repeat that problem.",
        "Start later with learned reference weighting and an explicit uniform-weight control. Add graph messages only if they help beyond that weighting and the fixed angle rule. Train and select on training/validation; reserve new generated data for a new unseen test. No new model or embedding transform was selected here.",
        "Before training, check the original encoder on equivalent atom orders, translations, rotations, and unit-cell choices. This study uses saved vectors and has not established those invariances. A representation artifact should not be passed to a larger model as physical evidence.",
        "The 1-degree threshold was chosen from validation results. It is not a universal chemical constant or a calibrated stability probability.",
        "",
        "## Reproduce and inspect",
        "",
        "Visible code: `evaluation/selection_audit.py`, `graphs/embedding_audit.py`, `structure/validity.py`, `structure/case_study.py`, `evaluation/mattersim_audit.py`, and `reporting/interpretation.py`.",
        "Local tables, full reference IDs, CIFs, and checkpoint provenance are in `../results/interpretation/`. Original frozen metrics are unchanged.",
        "Run the modules in that order, then this report module. SMACT and MatterSim use the separate ignored `environment/audit/` environment. The main environment and frozen baseline inputs were not replaced.",
        "",
        "Sources: [Crys-JEPA energy-weighted cosine loss and screening](https://arxiv.org/html/2605.14759v2#S4), [SMACT composition-rule limits](https://smact.readthedocs.io/en/latest/examples/validity.html), [nearest-neighbour popularity in high dimensions](https://www.jmlr.org/papers/v11/radovanovic10a.html).",
        "",
    ]
    (DOCS_DIR / "selection_interpretation.md").write_text("\n".join(lines))
    print("Wrote", DOCS_DIR / "selection_interpretation.md")


if __name__ == "__main__":
    main()
