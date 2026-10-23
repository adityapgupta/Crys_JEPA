"""Audit complete re-relaxations without reranking the fixed test selections."""

from reference_baselines.core.audit_io import read_csv

from concurrent.futures import ProcessPoolExecutor
import json
import warnings

import pandas as pd
from pymatgen.core import Structure

from reference_baselines.core.io import sha256
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.evaluation.selected_duplicates import compare_group
from reference_baselines.evaluation.selection_audit import OUT, ORIGINAL, ANGLE
from reference_baselines.paths import RESULTS_DIR
from reference_baselines.structure.heldout import init_novelty, novelty_group


def main():
    warnings.filterwarnings(
        "ignore", message="gcd is deprecated", category=FutureWarning
    )
    original = read_csv(OUT / "candidate_audit.csv")
    selected = original[
        original.split.eq("test") & original.selection_group.ne("neither")
    ]
    parts = [read_csv(OUT / f"rerelaxed_all_shard{i}of4.csv") for i in range(4)]
    relaxed = pd.concat(parts, ignore_index=True).sort_values("candidate_row")
    if not relaxed.candidate_row.is_unique or set(relaxed.candidate_row) != set(
        selected.candidate_row
    ):
        raise ValueError(
            "Full selected union is not yet re-relaxed; no partial-set performance claim"
        )
    if relaxed.steps.gt(300).any():
        raise ValueError("Re-relaxation exceeded the shared 300-step cap")
    converged = relaxed[relaxed.converged]
    if (
        converged.final_fmax.gt(0.050001).any()
        or converged.filtered_fmax.gt(0.050001).any()
    ):
        raise ValueError(
            "Convergence needs both actual atomic forces and cell-filter forces <=0.05"
        )
    for i in range(4):
        config = json.loads(
            (OUT / f"rerelaxed_all_protocol_shard{i}of4.json").read_text()
        )
        if config["max_steps"] != 300 or config["force_cutoff"] != 0.05:
            raise ValueError("Shards do not use the fixed re-relaxation protocol")
    relaxed.to_csv(OUT / "rerelaxed_all.csv", index=False)
    valid = read_csv(OUT / "validity_checks.csv").set_index("candidate_row")
    relaxed["composition_valid"] = valid.loc[
        relaxed.candidate_row, "composition_valid"
    ].to_numpy(bool)
    relaxed["usable"] = (
        relaxed.status.eq("ok")
        & relaxed.converged
        & relaxed.final_geometry_valid
        & relaxed.composition_valid
    )
    structures = []
    for row in relaxed[relaxed.usable].itertuples():
        structure = Structure.from_file(
            OUT / f"rerelaxed_all/candidate_{row.candidate_row}.cif"
        )
        structures.append((row.candidate_row, structure))
    groups = {}
    for row, structure in structures:
        groups.setdefault(structure.composition.reduced_formula, []).append(
            (row, structure)
        )
    pair_tasks = [
        (formula, sorted(records))
        for formula, records in groups.items()
        if len(records) > 1
    ]
    pairs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for records in pool.map(compare_group, pair_tasks, chunksize=8):
            pairs.extend(records)
    pd.DataFrame(pairs, columns=["candidate_row_a", "candidate_row_b"]).to_csv(
        OUT / "rerelaxed_duplicate_pairs.csv", index=False
    )
    adjacency = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    novelty = []
    with ProcessPoolExecutor(max_workers=4, initializer=init_novelty) as pool:
        for records in pool.map(novelty_group, list(groups.items()), chunksize=8):
            novelty.extend(records)
    novelty = pd.DataFrame(novelty).set_index("candidate_row")
    novelty.to_csv(OUT / "rerelaxed_novelty.csv")
    relaxed = relaxed.set_index("candidate_row", drop=False)
    relaxed["novel_after_relaxation"] = novelty.novel_against_mp2023.reindex(
        relaxed.index
    ).fillna(False)
    relaxed.to_csv(OUT / "rerelaxed_checked.csv", index=False)
    methods = read_csv(RESULTS_DIR / "heldout/selections.csv")
    metrics = []
    for method in [ORIGINAL, ANGLE]:
        chosen = (
            methods[methods.method.eq(method)]
            .sort_values("rank")
            .candidate_row.to_numpy()
        )
        f = relaxed.loc[chosen]
        usable = chosen[f.usable.to_numpy()]
        unique = unique_order(usable, adjacency)
        passing = relaxed.loc[unique, "rerelaxed_passing"].to_numpy(bool)
        novel = relaxed.loc[unique, "novel_after_relaxation"].to_numpy(bool)
        metrics.append(
            {
                "method": method,
                "fixed_budget": len(chosen),
                "converged": int(f.converged.sum()),
                "usable": int(f.usable.sum()),
                "stability_passing_before_dedup": int(
                    (f.rerelaxed_passing & f.usable).sum()
                ),
                "unique_usable_no_refill": len(unique),
                "unique_passing": int(passing.sum()),
                "yield_per_original_slot": float(passing.sum() / len(chosen)),
                "unique_novel_passing": int((passing & novel).sum()),
            }
        )
    pd.DataFrame(metrics).to_csv(OUT / "rerelaxed_method_metrics.csv", index=False)
    group_metrics = []
    for name, f in relaxed.groupby("selection_group"):
        group_metrics.append(
            {
                "selection_group": name,
                "count": len(f),
                "converged": int(f.converged.sum()),
                "errors": int(f.status.ne("ok").sum()),
                "valid_geometry": int(f.final_geometry_valid.sum()),
                "composition_valid": int(f.composition_valid.sum()),
                "passing_and_usable_before_dedup": int(
                    (f.rerelaxed_passing & f.usable).sum()
                ),
                "maximum_final_force": float(f.final_fmax.max()),
            }
        )
    report = {
        "full_union_count": len(relaxed),
        "group_metrics": group_metrics,
        "methods": metrics,
        "matched_pairs_after_relaxation": len(pairs),
        "converged_count": int(relaxed.converged.sum()),
        "errors": int(relaxed.status.ne("ok").sum()),
        "selection_unchanged": True,
        "refill_after_relaxation": False,
        "unconverged_policy": "Retained as unresolved; excluded from confirmed usable/passing yield",
        "validity_policy": "Require convergence, finite ordered periodic geometry, SMACT14 composition rule; composition flags are heuristic",
        "no_independent_DFT": True,
        "no_new_model_training": True,
        "sha256": {
            name: sha256(OUT / name)
            for name in [
                "rerelaxed_checked.csv",
                "rerelaxed_duplicate_pairs.csv",
                "rerelaxed_novelty.csv",
                "rerelaxed_method_metrics.csv",
            ]
        },
    }
    (OUT / "rerelaxed_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
