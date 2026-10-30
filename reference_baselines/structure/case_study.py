"""Export fixed-selection wins, losses, and representative examples with references."""

from reference_baselines.core.audit_io import read_csv

import json
import warnings

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from scipy.stats import spearmanr

from reference_baselines.core.angular import raw_distances
from reference_baselines.evaluation.selection_audit import OUT
from reference_baselines.paths import ROOT, DATA_DIR, RESULTS_DIR
from reference_baselines.structure.descriptors import (
    setup,
    normalized_fingerprint,
    geometry_atoms,
)


def pick_cases(frame):
    cases, pairs, used = [], [], set()
    frame = frame[frame.split.eq("test")].copy()
    added, removed = (
        frame[frame.selection_group.eq("added")],
        frame[frame.selection_group.eq("removed")],
    )
    for kind, good, bad, limit in [
        (
            "angle_win_same_formula",
            added[added.e_above_hull_proxy.between(-0.05, 0.08)],
            removed[removed.e_above_hull_proxy.between(0.12, 2)],
            3,
        ),
        (
            "angle_loss_same_formula",
            added[added.e_above_hull_proxy.between(0.12, 2)],
            removed[removed.e_above_hull_proxy.between(-0.05, 0.08)],
            2,
        ),
    ]:
        merged = good.merge(bad, on="formula", suffixes=("_added", "_removed"))
        # Avoid presenting duplicate-blocking as an original raw-score failure.
        merged = merged[merged.original_score_added > merged.original_score_removed]
        merged = (
            merged.sort_values(
                ["formula", "candidate_row_added", "candidate_row_removed"]
            )
            .drop_duplicates("formula")
            .head(limit)
        )
        for _, row in merged.iterrows():
            pair = f"{kind}_{len(pairs) + 1}"
            a, b = int(row.candidate_row_added), int(row.candidate_row_removed)
            pairs.append(
                {
                    "pair": pair,
                    "kind": kind,
                    "formula": row.formula,
                    "added_row": a,
                    "removed_row": b,
                }
            )
            for candidate in [a, b]:
                if candidate not in used:
                    cases.append(
                        {"candidate_row": candidate, "case_reason": kind, "pair": pair}
                    )
                    used.add(candidate)
    # Median-angle examples make the report less dependent on spectacular wins.
    for name in ["added", "removed"]:
        group = frame[frame.selection_group.eq(name)]
        target = group.nearest_angle_degrees.median()
        ordered = group.assign(
            delta=(group.nearest_angle_degrees - target).abs()
        ).sort_values(["delta", "candidate_row"])
        row = next(int(r) for r in ordered.candidate_row if r not in used)
        cases.append(
            {"candidate_row": row, "case_reason": f"median_angle_{name}", "pair": ""}
        )
        used.add(row)
    return pd.DataFrame(cases).merge(
        frame, on="candidate_row", validate="one_to_one"
    ), pd.DataFrame(pairs)


def load_candidates(frame):
    structures = {}
    for folder, group in frame.groupby("eval_folder"):
        raw = pd.read_json(
            DATA_DIR
            / f"release/exp_logs/base/mp_20/eval/{folder}/relaxed_mattersim.json"
        )
        for row in group.itertuples():
            structures[row.candidate_row] = Structure.from_dict(
                raw.loc[row.idx, "structure"]
            )
    return structures


def main():
    warnings.filterwarnings(
        "ignore", message="gcd is deprecated", category=FutureWarning
    )
    frame = read_csv(OUT / "candidate_audit.csv")
    cases, pairs = pick_cases(frame)
    cases.to_csv(OUT / "cases.csv", index=False)
    pairs.to_csv(OUT / "case_pairs.csv", index=False)
    structures = load_candidates(cases)
    ref_vectors = np.asarray(np.load(ROOT / "embeddings/data/embeddings.npy"), float)
    query_vectors = np.load(
        ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r"
    )
    metadata = read_csv(ROOT / "embeddings/data/metadata.csv")
    source = read_csv(DATA_DIR / "mp_20_train.csv.gz").set_index("material_id")
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    setup()
    soap = {
        profile: np.load(RESULTS_DIR / f"soap_{profile}.npy")
        for profile in ["physical", "density_normalized"]
    }
    reference_examples, geometry_summaries, exported = [], [], {}
    case_dir = OUT / "structures"
    case_dir.mkdir(exist_ok=True)
    for record in cases.itertuples():
        row = record.candidate_row
        q = np.asarray(query_vectors[row], dtype=float)
        candidate = structures[row]
        candidate.to(filename=str(case_dir / f"candidate_{row}.cif"))
        pool = ids[ptr[row] : ptr[row + 1]]
        distance = raw_distances(
            np.asarray(query_vectors[row], float), ref_vectors[pool]
        )
        order = np.lexsort((pool, distance))
        references = {
            "nearest_global_cosine": int(record.nearest_reference_row),
            "nearest_chemical": int(pool[order[0]]),
            "median_chemical_distance": int(pool[order[len(order) // 2]]),
        }
        per_profile = {
            profile: normalized_fingerprint(
                geometry_atoms(candidate, profile == "density_normalized"), profile
            )
            for profile in soap
        }
        geometry_distance = np.clip(
            1 - soap["density_normalized"][pool] @ per_profile["density_normalized"],
            0,
            2,
        )
        references["best_geometry_in_chemical_pool"] = int(
            pool[np.lexsort((pool, geometry_distance))[0]]
        )
        for role, ref_id in references.items():
            meta = metadata.iloc[ref_id]
            known = Structure.from_str(source.loc[meta.material_id, "cif"], fmt="cif")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                CifWriter(known).write_file(str(case_dir / f"reference_{ref_id}.cif"))
            d = float(
                raw_distances(
                    np.asarray(query_vectors[row], float), ref_vectors[ref_id]
                )
            )
            angle = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(
                            np.dot(q, ref_vectors[ref_id])
                            / (np.linalg.norm(q) * np.linalg.norm(ref_vectors[ref_id])),
                            -1,
                            1,
                        )
                    )
                )
            )
            same_formula = (
                known.composition.fractional_composition
                == candidate.composition.fractional_composition
            )
            reference_examples.append(
                {
                    "candidate_row": row,
                    "role": role,
                    "reference_row": ref_id,
                    "material_id": meta.material_id,
                    "formula": meta.pretty_formula,
                    "chemical_system": meta.chemical_system,
                    "chemically_eligible": bool(ref_id in pool),
                    "raw_squared_distance": d,
                    "angle_degrees": angle,
                    "known_DFT_formation_energy_per_atom": meta.formation_energy_per_atom,
                    "known_DFT_hull_distance": meta.e_above_hull,
                    "same_composition": bool(same_formula),
                    "structure_match": bool(StructureMatcher().fit(candidate, known))
                    if same_formula
                    else False,
                    **{
                        f"SOAP_{profile}_distance": float(
                            np.clip(
                                1 - np.dot(per_profile[profile], values[ref_id]), 0, 2
                            )
                        )
                        for profile, values in soap.items()
                    },
                }
            )
        geometry_summaries.append(
            {
                "candidate_row": row,
                "pool_size": len(pool),
                "nearest_chemical_embedding_vs_geometry_spearman": float(
                    spearmanr(distance, geometry_distance).statistic
                )
                if len(pool) > 2
                else None,
                "chemical_pool_distance_min": float(distance.min()),
                "chemical_pool_distance_median": float(np.median(distance)),
                "chemical_pool_distance_max": float(distance.max()),
                "best_geometry_distance": float(geometry_distance.min()),
            }
        )
        exported[str(row)] = {
            "candidate": candidate.as_dict(),
            "references": references,
            "chemical_distance_quantiles": np.quantile(
                distance, [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
        }
    pd.DataFrame(reference_examples).to_csv(OUT / "case_references.csv", index=False)
    pd.DataFrame(geometry_summaries).to_csv(OUT / "case_geometry.csv", index=False)
    (OUT / "case_structures.json").write_text(json.dumps(exported) + "\n")
    (OUT / "case_protocol.json").write_text(
        json.dumps(
            {
                "selection": "post-hoc illustrative test examples: up to 3 wins and 2 losses matched exactly on reduced formula, then 2 median-angle examples",
                "pair_choice": "passing <=.08, failing >=.12 and <=2 eV/atom; passing not below -.05; sorted formula then row IDs; added original score worse than removed",
                "not_random_sample": True,
                "not_causal_evidence": True,
                "geometry": "same stored element-blind SOAP profiles as prior study; density-normalized profile removes uniform volume scale",
                "energy_comparison_limit": "Known reference formation energy is DFT; candidate total energy is MatterSim. Do not directly compare these columns as a physical energy difference.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        cases[
            [
                "candidate_row",
                "case_reason",
                "formula",
                "selection_group",
                "e_above_hull_proxy",
                "nearest_angle_degrees",
                "original_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
