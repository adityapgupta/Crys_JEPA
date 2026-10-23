"""Give each reduced reference formula equal total weight; retain polymorphs."""

import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition

from reference_baselines.baselines.mean_ablation import gate_ranks
from reference_baselines.core.angular import within_angle
from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "pre_gnn"


def formula_weights(formulas):
    _, inverse, counts = np.unique(formulas, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = read_csv(ROOT / "embeddings/data/metadata.csv")
    formulas = np.array(
        [Composition(f).reduced_formula for f in metadata.pretty_formula]
    )
    reference = np.asarray(
        np.load(ROOT / "embeddings/data/embeddings.npy"), dtype=float
    )
    query = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    frame = manifest[manifest.vun & manifest.split.isin(["validation", "test"])]
    rows = frame.index.to_numpy()
    positions = pd.Series(np.arange(len(rows)), index=rows)
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    values = np.full(len(rows), np.nan)
    pool_records = []
    for system, group in frame.groupby("chemical_system", sort=False):
        rr = group.index.to_numpy()
        pool = ids[ptr[rr[0]] : ptr[rr[0] + 1]]
        if not len(pool):
            continue
        weights = formula_weights(formulas[pool])
        elemental = metadata.iloc[pool].chemical_system.str.count("-").eq(0).to_numpy()
        pool_records.append(
            {
                "chemical_system": system,
                "reference_count": len(pool),
                "formula_count": len(np.unique(formulas[pool])),
                "max_formula_share_before": float(
                    pd.Series(formulas[pool]).value_counts().max() / len(pool)
                ),
                "max_formula_share_after": float(1 / len(np.unique(formulas[pool]))),
                "elemental_share_before": float(elemental.mean()),
                "elemental_share_after": float(weights[elemental].sum()),
            }
        )
        r = reference[pool]
        for start in range(0, len(rr), 256):
            chosen = rr[start : start + 256]
            q = np.asarray(query[chosen], dtype=float)
            distances = np.maximum(
                np.square(q).sum(1)[:, None]
                + np.square(r).sum(1)[None, :]
                - 2 * q @ r.T,
                0,
            )
            values[positions.loc[chosen].to_numpy()] = distances @ weights
    cosine = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")
    for split in ["validation", "test"]:
        mask = frame.split.eq(split).to_numpy()
        split_rows = rows[mask]
        gated = within_angle(cosine[split_rows, :10], 1).any(1)
        np.savez_compressed(
            OUT / f"{split}_formula_scores.npz",
            candidate_rows=split_rows,
            methods=np.array(
                ["formula_balanced_mean", "gate_then_formula_balanced_mean"]
            ),
            scores=np.column_stack(
                [values[mask], gate_ranks(split_rows, values[mask], gated)]
            ),
        )
    pd.DataFrame(pool_records).to_csv(OUT / "formula_pool_checks.csv", index=False)
    inputs = [
        ROOT / "embeddings/data/metadata.csv",
        ROOT / "embeddings/data/embeddings.npy",
        ROOT / "embeddings/data/candidate_embeddings.npy",
        RESULTS_DIR / "chemical_reference_indptr.npy",
        RESULTS_DIR / "chemical_reference_indices.npy",
        RESULTS_DIR / "candidate_cosine_distances.npy",
    ]
    protocol = {
        "weight": "Each reduced formula gets 1/F total weight; each of its n polymorphs gets 1/(F*n)",
        "pool": "Unchanged original chemical references",
        "candidate_labels_used": False,
        "retain_structures": True,
        "date": "2026-10-22",
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
    }
    (OUT / "formula_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Scored formula balance for", len(rows), "validation/test candidates")


if __name__ == "__main__":
    main()
