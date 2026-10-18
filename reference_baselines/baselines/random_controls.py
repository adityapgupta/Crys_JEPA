"""Match reference-set size and fallback gates to test whether neighbour choice adds value."""

import json

import numpy as np
import pandas as pd

from reference_baselines.core.angular import raw_distances, within_angle
from reference_baselines.paths import BASELINE_RESULTS_DIR, ROOT, RESULTS_DIR


def main():
    source = np.load(BASELINE_RESULTS_DIR / "validation_angular_scores.npz")
    rows = source["candidate_rows"]
    ref = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    query = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    euclidean = np.load(RESULTS_DIR / "candidate_knn_indices.npy", mmap_mode="r")
    cosine = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")
    angle = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")
    original = pd.read_csv(
        RESULTS_DIR / "candidate_reference_scores.csv"
    ).crys_jepa.to_numpy()
    names = [
        f"random_{kind}_seed{seed}"
        for seed in [2026, 2027, 2028]
        for kind in [
            "chemical_k5",
            "euclidean_intersection_size",
            "cosine_intersection_size",
            "theta1_gate_size",
        ]
    ]
    scores = {name: np.full(len(rows), np.nan) for name in names}
    for seed in [2026, 2027, 2028]:
        rng = np.random.default_rng(seed)
        for i, row in enumerate(rows):
            pool = ids[ptr[row] : ptr[row + 1]]
            e_count = int(np.isin(euclidean[row, :5], pool).sum())
            c_count = int(np.isin(cosine[row, :5], pool).sum())
            t_count = int(within_angle(angle[row, :10], 1).sum())
            for kind, count, bank in [
                ("chemical_k5", min(5, len(pool)), pool),
                ("euclidean_intersection_size", e_count, pool),
                ("cosine_intersection_size", c_count, pool),
                ("theta1_gate_size", t_count, len(ref)),
            ]:
                if count:
                    chosen = rng.choice(bank, count, replace=False)
                    scores[f"random_{kind}_seed{seed}"][i] = raw_distances(
                        query[row], ref[chosen]
                    ).mean()
                else:
                    scores[f"random_{kind}_seed{seed}"][i] = original[row]
        print("Random reference controls finished:", seed, flush=True)
    # A mixed tiny-neighbour/original score can act mainly as a gate. Test the gate directly.
    gated = within_angle(angle[rows, :10], 1).any(1)
    old = original[rows]
    order = np.lexsort((rows, np.nan_to_num(old, nan=np.inf), ~gated))
    ranks = np.empty(len(rows), dtype=float)
    ranks[order] = np.arange(len(rows))
    ranks[(~gated) & ~np.isfinite(old)] = np.nan
    scores["theta1_gate_then_original_score"] = ranks
    for seed in [2026, 2027, 2028]:
        rng = np.random.default_rng(seed)
        scores[f"theta1_gate_random_order_seed{seed}"] = (~gated).astype(
            float
        ) + rng.random(len(rows)) * 0.5
    scores["cosine_nearest_only_angle"] = angle[rows, 0]
    names = list(scores)
    np.savez_compressed(
        BASELINE_RESULTS_DIR / "validation_random_controls.npz",
        candidate_rows=rows,
        methods=np.array(names),
        scores=np.column_stack(list(scores.values())),
    )
    (BASELINE_RESULTS_DIR / "random_control_protocol.json").write_text(
        json.dumps(
            {
                "seeds": [2026, 2027, 2028],
                "reference_bank": "known MP20 only",
                "purpose": "Match reference count and empty-pool fallback gate to separate neighbour quality from small-pool and gating effects.",
                "theta_control": "Same availability gate and reference count as cosine theta=1 degree, k=10; replace chosen references by random known references.",
                "candidate_labels_used": False,
                "test_rows_scored": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
