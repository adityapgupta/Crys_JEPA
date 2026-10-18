"""Choose reference-pool size by graph conductance, without candidate outcome labels."""

import json

import numpy as np
import pandas as pd
from scipy import sparse

from reference_baselines.core.angular import raw_distances
from reference_baselines.core.settings import baseline_settings
from reference_baselines.paths import BASELINE_RESULTS_DIR, ROOT, RESULTS_DIR


def prefix_conductance(graph, degree, ordered, ks):
    """Weighted cut/volume on query-distance-ordered prefixes, not a global minimum cut."""
    total = float(degree.sum())
    adjacency = graph[ordered][:, ordered].toarray()
    internal = adjacency.cumsum(0).cumsum(1)
    volume = np.cumsum(degree[ordered])
    result = []
    for k in ks:
        cut = volume[k - 1] - internal[k - 1, k - 1]
        denominator = min(volume[k - 1], total - volume[k - 1])
        result.append(max(0.0, cut) / denominator if denominator > 0 else np.inf)
    return np.array(result)


def main():
    cfg = baseline_settings()
    out = BASELINE_RESULTS_DIR
    rows = np.load(out / "validation_angular_scores.npz")["candidate_rows"]
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    original = pd.read_csv(
        RESULTS_DIR / "candidate_reference_scores.csv"
    ).crys_jepa.to_numpy()[rows]
    ref = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    query = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    scores = {}
    choices = []
    for retrieval in ["euclidean", "cosine"]:
        neighbours = np.load(
            RESULTS_DIR
            / (
                "candidate_knn_indices.npy"
                if retrieval == "euclidean"
                else "candidate_cosine_indices.npy"
            ),
            mmap_mode="r",
        )[rows]
        distances = np.empty(neighbours.shape)
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            distances[start:stop] = raw_distances(
                np.asarray(query[rows[start:stop]], float)[:, None, :],
                ref[neighbours[start:stop]],
            )
        for metric in ["euclidean", "cosine"]:
            name = (
                f"reference_graph_k10.npz"
                if metric == "euclidean"
                else "reference_cosine_graph_k10.npz"
            )
            graph = sparse.load_npz(RESULTS_DIR / name).tocsr()
            degree = np.asarray(graph.sum(1)).ravel()
            values = np.array(
                [
                    prefix_conductance(graph, degree, seed, cfg["ks"])
                    for seed in neighbours
                ]
            )
            prefix = f"conductance_{metric}_seed_{retrieval}"
            for j, k in enumerate(cfg["ks"]):
                scores[f"{prefix}_k{k}"] = values[:, j]
            picked = np.argmin(values, axis=1)
            chosen = np.array(cfg["ks"])[picked]
            raw = np.empty(len(rows))
            intersection = np.empty(len(rows))
            for i, (row, k) in enumerate(zip(rows, chosen)):
                raw[i] = distances[i, :k].mean()
                mask = np.isin(neighbours[i, :k], ids[ptr[row] : ptr[row + 1]])
                intersection[i] = (
                    distances[i, :k][mask].mean() if mask.any() else original[i]
                )
            scores[f"{prefix}_adaptive_raw"] = raw
            scores[f"{prefix}_adaptive_intersection_fallback"] = intersection
            choices.append(
                {
                    "graph": metric,
                    "retrieval": retrieval,
                    "chosen_k_counts": {
                        str(k): int((chosen == k).sum()) for k in cfg["ks"]
                    },
                }
            )
            print("Conductance sweep finished:", metric, retrieval, flush=True)
    np.savez_compressed(
        out / "validation_conductance.npz",
        candidate_rows=rows,
        methods=np.array(list(scores)),
        scores=np.column_stack(list(scores.values())),
    )
    (out / "conductance_protocol.json").write_text(
        json.dumps(
            {
                "method_count": len(scores),
                "choices": choices,
                "definition": "cut(S,not S)/min(volume(S),volume(not S)); equal to normalized-Laplacian indicator Rayleigh quotient when S has smaller volume",
                "adaptive_rule": "choose minimum conductance among the distance-ordered prefixes k=5,10,20,50; ties choose smaller k",
                "limits": "A local sweep over these prefixes, not a Fiedler sweep or a global minimum cut.",
                "reference_nodes": "known MP20 only",
                "candidate_labels_used": False,
                "test_rows_scored": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
