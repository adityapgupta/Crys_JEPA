"""Build the known-reference graphs for the reference-utility experiment.

Nodes are the 27,136 known MP20 crystals. Edges come from fit-side
decompositions only. Generated candidates are never nodes.

Outputs (reference_baselines/results/reference_gnn/graphs/):
  physical.npz  co-occurrence in fit hull decompositions, log1p weights
  cosine.npz    symmetric cosine k=10 reference graph
  self.npz      identity only (self-edge control)
  permuted.npz  physical graph with permuted node labels

Run: python -m reference_baselines.graphs.build_reference_graph
"""

import argparse
import json

import numpy as np
from scipy import sparse

from reference_baselines.core.io import sha256
from reference_baselines.datasets.decomposition import build_targets, target_stats
from reference_baselines.datasets.reference_graph import OUT as SPLIT_DIR
from reference_baselines.graphs.cooccurrence import cooccurrence_graph, graph_stats, permuted_graph
from reference_baselines.paths import PACKAGE_DIR, RESULTS_DIR, ROOT


OUT = RESULTS_DIR / "reference_gnn" / "graphs"
REFERENCE_COUNT = 27136


def cosine_graph(k=10):
    """Symmetric cosine similarity graph with self-loops, weight 1 on loops."""
    indices = np.load(RESULTS_DIR / "reference_cosine_indices.npy", mmap_mode="r")[:, :k]
    distances = np.load(RESULTS_DIR / "reference_cosine_distances.npy", mmap_mode="r")[:, :k]
    source = np.repeat(np.arange(REFERENCE_COUNT), k)
    target = np.asarray(indices).reshape(-1).astype(np.int64)
    similarity = np.clip(1.0 - np.asarray(distances, dtype=np.float64).reshape(-1), 1e-6, 1.0)
    graph = sparse.csr_matrix((similarity, (source, target)), shape=(REFERENCE_COUNT, REFERENCE_COUNT))
    graph = graph.maximum(graph.T)
    graph = graph + sparse.eye(REFERENCE_COUNT, format="csr", dtype=np.float64)
    graph.sum_duplicates()
    return graph.tocsr()


def save(graph, name, stats):
    path = OUT / f"{name}.npz"
    sparse.save_npz(path, graph)
    return {"name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), **stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-seed", type=int, default=20261023)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fit_rows = np.load(SPLIT_DIR / "train_rows.npy").astype(np.int64)
    validation = np.load(SPLIT_DIR / "validation_rows.npy").astype(np.int64)
    reserved = np.load(SPLIT_DIR / "reserved_rows.npy").astype(np.int64)
    if np.intersect1d(fit_rows, validation).size or np.intersect1d(fit_rows, reserved).size:
        raise ValueError("Graph fit rows must not include validation or reserved rows")

    # Fit-side decompositions define the physical edges. Labels are not read.
    positives, unmapped, phase_counts = build_targets(fit_rows)
    physical, pair_counts = cooccurrence_graph(positives, REFERENCE_COUNT)
    permuted, permutation = permuted_graph(physical, args.permutation_seed)

    cosine = cosine_graph()
    identity = sparse.eye(REFERENCE_COUNT, format="csr")
    records = []
    records.append(save(physical, "physical", graph_stats(physical, "Fit-decomposition co-occurrence")))
    records.append(save(cosine, "cosine", graph_stats(cosine, "Symmetric cosine k=10")))
    records.append(save(identity, "self", graph_stats(identity, "Self-loops only")))
    records.append(save(permuted, "permuted", {
        **graph_stats(permuted, "Permuted physical graph"),
        "permutation_seed": args.permutation_seed,
    }))

    report = {
        "status": "passed",
        "nodes": REFERENCE_COUNT,
        "fit_rows": int(len(fit_rows)),
        "validation_rows_not_used": int(len(validation)),
        "reserved_rows_not_used": int(len(reserved)),
        "candidate_labels_read": False,
        "candidate_nodes": 0,
        "physical_edge_rule": "Two known references are adjacent when both occur in the same fit-candidate hull decomposition.",
        "physical_weight": "log1p(co-occurrence count); self-loop weight 1",
        "cosine_rule": "Symmetric max of each reference's 10 nearest JEPA neighbours; similarity = 1 - cosine distance.",
        "cooccurrence_pair_count": len(pair_counts),
        "target_stats": target_stats(fit_rows, positives, unmapped, phase_counts),
        "graphs": records,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                PACKAGE_DIR / "datasets/decomposition.py",
                PACKAGE_DIR / "graphs/cooccurrence.py",
                RESULTS_DIR / "candidate_decompositions.csv",
                RESULTS_DIR / "chemical_reference_indices.npy",
                RESULTS_DIR / "reference_cosine_indices.npy",
            ]
        },
    }
    (OUT / "graph_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
