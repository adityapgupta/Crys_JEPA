"""Build the complementary-wiring reference graphs for the connectivity experiment.

Nodes are the 27,136 known MP20 crystals (same node set as every other graph).
Only the *edges* differ -- this experiment varies connectivity, not node features.

Outputs (reference_baselines/results/reference_gnn/graphs/):
  soap.npz   symmetric SOAP-geometry kNN graph (k=10), self-loops, weight 1 on loops
  phase.npz  convex-hull tie-line adjacency (reuses relational_edge/hull_compatibility.npz),
             self-loops added

The ``physical`` and ``cosine`` graphs are produced by build_reference_graph.py and
are reused as-is (co-occurrence and embedding-cosine kNN respectively); ``none`` is a
built-in empty graph. This builder only adds the two complementary wirings, so the
full topology set is physical / cosine / soap / phase / none.

Fit-side / known-crystal inputs only -- no candidate outcomes, validation, or
reserved labels are read.

Run: python -m reference_baselines.graphs.connectivity_adjacency
"""

import argparse
import json

import numpy as np
from scipy import sparse

from reference_baselines.core.io import sha256
from reference_baselines.paths import RESULTS_DIR, ROOT


OUT = RESULTS_DIR / "reference_gnn" / "graphs"
REFERENCE_COUNT = 27136
SOAP_PATH = RESULTS_DIR / "soap_density_normalized.npy"
HULL_PATH = RESULTS_DIR / "relational_edge" / "hull_compatibility.npz"
SOAP_K = 10


def soap_knn_graph(k=SOAP_K, batch=1024):
    """Symmetric SOAP-geometry kNN graph with self-loops.

    Edges connect each reference to its ``k`` nearest references by cosine
    similarity of the element-blind, density-normalised SOAP fingerprint -- a
    geometry relation independent of the Crys-JEPA embedding. Weight = cosine
    similarity in [0, 1]; symmetrised by max; self-loop weight 1.
    """
    if not SOAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOAP_PATH}. Generate SOAP first: "
            "python -m reference_baselines.structure.descriptors"
        )
    soap = np.asarray(np.load(SOAP_PATH), dtype=np.float32)
    if soap.shape[0] != REFERENCE_COUNT:
        raise ValueError(f"SOAP has {soap.shape[0]} rows, expected {REFERENCE_COUNT}")
    norm = np.linalg.norm(soap, axis=1, keepdims=True)
    unit = soap / np.maximum(norm, 1e-12)

    source, target, weight = [], [], []
    for start in range(0, REFERENCE_COUNT, batch):
        stop = min(start + batch, REFERENCE_COUNT)
        similarity = unit[start:stop] @ unit.T  # (b, N) cosine similarity
        rows = np.arange(stop - start)
        similarity[rows, np.arange(start, stop)] = -np.inf  # exclude self
        nearest = np.argpartition(-similarity, k, axis=1)[:, :k]
        picked = similarity[rows[:, None], nearest]
        source.append(np.repeat(np.arange(start, stop), k))
        target.append(nearest.reshape(-1).astype(np.int64))
        weight.append(np.clip(picked.reshape(-1), 1e-6, 1.0).astype(np.float64))
    source = np.concatenate(source)
    target = np.concatenate(target)
    weight = np.concatenate(weight)

    graph = sparse.csr_matrix(
        (weight, (source, target)), shape=(REFERENCE_COUNT, REFERENCE_COUNT)
    )
    graph = graph.maximum(graph.T)
    graph = graph + sparse.eye(REFERENCE_COUNT, format="csr", dtype=np.float64)
    graph.sum_duplicates()
    return graph.tocsr()


def phase_graph():
    """Hull tie-line adjacency as a topology, with self-loops.

    Reuses the verified relational_edge/hull_compatibility.npz (1 where two
    references are a genuine two-phase tie-line on the convex hull of their union
    chemical system). Already symmetric 0/1; we add self-loops so every node keeps
    itself in aggregation.
    """
    if not HULL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HULL_PATH}. Build it first: "
            "python -m reference_baselines.graphs.relational_edge_features"
        )
    hull = sparse.load_npz(HULL_PATH).tocsr()
    if hull.shape != (REFERENCE_COUNT, REFERENCE_COUNT):
        raise ValueError(f"Hull adjacency shape {hull.shape} != reference square")
    hull = hull.maximum(hull.T)  # defensive; it is already symmetric
    graph = hull.astype(np.float64) + sparse.eye(
        REFERENCE_COUNT, format="csr", dtype=np.float64
    )
    graph.sum_duplicates()
    graph.data[:] = np.minimum(graph.data, 1.0)
    return graph.tocsr()


def stats(graph, description):
    off_diagonal = graph.nnz - (graph.diagonal() > 0).sum()
    return {
        "description": description,
        "nodes": int(graph.shape[0]),
        "edges_including_self": int(graph.nnz),
        "off_diagonal_edges": int(off_diagonal),
        "mean_degree": float(off_diagonal / graph.shape[0]),
    }


def save(graph, name, description):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.npz"
    sparse.save_npz(path, graph)
    return {"name": name, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            **stats(graph, description)}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = [
        save(soap_knn_graph(), "soap", f"Symmetric SOAP-geometry kNN k={SOAP_K}"),
        save(phase_graph(), "phase", "Convex-hull tie-line adjacency with self-loops"),
    ]
    report = {
        "status": "passed",
        "nodes": REFERENCE_COUNT,
        "candidate_labels_read": False,
        "soap_rule": f"Each reference to its {SOAP_K} nearest references by SOAP cosine "
        "similarity; symmetric max; self-loop weight 1.",
        "phase_rule": "Two references adjacent when they form a genuine two-phase "
        "hull tie-line (reused from hull_compatibility.npz); self-loops added.",
        "graphs": records,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [SOAP_PATH, HULL_PATH]
            if path.exists()
        },
    }
    (OUT / "connectivity_graph_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
