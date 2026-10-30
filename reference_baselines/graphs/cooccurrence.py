"""Build a known-reference graph from fit-side hull decompositions.

Nodes are the 27,136 known MP20 reference crystals. An edge joins two
references when both occur in the same saved decomposition of a fit candidate.
Generated candidates are never nodes. Candidate stability labels are never read.
"""

import numpy as np
from scipy import sparse


def cooccurrence_graph(positives, reference_count):
    """Return a symmetric weighted graph with self-loops.

    ``positives`` maps candidate row to reference rows in its decomposition.
    The edge weight is ``log1p(pair count)``. Self-loops have weight 1.
    """
    counts = {}
    for refs in positives.values():
        values = np.unique(np.asarray(refs, dtype=np.int64))
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                if left == right:
                    continue
                key = (int(left), int(right)) if left < right else (int(right), int(left))
                counts[key] = counts.get(key, 0) + 1
    if counts:
        keys = np.array(list(counts), dtype=np.int64)
        values = np.log1p(np.array(list(counts.values()), dtype=np.float64))
        rows = np.concatenate([keys[:, 0], keys[:, 1], np.arange(reference_count)])
        cols = np.concatenate([keys[:, 1], keys[:, 0], np.arange(reference_count)])
        data = np.concatenate([values, values, np.ones(reference_count)])
    else:
        rows = np.arange(reference_count)
        cols = np.arange(reference_count)
        data = np.ones(reference_count)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(reference_count, reference_count))
    graph.sum_duplicates()
    return graph, counts


def permuted_graph(graph, seed):
    """Permute node labels and keep the edge count and weight distribution."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(graph.shape[0])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    # New node p(i) takes the old edges of i.
    return graph[inverse][:, inverse].tocsr(), permutation


def graph_stats(graph, labels):
    """Describe graph coverage without candidate labels."""
    upper = sparse.triu(graph, k=1).tocoo()
    degree = np.asarray((graph - sparse.diags(graph.diagonal())).sum(1)).ravel()
    return {
        "graph": labels,
        "nodes": int(graph.shape[0]),
        "directed_nonself_entries": int(upper.nnz * 2),
        "undirected_nonself_edges": int(upper.nnz),
        "nodes_with_nonself_edge": int((degree > 0).sum()),
        "mean_weighted_degree": float(degree[degree > 0].mean()) if (degree > 0).any() else 0.0,
        "max_weighted_degree": float(degree.max()) if len(degree) else 0.0,
    }
