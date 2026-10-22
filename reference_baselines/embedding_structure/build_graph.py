"""Weighted kNN graphs for descriptive community analysis."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def load_embeddings(data_dir="data", emb_name="embeddings.npy", meta_name="metadata.csv"):
    """Load embeddings (N x d) and check metadata row count (not source identity)."""
    import os

    emb = np.load(os.path.join(data_dir, emb_name)).astype(np.float32)
    if emb.ndim != 2 or len(emb) < 2 or not np.isfinite(emb).all():
        raise ValueError('Embeddings must be a finite matrix with at least two rows')
    meta_path = os.path.join(data_dir, meta_name)
    meta = None
    if os.path.exists(meta_path):
        try:
            import pandas as pd
            meta = pd.read_csv(meta_path, keep_default_na=False)
        except ModuleNotFoundError:
            print("  [warn] pandas not installed -- returning meta=None. "
                  "`pip install pandas` to load the metadata table.")

    if meta is not None and len(meta) != emb.shape[0]:
        raise ValueError(
            f"ALIGNMENT BROKEN: {emb.shape[0]} embeddings but {len(meta)} metadata rows. "
            "Row i of embeddings must correspond to row i of metadata."
        )
    return emb, meta


def build_knn_graph(
    embeddings: np.ndarray,
    k: int,
    sigma: float | None = None,
    metric: str = "l2",
    symmetrize: str = "union",
    node_mask: np.ndarray | None = None,
    normalize: bool = False,
):
    """Return a weighted kNN graph and manifest; sigma defaults to median neighbour distance."""
    if metric != "l2":
        raise NotImplementedError(f"Unsupported metric: {metric}; use l2")

    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('Embeddings must be a finite two-dimensional matrix')
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')

    if node_mask is not None:
        mask = np.asarray(node_mask)
        if mask.dtype != bool or mask.shape != (len(X),):
            raise ValueError('node_mask must be a Boolean array matching the input rows')
        node_index = np.where(mask)[0]
        X = X[node_index]
    else:
        node_index = np.arange(X.shape[0])

    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Cosine geometry is undefined for zero vectors')
        X = X / norms

    M = X.shape[0]
    if k >= M:
        raise ValueError(f"k={k} must be smaller than the number of nodes ({M}).")

    # Remove self by identity, including when exact duplicates precede it.
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', algorithm='brute')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    dist = np.empty((M, k), dtype=float)
    idx = np.empty((M, k), dtype=int)
    for row in range(M):
        eligible = indices[row] != row
        ids, ds = indices[row][eligible], distances[row][eligible]
        order = np.lexsort((ids, ds))[:k]
        idx[row], dist[row] = ids[order], ds[order]

    if sigma is None:
        sigma = float(np.median(dist))
        if sigma <= 0:
            sigma = float(dist[dist > 0].min()) if np.any(dist > 0) else 1.0

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')

    weights = np.exp(-(dist ** 2) / (sigma ** 2))

    rows = np.repeat(np.arange(M), k)
    cols = idx.reshape(-1)
    vals = weights.reshape(-1)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(M, M))
    A.setdiag(0)
    A.eliminate_zeros()

    if symmetrize == "union":
        W = A.maximum(A.T)
    elif symmetrize == "mutual":
        W = A.minimum(A.T)
    elif symmetrize == "none":
        W = A
    else:
        raise ValueError("symmetrize must be 'union', 'mutual', or 'none'")
    W = W.tocsr()
    W.eliminate_zeros()

    n_edges = int(W.nnz // 2) if symmetrize != "none" else int(W.nnz)
    n_comp, _ = connected_components(W, directed=False)
    manifest = {
        "n_nodes": int(M),
        "n_edges": n_edges,
        "k": int(k),
        "sigma": float(sigma),
        "metric": metric,
        "symmetrize": symmetrize,
        "normalized": bool(normalize),
        "n_connected_components": int(n_comp),
        "mean_degree": float(np.asarray((W > 0).sum(axis=1)).mean()),
        "weight_convention": "exp(-d^2/sigma^2); median all-neighbour distance",
        "numeric_dtype": "float64",
        "tie_policy": "sort returned ties by row ID; ties at retrieval boundary are backend dependent",
        "node_index": node_index,
    }
    return W, manifest


def graph_summary(W: sp.spmatrix) -> dict:
    """Return basic health stats + run the invariants we always want to hold."""
    W = W.tocsr()
    deg = np.asarray((W > 0).sum(axis=1)).reshape(-1)
    is_symmetric = (abs(W - W.T) > 1e-6).nnz == 0
    has_self_loops = W.diagonal().any()
    n_comp, labels = connected_components(W, directed=False)
    sizes = np.bincount(labels)
    return {
        "n_nodes": W.shape[0],
        "n_edges": int(W.nnz // 2),
        "symmetric": bool(is_symmetric),
        "self_loops": bool(has_self_loops),
        "isolated_nodes": int((deg == 0).sum()),
        "degree_min/mean/max": (int(deg.min()), float(deg.mean()), int(deg.max())),
        "n_connected_components": int(n_comp),
        "largest_component_frac": float(sizes.max() / W.shape[0]),
    }
