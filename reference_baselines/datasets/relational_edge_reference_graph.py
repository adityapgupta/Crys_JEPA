"""Relational, non-redundant edge features on the frozen physical reference graph.

Each directed edge ``(i, j)`` receives three features, every one drawn from
information that is **not** in the Crys-JEPA node embedding (see
``docs/relational_edge_reference_gnn.md``):

  0. ``soap_distance``       ``1 - <soap_i, soap_j>``   element-blind local geometry
  1. ``cooccurrence_pmi``    ``log(N * c_ij / (c_i c_j))``  popularity-normalised competition
  2. ``hull_compatibility``  ``1`` if i,j share a stable hull tie-line, else ``0``

Topology, nodes, pools, decomposition targets and splits are unchanged from the
reference GNN; only edge information varies.  Features use known-crystal data and
fit-side decompositions only -- candidate outcomes and reserved labels are never
read.

``edge_mode`` controls how the feature rows are filled:
  ``default``   -- the real feature row of each edge
  ``shuffled``  -- each edge is given the **complete, self-consistent feature row of
                   another physical edge**, chosen by a fixed global edge permutation.
                   This preserves the marginal distribution of feature rows and only
                   breaks the alignment between a row and its edge (negative control).
  ``redundant`` -- slot 0 holds embedding cosine (a deliberately redundant feature),
                   slots 1-2 are zero (re-confirms redundant features add nothing)

Per-variant feature *masking* (scalar/geom/cooc/phase/rich/readout) is applied by
the training / evaluation loop, not here, so one ``default`` dataset serves them all.
"""

from functools import lru_cache

import numpy as np
from scipy import sparse
import torch

from reference_baselines.datasets.reference_graph import (
    REFERENCE_COUNT,
    PoolDataset,
    collate,
    load_graph,
)
from reference_baselines.paths import RESULTS_DIR


EDGE_FEATURES = ("soap_distance", "cooccurrence_pmi", "hull_compatibility")
EDGE_MODES = ("default", "shuffled", "redundant")
PRECOMPUTE = RESULTS_DIR / "relational_edge"

SOAP_PATH = RESULTS_DIR / "soap_density_normalized.npy"
FREQUENCY_PATH = PRECOMPUTE / "reference_decomp_frequency.npy"
HULL_PATH = PRECOMPUTE / "hull_compatibility.npz"
NORMALISER_PATH = PRECOMPUTE / "cooccurrence_normaliser.npy"

PMI_CLIP = 5.0  # pmi clipped to [-PMI_CLIP, PMI_CLIP] then scaled to [-1, 1]


@lru_cache(maxsize=None)
def _soap():
    if not SOAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOAP_PATH}. Run: python -m reference_baselines.structure.descriptors"
        )
    return np.nan_to_num(np.load(SOAP_PATH).astype(np.float64), nan=0.0)


@lru_cache(maxsize=None)
def _frequency():
    if not FREQUENCY_PATH.exists() or not NORMALISER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FREQUENCY_PATH} / {NORMALISER_PATH}. Run: "
            "python -m reference_baselines.graphs.relational_edge_features"
        )
    return np.load(FREQUENCY_PATH).astype(np.float64), float(np.load(NORMALISER_PATH))


@lru_cache(maxsize=None)
def _hull():
    if not HULL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HULL_PATH}. Run: "
            "python -m reference_baselines.graphs.relational_edge_features"
        )
    return sparse.load_npz(HULL_PATH).tocsr()


@lru_cache(maxsize=None)
def _physical_edges():
    """Co-occurrence counts and an ordinal index over the physical graph's edges."""
    counts = load_graph("physical").tocsr().copy()
    counts.data = np.expm1(counts.data)  # physical weight is log1p(count)
    coo = load_graph("physical").tocoo()
    edge_i = coo.row.astype(np.int64)
    edge_j = coo.col.astype(np.int64)
    size = len(edge_i)
    ordinal = sparse.csr_matrix(
        (np.arange(1, size + 1), (edge_i, edge_j)), shape=(REFERENCE_COUNT, REFERENCE_COUNT)
    ).tocsr()
    return counts, edge_i, edge_j, ordinal, size


def soap_distance(soap, source_global, target_global):
    """1 - cosine of unit-normalised SOAP fingerprints; in [0, 2]."""
    dot = np.sum(soap[source_global] * soap[target_global], axis=1)
    return 1.0 - np.clip(dot, -1.0, 1.0)


def cooccurrence_pmi(count, frequency, normaliser, source_global, target_global):
    """PMI from raw co-occurrence counts and per-reference frequency; scaled to [-1, 1]."""
    count = np.asarray(count, dtype=np.float64)
    ci = np.maximum(frequency[source_global], 1.0)
    cj = np.maximum(frequency[target_global], 1.0)
    ratio = (normaliser * count) / (ci * cj)
    pmi = np.log(np.maximum(ratio, 1e-6))
    return np.clip(pmi, -PMI_CLIP, PMI_CLIP) / PMI_CLIP


def hull_compatibility(hull, source_global, target_global):
    """0/1 tie-line adjacency looked up from the precomputed sparse matrix."""
    if len(source_global) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(hull[source_global, target_global]).ravel().astype(np.float64)


def _lookup(matrix, rows, cols):
    if len(rows) == 0:
        return np.zeros(0)
    return np.asarray(matrix[rows, cols]).ravel()


class RelationalEdgePoolDataset(PoolDataset):
    """Original reference pools augmented with non-redundant relational edge features."""

    def __init__(self, *args, edge_mode="default", edge_seed=2026, **kwargs):
        if edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {EDGE_MODES}")
        super().__init__(*args, **kwargs)
        self.edge_mode = edge_mode
        self.edge_seed = edge_seed
        if edge_mode == "redundant":
            return
        self.soap = _soap()
        self.frequency, self.normaliser = _frequency()
        self.hull = _hull()
        self.counts, self.edge_i, self.edge_j, self.ordinal, self.edge_total = _physical_edges()
        if len(self.soap) != REFERENCE_COUNT or len(self.frequency) != REFERENCE_COUNT:
            raise ValueError("SOAP / frequency arrays are not aligned with the references")
        # Global permutation of complete edge rows (identity unless shuffled).
        if edge_mode == "shuffled":
            self.edge_perm = np.random.default_rng(edge_seed).permutation(self.edge_total)
        else:
            self.edge_perm = np.arange(self.edge_total)

    def _reassign_shuffled(self, gi, gj):
        """Map each edge to the endpoints of another physical edge (its permuted row)."""
        order = _lookup(self.ordinal, gi, gj).astype(np.int64)  # 1-based ordinal, 0 if absent
        present = order > 0
        permuted = self.edge_perm[np.where(present, order - 1, 0)]
        return (
            np.where(present, self.edge_i[permuted], gi),
            np.where(present, self.edge_j[permuted], gj),
        )

    def _features(self, item):
        edge = item["edge"].numpy()
        source, target = edge[0], edge[1]
        refs = item["reference_ids"]
        if self.edge_mode == "redundant":
            embedding = item["x"][:, :512].numpy()
            norm = np.maximum(np.linalg.norm(embedding, axis=1), 1e-12)
            cosine = np.sum(embedding[source] * embedding[target], axis=1) / (norm[source] * norm[target])
            return np.column_stack(
                [np.clip(cosine, -1.0, 1.0), np.zeros(len(source)), np.zeros(len(source))]
            ).astype(np.float32)
        gi, gj = refs[source], refs[target]
        if self.edge_mode == "shuffled":
            gi, gj = self._reassign_shuffled(gi, gj)
        counts = _lookup(self.counts, gi, gj)  # global co-occurrence count c_ij
        return np.column_stack(
            [
                soap_distance(self.soap, gi, gj),
                cooccurrence_pmi(counts, self.frequency, self.normaliser, gi, gj),
                hull_compatibility(self.hull, gi, gj),
            ]
        ).astype(np.float32)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["edge_attr"] = torch.from_numpy(self._features(item))
        return item


def relational_edge_collate(items):
    """Original disjoint-graph collation plus the aligned edge-feature block."""
    batch = collate(items)
    batch["edge_attr"] = torch.cat([item["edge_attr"] for item in items])
    return batch
