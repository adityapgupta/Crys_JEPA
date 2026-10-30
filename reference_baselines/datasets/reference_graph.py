"""Reference-utility data for the clean GNN experiment.

Each row is one generated candidate. Its references are the chemically
eligible known crystals plus the five nearest known crystals in the frozen
JEPA space. The target of a reference is one when that reference appears in
the saved equilibrium decomposition of the candidate.

Candidate stability labels are never read. Legacy test rows and the reserved
partition are blocked from training.

Pool features are precomputed once into a compact table. Per-row Python work
would be much slower than the small graph forward pass.
"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from reference_baselines.datasets.decomposition import build_targets
from reference_baselines.paths import RESULTS_DIR, ROOT


OUT = RESULTS_DIR / "gnn_comparison"
EXPERIMENT = RESULTS_DIR / "reference_gnn"
GRAPHS = EXPERIMENT / "graphs"
SPLITS = ("train", "validation", "reserved")
GRAPH_NAMES = ("physical", "cosine", "self", "permuted", "none")
REFERENCE_COUNT = 27136
TABLE_SCHEMA = "v2"

_GRAPH_CACHE = {}
_TARGET_CACHE = {}
_ARRAYS = {}


def split_rows(split):
    """Return the frozen row index array for one permitted split."""
    if split not in SPLITS:
        raise ValueError(f"Split must be one of {SPLITS}, not {split!r}")
    path = OUT / f"{split}_rows.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen split file: {path}")
    rows = np.load(path).astype(np.int64)
    assert_split_isolation(split, rows)
    return rows


def assert_split_isolation(split, rows):
    """Reject legacy test rows; require the three splits to be disjoint."""
    legacy = RESULTS_DIR / "test_candidate_rows.npy"
    if legacy.exists() and np.intersect1d(rows, np.load(legacy)).size:
        raise ValueError("Legacy test rows are blocked from this experiment")
    reserved = np.load(OUT / "reserved_rows.npy")
    if split != "reserved" and np.intersect1d(rows, reserved).size:
        raise ValueError("Reserved rows entered a non-reserved split")
    if split == "train":
        others = np.concatenate([np.load(OUT / "validation_rows.npy"), reserved])
        if np.intersect1d(rows, others).size:
            raise ValueError("Training rows overlap validation or reserved rows")


def decomposition_targets(split):
    """Return per-row mapped decomposition references for one split."""
    if split not in _TARGET_CACHE:
        _TARGET_CACHE[split] = build_targets(split_rows(split))[0]
    return _TARGET_CACHE[split]


def load_graph(name):
    """Load one prepared 27,136-node reference graph."""
    if name not in GRAPH_NAMES:
        raise ValueError(f"Graph must be one of {GRAPH_NAMES}, not {name!r}")
    if name == "none":
        from scipy import sparse

        return sparse.csr_matrix((REFERENCE_COUNT, REFERENCE_COUNT), dtype=np.float32)
    if name not in _GRAPH_CACHE:
        from scipy import sparse

        path = GRAPHS / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing graph {path}. Run: python -m reference_baselines.graphs.build_reference_graph"
            )
        _GRAPH_CACHE[name] = sparse.load_npz(path).tocsr()
    return _GRAPH_CACHE[name]


def arrays():
    """Load the large frozen arrays once per process."""
    if not _ARRAYS:
        _ARRAYS.update(
            {
                "chemical_ptr": np.load(RESULTS_DIR / "chemical_reference_indptr.npy"),
                "chemical_ids": np.load(RESULTS_DIR / "chemical_reference_indices.npy"),
                "knn": np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r"),
                "query": np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r"),
                "reference": np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r"),
                "rcomp": np.load(RESULTS_DIR / "reference_composition.npy", mmap_mode="r"),
                "qcomp": np.load(RESULTS_DIR / "candidate_composition.npy", mmap_mode="r"),
            }
        )
    return _ARRAYS


def natural_pool(row, k=5):
    """Chemical references plus top-k cosine neighbours, sorted and unique."""
    table = arrays()
    chemical = table["chemical_ids"][
        table["chemical_ptr"][row] : table["chemical_ptr"][row + 1]
    ]
    return np.union1d(chemical, table["knn"][row, :k]).astype(np.int64)


def induce(graph, refs):
    """Return the subgraph on sorted ``refs`` as local source, target, weight."""
    sub = graph[refs]
    lookup = {int(value): i for i, value in enumerate(refs)}
    source, target, weight = [], [], []
    for i in range(len(refs)):
        for position in range(sub.indptr[i], sub.indptr[i + 1]):
            other = lookup.get(int(sub.indices[position]))
            if other is not None:
                source.append(i)
                target.append(other)
                weight.append(float(sub.data[position]))
    return (
        np.array(source, dtype=np.int32),
        np.array(target, dtype=np.int32),
        np.array(weight, dtype=np.float32),
    )


def table_path(split, graph="physical", k=5, pool="chemical_knn", seed=2026, limit=0):
    suffix = f"_limit{limit}" if limit else ""
    return (
        EXPERIMENT
        / "pools"
        / f"{TABLE_SCHEMA}_{split}_{pool}_k{k}_{graph}_seed{seed}{suffix}.npz"
    )


def build_table(split, graph="physical", k=5, pool="chemical_knn", seed=2026, limit=0):
    """Precompute per-row pools, decomposition targets, and local edges."""
    if pool not in ("chemical_knn", "random"):
        raise ValueError("Pool must be 'chemical_knn' or 'random'")
    rows = split_rows(split)
    if limit and limit < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(rows, limit, replace=False))
    positives = decomposition_targets(split)
    adjacency = load_graph(graph)
    table = arrays()
    rng = np.random.default_rng(seed)

    pool_indptr = [0]
    pool_values, target_values, prior_values, has_target = [], [], [], []
    edge_indptr = [0]
    edge_source, edge_target, edge_weight = [], [], []
    for row in rows:
        row = int(row)
        refs = natural_pool(row, k)
        if pool == "random":
            refs = np.sort(
                rng.choice(REFERENCE_COUNT, len(refs), replace=False)
            ).astype(np.int64)
        if not len(refs):
            raise ValueError(f"Candidate {row} has an empty reference pool")
        pool_values.append(refs)
        pool_indptr.append(pool_indptr[-1] + len(refs))
        phases = positives[row]
        has_target.append(len(phases) > 0)
        target_values.append(
            np.isin(refs, phases).astype(np.uint8)
            if len(phases)
            else np.zeros(len(refs), np.uint8)
        )
        composition = np.asarray(table["rcomp"][refs])
        _, inverse, counts = np.unique(
            composition, axis=0, return_inverse=True, return_counts=True
        )
        prior_values.append(np.log(1.0 / (len(counts) * counts[inverse])).astype(np.float32))
        source, destination, weight = induce(adjacency, refs)
        edge_source.append(source)
        edge_target.append(destination)
        edge_weight.append(weight)
        edge_indptr.append(edge_indptr[-1] + len(source))

    return {
        "rows": rows,
        "pool_indptr": np.array(pool_indptr, dtype=np.int64),
        "pool": np.concatenate(pool_values).astype(np.int32),
        "target": np.concatenate(target_values),
        "has_target": np.array(has_target, dtype=bool),
        "log_prior": np.concatenate(prior_values),
        "edge_indptr": np.array(edge_indptr, dtype=np.int64),
        "edge_source": np.concatenate(edge_source).astype(np.int32),
        "edge_target": np.concatenate(edge_target).astype(np.int32),
        "edge_weight": np.concatenate(edge_weight).astype(np.float32),
        "spec": np.array(
            json.dumps(
                {"split": split, "graph": graph, "k": k, "pool": pool, "seed": seed}
            )
        ),
    }


def prepare_table(
    split, graph="physical", k=5, pool="chemical_knn", seed=2026, limit=0, force=False
):
    """Save the compact table once, then reuse it."""
    path = table_path(split, graph, k, pool, seed, limit)
    if path.exists() and not force:
        return path
    data = build_table(split, graph, k, pool, seed, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


class PoolDataset(Dataset):
    """Compact precomputed pool table with vectorised feature assembly."""

    def __init__(self, split, graph="physical", k=5, pool="chemical_knn", seed=2026, limit=0):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split!r}")
        path = prepare_table(split, graph, k, pool, seed, limit)
        data = np.load(path)
        self.split, self.graph_name, self.k, self.pool_kind = split, graph, k, pool
        self.rows = data["rows"]
        self.pool_indptr = data["pool_indptr"]
        self.pool_values = data["pool"]
        self.targets = data["target"]
        self.has_target = data["has_target"]
        self.priors = data["log_prior"]
        self.edge_indptr = data["edge_indptr"]
        self.edge_source = data["edge_source"]
        self.edge_target = data["edge_target"]
        self.edge_weight = data["edge_weight"]
        self.table = arrays()
        self.index = np.arange(len(self.rows))

    def __len__(self):
        return len(self.index)

    def pool(self, index):
        start, stop = self.pool_indptr[index], self.pool_indptr[index + 1]
        return self.pool_values[start:stop]

    def __getitem__(self, index):
        position = int(self.index[index])
        row = int(self.rows[position])
        start, stop = self.pool_indptr[position], self.pool_indptr[position + 1]
        refs = self.pool_values[start:stop].astype(np.int64)
        embedding = np.asarray(self.table["query"][row], dtype=np.float32)
        query = np.concatenate(
            [embedding, np.asarray(self.table["qcomp"][row], dtype=np.float32)]
        )
        nodes = np.asarray(self.table["reference"][refs], dtype=np.float32)
        composition = np.asarray(self.table["rcomp"][refs], dtype=np.float32)
        distance = np.square(
            np.asarray(nodes, dtype=np.float64) - np.asarray(embedding, dtype=np.float64)
        ).sum(1)
        cosine = np.clip(
            1
            - (nodes @ embedding)
            / np.maximum(
                np.linalg.norm(nodes, axis=1)
                * max(float(np.linalg.norm(embedding)), 1e-12),
                1e-12,
            ),
            0,
            2,
        )
        chemical = self.table["chemical_ids"][
            self.table["chemical_ptr"][row] : self.table["chemical_ptr"][row + 1]
        ]
        eligible = np.isin(refs, chemical).astype(np.float32)
        pair = np.column_stack([np.log1p(distance), eligible, cosine]).astype(np.float32)
        edge_start, edge_stop = self.edge_indptr[position], self.edge_indptr[position + 1]
        return {
            "row": row,
            "reference_ids": refs,
            "x": torch.from_numpy(np.concatenate([nodes, composition], axis=1)),
            "query": torch.from_numpy(query),
            "pair": torch.from_numpy(pair),
            "edge": torch.from_numpy(
                np.stack(
                    [
                        self.edge_source[edge_start:edge_stop],
                        self.edge_target[edge_start:edge_stop],
                    ]
                )
            ),
            "weight": torch.from_numpy(self.edge_weight[edge_start:edge_stop]),
            "target": torch.from_numpy(self.targets[start:stop].astype(np.float32)),
            "has_target": bool(self.has_target[position]),
            "log_prior": torch.from_numpy(self.priors[start:stop]),
            "distance": torch.from_numpy(distance.astype(np.float32)),
        }


def collate(items):
    """Join variable-size reference pools into one disjoint graph batch."""
    sizes = torch.tensor([len(item["reference_ids"]) for item in items], dtype=torch.long)
    pointer = torch.cumsum(sizes, 0) - sizes
    owner = torch.repeat_interleave(torch.arange(len(items)), sizes)
    edges = [item["edge"] + int(pointer[i]) for i, item in enumerate(items)]
    return {
        "rows": torch.tensor([item["row"] for item in items], dtype=torch.long),
        "x": torch.cat([item["x"] for item in items]),
        "query": torch.stack([item["query"] for item in items]),
        "pair": torch.cat([item["pair"] for item in items]),
        "edge": torch.cat(edges, dim=1),
        "weight": torch.cat([item["weight"] for item in items]),
        "target": torch.cat([item["target"] for item in items]),
        "has_target": torch.tensor([item["has_target"] for item in items], dtype=torch.bool),
        "log_prior": torch.cat([item["log_prior"] for item in items]),
        "distance": torch.cat([item["distance"] for item in items]),
        "owner": owner,
        "pool_size": sizes,
        "pool_ptr": pointer,
        "reference_ids": np.concatenate([item["reference_ids"] for item in items]),
    }
