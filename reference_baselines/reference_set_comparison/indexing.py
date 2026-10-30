"""Persisted exact vector index and the original mean squared-distance score."""

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.core.retrieval import exact_knn
from .common import fingerprint, finish, receipt


def subsystems(system):
    elements = sorted(set(system.split("-")))
    return ["-".join(sub) for n in range(1, len(elements)+1) for sub in combinations(elements, n)]


class ExactIndex:
    """Memory-mapped raw vectors; existing deterministic float64 exact search."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.references = np.load(self.directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.norms = np.load(self.directory / "norms.npy", mmap_mode="r", allow_pickle=False)
        self.metadata = pd.read_csv(self.directory / "rows.csv", keep_default_na=False)

    @classmethod
    def build(cls, export, directory):
        export, directory = Path(export), Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        identity = fingerprint({name: sha256(export / name) for name in ("embeddings.npy", "metadata.csv", "provenance.json")})
        marker = directory / "index.json"
        if not receipt(marker, identity):
            vectors = np.load(export / "embeddings.npy", allow_pickle=False)
            metadata = pd.read_csv(export / "metadata.csv", keep_default_na=False)
            if vectors.ndim != 2 or len(vectors) != len(metadata) or not np.isfinite(vectors).all():
                raise ValueError("Invalid reference vectors or row count")
            if not metadata.material_id.is_unique:
                raise ValueError("Reference material IDs must be unique")
            np.testing.assert_array_equal(metadata.embedding_row, np.arange(len(metadata)))
            np.save(directory / "vectors.npy", vectors, allow_pickle=False)
            np.save(directory / "norms.npy", np.square(vectors.astype(np.float64)).sum(1), allow_pickle=False)
            metadata.to_csv(directory / "rows.csv", index=False)
            finish(marker, identity, [directory / name for name in ("vectors.npy", "norms.npy", "rows.csv")],
                   metric="squared Euclidean, unnormalized", ties="smaller reference_row")
        return cls(directory)

    def search(self, queries, count, batch_size=128):
        return exact_knn(queries, self.references, count, batch_size=batch_size)


def score_reference_pools(queries, reference_vectors, reference_metadata, candidate_metadata, neighbours, distances, counts):
    """Chemical mean, global kNN mean, and deduplicated union mean."""
    groups = {s: np.asarray(ids, dtype=int) for s, ids in reference_metadata.groupby("chemical_system").groups.items()}
    pools = {s: np.sort(np.concatenate([groups[sub] for sub in subsystems(s) if sub in groups]))
             if any(sub in groups for sub in subsystems(s)) else np.array([], dtype=int)
             for s in candidate_metadata.chemical_system.unique()}
    result = candidate_metadata.copy()
    result["chemical_reference_count"] = [len(pools[s]) for s in result.chemical_system]
    scores = {"chemical": np.full(len(queries), np.nan)}
    for k in counts:
        scores[f"knn_{k}"] = distances[:, :k].mean(1)
        scores[f"union_{k}"] = np.empty(len(queries))
    for row, system in enumerate(candidate_metadata.chemical_system):
        pool = pools[system]
        if len(pool):
            delta = np.asarray(reference_vectors[pool], dtype=np.float64) - np.asarray(queries[row], dtype=np.float64)
            chemical_sum = np.square(delta).sum()
            scores["chemical"][row] = chemical_sum / len(pool)
        else:
            chemical_sum = 0.
        for k in counts:
            overlap = np.isin(neighbours[row, :k], pool)
            total = chemical_sum + distances[row, :k][~overlap].sum()
            scores[f"union_{k}"][row] = total / (len(pool) + k - overlap.sum())
    for method, values in scores.items():
        result[method] = values
    return result, pools


def score_candidates(config):
    output = Path(config["output"])
    index = ExactIndex.build(config["reference_export"], Path(config["cache"]) / "reference_index")
    identity = fingerprint({"embed": sha256(output / "embed.json"), "index": sha256(index.directory / "index.json"),
                            "counts": config["neighbours"], "source": sha256(Path(__file__))})
    marker = output / "score.json"
    if receipt(marker, identity):
        return
    queries = np.load(output / "candidate_embeddings.npy", mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(output / "candidate_metadata.csv", keep_default_na=False)
    neighbours, distances = index.search(queries, max(config["neighbours"]), config["retrieval_batch_size"])
    result, pools = score_reference_pools(queries, index.references, index.metadata, metadata, neighbours, distances, config["neighbours"])
    result.to_csv(output / "scores.csv", index=False)
    np.save(output / "neighbour_ids.npy", neighbours, allow_pickle=False)
    np.save(output / "neighbour_squared_distances.npy", distances, allow_pickle=False)
    # CSR records the chemical pool for every candidate for downstream inspection.
    sizes = np.array([len(pools[s]) for s in metadata.chemical_system], dtype=int)
    np.save(output / "chemical_indptr.npy", np.r_[0, np.cumsum(sizes)])
    np.save(output / "chemical_ids.npy", np.concatenate([pools[s] for s in metadata.chemical_system]))
    finish(marker, identity, [output / name for name in ("scores.csv", "neighbour_ids.npy", "neighbour_squared_distances.npy", "chemical_indptr.npy", "chemical_ids.npy")])
