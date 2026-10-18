"""Build exact cosine retrieval and strict-stable-bank retrieval, without labels."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from reference_baselines.core.angular import unit_vectors
from reference_baselines.core.io import sha256
from reference_baselines.core.retrieval import KS, exact_knn
from reference_baselines.paths import ROOT, RESULTS_DIR


def save_graphs(indices, distance, prefix):
    for k in KS:
        scale = float(np.median(distance[:, k - 1]))
        if scale <= 0:
            raise ValueError("Cannot assign a positive graph scale.")
        weights = np.exp(-distance[:, :k].ravel() / (2 * scale))
        directed = sparse.csr_matrix(
            (weights, (np.repeat(np.arange(len(indices)), k), indices[:, :k].ravel())),
            shape=(len(indices), len(indices)),
        )
        sparse.save_npz(
            RESULTS_DIR / f"{prefix}_graph_k{k}.npz", directed.maximum(directed.T)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    refs = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    candidates = np.load(
        ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r"
    )
    metadata = pd.read_csv(ROOT / "embeddings/data/metadata.csv")
    r = unit_vectors(refs)
    q = unit_vectors(candidates)
    ri, rd = exact_knn(r, r, max(KS), args.batch_size, True, "cosine reference graph")
    ci, cd = exact_knn(
        q, r, max(KS), args.batch_size, False, "cosine candidate retrieval"
    )
    for name, values in [
        ("reference_cosine_indices", ri),
        ("reference_cosine_distances", np.clip(rd / 2, 0, 2)),
        ("candidate_cosine_indices", ci),
        ("candidate_cosine_distances", np.clip(cd / 2, 0, 2)),
    ]:
        np.save(RESULTS_DIR / f"{name}.npy", values)
    save_graphs(ri, np.clip(rd / 2, 0, 2), "reference_cosine")
    stable = np.flatnonzero(metadata.e_above_hull.to_numpy() <= 1e-8)
    np.save(RESULTS_DIR / "strict_stable_reference_rows.npy", stable)
    for metric, rr, qq in [
        ("euclidean", refs[stable], candidates),
        ("cosine", r[stable], q),
    ]:
        ids, distance = exact_knn(
            qq, rr, max(KS), args.batch_size, False, f"strict stable {metric}"
        )
        np.save(RESULTS_DIR / f"candidate_strict_{metric}_indices.npy", stable[ids])
        np.save(
            RESULTS_DIR / f"candidate_strict_{metric}_distances.npy",
            np.clip(distance / 2, 0, 2) if metric == "cosine" else distance,
        )
    report = {
        "reference_count": len(refs),
        "candidate_count": len(q),
        "strict_reference_count": len(stable),
        "all_bank_recorded_hull_max": float(metadata.e_above_hull.max()),
        "strict_filter": "known MP20 metadata e_above_hull <= 1e-8 eV/atom; generated crystals never enter the bank",
        "metric": "cosine distance = 1 - dot(unit(query),unit(reference)); exact float64 search; ties by original row ID",
        "graph": "symmetric union; exp(-cosine_distance/(2*median_kth_cosine_distance))",
        "ks": list(KS),
        "elapsed_seconds": time.perf_counter() - start,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                Path(__file__).resolve(),
                ROOT / "embeddings/data/metadata.csv",
                ROOT / "embeddings/data/embeddings.npy",
                ROOT / "embeddings/data/candidate_embeddings.npy",
            ]
        },
    }
    (RESULTS_DIR / "cosine_retrieval_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
