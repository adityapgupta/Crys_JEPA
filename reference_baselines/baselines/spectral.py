"""Retrieve known references in checked diffusion-map coordinates, then score raw JEPA distance."""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from reference_baselines.core.angular import raw_distances
from reference_baselines.core.retrieval import exact_knn
from reference_baselines.core.settings import baseline_settings
from reference_baselines.paths import BASELINE_RESULTS_DIR, ROOT, RESULTS_DIR


def diffusion_coordinates(values, vectors, degree, time_value):
    keep = (values > 1e-9) & (np.abs(1 - values) > 1e-6)
    mu = 1 - values[keep]
    functions = vectors[:, keep] / np.sqrt(degree[:, None])
    return functions * np.power(mu, time_value), functions, mu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-metric", choices=["euclidean", "cosine"], required=True
    )
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = baseline_settings()
    out = BASELINE_RESULTS_DIR
    status = json.loads((out / f"{args.graph_metric}_eigenmodes.json").read_text())
    if status["status"] != "passed":
        raise RuntimeError("No checked full-graph eigenvectors are available.")
    eig = np.load(out / f"{args.graph_metric}_eigenmodes.npz")
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv")
    rows = np.flatnonzero(manifest.vun & manifest.split.eq(args.split))
    ref = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    query = np.asarray(
        np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")[rows],
        dtype=np.float64,
    )
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    original = pd.read_csv(
        RESULTS_DIR / "candidate_reference_scores.csv"
    ).crys_jepa.to_numpy()[rows]
    scores = {}
    overlaps = []
    for seed_metric in ["euclidean", "cosine"]:
        seed = np.load(
            RESULTS_DIR
            / (
                "candidate_knn_indices.npy"
                if seed_metric == "euclidean"
                else "candidate_cosine_indices.npy"
            ),
            mmap_mode="r",
        )[rows, :10]
        for time_value in cfg["spectral_times"]:
            coords, functions, mu = diffusion_coordinates(
                eig["values"], eig["vectors"], eig["degree"], time_value
            )
            if not coords.shape[1]:
                raise ValueError("No nonconstant spectral modes survived.")
            qcoords = functions[seed].mean(1) / mu * np.power(mu, time_value)
            neighbours, _ = exact_knn(
                qcoords,
                coords,
                max(cfg["ks"]),
                512,
                False,
                f"spectral {args.graph_metric}/{seed_metric} t={time_value}",
            )
            raw = np.empty(neighbours.shape)
            mask = np.zeros(neighbours.shape, dtype=bool)
            for start in range(0, len(rows), 256):
                stop = min(start + 256, len(rows))
                raw[start:stop] = raw_distances(
                    query[start:stop, None, :], ref[neighbours[start:stop]]
                )
            for i, row in enumerate(rows):
                mask[i] = np.isin(neighbours[i], ids[ptr[row] : ptr[row + 1]])
            for k in cfg["ks"]:
                prefix = f"spectral_{args.graph_metric}_seed_{seed_metric}_t{time_value}_k{k}"
                scores[prefix] = raw[:, :k].mean(1)
                count = mask[:, :k].sum(1)
                mean = np.divide(
                    np.where(mask[:, :k], raw[:, :k], 0).sum(1),
                    count,
                    out=np.full(len(rows), np.nan),
                    where=count > 0,
                )
                scores[prefix + "_intersection_fallback"] = np.where(
                    count > 0, mean, original
                )
                overlaps.append(
                    {
                        "method": prefix,
                        "fraction_with_chemical_intersection": float(
                            (count > 0).mean()
                        ),
                    }
                )
    np.savez_compressed(
        out / f"{args.split}_spectral_{args.graph_metric}.npz",
        candidate_rows=rows,
        methods=np.array(list(scores)),
        scores=np.column_stack(list(scores.values())),
    )
    report = {
        "graph_metric": args.graph_metric,
        "candidate_count": len(rows),
        "method_count": len(scores),
        "out_of_sample": "uniform 10-neighbour Nystrom random-walk extension: psi(query)=mean(psi(seeds))/mu; references remain fixed",
        "coordinates": "nonconstant normalized-Laplacian eigenfunctions times mu^t, mu=1-lambda; truncated diffusion map",
        "score": "retrieve in spectral coordinates, score by mean RAW squared JEPA distance; fallback is original chemical score",
        "reference_bank": "known MP20 training crystals, never generated candidates",
        "candidate_labels_used": False,
        "test_scores_produced": False,
        "chemical_overlap": overlaps,
    }
    (out / f"{args.split}_spectral_{args.graph_metric}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Saved", len(scores), "spectral methods.", flush=True)


if __name__ == "__main__":
    main()
