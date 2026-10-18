"""Compare cosine/angle reference pools and a strict-stable known-reference bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reference_baselines.core.angular import raw_distances, unit_vectors, within_angle
from reference_baselines.core.io import sha256
from reference_baselines.core.settings import baseline_settings
from reference_baselines.paths import (
    BASELINE_RESULTS_DIR,
    CONFIG_DIR,
    ROOT,
    RESULTS_DIR,
)


def masked_mean(values, mask):
    n = mask.sum(1)
    return np.divide(
        np.where(mask, values, 0).sum(1),
        n,
        out=np.full(len(values), np.nan),
        where=n > 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    args = parser.parse_args()
    cfg = baseline_settings()
    out = BASELINE_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv")
    rows = np.flatnonzero(manifest.vun & manifest.split.eq(args.split))
    ref = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    candidate = np.load(
        ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r"
    )
    q = np.asarray(candidate[rows], dtype=np.float64)
    ru = unit_vectors(ref)
    qu = unit_vectors(q)
    ci = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")[rows]
    cd = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")[rows]
    distances = np.empty(ci.shape, dtype=np.float64)
    for start in range(0, len(rows), 256):
        end = min(start + 256, len(rows))
        distances[start:end] = raw_distances(q[start:end, None, :], ref[ci[start:end]])
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    strict = np.zeros(len(ref), dtype=bool)
    strict[np.load(RESULTS_DIR / "strict_stable_reference_rows.npy")] = True
    baseline = pd.read_csv(RESULTS_DIR / "candidate_reference_scores.csv").iloc[rows]
    old_names = json.loads(
        (RESULTS_DIR / "reference_experiment_provenance.json").read_text()
    )["score_methods"]
    scores = {name: baseline[name].to_numpy(dtype=float) for name in old_names}
    scores["chemical_cosine_mean"] = np.full(len(rows), np.nan)
    scores["strict_chemical_mean"] = np.full(len(rows), np.nan)
    eligible = np.zeros(ci.shape, dtype=bool)
    strict_eligible = np.zeros(ci.shape, dtype=bool)
    chemical_count = np.zeros(len(rows), dtype=int)
    for k in cfg["ks"]:
        scores[f"cosine_chemical_knn_raw_k{k}"] = np.full(len(rows), np.nan)
        scores[f"cosine_chemical_knn_cos_k{k}"] = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        pool = ids[ptr[row] : ptr[row + 1]]
        chemical_count[i] = len(pool)
        eligible[i] = np.isin(ci[i], pool)
        strict_pool = pool[strict[pool]]
        strict_eligible[i] = eligible[i] & strict[ci[i]]
        if len(pool):
            cos = np.clip(1 - ru[pool] @ qu[i], 0, 2)
            scores["chemical_cosine_mean"][i] = cos.mean()
            order = np.lexsort((pool, cos))
            raw = raw_distances(q[i], ref[pool])
            for k in cfg["ks"]:
                scores[f"cosine_chemical_knn_raw_k{k}"][i] = raw[order[:k]].mean()
                scores[f"cosine_chemical_knn_cos_k{k}"][i] = cos[order[:k]].mean()
        if len(strict_pool):
            scores["strict_chemical_mean"][i] = raw_distances(
                q[i], ref[strict_pool]
            ).mean()
        if (i + 1) % 10000 == 0:
            print("Angular chemical pools:", i + 1, "/", len(rows), flush=True)
    original = scores["crys_jepa"]
    for k in cfg["ks"]:
        d = distances[:, :k]
        c = cd[:, :k]
        mask = eligible[:, :k]
        intersection = masked_mean(d, mask)
        scores[f"cosine_knn_raw_k{k}"] = d.mean(1)
        scores[f"cosine_knn_cos_k{k}"] = c.mean(1)
        scores[f"cosine_intersection_k{k}"] = intersection
        scores[f"cosine_intersection_fallback_k{k}"] = np.where(
            mask.any(1), intersection, original
        )
        scores[f"cosine_union_k{k}"] = (
            np.nan_to_num(original) * chemical_count
            + d.sum(1)
            - np.where(mask, d, 0).sum(1)
        ) / (chemical_count + k - mask.sum(1))
        for theta in cfg["theta_degrees"]:
            angular = within_angle(c, theta)
            value = masked_mean(d, angular)
            scores[f"cosine_theta{theta}_k{k}"] = value
            scores[f"cosine_theta{theta}_fallback_k{k}"] = np.where(
                angular.any(1), value, original
            )
            both = angular & mask
            value = masked_mean(d, both)
            scores[f"cosine_theta{theta}_intersection_fallback_k{k}"] = np.where(
                both.any(1), value, original
            )
    for metric in ["euclidean", "cosine"]:
        neighbours = np.load(
            RESULTS_DIR / f"candidate_strict_{metric}_indices.npy", mmap_mode="r"
        )[rows]
        raw = np.empty(neighbours.shape, dtype=float)
        mask = np.zeros(neighbours.shape, dtype=bool)
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            raw[start:stop] = raw_distances(
                q[start:stop, None, :], ref[neighbours[start:stop]]
            )
        for i, row in enumerate(rows):
            pool = ids[ptr[row] : ptr[row + 1]]
            mask[i] = np.isin(neighbours[i], pool)
        for k in cfg["ks"]:
            scores[f"strict_{metric}_knn_k{k}"] = raw[:, :k].mean(1)
            value = masked_mean(raw[:, :k], mask[:, :k])
            scores[f"strict_{metric}_intersection_fallback_k{k}"] = np.where(
                mask[:, :k].any(1), value, scores["strict_chemical_mean"]
            )
    names = list(scores)
    matrix = np.column_stack([scores[name] for name in names])
    assert not np.isinf(matrix).any()
    np.savez_compressed(
        out / f"{args.split}_angular_scores.npz",
        candidate_rows=rows,
        methods=np.array(names),
        scores=matrix,
    )
    # Describe angle filtering without reading any candidate outcome label.
    pd.DataFrame(
        [
            {
                "k": k,
                "theta_degrees": theta,
                "fraction_with_any_neighbour": float(
                    within_angle(cd[:, :k], theta).any(1).mean()
                ),
                "fraction_with_chemical_intersection": float(
                    (within_angle(cd[:, :k], theta) & eligible[:, :k]).any(1).mean()
                ),
            }
            for k in cfg["ks"]
            for theta in cfg["theta_degrees"]
        ]
    ).to_csv(out / f"{args.split}_theta_coverage.csv", index=False)
    report = {
        "split": args.split,
        "candidate_count": len(rows),
        "method_count": len(names),
        "methods": names,
        "theta": "maximum query-reference angle in degrees among top k cosine neighbours; no radius search beyond k",
        "primary_score": "mean RAW squared JEPA distance so retrieval changes are isolated; *_cos methods change the scoring metric explicitly",
        "bank": "known MP20 training references only; strict_* filters their recorded DFT hull distance to <=1e-8",
        "fallback": "original score only if filtered pool is empty; pure variants abstain with NaN",
        "candidate_outcome_labels_used_in_scoring": False,
        "test_scores_produced": False,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [Path(__file__).resolve(), CONFIG_DIR / "nonparametric.toml"]
        },
    }
    (out / f"{args.split}_angular_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "Saved",
        len(names),
        "methods for",
        len(rows),
        args.split,
        "candidates.",
        flush=True,
    )


if __name__ == "__main__":
    main()
