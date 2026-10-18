"""Score fixed held-out methods without reading candidate energy outcomes."""

import numpy as np
import pandas as pd
from scipy import sparse

from reference_baselines.baselines.angular import masked_mean
from reference_baselines.core.angular import raw_distances, within_angle
from reference_baselines.graphs.diffusion import (
    heat_features,
    resolvent_features,
    score_moments,
    transition_matrix,
)
from reference_baselines.paths import ROOT, RESULTS_DIR


def fixed_scores(rows, cfg):
    """Return the frozen method list; graphs contain known references only."""
    original_table = pd.read_csv(RESULTS_DIR / "candidate_reference_scores.csv")
    np.testing.assert_array_equal(
        original_table.candidate_row, np.arange(len(original_table))
    )
    scores = {
        name: original_table[name].to_numpy()[rows]
        for name in ["crys_jepa", "knn_k5", "intersection_fallback_k5"]
    }
    ref = np.asarray(np.load(ROOT / "embeddings/data/embeddings.npy"), dtype=float)
    query = np.asarray(
        np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")[rows],
        dtype=float,
    )
    k = cfg["knn_k"]
    limit = cfg["theta_k"]
    cosine = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")[
        rows, :limit
    ]
    cosdist = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")[
        rows, :limit
    ]
    distance = np.empty(cosine.shape)
    for start in range(0, len(rows), 256):
        stop = min(start + 256, len(rows))
        distance[start:stop] = raw_distances(
            query[start:stop, None, :], ref[cosine[start:stop]]
        )
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    mask = np.array(
        [
            np.isin(neighbours[:k], ids[ptr[row] : ptr[row + 1]])
            for row, neighbours in zip(rows, cosine)
        ]
    )
    original = scores["crys_jepa"]
    scores["cosine_knn_raw_k5"] = distance[:, :k].mean(1)
    scores["cosine_knn_cos_k5"] = cosdist[:, :k].mean(1)
    scores["cosine_intersection_fallback_k5"] = np.where(
        mask.any(1), masked_mean(distance[:, :k], mask), original
    )
    angular = within_angle(cosdist, cfg["theta_degrees"])
    gated = angular.any(1)
    scores["cosine_theta1_fallback_k10"] = np.where(
        gated, masked_mean(distance, angular), original
    )
    order = np.lexsort((rows, np.nan_to_num(original, nan=np.inf), ~gated))
    ranks = np.empty(len(rows), dtype=float)
    ranks[order] = np.arange(len(rows))
    ranks[(~gated) & ~np.isfinite(original)] = np.nan
    scores["theta1_gate_then_original_score"] = ranks
    checks = []
    features = np.column_stack([ref, np.square(ref).sum(1)])
    for metric, operator, parameter, seed_k, name in [
        (
            "euclidean",
            resolvent_features,
            cfg["resolvent_alpha"],
            cfg["resolvent_seed_k"],
            "euclidean_resolvent_0.25_seed_euclidean_k5",
        ),
        (
            "cosine",
            heat_features,
            cfg["heat_time"],
            cfg["heat_seed_k"],
            "cosine_heat_1_seed_cosine_k5",
        ),
    ]:
        filename = f"reference_{'cosine_' if metric == 'cosine' else ''}graph_k{cfg['graph_k']}.npz"
        p = transition_matrix(sparse.load_npz(RESULTS_DIR / filename))
        moments, diagnostic = operator(p, features, parameter, 1e-6, 200)
        if not diagnostic["converged"]:
            raise RuntimeError(f"Unconverged graph filter: {diagnostic}")
        ni = np.load(
            RESULTS_DIR
            / f"candidate_{'cosine' if metric == 'cosine' else 'knn'}_indices.npy",
            mmap_mode="r",
        )[rows, :seed_k]
        values = np.empty(len(rows))
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            values[start:stop] = score_moments(
                query[start:stop], moments[ni[start:stop]].mean(1)
            )
        scores[name] = values
        checks.append({"method": name, **diagnostic})
        print("Scored", name, flush=True)
    if set(scores) != set(cfg["methods"]):
        raise ValueError("Implementation and frozen methods differ")
    return np.column_stack([scores[name] for name in cfg["methods"]]), checks
