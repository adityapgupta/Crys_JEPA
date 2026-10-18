"""Label-free random-walk resolvent and Laplacian heat filters on known references."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from reference_baselines.core.io import sha256
from reference_baselines.core.settings import baseline_settings
from reference_baselines.paths import (
    BASELINE_RESULTS_DIR,
    CONFIG_DIR,
    ROOT,
    RESULTS_DIR,
)


def transition_matrix(graph):
    graph = graph.tocsr().astype(np.float64)
    degree = np.asarray(graph.sum(1)).ravel()
    dangling = degree == 0
    if dangling.any():
        graph = graph + sparse.diags(dangling.astype(float))
        degree = np.asarray(graph.sum(1)).ravel()
    return sparse.diags(1 / degree) @ graph


def resolvent_features(p, features, alpha, tolerance=1e-6, max_iterations=100):
    """(1-alpha)(I-alpha P)^-1 F, with a sup-norm contraction error bound."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1).")
    h = np.asarray(features, dtype=np.float64).copy()
    error = 0.0
    for iteration in range(1, max_iterations + 1):
        updated = (1 - alpha) * features + alpha * (p @ h)
        delta = float(np.max(np.abs(updated - h)))
        h = updated
        error = alpha * delta / (1 - alpha)
        if error <= tolerance:
            break
    return h, {
        "iterations": iteration,
        "feature_sup_error_bound": error,
        "converged": error <= tolerance,
    }


def heat_features(p, features, time_value, tolerance=1e-6, max_iterations=100):
    """exp(-t L_rw)F using a nonnegative Poisson expansion; L_rw=I-P."""
    if time_value < 0:
        raise ValueError("Heat time must be nonnegative.")
    power = np.asarray(features, dtype=np.float64).copy()
    probability = math.exp(-time_value)
    h = probability * power
    mass = probability
    largest = float(np.max(np.abs(features)))
    for iteration in range(1, max_iterations + 1):
        power = p @ power
        probability *= time_value / iteration
        h += probability * power
        mass += probability
        error = max(0.0, 1 - mass) * largest
        if 2 * error <= tolerance:
            break
    # Retain the remaining Poisson mass at the current walk power. Rows stay probability distributions.
    tail = max(0.0, 1 - mass)
    h += tail * power
    return h, {
        "iterations": iteration,
        "feature_sup_error_bound": 2 * tail * largest,
        "converged": 2 * tail * largest <= tolerance,
        "poisson_tail_mass": tail,
    }


def score_moments(query, moments):
    return np.maximum(
        np.square(query).sum(1) + moments[:, -1] - 2 * (query * moments[:, :-1]).sum(1),
        0.0,
    )


def heat_seed_distribution(p, seed, time_value, tolerance=1e-11, max_iterations=100):
    """Dual probability walk: exp(-t L_rw.T) seed, with a total-mass error bound."""
    seed = np.asarray(seed, dtype=float).ravel()
    if np.any(seed < 0) or not np.isclose(seed.sum(), 1):
        raise ValueError("Seed must be a probability distribution.")
    power = seed.copy()
    probability = math.exp(-time_value)
    result = probability * power
    mass = probability
    for iteration in range(1, max_iterations + 1):
        power = p.T @ power
        probability *= time_value / iteration
        result += probability * power
        mass += probability
        if 2 * max(0.0, 1 - mass) <= tolerance:
            break
    tail = max(0.0, 1 - mass)
    result += tail * power
    return result, {"iterations": iteration, "probability_l1_error_bound": 2 * tail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=["train", "validation"], default="validation"
    )
    parser.add_argument(
        "--graph-metric", choices=["euclidean", "cosine"], required=True
    )
    args = parser.parse_args()
    cfg = baseline_settings()
    out = BASELINE_RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv")
    rows = np.flatnonzero(manifest.vun & manifest.split.eq(args.split))
    ref = np.asarray(np.load(ROOT / "embeddings/data/embeddings.npy"), dtype=np.float64)
    query = np.asarray(
        np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")[rows],
        dtype=np.float64,
    )
    features = np.column_stack([ref, np.square(ref).sum(1)])
    filename = (
        f"reference_graph_k{cfg['graph_k']}.npz"
        if args.graph_metric == "euclidean"
        else f"reference_cosine_graph_k{cfg['graph_k']}.npz"
    )
    graph = sparse.load_npz(RESULTS_DIR / filename)
    p = transition_matrix(graph)
    np.testing.assert_allclose(np.asarray(p.sum(1)).ravel(), 1.0, atol=1e-12)
    neighbours = {
        metric: np.load(
            RESULTS_DIR
            / (
                "candidate_knn_indices.npy"
                if metric == "euclidean"
                else "candidate_cosine_indices.npy"
            ),
            mmap_mode="r",
        )[rows]
        for metric in ["euclidean", "cosine"]
    }
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    pools = [ids[ptr[row] : ptr[row + 1]] for row in rows]
    scores = {}
    checks = []
    details = []
    schedules = [("resolvent", alpha) for alpha in cfg["diffusion_alpha"]] + [
        ("heat", t) for t in cfg["spectral_times"]
    ]
    for family, parameter in schedules:
        started = time.perf_counter()
        if family == "resolvent":
            h, diagnostic = resolvent_features(
                p,
                features,
                parameter,
                cfg["diffusion_tolerance"],
                cfg["diffusion_max_iterations"],
            )
        else:
            h, diagnostic = heat_features(
                p,
                features,
                parameter,
                cfg["diffusion_tolerance"],
                cfg["diffusion_max_iterations"],
            )
        prefix = f"{args.graph_metric}_{family}_{parameter:g}"
        for metric, indices in neighbours.items():
            for k in cfg["diffusion_seed_k"]:
                values = np.empty(len(rows))
                for start in range(0, len(rows), 256):
                    end = min(start + 256, len(rows))
                    moments = h[indices[start:end, :k]].mean(1)
                    values[start:end] = score_moments(query[start:end], moments)
                scores[f"{prefix}_seed_{metric}_k{k}"] = values
        moments = np.array(
            [
                h[pool].mean(0) if len(pool) else np.full(h.shape[1], np.nan)
                for pool in pools
            ]
        )
        scores[f"{prefix}_seed_chemical"] = score_moments(query, moments)
        diagnostic.update(
            family=family, parameter=parameter, seconds=time.perf_counter() - started
        )
        details.append(diagnostic)
        print(args.graph_metric, family, parameter, diagnostic, flush=True)
    # Alpha=0 must recover the original kNN mean distances exactly, independent of graph topology.
    for metric, indices in neighbours.items():
        sample = np.arange(0, len(rows), max(1, len(rows) // 64))
        expected = (
            np.square(query[sample, None, :] - ref[indices[sample, :5]]).sum(2).mean(1)
        )
        actual = score_moments(query[sample], features[indices[sample, :5]].mean(1))
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        checks.append(
            {
                "metric": metric,
                "zero_diffusion_queries": len(sample),
                "max_error": float(np.max(np.abs(actual - expected))),
            }
        )
    np.savez_compressed(
        out / f"{args.split}_diffusion_{args.graph_metric}.npz",
        candidate_rows=rows,
        methods=np.array(list(scores)),
        scores=np.column_stack(list(scores.values())),
    )
    report = {
        "graph_metric": args.graph_metric,
        "graph_k": cfg["graph_k"],
        "graph_nodes": len(ref),
        "components": int(
            connected_components(graph, directed=False, return_labels=False)
        ),
        "split": args.split,
        "candidate_count": len(rows),
        "method_count": len(scores),
        "filters": details,
        "checks": checks,
        "references": "known MP20 nodes only; candidates provide query seeds and never become reference nodes",
        "score": "expected RAW squared JEPA distance under query-seeded reference diffusion; computed exactly from first/second moments up to reported filter error",
        "spectral_relation": "L_rw=I-P; resolvent eigenfilter (1-alpha)/(1-alpha+alpha*lambda); heat eigenfilter exp(-t*lambda)",
        "label_use": "no candidate labels or reference stability labels in graph filters",
        "test_performance_evaluated": False,
        "sha256": {
            str(f.relative_to(ROOT)): sha256(f)
            for f in [
                Path(__file__).resolve(),
                CONFIG_DIR / "nonparametric.toml",
                RESULTS_DIR / filename,
            ]
        },
    }
    (out / f"{args.split}_diffusion_{args.graph_metric}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
