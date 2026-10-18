"""Independent full-sort cosine checks and scalar checks of the new reference scores."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from reference_baselines.core.angular import unit_vectors
from reference_baselines.core.io import sha256
from reference_baselines.paths import BASELINE_RESULTS_DIR, ROOT, RESULTS_DIR


def main():
    out = BASELINE_RESULTS_DIR
    ref = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    candidate = np.load(
        ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r"
    )
    rm = pd.read_csv(ROOT / "embeddings/data/metadata.csv")
    cm = pd.read_csv(ROOT / "embeddings/data/candidate_metadata.csv")
    ru = unit_vectors(ref)
    rng = np.random.default_rng(2026)
    max_error = 0.0
    search_checks = 0
    for bank, metric in [
        ("full", "cosine"),
        ("strict", "cosine"),
        ("strict", "euclidean"),
    ]:
        prefix = "candidate_cosine" if bank == "full" else f"candidate_strict_{metric}"
        indices = np.load(RESULTS_DIR / f"{prefix}_indices.npy", mmap_mode="r")
        distance = np.load(RESULTS_DIR / f"{prefix}_distances.npy", mmap_mode="r")
        assert indices.shape == (len(cm), 50)
        assert indices.min() >= 0 and indices.max() < len(ref)
        if bank == "strict":
            assert (rm.e_above_hull.to_numpy()[indices] <= 1e-8).all()
        for row in rng.choice(len(cm), 64, replace=False):
            allowed = (
                np.arange(len(ref))
                if bank == "full"
                else np.flatnonzero(rm.e_above_hull.to_numpy() <= 1e-8)
            )
            if metric == "cosine":
                query = unit_vectors(candidate[row : row + 1])[0]
                exact = np.clip(1 - ru[allowed] @ query, 0, 2)
                actual = np.clip(1 - ru[indices[row]] @ query, 0, 2)
            else:
                exact = np.square(np.asarray(ref[allowed], float) - candidate[row]).sum(
                    1
                )
                actual = np.square(
                    np.asarray(ref[indices[row]], float) - candidate[row]
                ).sum(1)
            expected = np.sort(exact)[:50]
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(actual, distance[row], atol=1e-10, rtol=1e-10)
            max_error = max(max_error, float(np.max(np.abs(actual - distance[row]))))
            search_checks += 1
    for k in [5, 10, 20, 50]:
        graph = sparse.load_npz(RESULTS_DIR / f"reference_cosine_graph_k{k}.npz")
        assert graph.shape == (len(ref), len(ref)) and (graph - graph.T).nnz == 0
        assert (
            graph.diagonal().sum() == 0
            and graph.data.min() > 0
            and graph.data.max() <= 1
        )
    angular = np.load(out / "validation_angular_scores.npz")
    rows = angular["candidate_rows"]
    table = pd.DataFrame(angular["scores"], columns=angular["methods"], index=rows)
    original = pd.read_csv(
        RESULTS_DIR / "candidate_reference_scores.csv"
    ).crys_jepa.to_numpy()
    ci = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")
    scalar_checks = 0
    for row in rng.choice(rows, 64, replace=False):
        chemical = set(cm.loc[row, "chemical_system"].split("-"))
        for k in [5, 10, 20, 50]:
            pool = ci[row, :k]
            direct = np.square(np.asarray(ref[pool], float) - candidate[row]).sum(1)
            np.testing.assert_allclose(
                table.loc[row, f"cosine_knn_raw_k{k}"], direct.mean(), atol=1e-10
            )
            eligible = np.array(
                [
                    set(rm.loc[r, "chemical_system"].split("-")).issubset(chemical)
                    for r in pool
                ]
            )
            expected = direct[eligible].mean() if eligible.any() else original[row]
            np.testing.assert_allclose(
                table.loc[row, f"cosine_intersection_fallback_k{k}"],
                expected,
                atol=1e-10,
                equal_nan=True,
            )
            scalar_checks += 2
    diffusion = []
    for metric in ["euclidean", "cosine"]:
        report = json.loads((out / f"validation_diffusion_{metric}.json").read_text())
        assert all(item["converged"] for item in report["filters"])
        diffusion.append(
            {
                "metric": metric,
                "all_filters_converged": True,
                "max_feature_error_bound": max(
                    item["feature_sup_error_bound"] for item in report["filters"]
                ),
            }
        )
    result = {
        "status": "passed",
        "full_sort_query_checks": search_checks,
        "max_distance_error": max_error,
        "scalar_angular_score_checks": scalar_checks,
        "graph_checks": 4,
        "diffusion": diffusion,
        "reference_source_checked": "all retrieved IDs index known MP20 rows; strict bank verified against known reference metadata",
        "test_outcome_labels_read": False,
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [Path(__file__).resolve(), out / "validation_angular_scores.npz"]
        },
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
