"""Export actual known-reference IDs and audit physical phase recall on validation."""

import json
import re

import numpy as np
import pandas as pd
from scipy import sparse

from reference_baselines.core.angular import within_angle
from reference_baselines.graphs.diffusion import (
    heat_seed_distribution,
    transition_matrix,
)
from reference_baselines.paths import BASELINE_RESULTS_DIR, ROOT, RESULTS_DIR


def main():
    out = BASELINE_RESULTS_DIR
    rows = np.load(out / "validation_angular_scores.npz")["candidate_rows"]
    rm = pd.read_csv(ROOT / "embeddings/data/metadata.csv")
    cm = pd.read_csv(ROOT / "embeddings/data/candidate_metadata.csv")
    ref = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    q = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    ei = np.load(RESULTS_DIR / "candidate_knn_indices.npy", mmap_mode="r")
    ci = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")
    angles = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")
    p = transition_matrix(
        sparse.load_npz(RESULTS_DIR / "reference_cosine_graph_k10.npz")
    )
    examples = []
    direct_errors = []
    diffusion = np.load(out / "validation_diffusion_cosine.npz")
    method = list(diffusion["methods"]).index("cosine_heat_1_seed_cosine_k5")
    for i in range(min(8, len(rows))):
        row = int(rows[i])
        chemical = ids[ptr[row] : ptr[row + 1]]
        pools = {"original_chemical": chemical}
        for name, retrieved in [
            ("euclidean_intersection_fallback_k5", ei[row, :5]),
            ("cosine_intersection_fallback_k5", ci[row, :5]),
        ]:
            pool = np.intersect1d(chemical, retrieved)
            pools[name] = pool if len(pool) else chemical
        pool = ci[row, :10][within_angle(angles[row, :10], 1)]
        pools["cosine_theta1_fallback_k10"] = pool if len(pool) else chemical
        for name, pool in pools.items():
            for refrow in pool:
                examples.append(
                    {
                        "candidate_row": row,
                        "eval_folder": cm.loc[row, "eval_folder"],
                        "candidate_idx": cm.loc[row, "idx"],
                        "method": name,
                        "reference_row": int(refrow),
                        "material_id": rm.loc[refrow, "material_id"],
                        "formula": rm.loc[refrow, "pretty_formula"],
                        "reference_recorded_e_above_hull": rm.loc[
                            refrow, "e_above_hull"
                        ],
                        "weight": 1 / len(pool),
                        "chemically_eligible": bool(refrow in chemical),
                    }
                )
        seed = np.zeros((len(ref), 1))
        seed[ci[row, :5], 0] = 1 / 5
        weights, _ = heat_seed_distribution(
            p, seed, 1, tolerance=1e-11, max_iterations=100
        )
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-9)
        assert weights.min() >= -1e-12
        raw = np.square(np.asarray(ref, float) - q[row]).sum(1)
        actual = float(weights @ raw)
        expected = diffusion["scores"][i, method]
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-6)
        direct_errors.append(abs(actual - expected))
        for refrow in np.lexsort((np.arange(len(ref)), -weights))[:20]:
            examples.append(
                {
                    "candidate_row": row,
                    "eval_folder": cm.loc[row, "eval_folder"],
                    "candidate_idx": cm.loc[row, "idx"],
                    "method": "cosine_heat_1_seed_cosine_k5_top20_weights",
                    "reference_row": int(refrow),
                    "material_id": rm.loc[refrow, "material_id"],
                    "formula": rm.loc[refrow, "pretty_formula"],
                    "reference_recorded_e_above_hull": rm.loc[refrow, "e_above_hull"],
                    "weight": weights[refrow],
                    "chemically_eligible": bool(refrow in chemical),
                }
            )
    pd.DataFrame(examples).to_csv(out / "reference_examples.csv", index=False)
    labels = pd.read_csv(RESULTS_DIR / "candidate_labels.csv").set_index(
        "candidate_row"
    )
    material_to_row = {str(material): i for i, material in enumerate(rm.material_id)}
    physical = []
    for row in rows:
        targets = {}
        for product in json.loads(labels.loc[row, "decomposition"]):
            match = re.match(r"^(mp-\d+)", product["entry_id"])
            if match and match.group(1) in material_to_row:
                target = material_to_row[match.group(1)]
                targets[target] = targets.get(target, 0) + product["fraction"]
        available = sum(targets.values())
        strict_available = sum(
            v for r, v in targets.items() if rm.loc[r, "e_above_hull"] <= 1e-8
        )
        for metric, knn in [("euclidean", ei), ("cosine", ci)]:
            for k in [5, 10, 20, 50]:
                pool = set(knn[row, :k])
                recovered = sum(v for r, v in targets.items() if r in pool)
                physical.append(
                    {
                        "metric": metric,
                        "k": k,
                        "available_weight": available,
                        "strict_available_weight": strict_available,
                        "recovered_weight": recovered,
                        "recall_when_available": recovered / available
                        if available
                        else np.nan,
                    }
                )
    pd.DataFrame(physical).groupby(["metric", "k"]).agg(
        queries=("available_weight", "size"),
        queries_with_available_phase=("recall_when_available", "count"),
        available_weight=("available_weight", "mean"),
        strict_available_weight=("strict_available_weight", "mean"),
        recovered_weight=("recovered_weight", "mean"),
        recall_when_available=("recall_when_available", "mean"),
    ).to_csv(out / "decomposition_recall.csv")
    report = {
        "status": "passed",
        "example_queries": min(8, len(rows)),
        "diffusion_scalar_checks": len(direct_errors),
        "max_diffusion_score_error": max(direct_errors),
        "example_scope": "first eight validation queries; not selected for success",
        "weight_note": "heat examples export top 20 weights only; their sum can be below one",
        "test_outcome_labels_read": False,
    }
    (out / "reference_example_checks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
