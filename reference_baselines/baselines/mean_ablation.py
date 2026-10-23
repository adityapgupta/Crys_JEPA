"""Separate reference-pool changes, averaging, and angular candidate gating."""

import json

import numpy as np

from reference_baselines.core.angular import raw_distances, within_angle
from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "mean_ablation"
TEMPERATURE = 10.0
POOLS = (
    "chemical",
    "union_cosine5",
    "intersection_cosine5_fallback",
    "cosine5",
    "theta1_top10_fallback",
)


def aggregate(distances, temperature=None):
    """Mean squared distances, optionally with normalized exp(-d/T) weights.

    Distances, rather than reference embedding vectors, are averaged. Weights
    use no candidate energy labels and are recomputed within each chosen pool.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[1] == 0 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Need finite nonnegative distances and a nonempty pool")
    if temperature is None:
        return d.mean(axis=1)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    weights = np.exp(-(d - d.min(axis=1, keepdims=True)) / temperature)
    return (weights * d).sum(axis=1) / weights.sum(axis=1)


def pool_ids(chemical, neighbours):
    """Enhance the reference set without counting an existing crystal twice."""
    return np.union1d(chemical, neighbours)


def gate_ranks(rows, scores, gated):
    order = np.lexsort((rows, np.nan_to_num(scores, nan=np.inf), ~gated))
    ranks = np.empty(len(rows), dtype=float)
    ranks[order] = np.arange(len(rows))
    ranks[(~gated) & ~np.isfinite(scores)] = np.nan
    return ranks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    frame = manifest[manifest.vun & manifest.split.isin(["validation", "test"])]
    rows = frame.index.to_numpy()
    lookup = {int(row): i for i, row in enumerate(rows)}
    ref = np.asarray(np.load(ROOT / "embeddings/data/embeddings.npy"), dtype=float)
    query = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    neighbours = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")
    cosine = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    stored = read_csv(RESULTS_DIR / "candidate_reference_scores.csv").set_index(
        "candidate_row"
    )
    scores = {
        f"{pool}_{mean}": np.full(len(rows), np.nan)
        for pool in POOLS
        for mean in ["mean", "weighted"]
    }
    pool_sizes = {pool: np.zeros(len(rows), dtype=int) for pool in POOLS}
    fallback = {pool: np.zeros(len(rows), dtype=bool) for pool in POOLS}
    # Compute full chemical-pool means in groups, then the small neighbour pools.
    for _, group in frame.groupby("chemical_system", sort=False):
        rr = group.index.to_numpy()
        chemical = ids[ptr[rr[0]] : ptr[rr[0] + 1]]
        if not len(chemical):
            continue
        r = ref[chemical]
        for start in range(0, len(rr), 256):
            chosen = rr[start : start + 256]
            position = np.array([lookup[int(row)] for row in chosen])
            q = np.asarray(query[chosen], dtype=float)
            distance = np.maximum(
                np.square(q).sum(1)[:, None]
                + np.square(r).sum(1)[None, :]
                - 2 * q @ r.T,
                0,
            )
            scores["chemical_mean"][position] = aggregate(distance)
            scores["chemical_weighted"][position] = aggregate(distance, TEMPERATURE)
            pool_sizes["chemical"][position] = len(chemical)
    np.testing.assert_allclose(
        scores["chemical_mean"], stored.loc[rows, "crys_jepa"], rtol=2e-6, atol=3e-5
    )
    np.testing.assert_allclose(
        scores["chemical_weighted"],
        stored.loc[rows, "chemical_softmax_t10"],
        rtol=2e-6,
        atol=3e-5,
    )
    # Preserve the exact released original score and tie order for its control.
    scores["chemical_mean"] = stored.loc[rows, "crys_jepa"].to_numpy(float)
    gated = within_angle(cosine[rows, :10], 1).any(1)
    for i, row in enumerate(rows):
        chemical = ids[ptr[row] : ptr[row + 1]]
        near = neighbours[row, :5]
        sets = {
            "union_cosine5": pool_ids(chemical, near),
            "intersection_cosine5_fallback": np.intersect1d(chemical, near),
            "cosine5": near,
            "theta1_top10_fallback": neighbours[row, :10][
                within_angle(cosine[row, :10], 1)
            ],
        }
        for name, pool in sets.items():
            if not len(pool):
                pool_sizes[name][i] = len(chemical)
                fallback[name][i] = True
                for mean in ["mean", "weighted"]:
                    scores[f"{name}_{mean}"][i] = scores[f"chemical_{mean}"][i]
                continue
            pool_sizes[name][i] = len(pool)
            distance = raw_distances(np.asarray(query[row], dtype=float), ref[pool])[
                None, :
            ]
            scores[f"{name}_mean"][i] = aggregate(distance)[0]
            scores[f"{name}_weighted"][i] = aggregate(distance, TEMPERATURE)[0]
        if (i + 1) % 15000 == 0:
            print("Pool ablation scored:", i + 1, "/", len(rows), flush=True)
    for split in ["validation", "test"]:
        mask = frame.split.eq(split).to_numpy()
        split_rows = rows[mask]
        values = {name: value[mask] for name, value in scores.items()}
        for mean in ["mean", "weighted"]:
            values[f"gate_then_chemical_{mean}"] = gate_ranks(
                split_rows, values[f"chemical_{mean}"], gated[mask]
            )
        np.savez_compressed(
            OUT / f"{split}_scores.npz",
            candidate_rows=split_rows,
            methods=np.array(list(values)),
            scores=np.column_stack(list(values.values())),
        )
    frame = frame.copy()
    for name in POOLS:
        frame[f"{name}_size"] = pool_sizes[name]
        frame[f"{name}_fallback"] = fallback[name]
    frame.to_csv(OUT / "pool_checks.csv")
    paths = [
        ROOT / "embeddings/data" / name
        for name in ["embeddings.npy", "candidate_embeddings.npy"]
    ]
    paths += [
        RESULTS_DIR / name
        for name in [
            "candidate_reference_scores.csv",
            "split_manifest.csv",
            "chemical_reference_indptr.npy",
            "chemical_reference_indices.npy",
            "candidate_cosine_indices.npy",
            "candidate_cosine_distances.npy",
        ]
    ]
    protocol = {
        "k": 5,
        "angle_degrees": 1,
        "angle_neighbour_cap": 10,
        "temperature": TEMPERATURE,
        "weight_formula": "exp(-(squared_distance - minimum_in_pool) / 10), normalized within pool",
        "temperature_reason": "Existing chemical_softmax_t10 baseline; fixed for this attribution study, not selected from these outcomes",
        "primary_pool_change": "chemical references union five global cosine neighbours, unique reference row IDs",
        "score": "mean or weighted mean of raw squared Euclidean distances, never distance to a mean embedding",
        "fallback": "Empty intersection/angle pool uses full chemical pool with the same averaging rule",
        "gate_control": "Prioritize any known reference within 1 degree; rank each group with full chemical mean or weighted mean",
        "candidate_energy_labels_read_for_scoring": False,
        "training": False,
        "status": "Exploratory attribution on validation and already-open test; no independent new test claim",
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Saved validation and test scores", flush=True)


if __name__ == "__main__":
    main()
