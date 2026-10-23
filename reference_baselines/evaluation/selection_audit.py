"""Explain fixed selections after evaluation; never change scores or methods."""

from reference_baselines.core.audit_io import read_csv

import json

import numpy as np
import pandas as pd

from reference_baselines.core.angular import within_angle
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.paths import BASELINE_RESULTS_DIR, ROOT, RESULTS_DIR

OUT = RESULTS_DIR / "interpretation"
ORIGINAL = "crys_jepa"
ANGLE = "theta1_gate_then_original_score"


def membership(rows, original, angle):
    a, b = np.isin(rows, original), np.isin(rows, angle)
    return np.select(
        [a & b, b & ~a, a & ~b], ["shared", "added", "removed"], default="neither"
    )


def score_parts(query, references):
    """Exact mean squared-distance identity, exposing pool spread."""
    query = np.asarray(query, dtype=float)
    references = np.asarray(references, dtype=float)
    centroid = references.mean(0)
    spread = np.square(references - centroid).sum(1).mean()
    centroid_distance = np.square(query - centroid).sum(-1)
    return centroid_distance, float(spread)


def split_selections(split, rows):
    if split == "test":
        table = read_csv(RESULTS_DIR / "heldout/selections.csv")
        return {
            method: table.loc[table.method.eq(method), "candidate_row"].to_numpy()
            for method in [ORIGINAL, ANGLE]
        }
    source = np.load(BASELINE_RESULTS_DIR / "validation_random_controls.npz")
    np.testing.assert_array_equal(source["candidate_rows"], rows)
    baseline = read_csv(
        RESULTS_DIR / "candidate_reference_scores.csv"
    ).crys_jepa.to_numpy()[rows]
    angle = source["scores"][:, source["methods"].tolist().index(ANGLE)]
    adjacency = {}
    for a, b in read_csv(RESULTS_DIR / "validation_duplicate_pairs.csv").itertuples(
        index=False, name=None
    ):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    selected = {}
    for name, values in [(ORIGINAL, baseline), (ANGLE, angle)]:
        available = np.isfinite(values)
        order = rows[available][np.lexsort((rows[available], values[available]))]
        selected[name] = unique_order(order, adjacency)[: int(len(rows) * 0.05)]
    return selected


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index(
        "candidate_row", drop=False
    )
    labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index("candidate_row")
    scores = read_csv(RESULTS_DIR / "candidate_reference_scores.csv").set_index(
        "candidate_row"
    )
    references = np.asarray(
        np.load(ROOT / "embeddings/data/embeddings.npy"), dtype=float
    )
    reference_metadata = read_csv(ROOT / "embeddings/data/metadata.csv")
    query = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    nearest = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")
    cosine = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")
    chemical_nearest = np.load(RESULTS_DIR / "chemical_knn_indices.npy", mmap_mode="r")
    ptr = np.load(RESULTS_DIR / "chemical_reference_indptr.npy")
    ids = np.load(RESULTS_DIR / "chemical_reference_indices.npy")
    frames, summaries = [], []
    for split in ["validation", "test"]:
        rows = manifest.index[manifest.vun & manifest.split.eq(split)].to_numpy()
        selected = split_selections(split, rows)
        f = manifest.loc[rows].copy()
        for key in [
            "formula",
            "num_atoms",
            "corrected_energy_per_atom",
            "e_above_hull_proxy",
        ]:
            f[key] = labels.loc[rows, key]
        f["passing"] = f.e_above_hull_proxy.le(0.1)
        f["selection_group"] = membership(rows, selected[ORIGINAL], selected[ANGLE])
        f["original_score"] = scores.loc[rows, ORIGINAL]
        f["chemical_reference_count"] = scores.loc[rows, "chemical_reference_count"]
        f["nearest_reference_row"] = nearest[rows, 0]
        f["nearest_chemical_reference_row"] = chemical_nearest[rows, 0]
        f["nearest_angle_degrees"] = np.rad2deg(
            np.arccos(np.clip(1 - cosine[rows, 0], -1, 1))
        )
        f["angle_gate"] = within_angle(cosine[rows, :10], 1).any(1)
        f["embedding_norm"] = np.linalg.norm(np.asarray(query[rows], float), axis=1)
        f["nearest_raw_squared_distance"] = np.square(
            np.asarray(query[rows], float) - references[nearest[rows, 0]]
        ).sum(1)
        f["nearest_radial_squared_distance"] = np.square(
            f.embedding_norm.to_numpy()
            - np.linalg.norm(references[nearest[rows, 0]], axis=1)
        )
        f["nearest_angular_squared_distance"] = (
            2
            * f.embedding_norm.to_numpy()
            * np.linalg.norm(references[nearest[rows, 0]], axis=1)
            * cosine[rows, 0]
        )
        np.testing.assert_allclose(
            f.nearest_raw_squared_distance,
            f.nearest_radial_squared_distance + f.nearest_angular_squared_distance,
            atol=1e-10,
        )
        f["centroid_squared_distance"] = np.nan
        f["chemical_pool_spread"] = np.nan
        f["elemental_reference_fraction"] = np.nan
        f["exact_system_reference_count"] = 0
        f["reference_formation_energy_std"] = np.nan
        f["reference_unique_formula_count"] = 0
        f["nearest_is_chemical_reference"] = False
        for _, group in f.groupby("chemical_system", sort=False):
            rr = group.candidate_row.to_numpy()
            pool = ids[ptr[rr[0]] : ptr[rr[0] + 1]]
            if not len(pool):
                continue
            distance, spread = score_parts(
                np.asarray(query[rr], float), references[pool]
            )
            f.loc[rr, "centroid_squared_distance"] = distance
            f.loc[rr, "chemical_pool_spread"] = spread
            rm = reference_metadata.iloc[pool]
            f.loc[rr, "elemental_reference_fraction"] = (
                rm.chemical_system.str.count("-").eq(0).mean()
            )
            f.loc[rr, "exact_system_reference_count"] = rm.chemical_system.eq(
                group.chemical_system.iloc[0]
            ).sum()
            f.loc[rr, "reference_formation_energy_std"] = (
                rm.formation_energy_per_atom.std(ddof=0)
            )
            f.loc[rr, "reference_unique_formula_count"] = rm.pretty_formula.nunique()
            f.loc[rr, "nearest_is_chemical_reference"] = np.isin(nearest[rr, 0], pool)
        reconstructed = f.centroid_squared_distance + f.chemical_pool_spread
        available = np.isfinite(f.original_score)
        error = np.abs(reconstructed[available] - f.original_score[available])
        np.testing.assert_allclose(
            reconstructed[available], f.original_score[available], atol=3e-5, rtol=2e-6
        )
        rank = pd.Series(np.nan, index=rows)
        finite_rows = f.index[available].to_numpy()
        ordered = finite_rows[
            np.lexsort((finite_rows, f.loc[finite_rows, "original_score"]))
        ]
        rank.loc[ordered] = np.arange(1, len(ordered) + 1)
        f["original_raw_rank"] = rank
        for name, values in selected.items():
            f[name + "_selected_rank"] = pd.Series(
                np.arange(1, len(values) + 1), index=values
            )
        counts = {}
        for group, values in f.groupby("selection_group"):
            counts[group] = {
                "count": len(values),
                "passing": int(values.passing.sum()),
                "pass_rate": float(values.passing.mean()),
                "median_original_score": float(values.original_score.median()),
                "median_pool_spread": float(values.chemical_pool_spread.median()),
                "median_angle": float(values.nearest_angle_degrees.median()),
                "nearest_chemically_eligible_fraction": float(
                    values.nearest_is_chemical_reference.mean()
                ),
            }
        gain = counts["added"]["passing"] - counts["removed"]["passing"]
        assert gain == int(
            f.loc[selected[ANGLE], "passing"].sum()
            - f.loc[selected[ORIGINAL], "passing"].sum()
        )
        summaries.append(
            {
                "split": split,
                "groups": counts,
                "net_passing_gain": gain,
                "mean_score_identity_max_error": float(error.max()),
            }
        )
        frames.append(f)
    frame = pd.concat(frames)
    frame.to_csv(OUT / "candidate_audit.csv", index=False)
    contributions = []
    for (split, system), group in frame.groupby(["split", "chemical_system"]):
        counts = {
            name: int(group.loc[group.selection_group.eq(name), "passing"].sum())
            for name in ["added", "removed"]
        }
        contributions.append(
            {
                "split": split,
                "chemical_system": system,
                **counts,
                "gain": counts["added"] - counts["removed"],
            }
        )
    pd.DataFrame(contributions).to_csv(OUT / "chemical_contributions.csv", index=False)
    write = {
        "purpose": "Post-hoc description of fixed validation/test selections, no tuning or new claimed test winner",
        "summaries": summaries,
        "candidate_energy_source": "saved MatterSim energies with MP corrections and fixed MP2023 hull",
        "score_identity": "mean ||q-r||^2 = ||q-mean(r)||^2 + mean ||r-mean(r)||^2",
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                ROOT / "reference_baselines/evaluation/selection_audit.py",
                RESULTS_DIR / "heldout/selections.csv",
                RESULTS_DIR / "candidate_labels.csv",
                OUT / "candidate_audit.csv",
            ]
        },
    }
    (OUT / "selection_audit.json").write_text(json.dumps(write, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
