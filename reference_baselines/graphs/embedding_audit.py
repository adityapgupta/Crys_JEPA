"""Describe stored embedding geometry without fitting a candidate predictor."""

from reference_baselines.core.audit_io import read_csv

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reference_baselines.core.io import sha256
from reference_baselines.evaluation.selection_audit import OUT
from reference_baselines.paths import ROOT, RESULTS_DIR


def pca_geometry(values):
    values = np.asarray(values, float)
    mean = values.mean(0)
    centered = values - mean
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    ordering = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[ordering], 0)
    return mean, eigenvalues, vectors[:, ordering]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = np.asarray(np.load(ROOT / "embeddings/data/embeddings.npy"), float)
    metadata = read_csv(ROOT / "embeddings/data/metadata.csv")
    mean, eigenvalues, axes = pca_geometry(refs)
    explained = eigenvalues / eigenvalues.sum()
    coordinates = (refs - mean) @ axes[:, :10]
    pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "variance_fraction": explained,
            "cumulative_variance": np.cumsum(explained),
        }
    ).to_csv(OUT / "pca_spectrum.csv", index=False)
    reference_frame = metadata[
        [
            "material_id",
            "pretty_formula",
            "chemical_system",
            "formation_energy_per_atom",
            "e_above_hull",
            "num_id",
        ]
    ].copy()
    reference_frame["embedding_norm"] = np.linalg.norm(refs, axis=1)
    for j in range(10):
        reference_frame[f"PC{j + 1}"] = coordinates[:, j]
    reference_frame.to_csv(OUT / "known_embedding_projection.csv", index=False)
    candidates = read_csv(OUT / "candidate_audit.csv")
    array = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    candidates["PC1"] = (
        np.asarray(array[candidates.candidate_row], float) - mean
    ) @ axes[:, 0]
    candidates["PC2"] = (
        np.asarray(array[candidates.candidate_row], float) - mean
    ) @ axes[:, 1]
    candidates.to_csv(OUT / "candidate_projection.csv", index=False)
    # Random pair / neighbour diagnostics on known crystals, not candidate outcomes.
    rng = np.random.default_rng(2026)
    a = rng.integers(0, len(refs), 30000)
    b = rng.integers(0, len(refs) - 1, 30000)
    b += b >= a
    norm = np.linalg.norm(refs, axis=1)
    raw = np.square(refs[a] - refs[b]).sum(1)
    cosine = np.clip(1 - (refs[a] * refs[b]).sum(1) / (norm[a] * norm[b]), 0, 2)
    radial = np.square(norm[a] - norm[b])
    angles = np.rad2deg(np.arccos(1 - cosine))
    energy = metadata.formation_energy_per_atom.to_numpy()
    gap = np.abs(energy[a] - energy[b])
    pd.DataFrame(
        {
            "reference_a": a,
            "reference_b": b,
            "angle_degrees": angles,
            "raw_squared_distance": raw,
            "formation_energy_gap": gap,
        }
    ).to_csv(OUT / "known_pair_diagnostics.csv", index=False)
    counts = np.bincount(
        np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")[
            candidates.loc[candidates.split.eq("validation"), "candidate_row"], :5
        ].ravel(),
        minlength=len(refs),
    )
    hubs = metadata[
        [
            "material_id",
            "pretty_formula",
            "chemical_system",
            "formation_energy_per_atom",
        ]
    ].copy()
    hubs["reference_row"] = np.arange(len(refs))
    hubs["validation_top5_retrieval_count"] = counts
    hubs.sort_values("validation_top5_retrieval_count", ascending=False).to_csv(
        OUT / "reference_hubs.csv", index=False
    )
    q = lambda x: {str(p): float(np.quantile(x, p)) for p in [0.05, 0.5, 0.95]}
    summary = {
        "pca_fit": "known MP20 references only; no candidate outcome labels",
        "shape": list(refs.shape),
        "vector_norm_quantiles": q(norm),
        "norm_coefficient_of_variation": float(norm.std() / norm.mean()),
        "centroid_norm": float(np.linalg.norm(mean)),
        "top1_variance": float(explained[0]),
        "top2_variance": float(explained[:2].sum()),
        "top10_variance": float(explained[:10].sum()),
        "components_for_90_percent": int(
            np.searchsorted(np.cumsum(explained), 0.9) + 1
        ),
        "participation_ratio": float(
            eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
        ),
        "known_norm_vs_formation_energy_spearman": float(
            spearmanr(norm, energy).statistic
        ),
        "known_PC_vs_formation_energy_spearman": [
            float(spearmanr(coordinates[:, j], energy).statistic) for j in range(10)
        ],
        "random_pair_count": len(a),
        "random_pair_angle_quantiles": q(angles),
        "random_pair_radial_fraction_quantiles": q(radial / np.maximum(raw, 1e-20)),
        "random_pair_raw_vs_cosine_spearman": float(spearmanr(raw, cosine).statistic),
        "random_pair_raw_vs_formation_gap_spearman": float(
            spearmanr(raw, gap).statistic
        ),
        "random_pair_cosine_vs_formation_gap_spearman": float(
            spearmanr(cosine, gap).statistic
        ),
        "top1_percent_known_references_share_of_validation_top5_links": float(
            np.sort(counts)[-int(np.ceil(0.01 * len(refs))) :].sum() / counts.sum()
        ),
        "unused_reference_fraction_validation_top5": float((counts == 0).mean()),
        "encoder_source": "components/jepa/frame/jepa.py:get_loss uses cosine similarity with formation-energy weights; encode returns a whole-crystal CLS vector",
        "interpretation_limits": [
            "PCA axes summarize variance, not a physical reaction coordinate.",
            "Known reference crystals may have been seen by the encoder; these diagnostics do not prove generalization.",
            "Post-hoc test projections are descriptive only; no embedding transform or model selection is tested here.",
            "A chemistry-independent energy-aware distance is not an energy-above-hull calculation.",
        ],
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                ROOT / "components/jepa/frame/jepa.py",
                ROOT / "embeddings/data/embeddings.npy",
            ]
        },
    }
    (OUT / "embedding_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
