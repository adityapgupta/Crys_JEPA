"""Label-free UMAP of a verified reference export, cached by inputs and settings."""

from importlib.metadata import version
from pathlib import Path
import os
import time
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from .common import fingerprint, read_json, write_json


def project_reference_export(export_dir, n_neighbors=15, min_dist=.1, seed=2026):
    export_dir = Path(export_dir)
    # Keep compilation caches writable without changing the installed environment.
    compiler_cache = export_dir.resolve() / "umap" / ".numba-cache"
    compiler_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(compiler_cache))
    from umap import UMAP

    source = read_json(export_dir / "provenance.json")
    hashes = {name: sha256(export_dir / name) for name in ("embeddings.npy", "metadata.csv")}
    for name, key in (("embeddings.npy", "embeddings_sha256"), ("metadata.csv", "metadata_sha256")):
        if hashes[name] != source[key]:
            raise ValueError(f"Reference export checksum mismatch: {name}")
    embeddings = np.load(export_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(export_dir / "metadata.csv", keep_default_na=False)
    if embeddings.shape != (len(metadata), 512) or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite, row-aligned 512-dimensional MP-20 embeddings")
    np.testing.assert_array_equal(metadata.embedding_row, np.arange(len(metadata)))
    if not metadata.material_id.is_unique or not 2 <= n_neighbors < len(metadata):
        raise ValueError("Need unique IDs and 2 <= n_neighbors < number of rows")
    if not 0 <= min_dist <= 1:
        raise ValueError("min_dist must lie in [0,1]")
    settings = {"n_components": 2, "n_neighbors": int(n_neighbors), "min_dist": float(min_dist),
                "metric": "euclidean", "init": "spectral", "random_state": int(seed),
                "n_jobs": 1, "low_memory": True}
    identity = {"input_sha256": hashes, "parameters": settings,
                "packages": {name: version(name) for name in ("numpy", "umap-learn", "numba", "pynndescent", "scikit-learn")},
                "source_sha256": sha256(Path(__file__)), "normalization": "none", "PCA": "none",
                "fit_features": "all 512 raw embedding dimensions; no energy/metadata features"}
    directory = export_dir / "umap" / fingerprint(identity)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    coordinates = directory / "coordinates.npy"
    receipt = directory / "projection.json"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["identity"] != identity or sha256(coordinates) != saved["coordinates_sha256"]:
            raise ValueError("Modified or inconsistent cached UMAP projection")
        xy = np.load(coordinates, allow_pickle=False)
        print(f"Reusing full {len(metadata):,}-crystal UMAP: {directory}", flush=True)
    else:
        start = time.monotonic()
        print(f"Fitting UMAP to {embeddings.shape}; energy labels are not used for fitting.", flush=True)
        xy = UMAP(**settings, verbose=True).fit_transform(np.asarray(embeddings, dtype=np.float32))
        if xy.shape != (len(metadata), 2) or not np.isfinite(xy).all():
            raise ValueError("Invalid UMAP coordinates")
        temporary = directory / "coordinates.part.npy"
        np.save(temporary, xy, allow_pickle=False)
        temporary.replace(coordinates)
        write_json(receipt, {"identity": identity, "coordinates_sha256": sha256(coordinates),
                             "rows": len(metadata), "shape": list(xy.shape), "seconds": time.monotonic()-start,
                             "row_alignment": "coordinates[i] corresponds to metadata.iloc[i] and embeddings[i]",
                             "reference_export": source})
    if xy.shape != (len(metadata), 2) or not np.isfinite(xy).all():
        raise ValueError("Cached UMAP shape or values are invalid")
    return xy, metadata, directory


def equal_energy_groups(energies, groups=5):
    """Stable rank quantiles: equal counts; exact energy ties break by source row."""
    energy = np.asarray(energies, dtype=float)
    if energy.ndim != 1 or not np.isfinite(energy).all() or not 1 <= groups <= len(energy):
        raise ValueError("Need a finite energy vector with at least one row per group")
    labels = np.empty(len(energy), dtype=np.int32)
    summaries = []
    for label, rows in enumerate(np.array_split(np.argsort(energy, kind="stable"), groups)):
        labels[rows] = label
        summaries.append({"energy_group": label+1, "count": len(rows),
                          "min_eV_per_atom": float(energy[rows].min()),
                          "max_eV_per_atom": float(energy[rows].max())})
    return labels, pd.DataFrame(summaries)
