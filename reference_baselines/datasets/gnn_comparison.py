"""Freeze a future GNN holdout and load reference-only cosine graphs."""

import hashlib
import json

import numpy as np
from scipy import sparse
import torch

from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.datasets.training import ReferenceDataset
from reference_baselines.paths import RESULTS_DIR

OUT = RESULTS_DIR / "gnn_comparison"


def reserve_system(system):
    digest = hashlib.sha256(("gnn-reserve-v1|" + system).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < 0.2


def prepare():
    OUT.mkdir(exist_ok=True)
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    original = np.load(RESULTS_DIR / "training_train_rows.npy")
    validation = np.load(RESULTS_DIR / "training_validation_rows.npy")
    # These four training examples previously appeared in gradient/debug checks.
    smoke = json.loads((RESULTS_DIR / "training_smoke_check.json").read_text())
    touched_systems = set(
        manifest.loc[smoke["actual_training_rows_checked"], "chemical_system"]
    )
    held = (
        manifest.loc[original, "chemical_system"]
        .map(lambda s: reserve_system(s) and s not in touched_systems)
        .to_numpy(bool)
    )
    train, reserve = original[~held], original[held]
    if not len(train) or not len(reserve):
        raise ValueError("Empty fit or reserved set")
    for a, b in [(train, reserve), (train, validation), (reserve, validation)]:
        if set(manifest.loc[a, "chemical_system"]) & set(
            manifest.loc[b, "chemical_system"]
        ):
            raise ValueError("Chemical systems cross GNN splits")
    inputs = [
        RESULTS_DIR / "split_manifest.csv",
        RESULTS_DIR / "training_train_rows.npy",
        RESULTS_DIR / "training_validation_rows.npy",
        RESULTS_DIR / "training_smoke_check.json",
    ]
    record = {
        "date": "2026-10-22",
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "reserved": len(reserve),
        },
        "rule": "Hash old-training chemical systems with gnn-reserve-v1; reserve fraction 0.2; previously debugged systems stay in fit set",
        "reserved_labels_examined_for_this_split": False,
        "old_test_used": False,
        "caveat": "Prospective GNN holdout, not a wholly untouched research dataset: old train pool had aggregate and extreme-label quality checks. Do not call it independent publication validation.",
        "source_sha256": {str(p.relative_to(RESULTS_DIR)): sha256(p) for p in inputs},
    }
    protocol = OUT / "split_protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != record:
        raise ValueError("Frozen GNN allocation changed")
    for name, rows in [
        ("train", train),
        ("validation", validation),
        ("reserved", reserve),
    ]:
        path = OUT / f"{name}_rows.npy"
        if path.exists():
            np.testing.assert_array_equal(np.load(path), rows)
        else:
            np.save(path, rows)
    protocol.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


class CosineReferenceDataset(ReferenceDataset):
    """Known reference nodes only; candidates query each graph independently."""

    def __init__(self, split, limit=0, seed=2026):
        if split not in ["train", "validation"]:
            raise ValueError("Training code cannot open reserved or legacy test rows")
        super().__init__(split, pool="union", k=5)
        self.rows = np.load(OUT / f"{split}_rows.npy")
        if limit and limit < len(self.rows):
            self.rows = np.sort(
                np.random.default_rng(seed).choice(self.rows, limit, replace=False)
            )
        self.knn = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")
        self.graph = sparse.load_npz(
            RESULTS_DIR / "reference_cosine_graph_k10.npz"
        ).tocsr()

    def __getitem__(self, index):
        item = super().__getitem__(index)
        refs = item["reference_ids"]
        q = np.asarray(self.query[item["row"]], dtype=float)
        r = np.asarray(self.reference[refs], dtype=float)
        cos = np.clip(1 - r @ q / (np.linalg.norm(r, axis=1) * np.linalg.norm(q)), 0, 2)
        # Composition fractions identify reduced formulas without reading energy labels.
        _, inverse, count = np.unique(
            self.rcomp[refs], axis=0, return_inverse=True, return_counts=True
        )
        prior = 1.0 / (len(count) * count[inverse]) if len(count) else np.empty(0)
        item["pair"] = torch.from_numpy(
            np.column_stack([item["pair"].numpy(), cos]).astype(np.float32)
        )
        item["log_prior"] = torch.from_numpy(np.log(prior).astype(np.float32))
        return item


def collate(items):
    from reference_baselines.datasets.training import collate as original_collate

    batch = original_collate(items)
    batch["log_prior"] = torch.cat([item["log_prior"] for item in items])
    return batch


if __name__ == "__main__":
    prepare()
