"""Edge attribute features on the frozen reference-utility graph and pools.

This module wraps the original ``PoolDataset`` without changing its cached
tables.  The graph topology, node features, query features, pools, targets and
split isolation therefore remain exactly the same as in the frozen reference
GNN experiment.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from reference_baselines.datasets.reference_graph import PoolDataset, collate
from reference_baselines.paths import ROOT


EDGE_FEATURES = (
    "embedding_cosine_similarity",
    "same_space_group",
    "electronegativity_profile_distance",
)
EDGE_MODES = ("rich", "shuffled")


@lru_cache(maxsize=None)
def pauling_electronegativities(width=100):
    """Return Pauling electronegativity by atomic number; missing values are NaN."""
    from pymatgen.core import Element

    values = np.full(width, np.nan, dtype=np.float64)
    for atomic_number in range(1, width):
        value = Element.from_Z(atomic_number).X
        if value is not None and np.isfinite(value):
            values[atomic_number] = float(value)
    return values


def electronegativity_profiles(composition, electronegativity=None):
    """Composition-weighted EN mean, spread and missing-mass, then standardise.

    Including missing electronegativity mass avoids silently treating elements
    without a Pauling value as having zero electronegativity.  The final edge
    feature is Euclidean distance in this three-component profile.
    """
    composition = np.asarray(composition, dtype=np.float64)
    values = (
        pauling_electronegativities(composition.shape[1])
        if electronegativity is None
        else np.asarray(electronegativity, dtype=np.float64)
    )
    if composition.ndim != 2 or values.shape != (composition.shape[1],):
        raise ValueError("Composition and electronegativity dimensions disagree")
    known = np.isfinite(values)
    known_mass = composition[:, known].sum(axis=1)
    weighted = composition[:, known] * values[known]
    mean = np.divide(
        weighted.sum(axis=1),
        known_mass,
        out=np.zeros(len(composition)),
        where=known_mass > 0,
    )
    variance = np.divide(
        (composition[:, known] * np.square(values[known] - mean[:, None])).sum(axis=1),
        known_mass,
        out=np.zeros(len(composition)),
        where=known_mass > 0,
    )
    raw = np.column_stack(
        [mean, np.sqrt(np.maximum(variance, 0)), np.clip(1 - known_mass, 0, 1)]
    )
    scale = raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return ((raw - raw.mean(axis=0)) / scale).astype(np.float32)


def edge_attributes(nodes, edge, space_group, en_profile):
    """Build the three edge features from aligned per-node descriptors."""
    nodes = np.asarray(nodes, dtype=np.float64)
    edge = np.asarray(edge, dtype=np.int64)
    source, target = edge
    norm = np.maximum(np.linalg.norm(nodes, axis=1), 1e-12)
    cosine = np.sum(nodes[source] * nodes[target], axis=1) / (norm[source] * norm[target])
    same_group = np.asarray(space_group)[source] == np.asarray(space_group)[target]
    en_distance = np.linalg.norm(
        np.asarray(en_profile)[source] - np.asarray(en_profile)[target], axis=1
    ) / np.sqrt(np.asarray(en_profile).shape[1])
    return np.column_stack(
        [np.clip(cosine, -1, 1), same_group.astype(float), en_distance]
    ).astype(np.float32)


class EdgeAttributePoolDataset(PoolDataset):
    """Original reference pools augmented with label-free edge attributes."""

    def __init__(self, *args, edge_mode="rich", edge_seed=2026, **kwargs):
        if edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {EDGE_MODES}")
        super().__init__(*args, **kwargs)
        metadata = pd.read_csv(
            ROOT / "embeddings/data/metadata.csv", usecols=["spacegroup.number"]
        )
        if len(metadata) != len(self.table["reference"]):
            raise ValueError("Reference metadata and embeddings are not aligned")
        space_group = metadata["spacegroup.number"].to_numpy()
        if pd.isna(space_group).any():
            raise ValueError("Reference space groups must be present")
        profile = electronegativity_profiles(self.table["rcomp"])
        if edge_mode == "shuffled":
            # Shuffle SG and EN jointly, retaining their mutual distribution but
            # breaking their alignment to reference identity and graph position.
            order = np.random.default_rng(edge_seed).permutation(len(space_group))
            space_group, profile = space_group[order], profile[order]
        self.edge_mode = edge_mode
        self.edge_seed = edge_seed
        self.reference_space_group = space_group
        self.reference_en_profile = profile

    def __getitem__(self, index):
        item = super().__getitem__(index)
        refs = item["reference_ids"]
        item["edge_attr"] = torch.from_numpy(
            edge_attributes(
                item["x"][:, :512].numpy(),
                item["edge"].numpy(),
                self.reference_space_group[refs],
                self.reference_en_profile[refs],
            )
        )
        return item


def edge_attribute_collate(items):
    """Original disjoint-graph collation plus aligned edge attributes."""
    batch = collate(items)
    batch["edge_attr"] = torch.cat([item["edge_attr"] for item in items])
    return batch
