"""Model builder and variant masks for the relational edge-feature experiment.

The gate / attention layers are reused unchanged from the edge-attribute
experiment (both take three edge features).  Per-feature variants are expressed
as a mask over the three feature slots, applied by the training / evaluation loop
before the model sees ``edge_attr`` -- so one ``default`` dataset serves
scalar/geom/cooc/phase/rich/readout, and only shuffled/redundant need their own.
"""

import torch

from reference_baselines.models.edge_attribute_reference_gnn import (
    EdgeAttributeReferenceGNN,
    EdgeReadoutReferenceModel,
)
from reference_baselines.models.reference_gnn import ReferenceGNN


VARIANTS = ("scalar", "geom", "cooc", "phase", "rich", "shuffled", "readout", "redundant")

# mask over (soap_distance, cooccurrence_pmi, hull_compatibility)
FEATURE_MASK = {
    "scalar": (0.0, 0.0, 0.0),
    "geom": (1.0, 0.0, 0.0),
    "cooc": (0.0, 1.0, 0.0),
    "phase": (0.0, 0.0, 1.0),
    "rich": (1.0, 1.0, 1.0),
    "shuffled": (1.0, 1.0, 1.0),
    "readout": (1.0, 1.0, 1.0),
    "redundant": (1.0, 0.0, 0.0),  # slot 0 carries embedding cosine in 'redundant' mode
}

# which dataset edge_mode each variant reads
EDGE_MODE = {
    "scalar": "default",
    "geom": "default",
    "cooc": "default",
    "phase": "default",
    "rich": "default",
    "readout": "default",
    "shuffled": "shuffled",
    "redundant": "redundant",
}


def mask_tensor(variant, device=None):
    return torch.tensor(FEATURE_MASK[variant], dtype=torch.float32, device=device)


def apply_mask(batch, variant):
    """Zero the feature slots this variant does not use (in place)."""
    if "edge_attr" in batch and batch["edge_attr"].numel():
        batch["edge_attr"] = batch["edge_attr"] * mask_tensor(variant, batch["edge_attr"].device)
    return batch


def build_model(variant, kind, hidden=64, layers=2):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant == "scalar":
        return ReferenceGNN(kind, hidden, layers)  # baseline: ignores edge_attr
    if variant == "readout":
        if kind != "readout":
            raise ValueError("The readout variant uses kind='readout'")
        return EdgeReadoutReferenceModel(hidden)
    return EdgeAttributeReferenceGNN(kind, hidden, layers)
