"""Reference-utility models with explicit symmetry and chemistry edge features."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from reference_baselines.models.reference import segment_softmax, segment_sum
from reference_baselines.models.reference_gnn import ReferenceGNN


VARIANTS = ("scalar", "rich", "shuffled", "readout")


def reference_outputs(logits, batch, query_count, uniform_weights, prior_weights):
    """Apply the shared formula-prior weighting and candidate score readout."""
    owner = batch["owner"]
    if uniform_weights:
        weights = segment_softmax(torch.zeros_like(logits), owner, query_count)
    elif prior_weights:
        weights = segment_softmax(batch["log_prior"], owner, query_count)
    else:
        weights = segment_softmax(logits + batch["log_prior"], owner, query_count)
    score = segment_sum(weights * batch["distance"], owner, query_count)
    return logits, weights, score


class EdgeAttributeGCNLayer(nn.Module):
    def __init__(self, hidden, edge_features=3):
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_features, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge, weight, edge_attr):
        source, target = edge
        gate = 2 * torch.sigmoid(self.edge_gate(edge_attr).squeeze(1))
        effective = weight * gate
        degree = segment_sum(effective, target, len(x)).clamp_min(1e-12)
        scale = effective / (degree[source] * degree[target]).sqrt().clamp_min(1e-12)
        aggregate = segment_sum(self.message(x[source]) * scale[:, None], target, len(x))
        return self.norm(x + F.silu(self.update(torch.cat([x, aggregate], dim=1))))


class EdgeAttributeSAGELayer(nn.Module):
    def __init__(self, hidden, edge_features=3):
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_features, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge, weight, edge_attr):
        source, target = edge
        gate = 2 * torch.sigmoid(self.edge_gate(edge_attr).squeeze(1))
        effective = weight * gate
        degree = segment_sum(effective, target, len(x)).clamp_min(1e-12)
        aggregate = segment_sum(
            self.message(x[source]) * (effective / degree[target])[:, None],
            target,
            len(x),
        )
        return self.norm(x + F.silu(self.update(torch.cat([x, aggregate], dim=1))))


class EdgeAttributeGATLayer(nn.Module):
    def __init__(self, hidden, edge_features=3):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.message = nn.Linear(hidden, hidden)
        self.scalar_bias = nn.Linear(2, 1)
        self.rich_bias = nn.Sequential(
            nn.Linear(edge_features, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge, weight, edge_attr):
        source, target = edge
        scalar = torch.stack([weight, (source == target).to(weight.dtype)], dim=1)
        score = (self.q(x[target]) * self.k(x[source])).sum(1) / math.sqrt(x.shape[1])
        score = score + self.scalar_bias(scalar).squeeze(1)
        score = score + self.rich_bias(edge_attr).squeeze(1)
        attention = segment_softmax(score, target, len(x))
        aggregate = segment_sum(self.message(x[source]) * attention[:, None], target, len(x))
        return self.norm(x + F.silu(self.update(torch.cat([x, aggregate], dim=1))))


class EdgeAttributeReferenceGNN(ReferenceGNN):
    """The frozen architecture with rich features controlling graph messages."""

    def __init__(self, kind, hidden=64, layers=2):
        super().__init__(kind, hidden, layers=0)
        classes = {"gcn": EdgeAttributeGCNLayer, "gat": EdgeAttributeGATLayer, "sage": EdgeAttributeSAGELayer}
        if kind not in classes:
            raise ValueError("Edge attribute message passing requires gcn, gat or sage")
        self.layers = nn.ModuleList([classes[kind](hidden) for _ in range(layers)])

    def forward(self, batch, uniform_weights=False, prior_weights=False):
        query = self.encode(batch["query"])
        nodes = self.encode(batch["x"])
        owner = batch["owner"]
        for layer in self.layers:
            nodes = layer(nodes, batch["edge"], batch["weight"], batch["edge_attr"])
        logits = self.readout(
            torch.cat([query[owner], nodes, batch["pair"]], dim=1)
        ).squeeze(1)
        return reference_outputs(
            logits, batch, len(query), uniform_weights, prior_weights
        )


class EdgeReadoutReferenceModel(ReferenceGNN):
    """Uses edge summaries at readout, but performs no neighbour message passing."""

    def __init__(self, hidden=64):
        super().__init__("no_graph", hidden, layers=0)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden + 3 + 3, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, batch, uniform_weights=False, prior_weights=False):
        query = self.encode(batch["query"])
        nodes = self.encode(batch["x"])
        owner = batch["owner"]
        source, target = batch["edge"]
        nonself = source != target
        count = segment_sum(
            nonself.to(nodes.dtype), target, len(nodes)
        ).clamp_min(1)
        summary = segment_sum(
            batch["edge_attr"] * nonself[:, None], target, len(nodes)
        ) / count[:, None]
        logits = self.readout(
            torch.cat([query[owner], nodes, batch["pair"], summary], dim=1)
        ).squeeze(1)
        return reference_outputs(
            logits, batch, len(query), uniform_weights, prior_weights
        )


def build_model(variant, kind, hidden=64, layers=2):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if variant == "scalar":
        return ReferenceGNN(kind, hidden, layers)
    if variant == "readout":
        if kind != "readout":
            raise ValueError("The readout variant uses kind='readout'")
        return EdgeReadoutReferenceModel(hidden)
    return EdgeAttributeReferenceGNN(kind, hidden, layers)
