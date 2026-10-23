"""Graph models that predict the usefulness of each known reference crystal.

The model reads known-reference nodes and reference-reference edges. Each node
receives a query-conditioned score. The normalised scores give reference
weights. The candidate score is the weighted mean JEPA distance.

The model never reads a candidate stability label.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from reference_baselines.models.reference import segment_sum, segment_softmax


KINDS = ("gcn", "gat", "no_graph")


class GCNLayer(nn.Module):
    """Symmetric degree-normalised convolution with edge weights."""

    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge, weight):
        source, target = edge
        degree = segment_sum(weight, target, len(x)).clamp_min(1e-12)
        scale = weight / (degree[source] * degree[target]).sqrt().clamp_min(1e-12)
        aggregate = segment_sum(self.message(x[source]) * scale[:, None], target, len(x))
        return self.norm(x + F.silu(self.update(torch.cat([x, aggregate], dim=1))))


class GATLayer(nn.Module):
    """Single-head edge-aware attention inside each candidate's pool."""

    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.message = nn.Linear(hidden, hidden)
        self.edge_bias = nn.Linear(2, 1)
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge, weight, owner):
        source, target = edge
        features = torch.stack([weight, (source == target).to(weight.dtype)], dim=1)
        score = (self.q(x[target]) * self.k(x[source])).sum(1) / math.sqrt(x.shape[1])
        score = score + self.edge_bias(features).squeeze(1)
        attention = segment_softmax(score, target, len(x))
        # A node without incident edges keeps zero attention to other nodes.
        aggregate = segment_sum(self.message(x[source]) * attention[:, None], target, len(x))
        return self.norm(x + F.silu(self.update(torch.cat([x, aggregate], dim=1))))


class ReferenceGNN(nn.Module):
    """Predict per-reference utility, then weight reference distances."""

    def __init__(self, kind, hidden=64, layers=2):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Kind must be one of {KINDS}, not {kind!r}")
        self.kind = kind
        self.embedding_norm = nn.LayerNorm(512)
        self.project = nn.Sequential(nn.Linear(613, hidden), nn.SiLU())
        if kind == "gcn":
            self.layers = nn.ModuleList([GCNLayer(hidden) for _ in range(layers)])
        elif kind == "gat":
            self.layers = nn.ModuleList([GATLayer(hidden) for _ in range(layers)])
        else:
            self.layers = nn.ModuleList()
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden + 3, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def encode(self, x):
        magnitude = torch.log1p(x[:, :512].norm(dim=1, keepdim=True))
        return self.project(
            torch.cat([self.embedding_norm(x[:, :512]), x[:, 512:], magnitude], dim=1)
        )

    def forward(
        self, batch, self_only=False, uniform_weights=False, prior_weights=False
    ):
        """Return per-reference logits, reference weights, and candidate scores."""
        query = self.encode(batch["query"])
        nodes = self.encode(batch["x"])
        owner = batch["owner"]
        edge, weight = batch["edge"], batch["weight"]
        if self_only:
            keep = edge[0] == edge[1]
            edge, weight = edge[:, keep], weight[keep]
        for layer in self.layers:
            if isinstance(layer, GATLayer):
                nodes = layer(nodes, edge, weight, owner)
            else:
                nodes = layer(nodes, edge, weight)
        logits = self.readout(
            torch.cat([query[owner], nodes, batch["pair"]], dim=1)
        ).squeeze(1)
        if uniform_weights:
            # Equal weight over the pool. This is the no-prior control.
            weights = segment_softmax(
                torch.zeros_like(logits), owner, len(query)
            )
        elif prior_weights:
            # Formula prior only. No learned reference score is used.
            weights = segment_softmax(batch["log_prior"], owner, len(query))
        else:
            # The formula prior is a prior only; logits can still reorder polymorphs.
            weights = segment_softmax(logits + batch["log_prior"], owner, len(query))
        score = segment_sum(weights * batch["distance"], owner, len(query))
        return logits, weights, score


def reference_loss(logits, batch):
    """Mean binary cross entropy on decomposition membership."""
    target = batch["target"]
    per_reference = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if batch["has_target"].any():
        keep = batch["has_target"][batch["owner"]]
        if not keep.any():
            return logits.sum() * 0.0
        return per_reference[keep].mean()
    return logits.sum() * 0.0
