"""Matched GCN/MPNN encoders with the same candidate-conditioned readout."""

import torch
from torch import nn

from reference_baselines.models.reference import segment_sum, segment_softmax


class MessageLayer(nn.Module):
    def __init__(self, kind, hidden):
        super().__init__()
        self.kind = kind
        if kind == "gcn":
            self.message = nn.Linear(hidden, hidden)
        elif kind == "mpnn":
            self.message = nn.Sequential(
                nn.Linear(2 * hidden + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
            )
        else:
            raise ValueError(kind)
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, edge, weight):
        source, target = edge
        degree = segment_sum(weight, target, len(x)).clamp_min(1e-12)
        if self.kind == "gcn":
            scale = weight / (degree[source] * degree[target]).sqrt()
            message = self.message(x[source]) * scale[:, None]
        else:
            features = torch.stack([weight, (source == target).to(weight.dtype)], dim=1)
            message = self.message(torch.cat([x[source], x[target], features], dim=1))
            message = message * (weight / degree[target])[:, None]
        aggregate = segment_sum(message, target, len(x))
        return self.norm(
            x + torch.nn.functional.silu(self.update(torch.cat([x, aggregate], dim=1)))
        )


class ReferenceGNN(nn.Module):
    def __init__(self, kind, hidden=64, layers=2):
        super().__init__()
        if kind not in ["gcn", "mpnn", "no_graph"]:
            raise ValueError(kind)
        self.kind = kind
        self.embedding_norm = nn.LayerNorm(512)
        self.project = nn.Sequential(nn.Linear(613, hidden), nn.SiLU())
        self.layers = nn.ModuleList(
            [MessageLayer(kind, hidden) for _ in range(layers)]
            if kind != "no_graph"
            else []
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden + 3, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def encode(self, x):
        magnitude = torch.log1p(x[:, :512].norm(dim=1, keepdim=True))
        return self.project(
            torch.cat([self.embedding_norm(x[:, :512]), x[:, 512:], magnitude], dim=1)
        )

    def forward(self, batch, self_only=False, uniform_readout=False):
        query, x = self.encode(batch["query"]), self.encode(batch["x"])
        edge, weight = batch["edge"], batch["weight"]
        if self_only:
            keep = edge[0] == edge[1]
            edge, weight = edge[:, keep], weight[keep]
        for layer in self.layers:
            x = layer(x, edge, weight)
        owner = batch["batch"]
        logits = self.readout(
            torch.cat([query[owner], x, batch["pair"]], dim=1)
        ).squeeze(1)
        if uniform_readout:
            logits = torch.zeros_like(logits)
        # Formula-balanced prior; learned weights can distinguish its polymorphs.
        weights = segment_softmax(logits + batch["log_prior"], owner, len(query))
        context = segment_sum(x * weights[:, None], owner, len(query))
        return self.head(torch.cat([query, context], dim=1)).squeeze(1), weights
