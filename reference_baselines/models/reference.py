"""Small reference-selection models in plain PyTorch, with a shared attention readout."""

import math

from torch import nn
import torch


MODEL_NAMES = ['candidate_only', 'reference_attention', 'gcn', 'graphsage', 'mpnn', 'graph_transformer']


def segment_sum(x, group, size):
    result = x.new_zeros((size,) + x.shape[1:])
    return result.index_add(0, group, x)


def segment_softmax(value, group, size):
    maximum = value.new_full((size,), -torch.inf)
    maximum = maximum.scatter_reduce(0, group, value, reduce='amax', include_self=True)
    exponent = (value - maximum[group]).exp()
    denominator = segment_sum(exponent, group, size)
    return exponent / denominator[group].clamp_min(1e-12)


class GraphLayer(nn.Module):
    def __init__(self, hidden, kind):
        super().__init__()
        self.kind = kind
        self.message = nn.Linear(hidden, hidden)
        self.update = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        if kind == 'mpnn':
            self.edge_mlp = nn.Sequential(nn.Linear(hidden * 2 + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        if kind == 'graph_transformer':
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.k = nn.Linear(hidden, hidden, bias=False)
            self.edge_bias = nn.Linear(2, 1)

    def forward(self, x, edge, weight):
        source, target = edge
        degree = segment_sum(weight, target, len(x)).clamp_min(1e-12)
        edge_feature = torch.stack([weight, (source == target).to(weight.dtype)], dim=1)
        if self.kind == 'gcn':
            scale = weight / (degree[source] * degree[target]).sqrt()
            aggregate = segment_sum(self.message(x[source]) * scale[:, None], target, len(x))
        elif self.kind == 'graphsage':
            # Self contribution is a separate branch; exclude loops from neighbour mean.
            nonself = source != target
            sw = weight * nonself
            denominator = segment_sum(sw, target, len(x)).clamp_min(1e-12)
            aggregate = segment_sum(self.message(x[source]) * sw[:, None], target, len(x)) / denominator[:, None]
        elif self.kind == 'mpnn':
            message = self.edge_mlp(torch.cat([x[source], x[target], edge_feature], dim=1))
            aggregate = segment_sum(message * weight[:, None], target, len(x)) / degree[:, None]
        elif self.kind == 'graph_transformer':
            score = (self.q(x[target]) * self.k(x[source])).sum(1) / math.sqrt(x.shape[1])
            score = score + self.edge_bias(edge_feature).squeeze(1)
            attention = segment_softmax(score, target, len(x))
            aggregate = segment_sum(self.message(x[source]) * attention[:, None], target, len(x))
        else:
            raise ValueError(self.kind)
        return self.norm(x + torch.nn.functional.silu(self.update(torch.cat([x, aggregate], dim=1))))


class ReferenceModel(nn.Module):
    def __init__(self, kind, hidden=128, layers=2):
        super().__init__()
        if kind not in MODEL_NAMES:
            raise ValueError(kind)
        self.kind = kind
        self.embedding_norm = nn.LayerNorm(512)
        self.project = nn.Sequential(nn.Linear(613, hidden), nn.SiLU())
        self.layers = nn.ModuleList([GraphLayer(hidden, kind) for _ in range(layers)]
                                   if kind in ['gcn', 'graphsage', 'mpnn', 'graph_transformer'] else [])
        if kind != 'candidate_only':
            self.attention = nn.Sequential(nn.Linear(hidden * 2 + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        width = hidden if kind == 'candidate_only' else hidden * 2
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def encode(self, x):
        norm = torch.log1p(x[:, :512].norm(dim=1, keepdim=True))
        return self.project(torch.cat([self.embedding_norm(x[:, :512]), x[:, 512:], norm], dim=1))

    def forward(self, batch, uniform_readout=False):
        query = self.encode(batch['query'])
        if self.kind == 'candidate_only':
            return self.head(query).squeeze(1), query.new_empty(0)
        x = self.encode(batch['x'])
        for layer in self.layers:
            x = layer(x, batch['edge'], batch['weight'])
        owner = batch['batch']
        score = self.attention(torch.cat([query[owner], x, batch['pair']], dim=1)).squeeze(1)
        attention = segment_softmax(torch.zeros_like(score) if uniform_readout else score, owner, len(query))
        context = segment_sum(x * attention[:, None], owner, len(query))
        return self.head(torch.cat([query, context], dim=1)).squeeze(1), attention
