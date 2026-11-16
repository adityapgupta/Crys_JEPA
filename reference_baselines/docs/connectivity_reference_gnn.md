# Connectivity reference GNN

> **Status:** **COMPLETE — null result.** Third experiment in the
> inductive-bias GNN series, following the edge-attribute and relational-edge
> experiments (both null). All 39 runs (4 topologies × GCN/GAT/GraphSAGE × 3 seeds,
> plus the shared no-graph readout × 3 seeds) trained to the frozen checkpoint rule,
> and the reserved partition (35,336 candidates) was opened once after the
> checkpoints were frozen. **Headline finding:** no topology makes the GNN beat the
> no-graph `readout` control, and no topology beats the current `physical` graph —
> graph wiring adds no measurable screening value. See **Results** below.

## Why this experiment (the lesson from experiments 1 and 2)

Experiments 1 and 2 both varied the **edge information** on a fixed graph and found
no screening gain. The decisive clue was the same in both: the no-message-passing
`readout` control **matched** the full GNNs (`rich − readout ≈ 0`). That says the
problem is not what the edges carry — it is that **message passing on this graph
adds nothing over using the node features directly.**

Both experiments fixed the *same* topology (decomposition co-occurrence) and changed
features or architecture. The one lever never varied is **connectivity: which
references are connected at all.** A graph adds value only if its wiring encodes a
relationship *not already recoverable from the node features*; if propagation over a
given wiring never beats `readout`, that wiring carries no usable inductive bias.
This experiment varies the wiring and keeps everything else fixed.

## The embedding-space question (explicit)

A natural objection: *"this is a Crys-JEPA project — shouldn't the graph be built in
the embedding space?"* The embeddings **do** stay central: they remain the node
features and the screening score (mean squared embedding distance to the pool). But
a graph built *from* the embeddings (nearest neighbours in embedding space) is a
function of the node features, so message passing over it can only reshuffle
information the nodes already hold — the `readout ≈ GNN` regime, by construction. An
embedding-kNN graph is therefore the **most-redundant** wiring, not the most
promising one. We include it as a named control (`cosine`) to test that prediction
directly, alongside wirings built from information **complementary** to the
embeddings.

## Question

On the same nodes, candidate-reference pools, training target, and screening score,
does **changing the graph topology** make message passing beat the no-propagation
`readout` control — and does any topology beat the current co-occurrence graph at a
fixed screening budget? As before, the GNN predicts reference
decomposition-membership (never candidate stability), so there is no label leakage.

## Topologies

Each topology defines the reference→reference edges. Nodes (Crys-JEPA embeddings),
pools, target, checkpoint rule, and screening score are identical across all of
them; a single scalar edge weight is used (no edge features — experiments 1–2 showed
they do not matter, so the only moving part here is *who connects to whom*).

| Topology | Edges connect two references when… | Edge weight | Built from | Role |
|---|---|---|---|---|
| `physical` (baseline) | they co-occur in a fit-side hull decomposition | `log1p(count)` | decomposition data | the current graph; reference point |
| `cosine` | each reference ↔ its top-k embedding-nearest references | cosine similarity | **the embeddings** | redundancy control — the "embedding-space graph" |
| `soap` | each reference ↔ its top-k SOAP-geometry-nearest references | `1 − SOAP distance` | atomic structure (independent of JEPA) | complementary wiring — best shot at a gain |
| `phase` | the two references are convex-hull tie-line adjacent | 1 | reference formation energies | complementary wiring — thermodynamic competitors |

The floor is the `no_graph` **readout** control (no edges at all, no message
passing) — the same pre-registered control used in experiments 1–2 — so a separate
empty-graph topology is not needed.

`soap` and `phase` reuse artifacts already built for experiment 2
(`soap_density_normalized.npy`, `hull_compatibility.npz`); the adjacency builder is
the one new precompute. `cosine` reuses the frozen embeddings and is already built
by `graphs/build_reference_graph.py`. `soap` and `phase` are registered in
`datasets/reference_graph.py` `GRAPH_NAMES`, so the base dataset, trainer, and
evaluation handle them with no further changes.

## Controls and the decisive comparison

For every topology: GCN / GAT / GraphSAGE × 3 seeds, with a single scalar edge
weight. The shared, topology-independent `readout` control (node features only, no
message passing) × 3 seeds is the yardstick.

The **decisive, pre-registered comparison is `GNN − readout` on each topology.** A
topology carries usable structure only if its GNN beats `readout`. Secondary
comparison: each topology's GNN vs the `physical` GNN and vs the standing-best
non-GNN baseline (`angle_gate_1deg`).

Run count: 4 topologies × 3 models × 3 seeds = 36 GNN runs, plus the `no_graph`
readout ×3 seeds = **39 runs** across 13 `(graph, model)` specs — one Slurm array,
same shape as experiment 2. The readout is trained once (under `physical`, since it
ignores edges) and reused as the floor for every topology.

## Evaluation

Reuse the frozen-checkpoint harness unchanged. After all checkpoints are frozen,
open the reserved partition once (35,336 candidates). Metric: MatterSim proxy pass
rate (≤ 0.1 eV/atom above the fixed MP2023 hull) at budgets 1/2/5/7%, primary **5%**.
Paired chemical-system bootstrap confidence intervals for:

- `GNN − readout` on each topology (does this wiring make propagation pay off?);
- each topology's GNN − `physical` GNN (does any wiring beat the current graph?);
- each topology's GNN vs `angle_gate_1deg` (does it clear the simple-rule bar?).

**Pre-registered success criteria.** A topology *helps* if, at 5%, its `GNN − readout`
gain has a bootstrap CI excluding zero **and** it beats the `physical` GNN. If **no**
topology makes the GNN beat `readout`, the conclusion is the strong, general null:
**graph structure of any tested wiring adds nothing to Crys-JEPA screening — the
node embeddings are already saturated for this task.** That outcome, combined with
the two edge-feature nulls, is the clean negative result for the inductive-bias
analysis and the point at which we stop adding structure.

## Results (reserved partition)

All 39 runs reached the frozen checkpoint rule; the reserved partition
(35,336 candidates) was then opened once. Yield is the MatterSim-proxy pass rate at
the fixed 5% unique-candidate budget (1,766 selected), averaged over the three seeds
and the three GNN architectures for each topology.

| Method (5% budget) | Mean passes | Yield | `GNN − readout` (mean, significant of 9) |
|---|---:|---:|---|
| `readout` (no graph) | 782 | 44.3% | — (floor) |
| `physical` (co-occurrence) | 784 | 44.4% | +0.13 pts, 0 / 9 |
| `cosine` (embedding kNN) | 786 | 44.5% | +0.23 pts, 0 / 9 |
| `soap` (geometry kNN) | 790 | 44.7% | +0.46 pts, 0 / 9 |
| `phase` (hull tie-lines) | 786 | 44.5% | +0.21 pts, 0 / 9 |

Reference points at the same budget: `angle_gate_1deg` = 45.8% (809 passes);
original Crys-JEPA screening = 42.4% (749 passes).

Paired chemical-system bootstrap confidence intervals (5% budget, one interval per
seed and model family):

- **`GNN − readout`** — for every topology, all nine intervals include zero. No
  wiring makes message passing beat the no-graph readout control.
- **`GNN − physical`** — for every topology, all nine intervals include zero. No
  wiring beats the current co-occurrence graph.

**Conclusion.** This is the pre-registered strong null. Changing the graph wiring
does not make message passing pay off: the no-graph readout matches every topology,
and none clears the `angle_gate_1deg` bar. The geometry wiring (`soap`) is the
strongest of the four and the only one whose mean yield exceeds the readout by more
than a few tenths of a point — consistent with the hypothesis that geometry is the
most complementary signal — but the gain is within noise (0 of 9 intervals exclude
zero). The energy-derived `phase` wiring adds nothing, so the conclusion rests on
the energy-free topologies. The embedding `cosine` wiring also does not beat readout,
as predicted for a graph built from the node features themselves.

Together with the two edge-feature experiments, the inductive-bias series is now
complete: neither richer edge information nor alternative graph wiring improves
Crys-JEPA reference screening. The frozen embeddings already carry what the task
needs, and graph propagation over any tested structure adds no measurable value.

## Prerequisites and leakage guards

- **Embeddings:** frozen Crys-JEPA reference embeddings (already present).
- **`physical` / `cosine` graphs:** already built by `graphs/build_reference_graph.py`
  and reused as-is (`reference_gnn/graphs/{physical,cosine}.npz`).
- **`soap` adjacency:** reference→reference top-k kNN from `soap_density_normalized.npy`
  (new precompute, `graphs/connectivity_adjacency.py`; reuses experiment-2 SOAP).
- **`phase` adjacency:** reuse `relational_edge/hull_compatibility.npz` directly as
  the edge set (already built and verified: 34,878 tie-line-adjacent pairs),
  self-loops added — also produced by `graphs/connectivity_adjacency.py`.
- **k:** fix `k = 10` for the SOAP kNN graph in the main run, matching the prebuilt
  `cosine` graph's k (a small k-sweep is a follow-up, not part of the pre-registered
  test).

All adjacencies use known-crystal properties and fit-side data only — never
candidate outcomes, validation, or reserved labels. Splits, pools, target,
checkpoint rule, and screening score are identical to experiments 1–2, so the only
moving part is the graph topology.

## Assumptions and limits

- MatterSim is a proxy screen, not DFT confirmation.
- Only node-local edge weights and these four wirings are tested; a negative result
  bounds *these* topologies, not every conceivable graph.
- kNN graphs depend on `k`; the main run fixes one value.
- A positive result shows improved ranking under this pipeline — not DFT stability
  or synthesizability.

## Implementation plan (reuse the base reference GNN)

Maximum reuse — the base reference GNN is already parameterised by graph name and
ships a `no_graph` readout, so there is **no new model or training code**:

- `graphs/connectivity_adjacency.py` — build the `soap` kNN adjacency; wrap the
  existing `hull_compatibility.npz` as the `phase` adjacency (`cosine`/`physical`
  are reused from `build_reference_graph.py`).
- `datasets/reference_graph.py` — register `soap`/`phase` in `GRAPH_NAMES` (the only
  base-file change; additive).
- **Training reuses `training/train_reference_gnn.py`** unchanged — the cluster array
  calls it per `(graph, model)` with the connectivity config and run-dir.
- **Models reuse `models/reference_gnn.py`** unchanged (`gcn/gat/sage/no_graph`).
- `evaluation/connectivity_reference_gnn.py` — `--prepare` / `--evaluate`, the
  decisive `GNN − readout` and `topology − physical` contrasts (reuses the base eval
  helpers).
- `configs/connectivity_reference_gnn.toml`, tests, and cluster/Slurm-array scripts
  mirroring experiment 2.

## Relation to the series

This is the **connectivity** arm named in the relational-edge doc. Experiments 1–2
asked "do better edge features help?" (no). This asks "does a better graph wiring
help?" Together they separate *edge information* from *graph structure*. If this is
also null, the series conclusion is that graph-based inductive bias does not improve
Crys-JEPA screening because the embeddings already capture what the task needs.
