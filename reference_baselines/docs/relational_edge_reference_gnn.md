# Relational edge-feature reference GNN

> **Status:** **DONE — null result (owned by Aditi).** Second experiment in the
> inductive-bias GNN series, following the edge-attribute experiment
> ([edge_attribute_reference_gnn.md](edge_attribute_reference_gnn.md), also null).
> Non-redundant relational edge features (element-blind SOAP geometry, co-occurrence
> PMI, convex-hull tie-line compatibility) add **no screening gain** over the scalar
> node-only baseline on the reserved partition; the pre-registered null branch holds.
> See [Results](#results). The design, controls and pre-registered criteria below
> were fixed before the reserved partition was opened.

## Why this experiment (the lesson from experiment 1)

The edge-attribute experiment added three edge features — embedding cosine
similarity, a same-space-group flag, and electronegativity-profile distance — and
found **no screening gain** over a scalar co-occurrence weight, indistinguishable
from a randomized-feature control. The diagnosis: **all three features were
largely re-derivable from the node features the model already had.**

- *Embedding cosine* is computed directly from the node JEPA embeddings — fully redundant.
- *Same space group* is partly encoded in the embeddings already (our symmetry
  analysis: linear-probe space group 34.0% vs 14.3% chance, community AMI 0.148) —
  so the flag mostly duplicates information in the node vectors.
- *Electronegativity profile* is a function of composition, which the embeddings
  encode strongly (oxygen-presence probe 95.1%) — largely redundant too.

Feeding a GNN edge features that are functions of its own node features adds little
information to pass along. This experiment fixes that with a single design
principle for edge features:

> **Non-redundant and relational.** Each edge feature must draw on information that
> is **not present in the Crys-JEPA node embeddings** (independent geometry,
> phase-diagram structure, decomposition statistics), and must describe a
> **relationship between two references** that bears on decomposition / competition.

A second lesson constrains the conclusion we can draw: in experiment 1 the
no-message-passing `readout` control matched the full GNN. So edge features alone
may not be the bottleneck — the message-passing-on-this-topology step added
little. This experiment keeps the topology fixed (a clean test of *edge
information*) but includes the `readout` control as a diagnostic, and names the
complementary *new-connectivity* experiment as the other arm (out of scope here).

## Question

On the same physical reference graph and candidate-reference pools, do relational
edge features drawn from information **outside** the Crys-JEPA embedding improve
MatterSim screening yield at a fixed budget over the scalar-weight baseline — and
do they make message passing add measurable value over a no-propagation readout?

As before, the GNN predicts **reference decomposition-membership** (which known
references appear in a candidate's fixed-hull decomposition), never candidate
stability, so there is no label leakage. The learned per-reference weights feed the
screening score (weighted mean squared embedding distance; lower ranks first).

## Edge features

Every directed edge `(i, j)` of the fixed physical graph receives three features.
Each is justified as non-redundant with the node embeddings/composition.

1. **SOAP geometric distance.** `d_SOAP(i,j) = 1 − ⟨s_i, s_j⟩`, where `s` is the
   unit-normalized, element-blind, density-normalized SOAP fingerprint from
   `structure/descriptors.py` (`soap_density_normalized.npy`). SOAP captures local
   atomic geometry independently of the energy-aware JEPA embedding — exactly the
   axis the symmetry analysis showed the embedding is weakest on. *Not a function
   of the node vectors.*

2. **Decomposition co-occurrence PMI.** From fit-side decomposition co-occurrence
   counts (`graphs/cooccurrence.py`): `pmi(i,j) = log( (N · c_ij) / (c_i · c_j) )`,
   clipped to a sensible range, where `c_ij` is the count of candidates whose
   decomposition contains both references, `c_i`/`c_j` the per-reference totals, `N`
   the normaliser. This refines the current `log1p(c_ij)` edge weight by controlling
   for each reference's base popularity — "do these two compete together *more than
   their individual frequency predicts*." Relational; from decomposition data only.

3. **Hull phase-compatibility.** For the smallest chemical (sub)system containing
   both references, build the known-crystal convex hull from reference formation
   energies (`core/thermodynamics.py`, `reference_set_comparison/hulls.py`). Feature
   = 1 if the two references share a stable tie-line / facet (they can coexist as a
   two-phase decomposition region), else 0; optionally graded by the composition
   span of their shared facet. Encodes competing-phase structure directly — the
   thing the membership target is really about — and is not in the embeddings.

**Deliberately excluded** (and kept only as a redundancy re-check): embedding cosine
and same-space-group. These were shown redundant in experiment 1; `redundant` below
re-confirms they add nothing.

## Modelling: a learned edge encoder

Standardise the 3-D edge-feature vector `e_ij` over the fit edges. A small MLP edge
encoder `φ` maps it to a per-message modulation:

```
g_ij = 2 · sigmoid( φ(e_ij) )          # φ: Linear(3→h) → SiLU → Linear(h→1)
```

- **GCN / GraphSAGE:** multiply the edge weight by `g_ij` before aggregation
  (same gating mechanism as experiment 1, so the comparison is clean).
- **GAT:** add `φ(e_ij)` as an additive bias to the attention logit, alongside the
  existing scalar bias.

This is the "learned edge encoder" follow-up named in experiment 1: features are
learned-combined rather than hand-weighted. Topology, nodes, pools, targets and the
candidate score are all unchanged.

## Controlled variants

Each variant runs GCN / GAT / GraphSAGE × 3 seeds, except `readout` (architecture-independent).

| Variant | Edge information | What it isolates |
|---|---|---|
| `scalar` | co-occurrence weight only | baseline (experiment-1 scalar) |
| `geom` | + SOAP distance | value of independent geometry |
| `cooc` | + co-occurrence PMI | value of popularity-normalised competition |
| `phase` | + hull compatibility | value of phase-diagram competition |
| `rich` | + all three | combined non-redundant edge value |
| `shuffled` | `rich` features permuted across edges | are the features real signal? |
| `readout` | all three at readout, no message passing | does propagation add value? |
| `redundant` | + embedding cosine only | re-confirms redundant features add nothing |

Per-feature variants (`geom`/`cooc`/`phase`) show *which* relation carries signal,
not just the all-in bundle.

## Evaluation

Reuse the frozen-checkpoint harness. After every checkpoint is frozen, open the
reserved partition once (35,336 candidates). Metric: MatterSim pass rate (energy
≤ 0.1 eV/atom above the fixed MP2023 hull) at budgets 1/2/5/7%, primary **5%**.
Paired chemical-system bootstrap confidence intervals for:

- each variant − `scalar` (does it beat baseline?);
- `rich` − `shuffled` (are the features real?);
- `rich` − `readout` and `readout` − `scalar` (does message passing add value?);
- each variant vs the **standing-best non-GNN baseline** (`angle_gate_1deg` /
  `intersection_fallback_k5`) — the real bar for "better screening" is beating the
  best simple rule, not just the mean.

**Pre-registered success criteria.** A variant *helps* if, at 5%, its gain over
`scalar` has a bootstrap CI excluding zero **and** it is at least as high as the
best non-GNN baseline **and** it beats `shuffled`. If no variant beats `scalar` and
`readout ≈` the GNNs again, the conclusion is that the bottleneck is the
topology / message-passing step, not the edge features — which motivates the
new-connectivity experiment rather than more features.

## Results

Reserved partition (≈35.3k candidates), primary budget **5%**, 3 models × 3 seeds
per variant, 1000-draw paired chemical-system bootstrap. Scalar baseline yield
≈ **44.4%** at 5% (in line with the edge-attribute experiment).

**No variant beats the scalar baseline.** Mean yield delta vs `scalar`, averaged
over the 9 model×seed runs, with the count of runs whose bootstrap CI excludes zero
in the treatment's favour:

| Contrast | Mean Δyield | Runs favouring treatment (of 9) |
|---|---|---|
| `rich − scalar` (all three features) | −0.24 pp | 0 |
| `geom − scalar` (SOAP distance) | −0.69 pp | 0 (1 run significantly *worse*) |
| `cooc − scalar` (co-occurrence PMI) | −0.32 pp | 0 |
| `phase − scalar` (hull compatibility) | −0.03 pp | 0 |
| `rich − shuffled` (features real?) | +0.40 pp | 1 |
| `rich − readout` (propagation adds value?) | −0.16 pp | 0 |
| `readout − scalar` (control) | −0.08 pp | 0 |
| `redundant − scalar` (control) | −0.07 pp | 0 |

Every delta is within ±0.7 pp of zero, and no contrast clears the pre-registered
bar. `rich` does not reliably beat its own `shuffled` control (1/9), and `readout`
≈ the GNNs (`rich − readout` ≈ 0), repeating experiment 1.

**Conclusion (the pre-registered null branch).** Even deliberately non-redundant,
relational edge features — element-blind SOAP geometry (independent of the JEPA
embedding), popularity-normalised decomposition PMI, and convex-hull tie-line
compatibility — carry **no screening signal beyond the Crys-JEPA node embeddings**.
Combined with `readout ≈ GNN`, the bottleneck is not the edge *information* but the
fixed topology / message-passing step. This is the stronger form of the experiment-1
null (it also removes the redundancy objection), and it motivates the
**connectivity** arm — building the graph from phase/geometry relations — rather
than further edge features.

Artifacts: `results/relational_edge_reference_gnn/reserved/{controlled_comparisons.csv,
metrics.csv, scores.csv, evaluation.json}`.

## Prerequisites and leakage guards

- **SOAP:** `python -m reference_baselines.structure.descriptors` → `soap_density_normalized.npy` (needs `dscribe` and `ase`).
- **Hull energies (must be supplied):** `results/reference_hull_energies.csv`
  with columns `reference_row, formula, energy_per_atom`, aligned to the 27,136
  references, from the project's MP2023 reference energies. The precompute does
  **not** generate this file.
- **Frequency + hull compatibility:** `python -m reference_baselines.graphs.relational_edge_features`.
  Hull compatibility is built **per physical edge from the exact union chemical
  system** of the pair (not inside an unrelated larger system). Compatible = the
  midpoint of the pair decomposes into exactly those two phases (a genuine two-phase
  tie-line); this is equivalent to "both stable and sharing a hull facet" (every
  simplex edge is a tie-line), and immiscible pairs are correctly compatible. 34,878
  physical edges are compatible; spot-checked and verified.
- **Cluster env:** the Slurm script installs torch/numpy/pandas/scipy/pymatgen and
  `dscribe`/`ase`.

All edge features use **known-crystal properties and fit-side decompositions only** —
never candidate outcomes, validation, or reserved labels. Splits, pools, targets,
checkpoint rule (lowest validation reference BCE) and candidate score are identical
to experiment 1, so the only moving part is the edge information.

## Assumptions and limits

- MatterSim is a proxy screen, not DFT confirmation.
- Topology is fixed on purpose; chemistry/geometry-based *connectivity* is a
  separate experiment, not tested here.
- SOAP averages local environments and is not a unique crystal representation;
  hull compatibility uses known-crystal energies only.
- SOAP and hull precomputes add cost; they run once.
- A positive result shows improved ranking under this pipeline — not DFT stability,
  synthesizability, or that geometry/phase relations are generally decisive.

## Implementation plan (mirror the edge-attribute files)

- `datasets/relational_edge_reference_graph.py` — compute the three features, wrap
  `PoolDataset`, precompute the frozen edge-feature table.
- `models/relational_edge_reference_gnn.py` — edge-encoder GCN/GAT/SAGE layers,
  readout control, `build_model(variant, kind)`.
- `training/train_relational_edge_reference_gnn.py` — training loop (reuse
  `reference_loss`, `reference_metrics`, `seed_everything`).
- `evaluation/relational_edge_reference_gnn.py` — `--prepare` / `--evaluate`,
  metrics + controlled-comparison tables.
- `configs/relational_edge_reference_gnn.toml` — models, variants, seeds, budgets.
- `tests/test_relational_edge_reference_gnn.py` and cluster scripts.

## Relation to the series

This is the **edge-information** arm: do non-redundant relational features help on a
fixed graph? The complementary **connectivity** arm — building the graph from phase
or geometry relationships rather than only hull co-occurrence — is the next
experiment. Run together, they separate "better edge features" from "better graph."
