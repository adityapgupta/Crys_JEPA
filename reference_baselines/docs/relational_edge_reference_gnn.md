# Relational edge-feature reference GNN

> **Status:** **DONE — null result (corrected controls).** Second experiment in the
> inductive-bias GNN series, following the edge-attribute experiment
> ([edge_attribute_reference_gnn.md](edge_attribute_reference_gnn.md), also null).
> It tests whether element-blind SOAP geometry, co-occurrence PMI, and convex-hull
> tie-line compatibility — edge features **constructed to be independent of the node
> embedding** — improve screening over the scalar node-only baseline on the reserved
> partition.
>
> The first run (commit `cdfe23e`) returned a narrow null but had two control defects:
> the shuffled control permuted *directed* edges independently, breaking the symmetry
> of the (symmetric) features, and the documented fit-edge standardisation was not
> implemented. Both were fixed in `datasets/relational_edge_reference_graph.py` and the
> full experiment was **re-run from scratch**. The corrected result is the same — and
> cleaner: no variant reliably beats the scalar baseline, and with the symmetric shuffle
> `rich` no longer beats its own scrambled control. See [Results](#results).
>
> Scope of the claim: this is a **"no reliable improvement"** result on one fixed
> topology, not a proof of equivalence. "Independent" started as independent *by
> construction of the input*; a recovery probe ([below](#recoverability-probe)) now
> measures it, and it holds cleanly for geometry (SOAP R²=0.05), mostly for
> decomposition PMI (R²=0.18), and only partially for hull compatibility (AUC=0.71 —
> meaningfully recoverable from the embeddings). The design and success criteria below
> were written before the reserved partition was opened, though Git does not
> independently witness that ordering (commit `cdfe23e` already held both criteria and
> the first results).

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

> **Independently sourced and relational.** Each edge feature is *computed* from a
> source outside the Crys-JEPA node embeddings (independent geometry, phase-diagram
> structure, decomposition statistics), and must describe a **relationship between two
> references** that bears on decomposition / competition.

Note the scope of the claim: independence here is **by construction of the input**.
We now also measure it directly with an embedding-recoverability probe
([below](#recoverability-probe)): SOAP distance is essentially unrecoverable from the
endpoint embeddings (R²=0.05), PMI is weakly recoverable (R²=0.18 — expected, since it
reweights the same co-occurrence counts the scalar edge already carries), and hull
compatibility is partially recoverable (AUC=0.71). So "non-redundant" holds cleanly for
geometry, mostly for PMI, and only partially for the thermodynamic feature — and in
every case this is "drawn from a largely independent source," not "proven to add
screening value," which the Results section addresses separately.

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
| `shuffled` | `rich` features permuted across undirected edge pairs | are the features real signal? |
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

Reserved partition (35,336 candidates), primary budget **5%**, 3 models × 3 seeds
per variant, 1000-draw paired chemical-system bootstrap. Scalar baseline yield
**44.65%** at 5%. (Edge-reading variant means at 5%: `geom` 44.26%, `cooc` 44.49%,
`phase` 44.27%, `rich` 44.50%, `shuffled` 44.94%, `readout` 44.53%, `redundant` 44.67%.)

*On comparability to experiment 1.* The two experiments share nodes, pools, topology,
targets, checkpoint rule and ranking, but are **not identical**: this one uses batch
size 64, the edge-attribute experiment used 32, and the scalar baselines differ
accordingly (44.65% here vs 44.41% there). They are closely comparable, not a
drop-in match; treat small cross-experiment deltas with that caveat. (Scalar also moved
from 44.74% in the first, pre-fix run of *this* experiment to 44.65% here; it never
reads `edge_attr`, so that ≈0.1 pp is ordinary CPU-training nondeterminism across nodes
— a useful read on the run-to-run noise floor against which the deltas below sit.)

**No variant reliably beat the scalar baseline.** Mean yield delta vs `scalar`,
averaged over the 9 model×seed runs, with the count of runs whose bootstrap CI excludes
zero in the treatment's favour:

| Contrast | Mean Δyield | Runs favouring treatment (of 9) |
|---|---|---|
| `rich − scalar` (all three features) | −0.14 pp | 0 |
| `geom − scalar` (SOAP distance) | −0.39 pp | 0 |
| `cooc − scalar` (co-occurrence PMI) | −0.16 pp | 0 |
| `phase − scalar` (hull compatibility) | −0.38 pp | 0 |
| `rich − shuffled` (features real?) | −0.43 pp | 0 (1 run significantly *worse*) |
| `rich − readout` (propagation adds value?) | −0.03 pp | 0 |
| `readout − scalar` (control) | −0.12 pp | 0 |
| `redundant − scalar` (control) | +0.03 pp | 0 |

Every delta is within ±0.5 pp of zero and no contrast clears the success bar. The
corrected (symmetric) shuffle sharpens the "are the features real?" test: `shuffled`
now scores *highest* of all variants, so `rich − shuffled` is **−0.43 pp** (one run
significantly worse, none favouring `rich`) — the features do not beat their own
scrambled version. For context, the standing-best non-GNN baseline (`angle_gate_1deg`)
scores **45.81%** at 5% — above every GNN variant mean here, so none of these variants
beats the real screening bar regardless of the CIs.

**What this supports, and what it does not.** The analysis shows *non-significance*:
no tested variant demonstrated a reliable gain over `scalar` on this fixed topology.
It does **not** establish *equivalence* — the study reports per-run CI counts and an
average delta, but no aggregate uncertainty interval or non-inferiority / equivalence
test, so it cannot claim the features carry zero complementary information. Likewise
`readout ≈ GNN` is *consistent with* a topology/propagation bottleneck but does not
prove it. A fair summary:

> On the reserved MatterSim-proxy screen, none of the tested SOAP-, PMI-, or
> hull-compatibility edge-feature variants showed a reliable improvement over the
> scalar co-occurrence baseline on the fixed physical topology. This motivates testing
> alternative connectivity, but does not establish that the features contain no
> complementary information or that topology is definitively the bottleneck.

This remains the weaker (and defensible) form of the experiment-1 null, and it
motivates the **connectivity** arm — building the graph from phase/geometry relations —
as the next test rather than further edge features.

Artifacts (corrected-control run): `results/relational_edge_reference_gnn/reserved/{controlled_comparisons.csv,
metrics.csv, scores.csv, evaluation.json}`. The superseded pre-fix artifacts are kept
alongside in `runs_prefix/` and `reserved_prefix/`.

### Recoverability probe

The "independent by construction" claim was tested directly:
`evaluation/embedding_recoverability_probe.py` tries to predict each edge feature from
the two endpoint Crys-JEPA embeddings. Inputs are symmetric in `(i,j)`
(`[emb_i+emb_j, |emb_i−emb_j|]`, 1024-D) and the split is **reference-disjoint** — an
edge is a test edge only when *both* endpoints are held-out references, so no test
reference's embedding is seen in training. Closed-form ridge, 5 split seeds, reported on
the held-out edges. Low score ⇒ the embeddings cannot reproduce the feature ⇒ genuinely
non-redundant; high score ⇒ the feature is already latent in the embeddings.

| Edge feature | Metric | Score (mean ± std) | Reading |
|---|---|---|---|
| SOAP distance | test R² | **0.05 ± 0.06** | essentially unrecoverable — genuinely non-redundant |
| Co-occurrence PMI | test R² | **0.18 ± 0.06** | weakly recoverable — mostly independent |
| Hull compatibility | test ROC-AUC | **0.71 ± 0.03** | clearly above chance — *partially* latent in the embeddings |

(16,041 train / 1,120 held-out edges per split; `ridge_lambda=10`.) This is the honest
upgrade over the construction-only claim: geometry is independent as designed, PMI is
mostly so (its residual recoverability is expected — it reweights the same counts the
scalar edge carries), and hull compatibility is the one feature the embeddings can
already predict above chance. That ordering is itself consistent with the null: the
feature with the most to add that *isn't* already in the model (geometry) still produced
no screening gain, and the one with real overlap (thermodynamics) could not have added
much. It does **not** change the result — recoverability bounds how *new* a feature is,
not whether it helps ranking — but it removes the experiment-1 escape hatch ("the
features were just redundant") for geometry and, largely, PMI. Artifact:
`results/relational_edge_reference_gnn/recoverability_probe.json`.

## Reproducing

The shuffle and standardisation fixes change the training inputs, so the run was
regenerated from scratch in a fresh run directory (the training script refuses to
reuse a directory whose recorded source hashes changed). No precompute changes are
needed — standardisation reuses the SOAP / frequency / hull arrays already loaded. To
reproduce (on the cluster, torch required):

```bash
# from the repo root; retire any prior runs/ and reserved/ first.
sbatch --array=0-21 reference_baselines/cluster/relational_edge_train_array.slurm   # ~5 h; %8 to cap concurrency
# once all 66 runs carry complete.json, on a compute node (login node has no torch):
sbatch --export=ALL,STAGE=prepare  reference_baselines/cluster/relational_edge_reference_gnn.slurm
sbatch --export=ALL,STAGE=evaluate reference_baselines/cluster/relational_edge_reference_gnn.slurm   # opens reserved
```

Point the array at a clean `runs/` directory (move or delete the pre-fix one first).
Rough cost: ~1 h median per model×seed run (37–92 min observed); each array task runs
3 seeds (~2–5 h), so with 8 concurrent tasks the training wall-clock is ~6–10 h plus
queue and venv build, then prepare/evaluate.

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
never candidate outcomes, validation, or reserved labels. Within this experiment the
only moving part across variants is the edge information: splits, pools, nodes,
topology, targets, checkpoint rule (lowest validation reference BCE) and candidate
score are all shared. These structural components also match experiment 1, though the
two experiments are not a drop-in match (this one trains with batch size 64 vs 32).

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
