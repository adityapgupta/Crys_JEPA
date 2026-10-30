# Edge-attribute reference GNN experiment

> **Status:** **COMPLETE — null result** Design, implementation, inputs
> and pipeline are done. The full 30-run training (3 seeds × `scalar`/`rich`/
> `shuffled` × GCN/GAT/GraphSAGE, plus the `readout` control) and the single
> internal reserved evaluation ran on the SoC cluster; input provenance was
> verified by SHA-256 against the local frozen embeddings and metadata. **Headline
> finding:** the three chemistry/symmetry edge features add **no** statistically
> significant screening value over the scalar hull-co-occurrence weight, and are
> statistically indistinguishable from a randomized-feature control. See
> **Results** below. Result files are not committed (repo convention: `results/`
> is gitignored); the numbers are recorded in this document.

## Summary

The existing reference GNN connects known crystals when they co-occur in a
fit-side hull decomposition. Its edges carry only a scalar co-occurrence weight;
any chemistry or structural information otherwise comes indirectly from the
node's CrysJEPA embedding and composition. This experiment asks whether explicitly
describing the relationship between two connected references helps the GNN choose
more useful references and thereby improves downstream candidate screening.

The graph topology is intentionally unchanged. Each existing edge is augmented
with embedding similarity, symmetry agreement, and one transferable chemical
descriptor: electronegativity-profile distance. Keeping the nodes, candidate
pools, targets, splits, topology and training settings fixed turns this into a
controlled test of edge information rather than a simultaneous test of a new
graph-building rule.

## Question

On the same physical reference graph and candidate-reference pools, does explicit
symmetry and chemistry information improve reference weighting beyond the scalar
hull-co-occurrence edge weight? This experiment does **not** test a new graph
connectivity rule: fixing topology is deliberate, so any difference is attributable
to edge information rather than a different set of neighbours.

## Edge features

Every directed entry of the induced physical graph receives:

1. cosine similarity between the two frozen CrysJEPA embeddings;
2. an exact same-space-group indicator;
3. distance between composition-weighted Pauling-electronegativity profiles.

The electronegativity profile contains the weighted mean, weighted standard
deviation, and fraction of composition whose elements lack a Pauling value. These
components are standardized over the reference bank before their Euclidean
distance is taken. No formation energy, hull energy, candidate outcome or reserved
label is used to construct an edge feature.

Electronegativity is used because it summarizes how strongly a composition tends
to attract electrons and therefore provides a coarse signal about bonding
character. It is not itself a stability criterion: oxidation state, coordination,
pressure, competing phases and many other factors can override it.

## Experimental protocol

1. Use the existing known-reference nodes and physical hull-co-occurrence graph.
2. For each generated candidate, induce the same chemically eligible plus top-5
   CrysJEPA-neighbour reference pool used by the original reference GNN.
3. Train the model to identify which pool references occur in the candidate's
   saved fixed-hull decomposition. Candidate stability is not the training target.
4. Choose checkpoints only by validation reference-level binary cross-entropy.
5. After every checkpoint is frozen, compare screening yield at the fixed 5%
   unique-candidate budget on the reserved MatterSim-proxy partition.

The learned candidate score remains the weighted mean CrysJEPA distance to its
reference pool; lower scores rank earlier for screening. Three seeds are used for
each model family so a single favourable initialization is not treated as evidence.

## Controlled variants

| Variant | Topology | Edge information | What it isolates |
|---|---|---|---|
| `scalar` | Physical | Existing hull co-occurrence weight | Original GNN baseline |
| `rich` | Physical | Scalar weight plus the three new features | Incremental edge-feature value |
| `shuffled` | Physical | Cosine intact; SG and EN profiles jointly permuted across references | Whether chemistry/symmetry must align with node identity |
| `readout` | Physical | Per-node mean of the three features, no neighbour message passing | Feature availability versus graph propagation |

All variants retain the original node features, pair features, formula prior,
decomposition-membership target, frozen fit/validation/reserved splits, checkpoint
rule, seeds and candidate score. GCN, GAT and GraphSAGE are compared within the
same family; the readout control is architecture-independent.

## Results (internal reserved partition)

The experiment ran to completion on the SoC cluster (CPU-only, `long` partition):
all 30 runs reached the frozen checkpoint rule, then a single evaluation opened the
reserved partition (**35,336 unique candidates**). Yield is the MatterSim-proxy
pass rate at the fixed **5% unique-candidate budget** (1,766 selected), averaged
over the three seeds.

| Method (5% budget) | GCN | GAT | GraphSAGE |
|---|---|---|---|
| `scalar` (baseline weight)     | 0.445 | 0.440 | 0.448 |
| `rich` (+ 3 edge features)     | 0.437 | 0.442 | 0.452 |
| `shuffled` (permuted features) | 0.440 | 0.448 | 0.439 |

`readout` control (no message passing, architecture-independent): **0.442**.
Reference points at the same budget: original CrysJEPA screening `D_C` = **0.424**;
prior `angle_gate_1deg` baseline = **0.458**.

Paired bootstrap confidence intervals on the controlled contrasts (5% budget, one
interval per seed and model family — 36 comparisons in total):

- **`rich − scalar`** — all 9 CIs include zero (per-seed deltas range −0.019 to
  +0.015). No evidence the edge features improve on the scalar weight.
- **`rich − shuffled`** — 8 of 9 CIs include zero. The lone exception (GAT, seed
  2028) is `[−0.037, −0.0001]`: it *favours the shuffled control* and sits on the
  significance boundary — one result consistent with chance across 36 comparisons,
  not a signal. The real features are statistically indistinguishable from
  randomized ones.
- **`rich − readout`** and **`readout − scalar`** — all 18 CIs include zero.
  Message passing on this topology adds no measurable value over a no-propagation
  readout.

**Conclusion.** Under the fixed physical topology, the three explicit
chemistry/symmetry edge features add no demonstrated screening value over the
scalar hull-co-occurrence weight. No learned variant beats the prior
`angle_gate_1deg` baseline on average. One `rich` GraphSAGE seed reached 812
passes versus the gate's 809, but the effect did not repeat across seeds and the
controlled confidence intervals do not support it as a reliable win. This is the
"No variant beats `scalar`" row of the interpretation table below: it bounds the
*value of these features on this topology* — not whether the features contain any
chemical information — and it is exactly the result that motivates the follow-up
experiments (new connectivity rules, learned edge encoders) listed later in this
document.

## How to interpret the eventual outcomes

| Observation | Supported interpretation | What it would not prove |
|---|---|---|
| `rich` beats both `scalar` and `shuffled` consistently | Correctly aligned explicit edge information adds screening value | That every individual feature is necessary |
| `rich` beats `scalar` but not `shuffled` | Extra parameters or stochastic variation may explain the gain | Chemistry-aware edges helped |
| `rich` beats `readout` | Neighbour message passing adds value beyond exposing edge summaries | The physical topology is optimal |
| `readout` matches `rich` and beats `scalar` | Edge summaries help, but graph propagation is unnecessary | GNNs improve screening |
| Results vary strongly by seed or model family | Evidence is fragile and architecture-dependent | A robust screening improvement |
| No variant beats `scalar` | These three explicit edge features add no demonstrated value under this topology | That the features contain no chemical information |

## Assumptions and gaps

- Space-group equality is a coarse symmetry relation; crystals in different space
  groups can still be structurally similar, and crystals in one space group can
  have very different local environments.
- The electronegativity profile discards oxidation state, coordination and bond
  geometry. Missing Pauling values are tracked rather than silently set to zero,
  but this does not remove the descriptor's chemical limitations.
- The physical graph is derived from saved fit-side hull decompositions. Fixing it
  makes the feature comparison clean but does not test chemistry-based connectivity.
- The shuffled control jointly permutes space group and electronegativity profiles,
  while leaving embedding cosine similarity intact. It tests correct alignment of
  the explicit chemistry/symmetry fields, not a complete random-edge null.
- Reserved outcomes are MatterSim-derived stability proxies, not fresh DFT,
  experimental synthesis, phonon stability or finite-temperature stability.
- The reserved partition is a chemistry-disjoint internal engineering holdout
  reconstructed from the old training pool. It is shared by every variant, so it
  supports the controlled comparison here, but it is not a pristine external test
  and is not bit-identical to the team's earlier split files.
- A positive result would show improved ranking under this pipeline, not a general
  law that chemically similar materials have similar stability.

## Follow-up experiments

This edge-attribute experiment is the first in a series of **inductive-bias GNN** studies
for reference selection. Each follow-up is a controlled change on the *same* pools,
decomposition targets, frozen splits and candidate score, so any difference is
attributable to the inductive bias rather than a changed evaluation:

1. **Richer / learned edge encoders** — replace the fixed three-feature edge gate with
   a learned edge embedding (an MLP over a larger descriptor set) and test whether more
   edge information helps beyond the current three.
2. **New connectivity rules** (not only hull co-occurrence) — same-formula, subsystem,
   and cross-chemistry nearest-reference edges, each compared against the physical graph
   and matched random-edge controls.
3. **Typed / heterogeneous edges** — a distinct message function per edge relation type
   rather than one shared gate.
4. **Edge-conditioned attention** — attention that weights references by relation,
   beyond the current GAT scalar+rich bias.

## Run

From the repository root, in the environment used by the reference GNN:

```sh
python -m unittest reference_baselines.tests.test_edge_attribute_reference_gnn -v
python -m reference_baselines.training.train_edge_attribute_reference_gnn --smoke
python -m reference_baselines.evaluation.edge_attribute_reference_gnn --smoke --prepare
python -m reference_baselines.evaluation.edge_attribute_reference_gnn --smoke --evaluate
```

The smoke path uses validation rows and separate output directories. For the full
experiment, omit `--smoke`; complete every checkpoint before running evaluation:

```sh
python -m reference_baselines.training.train_edge_attribute_reference_gnn
python -m reference_baselines.evaluation.edge_attribute_reference_gnn --prepare
python -m reference_baselines.evaluation.edge_attribute_reference_gnn --evaluate
```

The last command opens the reserved MatterSim proxy outcomes only after verifying
that all checkpoints exist and match the current configuration. Primary evidence
requires both `rich_minus_scalar` and `rich_minus_shuffled` to be positive at the
fixed 5% unique screening budget, consistently across seeds and model families.
`rich_minus_readout` asks whether graph propagation adds value beyond merely
exposing feature summaries. None of these results alone establishes DFT stability,
synthesis feasibility, or that chemistry-aware connectivity would be better.

## Cluster transfer

The transfer helper follows the current SoC workflow: it makes one small source
archive, uploads it using `scp -J` through the NUS jump host, and extracts it in
the existing remote checkout. From a Bash shell at the local repository root,
first preview and then upload only the new source files:

```sh
DRY_RUN=1 bash reference_baselines/cluster/sync_edge_attribute_to_cluster.sh
bash reference_baselines/cluster/sync_edge_attribute_to_cluster.sh
```

The defaults are `userid@xlogin.comp.nus.edu.sg`, the `~/Crys_JEPA` checkout,
and `userid@stujump.comp.nus.edu.sg`. Override `REMOTE_USER_HOST`,
`REMOTE_ROOT`, or `SSH_JUMP` if the checkout moved. This helper transfers source
only; it does not try to place the large input arrays in the small home quota.

On the cluster, the first submitted job should be tests only:

```sh
cd ~/Crys_JEPA
sbatch --partition=normal --time=01:00:00 \
  --export=ALL,STAGE=tests \
  reference_baselines/cluster/edge_attribute_reference_gnn.slurm
```

The Slurm wrapper uses `/usr/bin/python3.12`, creates its venv and pip temporary
directory under `/tmp/$USER`, and installs the CPU-only PyTorch wheel. This GNN
uses small per-candidate graphs and does not require a GPU or `torch_geometric`.

After later runs, copy the complete edge-attribute result directory and matching logs
back without deleting either side:

```sh
bash reference_baselines/cluster/sync_edge_attribute_from_cluster.sh
```
