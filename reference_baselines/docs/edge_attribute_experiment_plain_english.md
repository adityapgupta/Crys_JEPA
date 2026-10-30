# Did chemistry-aware GNN edges improve screening?

**Experiment status:** complete

**Result:** no reliable improvement was demonstrated

## The short version

We tested whether a graph neural network (GNN) could choose better reference
crystals if every graph connection explicitly described both crystals' embedding
similarity, symmetry, and chemistry. It could not: across three model types and
three random seeds, the real edge attributes performed about the same as the old
scalar edge weight, randomized attributes, and a control that did no graph message
passing. The experiment is still useful because it narrows the claim precisely:
**these three attributes did not add screening value on this fixed graph; it does
not show that symmetry or chemistry is irrelevant.**

## What we were trying to improve

Crys-JEPA screens a large set of generated crystals by comparing each candidate
with known reference crystals. The existing "physical" reference graph connects
two known crystals when they appear together in a fit-side phase-decomposition
solution. The connection has one number: how often that pair co-occurs, transformed
with `log(1 + count)`.

That graph therefore records a real relationship from the reference data, but it
does not explicitly tell the GNN *how* the two connected crystals are similar. We
asked whether adding that information would help the model assign better weights
to references and, in turn, rank more near-stable candidates near the top of the
screening list.

## What we changed

We kept the graph's nodes and connections fixed and added three descriptions to
every existing edge:

1. **Embedding similarity:** cosine similarity between the two frozen Crys-JEPA
   embeddings.
2. **Symmetry agreement:** whether the two crystals have exactly the same space
   group.
3. **Chemical similarity:** the distance between simple composition-weighted
   electronegativity profiles.

The chemical profile summarizes the mean and spread of Pauling electronegativity
in each composition and records how much of the composition has no Pauling value.
It is a coarse bonding descriptor, not a rule for stability.

Everything else was held constant: reference crystals, candidate pools, graph
connections, data partitions, training target, checkpoint selection, and final
screening score. This makes the experiment a clean test of **edge information**,
not a test of a new way to connect the graph.

## The four controls

| Version | Information available | Question answered |
|---|---|---|
| `scalar` | Original co-occurrence weight only | How well does the existing edge work? |
| `rich` | Original weight plus the three real attributes | Do the new attributes help? |
| `shuffled` | Embedding similarity kept; symmetry and chemistry assigned to the wrong reference identities | Does correct chemistry/symmetry alignment matter? |
| `readout` | Edge summaries are visible, but references do not exchange messages | Is graph propagation itself useful? |

We ran GCN, GAT, and GraphSAGE models with seeds 2026, 2027, and 2028. In total,
there were 30 frozen training runs: 27 GNN runs and three readout controls.

The model was trained to predict whether a known reference appears in a
candidate's saved hull decomposition. Candidate stability was **not** the training
target. After checkpoint selection using validation reference loss, the frozen
models ranked a chemistry-disjoint reserved partition. A candidate counted as a
screening pass when its saved MatterSim proxy hull distance was at most
0.1 eV/atom, using the same unique-candidate budget for every method.

## Result

The reserved partition contained 35,336 unique candidates. At the fixed 5% budget,
each method could select 1,766 candidates.

| Method | Mean passing candidates | Mean yield | Range across runs |
|---|---:|---:|---:|
| Original Crys-JEPA score | 749 | 42.4% | fixed method |
| `scalar` GNN | 784.2 | 44.4% | 769–802 |
| `rich` GNN | 783.6 | 44.4% | 767–812 |
| `shuffled` GNN | 781.2 | 44.2% | 771–799 |
| `readout` without message passing | 780.7 | 44.2% | 762–792 |
| Fixed 1-degree angle gate | 809 | 45.8% | fixed method |

The averages are the important comparison. One `rich` GraphSAGE run reached 812
passes, slightly above the fixed angle gate's 809, but this did not repeat across
seeds and is not evidence of a dependable improvement.

The paired uncertainty checks lead to the same conclusion:

- All nine `rich` versus `scalar` confidence intervals include zero.
- All nine `rich` versus `readout` confidence intervals include zero.
- Eight of nine `rich` versus `shuffled` intervals include zero. The only
  exception favours the shuffled control, not the real attributes, and is one
  uncorrected result among 36 comparisons.
- No model family shows a consistent improvement in validation loss from the
  real edge attributes.

## What the experiment shows

- Adding these particular embedding, space-group, and electronegativity edge
  attributes did not reliably improve screening on the existing physical graph.
- Correctly aligned symmetry and chemistry did not outperform the randomized
  control.
- Making the attributes available without message passing performed about as well
  as propagating them through the graph.
- The simple fixed angle gate remains stronger on average than the learned
  variants in this experiment.

## What it does not show

- It does **not** show that Crys-JEPA embeddings lack structural information.
- It does **not** show that space group or chemistry is unrelated to stability.
- It does **not** test a graph whose *connections* are built from chemistry or
  symmetry; the existing topology was deliberately frozen.
- It does **not** isolate the value of each new attribute individually.
- It does **not** establish DFT stability, synthesizability, phonon stability, or
  finite-temperature stability. The outcome labels are MatterSim-derived proxies.

## Important assumptions and gaps

- Exact space-group equality is a coarse comparison. The same space group can
  contain very different local structures, and related structures can have
  different space groups.
- The electronegativity profile omits oxidation state, coordination, bonding
  geometry, pressure, and competing phases.
- The shuffled control randomizes the explicit symmetry and electronegativity
  fields but keeps embedding cosine similarity intact. It is not a completely
  random-edge control.
- The reserved partition is an **internal engineering holdout** reconstructed
  from the old training pool by whole chemical systems. It is valid for comparing
  these variants because every version used the same partition, but it is not a
  pristine external publication test and is not bit-identical to the team's older
  split files.
- The result applies to this topology, feature definition, training objective,
  model grid, and screening pipeline. It should not be generalized beyond them.

## Validation performed after the run

The code and result package were checked again on 29 October 2026:

- all 30 run folders contain a completion record, history, and checkpoint;
- all 30 selected epochs and validation losses reproduce from the histories;
- every checkpoint hash matches the frozen evaluation record;
- the evaluated row IDs exactly match the 35,336-row reserved partition;
- all 30 model score columns are finite;
- all 136 reported metric rows and all 36 controlled contrasts were independently
  recalculated with zero mismatches;
- duplicate-aware evaluation used 13,379 recorded duplicate pairs;
- the five experiment modules pass Python syntax compilation; and
- manual review found no candidate-stability label in the edge features or model
  inputs, and the reserved outcomes were opened only after checkpoint freezing.

The local laptop environment does not currently contain PyTorch, so the PyTorch
unit tests were not rerun locally during this publication audit. The completed
cluster checkpoints, histories, hashes, and independently reproduced metrics were
available and were validated. Minor source-provenance differences were reconciled
to Windows line endings and a non-behavioural peak-memory logging compatibility
change; the edge-experiment source files themselves match the frozen protocol.

For the full technical protocol and per-model averages, see
[edge_attribute_reference_gnn.md](edge_attribute_reference_gnn.md).

