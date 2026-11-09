# Did non-redundant relational GNN edges improve screening?

**Experiment status:** complete

**Result:** no reliable improvement was demonstrated

## The short version

This is the follow-up to the edge-attribute experiment. That first attempt added
edge features (embedding similarity, same space group, electronegativity) and found
no gain — but those features could all be re-derived from the node embeddings the
model already had, so a skeptic could dismiss the null as "of course, the features
were redundant."

Here we fixed that objection. We added three edge features **deliberately chosen to
carry information the Crys-JEPA embeddings do not already encode**, then re-ran the
same test. The result still held: across three model types and three seeds, the new
edge features performed about the same as the plain baseline, their own shuffled
version, and a control that does no message passing. **The stronger conclusion:
even genuinely complementary edge information adds no screening value on this graph.**

## What we were trying to improve

Crys-JEPA screens generated crystals by comparing each candidate with known
reference crystals (lower embedding distance to a chemistry-matched pool ranks
first). The GNN's job is to learn a useful weight for each reference. The "physical"
reference graph connects two known crystals when they co-occur in a fit-side
phase-decomposition; each connection carries one number, `log(1 + co-occurrence)`.

We asked whether describing each connection with **richer, independent** information
would let the model weight references better and push more near-stable candidates
to the top of the list.

## What we changed

We kept the nodes and the connections fixed and added three descriptions to every
edge, each chosen to be **non-redundant with the embeddings**:

1. **SOAP geometric distance** — how different the two crystals' local atomic
   geometry is, from a structure descriptor computed independently of Crys-JEPA.
   (Geometry is the axis the symmetry analysis showed the embedding is weakest on.)
2. **Co-occurrence PMI** — do these two references team up in decompositions *more
   than their individual popularity predicts*? This is pure decomposition
   statistics, not a function of the embeddings.
3. **Hull compatibility** — real thermodynamics: do the two references form a
   genuine two-phase tie-line on the convex hull of their own chemical system (can
   they coexist as a stable mixture)? Built from reference formation energies.

Everything else was held constant — reference crystals, candidate pools, graph
connections, data splits, training target, checkpoint rule, and the final screening
score. So this is a clean test of **edge information**, not of a new way to wire the
graph.

## The variants (controls)

| Version | Edge information | Question answered |
|---|---|---|
| `scalar` | co-occurrence weight only | baseline — how well does the existing edge work? |
| `geom` | + SOAP distance | does independent geometry help? |
| `cooc` | + co-occurrence PMI | does popularity-normalised competition help? |
| `phase` | + hull compatibility | does phase-diagram competition help? |
| `rich` | + all three | do the new features help combined? |
| `shuffled` | the `rich` features permuted across edges | are the features real signal, not noise? |
| `readout` | features visible, but no message passing | is graph propagation itself useful? |
| `redundant` | + embedding cosine only | re-confirms a known-redundant feature adds nothing |

We ran GCN, GAT, and GraphSAGE with seeds 2026/2027/2028 for every variant, plus
the architecture-independent `readout` control — **66 frozen training runs** in
total. The model was trained to predict whether a reference appears in a candidate's
saved hull decomposition; candidate stability was never the training target. After
checkpoint selection on validation reference loss, the frozen models ranked a
chemistry-disjoint reserved partition. A candidate counts as a pass when its saved
MatterSim proxy hull distance is at most 0.1 eV/atom, using the same budget for
every method.

## Result

The reserved partition contained 35,336 candidates; at the 5% budget each method
selects 1,766. The scalar baseline passed ≈ 44.4% — identical to the edge-attribute
experiment's baseline, which confirms this ran on the same harness and the numbers
are directly comparable.

**No variant beat the baseline.** Mean yield difference versus `scalar`, averaged
over the nine model×seed runs, with how many of those nine bootstrap confidence
intervals exclude zero in the variant's favour:

| Comparison | Mean yield difference | Runs that beat the control (of 9) |
|---|---:|---:|
| `rich` − `scalar` (all three features) | −0.24 points | 0 |
| `geom` − `scalar` (SOAP distance) | −0.69 points | 0 (1 run was worse) |
| `cooc` − `scalar` (co-occurrence PMI) | −0.32 points | 0 |
| `phase` − `scalar` (hull compatibility) | −0.03 points | 0 |
| `rich` − `shuffled` (are features real?) | +0.40 points | 1 |
| `rich` − `readout` (does propagation help?) | −0.16 points | 0 |
| `readout` − `scalar` (control) | −0.08 points | 0 |
| `redundant` − `scalar` (control) | −0.07 points | 0 |

Every difference is within ±0.7 of a percentage point of zero, and essentially no
confidence interval excludes zero. `rich` does not reliably beat even its own
shuffled version (1 of 9). And `readout` — which turns message passing off — matches
the full GNNs (`rich` − `readout` ≈ 0).

## What the experiment shows

- Adding independent geometry, decomposition statistics, and phase-diagram
  compatibility as edge features did **not** improve screening on this graph.
- This holds even though the features were specifically chosen to be *not* derivable
  from the embeddings — so it is the stronger form of the edge-attribute null.
- Turning message passing off (`readout`) did about as well as the full GNNs, in
  both experiments. **This is the key clue: the bottleneck is not the edge
  information but the graph wiring / propagation step itself.**

## What it does not show

- It does **not** show the embeddings lack structural, geometric, or thermodynamic
  information — it shows adding that information *on the edges* does not help ranking.
- It does **not** test a graph whose *connections* are built from geometry or phase
  relationships; the topology was deliberately frozen. That is the next experiment.
- It does **not** establish DFT stability or synthesizability; labels are MatterSim
  proxies.
- The reserved partition is the same internal engineering holdout used throughout;
  valid for comparing variants, not a pristine external test.

## Where this points next

Because `readout ≈ GNN` across both edge experiments, the remaining untested lever
is **connectivity** — how the graph is wired, not what the edges say. The natural
next experiment rebuilds the topology from a complementary signal (e.g. SOAP
geometric neighbours) and compares the GNN against the `readout` control again,
including an embedding-similarity graph as the control that tests whether an
embedding-derived wiring is (as expected) redundant.

For the full technical protocol, feature definitions, and pre-registered criteria,
see [relational_edge_reference_gnn.md](relational_edge_reference_gnn.md).
