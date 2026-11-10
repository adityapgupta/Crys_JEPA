# Did non-redundant relational GNN edges improve screening?

**Experiment status:** complete — re-run from scratch after two controls were corrected

**Result:** no reliable improvement was demonstrated (and with the corrected controls,
the new edge features do not even beat their own shuffled version)

## The short version

This is the follow-up to the edge-attribute experiment. That first attempt added
edge features (embedding similarity, same space group, electronegativity) and found
no gain — but those features could all be re-derived from the node embeddings the
model already had, so a skeptic could dismiss the null as "of course, the features
were redundant."

Here we targeted that objection. We added three edge features **drawn from sources
the Crys-JEPA embeddings do not use** (independent geometry, decomposition statistics,
and phase-diagram thermodynamics), then re-ran the same test (and re-ran it again from
scratch after fixing two control bugs). The result held: across three model types and
three seeds, the new edge features performed about the same as the plain baseline — and
with the corrected controls, no better than their own shuffled version or a control
that does no message passing. **The reading: adding this edge information did not
improve screening on this graph** — a result consistent with, but not proof of, the
idea that the limitation is the graph's wiring rather than the edge information.

> **Caveat on "independent."** The features are independent *by how they're built* —
> and we also ran a test (a "recoverability probe") that tries to reconstruct each
> feature from the embeddings. Geometry (SOAP) came back essentially unrecoverable, so
> it's genuinely new information; the decomposition statistic (PMI) is mostly new; the
> thermodynamic feature (hull compatibility) is *partly* already inside the embeddings.
> So the "new information" claim holds strongly for geometry, mostly for PMI, and only
> partly for thermodynamics. Separately: this is a "no reliable improvement" result, not
> a proof that the features are worthless — those are different claims.

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
| `shuffled` | the `rich` features permuted across undirected edge pairs | are the features real signal, not noise? |
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

> **Note:** these are the corrected-control numbers. The first run had two control
> bugs — the "shuffled" baseline scrambled edge directions independently (breaking the
> symmetry of the features) and the planned feature standardization was never applied.
> Both were fixed and the whole experiment re-run from scratch. The conclusion didn't
> change; the shuffled-control comparison actually got cleaner.

The reserved partition contained 35,336 candidates; at the 5% budget each method
selects 1,766. The scalar baseline passed **44.65%**. (The edge-attribute experiment's
baseline was 44.41%; the two are close but not identical — this run used a larger
training batch size — so they're comparable rather than a drop-in match.)

**No variant beat the baseline.** Mean yield difference versus `scalar`, averaged over
the nine model×seed runs, with how many of those nine bootstrap confidence intervals
exclude zero in the variant's favour:

| Comparison | Mean yield difference | Runs that beat the control (of 9) |
|---|---:|---:|
| `rich` − `scalar` (all three features) | −0.14 points | 0 |
| `geom` − `scalar` (SOAP distance) | −0.39 points | 0 |
| `cooc` − `scalar` (co-occurrence PMI) | −0.16 points | 0 |
| `phase` − `scalar` (hull compatibility) | −0.38 points | 0 |
| `rich` − `shuffled` (are features real?) | −0.43 points | 0 (1 run was worse) |
| `rich` − `readout` (does propagation help?) | −0.03 points | 0 |
| `readout` − `scalar` (control) | −0.12 points | 0 |
| `redundant` − `scalar` (control) | +0.03 points | 0 |

Every difference is within ±0.5 of a percentage point of zero, and essentially no
confidence interval excludes zero. Tellingly, with the fixed (symmetry-preserving)
shuffle, the scrambled `shuffled` version actually scored a touch *higher* than
`rich` — so `rich` does not beat its own shuffled control (0 of 9 favour it, 1 run
worse). And `readout` — which turns message passing off — matches the full GNNs
(`rich` − `readout` ≈ 0). For reference, the best simple non-GNN rule
(`angle_gate_1deg`) scores 45.81% — above every GNN variant here — so none of these
variants cleared the real screening bar either.

## What the experiment shows

- Adding independent geometry, decomposition statistics, and phase-diagram
  compatibility as edge features did **not** reliably improve screening on this graph.
- The features were chosen to be independent *by construction*, and a recoverability
  probe largely confirmed it: the embeddings cannot reconstruct the geometry feature
  (R²≈0.05) and only weakly reconstruct the decomposition statistic (R²≈0.18), though
  they *can* partly predict the thermodynamic feature (AUC≈0.71). So for geometry and
  mostly PMI, "the model already knew this" is ruled out — yet they still didn't help.
  And "no reliable improvement" is not the same as "no information" — we showed the
  former, not the latter.
- Turning message passing off (`readout`) did about as well as the full GNNs, in
  both experiments. **This is suggestive: it points to the graph wiring / propagation
  step, rather than the edge information, as the likely limitation — a hypothesis to
  test next, not something these runs prove.**

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
