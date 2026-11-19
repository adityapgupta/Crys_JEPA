# Did changing the graph wiring improve screening?

**Experiment status:** complete — null result

**Result:** no graph wiring made message passing beat the no-graph baseline.

## The short version

This is the third and final experiment in the series. The first two added richer
information to the *edges* of a fixed graph and found no improvement; in both, a
control that turns message passing off (`readout`) did just as well as the full
graph network. That pointed at the graph's **wiring** — which crystals connect to
which — rather than the edge information, as the thing left to test.

This experiment keeps everything else fixed and changes only the wiring. It compares
four ways of connecting the reference crystals and asks one question: does any of
them let the graph network beat the no-graph baseline? The answer is no.

## What we changed

The reference crystals (the nodes), their embeddings, the training target, the data
splits, and the screening score were all held fixed. Only the edges changed:

- **physical** — the original graph: two references are connected when they appear
  together in a candidate's breakdown into competing phases. *(baseline)*
- **cosine** — connect each reference to its nearest neighbours in the embedding
  space. *(a control: a graph built from the embeddings can only reshuffle what the
  nodes already hold, so it is expected to add nothing)*
- **soap** — connect each reference to its nearest neighbours by local atomic
  geometry, from an independent structure descriptor. *(the main test: geometry is
  the information the embeddings carry least well)*
- **phase** — connect two references when they sit next to each other on the
  thermodynamic stability diagram.

The floor is the **readout** control: the same node information, with message passing
switched off.

## What we found

Reserved set of 35,336 candidates; at the 5% budget each method selects 1,766.

| Wiring | Pass rate at 5% |
|---|---:|
| readout (no graph) | 44.3% |
| physical | 44.4% |
| cosine | 44.5% |
| soap | 44.7% |
| phase | 44.5% |
| `angle_gate_1deg` (simple rule, for reference) | 45.8% |

For every wiring, the difference from the no-graph readout is within noise — none of
the confidence intervals exclude zero, across all three network types and three
seeds. No wiring beats the original graph either, and none reaches the simple
angle-gate rule.

The geometry wiring (`soap`) is the strongest of the four and the only one clearly
above the readout on average, which matches the idea that geometry is the most
complementary signal — but the margin is still within noise, so it is not a
demonstrated improvement.

## What it means

Together with the two edge-feature experiments, the series now gives one consistent
answer: **neither richer edge information nor a different graph wiring improves
Crys-JEPA reference screening.** The no-graph readout matches the graph networks in
every case. The frozen embeddings already appear to carry what this task needs, and
message passing over any tested graph adds no measurable value.

## What it does not show

- It does not test every possible graph; it bounds the four wirings and the single
  neighbour count (k = 10) used here.
- The labels are MatterSim stability proxies, not DFT or experimental results.
- The reserved set is an internal engineering holdout, suitable for comparing
  methods rather than a fully external test.

See [connectivity_reference_gnn.md](connectivity_reference_gnn.md) for the full
protocol, controls, pre-registered criteria, and the complete results table.
