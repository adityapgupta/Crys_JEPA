# GNN comparison summary

## Scope

This is the exploratory comparison shown in `presentations/idea_presentation.tex`.
Each method ranks the same generated candidates and selects a fixed budget. The
selected candidates are then evaluated with MatterSim. A pass means corrected
energy above the fixed hull is at most 0.1 eV/atom.

The values below are the presentation values supplied for this comparison. They
are not a replacement for the raw experiment files or for later experiments in
this directory.

## Baselines at 5%

| Method | MatterSim passes | Yield |
|---|---:|---:|
| Original Crys-JEPA | 721 | 40.8% |
| Cosine KNN | 722 | 40.9% |
| Intersection KNN | 760 | 43.0% |
| Formula prior | 693 | 39.2% |
| Random references | 679 | 38.4% |

Intersection KNN keeps the overlap between the five nearest JEPA references and
the chemically eligible reference pool. If the overlap is empty, it uses the
original Crys-JEPA score.

## Graph setup

The graph nodes are known MP20 reference crystals. Each node has a frozen JEPA
embedding and composition features. Edges connect related reference crystals.
A generated candidate is the query, not a graph node. The model ranks the
candidate by assigning weights to its reference pool; MatterSim is used only
after selection.

- GCN: normalized neighbour aggregation.
- GAT: learned attention over neighbours.
- GraphSAGE: mean neighbour aggregation.

## Graph results at 5%

| Model family | MatterSim passes | Gain vs original |
|---|---:|---:|
| GCN | 774 | +53 |
| GAT | 745 | +24 |
| GraphSAGE | 745 | +24 |

The presentation reports the GCN result as the strongest graph result in this
exploratory comparison.

## Budget sweep

| Method | 1% | 2% | 5% | 7% |
|---|---:|---:|---:|---:|
| Original Crys-JEPA | 189 | 328 | 721 | 954 |
| Cosine KNN | 175 | 318 | 722 | 950 |
| Intersection KNN | 186 | 333 | 760 | 1,013 |
| Angle gate | 193 | 348 | 769 | 1,042 |
| GCN | 188 | 333 | 774 | 1,067 |
| GAT | 175 | 329 | 745 | 1,022 |
| GraphSAGE | 183 | 325 | 745 | 1,018 |

The result is a MatterSim screening comparison. It does not prove lower true
formation energy or DFT stability.

## Relationship to other reports

Other files in `reference_baselines/docs/` describe separate validation splits,
controls, and later graph experiments. Their numbers must not be mixed with this
presentation table. The raw CSVs remain the source for reproducing each run.
