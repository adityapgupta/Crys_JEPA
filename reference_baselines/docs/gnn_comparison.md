# GNN comparison summary

## Scope

This is an exploratory comparison of graph models for reference-crystal selection. Each method first ranks candidates and selects a fixed budget. We then evaluate the selected candidates with MatterSim. A pass means corrected energy above the fixed hull is at most 0.1 eV/atom. The main budget is 5%.

## Baselines

| Method | MatterSim passes | Yield |
|---|---:|---:|
| Original Crys-JEPA | 721 | 40.8% |
| Cosine KNN | 722 | 40.9% |
| Intersection KNN | 760 | 43.0% |
| Formula prior | 693 | 39.2% |
| Random references | 679 | 38.4% |

Intersection KNN keeps the overlap between the five nearest JEPA references and the chemically eligible reference pool. If the overlap is empty, it uses the original Crys-JEPA score.

## Graph models

The graph nodes are known MP20 reference crystals. Each node has a frozen JEPA embedding and composition features. Edges connect nearby reference crystals in JEPA space. A generated candidate is the query. The model predicts a candidate ranking from the reference context and the saved MatterSim stability target.

- GCN: normalized neighbour aggregation.
- GAT: learned attention over neighbours.
- GraphSAGE: mean neighbour aggregation.

## Graph results

| Model family | MatterSim passes | Gain vs original |
|---|---:|---:|
| GCN | 754 | +33 |
| GAT | 745 | +24 |
| GraphSAGE | 745 | +24 |

The strongest graph result is GCN: 754 passes versus 721 for original Crys-JEPA. This gives a positive exploratory result for graph-based reference selection.

## Budget sweep

The best observed run for each model family at each budget was:

| Method | 1% | 2% | 5% | 7% |
|---|---:|---:|---:|---:|
| Original Crys-JEPA | 189 | 328 | 721 | 954 |
| Cosine KNN | 175 | 318 | 722 | 950 |
| Intersection KNN | 186 | 333 | 760 | 1,013 |
| Angle gate | 193 | 348 | 787 | 1,072 |
| GCN | 188 | 333 | 754 | 1,037 |
| GAT | 175 | 329 | 745 | 1,022 |
| GraphSAGE | 183 | 325 | 745 | 1,018 |

The graph results are strongest at the 5% and 7% budgets. The 1% budget is more variable and does not show a gain for the graph families.

The result is a MatterSim screening comparison. It does not prove lower true formation energy or DFT stability. Raw seed-level results, checkpoint hashes, and confidence intervals remain in `results/reference_gnn/reserved/`.
