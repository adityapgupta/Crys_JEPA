# Crys-JEPA GraphML project report

**Report date:** 23 October 2026  
**Status:** Graph pilot and full validation inference are complete. Full-data training is pending.

> **Correction for the new reference-selection experiment.** The learned runs in
> Section 9 are invalid for the new question. They predicted candidate stability
> and used candidate stability labels for training or checkpoint choice. The new
> experiment predicts reference utility, not candidate stability. Candidate
> stability comes only from MatterSim relaxation against the fixed MP2023 hull.
> Section 9 stays as a record of the old work. Do not report its learned yields
> as evidence for the reference-utility model. See
> [the new experiment design](gnn_reference_selection.md).

## 1. Executive summary

This project studies how to select known reference crystals for generated Crys-JEPA candidates.
The reference bank has 27,136 known MP20 training crystals. Generated crystals are query candidates.
They are not reference nodes.

The original Crys-JEPA method uses the mean squared JEPA distance to all chemically eligible references.
The main fixed alternative uses a 1-degree cosine-angle gate before the original score.
At a 5% test budget, the original method found 644 MatterSim passes and the angle gate found 782.
Fresh re-relaxation gave 662 and 808 unique passes. These are MatterSim results, not DFT results.

The later learned study compared a GCN, an MPNN, and a no-graph ablation.
At the 5% validation budget, all learned models gave a yield near 60%.
The no-graph ablation had the best mean result. The self-edge-only graph ablation also gave almost the
same result as the full graph. Thus, the current experiment does not show a useful graph-message effect.

The final project direction will remain graph-based because this is a GraphML project.
The no-graph model will stay only as a required scientific control. The next graph model must show that
graph edges add value beyond learned reference weighting.

## 2. Scientific question

The main question is:

> Can a graph-based reference model select more valid generated candidates than the original
> Crys-JEPA score, with the same candidate pool and selection budget?

A candidate passes the current stability screen when its corrected MatterSim energy above the fixed
MP2023 DFT hull is at most 0.1 eV/atom. This is a proxy screen. It is not DFT confirmation.

## 3. Data and split protocol

| Item | Count or status |
| --- | ---: |
| Known MP20 reference crystals | 27,136 |
| Released generated candidates | 324,622 |
| VUN candidates used in this work | 250,890 |
| Candidate training split | 176,458 |
| Candidate validation split | 37,438 |
| Old candidate test split | 36,994 |
| New fit partition from old training split | 141,126 |
| Reserved partition from old training split | 35,332 |

The candidate split is grouped by chemical system. The old test set was used for earlier fixed baseline
work. It must not be used to tune a new model. The reserved partition has not been evaluated.

The JEPA vectors have 512 dimensions and are frozen. The original checkpoint is
`reference_baselines/data/jepa_audit/model_1972.pt`. Its SHA256 is
`c0f452d329dd1ced5460621db20549d669c616b11201a78c5088351c8a890b9a`.

The original lattice scaler and the exact embedding extraction script are missing. Therefore, exact
encoder reproduction is not complete. The saved-vector experiments are valid for this project, but any
claim about exact encoder reproduction must state this limit.

## 4. Original Crys-JEPA reference score

For each candidate, the original method does the following:

1. Find every known reference whose elements are a subset of the candidate element set.
2. Calculate the raw squared distance between the candidate JEPA vector and each reference vector.
3. Use the mean distance as the candidate score.
4. Rank candidates from low score to high score.

The score gives equal weight to each reference structure. It does not give equal weight to each formula.
Thus, a formula with many polymorphs receives more total weight.

## 5. Fixed distance, angle, and graph experiments

The first phase used no supervised model. It tested nearest-reference pools, angle gates, and graph
operators on 37,438 validation candidates. Each method selected 1,871 unique candidates at the 5% budget.

| Validation method | MatterSim passes | Yield |
| --- | ---: | ---: |
| Original Crys-JEPA | 727 | 38.9% |
| Euclidean intersection with fallback, k=5 | 796 | 42.5% |
| Cosine intersection with fallback, k=5 | 798 | 42.7% |
| Cosine neighbors with raw-distance score, k=5 | 769 | 41.1% |
| Original score with strict on-hull references | 786 | 42.0% |
| 1-degree angle gate, then original score | **857** | **45.8%** |
| Best Laplacian resolvent setting | 754 | 40.3% |
| Best Laplacian heat setting | 747 | 39.9% |
| Best conductance reference sweep | 727 | 38.9% |

The graph operators used the full known-reference graph. Heat and resolvent diffusion spread query seed
weights across known reference nodes. Conductance selected graph prefixes. These graph methods were
numerically verified, but they did not beat the simple angle gate.

The graph eigenvector runs did not meet local convergence and time checks. Their results were excluded.

## 6. Fixed test result

The 1-degree gate was fixed before the old test outcome was opened. At the 5% budget, both methods selected
1,849 unique candidates.

| Test method | MatterSim passes | Yield |
| --- | ---: | ---: |
| Original Crys-JEPA | 644 | 34.8% |
| 1-degree gate, then original score | 782 | 42.3% |

The difference was 138 passes, or 7.46 percentage points. The paired chemical-system bootstrap interval
was +3.30 to +11.25 percentage points.

The two selections shared 455 candidates. The angle method added 1,394 candidates and removed 1,394.
The added group had 613 saved MatterSim passes. The removed group had 475 passes.

## 7. Fresh MatterSim validation

All 3,243 candidates in the union of the two test selections received new MatterSim energy and force
inference. They also received atom-and-cell re-relaxation. The same settings were used for both methods.

| Method | Unique usable | Unique passes | Yield per 1,849 slots | Unique novel passes |
| --- | ---: | ---: | ---: | ---: |
| Original Crys-JEPA | 1,842 | 662 | 35.80% | 653 |
| 1-degree gate | 1,849 | 808 | 43.70% | 800 |

The fresh difference was 146 passes, or 7.90 percentage points. All 1,394 candidates added by the angle
rule converged and passed the geometry and composition checks. Only 635 passed the hull threshold after
re-relaxation. Thus, relaxation completion does not mean stability.

## 8. Why the fixed gain occurred

The angle rule changed the chemistry allocation. It selected 1,063 chemical systems, compared with 653
for the original score. When both methods were forced to use the same chemical-system counts, most of the
global gain disappeared.

At the 5% test budget:

| Control | Original | Formula-balanced | Angle gate |
| --- | ---: | ---: | ---: |
| Global comparison | 644 | 696 | 782 |
| Original chemistry allocation | 644 | 650 | 640 |
| Angle chemistry allocation | 757 | 756 | 759 |
| Original formula allocation | 644 | 646 | 645 |
| Angle formula allocation | 772 | 761 | 759 |

Same-formula pair ranking was also near random. On 4,622 opposite-label test pairs, the original method
had 50.8% accuracy and the angle gate had 50.7% accuracy.

The evidence supports better allocation across chemistry groups. It does not yet show reliable ranking
of polymorphs with the same formula.

## 9. Learned graph experiment

### 9.1 Models

The learned comparison contains three matched models:

- **GCN:** Graph convolution on the known-reference graph.
- **MPNN:** Edge-aware message passing on the known-reference graph.
- **No-graph ablation:** The same encoder and candidate-conditioned readout, but no reference-to-reference
  messages.

The no-graph model is not the proposed final model. It tests whether a gain comes from graph messages or
only from learned reference weights.

### 9.2 Graph and features

The graph nodes are known MP20 reference crystals. The graph is a symmetric cosine k-nearest-neighbor
graph with k=10 and self-loops. Generated candidates are separate queries and are not graph nodes.

Each candidate query and reference node uses:

- the frozen 512-dimensional JEPA vector;
- a 100-dimensional element-fraction vector;
- the log vector magnitude.

The candidate-reference readout also uses:

- log raw squared JEPA distance;
- a chemical-eligibility flag;
- cosine distance.

The graph message layers use the graph weight and a self-loop flag. A formula-balanced prior gives equal
total initial weight to each reference formula. A learned readout can still distinguish its polymorphs.

### 9.3 Training protocol

The pilot used 4,096 fit candidates and 2,048 pilot-validation candidates. It used five epochs, hidden
size 64, two layers, learning rate 0.001, weight decay 0.0001, and seeds 2026, 2027, and 2028.

The full 37,438-row validation run used the saved pilot checkpoints. It made no additional optimizer
steps. The old test and the reserved partition were not evaluated.

The mean local graph had 28.14 reference nodes and 33.96 non-self edges per query. A total of 99.95% of
queries had at least one non-self edge.

### 9.4 Full validation results

At the 5% budget, each model selected 1,871 unique candidates.

| Model | Seed 2026 | Seed 2027 | Seed 2028 | Mean yield |
| --- | ---: | ---: | ---: | ---: |
| No-graph ablation | 1,101 (58.85%) | 1,135 (60.66%) | **1,144 (61.14%)** | **60.22%** |
| GCN | 1,122 (59.97%) | 1,122 (59.97%) | 1,126 (60.18%) | 60.04% |
| MPNN | **1,127 (60.24%)** | 1,110 (59.33%) | 1,114 (59.54%) | 59.70% |

The fixed baselines on this validation set were:

| Fixed baseline | Passes | Yield |
| --- | ---: | ---: |
| Original mean | 727 | 38.86% |
| Formula-balanced mean | 803 | 42.92% |
| Angle gate | 857 | 45.80% |

The learned results are not a final fair comparison with the fixed methods. The learned models used
MatterSim target labels and used a small related training sample. The pilot-validation rows also form part
of the reported full validation set.

### 9.5 Did graph messages help?

No clear graph gain was found.

- All six graph-versus-no-graph 95% intervals included zero.
- At 5%, GCN changed the no-graph count by +21, -13, and -18 across the three seeds.
- At 5%, MPNN changed it by +26, -25, and -30.
- The self-edge-only inference ablation changed most counts by zero or one candidate.
- Mean same-formula accuracy was 53.28% for no graph, 52.99% for GCN, and 52.69% for MPNN.

The learned readout improved screening, but the present message-passing layers did not add measurable
value. A likely reason is that the direct candidate-reference features already dominate the prediction.
The small training sample and five-epoch limit can also prevent useful graph learning. This is an
interpretation, not a proven cause.

## 10. Verification status

On 23 October 2026:

- 48 tests passed;
- 86 subtests passed;
- the test run had 31 deprecation warnings and no failures;
- the frozen pre-GNN audit passed;
- 432 selection groups were checked;
- exact counts and stability labels passed;
- matched quotas passed;
- frozen artifacts were unchanged;
- exact encoder reproduction remained incomplete.

The GNN tests cover split determinism, blocked test/reserved loaders, finite gradients, target isolation,
query isolation, node permutation invariance, and duplicate-aware fixed-budget scoring.

## 11. GraphML direction

The primary project model will use a graph. The no-graph model will remain an ablation because it is needed
to measure the value of graph edges.

The next graph experiment should use an edge-aware MPNN as the main model and a GCN as the simple graph
baseline. The graph should make reference roles explicit. Useful edge or relation types include:

- same formula;
- same chemical system;
- subsystem relation;
- cosine-neighbor relation;
- cross-chemistry nearest-reference relation.

The design should prevent indiscriminate neighbor averaging. It should keep candidate-reference features
separate from reference-reference edge features. It should also keep formula-balanced readout weights.

Required ablations are:

1. Full graph versus self-edges only.
2. Full graph versus the no-graph learned-readout control.
3. Real edges versus permuted or random edges with matched edge counts.
4. Formula-balanced prior versus a uniform structure prior.
5. Chemistry-matched evaluation and same-formula pair evaluation.

The model and settings must be frozen before the reserved partition is opened. The old test set must not
be used for model selection. Full training must use only the 141,126-row fit partition.

The graph model should be considered successful only if it gives a repeatable gain over the no-graph
control and if the self-edge ablation removes that gain. A global yield gain alone is not sufficient. The
report must also show chemistry allocation and within-formula behavior.

## 12. Current limits

- MatterSim provides the current labels. DFT confirmation is pending.
- The old validation and test outcomes are open.
- The learned pilot used only 4,096 training candidates and five epochs.
- The full validation set contains the pilot-validation subset used for checkpoint choice.
- The reserved partition is an engineering holdout, not a pristine publication test.
- The original JEPA scaler and embedding extraction script are missing.
- Current models do not show strong same-formula structural ranking.
- Current results do not prove physical decomposition-phase selection or synthesis success.

## 13. Main project files

- [Project handoff](HANDOFF.md)
- [Reference-utility GNN design](gnn_reference_selection.md)
- [Cluster runbook](CLUSTER_GNN_RUNBOOK.md)
- [Fixed experiment history](EXPERIMENT_REPORT.md)
- [Distance, angle, and graph baselines](nonparametric_experiments.md)
- [Fixed test comparison](heldout_experiment.md)
- [Fresh MatterSim validation](mattersim_validation.md)
- [Pre-GNN controls](pre_gnn_report.md)
- [Mean and reference-pool ablations](mean_ablation.md)
- [Selection interpretation](selection_interpretation.md)
- [Reference-utility GNN model](../models/reference_gnn.py)
- [Reference-utility data loader](../datasets/reference_graph.py)
- [Reference graph builder](../graphs/build_reference_graph.py)
- [Reference-utility training program](../training/train_reference_gnn.py)
- [Reference-utility evaluation program](../evaluation/reference_gnn.py)
- [Reference-utility configuration](../configs/reference_gnn.toml)

Local result tables are in `reference_baselines/results/gnn_comparison/`. Model checkpoints, arrays,
embeddings, environments, and generated result dumps must not be committed.
