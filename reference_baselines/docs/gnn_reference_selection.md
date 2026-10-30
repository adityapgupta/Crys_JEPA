# Graph learning for reference selection (approach 1B + 2A)

## 1. Research question

The screening method must send a larger number of valid crystals to refining.
The screening step is a fixed-budget ranking problem. Each candidate receives a
score. The lowest scores are selected first.

The question is:

> Can a graph model choose better known references for each candidate, and thus
> improve the fixed-budget pass rate?

The graph model does **not** predict candidate stability. Candidate stability
comes only from MatterSim relaxation against the fixed MP2023 hull.

## 2. Why the old GNN results are invalid

The old GNN, MPNN, and GCN runs predicted candidate stability. They used
candidate stability labels during training or checkpoint choice. That target is
the same target as the final evaluation. The reported gains are therefore not
evidence for reference selection.

The old numbers stay in the repository only as history. Do not compare them with
the new results.

## 3. Target

Each candidate has a saved hull decomposition. The decomposition is a list of
known phases. Map each phase to one known reference row:

1. Read `entry_id` from `results/candidate_decompositions.csv`.
2. Remove the `-GGA`, `-GGA+U`, or `-R2SCAN` suffix.
3. Look up the MP material ID in `embeddings/data/metadata.csv`.

The target of a reference is **1** when that reference appears in the saved
decomposition of the candidate. The target is **0** for every other reference
in the candidate pool.

This target is stable by construction. It is a property of the candidate, not of
the old screening score. It uses no candidate stability label.

This design does not claim that decomposition members are stable. It asks a
narrow question: which known references does the fixed-hull solution use?

## 4. Candidate pool

The pool for one candidate is:

- all references with an eligible chemical system, from
  `chemical_reference_indptr.npy` and `chemical_reference_indices.npy`;
- plus the top 5 nearest known references in the frozen JEPA space, from
  `candidate_cosine_indices.npy`;
- sorted and de-duplicated.

The audit confirms that every mapped decomposition target is in this pool for
all three splits (fit, validation, reserved).

## 5. Graphs

Nodes are the 27,136 known MP20 reference crystals. Generated candidates are
never nodes.

| Graph | Edge rule | Use |
| --- | --- | --- |
| `physical` | Two references are adjacent when both occur in the same fit-side hull decomposition. Weight is `log1p(count)`. | Primary method |
| `cosine` | Symmetric maximum of each reference's 10 nearest JEPA neighbours. Weight is cosine similarity. | Ablation |
| `self` | Self-loops only. | Removes all reference-reference messages |
| `permuted` | The physical graph with permuted node labels. | Tests whether physical structure matters |

All graphs have unit self-loops. The graph is built from fit rows only. The
validation and reserved rows are blocked by an explicit check.

## 6. Model

Input for one reference node:

- 512 frozen JEPA dimensions, layer-normalised;
- 100 atomic fractions;
- the log of the embedding norm.

The model projects this input to a hidden vector. It then applies graph layers.
Three model kinds use the same readout:

| Kind | Message passing |
| --- | --- |
| `gcn` | Symmetric degree-normalised convolution with edge weights |
| `gat` | Single-head edge-aware attention inside one candidate pool |
| `sage` | GraphSAGE mean aggregation, concatenated with the self vector |
| `no_graph` | No message passing. Node features and query only |

The readout concatenates:

- the projected query;
- the node vector;
- three pair features: log distance, chemical eligibility flag, cosine distance.

The readout gives one logit per reference. Weights come from a softmax inside
one candidate pool. The formula prior is added to the logit:

```text
weight_i = softmax(logit_i + log_prior_i)
score    = sum_i weight_i * distance_i / sum_i weight_i
```

The prior is the negative log of the reference composition frequency. The
learned term can still reorder polymorphs with the same composition.

Lower score is better.

## 7. Loss and checkpoint rule

Loss is binary cross entropy on reference decomposition membership. The loss is
computed only for references inside each candidate pool. It is masked for
candidates without a mapped decomposition phase.

Checkpoint rule, frozen before the run:

> Use the epoch with the lowest validation reference BCE. On a tie, use the
> earlier epoch. Early stopping patience is 4 epochs.

The model selection uses reference targets only. It never uses candidate
stability.

## 8. Splits

| Split | Rows | Use |
| --- | --- | --- |
| Fit | 141,126 | Graph edges and model training |
| Validation | 37,438 | Early stopping and checkpoint choice |
| Reserved | 35,332 | One final evaluation, after all checkpoints are frozen |

The legacy test rows are blocked from all three splits. The reserved partition
is an engineering holdout. It was inspected for duplicate structures before the
model run, but it is not a pristine holdout for publication claims.

## 9. Fixed baselines and controls

Every method uses the same candidate rows, the same duplicate rule, and the same
fixed budgets.

Fixed baselines:

- `original_crys_jepa` — the original score;
- `knn_k5` — mean distance to 5 nearest references;
- `intersection_fallback_k5` — chemical and nearest-reference intersection, with
  a fallback;
- `angle_gate_1deg` — 1-degree gate, then original score.

Controls:

- `uniform_natural_pool` — equal weights on the natural pool;
- `formula_prior_natural_pool` — formula prior only, no learned logits;
- `random_references` — matched random pool, equal weights;
- `gcn_self_only`, `gat_self_only` — learned readout with self-loops only;
- `no_graph` — learned readout without graph messages (supported, but not in
  the default model list);
- all learned models on the `cosine`, `self`, and `permuted` graphs.

These controls separate the effect of graph messages from the effect of the
learned per-reference readout.

## 10. Evaluation

Run the evaluation once, after all 36 checkpoints exist and pass the frozen
checkpoint check:

```sh
python -m reference_baselines.evaluation.reference_gnn --evaluate
```

Budgets are 1%, 2%, 5%, and 7% of the reserved rows. At each budget, greedy
selection removes duplicate structures with the StructureMatcher adjacency
table. No per-budget tuning is allowed.

Outputs:

- `metrics.csv` — passes and yield for each method and budget;
- `comparisons.csv` — paired chemical-system bootstrap intervals against
  `original_crys_jepa` at 5%;
- `selections.csv` — ranked candidate rows;
- `checkpoints.csv` — checkpoint hash, epoch, and validation BCE;
- `evaluation.json` — record and source hashes.

The primary comparison is physical graph versus the fixed baselines and the
controls at 5%. The other graphs show whether message passing helps.

## 11. Limitations

- The target is the saved decomposition, not DFT stability.
- MatterSim is a proxy screen, not DFT confirmation.
- About 56% of fit phases (198,557 of 352,942) have no mapped bank row. They stay
  negative for every bank reference. About 26% of fit candidates have no mapped
  phase; their loss is masked.
- The formula prior uses reference composition frequency. It is a heuristic.
- The reserved split is not pristine for publication claims.
- Full cluster runtime is not yet measured.
- The model cannot add new references to the bank.
