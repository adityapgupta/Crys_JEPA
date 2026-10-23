# What we are doing, explained from the start

This document explains the whole experiment in plain words. It starts from the
research goal and ends with the commands we ran on the cluster.

---

## 1. The goal

We generate many possible crystals with a diffusion model. Most are not useful.
We want to send as many *useful* crystals as possible to the refining step.

Refining is expensive. We can only refine a small number. So we must rank the
generated crystals and take the best ones first.

The question of this experiment is narrow:

> Can a graph model rank generated crystals better than the current ranking
> method, when both methods are allowed to refine the same number of crystals?

The graph model does **not** decide if a crystal is stable. It only changes the
ranking order.

---

## 2. The words we use, and what they mean

**Candidate.** One crystal that the diffusion model generated. We have 324,622
of them. Each candidate is a possible crystal structure, not yet accepted.

**Reference crystal.** One crystal that we already know and trust. These come
from the MP20 database and from our earlier relaxation runs. We have 27,136 of
them. They are the measuring sticks.

**Embedding.** A list of 512 numbers that describes a crystal. Two crystals with
similar embeddings are expected to behave in a similar way. We already computed
these numbers with a pretrained model called JEPA. We do not train that model.
We only read the numbers.

**Distance.** A single number that says how far one candidate is from one
reference in embedding space. Small distance means "similar". Large distance
means "different".

**Score.** One number per candidate that we use for ranking. We build it from
distances to several reference crystals. Lower score means "more promising", so
we refine it earlier.

**Stability.** Whether a crystal keeps its structure and does not fall apart
into other compounds. We measure it with MatterSim. MatterSim relaxes the
structure and reports an energy. If that energy is at most 0.1 eV per atom above
the known stable energy limit (the "hull"), we count the crystal as a pass.

**Hull.** For each mix of chemical elements, the hull is the set of lowest
known energies. A crystal that sits on the hull is stable. A crystal above the
hull has extra energy and is less likely to survive.

---

## 3. How the score is built today, and why references matter

Today the score of a candidate is an average of its distances to a small set of
reference crystals:

```text
score = weighted average of distances to the chosen references
```

If you pick good references, the score is meaningful. A candidate that looks
like several known stable crystals gets a low score. A candidate that looks
like nothing known gets a high score.

If you pick bad references, the score is noise.

So the real question is:

> **Which reference crystals should we use for each candidate?**

Everything in this experiment is about answering that question better.

---

## 4. The score and stability are two different things

This point caused confusion before, so we state it clearly.

**The score is a ranking tool.** It is cheap. It needs only embeddings and
distances. We compute it for all 324,622 candidates.

**Stability is the truth we want.** It is expensive. It needs a MatterSim
relaxation. We compute it only for the candidates we choose.

The score is used to *guess* which candidates are worth the expensive check. The
MatterSim relaxation decides the real answer. The score never overrides
MatterSim.

The pipeline is:

```text
all candidates
    -> compute score
    -> rank by score
    -> take the top N (our budget)
    -> run MatterSim on those N
    -> count how many pass
```

The number that matters is **how many pass inside the budget N**.

---

## 5. What the old graph run did wrong

An earlier version trained a graph model to predict *candidate stability*. That
is the same quantity that the final evaluation measures. The model was trained
on the answer it was later graded on.

That makes the reported gain meaningless. It measures memorisation, not
ranking skill.

We keep those old numbers only as history. We do not compare them with the new
results.

The new experiment does not touch candidate stability during training. Not in
the loss. Not in early stopping. Not in checkpoint choice. Candidate stability
is read once, at the very end, by the evaluator.

---

## 6. The new training target, explained slowly

When we relax a candidate against the hull, we get a **decomposition**. The
decomposition says:

> "This candidate, at its lowest energy, is a mix of these known crystals."

Example: a candidate for composition A2B3 might decompose into A + AB + AB3.
A, AB and AB3 are known crystals from our reference bank.

So each candidate gives us a small list of known crystals that the hull solver
actually used.

That list is our training target. For one candidate:

- Set the target to **1** for a reference that appears in its decomposition.
- Set the target to **0** for every other reference we are considering.

Then we train the model to answer, for each candidate:

> "Which of these references does the hull use for this candidate?"

Why this is a fair target:

- It is a property of the candidate and the hull. It does not depend on the old
  score, and it does not depend on stability.
- It is available for fit rows. We never need stability labels.
- It gives the model a real signal: good references are the ones the hull
  actually chose.

What this target does **not** claim: it does not say these references are
stable. It says the hull used them. That is all.

---

## 7. The reference pool for one candidate

We cannot let the model choose from all 27,136 references for every candidate.
That is too wide, and most are chemically impossible.

So each candidate gets a short **pool**: a small list of references that could
plausibly matter. The pool has two parts:

1. **Chemically eligible references.** A crystal can only decompose into phases
   made of elements it already contains. This part comes from a precomputed
   table of allowed element systems.
2. **Nearest references by embedding.** The 5 reference crystals closest to the
   candidate in JEPA space.

We merge the two lists, remove duplicates, and keep the result.

We checked this pool. Every reference that appears in a candidate's saved
decomposition is inside that candidate's pool, in all three data splits. If
this check failed, the target would be impossible to learn, and we would stop.

---

## 8. The graph, explained

A graph is a set of **nodes** and **edges**.

- A **node** is one known reference crystal. We have 27,136 nodes.
- An **edge** connects two references that we want to treat as related.
- An **edge weight** is a number that says how strong the relation is.

Generated candidates are never nodes. They are the things we score, not part of
the reference bank. We do not add generated crystals to the reference bank,
because then the bank would contain the answers we are trying to judge.

The purpose of the graph is to let the model share information between related
references. If reference A is useful for a candidate and reference B is closely
related to A, then B is probably useful too. A model with a graph can learn that
pattern. A model without a graph cannot.

---

## 9. The four graphs

We build four different graphs. They differ only in how we connect the
reference crystals. Using the same models on all four tells us whether the
connection pattern matters.

### Graph 1: `physical` — the main method

Two references get an edge when they appear together in the same hull
decomposition of a fit candidate. The weight grows with how often they appear
together.

The idea: the hull already tells us which crystals coexist. Crystals that
coexist for one candidate are likely to be useful for similar candidates.

This graph is built **only from fit rows**. Validation and reserved rows are
blocked by an explicit check, so the graph cannot leak the evaluation answers.

### Graph 2: `cosine` — similarity graph

Two references get an edge when they are close in JEPA embedding space. We take
the 10 nearest neighbours of each reference and connect them. The weight is the
cosine similarity.

This graph uses no decomposition information. It is a strong alternative built
only from the embeddings.

### Graph 3: `self` — no information between references

Each reference connects only to itself. No message passes from one reference to
another.

This is a control. It shows what the model can do when the graph carries no
useful relation at all.

### Graph 4: `permuted` — a scrambled graph

We take the `physical` graph and randomly relabel the nodes. The number of edges
and their weights stay the same, but the structure becomes meaningless.

This is a control for a subtle failure. If a model scores well on a scrambled
graph, then it is not using the graph structure. It is only using node features.
That result would weaken the claim that the graph matters.

---

## 10. About "four learning objectives"

There is **one** learning objective, not four. The model has one job:

> For this candidate, give a higher probability to the references that appear in
> its saved decomposition.

What varies is how the model passes information between references. We compare
three graph designs, and we test each one on four graphs.

So the count is:

```text
3 model designs × 4 graphs = 12 combinations
```

and with three random seeds each:

```text
3 × 4 × 3 = 36 training runs
```

---

## 11. The three model designs

All three use the same input, the same readout, and the same loss. They differ
only in how a reference combines information from its neighbours.

### `gcn` — Graph Convolutional Network

Each reference averages the messages from its neighbours. The average is scaled
by how connected each neighbour is, so a reference with 900 edges does not
dominate one with 3 edges.

Picture: every node listens to its neighbours, and loud neighbours are turned
down.

### `gat` — Graph Attention Network

Each reference also listens to its neighbours, but it learns *how much* to
listen to each one. The weight is not fixed by degree. It is predicted from the
features of the two nodes and from the edge weight.

Picture: every node decides which neighbours are worth listening to.

### `sage` — GraphSAGE

Each reference averages its neighbours, but it keeps its own vector separate and
concatenates the two. It does not blend itself into the average.

Picture: every node keeps its own opinion and also records what the neighbours
said, side by side.

### `no_graph` — the optional control

No message passing at all. Each reference is scored from its own features only.

This design is still available in the code, but it is **not** in the default run
list. We replaced it with `sage` so that we compare three real graph designs. If
a reviewer asks for the ablation, it can be run with `--model no_graph`.

---

## 12. What the model actually reads

For one reference, the model reads:

- the 512 JEPA numbers, normalised;
- 100 atomic fractions, which describe the chemical mix;
- the logarithm of the embedding length, which is a rough size signal.

For one candidate-reference pair, the model also reads:

- the distance;
- whether the reference is chemically allowed for this candidate;
- the cosine distance.

For one candidate, the model also reads the candidate's own 512 numbers. This
lets the model give different reference weights to different candidates.

---

## 13. How the model turns probabilities into a score

Step by step, for one candidate:

1. The model gives each reference in the pool a raw number (a **logit**).
2. We add a small fixed prior to each logit. The prior favours references whose
   chemical composition is common. This is a mild tie-breaker, not a decision.
3. We turn the adjusted numbers into weights with a softmax. A softmax makes all
   weights positive and makes them sum to 1.
4. The score is the weighted average of the distances.

```text
weight_i = softmax(logit_i + log_prior_i)
score    = sum_i (weight_i × distance_i)
```

Lower score is better. The score is still an average distance. The model only
decides how much each reference counts.

---

## 14. The loss, and how a checkpoint is chosen

**Loss** is the number we minimise during training. We use binary cross entropy
on the reference targets. In plain words: the model predicts a probability for
each reference; the loss punishes the model when a true reference gets a low
probability or a false one gets a high probability.

Two details:

- The loss is computed only over references inside the candidate's pool.
- Candidates with no mapped decomposition phase are masked out. They have no
  usable target, so they do not contribute.

**Checkpoint** means the saved model weights at one epoch. We save one checkpoint
per run. The rule is fixed before training starts:

> Keep the epoch with the lowest validation loss. On a tie, keep the earlier
> epoch. Stop early if validation loss does not improve for 4 epochs.

The choice uses reference targets only. It never uses candidate stability.

---

## 15. The three data splits

| Split | Rows | What it is used for |
| --- | --- | --- |
| Fit | 141,126 | Build graph edges and train the model |
| Validation | 37,438 | Pick the best epoch, decide when to stop |
| Reserved | 35,332 | One final evaluation, after all checkpoints are frozen |

The rows do not overlap. A separate check blocks reserved rows from entering the
fit split.

The reserved split is an engineering holdout. We inspected it for duplicate
structures before training, so it is not a pristine holdout for a publication
claim. It is still untouched by training and by model selection.

### Why the freeze matters

The evaluator loads every checkpoint and checks its hash *before* it opens the
reserved stability labels. If a checkpoint is missing or changed, the run stops
before any label is read. This prevents quiet tuning against the evaluation set.

---

## 16. The comparisons

Every method ranks the same reserved candidates. Every method uses the same
duplicate rule and the same budgets. Only the ranking differs.

### Baselines: the methods we must beat

| Name | What it does |
| --- | --- |
| `original_crys_jepa` | The current score, with the current reference choice |
| `knn_k5` | Average distance to the 5 nearest references |
| `intersection_fallback_k5` | Chemical and nearest-reference overlap, with a fallback |
| `angle_gate_1deg` | Reject candidates more than 1 degree off, then use the current score |

### Controls: methods that should *not* win

| Name | What it tests |
| --- | --- |
| `uniform_natural_pool` | Equal weight on all pool references. No learning. |
| `formula_prior_natural_pool` | Common-composition prior only. No learning. |
| `random_references` | Random references, matched in count. |
| `gcn_self_only`, `gat_self_only` | Learned model with self-loops only |
| `cosine`, `self`, `permuted` graphs | Same models, different or broken graph |

If the learned models cannot beat these controls, the gain comes from the
reference pool or from the prior, not from the graph.

---

## 17. How the evaluation works

For each method:

1. Score every reserved candidate.
2. Sort by score, lowest first.
3. Take the top 1%, then 2%, 5%, and 7% of the reserved rows.
4. Walk down the sorted list and skip a candidate if it duplicates one already
   selected. Duplicates are found with pymatgen StructureMatcher.
5. Count how many selected candidates pass the MatterSim rule.

The **pass rule** is: corrected MatterSim energy is at most 0.1 eV per atom
above the fixed MP2023 hull.

**Gain** is the number of extra passes compared with `original_crys_jepa` at the
same budget. We also report a paired 5% confidence interval. The pairing uses
the chemical system, so candidates from the same system are compared together.

The main comparison is the `physical` graph at the 5% budget. The other graphs
and budgets show whether the result is stable or a coincidence.

---

## 18. What we ran on the cluster

### Step 1 — copy the data

The code lives in Git. The data is too large for Git: about 1.4 GB. We copied
it with `rsync` through the jump host:

```sh
rsync -avP \
  -e 'ssh -J anirudh8@stujump.comp.nus.edu.sg' \
  --files-from=reference_baselines/cluster/cluster_inputs.txt \
  ./ anirudh8@xlogin.comp.nus.edu.sg:/home/a/anirudh8/Crys_JEPA/
```

We verified the copy two ways: a file-by-file existence check, and a
`rsync --checksum --dry-run` that listed nothing left to transfer.

### Step 2 — build the Python environment

We did not install on the login node. Its small `/tmp` caused a quota error. We
used a Slurm job instead, with temp files inside the home directory:

```sh
sbatch install.sh
```

The script made `~/venv-gnn` with Python 3.12, numpy, pandas, scipy, pymatgen,
and the **CPU** build of torch. CPU is the right choice: the model is small and
each candidate pool has only a handful of references, so a GPU would add
overhead without saving time.

### Step 3 — run the unit tests

```sh
sbatch --partition=normal --time=01:00:00 --cpus-per-task=8 --mem=16G \
  --output=tests-%j.log \
  --wrap="cd /home/a/anirudh8/Crys_JEPA && \
    PYTHONPATH=/home/a/anirudh8/Crys_JEPA \
    /home/a/anirudh8/venv-gnn/bin/python -m unittest \
    reference_baselines.tests.test_reference_gnn -v"
```

Result: **21 tests pass**. One test failed at first because it needed a
prebuilt graph. That was a test-isolation bug, now fixed.

### Step 4 — build the reference graphs

```sh
sbatch --partition=normal --time=03:00:00 \
  --export=ALL,STAGE=graphs \
  reference_baselines/cluster/reference_gnn.slurm
```

Output: four graph files and a report. The report confirmed:

```json
"status": "passed",
"candidate_labels_read": false,
"nodes": 27136,
"fit_rows": 141126,
"cooccurrence_pair_count": 25943
```

The graph used only fit rows. Candidate labels were not read.

### Step 5 — prepare the candidate pools

```sh
sbatch --partition=normal --time=03:00:00 \
  --export=ALL,STAGE=prepare \
  reference_baselines/cluster/reference_gnn.slurm
```

This built the pool and edge tables for the validation and reserved rows, for
each graph, plus the random control. The report again confirmed that no
candidate labels were read.

### Step 6 — train

We run one job per graph, in parallel:

```sh
for g in physical cosine self permuted; do
  sbatch --partition=long --time=1-00:00:00 --cpus-per-task=8 --mem=32G \
    --job-name=train-$g \
    --output=reference_baselines/results/logs/train-$g-%j.log \
    --export=ALL,STAGE=train,GRAPH_ONLY=$g \
    reference_baselines/cluster/reference_gnn.slurm
done
```

Each job trains 3 models × 3 seeds = 9 runs. Total: 36 runs.

Each run writes three files:

```text
best.pt          the saved weights
history.json     the loss at each epoch
complete.json    proof the run finished, with the chosen epoch
```

If a job dies, resubmit it. Finished runs are skipped. A partial run is
retrained.

### Step 7 — evaluate, once

```sh
sbatch --partition=normal --time=03:00:00 \
  --dependency=afterok:JOB1:JOB2:JOB3:JOB4 \
  --export=ALL,STAGE=evaluate \
  reference_baselines/cluster/reference_gnn.slurm
```

This is the only step that reads the reserved stability labels. It first checks
that all 36 checkpoints exist and match their recorded hashes.

---

## 19. The five files the evaluation writes

| File | Content |
| --- | --- |
| `metrics.csv` | Passes and yield for every method at every budget |
| `comparisons.csv` | Gain over the original method, with confidence intervals |
| `selections.csv` | The ranked candidate order for each method |
| `checkpoints.csv` | Hash, chosen epoch, and validation loss per checkpoint |
| `evaluation.json` | Record of the run: settings, hashes, and inputs |

---

## 20. How to read the result

Start with `metrics.csv` at the 5% budget. Look at the `passes` column.

Ask these questions in order:

1. Does `physical` beat `original_crys_jepa`? If not, the idea does not work at
   this budget.
2. Does `physical` beat `uniform_natural_pool` and
   `formula_prior_natural_pool`? If not, the gain comes from the pool or the
   prior, not from the graph.
3. Does `physical` beat `self` and `permuted`? If not, the graph structure is not
   doing real work.
4. Does the gain hold at 1%, 2%, and 7%? A gain at one budget only is weak
   evidence.
5. Do `gcn`, `gat`, and `sage` agree? If one model wins and two lose, the result
   is fragile.
6. Is the confidence interval in `comparisons.csv` above zero? If it crosses
   zero, the gain is not distinguishable from noise.

---

## 21. Limits you must state when you report this

- MatterSim is a proxy screen. It is not a DFT calculation. A pass is not proof
  of stability.
- The training target is the saved hull decomposition, not stability.
- About 56% of phases in the fit decompositions have no matching reference row.
  They stay negative for every reference. About 26% of fit candidates have no
  mapped phase at all, and their loss is masked.
- The composition prior is a heuristic.
- The reserved split was inspected for duplicates before training, so it is not
  a pristine holdout.
- The model cannot add new references. It only reweights the existing bank.
- Cluster runtime is measured only after the training jobs finish.

---

## 22. One-paragraph summary

We score each generated crystal by its weighted average distance to a handful of
known reference crystals. The old method picked those references with a fixed
nearest-neighbour rule. We replace that rule with a graph neural network. The
network is trained to guess which references the hull solver actually used for a
candidate. It never sees stability. Those guessed references set the weights,
the weights set the score, the score sets the ranking, and MatterSim checks the
top of the ranking. We compare the new ranking with the old one and with several
controls, at the same number of refinements per method.
