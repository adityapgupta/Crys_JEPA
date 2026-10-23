# Crys-JEPA reference-crystal project: handoff

Date: 2026-10-22

> **Superseded design note (2026-10-23).** Sections 9 and 10 describe the old
> GNN work, which predicted candidate stability. Those runs are invalid for the
> current question. The clean reference-utility design is in
> [gnn_reference_selection.md](gnn_reference_selection.md). The cluster steps are
> in [CLUSTER_GNN_RUNBOOK.md](CLUSTER_GNN_RUNBOOK.md).

This file is the working handoff for the next agent. Read it before changing code.

## 1. Project goal

The project improves how Crys-JEPA chooses known reference crystals for each generated candidate.

The source of references is the known MP20 training bank: 27,136 crystals. Generated crystals are query candidates only. They are not reference nodes.

The released generated candidates are already relaxed with MatterSim, but this does not mean that all are stable. Stability is checked later with corrected MatterSim energy above the fixed MP2023 DFT hull. The working pass rule is `energy_above_hull <= 0.1 eV/atom`.

The main question is whether a better reference-selection score gives more valid generated candidates than the original Crys-JEPA score, on the same candidate pool and budget.

## 2. Data and split facts

- Known reference crystals: 27,136 MP20 train structures.
- Generated candidates: 324,622 released candidates; 250,890 are in the VUN subset used in this work.
- Existing candidate split, grouped by chemical system: 176,458 train, 37,438 validation, 36,994 test.
- The old test set was used for early baseline studies. It must not be used to tune new GNN settings.
- Existing candidate structures have MatterSim relaxation results and corrected hull labels.
- JEPA embeddings are frozen 512-dimensional vectors. The encoder was trained for formation-energy-related representation learning, not directly on hull labels.
- Original checkpoint: `reference_baselines/data/jepa_audit/model_1972.pt`.
- Checkpoint SHA256: `c0f452d329dd1ced5460621db20549d669c616b11201a78c5088351c8a890b9a`.
- Exact original scaler and embedding extraction script are still missing. This is a provenance limitation. Frozen-vector experiments can continue, but claims about exact encoder reproduction must remain qualified.

## 3. What the original Crys-JEPA procedure does

For a generated candidate, collect known MP20 references whose elements are a subset of the candidate element system. For each eligible reference, compute the raw squared distance between the candidate and reference JEPA embeddings. The original score is the mean distance over eligible reference structures. Lower score is better.

The original method then selects references/candidates under the paper's budget procedure. The mean gives each reference structure equal weight. It does not give each formula equal weight. Several polymorphs of one formula therefore receive separate weight.

The generated candidates are later evaluated by MatterSim against the fixed hull. The selection rule does not make a crystal stable; it only changes which candidates are selected for evaluation.

## 4. Earlier reference-selection experiments

The strongest simple rule tested was an angle gate:

1. Keep a candidate if it has at least one known MP20 reference with cosine angle <= 1 degree in JEPA space.
2. Rank the kept candidates with the original distance score.

This used known MP20 references, not generated candidates. It used no graph and no learned weights.

On the frozen test 5% budget, the original rule selected 644 MatterSim passes and the 1-degree gate selected 782: +138 candidates, or +7.46 percentage points. A fresh union re-relaxation gave 662 versus 808 (+146, +7.90 percentage points). These are MatterSim comparative results, not DFT results.

Important controls:

- Formula-balanced weighting changed results, so part of the gain can come from changing chemistry allocation.
- When both methods were forced to select the same number from each chemical system, the gain was almost removed.
- Same-formula opposite-label pairs were close to random: original 50.8%, gate 50.7%.
- Therefore the current evidence does not prove that the JEPA graph finds physically better structures. It shows a change in candidate allocation and screening yield under MatterSim.
- A real within-system win exists, such as a Li3Ni2O5 example, but wins and losses occur across systems.

The fair ablation set was defined as:

- original paper with weighted mean;
- new neighbour/gate method with normal mean;
- new neighbour/gate method with weighted mean.

Do not attribute the total gain to neighbours until these components are compared on a clean split.

## 5. Why the chemistry point matters

Two crystals can have the same chemical formula but different space groups, atom positions, lattice, and energies. They are different reference structures. A chemical system is the set of elements, such as Na-Cl. A reference subsystem is any known reference whose elements are contained in the candidate system, such as Na, Cl, or Na-Cl for a Na-Cl candidate.

The current rule can favour some chemical systems and reduce others because it does not force equal system counts. This is a known source of bias. It is useful to report both unconstrained selection and chemistry-balanced controls.

## 6. GNN work completed

The user asked to compare a simple GCN and a message-passing neural network (MPNN), without moving to transformers.

New code:

- `reference_baselines/datasets/gnn_comparison.py`
- `reference_baselines/models/message_passing.py`
- `reference_baselines/training/compare_gnns.py`
- `reference_baselines/evaluation/gnn_comparison.py`
- `reference_baselines/configs/gnn_comparison.toml`
- `tests/test_gnn_comparison.py`

The comparison also has a no-graph learned-reference control. This control uses the same candidate-conditioned readout and formula-balanced prior, but no graph message passing. It is needed to test if any gain comes from learned reference weighting alone.

### GNN inputs

- Graph nodes: known MP20 reference crystals.
- Node features: frozen 512-D JEPA vector plus 100-D elemental-composition fractions.
- Graph: symmetric k=10 cosine-neighbour graph among known reference crystals, with self loops.
- Edge features: log raw squared embedding distance, chemical eligibility flag, and cosine distance.
- Query: one generated candidate. It is not inserted as a graph node.
- Readout prior: formula-balanced prior, `1/(number of formulas * polymorph count)`.
- Target: saved MatterSim stability pass label.

The model predicts a screening score. Its weights are not physical phase coefficients.

### Data isolation

A new deterministic reserve was made before fitting:

- fit: 141,126 candidates;
- original validation: 37,438 candidates;
- reserved from old train: 35,332 candidates.

The reserve is grouped by chemical system and was selected without energy labels. It is a prospective engineering holdout, not a pristine independent publication test, because the old train pool had earlier aggregate/extreme-label checks. The code rejects old `test` and `reserved` loaders.

The pilot used 4,096 fit candidates, 2,048 validation candidates, 5 epochs, hidden size 64, 2 layers, learning rate 0.001, weight decay 0.0001, and seeds 2026, 2027, 2028. This is an engineering pilot, not converged full training.

## 7. GNN results so far

Full validation has 37,438 candidates. The saved checkpoints were evaluated without more fitting.

At the 5% budget (1,871 selections):

| model | seed 2026 | seed 2027 | seed 2028 |
|---|---:|---:|---:|
| no graph | 1,101 (58.85%) | 1,135 (60.66%) | 1,144 (61.14%) |
| GCN | 1,122 (59.97%) | 1,122 (59.97%) | 1,126 (60.18%) |
| MPNN | 1,127 (60.24%) | 1,110 (59.33%) | 1,114 (59.54%) |

Fixed non-learned baselines on the same validation set:

- original mean: 727 (38.86%);
- angle gate: 857 (45.80%);
- formula-balanced mean: 803 (42.92%).

The learned models are not directly comparable to the fixed baselines as a final scientific result yet: they were trained on a subset of related data and use target labels. The full validation set includes the pilot validation rows used for epoch choice. Treat these numbers as development evidence.

Main interpretation:

- GCN and MPNN both run correctly.
- Neither is a clear winner across seeds.
- The no-graph control is often as good as or better than a graph model.
- This means a graph advantage is not established.
- The self-edge-only ablation gives almost the same scores as full graph, which further suggests that the current message passing is not yet using useful neighbour information.
- Same-formula pair accuracy is only about 0.51–0.55, so the learned scores do not yet show strong within-formula energy ordering.

Pilot runtime was about 62 seconds total on the local CPU using two threads. Peak process memory was about 1.37 GiB. Full-data training will be slower; this pilot does not measure full graph-cache cost.

## 8. Tests and checks

Before GNN work, 42 tests passed. Six GNN tests were added. Expected total is 48 tests.

The GNN tests cover:

- deterministic reserve allocation;
- rejection of test/reserved loaders;
- finite gradients for all parameters;
- target and query isolation;
- node permutation invariance;
- duplicate-aware fixed-budget scoring.

Run the complete suite before changing the model:

```bash
reference_baselines/environment/audit/bin/python -m pytest -q
```

Run the frozen pre-GNN audit:

```bash
reference_baselines/environment/audit/bin/python -m reference_baselines.checks.pre_gnn
```

## 9. Output locations

GNN outputs are ignored local artifacts under:

- `reference_baselines/results/gnn_comparison/`
- `reference_baselines/results/logs/gnn_comparison_pilot.log`
- `reference_baselines/results/logs/gnn_full_validation.log`

Important files are `full_validation_metrics.csv`, `full_validation_baselines.csv`, `full_validation_intervals.csv`, `full_validation_same_formula.csv`, and the per-model prediction CSV files.

Do not commit checkpoints, `.pt` files, JSON result dumps, arrays, embeddings, or environments.

## 10. Exact next work

1. Run the full 48-test suite and the frozen pre-GNN audit.
2. Read the full validation CSVs and add a short GNN report at `reference_baselines/docs/gnn_comparison.md`.
3. Update `reference_baselines/README.md`, `reference_baselines/docs/EXPERIMENT_REPORT.md`, and `reference_baselines/docs/PRE_GNN_PROGRESS.md` so they state that the GNN pilot and full validation are complete, but full training is pending.
4. Add the GNN result and the missing-scaler limitation to the thesis progress report at `/Users/anirudhtn/repos/crystal-ml-thesis/PROGRESS.MD`. Only edit the relevant section.
5. Do not use the old test set to tune architecture, epoch, or thresholds.
6. Use the reserved set only after the model and all settings are frozen. Report it as a development holdout, with its provenance limitation.
7. Train GCN, MPNN, and no-graph control on the full fit set only after the pilot design is accepted. Keep seeds and settings fixed first.
8. Add the clean mean/weighted-mean and neighbour ablations on a new split. This is needed to separate the effect of reference weighting from the effect of JEPA neighbours.
9. Study why the gain occurs by reporting chemical-system allocation, formula/polymorph counts, nearest-reference distances, and examples where original and new rules disagree.
10. Keep MatterSim as the current comparison evaluator. DFT is a final publication step with the chemistry team, not a current blocker.

## 11. Rules for the next agent

- Do not call all generated candidates stable. Relaxed means geometry optimisation finished; stability is a separate hull test.
- Do not say the GNN found physically better crystals unless an independent stability evaluation supports that claim.
- Do not mix generated candidates with the known MP20 reference bank.
- Do not change old frozen modules or rerun the old test-based selection protocol under new settings.
- Keep reports short and use plain English.
- No transformers yet. First establish whether simple MPNN or GCN adds value over no-graph weighting.
