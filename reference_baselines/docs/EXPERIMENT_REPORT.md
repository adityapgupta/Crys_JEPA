# Reference experiments and training readiness

## Completed MEGNet/eSEN screening experiment — analysis updated 28 October 2026

The SOC run for **eval folders 20–29 inclusive** and the combined report are
complete. The downloaded report files match all 11 hashes in their cluster
completion receipt. The notebook's earlier MEGNet-only block has been replaced
with a reusable analysis of both proxies, their coverage, and their agreement.
The original downloaded report files and the historical studies below are
unchanged.

Artifacts: [executed notebook](../reference_set_comparison/inspect_train.ipynb),
[combined cluster report](../reference_set_comparison/outputs/comparison_20_29/reports/REPORT.md),
[source tables](../reference_set_comparison/outputs/comparison_20_29/reports/pooled_metrics.csv),
and [derived tables/figures](../reference_set_comparison/outputs/comparison_20_29/reports/analysis/).
The reusable implementation is [analysis.py](../reference_set_comparison/analysis.py).

### What is being compared, and what union means

This tests **reference selection for a frozen JEPA screening score**, not a newly
trained graph encoder. Each node here represents an entire reference crystal.
For candidate embedding `h_C`, chemical references are the MP-20 references whose
elements are a subset of the candidate's chemical system. Global kNN retrieves
the nearest `q` reference embeddings without a chemistry restriction.

The exact index uses raw, unnormalised 512-dimensional embeddings and squared
Euclidean distance:

$$
\begin{aligned}
d^2(C,r) &= \sum_{j=1}^{512}(h_{C,j}-h_{r,j})^2, \\
D_R(C) &= \frac{1}{|R|}\sum_{r\in R}d^2(C,r).
\end{aligned}
$$

Lower `D_R` ranks first. It is an embedding-distance score, not a probability,
formation energy, or energy above hull. Exact-distance ties break by reference
row ID; score-ranking ties break by preserved `candidate_row`. The nine scored
configurations are chemical, kNN-5/10/15/25, and union-5/10/15/25; a deterministic
random candidate ranking is a tenth control.

For each candidate, **`R_union,q = R_chemical ∪ R_kNN,q`**, deduplicated by
reference ID, and its score is the mean squared distance over this combined
pool. This is implemented in
[indexing.py](../reference_set_comparison/indexing.py). It is not the union of
generated candidates selected by two methods.

**Clarification of the user's later expectation:** the union pool was **not**
used to calculate a separate energy-above-hull value. All selected candidates
were evaluated against the same fixed MP2023 phase bank per proxy, as specified
in the original independent-proxy plan. Changing the hull with the screening
method would change the label as well as the ranking. Some unrestricted kNN
references also contain elements outside the candidate system and cannot be
decomposition phases. A study of approximate hulls from restricted/union pools
would be a separate experiment, assessed against the common fixed hull; these
results do not claim to test it.

### Proxy stability and fixed-hull calculation

The released geometries are already MatterSim-relaxed. Their `e_relax` is total
cell energy in eV, not formation energy or energy above hull. This experiment
does not relax them again. MEGNet and eSEN perform single-point predictions on
these candidate geometries and on the stored MP2023 reference geometries.

MEGNet supplies predicted formation energy per atom. eSEN supplies total cell
energy, divided by the represented atom count. A separate fixed hull is built
for each model in its **own native energy convention**; no DFT energies or
additional MP compatibility corrections are mixed into these hulls.

For atomic-fraction vectors $x_C$, $x_j$ and proxy per-atom energies $E_m$, solve

$$
\begin{aligned}
H_m(x_C) &= \min_{\lambda_j\ge0}\sum_j\lambda_j E_m(P_j), \\
\text{subject to}\quad \sum_j\lambda_j x_j &= x_C,
\qquad \sum_j\lambda_j=1, \\
\Delta E_m(C) &= E_m(C)-H_m(x_C).
\end{aligned}
$$

The phase bank contains all relevant chemical subsystems, including elemental
phases and all available polymorphs before model prediction. There is no pruning
by lowest DFT energy. Generated candidates never enter the fixed hull. The
shared `FixedHull` solver uses a phase diagram for small systems and a linear
program fallback, checking composition balance, nonnegative fractions, and their
sum. A candidate can lie below this external bank; negative differences are kept.

For eSEN, optional formation energy is `E_t/N − sum_i f_i mu_i`, with each
elemental `mu_i` obtained from the same model's fixed unary references. Those
elemental terms cancel between the candidate and a composition-balanced mixture,
so eSEN hull distances can be calculated directly from its total energies per
atom. MEGNet and eSEN energies are never averaged together.

The primary positive label is **`Delta E_m < 0.1 eV/atom`**, meaning model-based
near-stability against this bank. Secondary thresholds are `<0.05` and `<=0`;
exact-threshold counts are retained. This strict 0.1 rule differs from the older
MatterSim study's `<=0.1` convention. Unknown predictions/incomplete reference
hulls remain unknown rather than being silently dropped or marked stable.

These are **proxy labels**, not DFT confirmation or evidence of synthesizability.
Different learned potentials can disagree, particularly on generated structures.
The models also share historical materials data, so two-proxy agreement is not
independent experimental validation.

### Parameters, provenance, and verification

| Item | Recorded configuration |
| --- | --- |
| Source population | 8,113 released rows; 6,258 eligible VUN candidates; 1,855 VUN exclusions |
| Identity/filtering | Original eval folder/JSON row preserved; successful structure parsing, finite recorded energy, no relaxation exception, `vu AND vn` |
| Reference embeddings | 27,136 x 512 MP-20 training references; frozen JEPA `model_1972.pt` |
| Embedding convention | Same reference/candidate checkpoint, config, primitive/reduced/type-sorted preprocessing and scaler, verified by hashes |
| Scaler caveat | Reconstructed from up to 20,000 crystals per JEPA training source; adapted embedding space, not an exact reproduction using the original unavailable scaler |
| Primary ranking scope | One ten-folder group, `eval_folder // 10 = 2`; pooled results aggregate that same group |
| Selection | Ascending distance; `floor(p * eligible population)` = 62 / 312 / 625 slots at p = 1% / 5% / 10% |
| Duplicate policies | Raw and greedy StructureMatcher unique-refill; primary results below use unique-refill |
| Random/interval settings | Seed 2026; 2,000 paired chemical-system bootstrap resamples; intervals descriptive and unadjusted |
| Evaluation subset | Union of all required method/budget/policy/scope selections and controls: 2,550 candidate rows |
| Fixed phase bank | 58,377 MP2023 reference structures; zero structure-import failures; independent of which method selected a candidate |
| Bank SHA-256 | `01ca5b8c8c42dee3c995d1f4f458cc3be61c8b2011aea0fee41f1e6dd5bd85ce` |
| MEGNet weights | `materialyze/MEGNet-Eform-MP-2018.6.1`, revision `10c13f7ebd5d89fcb376c037770b2e9b7bd87658` |
| eSEN weights | `facebook/OMAT24`, `esen_30m_mptrj.pt`, revision `8a5a78c7ba7b250a17e85fe85943c4608499d895` |
| Runtime environments | MEGNet: torch 2.6.0+cu124 / MatGL 4.1.0 / NumPy 2.2.6; eSEN: torch 2.4.1+cu124 / fairchem-core 1.10.0 / NumPy 1.26.4 |

Source tests cover deterministic exhaustive neighbours/ties, chemical eligibility,
union deduplication and direct-distance equivalence, selection rounding/refill,
reference alignment, provenance checks, binary hull solutions, LP/phase-diagram
agreement, elemental-offset cancellation, strict thresholds, incomplete hulls,
failure isolation, and resumable caches. Both models passed real-weight GPU
smoke tests on SOC: finite energies, singleton/chunk agreement and replicated-cell
per-atom units. No model inference was repeated on the laptop for this analysis.

The report-analysis tests additionally cover missing-label denominators,
precision-versus-yield winner differences, consensus denominators, membership
overlap checks, modified transfers, and receipt path traversal. The executed
notebook audits every reported count/denominator and selection overlap, reconciles
joint-model counts, verifies pooled/turn aggregation, and confirms **all MEGNet
metrics and its model identity match the earlier standalone report**.
On this update, **20 targeted tests pass**: ten proxy/pipeline checks, six
reference-export checks, and four new report-analysis checks. Seven model and
comparison figures are embedded in the executed notebook. This is targeted
verification, not a claim that every historical repository test was rerun.

### Primary results: unique top 1%, 62 requested slots per method

MEGNet labels all 62 selections for every method. eSEN has the unknown counts
shown below. Precision denominators are valid labels; yield bounds use all 62
requested slots, treating unknowns as failing/passing at the two extremes.

| Method | MEGNet pass/valid (precision) | eSEN pass/valid (precision) | eSEN unknown | eSEN slot-yield bounds | Confirmed both-pass / 62 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chemical | 41/62 (66.1%) | 37/62 (59.7%) | 0 | 59.7–59.7% | 27 (43.5%) |
| kNN-5 | 53/62 (85.5%) | 31/59 (52.5%) | 3 | 50.0–54.8% | 27 (43.5%) |
| kNN-10 | 51/62 (82.3%) | 28/60 (46.7%) | 2 | 45.2–48.4% | 25 (40.3%) |
| kNN-15 | 51/62 (82.3%) | 28/60 (46.7%) | 2 | 45.2–48.4% | 24 (38.7%) |
| kNN-25 | 56/62 (90.3%) | 28/59 (47.5%) | 3 | 45.2–50.0% | 26 (41.9%) |
| Union-5 | 41/62 (66.1%) | 29/60 (48.3%) | 2 | 46.8–50.0% | 22 (35.5%) |
| Union-10 | 43/62 (69.4%) | 27/58 (46.6%) | 4 | 43.5–50.0% | 22 (35.5%) |
| Union-15 | 42/62 (67.7%) | 27/58 (46.6%) | 4 | 43.5–50.0% | 22 (35.5%) |
| Union-25 | 42/62 (67.7%) | 24/58 (41.4%) | 4 | 38.7–45.2% | 19 (30.6%) |
| Random control | 34/62 (54.8%) | 17/56 (30.4%) | 6 | 27.4–37.1% | 9 (14.5%) |

MEGNet kNN-25 improves valid-label precision over chemical by **24.19 percentage
points**, with descriptive paired chemical-system 95% interval **[+8.62,+40.30]
pp**. eSEN reverses the observed ordering: kNN-5 is **−7.14 pp**, interval
**[−23.47,+11.38] pp**; kNN-25 is **−12.22 pp**, interval **[−29.65,+7.14] pp**.
All pure-kNN eSEN intervals include zero. Chemical's observed primary count stays
higher even if every missing pure-kNN label is assumed passing, but that bound
does not eliminate sampling uncertainty or establish universal superiority.

### Budgets, model disagreement, and interpretation

| Budget | MEGNet highest observed precision | eSEN highest observed precision | eSEN highest confirmed slot yield |
| --- | --- | --- | --- |
| 1% / 62 slots | kNN-25: 56/62 = 90.3% | Chemical: 37/62 = 59.7% | Chemical: 37/62 = 59.7% |
| 5% / 312 slots | kNN-5 and kNN-10: 259/312 = 83.0% | kNN-5: 158/302 = 52.3% | kNN-5: 158/312 = 50.6% |
| 10% / 625 slots | kNN-10: 495/625 = 79.2% | Chemical: 266/575 = 46.3% | kNN-10: 273/625 = 43.7% |

At top 1%, chemical and kNN-25 share **zero selections**. That measures retrieval
overlap, not correctness. For kNN-25's 59 jointly labelled selections, **26 pass
both**, **27 pass MEGNet only**, **two pass eSEN only**, and **four fail both**;
three of its 62 selections are unknown in eSEN. Label agreement among the 59 is
**50.8%**. Chemical's agreement is 38/62 = **61.3%**. Chemical and kNN-5 tie at
27 confirmed dual-proxy successes per 62 slots; unknowns leave their ultimate
dual-proxy rates uncertain. Do not average model energies or precisions to
manufacture a single overall winner.

Unique refill matters: chemical screening has 39/62 eSEN positives under the raw
policy versus 37/62 for the primary unique policy. Do not mix those endpoints.
Sixty matching structure pairs were detected across the candidate pool; that is
a pair count, not sixty duplicate crystals. Per-folder heatmaps use their own
approximately six-slot rankings, not a decomposition of the primary selection.

**Conclusion:** pure kNN is stronger under MEGNet, but its primary gain is not
corroborated by eSEN or a clear gain in confirmed both-model yield. The union
score does not consistently improve both proxies. This one exploratory group
does **not** establish a proxy-independent best rule or a learned graph-model
improvement. Freeze settings before fresh-batch validation. Do not derive
full-population recall, score correlations, PR curves, or false-negative rates
from the selected-union-only labels.

The aggregate download does not identify the exact cause of each unknown eSEN
label. Inspect `esen_labels.csv`, `esen_hull_coverage.csv`, and prediction status
on SOC before scaling up. Import success alone does not prove every reference
prediction/hull succeeded. Recorded runtime rows omit hard-killed allocations
whose completion timer was never written, so their sum is not total eSEN GPU
time. Original protocol implementation notes and earlier studies follow below.

## Independent proxy comparison implementation - 24 October 2026

The new staged workflow in `../reference_set_comparison/` compares chemical
references, kNN (5, 10, 15, 25), and deduplicated unions at top-1%, 5%, and 10%
budgets. It reuses the new reference export's exact checkpoint/scaler/preprocessing,
and constructs separate fixed, model-native MEGNet and eSEN hulls from all relevant
MP2023 reference structures. It does not change the historical results below.

The actual sampled release batches 20-29 were checked locally: 8,113 source
rows, 6,258 eligible VUN candidates, and 1,855 VUN exclusions. Offline tests cover
the real JEPA architecture on untrained fixtures, deterministic retrieval,
selection/uniqueness, fixed-hull calculations, missing-label policy, and resumed
proxy caches using synthetic predictions. These are implementation checks.

**Historical implementation snapshot (before the completed SOC run above): no
public MEGNet/eSEN weights were downloaded or run on the laptop, and production
results were not yet available.** The SOC
guide `../reference_set_comparison/COMMANDS_TO_EXECUTE.md` includes isolated
environments, real-weight smoke tests, staged Slurm commands, cache resumption,
and result retrieval. The batch range is configurable; 20-29 is the initial run.

> **Correction — 23 October 2026.** The learned GNN/MPNN results in the earlier
> GNN section are invalid for the new reference-selection question. Those models
> predicted candidate stability and used candidate stability labels for training
> or checkpoint choice. The new model predicts reference utility only. Candidate
> stability comes from MatterSim relaxation. See
> [the new design](gnn_reference_selection.md). The historical numbers below are
> kept as a record and must not be reported as results for the new method.

## Pre-GNN controls — 22 October 2026

The new controls change the interpretation of the earlier global gain. At test
5%, original mean gives 644 passes, formula-balanced mean 696, and the fixed
angle gate 782. With a shared unique candidate set and identical chemical-system
quotas, the gate gives 640 versus 644 under original quotas, and 759 versus 757
under angle-rule quotas. Both gain intervals include zero. This supports a
chemistry-mix explanation for much of the global gain; it is not proof of
improved same-formula structural ranking or a causal decomposition.

A seeded same-formula study includes 4,622 passing/failing test pairs across
1,262 formulas. Original and angle-gate pair-order accuracies are 50.8% and 50.7%.
Direct ranking by the exact evaluation hull labels gives 100% passes at the
requested global budgets; it is an explicitly labelled oracle, not an independent
predictor. These controls use the existing saved MatterSim labels and do not mix
in selective extra-relaxation results. All four budgets and both splits are saved.

The official JEPA checkpoint was obtained, but its exact lattice scaler and the
script that produced our saved vectors were not found locally or in the public
release listing. Raw feature and geometric equivalence checks are complete;
encoder reproduction and invariance remain pending those inputs. A fresh final
evaluation protocol is prepared, but new candidates must be collected only after
model settings are frozen. No GNN training or new generation has started.

See [full pre-GNN findings](pre_gnn_report.md) and
[the progress tracker](PRE_GNN_PROGRESS.md). Earlier sections below record the
chronology and their original numerical results remain unchanged.

## Mean and reference-pool controls — 19 October 2026

The previous 1° candidate-gate gain did not use weighted averaging. The paper's
screening mean is also uniform (Eq. 11); its energy-weighted training loss is a
separate operation. New controls vary the pool and averaging rule separately,
using the same saved MatterSim labels at 1%, 2%, 5%, and 7% in both splits.

At 5% on the test set, original chemical references give 644 passes with a plain
mean and 692 with normalized exp(-distance/10) weights. The 1° reference pool
(nearest ten, original-pool fallback) gives 785 plain and 793 weighted. Weighting
alone does not explain the angular gain. The original-plus-five-cosine-neighbour
union gives 682 plain and 648 weighted: enlarging the pool is not a reliable gain
under these controls. The plain angle-pool and previous candidate gate overlap
in 1,783/1,849 selections (96.4%) and differ by only three passes, so better
physical reference choice is still not established. These are exploratory
comparisons on already inspected splits. All 36 tests pass.

See [the mean ablation report](mean_ablation.md) for all 12 controls, all budgets,
fixed weight settings, exact effects and uncertainty ranges. No new training or
relaxation was run; extra re-relaxation results were not mixed into this table.

## Budget study — 19 October 2026

Both fixed rules were compared at 1%, 2%, 5%, and 7% of the same 36,994 test
candidates. On the same saved MatterSim energies, original versus angle-rule
stability passes are 146/157, 269/328, 644/782, and 905/1,042. The gains are
2.98, 7.98, 7.46, and 5.29 percentage points. The 1% uncertainty range includes
zero. No rule, angle threshold, or generator was changed. This is screening,
not new generation, and the test outcomes were already open.

Separate extra-relaxation checks give 149/162 at 1%, 273/339 at 2%, and 662/808
at 5%. The full 7% original selection has not received that extra check, so no
complete fresh 7% comparison is claimed. See the [budget study](budget_study.md)
for the selection counts, uncertainty, check coverage, and commands.

## Current status — 18 October 2026

Fresh MatterSim validation and a qualitative study are now complete. All 1,394
added test selections were checked again and re-relaxed; all converged and pass
the basic geometry/composition rules. Of these, 635 meet the stability cutoff.
Both methods received the same checks. After convergence, validity, and duplicate
checks without refill, the angle rule has 808 unique stability passes versus 662
for original Crys-JEPA, from 1,849 original slots each. These counts describe the
new re-relaxation stage; the frozen results below remain unchanged.

See [fresh MatterSim validation](mattersim_validation.md) and
[why the rule helped](selection_interpretation.md), including matched wins,
counterexamples, structures, reference pools, and embedding plots. All 27 tests
pass. DFT is reserved for final publication checks with the chemistry team.
Neural training remains deferred while the representation and reference roles
are reviewed.

The fixed baseline study and held-out test are complete. Neural training remains
deferred. The main rule uses a 1-degree cosine angle filter, then the original
Crys-JEPA score. It uses existing JEPA embeddings and known MP20 references;
it does not need a graph or new model training.

On 36,994 test candidates, each method selects 1,849 unique crystals. Original
Crys-JEPA has 644 passing selections (34.8%); the fixed angle rule has 782 (42.3%).
The gain is 138 crystals, or 7.46 percentage points. The paired chemical-system
95% bootstrap interval is +3.30 to +11.25 percentage points.

These are MatterSim estimates against a fixed MP2023 hull. They support this
screening rule, not physically better references or a trained graph model. The
rule strongly favors close embedding neighbours. Chemistry checks cover only a
small shared subset and cannot rule out bias across the full test pool.

Cross-batch duplicate checks and selected-set MP2023 novelty checks are complete.
All 19 tests pass. No independent DFT calculation, full neural training,
generation-loop test, or synthesis check has been run.

Next: review the matched examples and complete original-encoder invariance
checks before choosing a learned reference-weighting model. Continue comparative
work with MatterSim. Independent DFT is a final publication check rather than a
block on model development. The test set is open and cannot be reused as an
unseen test after further tuning.

See the [full test report](heldout_experiment.md) and the
[validation baseline study](nonparametric_experiments.md).

The sections below record the earlier preparation phase. Their validation results
remain valid, but their earlier training proposal is inactive.

## Validation results

A positive label means corrected MatterSim energy at most 0.1 eV/atom above the fixed MP2023 DFT hull. These are proxy results. The budget is 1% or 5% of 37,438 validation candidates.

For unique results, remove repeated structures across batches in score order. Refill from the same list to give each method the same number of unique crystals. Test performance was unused during this validation phase; the later fixed test is reported above.

| Method | Raw positive rate, 1% | Unique positive rate, 1% | Unique positive rate, 5% |
| --- | ---: | ---: | ---: |
| Crys-JEPA | 54.0% | 47.1% | 38.9% |
| kNN, k=10 | 48.4% | 45.7% | 41.0% |
| Union, k=10 | 51.9% | 47.1% | 41.4% |
| Intersection, k=5, with fallback | 62.3% | 52.7% | 42.5% |
| Chemical softmax, temperature=1 | 58.6% | 51.3% | 39.7% |
| Random candidates (100 repeats) | 38.2% | 37.3% | 37.4% |

The paired chemical-system bootstrap interval for intersection-fallback minus Crys-JEPA at the unique 5% budget is [0.8, 6.6] percentage points. Intervals are descriptive, with no multiple-comparison correction. These runs do not establish a reliable winner.

The pure k=5 intersection covers only 4.3% of validation candidates. The fallback uses the original chemical-pool mean when the intersection is empty. A larger union with a fixed mean does not give a clear small-budget gain.

## Reference quality

- All 27,136 known CIFs pass ID, formula, and descriptor checks. JEPA neighbours have closer geometry-only SOAP descriptors than energy-gap-matched controls. This supports a geometry signal, not screening success.
- The k=10 reference graph is connected. Formation energy is smooth on the graph. The eigensolver did not converge, so no spectral-gap result is claimed.
- On average, 45.4% of full-hull decomposition weight is present in the small reference bank. At least one bank phase exists for 27,980 validation queries. Recall is assessed only for phases available in this bank.
- Original chemical pools and unions retain all available decomposition phases by construction. Unrestricted k=10 neighbours recover about 1.0% on average among queries with an available phase. Cross-system references can provide context, but cannot supply decomposition phases with absent atoms.
- Distance, formation energy, and attention weights are not physical correctness labels for a reference.

## Data and checks

- References: 27,136 x 512. Candidates: 324,622 x 512. Every candidate source ID, chemical system, and VUN flag agrees with the supplied arrays. Folder 84 is absent in both sources; it is not regenerated.
- All 324,622 candidates receive usable labels. Candidate corrections follow the repository path. Evaluation uses the pinned full MP2023 hull, rather than the smaller MP20 screening hull.
- Training: 176,458 VUN rows; validation: 37,438; test: 36,994. Exact chemical systems do not cross these splits. This holds out supervised candidate labels, not encoder pretraining chemistry. Known reference nodes remain available to every split.
- Independent checks pass: 56 hull solutions against linear programming, decomposition balance, 128 stored-energy correction checks, and 768 direct reference-score checks. CSV parsing preserves the valid formula `NaN`.
- Global validation uniqueness checks cover 54,338 same-formula pairs and find 12,998 matching pairs. Baselines and training use the same greedy StructureMatcher rule.
- Six models pass finite forward/backward, target isolation, batch isolation, reference-order invariance, empty-pool, and attention-normalization checks. Checkpoint loading and reference export pass with a temporary untrained fixture. No optimizer step is taken.

## Earlier training proposal — inactive

This section records prototype code and the earlier proposal. It is not the
current training plan. Candidate-only and transformer models are excluded from
the current plan, and `../configs/training.toml` disables neural training.

Models: candidate-only MLP, reference attention, weighted GCN variant, weighted GraphSAGE mean variant, edge-aware MPNN, and sparse single-head graph transformer. They are small task-specific implementations, not exact reproductions. Each graph node is one whole reference crystal, not an atom.

Primary pool: original chemical references union top 10 JEPA neighbours. Each candidate has its own induced reference graph. Reference models share the same query-to-reference attention readout. Features are frozen JEPA vectors, composition, vector norm, distances, and chemical eligibility. Energy labels never enter the inputs.

The target is the MatterSim near-stable label. Use binary cross entropy, two learning rates, three seeds, and the fixed settings in `../configs/training.toml`. Stop on validation unique positive rate at the 5% budget. Pool and k ablations follow the primary comparison. Attention ranks context; it does not predict physical decomposition probabilities.

The earlier proposal included learned-versus-uniform readout checks. It has not
been run. Any later training plan must use the completed fixed baselines as its
controls and reserve fresh data for an unseen test.

## Limits and next stage

MatterSim error and estimated compatibility corrections affect these labels. Final force convergence is not recorded. Release VUN flags do not establish novelty against MP2023. Training rows can repeat within their own split; chemical-system grouping prevents such structures from crossing downstream splits.

The label audit flags 165 train/validation rows below -0.5 eV/atom relative to the fixed hull (38 validation rows). These extreme values can reflect force-field errors, correction mismatch, or sparse hull coverage. They are not treated as proven discoveries or silently removed. `validation_extreme_label_sensitivity.csv` reports results after excluding values below -1, -0.5, and -0.25 eV/atom. These cutoffs are diagnostic, not physical validity rules.

Independent DFT validation remains required for stability claims. After model selection, fix a balanced subset from Crys-JEPA, fixed alternatives, learned models, and random controls. Use the same DFT settings, corrections, hull, and failure policy. No gain in generation or synthesis is established by these offline results.

## Key files

- `../results/validation_unique_baseline_metrics.csv`: primary unique comparison, with and without refill.
- `../results/validation_baseline_metrics.csv`: 25 reference methods, two reference-free controls, random selection, and the label oracle.
- `../results/validation_per_turn_summary.csv`, `validation_threshold_sensitivity.csv`: turn and threshold checks.
- `../results/validation_paired_comparisons.csv`: uncertainty for fixed selections.
- `../results/validation_decomposition_summary.csv`: phase recall and bank coverage.
- `../results/training_data_audit.json`, `training_smoke_check.json`, `training_readiness.json`: readiness evidence.
- `../results/label_quality_audit.json`, `extreme_proxy_energy_flags.csv`: proxy-label concerns and audit.
- `../README.md`: reproduction commands. `../configs/training.toml`: exact training grid.

## Sources

- [Pinned Crys-JEPA data source](https://huggingface.co/datasets/liun-online/Crys_JEPA): structures, energies, reference hull.
- [Materials Project phase diagrams](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds): composition-constrained phase competition.
- [Matbench Discovery](https://arxiv.org/abs/2308.14920): fixed DFT hull and discovery selection metrics.
- [Active learning for crystal prediction](https://arxiv.org/abs/1802.07605) and [robust potential training](https://doi.org/10.1038/s41524-025-01602-9): why predicted low-energy structures still need DFT checks and transferability tests.
- [SOAP across structural and alchemical space](https://arxiv.org/abs/1601.04077), [DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html): independent geometry comparison.
- [GCN](https://arxiv.org/abs/1609.02907), [GraphSAGE](https://arxiv.org/abs/1706.02216), [MPNN](https://arxiv.org/abs/1704.01212), [graph transformers](https://arxiv.org/abs/2012.09699): model families for controlled alternatives.
