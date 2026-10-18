# Fixed test comparison

This run uses settings fixed before test energy outcomes were opened.

## Where we are now — 18 October 2026

The fixed baseline study and the held-out test are complete. The current rule is:
give priority to candidates within 1 degree of any known reference in JEPA space,
then use the original Crys-JEPA score within each group. References still come
from the known MP20 bank. No new reference model or crystal generator was trained.

The fixed rule improves the test stability pass rate from 34.8% to 42.3%. This
supports candidate screening with the existing embeddings. It does not prove
that the selected references are physically better or that graph learning helps.
The rule has a clear bias toward close embedding neighbours.

Completed: validation baselines, fixed test comparison, cross-batch duplicate
checks, MP2023 novelty checks, chemistry summaries, and numerical checks. All
19 tests pass. The graph methods are secondary comparisons; the main rule does
not use a graph.

Still open: independent DFT checks, the cause of the gain, and whether a trained
reference model adds value. No synthesis or generation-loop gain is established.

Next: fix a balanced DFT sample from the original and angle-rule selections,
including shared and different selections. Use the same DFT settings and failure
rules for every group. Review those results before choosing a training run.
Neural training remains deferred. The test set is now open; later method tuning
needs another held-out set or new generated candidates.

- Test candidates: 36,994. Reference bank: 27,136 known MP20 training crystals.
- Each method selects 1,849 unique candidates (5%, rounded down).
- Pass: MatterSim proxy energy at most 0.1 eV/atom above the fixed MP2023 hull.
- Known references define scores. They do not replace the evaluation hull.

## Main comparison

Original Crys-JEPA: 644 passing (34.8%).
Fixed 1-degree gate then original score: 782 passing (42.3%).
Change: +138 passing candidates; +7.46 percentage points.
Paired chemical-system bootstrap 95% interval: [+3.30, +11.25] percentage points.
This interval describes this fixed comparison. It does not include MatterSim error.

## Fixed methods

| Method | Passing | Pass rate | Novel against MP2023 | Novel and passing |
| --- | ---: | ---: | ---: | ---: |
| crys_jepa | 644 | 34.8% | 1823 | 635 |
| knn_k5 | 736 | 39.8% | 1848 | 736 |
| intersection_fallback_k5 | 728 | 39.4% | 1817 | 710 |
| cosine_knn_raw_k5 | 736 | 39.8% | 1848 | 736 |
| cosine_knn_cos_k5 | 737 | 39.9% | 1848 | 737 |
| cosine_intersection_fallback_k5 | 725 | 39.2% | 1817 | 707 |
| theta1_gate_then_original_score | 782 | 42.3% | 1835 | 774 |
| cosine_theta1_fallback_k10 | 785 | 42.5% | 1836 | 778 |
| euclidean_resolvent_0.25_seed_euclidean_k5 | 724 | 39.2% | 1848 | 724 |
| cosine_heat_1_seed_cosine_k5 | 698 | 37.8% | 1848 | 698 |

All other methods are secondary checks. Their settings were chosen from validation, then fixed. We do not choose another rule from these test results.

## Bias checks

Selected chemical systems: original 653; angle gate 1063.
Median angle to a known reference: original 2.52 degrees; gate 0.94 degrees.
Angle measures embedding proximity. It is not a measure of physical novelty.
MP2023 novelty: original 98.6%; gate 99.2%.
Novelty was checked again with StructureMatcher against every same-composition MP2023 polymorph. There is no refill after the novelty check.
Within shared chemical system groups with at least 5 selections from each method: 20 groups; population-weighted gain +6.57 percentage points.
Within shared composition groups with at least 5 selections from each method: 10 groups; population-weighted gain +6.19 percentage points.
These within-group summaries are descriptive. The shared groups cover only part of the test pool; they do not prove a cause.

### Further descriptive checks after opening the test results

- The gate selects 1,478 crystals within 1 degree of a known reference. Original Crys-JEPA selects 84 in that band. This is a strong shift toward close embedding neighbours. The overall gain does not prove better ranking within each distance band.
- The shared chemical-system check above covers 1,016 of 36,994 test rows (2.7%). The shared-composition check covers 176 rows (0.5%). These checks are too narrow to rule out chemistry bias across the whole test pool.
- Of the angle rule's 1,849 selections, 1,358 have no same-composition entry in MP2023. For the original method, the count is 1,149. Much of the high novelty count therefore comes from compositions absent from the bank. It does not prove more distinct atomic arrangements at a fixed composition.
- Tightening the energy cutoff keeps the direction of the gain: at 0.05 eV/atom, original Crys-JEPA has 313 passing selections and the gate has 450. At zero, the counts are 69 and 142. These remain MatterSim estimates.
- One gate selection has a proxy hull distance below -0.5 eV/atom. Removing it leaves 781 passing selections out of 1,848 retained selections. The original set has no such extreme case. No refill is used in this check.

The result supports this fixed screening rule on these generated candidates. It does not establish physically better references, a learned graph model, or synthesis success.

## Limits and next step

No new model was trained. No independent DFT check was done. Release force convergence is not recorded.
The split keeps generated chemical systems separate. JEPA pretraining and the known reference bank can contain these chemical systems.
This test set is now open. Do not use it to tune a new rule and call the next result unseen.
Review these results, then plan an independent DFT sample. New method choices need another held-out set or new generated crystals.

## Reproduce

Run these commands from the repo root with the prepared local inputs:

```sh
python -m reference_baselines.evaluation.heldout freeze
python -m reference_baselines.evaluation.heldout score
python -m reference_baselines.evaluation.heldout structures --workers 4
python -m reference_baselines.evaluation.heldout evaluate
python -m reference_baselines.evaluation.heldout report
```

Freeze, score, structures, and evaluate refuse to overwrite a completed phase. The protocol stores source and input hashes.
Local outputs: `reference_baselines/results/heldout/`. Input and output data stay out of Git.
