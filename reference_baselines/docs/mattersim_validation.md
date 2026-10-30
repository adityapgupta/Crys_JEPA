# Fresh MatterSim validation

## Answer

Every one of the 1,394 additional test candidates was run through fresh MatterSim energy/force calculations and atom-plus-cell relaxation.
All 1,394 converged. All 1,394 pass the periodic geometry checks, and all 1,394 pass the fixed SMACT14 composition rules.
After re-relaxation, 635 meet the energy-above-hull cutoff of 0.1 eV/atom. The other 759 do not. Therefore, passing the angle filter does not mean that every candidate is stable.
Of the added stability passes, 631 are also unmatched against MP2023 after relaxation under the fixed structure-matching rule. This is novelty against that bank, not proof of a new synthesizable material.
These counts are before removing any structures that merged during re-relaxation. The fixed-budget method comparison below includes that check.

## What was run

1. Fixed both existing selection lists. No candidates were generated, reranked, or added to fill gaps.
2. Recomputed energy and forces for all 3,243 crystals selected by either method, including all added candidates and the original-method controls.
3. Verified saved energies: maximum difference 3.81e-06 eV/atom; 0 pass/fail changes at the saved structures.
4. Re-relaxed atoms and unit cells with the same settings for both methods. Recomputed MP2020 corrections on each final structure and compared energies against the same fixed MP2023 hull.
5. Checked geometry, composition, duplicates, and MP2023 novelty again. No refill was used after the new checks.

## Complete re-relaxation counts

| Selection group | Candidates | Converged | Valid geometry | Composition passes | Converged, valid stability passes before dedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| added | 1394 | 1394 | 1394 | 1394 | 635 |
| removed | 1394 | 1390 | 1394 | 1392 | 490 |
| shared | 455 | 455 | 455 | 455 | 173 |

Across the full union, 3,243 calculations were attempted; 3,239 converged. There were 0 calculation errors. Unconverged cases are unresolved and excluded from confirmed usable yield.
Composition-rule failures are review flags. Single-state rules can reject real mixed-valence compounds; a passing assignment is not proof of actual oxidation states.

## Fixed-budget comparison after all checks

Each method keeps its original budget of 1,849 slots. Failed, unconverged, or duplicate slots are not refilled. Uniqueness is applied in the original selected order. Novelty is checked against all same-composition MP2023 polymorphs.

| Method | Unique usable | Unique stability passes | Passes per 1,849 slots | Unique, novel stability passes |
| --- | ---: | ---: | ---: | ---: |
| Original Crys-JEPA | 1842 | 662 | 35.80% | 653 |
| 1-degree gate then original score | 1849 | 808 | 43.70% | 800 |

After these fresh checks, the angle rule has +146 more unique stability passes, a change of +7.90 percentage points per original slot.
This is a separate, more complete validation stage. It does not replace or silently alter the original frozen 782-versus-644 result, which used the released relaxed structures and the earlier refill policy.

## Examples after fresh relaxation

These examples were chosen to explain wins and failures, not to estimate the population pass rate.

| Row | Formula | Selection | Final hull distance (eV/atom) | Converged | Stability pass |
| --- | --- | --- | ---: | --- | --- |
| 22551 | Li3Ni2O5 | removed | 0.2340 | True | False |
| 42259 | MgGaAg | removed | 0.0700 | True | True |
| 59917 | CeSn2 | added | 0.1896 | True | False |
| 66383 | CeSn2 | removed | -0.0217 | True | True |
| 76528 | YbGa3 | removed | 0.1492 | True | False |
| 94292 | MgGaAg | added | 0.0846 | True | True |
| 119737 | MgGaAg | added | 0.1737 | True | False |
| 138335 | YbAl3 | added | 0.0794 | True | True |
| 179578 | YbAl3 | removed | 0.1210 | True | False |
| 224026 | CuSe | removed | 0.0367 | True | True |
| 267519 | YbGa3 | added | 0.0631 | True | True |
| 310813 | Li3Ni2O5 | added | 0.0281 | True | True |

## Calculation settings and checks

- Official MatterSim-v1.0.0-1M weights; MatterSim package 1.2.5; CPU inference. No model training.
- Checkpoint SHA256: `28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be`.
- ASE FIRE optimizer; FrechetCellFilter allows atoms and cell to move; at most 300 additional steps; force/filter cutoff 0.05 eV/angstrom.
- Same settings, fixed hull, corrections, and failure policy for both selection methods.
- Accepted inference uses one graph per batch and matches the official ASE calculator in energy, forces, and input-order checks.
- The first multi-structure legacy batching run was rejected because it disagreed with the individual calculator. Those outputs are marked rejected and are not used.
- A fixed analytical basis buffer added by recent MatterSim versions is initialized from the package. No learned checkpoint parameter is missing or ignored.
- MatterSim and basic validity checks support this model-based stability conclusion. Phonon stability, synthesis, and independent DFT remain untested; DFT is reserved for the final publication stage.

## Files

Accepted local outputs are in `../results/interpretation/`: `fresh_mattersim.csv`, `rerelaxed_checked.csv`, `rerelaxed_method_metrics.csv`, `rerelaxed_novelty.csv`, and `rerelaxed_all/` (final CIFs).
The checkpoint and its pinned source are in `../data/mattersim/`. These generated data and model files stay out of Git.
For the mechanism, structures, references, and embedding plots, see [the interpretation report](selection_interpretation.md).
