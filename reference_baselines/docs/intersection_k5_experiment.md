# Reference intersection experiment: k = 5

## What we tested

We changed how we select references and score existing candidate crystals.
We did not generate new crystals or train a model for this experiment.

We tested k = 5, 10, 20, and 50. The result discussed here uses **k = 5**.
This means five nearest references per candidate, before the intersection.
It is separate from the k = 10 graph prepared for future model training.

## Method

For each candidate:

1. Find its five nearest known references using the JEPA vectors and squared
   Euclidean distance.
2. Find its original Crys-JEPA reference set. These references contain only
   elements present in the candidate; they can contain fewer elements.
3. Keep the references found in both sets. This is the intersection.
4. Use the mean squared distance to the retained references as the score.
5. If the intersection is empty, use the original Crys-JEPA score. This is the
   fallback. If the original set is also empty, leave the candidate unscored.

A lower score gives a higher rank. The intersection can contain fewer than five
references. No Laplacian calculation or trained graph model sets this score.

## How we checked it

We used 37,438 validation candidates. Each method selected 1,871 unique crystals:
5% of that set, rounded down. We removed repeated structures across batches and
continued down each ranked list until it had 1,871 unique choices.

A selected crystal passed if its corrected MatterSim energy was no more than
0.1 eV per atom above the fixed MP2023 reference hull. This is an estimated
stability check. It is not a new DFT calculation or a test of structural validity.

| Method | Unique crystals selected | Passed the stability cutoff | Fraction that passed |
| --- | ---: | ---: | ---: |
| Original Crys-JEPA | 1,871 | 727 | 38.9% |
| Intersection with fallback, k = 5 | 1,871 | 796 | 42.5% |

The gain is **69 extra passing crystals**, or **3.7 percentage points** using
the unrounded rates. It is not a 3.7% increase in the number generated.

## What this means

The result suggests that a smaller, more focused reference set can help rank
candidates. It does not prove that this rule is best. We compared several methods
on one validation split. MatterSim errors can affect the result.

The pure intersection gives a score to only 4.3% of validation candidates.
The fallback is needed to cover more candidates. Test performance remains unused.

## Next step

Train the prepared models and compare them with Crys-JEPA and this fixed rule.
Choose the settings using validation results. Then evaluate the chosen method on
the untouched test set. Use independent DFT checks before making stability claims.

## Files

- Method: [reference_sets.py](../baselines/reference_sets.py).
- Numbers: [validation_unique_baseline_metrics.csv](../results/validation_unique_baseline_metrics.csv).
- Full report: [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).

The result CSV stays local and is not committed. It can be rebuilt with the
commands in the [package guide](../README.md).
