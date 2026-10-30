# Budget study

Both rules rank the same 36,994 generated candidates. Neither rule generates crystals.
The rule and the 1° angle stay fixed. Only the number selected changes.

## Same saved MatterSim check for every budget

All numbers in this table use the same saved MatterSim-relaxed structures and energies.
A pass means corrected energy above the fixed MP2023 hull is at most 0.1 eV/atom.
Selections remove structural duplicates before taking the budget. This table counts stability passes;
it does not add the extra convergence, composition, novelty, or post-relaxation checks below.

| Budget | Selected each | Crys-JEPA passes | Angle-rule passes | Extra passes | Gain in percentage points | Relative gain |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 369 | 146 (39.6%) | 157 (42.5%) | +11 | +2.98 | +7.5% |
| 2% | 739 | 269 (36.4%) | 328 (44.4%) | +59 | +7.98 | +21.9% |
| 5% | 1,849 | 644 (34.8%) | 782 (42.3%) | +138 | +7.46 | +21.4% |
| 7% | 2,589 | 905 (35.0%) | 1,042 (40.2%) | +137 | +5.29 | +15.1% |

Descriptive 95% uncertainty ranges for the gain, from 1,000 paired chemical-system bootstrap draws:

| Budget | Gain range in percentage points |
|---|---:|
| 1% | -6.38 to +11.41 |
| 2% | +2.09 to +14.25 |
| 5% | +3.48 to +11.46 |
| 7% | +2.78 to +7.58 |

At 1%, the range includes zero: this small gain is uncertain. The other three ranges are positive.
These four ranges are descriptive and do not adjust for multiple comparisons.
From 5% to 7%, each rule receives 740 more slots. Crys-JEPA adds 261 passes; the angle rule adds 260.
Thus the extra pass count stays nearly the same (+138 to +137), while the gain in pass rate falls.

The 5% row must reproduce the frozen 644 versus 782 result. It is a different stage
from the 662 versus 808 result after extra relaxation and checks.

## Extra MatterSim relaxation: separate check

These counts also require convergence, geometry and composition checks, and remove duplicates again.
Failed or duplicate slots are not replaced. Missing checks are not treated as passes or failures.

| Budget | Crys-JEPA fresh passes | Angle-rule fresh passes | Coverage |
|---|---:|---:|---|
| 1% | 149 | 162 | 369/369 original; 369/369 angle |
| 2% | 273 | 339 | 739/739 original; 739/739 angle |
| 5% | 662 | 808 | 1,849/1,849 original; 1,849/1,849 angle |
| 7% | Pending | Pending | 1,872/2,589 original; 2,589/2,589 angle |

## How to read this

A larger budget can add more passes while lowering the pass rate. Compare both counts and rates.
Both rules get the same budget, so a difference is not due to one rule receiving more slots.
This tests screening quality. It does not show that either rule generates more crystals.
The test outcomes were already open. This is a budget study, not a new unseen test.
Do not choose a budget from this table and call its result independent validation.
The two rules still use different reference eligibility: the angle gate can use cross-chemistry neighbours.
This study does not separate that effect from the angle rule itself.

## Reproduce

```sh
reference_baselines/environment/audit/bin/python -m reference_baselines.evaluation.budget_sweep
```

Full selections, counts, bootstrap intervals and input hashes: `reference_baselines/results/budget_sweep/`.
No new model training or new relaxation runs are needed for the saved-energy table.
