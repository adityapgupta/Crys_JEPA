# Reference pools and mean weighting

## What was actually tested before

The reported 1° rule did not use a weighted mean. It first prioritizes candidates with any known
reference within 1°, then ranks them by the original plain chemical-reference mean.
It changes candidate ranking. It does not replace that mean with new reference weights.

The paper's screening score is a uniform mean of squared embedding distances (Eq. 11, section 4.2).
Its energy-weighted training loss is separate from this screening score.
Source: [Crys-JEPA paper](https://arxiv.org/html/2605.14759v2#S4.SS2).

## Fair controls

We change the reference pool and averaging rule separately. All scores average the same raw squared
Euclidean embedding distance. Lower is better. We do not average embeddings before computing distance.

- Original pool: all known MP20 references within the candidate chemical system and its subsystems.
- Enhanced pool: original pool plus five global cosine neighbours, with duplicate reference IDs removed.
- Intersection control: only those five neighbours allowed by chemistry; if empty, use the original pool.
- Neighbours-only control: five global cosine neighbours, without the original pool.
- Angle-pool control: references within 1° among the nearest ten; if empty, use the original pool.
- Candidate-gate control: keep the original pool and change candidate priority using the 1° gate.

Each pool gets both a plain mean and a weighted mean. The gate also gets both means.
Weighted mean uses w = exp(-(d - smallest d in the pool) / 10), then score = sum(w*d)/sum(w).
Here d is squared embedding distance. Nearby references get more weight; no candidate energies enter the weights.
The scale 10 is the existing chemical_softmax_t10 setting. It was fixed for this study, not tuned on these outcomes.
This tests one stated weighting rule. It does not cover all possible weights or physical phase fractions.

All rules receive the same candidates, MP20 bank, budget, duplicate checks, fixed MP2023 hull and saved MatterSim
energies. A stability pass requires energy above hull <= 0.1 eV/atom. Extra re-relaxation counts are not mixed in.

## Main result at 5%

| Reference pool | Plain mean passes | Weighted mean passes |
|---|---:|---:|
| Original chemical references | 644 | 692 |
| 1° references, with chemical fallback | 785 | 793 |
| Original plus five cosine neighbours | 682 | 648 |

Weighting alone adds 48 passes (644 to 692). The angle-reference pool with a plain mean adds 141
(644 to 785). Holding weighting fixed, the angle pool adds 101 (692 to 793).
Thus this weighting rule does not explain the full angular-screening gain.

However, adding five cosine neighbours to the original pool gives only 38 extra passes with a plain mean
and loses 44 passes when compared with the weighted original pool. Its uncertainty ranges include zero.
We cannot claim that simply enlarging the reference set reliably improves screening.

The angle-reference plain mean and the reported candidate gate share 1,783 of 1,849
selections (96.4%). Their passing counts differ by only three (785 versus 782).
This still leaves angular candidate screening as a strong explanation. These results alone do not prove
that better reference identities, rather than the gate and smaller pools, cause the gain.

## Test results: passing unique candidates

| Reference pool and mean | 1% | 2% | 5% | 7% |
|---|---:|---:|---:|---:|
| Original references, plain mean | 146/369 (39.6%) | 269/739 (36.4%) | 644/1,849 (34.8%) | 905/2,589 (35.0%) |
| Original references, weighted mean | 142/369 (38.5%) | 289/739 (39.1%) | 692/1,849 (37.4%) | 961/2,589 (37.1%) |
| Original + five cosine neighbours, plain mean | 144/369 (39.0%) | 274/739 (37.1%) | 682/1,849 (36.9%) | 948/2,589 (36.6%) |
| Original + five cosine neighbours, weighted mean | 147/369 (39.8%) | 271/739 (36.7%) | 648/1,849 (35.0%) | 885/2,589 (34.2%) |
| Chemical intersection, plain mean | 177/369 (48.0%) | 333/739 (45.1%) | 725/1,849 (39.2%) | 977/2,589 (37.7%) |
| Chemical intersection, weighted mean | 178/369 (48.2%) | 332/739 (44.9%) | 739/1,849 (40.0%) | 990/2,589 (38.2%) |
| Five cosine neighbours only, plain mean | 169/369 (45.8%) | 334/739 (45.2%) | 736/1,849 (39.8%) | 963/2,589 (37.2%) |
| Five cosine neighbours only, weighted mean | 169/369 (45.8%) | 334/739 (45.2%) | 736/1,849 (39.8%) | 963/2,589 (37.2%) |
| 1° reference pool, plain mean | 204/369 (55.3%) | 373/739 (50.5%) | 785/1,849 (42.5%) | 1,044/2,589 (40.3%) |
| 1° reference pool, weighted mean | 204/369 (55.3%) | 373/739 (50.5%) | 793/1,849 (42.9%) | 1,063/2,589 (41.1%) |
| 1° candidate gate + original plain mean (reported rule) | 157/369 (42.5%) | 328/739 (44.4%) | 782/1,849 (42.3%) | 1,042/2,589 (40.2%) |
| 1° candidate gate + original weighted mean | 186/369 (50.4%) | 337/739 (45.6%) | 791/1,849 (42.8%) | 1,061/2,589 (41.0%) |

## Validation results: passing unique candidates

| Reference pool and mean | 1% | 2% | 5% | 7% |
|---|---:|---:|---:|---:|
| Original references, plain mean | 176/374 (47.1%) | 321/748 (42.9%) | 727/1,871 (38.9%) | 1,033/2,620 (39.4%) |
| Original references, weighted mean | 161/374 (43.0%) | 324/748 (43.3%) | 735/1,871 (39.3%) | 1,026/2,620 (39.2%) |
| Original + five cosine neighbours, plain mean | 170/374 (45.5%) | 320/748 (42.8%) | 765/1,871 (40.9%) | 1,048/2,620 (40.0%) |
| Original + five cosine neighbours, weighted mean | 140/374 (37.4%) | 270/748 (36.1%) | 678/1,871 (36.2%) | 928/2,620 (35.4%) |
| Chemical intersection, plain mean | 197/374 (52.7%) | 348/748 (46.5%) | 798/1,871 (42.7%) | 1,078/2,620 (41.1%) |
| Chemical intersection, weighted mean | 197/374 (52.7%) | 340/748 (45.5%) | 777/1,871 (41.5%) | 1,051/2,620 (40.1%) |
| Five cosine neighbours only, plain mean | 175/374 (46.8%) | 329/748 (44.0%) | 769/1,871 (41.1%) | 1,027/2,620 (39.2%) |
| Five cosine neighbours only, weighted mean | 175/374 (46.8%) | 329/748 (44.0%) | 769/1,871 (41.1%) | 1,026/2,620 (39.2%) |
| 1° reference pool, plain mean | 218/374 (58.3%) | 387/748 (51.7%) | 857/1,871 (45.8%) | 1,134/2,620 (43.3%) |
| 1° reference pool, weighted mean | 218/374 (58.3%) | 387/748 (51.7%) | 849/1,871 (45.4%) | 1,129/2,620 (43.1%) |
| 1° candidate gate + original plain mean (reported rule) | 165/374 (44.1%) | 331/748 (44.3%) | 857/1,871 (45.8%) | 1,134/2,620 (43.3%) |
| 1° candidate gate + original weighted mean | 162/374 (43.3%) | 346/748 (46.3%) | 849/1,871 (45.4%) | 1,129/2,620 (43.1%) |

## What each change adds: test at 5%

These are changes in passing counts at the same 1,849-candidate budget.

| One change at a time | Extra passes | Gain in percentage points | Descriptive 95% range |
|---|---:|---:|---:|
| Weight original references; same pool | +48 | +2.60 | -1.58 to +6.67 |
| Add five cosine neighbours; keep plain mean | +38 | +2.06 | -1.83 to +5.87 |
| Add five cosine neighbours; keep weighted mean | -44 | -2.38 | -5.84 to +1.52 |
| Weight enhanced pool; same enhanced pool | -34 | -1.84 | -7.00 to +2.86 |
| Use 1° reference pool; keep plain mean | +141 | +7.63 | +3.59 to +11.63 |
| Use 1° reference pool; keep weighted mean | +101 | +5.46 | +2.78 to +8.21 |
| Add 1° candidate gate; keep plain chemical mean | +138 | +7.46 | +3.48 to +11.46 |
| Add 1° candidate gate; keep weighted chemical mean | +99 | +5.35 | +2.68 to +8.13 |
| Angle-reference pool versus candidate gate; plain means | +3 | +0.16 | +0.03 to +0.32 |
| Angle-reference pool versus candidate gate; weighted means | +2 | +0.11 | +0.00 to +0.27 |

## Interpretation and limits

The original weighted mean tests whether weighting alone explains the gain.
The enhanced plain mean tests whether adding neighbours helps without weighting.
The enhanced weighted mean versus original weighted mean tests the added pool under the same weighting rule.
The interaction in contrasts.csv shows whether the two changes add independently; do not add their gains blindly.
The angle candidate gate is a different operation from changing a reference pool. Its result must be named separately.
Global cosine neighbours may have different elements. The intersection control keeps chemistry fixed.
This is a screening attribution study. It does not prove better physical decomposition references or better generation.
Both validation and test outcomes were already inspected. All these comparisons are exploratory, not a new unseen test.
Bootstrap ranges are descriptive and unadjusted for multiple comparisons. No winner or temperature is chosen here.

## Reproduce

Use reference_baselines/environment/audit/bin/python for these commands:

```sh
python -m reference_baselines.baselines.mean_ablation
python -m reference_baselines.evaluation.mean_ablation
python -m reference_baselines.reporting.mean_ablation
```

Scores, exact selection IDs, pool sizes, fallbacks, contrasts, and input hashes are saved under
reference_baselines/results/mean_ablation/ (local, Git-ignored). No new training or relaxation is run.
