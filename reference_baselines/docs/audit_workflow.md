# Interpretation and fresh MatterSim workflow

This work explains the fixed selections and checks them again. It does not train
a model or change the original rankings. Generated data and checkpoints stay in
the visible, Git-ignored `data/` and `results/` folders.

## Environment

Use the prepared scientific environment in `reference_baselines/environment/`.
The audit packages live in a separate environment, so their dependencies do not
replace that environment's NumPy, PyTorch, or Pymatgen installation.

```sh
reference_baselines/environment/bin/python -m venv --system-site-packages reference_baselines/environment/audit
reference_baselines/environment/audit/bin/python -m pip install --no-deps -r reference_baselines/requirements/audit_inference.txt
```

The audit environment also needs the parent environment's scientific packages.
Create this path file once, from the repo root:

```sh
reference_baselines/environment/audit/bin/python - <<'PY'
from pathlib import Path
import site
parent = next(Path('reference_baselines/environment').glob('lib/python*/site-packages')).resolve()
(Path(site.getsitepackages()[0]) / 'baseline_environment.pth').write_text(str(parent) + '\n')
PY
```

The requirements file lists the small inference dependency subset used here.
SMACT14 rules are explicit. SMACT 2.7's general dependency bounds are not used to
replace the working scientific stack. The exercised paths pass the tests below.

## Data and examples

Run these with the audit Python, in order:

```sh
python -m reference_baselines.evaluation.selection_audit
python -m reference_baselines.graphs.embedding_audit
python -m reference_baselines.structure.validity
python -m reference_baselines.structure.case_study
python -m reference_baselines.datasets.fetch_mattersim
```

`fetch_mattersim` downloads only the pinned official 1M checkpoint and checks its
SHA256. No training data or optimiser steps are needed.

## Fresh MatterSim

```sh
python -m reference_baselines.evaluation.mattersim_audit
```

This checks energies and forces for all 3,243 crystals in the two fixed test
selection lists. It uses one graph per inference batch. A regression test checks
energy, force, and input-order agreement with the public ASE calculator. The
multi-graph legacy batching attempt was rejected and its outputs are not used.

For complete atom-and-cell re-relaxation, run four disjoint shards. Each saves a
resume table after every completed crystal. Use one CPU thread per process.

```sh
for shard in 0 1 2 3; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m reference_baselines.evaluation.mattersim_audit \
    --rerelax-all --relax-steps 300 --threads 1 --shard "$shard" 4 &
done
wait
python -m reference_baselines.evaluation.mattersim_audit --tighten-forces
python -m reference_baselines.evaluation.rerelaxed
```

The force refinement checks any cell-filter-converged structure whose actual
atomic force remains above 0.05 eV/angstrom. It applies to both methods, does not
use energy to choose retries, and keeps the total 300-step limit. Final accepted
convergence requires both atomic and filter force norms at most 0.05.

The evaluator refuses incomplete shards or force-criterion violations. It
rechecks duplicates and MP2023 novelty after relaxation. It keeps the original
selection order and budget. It does not refill failed or duplicate slots.

## Reports and checks

```sh
MPLCONFIGDIR=reference_baselines/results/matplotlib python -m reference_baselines.reporting.interpretation
python -m reference_baselines.reporting.mattersim_validation
MPLCONFIGDIR=reference_baselines/results/matplotlib python -m unittest discover -s tests -v
```

Reports are in `reference_baselines/docs/`. Full tables, source IDs, CIFs, and
provenance are in `reference_baselines/results/interpretation/`. The original
fixed test artifacts in `results/heldout/` remain unchanged.
