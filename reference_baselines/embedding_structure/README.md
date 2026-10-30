# Embedding structure analysis

Community alignment, property probes, and space-group concentration for frozen
Crys-JEPA embeddings. See the [community study](../docs/embedding_structure_experiment.md)
and [concentration study](../docs/structure_capture_experiment.md) for results and limits.

## Layout

Paths below are relative to the repository root:

- `reference_baselines/embedding_structure/`: analysis modules and shared helpers.
- `reference_baselines/requirements/embedding_structure.txt`: dependencies.
- `tests/test_embedding_structure.py`: regression tests.
- `reference_baselines/results/embedding_structure/`: generated outputs, ignored by Git.
- `reference_baselines/docs/`: reports and selected figures.

## Run

From the repository root, with an isolated Python environment:

```sh
python -m pip install -r reference_baselines/requirements/embedding_structure.txt
python -m unittest discover -s tests -p test_embedding_structure.py -v
python -m reference_baselines.embedding_structure.spectral_community --data PATH_TO_DATA
python -m reference_baselines.embedding_structure.community_lists_and_map --data PATH_TO_DATA
python -m reference_baselines.embedding_structure.energy_analysis --data PATH_TO_DATA
python -m reference_baselines.embedding_structure.probe --data PATH_TO_DATA
python -m reference_baselines.embedding_structure.spacegroup_capture --draws 1000 --seed 2026
python -m reference_baselines.embedding_structure.capture_confound
```

`PATH_TO_DATA` contains aligned `embeddings.npy` and `metadata.csv` files.
Every analysis accepts `--out`; use the same output directory throughout.
Community and map runs default to k=10 and seed=42; pass matching values to both.
The optional `--spectrum` diagnostic checks residuals and makes no cluster-count claim.

## Conventions

### Energy-controlled concentration

After generating `crystals_by_community.csv`, run on CPU (no GPU required):

```sh
python -m reference_baselines.embedding_structure.energy_capture --data PATH_TO_DATA --draws 1000 --seed 2026
python -m unittest discover -s tests -p test_energy_capture.py -v
```

Only `metadata.csv` and the saved community table are needed; embeddings and maps
are not recomputed. `--out` must point to the directory containing the community
table if it is not in the default location. Rows are aligned by material ID.

The primary width is 0.01 eV/atom, with 0.025 and 0.05 as sensitivity checks
(`--widths 0.01 0.025 0.05`). Bins have a fixed zero origin. Permutations stay
inside each energy bin and reselect the best five communities on every draw.
All energies must be finite. A fresh unrestricted baseline is computed too.

Outputs are saved under `results/embedding_structure/energy_capture/`: a baseline
CSV, a concentration CSV and bin diagnostics per width, and `summary.json`.
Each group includes its fraction of crystals in bins with multiple space groups
and communities. Low fractions indicate limited mixing; no significance is not
proof of an absent structural signal. Report all widths rather than selecting
the most favourable result. BH adjustment is applied separately at each width.

Remaining concentration supports an association beyond this binned-energy null,
not full independence from energy or composition. `--draws 20` is useful for a
smoke run only; use 1,000 or more for the reported analysis.

### Graph and probe conventions

Graph weights are `exp(-distance**2/sigma**2)`, with sigma equal to the median
of all retained neighbour distances. This differs from `graphs/build.py`, which
uses kth-neighbour distances and a factor of two. Self-neighbours are excluded
by identity; exact boundary ties can depend on the search backend.

Concentration uses size-matched permutations and no captured/not-captured labels.
Formula parsing uses pymatgen. Null intervals are not effect confidence intervals.
Probes use a random row split, not a chemistry-held-out split.

## Outputs

JSON summaries record settings, hashes and versions. CSVs contain community and
space-group statistics; arrays contain graphs, projections and split indices.
Keep these generated files local. Commit source, tests, requirements, reports,
and only the figures referenced by those reports.
