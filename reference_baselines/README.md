# Reference-selection baselines

This folder contains reference-selection experiments, data checks, and training
code. Full model training has not started. See
[experiment report](docs/EXPERIMENT_REPORT.md) for current results and limits.

## Layout

```text
reference_baselines/
  core/          Shared retrieval and file helpers
  datasets/      Downloads, stability labels, and training data
  graphs/        Graph construction and analysis
  baselines/     Fixed reference-selection methods and screening
  structure/     Geometry descriptors and comparisons
  evaluation/    Selection metrics, duplicates, and label checks
  models/        Neural model definitions
  training/      Training and prediction commands
  checks/        Checks on real data and model behaviour
  reporting/     Report and readiness generation
  configs/       Fixed training settings
  requirements/  Python dependencies
  docs/          Experiment report
  paths.py       Shared input and output locations
  data/          Downloaded data (local, not staged)
  results/       Generated results and logs (local, not staged)
```

Run commands as Python modules from the repository root. This gives every module
the same package imports. `paths.py` keeps data and output locations independent
of the source file's folder. The original numbered pipeline commands remain in
the repository root. `V_screen_precomputed.py` calls `baselines/screening.py` for
compatibility; the screening implementation lives in the package.

Layout regression checks are in `tests/test_reference_layout.py`:

```sh
reference_baselines/environment/bin/python -m unittest discover -s tests -p test_reference_layout.py -v
```

## Run

From the repository root:

```sh
python3 -m reference_baselines.graphs.build
python3 -m reference_baselines.graphs.analysis
python3 -m reference_baselines.checks.retrieval
python3 -m reference_baselines.baselines.energy
```

Dependencies: NumPy, pandas, SciPy, and PyTorch. These are already available in the
current environment. No GPU, embedding model, or new embeddings are required.
`graphs/build.py` accepts `--data-dir`, `--out-dir`, `--batch-size`, and `--threads`.
The other scripts accept `--data-dir` and `--out-dir`.

## Definitions

- Input: `embeddings/data/`, with the two embedding arrays and matching CSV files.
- One graph node is one known reference crystal. Row IDs are zero-based CSV row
  positions, not atomic indices or material IDs.
- Search checks every reference using squared Euclidean distance in float64.
  Embeddings are not normalized or changed. Equal distances use the smaller
  reference row ID first.
- Each reference selects its nearest 50 other references. For k = 5, 10, 20, and
  50, use the first k columns. Self-connections are excluded; distinct crystals
  with identical embeddings remain eligible.
- Each candidate independently retrieves its nearest 50 known references. No
  candidate-to-candidate connections are created. All candidates are searched;
  summaries distinguish all candidates from those marked valid, unique, novel.
- Saved graph edges are bidirectional: retain an edge if either endpoint selects
  the other. Thus a node can have more than k neighbours. Directed k-NN lists are
  also saved and are used for neighbour-level statistics.
- Graph weights are `exp(-squared_distance / (2 * sigma**2))`. Each graph uses
  `sigma = median(kth-neighbour distance)` from known reference crystals. Graph
  weights do not change nearest-neighbour selection.
- Crys-JEPA eligibility means the reference's elements are a subset of the
  candidate's elements. It includes the exact system and all subsystems. Exact
  system matches are counted separately.

## Results

All outputs are in `reference_baselines/results/` by default. This folder is not
hidden and is not excluded by this implementation's ignore rules.

| File | Contents |
| --- | --- |
| `reference_knn_indices.npy` | N x 50 reference row IDs; self excluded |
| `reference_knn_squared_distances.npy` | N x 50 matching squared distances |
| `candidate_knn_indices.npy` | M x 50 retrieved reference row IDs |
| `candidate_knn_squared_distances.npy` | M x 50 matching squared distances |
| `reference_graph_k*.npz` | Four symmetric, weighted SciPy sparse matrices |
| `reference_rows.csv`, `candidate_rows.csv` | Metadata with explicit array row IDs |
| `candidate_reference_overlap.csv` | Original pool size and overlap at each k; also distance summaries |
| `most_retrieved_references.csv` | Top ten frequently selected references for each k and query population |
| `analysis.json` | Connectivity, incoming-edge concentration, chemistry overlap, and energy-gap statistics |
| `provenance.json` | Input/code hashes, versions, search settings, timing, and missing batch IDs |
| `verification.json` | Independent direct-distance, graph, row-ID, and chemistry-set checks |

For example, `candidate_knn_indices[10, :5]` gives the five selected reference
rows for candidate row 10. Look up those reference rows in `reference_rows.csv`.

## How to read the analysis

Overlap fraction is `number of chemically eligible retrieved references / k`.
Recall is `overlap / original reference count`, averaged only where that count is
positive. Jaccard overlap is `overlap / (original count + k - overlap)`.
An empty intersection is reported as empty; there is no silent fallback.
The original reference pool is not replaced or edited.

Random-overlap expectations assume uniform reference sampling. For known-crystal
queries, the same-system expectation excludes self. Random formation-energy gaps
use 100,000 uniformly sampled distinct ordered reference pairs, seed 1234.
These controls help interpret small overlap percentages; they do not establish
physical reference relevance.

Formation-energy gaps describe known-to-known edges only. Candidate metadata has
no energy labels. Similar formation energies do not imply similar stability or
structure. Connectivity and frequent retrieval are graph diagnostics, not proof
that a reference is good. Candidate outcome checks are described below.

Batch 84 is missing from the supplied candidate data. Matching row counts cannot
verify that the source structures, checkpoints, and scalers were aligned correctly.
The verification checks sampled exact distances, all stored neighbour IDs/order,
sampled graph weights, and sampled chemical reference intersections.

The initial k-NN baseline is followed by spectral analysis, outcome comparisons,
and model preparation below.

## Energy-only comparison

`baselines/energy.py` uses the 27,136 known crystals. It retrieves k = 5, 10, 20,
and 50 neighbours using absolute formation-energy differences alone. It excludes
self and resolves equal gaps by reference row ID. This checks how much the
JEPA reference sets overlap with a simple energy-only rule.

It also samples ten random controls per JEPA edge from the same absolute
formation-energy-gap bin (width 0.01 eV/atom). Sampling is uniform within the
bin, with replacement, and excludes the query crystal. This is a closer control
than sampling from the whole database. Each query's controls have approximately
the same energy differences as its JEPA neighbours.

The script compares chemical-system agreement, space-group agreement, and
band-gap differences for JEPA, energy-only, and energy-gap-matched retrieval.
Space-group agreement is only a weak structural clue. It does not establish
similar atomic arrangements. Band gaps are recorded properties, not structure
labels. These checks cannot establish stability prediction performance.

The primary descriptive comparison is k = 10. Its uncertainty intervals use
500 paired bootstrap resamples of query rows. Graph edges share nodes, and the
encoder may have seen these crystals during pretraining; these intervals are
not evidence of generalization to unseen chemistry. Gap matching is approximate
within bins. The report includes control-pool sizes to identify limited controls.

Outputs in `results/`:

- `energy_knn_indices.npy`: nearest reference rows under the energy-only rule.
- `energy_knn_absolute_gaps.npy`: matching energy gaps in eV/atom.
- `energy_comparison_per_reference.csv`: overlap and gap statistics per query.
- `energy_comparison.json`: comparisons, settings, input hashes, and verification.

The script verifies energy neighbours against a separate full-sort calculation
for 128 queries and a fixture with exact ties and tiny gaps. It checks all random
controls for self-exclusion and the requested energy-gap bounds. Options include
`--bin-width`, `--control-draws`, `--seed`, `--threads`, `--data-dir`, and `--out-dir`.

No candidate structures, candidate energy labels, or learned models are added
by this comparison. An independent atomic-structure comparison remains a later
step that requires the original structures.

### Initial results at k = 10

These results use all 27,136 known crystals, with self excluded.

| Check | JEPA neighbours | Energy-only neighbours | Energy-gap-matched random controls |
| --- | --- | --- | --- |
| Mean formation-energy gap (eV/atom) | 0.13535 | 0.00049 | 0.13545 |
| Same chemical system | 1.74% | 0.63% | 0.18% |
| Same space group | 21.23% | 5.37% | 5.16% |
| Mean band-gap difference (eV) | 0.779 | 0.887 | 0.902 |

JEPA and energy-only top-ten sets overlap by 0.95% on average. Different neighbour
sets alone do not show that JEPA captures useful extra information. However,
higher chemistry and symmetry agreement, and smaller band-gap differences than
the energy-gap-matched controls, provide evidence of additional patterns in this
dataset. This is not proof of structural similarity, stability, or performance on
new crystals. The independent structure check below tests the geometry directly.

## Independent structure check

The MP-20 reference structures can be downloaded from the paper's public dataset.
`datasets/fetch_structures.py` pins the dataset revision and verifies the download hash.
The source contains all 27,136 reference IDs. Before using a structure, the code
matches its material ID and checks all shared metadata fields against our CSV.
It also checks the formula and chemical system parsed from each CIF.

Run from the repository root:

```sh
python3 -m venv --system-site-packages reference_baselines/environment
reference_baselines/environment/bin/python -m pip install -r reference_baselines/requirements/structure.txt
reference_baselines/environment/bin/python -m reference_baselines.datasets.fetch_structures
reference_baselines/environment/bin/python -m reference_baselines.structure.descriptors
reference_baselines/environment/bin/python -m reference_baselines.structure.comparison
```

The separate environment avoids changes to the existing Python environment.
Only this generated environment and Python bytecode are excluded from Git in
this folder. All authored code, downloaded source data, and result files remain
visible under `reference_baselines/`.

### Descriptor and controls

The structure check uses SOAP, a descriptor of local atomic arrangements. It is
calculated directly from the CIF, without JEPA or energy labels. All atoms receive
the same label. Thus the check measures geometry without requiring the same
elements. The source CIF is not changed.

Two predefined profiles are used:

- `physical`: original lengths; cutoff 5 angstrom, Gaussian width 0.3 angstrom.
- `density_normalized`: divide all lengths by the cube root of volume per atom;
  cutoff 3 and Gaussian width 0.15 in the resulting relative length units.

Both profiles use periodic boundaries, six radial functions, angular degree up
to six, and an average over site power spectra. The 147-feature output is
normalized to unit length. Distance is one minus the vector dot product. A smaller
distance means closer fingerprints. This is a cosine comparison of DScribe's
outputs; it is not an exact crystal equivalence test.

The primary check is the density-normalized profile at k = 10. Original-length
results and k = 5, 20, 50 are secondary checks. We compare JEPA neighbours,
energy-only neighbours, and ten random references per edge from the same 0.01
eV/atom energy-gap bin. These are the same control rules as `baselines/energy.py`.
For the separate different-chemical-system comparison, both the JEPA neighbour
and retained random controls must have a different chemical system from the query.

Every primary comparison uses complete query rows shared by all three methods.
Failures are recorded, not imputed. The output reports the number of excluded
queries. Confidence intervals use 500 paired resamples of query rows. They do
not establish performance on unseen crystals or remove dependencies between
queries that share neighbours.

### Checks and output files

Descriptor checks cover translation, rotation, atom order, supercell replication,
element relabeling, and density-normalized scale invariance. A separate check
confirms that bcc and fcc arrangements are distinguished. Comparison distances
are checked against scalar dot products for sampled pairs.

| File | Contents |
| --- | --- |
| `data/mp_20_train.csv.gz`, `data/source.json` | Original CIFs and pinned source/hash |
| `results/soap_physical.npy` | N x 147 original-size fingerprints |
| `results/soap_density_normalized.npy` | N x 147 size-normalized fingerprints |
| `results/structure_fingerprint_status.csv` | Parse checks, atom counts, warnings, and failures per reference |
| `results/structure_fingerprint_provenance.json` | Descriptor settings, package versions, source hashes, and checks |
| `results/structure_comparison.json` | Overall and different-system comparison results |
| `results/structure_comparison_per_reference.csv` | Per-query k = 10 results |
| `results/structure_pair_examples.csv` | Close and distant cross-system pairs for inspection |

Pair examples are deliberately selected extremes, not a representative sample.
Averaged local fingerprints can miss differences in long-range arrangement.
Geometry similarity does not establish similar bonding, stability, or reference
usefulness for candidate screening.

Sources:

- [DScribe SOAP documentation](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html)
- [De et al., Comparing molecules and solids across structural and alchemical space](https://arxiv.org/abs/1601.04077)
- [Paper dataset](https://huggingface.co/datasets/liun-online/Crys_JEPA)

### Structure results at k = 10

All 27,136 reference structures passed parsing, composition, and fingerprint
checks. No query was excluded from the primary comparison.

| Geometry profile | JEPA mean distance | Energy-only mean distance | Energy-gap-matched random mean distance |
| --- | --- | --- | --- |
| Density-normalized (primary) | 0.004973 | 0.007510 | 0.007629 |
| Original physical size | 0.065045 | 0.093453 | 0.095290 |

Lower means closer under this descriptor. In the primary check, JEPA has a lower
mean distance than matched random controls for 82.7% of query crystals. For
different-chemical-system pairs alone, the primary distances are 0.005067 for
JEPA and 0.007636 for matched random controls (27,107 query crystals with eligible
pairs). The direction also holds when original physical size is retained.

This supports a local-geometry signal beyond the approximately matched energy
gaps in these known crystals. It does not establish identical structures,
stability, or a better reference set for generated candidates.

## Outcome experiments and training preparation

The experiment report contains the results, scientific limits, and readiness
decision. All code and outputs are visible in this folder. Data files are large;
they are local artifacts, and no Git commit or upload has been made.

Use the environment installed above. From the repository root, run these steps
in order (the two downloads require network access):

```sh
reference_baselines/environment/bin/python -m pip install -r reference_baselines/requirements/structure.txt
reference_baselines/environment/bin/python -m reference_baselines.datasets.fetch_release --extract-base-eval
reference_baselines/environment/bin/python -m reference_baselines.datasets.fetch_hull
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 reference_baselines/environment/bin/python -m reference_baselines.datasets.labels --stage entries
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 reference_baselines/environment/bin/python -m reference_baselines.datasets.labels --stage hull
reference_baselines/environment/bin/python -m reference_baselines.baselines.reference_sets
reference_baselines/environment/bin/python -m reference_baselines.graphs.spectral
reference_baselines/environment/bin/python -m reference_baselines.checks.outcomes
reference_baselines/environment/bin/python -m reference_baselines.evaluation.baselines
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 reference_baselines/environment/bin/python -m reference_baselines.evaluation.uniqueness
reference_baselines/environment/bin/python -m reference_baselines.evaluation.comparisons
reference_baselines/environment/bin/python -m reference_baselines.datasets.training
reference_baselines/environment/bin/python -m reference_baselines.checks.training
reference_baselines/environment/bin/python -m reference_baselines.evaluation.label_quality
reference_baselines/environment/bin/python -m reference_baselines.reporting.report
```

`baselines/reference_sets.py` also needs the initial screening scores. If rebuilding
from scratch, run this before that script:

```sh
python3 -m reference_baselines.baselines.screening --data-dir embeddings/data --out-dir screen_out
```

The hull builder uses standard phase diagrams for up to four elements and
composition-constrained linear programming for larger systems or diagram
failures. Negative distances to the fixed hull are retained. No labels or
performance metrics from the locked test split are used for method selection.

The baseline grid includes original Crys-JEPA, unrestricted kNN, chemical-only
kNN, intersection, intersection with explicit fallback, and deduplicated union,
at k=5/10/20/50 where applicable. Three chemical softmax temperatures and random
reference retrieval are also included. Two reference-free controls use vector
norm and global-centroid distance. The MatterSim hull oracle uses the labels
themselves and is only a diagnostic ceiling.

Global uniqueness is checked across all validation batches. Baseline and learned
model metrics share `evaluation/metrics.py`. Both fixed raw-selection budgets and
fixed unique-selection budgets are reported. The original per-turn selection
protocol and thresholds 0/0.05/0.1 eV/atom are separate sensitivity checks.

## Start model training

Preparation checks perform forward/backward passes but no optimizer steps. Full
training has not been run. The first two comparisons should use candidate-only
and reference-attention models, then GCN, GraphSAGE, MPNN, and graph transformer.
Use the full settings grid in `configs/training.toml` for the comparison.

Example command for one training run:

```sh
reference_baselines/environment/bin/python -m reference_baselines.training.train --model candidate_only --seed 2026 --learning-rate 0.001
```

Replace the model name with `reference_attention`, `gcn`, `graphsage`, `mpnn`, or
`graph_transformer`. Defaults are union pool, k=10, hidden width 128, two graph
layers, batch size 16, AdamW, and at most 50 epochs. Training stops after five
epochs without a better unique validation yield at the 5% budget. All models use
the same split and target. Existing run directories are never overwritten. CPU
is supported; CUDA is used when available. MPS and GPU execution were not tested.

After training, export reference weights using the actual checkpoint path:

```sh
reference_baselines/environment/bin/python -m reference_baselines.training.predict PATH_TO_BEST_PT --split validation --limit 128
```

Use `--limit 0` for the full validation split and `--uniform-readout` for the
fixed-checkpoint uniform-weight ablation. Candidate-only models export candidate
probabilities without reference weights. The CLI does not open test performance.

Attention weights express learned context, not physical decomposition fractions.
Graph layers operate on a per-query induced graph of known reference crystals.
They do not train an atomic crystal encoder or alter the JEPA embeddings.
