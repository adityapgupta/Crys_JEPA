# Reference-set comparison

## Independent proxy experiment

The staged experiment compares chemical-subsystem references, global kNN
(`q=5,10,15,25`), and their deduplicated union using the same JEPA mean squared
distance. It selects the top 1%, 5%, and 10% of eligible candidates within each
ten-folder group, with per-folder diagnostics and a fixed random control.
MEGNet and eSEN evaluate the combined selections against separate model-native
fixed MP2023 phase banks. Generated candidates never enter those hulls.

Start with [COMMANDS_TO_EXECUTE.md](COMMANDS_TO_EXECUTE.md). The configuration
is [experiment.yml](experiment.yml); `pipeline.py --help` lists stages. This
workflow reuses the released MatterSim-relaxed geometries and the existing
MP-20 reference export. It does not run a new relaxation or train JEPA.

Stages: `prepare -> embed -> score -> select`, then `fetch-hull -> phases`,
`download-model -> smoke-model -> predict -> hull` for each proxy, and `report`.
The phase import depends on all eligible chemistry rather than the selected
methods; every polymorph is retained until model prediction. Cached failed
reference predictions make affected hulls explicitly incomplete.

Reports include requested-budget yield bounds, precision on valid labels,
coverage, threshold sensitivity, overlap, baseline-only/alternative-only
precision, paired chemical-system intervals, model consensus, diversity,
duplicates, and timing. Full-population recall and correlations are omitted
because the proxies only evaluate selected candidates and controls.

To analyse one completed proxy while the other is running, use
`pipeline report --model megnet` (or `--model esen`). This requires only that
model's completed hull and writes to `outputs/<run>/reports/<model>/`.
The default `pipeline report` still requires both models and writes the combined
report, including consensus, to `outputs/<run>/reports/`. Single-model reports
omit cross-proxy consensus and do not change the experiment configuration.

Direct dependency versions are pinned in three separate requirements files.
Actual installed locks are exported on SOC; only rename them `.tested.txt`
after the corresponding real-weight smoke test passes. Local tests exercise
real JEPA on untrained fixtures and synthetic external predictions, not the
public MEGNet/eSEN checkpoints. No production GPU result is implied by these
tests.

Local verification on 24 October 2026: nine proxy-comparison tests and six
reference-export tests passed. The real sampled release input check accepted
6,258 of 8,113 rows (1,855 excluded by VUN). A broader repository test discovery
also ran but could not complete because the temporary local test environment
lacks `lmdb`, `scikit-learn`, and `dscribe` used by older analysis modules.
The actual public MP2023 LMDB and external model checkpoints remain SOC checks.

The MEGNet adapter uses a scoped PyTorch allowlist for the checkpoint's serialized
`matgl.data.transformer.Normalizer`, retaining restricted weights-only loading.
A serialization regression verifies preserved normalization parameters, cleanup
of the temporary allowlist, and rejection of unrelated classes. The cluster's
actual model smoke test is still required after uploading this compatibility fix.

The persistent vector store uses memory-mapped raw reference vectors, norms,
row metadata, and checksums with the existing deterministic float64 exact
retrieval. No external vector-database service is needed. MEGNet uses explicit
PyG batches; eSEN's public ASE calculator runs structures sequentially inside
checkpointed chunks. `prediction_batch_size` bounds cache commit chunks too.

Candidate `.pt` files and the reference LMDB use Python pickle. Load only the
official, checksum-verified release. Model weights are pinned on first download
and verified against their receipt on subsequent runs. Environment versions,
structure hashes and adapter source are included in prediction identities.

Use a separate output directory after changing scientific configuration. Changes
to device or throughput batch sizes are permitted on resumption. Run a model's
smoke test again after changing its environment. Do not run two copies of the
same model prediction stage against the same run/cache simultaneously.

`inspect_train.ipynb` downloads the pinned Hugging Face MP-20 train CSV, loads it,
and prints `head()`. The original local MatterGen CSV can still be selected.

`generate_embeddings.py` uses the published Crys-JEPA weights to export one
embedding per training row. It does not retrain JEPA or the generator.

## Environment

Run commands from the repository root. Install the appropriate PyTorch build for
your machine first; the repository baseline is PyTorch 2.6.0. Then install:

```sh
python -m pip install -r reference_baselines/reference_set_comparison/requirements.txt
```

The notebook needs pandas and a Jupyter kernel. Inference uses the existing JEPA
implementation. MatterSim, torch_scatter, W&B, and the full generative pipeline
are not needed. On SOC, environment setup, package installation, downloading,
scaler construction, and inference must run in a Slurm compute allocation.

## Data, weights, and reconstructed scaler

Inputs come from the [author's HF release](https://huggingface.co/datasets/liun-online/Crys_JEPA/tree/main),
pinned to revision `ad9bdac5cfd49188174617a2c11de48baad1d329`:

- `data/base/mp_20/train.csv.gz`: MP-20 reference crystals, kept in CSV order.
- `data/jepa/mp.csv.gz` and `data/jepa/mptrj.csv.gz`: sources for scaler fitting.
- `exp_logs/jepa/saved_model/model_1972.pt` inside `exp_logs.zip`: open weights.

Dataset files are checked against their published LFS SHA-256 hashes. The
checkpoint downloader reads only the roughly 291 MB compressed checkpoint member
using checked HTTP ranges, rather than downloading the full 7.39 GB ZIP. ZIP CRC
and a cached SHA-256 receipt are checked. All downloads stay under `artifacts/`.

The original JEPA mean/std scaler is not supplied in the public release. Following
the supplied `embed.py`, the default is to reconstruct an **approximate scaler**
from up to 20,000 crystals sampled from each JEPA pretraining file, seed 0. The
scaler is fitted independently of `--limit` and the MP-20 reference CSV. It is
saved once under `artifacts/scalers/` and reused. The receipt records sampled
source rows, source hashes, failures, package versions, and preprocessing.
Scaler-fit failures are recorded; embedding-export failures abort without silently
skipping reference rows. Zero/nonfinite standard deviations are rejected.

`--scaler-cap 0` uses every pretraining row and costs more time. Changing the cap
or seed selects a separate cache. Even full-data reconstruction is not verified
against the original scaler. **These are pretrained-weight embeddings with
reconstructed preprocessing, not verified reproduction of the original space.**

## Run locally or inside a compute allocation

Download only the checkpoint, without PyTorch or a scaler:

```sh
python -m reference_baselines.reference_set_comparison.generate_embeddings --download-only
```

First run eight training rows:

```sh
python -m reference_baselines.reference_set_comparison.generate_embeddings --limit 8 --out-dir reference_baselines/reference_set_comparison/outputs/smoke
```

The first run still downloads and prepares the default scaler sample. `--limit`
only limits embedding export. For a faster installation test, use a smaller
`--scaler-cap 100`; do not interpret that smoke-test scaler as the 20,000-row setup.

Then export the full training set using the same cached default scaler:

```sh
python -m reference_baselines.reference_set_comparison.generate_embeddings
```

To reuse an existing scaler (including the tuple saved by the supplied script):

```sh
python -m reference_baselines.reference_set_comparison.generate_embeddings --scaler /path/to/out/scaler.pt
```

Use `--train-csv mp_20/train.csv` for the downloaded local MatterGen version;
it is not assumed identical to the HF version. `--checkpoint`, `--config`,
`--artifact-dir`, `--device cpu|cuda|auto`, `--batch-size`, and `--out-dir` are
configurable. Existing output directories are never overwritten.

## Slurm

`embed.sh` follows the supplied GPU job example, with fail-fast execution and the
working directory taken from `SLURM_SUBMIT_DIR`. Submit from the repo root with
an existing environment. By default it uses `.venv/bin/python`. If using the
environment in the supplied example, set `PYTHON` first:

```sh
export PYTHON="$HOME/envs/crysjepa/bin/python"
squeue -u "$USER"
sbatch reference_baselines/reference_set_comparison/embed.sh --limit 8 --out-dir reference_baselines/reference_set_comparison/outputs/smoke
```

The template requests one GPU, four CPUs, 16 GB RAM, and two hours on `gpu`,
matching the supplied partition name. Adjust resource options to your actual
allocation; no completion time is guaranteed for initial scaler fitting.
No array or automatic follow-up jobs are submitted. Respect account-wide job
limits from the SOC instructions before submitting.

```sh
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode
# Replace JOB_ID below with the returned job number.
tail -f embed_JOB_ID.log
```

After a successful smoke run, submit the full export:

```sh
sbatch reference_baselines/reference_set_comparison/embed.sh
```

Check the log and output `provenance.json` before treating a job as successful.
Do not run a full export concurrently with another job fitting the same scaler;
wait for the smoke job to finish first. No cluster job was submitted when these
files were prepared.

## Output and alignment

The default output directory is `outputs/mp20_train/`:

- `embeddings.npy`: float32 `(number_of_input_rows, 512)` raw JEPA vectors.
- `metadata.csv`: original CSV columns except CIF text, explicit `embedding_row`
  and `source_row`, formula, chemical system, processed atom count, and a
  `band_gap` alias if the input has `dft_band_gap`.
- `provenance.json`: dataset/checkpoint/scaler/config/output hashes, scaler
  reconstruction details, settings, and output checks. Written last on success.

For every `i`, `embeddings[i]` describes `metadata.iloc[i]` and input CSV row `i`.
The original `Unnamed: 0` column is retained as metadata, not used as an array
index. Reference rows are never shuffled, skipped, sorted, or deduplicated.
A malformed reference structure aborts with its row and material ID.

Structure preprocessing follows the supplied script and the training-loader
convention: primitive cell, reduced cell, atom sorting by type/coordinate key,
float32 polar-decomposed lattice features, atom-count scaling, and lattice
normalisation. No augmentation or embedding L2 normalisation is applied.
This differs from `V_screen.py`, which preserves the input cell/site order.
Model weights load strictly; the first batch is checked against singleton
inference, and saved output row IDs are verified.

Later candidate embeddings must use the **same saved scaler, checkpoint, config,
preprocessing, and compatible package versions**. Pass that scaler explicitly
with `--scaler` rather than fitting on candidates. Do not mix these embeddings
with earlier arrays of unknown preprocessing provenance.

## Downloaded proxy report analysis

The independent-proxy section of `inspect_train.ipynb` reads the combined
`outputs/comparison_20_29/reports` bundle without model inference. Shared functions
in `analysis.py` validate report hashes, selection counts, missing-label
denominators, overlaps, pooled aggregation and cross-proxy agreement. Both
MEGNet and eSEN use the same plotting/table functions. Change the notebook's
report directory to analyse another completed run; keep the complete report
bundle together. Derived figures/tables and analysis provenance are written to
its `analysis/` child without changing the source report receipt.

The analysis distinguishes valid-label precision from success per requested
slot, with explicit unknown-label bounds. It does not average models into an
overall stability score. See `../docs/EXPERIMENT_REPORT.md` for the completed
20–29 experiment, equations, results, and the distinction between union screening
references and the fixed hull phase bank.

## Reference UMAP in the inspection notebook

The final section of `inspect_train.ipynb` projects every row of the MP-20
reference export to two dimensions. It fits UMAP on raw 512-dimensional
embeddings and then colours the points by five equal-count formation-energy
groups, with a second continuous-energy plot and an offline HTML hover view.
Install optional notebook packages from `requirements-visualization.txt` in
the notebook kernel if needed.

The section runs independently of earlier notebook cells. Input checksums and
row alignment are verified; coordinates, figures, hover metadata, parameters,
and package versions are saved under `outputs/mp20_train/umap/`. Re-running
unchanged inputs and settings reuses the coordinates. The first fit can take
several minutes on a laptop. Numerical compilation caches also stay in this
output directory.

This is a qualitative comparison with the paper's mixed MP/MPtrj sample of
100,000 crystals. Our smaller MP-20 reference population and reconstructed
lattice scaler differ. Similar shapes or colour bands cannot establish
stability or exact reproduction of the published embedding space.
