# SOC commands: chemical references versus kNN and union

Execute sections in order. The default batch range is **20 through 29 inclusive**.
The existing `~/Crys_JEPA/release_data/exp_logs` and `outputs/mp20_train` are reused.
Everything large is downloaded on a compute node. No model inference runs on xlogin.

## 1. Upload source changes from Windows PowerShell

```powershell
Set-Location "C:\Users\Atharv\OneDrive\Desktop\Books\NUS Courses\CS5284_Graph_Machine_Learning\Project"
tar -czf crysjepa-comparison-source.tar.gz --exclude=.git --exclude='.venv*' --exclude=__pycache__ --exclude=mp_20 --exclude=mp_20.zip --exclude=artifacts --exclude=outputs --exclude=release_data --exclude=Crys_JEPA/reference_baselines/data --exclude=Crys_JEPA/reference_baselines/results Crys_JEPA
scp -J atharvak@stujump.comp.nus.edu.sg .\crysjepa-comparison-source.tar.gz atharvak@xlogin.comp.nus.edu.sg:~/
ssh -J atharvak@stujump.comp.nus.edu.sg atharvak@xlogin.comp.nus.edu.sg
```

## 2. On xlogin: extract source and allocate a compute node

The archive updates source files. It does not replace excluded cluster datasets,
saved weights, existing exports, or environments.

```bash
cd ~
tar -xzf crysjepa-comparison-source.tar.gz
cd ~/Crys_JEPA
srun --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=02:00:00 --pty bash -l
```

Confirm the prompt now names a compute node. Then:

```bash
cd ~/Crys_JEPA
nvidia-smi
bash reference_baselines/reference_set_comparison/setup_environment.sh jepa
bash reference_baselines/reference_set_comparison/setup_environment.sh megnet
bash reference_baselines/reference_set_comparison/setup_environment.sh esen
```

These create `.venv-comparison`, `.venv-megnet`, and `.venv-esen`; your existing
`.venv` is preserved. Setup is one-time unless dependencies change. If allocation
time runs low, exit, allocate again, and run only unfinished setup commands.
Use `PYTHON_BOOTSTRAP=/path/to/python3.12 bash ...` if needed.

Direct package versions are pinned. Full platform-specific installed locks are
written under `artifacts/environment/`; they become tested locks only after the
checks below. The external GPU environments have not been validated from the laptop.

## 3. Check configuration and run offline tests

```bash
cat reference_baselines/reference_set_comparison/experiment.yml
.venv-comparison/bin/python -m unittest discover -s tests -p test_reference_set_comparison.py -v
.venv-comparison/bin/python -m unittest discover -s tests -p test_proxy_comparison.py -v
.venv-comparison/bin/python -m reference_baselines.reference_set_comparison.pipeline prepare
```

The default input is `release_data/exp_logs/base/mp_20/eval`, not the laptop's
`artifacts/release_samples`. The config points to the full 20,000-per-file scaler,
not the earlier 100-row smoke scaler. Missing files and mismatched hashes fail.

For a different range, change `start_batch`, `end_batch`, and `output` together
in a copied configuration file, then use `--config path/to/new.yml` on **every**
stage. Alternatively pass `--start-batch 20 --end-batch 29 --output path/to/run`
on every command. Never reuse an old run directory for different scientific settings.
Use complete ten-folder groups for the primary comparison; partial groups are
allowed and their smaller eligible populations are reported.

## 4. JEPA GPU export, exact index, and selections

While still inside the compute allocation:

```bash
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline embed
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline score
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline select
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline fetch-hull
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline phases
```

`phases.json` records how many reference structures require predictions. This can
be much larger than the selected candidate union. Every phase is scored once per
model and reused across candidate systems and subsequent runs. All polymorphs
are kept; there is no selection-dependent or DFT-based pre-pruning.

To run JEPA as a batch job instead, first `exit` to xlogin and submit:

```bash
cd ~/Crys_JEPA
sbatch reference_baselines/reference_set_comparison/run_gpu.sh embed
squeue -u "$USER"
```

After that job completes successfully, CPU stages can similarly be submitted one
at a time using `sbatch reference_baselines/reference_set_comparison/run_cpu.sh score`,
then `select`, `fetch-hull`, and `phases`. Wait for each prerequisite to complete.
The guide's direct Python commands always belong inside a compute allocation.

## 5. Download and smoke-test the two proxies

First accept the conditions at <https://huggingface.co/facebook/OMAT24> in your
browser. In a **compute allocation**, authenticate if the eSEN download is gated:

```bash
.venv-comparison/bin/huggingface-cli login
.venv-comparison/bin/python -m reference_baselines.reference_set_comparison.pipeline download-model --model megnet
.venv-comparison/bin/python -m reference_baselines.reference_set_comparison.pipeline download-model --model esen
.venv-megnet/bin/python -u -m reference_baselines.reference_set_comparison.pipeline smoke-model --model megnet
.venv-esen/bin/python -u -m reference_baselines.reference_set_comparison.pipeline smoke-model --model esen
```

Do not paste access tokens into source files or commands saved in this repository.
Smoke tests use actual weights and check finite per-atom outputs, singleton versus
chunk predictions, and invariance of per-atom energy under cell replication.
`predict` refuses to run without a passed smoke receipt for its exact environment.
If a test fails, stop and inspect the traceback rather than running production.

After they pass, export the tested environment locks:

```bash
.venv-comparison/bin/python -m pip freeze > reference_baselines/reference_set_comparison/artifacts/environment/jepa.tested.txt
.venv-megnet/bin/python -m pip freeze > reference_baselines/reference_set_comparison/artifacts/environment/megnet.tested.txt
.venv-esen/bin/python -m pip freeze > reference_baselines/reference_set_comparison/artifacts/environment/esen.tested.txt
```

Try a small resumable prediction chunk before full jobs:

```bash
.venv-megnet/bin/python -u -m reference_baselines.reference_set_comparison.pipeline predict --model megnet --max-items 16
.venv-esen/bin/python -u -m reference_baselines.reference_set_comparison.pipeline predict --model esen --max-items 16
exit
```

The `pending` count will remain positive after these small chunks; this is expected.

## 6. Submit production inference from xlogin

```bash
cd ~/Crys_JEPA
sbatch reference_baselines/reference_set_comparison/run_gpu.sh predict megnet
sbatch reference_baselines/reference_set_comparison/run_gpu.sh predict esen
squeue -u "$USER"
```

Logs are `compare_gpu_<jobid>.log`. Replace `JOBID` with the returned numeric ID:

```bash
tail -f compare_gpu_JOBID.log
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
```

Prediction caches commit after each chunk. If a job reaches its time limit,
resubmit the **same** command; completed structures are reused. Never run duplicate
jobs for the same proxy concurrently. The two proxy jobs use separate cache files.
Only proceed when both `<model>_prediction_status.json` files say `complete: true`.
Inspect them with:

```bash
cat reference_baselines/reference_set_comparison/outputs/comparison_20_29/megnet_prediction_status.json
cat reference_baselines/reference_set_comparison/outputs/comparison_20_29/esen_prediction_status.json
```

If a model environment is repaired, rerun its smoke test and use `--retry-errors`
to retry previously failed structures. Changing environment versions creates a
new prediction identity, so predictions from incompatible environments do not mix.

## 7. Calculate fixed hull labels and produce reports

From xlogin, submit these CPU stages **one at a time**, waiting for completion:

```bash
sbatch reference_baselines/reference_set_comparison/run_cpu.sh hull --model megnet
sbatch reference_baselines/reference_set_comparison/run_cpu.sh hull --model esen
sbatch reference_baselines/reference_set_comparison/run_cpu.sh report
```

Or use a CPU allocation and run them sequentially:

```bash
srun --partition=normal --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash -l
cd ~/Crys_JEPA
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline hull --model megnet
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline hull --model esen
.venv-comparison/bin/python -u -m reference_baselines.reference_set_comparison.pipeline report
```

Each proxy's fixed hull uses **all relevant MP2023 reference phases scored by
that proxy at stored reference geometries**, with no extra energy corrections.
It excludes generated candidates. Failed required phases make a hull incomplete.
Negative distances to this fixed hull are retained. MEGNet and eSEN single points
on MatterSim-relaxed candidates are proxy evidence, not new relaxation or DFT.

## 8. Download results to Windows

### MEGNet-only report while eSEN is still running

With the updated `pipeline.py` and `reporting.py` uploaded, submit from xlogin:

```bash
cd ~/Crys_JEPA
sbatch reference_baselines/reference_set_comparison/run_cpu.sh report --model megnet
```

This requires the completed `megnet_hull` stage, but no eSEN predictions or hull.
It writes `reports/megnet/REPORT.md` and the MEGNet metric/overlap tables, without
cross-model consensus. It preserves the combined report and all running inference.
Wait for the report job's main Slurm status to be `COMPLETED` with exit `0:0`.

Then in **Windows PowerShell** from the local repository root:

```powershell
$destination = "reference_baselines\reference_set_comparison\outputs\comparison_20_29\reports"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
scp -J atharvak@stujump.comp.nus.edu.sg -r atharvak@xlogin.comp.nus.edu.sg:~/Crys_JEPA/reference_baselines/reference_set_comparison/outputs/comparison_20_29/reports/megnet $destination
```

Open `outputs/comparison_20_29/reports/megnet/REPORT.md` or `pooled_metrics.csv`.
For the main unique top-1% comparison, filter the CSV to `policy=unique_refill`,
`fraction=0.01`, and `threshold_eV_per_atom=0.1`. Also inspect coverage and
requested-slot yield bounds. Once eSEN finishes, run the ordinary `report` stage
without `--model` to add the combined report and model-consensus tables.

### Download the full run or combined reports

Run in **local Windows PowerShell**, not on the cluster:

```powershell
Set-Location "C:\Users\Atharv\OneDrive\Desktop\Books\NUS Courses\CS5284_Graph_Machine_Learning\Project\Crys_JEPA"
New-Item -ItemType Directory -Force reference_baselines\reference_set_comparison\outputs | Out-Null
scp -J atharvak@stujump.comp.nus.edu.sg -r atharvak@xlogin.comp.nus.edu.sg:~/Crys_JEPA/reference_baselines/reference_set_comparison/outputs/comparison_20_29 reference_baselines\reference_set_comparison\outputs\
```

For reports only, download the `comparison_20_29/reports` subdirectory instead.
Large model caches and the MP2023 database stay on SOC. The main report is
`reports/REPORT.md`; detailed tables include:

| File | Purpose |
|---|---|
| `turn_metrics.csv` / `pooled_metrics.csv` | Primary top-1% precision and 5%/10% sensitivities |
| `batch_metrics.csv` | Per-folder diagnostic selections |
| `paired_comparisons.csv` | Difference versus chemical baseline, uncertainty and exclusive-cohort precision |
| `overlap.csv` / `selection_membership.csv` | Selection overlap and candidate-level disagreement |
| `proxy_consensus.csv` | Both/either/neither model pass rates and label coverage |
| `runtime.csv` / `provenance.json` | Time, inputs, model weights and evaluation settings |

Threshold: `<0.1` eV/atom, not formation energy `<0.1`. The threshold applies to
the difference from the fixed composition-balanced hull. Precision is among valid
labels; requested-slot yields and unknown-label bounds prevent failures or lack
of coverage from appearing as improvements.
