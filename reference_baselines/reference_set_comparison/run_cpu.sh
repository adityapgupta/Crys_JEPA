#!/usr/bin/env bash
#SBATCH --job-name=crys-compare-cpu
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=compare_cpu_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Crys_JEPA repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/${USER}/crys-compare-${SLURM_JOB_ID:-interactive}}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python_bin="${PYTHON:-$PWD/.venv-comparison/bin/python}"
stage="${1:?Supply a stage: prepare, score, select, fetch-hull, phases, hull, report}"
shift
"$python_bin" -m reference_baselines.reference_set_comparison.pipeline "$stage" "$@"
