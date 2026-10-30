#!/usr/bin/env bash
#SBATCH --job-name=crys-compare-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=compare_gpu_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Crys_JEPA repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/crys-compare-${SLURM_JOB_ID:-interactive}}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"
stage="${1:?Supply embed, smoke-model, or predict}"
shift
if [[ "$stage" == "embed" ]]; then
    python_bin="${JEPA_PYTHON:-$PWD/.venv-comparison/bin/python}"
else
    model="${1:?Supply megnet or esen after the stage}"
    shift
    case "$model" in
        megnet) python_bin="${MEGNET_PYTHON:-$PWD/.venv-megnet/bin/python}" ;;
        esen) python_bin="${ESEN_PYTHON:-$PWD/.venv-esen/bin/python}" ;;
        *) echo "Unknown model: $model" >&2; exit 2 ;;
    esac
    set -- --model "$model" "$@"
fi
"$python_bin" -m reference_baselines.reference_set_comparison.pipeline "$stage" "$@"
