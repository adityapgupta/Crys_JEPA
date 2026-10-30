#!/usr/bin/env bash
#SBATCH --job-name=crysjepa-embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=embed_%j.log

# Submit from the repository root. Requires an existing environment; setup and
# package installation must also run on a compute node, not the login node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
test -f configs/jepa/mp.yml || { echo "Submit from the Crys_JEPA repo root" >&2; exit 1; }
python_bin="${PYTHON:-$PWD/.venv/bin/python}"
test -x "$python_bin" || { echo "Set PYTHON to your environment's absolute Python path" >&2; exit 1; }
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/crysjepa-${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# No generator training or relaxation. Extra arguments go to the exporter.
"$python_bin" -m reference_baselines.reference_set_comparison.generate_embeddings --device cuda "$@"
