#!/usr/bin/env bash
#SBATCH --job-name=edge-attribute-gnn
#SBATCH --account=allusers
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=reference_baselines/results/logs/edge-attribute-%j.log

# Safe default: tests only. Set STAGE explicitly for smoke/train/prepare/evaluate.
# This experiment is CPU-bound; build its environment in node-local /tmp.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export STAGE="${STAGE:-tests}"
if [[ -z "${PYTHON:-}" ]]; then
  JOB_WORK="/tmp/$USER/edge-attribute-${SLURM_JOB_ID}"
  export TMPDIR="$JOB_WORK/pip"
  VENV="$JOB_WORK/venv"
  mkdir -p "$TMPDIR"
  /usr/bin/python3.12 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir --upgrade pip
  python -m pip install --no-cache-dir numpy pandas scipy pymatgen
  python -m pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu
  export PYTHON="$VENV/bin/python"
fi
bash reference_baselines/cluster/run_edge_attribute_reference_gnn.sh
