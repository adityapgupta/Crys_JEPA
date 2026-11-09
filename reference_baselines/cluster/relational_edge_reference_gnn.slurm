#!/usr/bin/env bash
#SBATCH --job-name=relational-edge-gnn
#SBATCH --account=allusers
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=reference_baselines/results/logs/relational-edge-%j.log

# Safe default: tests only. Set STAGE explicitly for precompute/smoke/train/prepare/evaluate.
# CPU-bound; build the environment in node-local /tmp.
# pymatgen (hull precompute) and dscribe/ase (SOAP generation) are installed below.
# SOAP itself is generated separately via reference_baselines.structure.descriptors;
# reference_hull_energies.csv must be supplied (see the experiment doc).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export STAGE="${STAGE:-tests}"
if [[ -z "${PYTHON:-}" ]]; then
  JOB_WORK="/tmp/$USER/relational-edge-${SLURM_JOB_ID}"
  export TMPDIR="$JOB_WORK/pip"
  VENV="$JOB_WORK/venv"
  mkdir -p "$TMPDIR"
  /usr/bin/python3.12 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir --upgrade pip
  python -m pip install --no-cache-dir numpy pandas scipy pymatgen
  python -m pip install --no-cache-dir dscribe ase   # SOAP feature generation
  python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  export PYTHON="$VENV/bin/python"
fi
bash reference_baselines/cluster/run_relational_edge_reference_gnn.sh
