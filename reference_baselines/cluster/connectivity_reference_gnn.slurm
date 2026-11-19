#!/usr/bin/env bash
#SBATCH --job-name=connectivity-gnn
#SBATCH --account=allusers
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=reference_baselines/results/logs/connectivity-%j.log

# Safe default: tests only. Set STAGE explicitly for adjacency/smoke/prepare/evaluate.
# CPU-bound; build the environment in node-local /tmp. pymatgen is required because the
# shared evaluation.reserved_duplicates module imports it at top level (even though we
# reuse the cached reserved_duplicate_pairs.csv and never recompute). dscribe/ase are NOT
# needed -- the soap/phase graphs reuse already-built soap_density_normalized.npy and
# hull_compatibility.npz.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export STAGE="${STAGE:-tests}"
if [[ -z "${PYTHON:-}" ]]; then
  JOB_WORK="/tmp/$USER/connectivity-${SLURM_JOB_ID}"
  export TMPDIR="$JOB_WORK/pip"
  VENV="$JOB_WORK/venv"
  mkdir -p "$TMPDIR"
  /usr/bin/python3.12 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir --upgrade pip
  python -m pip install --no-cache-dir numpy pandas scipy pymatgen
  python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  export PYTHON="$VENV/bin/python"
fi
bash reference_baselines/cluster/run_connectivity_reference_gnn.sh
