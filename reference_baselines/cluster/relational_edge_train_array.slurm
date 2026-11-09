#!/usr/bin/env bash
#SBATCH --job-name=rel-edge-train
#SBATCH --account=allusers
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=reference_baselines/results/logs/rel-edge-train-%A_%a.log

# Training array for the relational edge-feature reference GNN.
#
# One array task per (variant, model) spec; each task runs all 3 seeds via the
# training script, which skips any run that already has complete.json (safe to
# requeue).  22 specs -> submit with:
#     sbatch --array=0-21 reference_baselines/cluster/relational_edge_train_array.slurm
# (add %N to cap concurrency, e.g. --array=0-21%8).
#
# Minimal env: training imports numpy/pandas/scipy/torch only -- pymatgen and
# dscribe/ase are needed for the precompute/soap stages, not here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Spec table -- mirrors run_specs() order in train_relational_edge_reference_gnn.py.
# 7 variants x 3 models + readout(1) = 22 specs.
SPECS=(
  "scalar gcn" "scalar gat" "scalar sage"
  "geom gcn" "geom gat" "geom sage"
  "cooc gcn" "cooc gat" "cooc sage"
  "phase gcn" "phase gat" "phase sage"
  "rich gcn" "rich gat" "rich sage"
  "shuffled gcn" "shuffled gat" "shuffled sage"
  "readout readout"
  "redundant gcn" "redundant gat" "redundant sage"
)
if [[ "${#SPECS[@]}" -ne 22 ]]; then
  echo "Spec table is ${#SPECS[@]}, expected 22" >&2; exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-21}"
if (( IDX < 0 || IDX >= ${#SPECS[@]} )); then
  echo "Array index $IDX out of range 0..$(( ${#SPECS[@]} - 1 ))" >&2; exit 1
fi
read -r VARIANT MODEL <<< "${SPECS[$IDX]}"
echo "Array task $IDX -> variant=$VARIANT model=$MODEL (all seeds)"

if [[ -z "${PYTHON:-}" ]]; then
  JOB_WORK="/tmp/$USER/rel-edge-train-${SLURM_JOB_ID}"
  export TMPDIR="$JOB_WORK/pip"
  VENV="$JOB_WORK/venv"
  mkdir -p "$TMPDIR"
  /usr/bin/python3.12 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --no-cache-dir --upgrade pip
  python -m pip install --no-cache-dir numpy pandas scipy
  python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
  export PYTHON="$VENV/bin/python"
fi

export STAGE="train"
export VARIANT MODEL
export SEED=""   # empty -> training script loops all cfg seeds
bash reference_baselines/cluster/run_relational_edge_reference_gnn.sh
