#!/usr/bin/env bash
#SBATCH --job-name=conn-train
#SBATCH --account=allusers
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=reference_baselines/results/logs/conn-train-%A_%a.log

# Training array for the connectivity reference GNN.
#
# One array task per (graph, model) spec; each task runs all 3 seeds via the base
# trainer, which skips any run that already has complete.json (safe to requeue).
# 13 specs -> submit with:
#     sbatch --array=0-12 reference_baselines/cluster/connectivity_train_array.slurm
# (add %N to cap concurrency, e.g. --array=0-12%8).
#
# Reuses the base trainer (reference_baselines.training.train_reference_gnn) with a
# per-spec --graph/--model and the connectivity config and run-dir.
# Requires the soap/phase graphs to exist first (STAGE=adjacency).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

CONFIG="reference_baselines/configs/connectivity_reference_gnn.toml"
RUN_DIR="reference_baselines/results/connectivity_reference_gnn/runs"
GRAPHS="reference_baselines/results/reference_gnn/graphs"
for g in physical cosine soap phase; do
  if [[ ! -f "$GRAPHS/$g.npz" ]]; then
    echo "Missing graph $GRAPHS/$g.npz -- run STAGE=adjacency first" >&2; exit 1
  fi
done

# 4 topologies x 3 GNN models + 1 readout (no_graph, trained once under physical).
SPECS=(
  "physical gcn" "physical gat" "physical sage"
  "cosine gcn" "cosine gat" "cosine sage"
  "soap gcn" "soap gat" "soap sage"
  "phase gcn" "phase gat" "phase sage"
  "physical no_graph"
)
if [[ "${#SPECS[@]}" -ne 13 ]]; then
  echo "Spec table is ${#SPECS[@]}, expected 13" >&2; exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-12}"
if (( IDX < 0 || IDX >= ${#SPECS[@]} )); then
  echo "Array index $IDX out of range 0..$(( ${#SPECS[@]} - 1 ))" >&2; exit 1
fi
read -r GRAPH MODEL <<< "${SPECS[$IDX]}"
echo "Array task $IDX -> graph=$GRAPH model=$MODEL (all seeds)"

if [[ -z "${PYTHON:-}" ]]; then
  JOB_WORK="/tmp/$USER/conn-train-${SLURM_JOB_ID}"
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

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p reference_baselines/results/logs
"$PYTHON" -m reference_baselines.training.train_reference_gnn \
  --config "$CONFIG" --run-dir "$RUN_DIR" --graph "$GRAPH" --model "$MODEL"
