#!/usr/bin/env bash
# Run one stage of the connectivity reference GNN experiment.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-tests}"
DRY_RUN="${DRY_RUN:-0}"
GRAPH="${GRAPH:-}"
MODEL="${MODEL:-}"
SEED="${SEED:-}"
CONFIG="reference_baselines/configs/connectivity_reference_gnn.toml"
RUN_DIR="reference_baselines/results/connectivity_reference_gnn/runs"
GRAPHS="reference_baselines/results/reference_gnn/graphs"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p reference_baselines/results/logs

run() { printf '%q ' "$@"; printf '\n'; if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi; }
require_file() { if [[ ! -f "$1" ]]; then echo "Missing required file: $1" >&2; exit 1; fi; }

adjacency() {
  # Build the soap and phase topologies (reuses soap_density_normalized.npy and
  # relational_edge/hull_compatibility.npz; physical/cosine are prebuilt).
  run "$PYTHON" -m reference_baselines.graphs.connectivity_adjacency
}

check_inputs() {
  require_file "$GRAPHS/physical.npz"
  require_file "$GRAPHS/cosine.npz"
  require_file "$GRAPHS/soap.npz"
  require_file "$GRAPHS/phase.npz"
  require_file reference_baselines/results/reference_gnn/reserved_duplicate_pairs.csv
  echo "Connectivity input check passed."
}

tests() {
  run "$PYTHON" -m unittest reference_baselines.tests.test_connectivity_reference_gnn -v
}

train() {
  local command=(
    "$PYTHON" -m reference_baselines.training.train_reference_gnn
    --config "$CONFIG" --run-dir "$RUN_DIR"
  )
  [[ -n "$GRAPH" ]] && command+=(--graph "$GRAPH")
  [[ -n "$MODEL" ]] && command+=(--model "$MODEL")
  [[ -n "$SEED" ]] && command+=(--seed "$SEED")
  run "${command[@]}"
}

prepare() {
  run "$PYTHON" -m reference_baselines.evaluation.connectivity_reference_gnn --config "$CONFIG" --run-dir "$RUN_DIR" --prepare
}

evaluate() {
  echo "WARNING: this stage opens the reserved MatterSim proxy outcomes."
  run "$PYTHON" -m reference_baselines.evaluation.connectivity_reference_gnn --config "$CONFIG" --run-dir "$RUN_DIR" --evaluate
}

smoke() {
  tests
  adjacency
  local smoke_dir="reference_baselines/results/connectivity_reference_gnn/smoke"
  run "$PYTHON" -m reference_baselines.training.train_reference_gnn --config "$CONFIG" --run-dir "$smoke_dir" --graph physical --model gcn --smoke
  run "$PYTHON" -m reference_baselines.training.train_reference_gnn --config "$CONFIG" --run-dir "$smoke_dir" --graph physical --model no_graph --smoke
  run "$PYTHON" -m reference_baselines.evaluation.connectivity_reference_gnn --config "$CONFIG" --run-dir "$smoke_dir" --smoke --prepare
}

echo "Python: $PYTHON"
echo "Stage: $STAGE"

case "$STAGE" in
  adjacency) adjacency ;;
  check) check_inputs ;;
  tests) tests ;;
  smoke) smoke ;;
  train) check_inputs; train ;;
  prepare) check_inputs; prepare ;;
  evaluate) check_inputs; evaluate ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    echo "Use adjacency, check, tests, smoke, train, prepare, or evaluate." >&2
    exit 1
    ;;
esac
