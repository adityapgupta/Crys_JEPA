#!/usr/bin/env bash
# Run one stage of the relational edge-feature reference GNN experiment.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-tests}"
DRY_RUN="${DRY_RUN:-0}"
VARIANT="${VARIANT:-}"
MODEL="${MODEL:-}"
SEED="${SEED:-}"
CONFIG="reference_baselines/configs/relational_edge_reference_gnn.toml"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p reference_baselines/results/logs

run() {
  printf '%q ' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}

require_file() {
  if [[ ! -f "$1" ]]; then echo "Missing required file: $1" >&2; exit 1; fi
}

check_inputs() {
  require_file reference_baselines/results/reference_gnn/graphs/physical.npz
  require_file reference_baselines/results/soap_density_normalized.npy
  require_file reference_baselines/results/relational_edge/reference_decomp_frequency.npy
  require_file reference_baselines/results/relational_edge/cooccurrence_normaliser.npy
  require_file reference_baselines/results/relational_edge/hull_compatibility.npz
  echo "Relational-edge input check passed."
}

soap() {
  # Generate element-blind SOAP fingerprints (needs dscribe/ase; reads data/mp_20_train.csv.gz).
  run "$PYTHON" -m reference_baselines.structure.descriptors --workers "${SOAP_WORKERS:-8}"
}

precompute() {
  # Builds co-occurrence frequency + hull compatibility (needs reference_hull_energies.csv).
  run "$PYTHON" -m reference_baselines.graphs.relational_edge_features
}

tests() {
  run "$PYTHON" -m unittest \
    reference_baselines.tests.test_reference_gnn \
    reference_baselines.tests.test_relational_edge_reference_gnn -v
}

train() {
  local command=(
    "$PYTHON" -m reference_baselines.training.train_relational_edge_reference_gnn --config "$CONFIG"
  )
  [[ -n "$VARIANT" ]] && command+=(--variant "$VARIANT")
  [[ -n "$MODEL" ]] && command+=(--model "$MODEL")
  [[ -n "$SEED" ]] && command+=(--seed "$SEED")
  run "${command[@]}"
}

prepare() {
  run "$PYTHON" -m reference_baselines.evaluation.relational_edge_reference_gnn --config "$CONFIG" --prepare
}

evaluate() {
  echo "WARNING: this stage opens the reserved MatterSim proxy outcomes."
  run "$PYTHON" -m reference_baselines.evaluation.relational_edge_reference_gnn --config "$CONFIG" --evaluate
}

smoke() {
  tests
  run "$PYTHON" -m reference_baselines.training.train_relational_edge_reference_gnn --config "$CONFIG" --smoke
  run "$PYTHON" -m reference_baselines.evaluation.relational_edge_reference_gnn --config "$CONFIG" --smoke --prepare
  run "$PYTHON" -m reference_baselines.evaluation.relational_edge_reference_gnn --config "$CONFIG" --smoke --evaluate
}

echo "Python: $PYTHON"
echo "Stage: $STAGE"

case "$STAGE" in
  tests) tests ;;
  check) check_inputs ;;
  soap) soap ;;
  precompute) precompute ;;
  smoke) check_inputs; smoke ;;
  train) check_inputs; train ;;
  prepare) check_inputs; prepare ;;
  evaluate) check_inputs; evaluate ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    echo "Use tests, check, soap, precompute, smoke, train, prepare, or evaluate." >&2
    exit 1
    ;;
esac
