#!/usr/bin/env bash
# Run one stage of the fixed-topology edge-attribute reference GNN experiment.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-tests}"
DRY_RUN="${DRY_RUN:-0}"
VARIANT="${VARIANT:-}"
MODEL="${MODEL:-}"
SEED="${SEED:-}"
CONFIG="reference_baselines/configs/edge_attribute_reference_gnn.toml"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p reference_baselines/results/logs

run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

check_inputs() {
  local input
  while IFS= read -r input; do
    [[ -z "$input" || "$input" == \#* ]] && continue
    require_file "$input"
  done < reference_baselines/cluster/cluster_inputs.txt
  require_file reference_baselines/results/reference_gnn/graphs/physical.npz
  echo "Edge-attribute input check passed."
}

tests() {
  run "$PYTHON" -m unittest \
    reference_baselines.tests.test_reference_gnn \
    reference_baselines.tests.test_edge_attribute_reference_gnn -v
}

train() {
  local command=(
    "$PYTHON" -m reference_baselines.training.train_edge_attribute_reference_gnn
    --config "$CONFIG"
  )
  [[ -n "$VARIANT" ]] && command+=(--variant "$VARIANT")
  [[ -n "$MODEL" ]] && command+=(--model "$MODEL")
  [[ -n "$SEED" ]] && command+=(--seed "$SEED")
  run "${command[@]}"
}

prepare() {
  run "$PYTHON" -m reference_baselines.evaluation.edge_attribute_reference_gnn \
    --config "$CONFIG" --prepare
}

evaluate() {
  echo "WARNING: this stage opens the reserved MatterSim proxy outcomes."
  run "$PYTHON" -m reference_baselines.evaluation.edge_attribute_reference_gnn \
    --config "$CONFIG" --evaluate
}

smoke() {
  tests
  run "$PYTHON" -m reference_baselines.training.train_edge_attribute_reference_gnn \
    --config "$CONFIG" --smoke
  run "$PYTHON" -m reference_baselines.evaluation.edge_attribute_reference_gnn \
    --config "$CONFIG" --smoke --prepare
  run "$PYTHON" -m reference_baselines.evaluation.edge_attribute_reference_gnn \
    --config "$CONFIG" --smoke --evaluate
}

echo "Python: $PYTHON"
echo "Stage: $STAGE"

case "$STAGE" in
  tests) tests ;;
  check) check_inputs ;;
  smoke) check_inputs; smoke ;;
  train) check_inputs; train ;;
  prepare) check_inputs; prepare ;;
  evaluate) check_inputs; evaluate ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    echo "Use tests, check, smoke, train, prepare, or evaluate." >&2
    exit 1
    ;;
esac
