#!/usr/bin/env bash
# Run the reference-utility GNN experiment from a prepared cluster checkout.
# See reference_baselines/docs/CLUSTER_GNN_RUNBOOK.md before you start.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-all}"
BUILD_RESERVED_DUPLICATES="${BUILD_RESERVED_DUPLICATES:-0}"
DRY_RUN="${DRY_RUN:-0}"
GRAPH_ONLY="${GRAPH_ONLY:-}"

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
    echo "See reference_baselines/cluster/cluster_inputs.txt" >&2
    exit 1
  fi
}

check_inputs() {
  local input
  while IFS= read -r input; do
    [[ -z "$input" || "$input" == \#* ]] && continue
    if [[ "$BUILD_RESERVED_DUPLICATES" == 1 &&
          "$input" == reference_baselines/results/reference_gnn/reserved_duplicate_* ]]; then
      echo "Skipped (will rebuild): $input"
      continue
    fi
    require_file "$input"
  done < reference_baselines/cluster/cluster_inputs.txt
  echo "Input check passed."
}

graphs() {
  run "$PYTHON" -m reference_baselines.graphs.build_reference_graph
}

prepare() {
  if [[ "$BUILD_RESERVED_DUPLICATES" != 1 ]]; then
    echo "Using the precomputed reserved duplicate table."
    require_file reference_baselines/results/reference_gnn/reserved_duplicate_pairs.csv
    run "$PYTHON" -m reference_baselines.evaluation.reference_gnn --prepare \
      --reuse-duplicates
  else
    run "$PYTHON" -m reference_baselines.evaluation.reference_gnn --prepare
  fi
}

train() {
  local graphs_to_run=("physical" "cosine" "self" "permuted")
  if [[ -n "$GRAPH_ONLY" ]]; then
    graphs_to_run=("$GRAPH_ONLY")
  fi
  for graph in "${graphs_to_run[@]}"; do
    run "$PYTHON" -m reference_baselines.training.train_reference_gnn \
      --graph "$graph"
  done
}

evaluate() {
  run "$PYTHON" -m reference_baselines.evaluation.reference_gnn --evaluate
}

echo "Python: $PYTHON"
echo "Stage: $STAGE"
check_inputs

case "$STAGE" in
  graphs) graphs ;;
  prepare) prepare ;;
  train) train ;;
  evaluate) evaluate ;;
  all)
    graphs
    prepare
    train
    evaluate
    ;;
  *)
    echo "Unknown STAGE: $STAGE" >&2
    echo "Use graphs, prepare, train, evaluate, or all." >&2
    exit 1
    ;;
esac
