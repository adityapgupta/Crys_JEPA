#!/usr/bin/env bash
# Run from a checkout with the prepared input artifacts listed in cluster_inputs.txt.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/reference_baselines/results/matplotlib}"
mkdir -p reference_baselines/results/logs

run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    "$@"
  fi
}

if [[ "${REBUILD_RETRIEVAL:-0}" == 1 ]] || [[ ! -f reference_baselines/results/cosine_retrieval_provenance.json ]]; then
  run "$PYTHON" -m reference_baselines.graphs.cosine --threads "${SLURM_CPUS_PER_TASK:-6}"
fi
run "$PYTHON" -m reference_baselines.baselines.angular
run "$PYTHON" -m reference_baselines.baselines.random_controls
run "$PYTHON" -m reference_baselines.graphs.diffusion --graph-metric euclidean
run "$PYTHON" -m reference_baselines.graphs.diffusion --graph-metric cosine
run "$PYTHON" -m reference_baselines.baselines.conductance

if [[ "${RUN_EIGENVECTORS:-0}" == 1 ]]; then
  for metric in euclidean cosine; do
    run "$PYTHON" -m reference_baselines.graphs.eigenmodes --graph-metric "$metric" \
      --solver "${EIGEN_SOLVER:-iterative-shift}" --max-seconds "${EIGEN_MAX_SECONDS:-18000}"
    run "$PYTHON" -m reference_baselines.baselines.spectral --graph-metric "$metric" \
      --threads "${SLURM_CPUS_PER_TASK:-6}"
  done
fi

run "$PYTHON" -m reference_baselines.checks.nonparametric
run "$PYTHON" -m reference_baselines.reporting.reference_examples
if [[ "${RUN_EIGENVECTORS:-0}" == 1 ]]; then
  run "$PYTHON" -m reference_baselines.evaluation.nonparametric --require-spectral
else
  run "$PYTHON" -m reference_baselines.evaluation.nonparametric
fi
run "$PYTHON" -m unittest discover -s tests -p 'test_nonparametric.py' -v
run "$PYTHON" -m reference_baselines.reporting.nonparametric
