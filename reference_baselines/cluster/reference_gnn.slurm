#!/usr/bin/env bash
#SBATCH --job-name=crys-reference-gnn
#SBATCH --account=allusers
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=reference_baselines/results/logs/reference-gnn-%j.log

# Submit from the repository root after you activate the cluster environment.
# The `long` partition allows up to 3 days. `normal` allows only 3 hours.
# The time limit covers the whole STAGE=all job: 4 graphs x 3 models x 3 seeds.
# Cluster runtime is not measured. Measure one graph before the full job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export STAGE="${STAGE:-all}"
export BUILD_RESERVED_DUPLICATES="${BUILD_RESERVED_DUPLICATES:-0}"
if [[ -x "$HOME/venv-gnn/bin/python" ]]; then
  export PYTHON="${PYTHON:-$HOME/venv-gnn/bin/python}"
fi
bash reference_baselines/cluster/run_reference_gnn.sh
