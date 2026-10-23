#!/usr/bin/env bash
#SBATCH --job-name=crys-reference-gnn
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=reference_baselines/results/logs/reference-gnn-%j.log

# Submit from the repository root after you activate the cluster environment.
# Add your site's account and partition options to the sbatch command.
# The time limit covers the whole STAGE=all job: 4 graphs x 3 models x 3 seeds.
# Cluster runtime is not measured. Measure one graph before the full job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export STAGE="${STAGE:-all}"
export BUILD_RESERVED_DUPLICATES="${BUILD_RESERVED_DUPLICATES:-0}"
bash reference_baselines/cluster/run_reference_gnn.sh
