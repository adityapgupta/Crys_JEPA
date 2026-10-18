#!/usr/bin/env bash
#SBATCH --job-name=crys-reference-baselines
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=reference_baselines/results/logs/baselines-%j.log

# Submit from the repository root after activating the cluster Python environment.
# Add your site's account and partition options to the sbatch command.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
bash reference_baselines/cluster/run_baselines.sh
