#!/usr/bin/env bash
#SBATCH --job-name=crys-install
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=install-%j.log

# Build the CPU-only Python environment on a compute node.
# Submit from the repository root:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION install.sh
#
# The environment and temp files stay in shared home, not in /tmp.
# Set RECREATE=1 to rebuild an existing environment.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

VENV="${VENV:-$HOME/venv-gnn}"
TMPDIR="${TMPDIR:-$HOME/tmp}"
export TMPDIR
mkdir -p "$TMPDIR"

echo "Host: $(hostname)"
echo "Repo: $PWD"
echo "Venv: $VENV"
echo "TMPDIR: $TMPDIR"

if [[ "${RECREATE:-0}" == 1 && -d "$VENV" ]]; then
  rm -rf "$VENV"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir numpy pandas scipy pymatgen

# CPU wheel. Do not install the default CUDA wheel on a CPU job.
python -m pip install --no-cache-dir torch \
  --index-url https://download.pytorch.org/whl/cpu

python - <<'PY'
import sys
import numpy, pandas, scipy, torch, pymatgen

print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("torch", torch.__version__)
print("pymatgen", pymatgen.__version__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Install complete. Use PYTHON=$VENV/bin/python"
