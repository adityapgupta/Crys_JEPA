#!/usr/bin/env bash
#SBATCH --job-name=crys-install
#SBATCH --account=allusers
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=install-%j.log

# Build the CPU-only Python environment on a compute node.
# Submit from the repository root:
#   sbatch install.sh
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

if [[ -n "${PYTHON_MODULE:-}" ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load "$PYTHON_MODULE"
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "Need Python 3.11 or later. Set PYTHON_MODULE, for example:" >&2
  echo "  PYTHON_MODULE=python/3.11 sbatch install.sh" >&2
  exit 1
fi

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
from importlib.metadata import version
import numpy, pandas, scipy, torch, pymatgen
from pymatgen.core import Structure

for name, module in (
    ("python", None),
    ("numpy", numpy),
    ("pandas", pandas),
    ("scipy", scipy),
    ("torch", torch),
    ("pymatgen", pymatgen),
):
    found = sys.version.split()[0] if module is None else version(name)
    print(name, found)
print("Structure import ok:", Structure.__name__)
print("cuda_available", torch.cuda.is_available())
PY

echo "Install complete. Use PYTHON=$VENV/bin/python"
