#!/usr/bin/env bash
# Run INSIDE a Slurm compute allocation. No downloads on the laptop.
set -euo pipefail
test -f configs/jepa/mp.yml || { echo "Run this from the Crys_JEPA repository root" >&2; exit 2; }
name="${1:?Supply jepa, megnet, or esen}"
python_bin="${PYTHON_BOOTSTRAP:-/usr/bin/python3.12}"
case "$name" in
    jepa) env_dir="$PWD/.venv-comparison"; torch_version="2.6.0" ;;
    megnet) env_dir="$PWD/.venv-megnet"; torch_version="2.6.0" ;;
    esen) env_dir="$PWD/.venv-esen"; torch_version="2.4.1" ;;
    *) echo "Unknown environment $name" >&2; exit 2 ;;
esac
"$python_bin" -m venv "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip
"$env_dir/bin/python" -m pip install "torch==$torch_version" --index-url https://download.pytorch.org/whl/cu124
"$env_dir/bin/python" -m pip install -r "reference_baselines/reference_set_comparison/requirements-$name.txt"
if [[ "$name" == "esen" ]]; then
    "$env_dir/bin/python" -m pip install torch-scatter torch-sparse torch-cluster \
        --only-binary=:all: -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
fi
"$env_dir/bin/python" -m pip check
# This is an installed-environment record, not yet evidence of a successful model smoke test.
mkdir -p reference_baselines/reference_set_comparison/artifacts/environment
"$env_dir/bin/python" -m pip freeze > "reference_baselines/reference_set_comparison/artifacts/environment/$name.installed.txt"
echo "Environment ready: $env_dir. Next run the real-weight smoke test."
