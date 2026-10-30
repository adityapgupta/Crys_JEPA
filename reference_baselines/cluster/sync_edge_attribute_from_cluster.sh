#!/usr/bin/env bash
# Stage edge-attribute outputs in remote home, then download them through xlogin.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE_USER_HOST="${REMOTE_USER_HOST:-userid@xlogin.comp.nus.edu.sg}"
REMOTE_ROOT="${REMOTE_ROOT:-Crys_JEPA}"
SSH_JUMP="${SSH_JUMP:-userid@stujump.comp.nus.edu.sg}"
DRY_RUN="${DRY_RUN:-0}"
ARCHIVE="edge_attribute_results.tar.gz"

stage_command="cd ~/$REMOTE_ROOT/reference_baselines/results && mkdir -p ~/out && tar -czf ~/out/$ARCHIVE edge_attribute_reference_gnn \$(find logs -maxdepth 1 -name 'edge-attribute-*.log' -print)"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'ssh -J %q %q %q\n' "$SSH_JUMP" "$REMOTE_USER_HOST" "$stage_command"
  printf 'scp -J %q %q %q\n' \
    "$SSH_JUMP" "$REMOTE_USER_HOST:~/out/$ARCHIVE" "$PROJECT_ROOT/$ARCHIVE"
  exit 0
fi

ssh -J "$SSH_JUMP" "$REMOTE_USER_HOST" "$stage_command"
scp -J "$SSH_JUMP" "$REMOTE_USER_HOST:~/out/$ARCHIVE" "$PROJECT_ROOT/$ARCHIVE"
tar -xzf "$ARCHIVE" -C reference_baselines/results
rm "$ARCHIVE"
ssh -J "$SSH_JUMP" "$REMOTE_USER_HOST" "rm ~/out/$ARCHIVE"

echo "Download complete. Results are under reference_baselines/results/."
