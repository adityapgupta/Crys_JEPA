#!/usr/bin/env bash
# Package the new source files and upload them through the NUS jump host.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

REMOTE_USER_HOST="${REMOTE_USER_HOST:-userid@xlogin.comp.nus.edu.sg}"
REMOTE_ROOT="${REMOTE_ROOT:-Crys_JEPA}"
SSH_JUMP="${SSH_JUMP:-userid@stujump.comp.nus.edu.sg}"
DRY_RUN="${DRY_RUN:-0}"
SOURCE_LIST="reference_baselines/cluster/edge_attribute_source_files.txt"
REMOTE_ARCHIVE="edge_attribute_source.tar.gz"

bundle_dir="$(mktemp -d)"
trap 'rm -rf "$bundle_dir"' EXIT
file_list="$bundle_dir/files.txt"
archive="$bundle_dir/$REMOTE_ARCHIVE"

awk 'NF && $1 !~ /^#/' "$SOURCE_LIST" > "$file_list"
while IFS= read -r file; do
  if [[ ! -f "$file" ]]; then
    echo "Missing source file: $file" >&2
    exit 1
  fi
done < "$file_list"
tar -czf "$archive" -T "$file_list"

echo "Archive: $archive ($(wc -c < "$archive") bytes)"
echo "Destination: $REMOTE_USER_HOST:~/$REMOTE_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'scp -J %q %q %q\n' \
    "$SSH_JUMP" "$archive" "$REMOTE_USER_HOST:~/$REMOTE_ARCHIVE"
  printf 'ssh -J %q %q %q\n' \
    "$SSH_JUMP" "$REMOTE_USER_HOST" \
    "mkdir -p ~/$REMOTE_ROOT && tar -xzf ~/$REMOTE_ARCHIVE -C ~/$REMOTE_ROOT && rm ~/$REMOTE_ARCHIVE"
  exit 0
fi

scp -J "$SSH_JUMP" "$archive" "$REMOTE_USER_HOST:~/$REMOTE_ARCHIVE"
ssh -J "$SSH_JUMP" "$REMOTE_USER_HOST" \
  "mkdir -p ~/$REMOTE_ROOT && tar -xzf ~/$REMOTE_ARCHIVE -C ~/$REMOTE_ROOT && rm ~/$REMOTE_ARCHIVE"

echo "Upload complete. Next: submit the tests stage from ~/$REMOTE_ROOT."
