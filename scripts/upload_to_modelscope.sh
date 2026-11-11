#!/usr/bin/env bash
# One-command ModelScope upload for the validated merged Qwen-Image-2.1 pipeline.
#
# Usage (from the repo root, in a real Terminal so the token prompt can hide input):
#   bash scripts/upload_to_modelscope.sh
#   bash scripts/upload_to_modelscope.sh /path/to/other/merged_dir
#
# It asks for your ModelScope username (repo owner) and then for the SDK token with
# hidden input. The token is never written to disk, argv, or any project file.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -ge 1 ] && [ "${1#-}" = "$1" ]; then
  MERGED="$1"
  shift
else
  MERGED="$ROOT/output/ovb-e2b_cap10k_s42_10000_merged"
fi
# Remaining flags (e.g. --no-create, --public) are forwarded to the python script.
EXTRA=("$@")

if [ ! -d "$MERGED" ]; then
  echo "Merged output directory not found: $MERGED" >&2
  exit 1
fi

REPO_NAME="$(basename "$MERGED")"

echo "Merged output: $MERGED"
echo "Will upload to: https://www.modelscope.cn/models/<your username>/$REPO_NAME (private by default)"
echo
printf 'ModelScope username/organization: '
read -r OWNER
if [ -z "${OWNER:-}" ]; then
  echo "A username is required, exiting." >&2
  exit 1
fi

PY="${PYTHON:-python3}"

if ! "$PY" -c 'import modelscope' >/dev/null 2>&1; then
  echo "modelscope not found, installing it (SDK only, training dependencies such as torch are untouched)..."
  "$PY" -m pip install -U modelscope
fi

echo
echo "== Step 1: local check (no network, no authentication) =="
"$PY" -u "$ROOT/scripts/publish_to_hub.py" \
  --checkpoint "$MERGED" \
  --merged-pipeline \
  --target modelscope \
  --repo-id "$OWNER/$REPO_NAME" \
  --dry-run

echo
printf 'If the file list looks right, type yes to start the upload: '
read -r CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Cancelled, nothing was uploaded."
  exit 0
fi

echo
echo "== Step 2: upload (enter the hidden token next) =="
"$PY" -u "$ROOT/scripts/publish_to_hub.py" \
  --checkpoint "$MERGED" \
  --merged-pipeline \
  --target modelscope \
  --repo-id "$OWNER/$REPO_NAME" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo
echo "Done. If creating the repo fails with a license error, create a private model repo by hand on modelscope.cn (license: Other),"
echo "then rerun bash scripts/upload_to_modelscope.sh --no-create; files already uploaded are skipped."
