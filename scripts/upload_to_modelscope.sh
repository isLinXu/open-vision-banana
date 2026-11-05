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
  echo "找不到合并产物目录: $MERGED" >&2
  exit 1
fi

REPO_NAME="$(basename "$MERGED")"

echo "合并产物: $MERGED"
echo "将上传到: https://www.modelscope.cn/models/<你的用户名>/$REPO_NAME （默认私有）"
echo
printf 'ModelScope 用户名/组织名: '
read -r OWNER
if [ -z "${OWNER:-}" ]; then
  echo "必须填写用户名，已退出。" >&2
  exit 1
fi

PY="${PYTHON:-python3}"

if ! "$PY" -c 'import modelscope' >/dev/null 2>&1; then
  echo "未检测到 modelscope，正在安装（仅安装 SDK，不改动 torch 等训练依赖）..."
  "$PY" -m pip install -U modelscope
fi

echo
echo "== 第一步：本地校验（不联网、不认证）=="
"$PY" -u "$ROOT/scripts/publish_to_hub.py" \
  --checkpoint "$MERGED" \
  --merged-pipeline \
  --target modelscope \
  --repo-id "$OWNER/$REPO_NAME" \
  --dry-run

echo
printf '文件清单无误则开始正式上传，输入 yes 继续: '
read -r CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "已取消，未上传任何内容。"
  exit 0
fi

echo
echo "== 第二步：正式上传（接下来请输入隐藏 token）=="
"$PY" -u "$ROOT/scripts/publish_to_hub.py" \
  --checkpoint "$MERGED" \
  --merged-pipeline \
  --target modelscope \
  --repo-id "$OWNER/$REPO_NAME" \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo
echo "完成。若建库时报 license 错误：去 modelscope.cn 手动创建私有模型库（许可证选「其他」），"
echo "然后执行 bash scripts/upload_to_modelscope.sh --no-create 重跑，已上传文件会自动跳过。"
