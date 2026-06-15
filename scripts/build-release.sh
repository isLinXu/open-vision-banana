#!/usr/bin/env bash
# ============================================================
# build-release.sh — Crablet 桌面端统一分发打包脚本
# 支持 macOS (Intel + Apple Silicon) / Linux / Windows
# ============================================================

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESKTOP_DIR="$PROJECT_ROOT/desktop"
BINARIES_DIR="$DESKTOP_DIR/binaries"
OUTPUT_DIR="$PROJECT_ROOT/output"
BUNDLE_DIR="$PROJECT_ROOT/target/release/bundle"

# 版本号（从 Cargo.toml 读取）
VERSION=$(grep -m1 '^version' "$PROJECT_ROOT/Cargo.toml" | sed 's/.*= "\([^"]*\)".*/\1/')
if [ -z "$VERSION" ]; then
    VERSION="0.1.0"
fi

# Logo 路径
LOGO_PNG="/Users/gatilin/Downloads/微信图片_20260601105735_9720_209.png"

# Sidecar 名称
SIDEcar_NAME="crablet"

# 平台检测
OS="$(uname -s)"
ARCH="$(uname -m)"

# 确定 host target
if [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        HOST_TARGET="aarch64-apple-darwin"
    else
        HOST_TARGET="x86_64-apple-darwin"
    fi
elif [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "aarch64" ]; then
        HOST_TARGET="aarch64-unknown-linux-gnu"
    else
        HOST_TARGET="x86_64-unknown-linux-gnu"
    fi
fi

# 目标平台列表
TARGETS=(
    "aarch64-apple-darwin"
    "x86_64-apple-darwin"
    "x86_64-unknown-linux-gnu"
    "aarch64-unknown-linux-gnu"
    "x86_64-pc-windows-msvc"
    "aarch64-pc-windows-msvc"
)

# 编译 sidecar 二进制
compile_sidecar() {
    local target="$1"
    shift

    echo -e "${GREEN}🔨 编译 $target ...${NC}"

    # 检查目标工具链是否已安装
    if ! rustup target list --installed | grep -q "$target"; then
        echo -e "  ${YELLOW}⏭️  工具链未安装，跳过 $target (运行: rustup target add $target)${NC}"
        return 0
    fi

    # 编译
    cargo build --release --target "$target" "$@" 2>/dev/null || {
        echo -e "  ${RED}❌ $target 编译失败 (可能需要交叉编译工具链)${NC}"
        return 1
    }

    # 复制二进制
    local src="target/$target/release/$SIDEcar_NAME"
    local dst="$BINARIES_DIR/$SIDEcar_NAME-$target"
    [ -f "$dst" ] || cp "$src" "$dst"
    echo -e "  ${GREEN}✅ $target 完成${NC}"
}

# 如果是本地构建，同时创建无后缀的通用名（Tauri 在当前平台会查找无后缀名）
if [ "$HOST_TARGET" = "aarch64-apple-darwin" ] && [ -f "$BINARIES_DIR/$SIDEcar_NAME-aarch64-apple-darwin" ]; then
    cp "$BINARIES_DIR/$SIDEcar_NAME-aarch64-apple-darwin" "$BINARIES_DIR/$SIDEcar_NAME" 2>/dev/null || true
    echo -e "  ${GREEN}✅ 创建通用 sidecar 副本 (无后缀)${NC}"
fi
if [ "$HOST_TARGET" = "x86_64-apple-darwin" ] && [ -f "$BINARIES_DIR/$SIDEcar_NAME-x86_64-apple-darwin" ]; then
    cp "$BINARIES_DIR/$SIDEcar_NAME-x86_64-apple-darwin" "$BINARIES_DIR/$SIDEcar_NAME" 2>/dev/null || true
    echo -e "  ${GREEN}✅ 创建通用 sidecar 副本 (无后缀)${NC}"
fi
