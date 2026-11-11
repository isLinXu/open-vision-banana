#!/usr/bin/env bash
# ============================================================
# build-release.sh: unified packaging script for the Crablet desktop release
# Supports macOS (Intel + Apple Silicon) / Linux / Windows
# ============================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DESKTOP_DIR="$PROJECT_ROOT/desktop"
BINARIES_DIR="$DESKTOP_DIR/binaries"
OUTPUT_DIR="$PROJECT_ROOT/output"
BUNDLE_DIR="$PROJECT_ROOT/target/release/bundle"

# Version (read from Cargo.toml)
VERSION=$(grep -m1 '^version' "$PROJECT_ROOT/Cargo.toml" | sed 's/.*= "\([^"]*\)".*/\1/')
if [ -z "$VERSION" ]; then
    VERSION="0.1.0"
fi

# Logo path
LOGO_PNG="/Users/gatilin/Downloads/微信图片_20260601105735_9720_209.png"

# Sidecar name
SIDEcar_NAME="crablet"

# Platform detection
OS="$(uname -s)"
ARCH="$(uname -m)"

# Determine the host target
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

# Target platforms
TARGETS=(
    "aarch64-apple-darwin"
    "x86_64-apple-darwin"
    "x86_64-unknown-linux-gnu"
    "aarch64-unknown-linux-gnu"
    "x86_64-pc-windows-msvc"
    "aarch64-pc-windows-msvc"
)

# Build the sidecar binaries
compile_sidecar() {
    local target="$1"
    shift

    echo -e "${GREEN}🔨 Building $target ...${NC}"

    # Check whether the target toolchain is installed
    if ! rustup target list --installed | grep -q "$target"; then
        echo -e "  ${YELLOW}⏭️  Toolchain not installed, skipping $target (run: rustup target add $target)${NC}"
        return 0
    fi

    # Build
    cargo build --release --target "$target" "$@" 2>/dev/null || {
        echo -e "  ${RED}❌ $target build failed (a cross-compilation toolchain may be needed)${NC}"
        return 1
    }

    # Copy the binary
    local src="target/$target/release/$SIDEcar_NAME"
    local dst="$BINARIES_DIR/$SIDEcar_NAME-$target"
    [ -f "$dst" ] || cp "$src" "$dst"
    echo -e "  ${GREEN}✅ $target done${NC}"
}

# For a local build, also create the generic name without a suffix (Tauri looks for it on the current platform)
if [ "$HOST_TARGET" = "aarch64-apple-darwin" ] && [ -f "$BINARIES_DIR/$SIDEcar_NAME-aarch64-apple-darwin" ]; then
    cp "$BINARIES_DIR/$SIDEcar_NAME-aarch64-apple-darwin" "$BINARIES_DIR/$SIDEcar_NAME" 2>/dev/null || true
    echo -e "  ${GREEN}✅ Created generic sidecar copy (no suffix)${NC}"
fi
if [ "$HOST_TARGET" = "x86_64-apple-darwin" ] && [ -f "$BINARIES_DIR/$SIDEcar_NAME-x86_64-apple-darwin" ]; then
    cp "$BINARIES_DIR/$SIDEcar_NAME-x86_64-apple-darwin" "$BINARIES_DIR/$SIDEcar_NAME" 2>/dev/null || true
    echo -e "  ${GREEN}✅ Created generic sidecar copy (no suffix)${NC}"
fi
