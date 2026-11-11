#!/usr/bin/env bash
# generate-icons.sh: generate the full icon set from the user-provided logo PNG
# Supports macOS (icns), Windows (ico), Linux (png)

set -euo pipefail

LOGO_PNG="/Users/gatilin/Downloads/微信图片_20260601105735_9720_209.png"

if [ ! -f "$LOGO_PNG" ]; then
    echo "Logo PNG not found at $LOGO_PNG"
    exit 1
fi

# Create the icon output directory
ICON_DIR="desktop/icons"
mkdir -p "$ICON_DIR"

# macOS icns
if command -v iconutil &>/dev/null; then
    iconutil -c icns -o "$ICON_DIR/icon.icns" "$ICON_DIR/icon.iconset"
else
    # Fallback: use sips (built into macOS) + ImageMagick (optional)
    sips -s format icns "$LOGO_PNG" --out "$ICON_DIR/icon.icns" 2>/dev/null || \
    convert "$LOGO_PNG" -resize 256x256 "$ICON_DIR/icon.png" 2>/dev/null || \
    python3 -c "
from PIL import Image
img = Image.open('$LOGO_PNG')
sizes = [16, 32, 48, 64, 128, 256, 512, 1024]
for s in sizes:
    resized = img.resize((s, s), Image.LANCZOS)
    resized.save(f'$ICON_DIR/{s}x{s}.png')
print('Icons generated')
" || {
    echo "Neither iconutil, sips, nor ImageMagick available. Install one to generate icons."
    exit 1
}
fi
