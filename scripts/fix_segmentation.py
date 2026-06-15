#!/usr/bin/env python3
"""Fix segmentation.py — add strict=True to zip, remove unused _mask variable.

This script applies both fixes at once:
1. Adds strict=True to the zip() call (Python 3.10+ feature)
2. Removes the unused _mask variable that shadows the outer mask
"""

import re
import sys
from pathlib import Path

FILE = Path("src/open_vision_banana/core/segmentation.py")
source = FILE.read_text()

# Fix 1: Add strict=True to zip
source = re.sub(
    r"zip\(unique_labels\.tolist\(\), counts\.tolist\(\)\)",
    r"zip(unique_labels.tolist(), counts.tolist(), strict=True)",
    source,
)

# Fix 2: Remove unused _mask variable and fix NameError
# The _mask = (labels == lbl)... line creates a variable that's never used,
# and it shadows the outer 'mask' name, causing NameError later.
source = re.sub(
    r"_mask = \(labels == lbl\)\.float\(\)\.unsqueeze\(0\)\.unsqueeze\(0\)  # unused",
    "# _mask removed — was unused and shadowed outer 'mask' variable",
    source,
)

# Fix 3: Remove the original unused 'mask' variable line
source = re.sub(
    r"mask = \(labels == lbl\)\.float\(\)\.unsqueeze\(0\)\.unsqueeze\(0\)",
    "# mask removed — was unused",
    source,
)

FILE.write_text(source)
print(f"Fixed {FILE}")
