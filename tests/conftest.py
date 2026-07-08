"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Ensure src directory is on the path for test discovery
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Ensure project root is on the path so scripts/ imports work in tests
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
