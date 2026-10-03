"""Extensible, offline-first benchmark harness.

Run from a source checkout with ``python -m scripts.benchmark``. Heavy model
dependencies are loaded only when executing a benchmark, never during planning.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

SCHEMA_VERSION = 1
