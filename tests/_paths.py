"""Shared paths for tests that exercise standalone scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SCRIPTS = REPO_ROOT / "scripts" / "archive" / "pre_opensource"
