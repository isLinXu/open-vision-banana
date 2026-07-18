"""Structured diagnostics helpers for inference runs.

The inference API keeps returning tensors and task-specific values, while
this module provides a small JSONL sink for reproducibility and post-hoc
failure analysis.  It deliberately has no logging configuration side effects
and does not depend on optional inference packages.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def _json_safe(value: Any) -> Any:
    """Convert common metadata values into strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # Avoid importing torch just for diagnostics.  Tensor-like values should
    # not normally be passed, but scalar metadata objects can be represented.
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    raise TypeError(f"Unsupported diagnostics value: {type(value).__name__}")


class InferenceDiagnosticsWriter:
    """Append strict JSON records to a UTF-8 JSONL file.

    The instance is callable so it can be passed directly as
    ``diagnostics_callback`` to :meth:`VisionBanana.predict`.
    """

    def __init__(self, path: str | Path, *, flush: bool = True) -> None:
        self.path = Path(path)
        self.flush = flush
        self._handle: TextIO | None = None

    def _ensure_open(self):
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def write(self, record: dict[str, Any]) -> None:
        """Write one record, adding a UTC timestamp if none is supplied."""
        if not isinstance(record, dict):
            raise TypeError(f"diagnostics record must be a dict, got {type(record).__name__}")
        payload = dict(record)
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        handle = self._ensure_open()
        handle.write(json.dumps(_json_safe(payload), allow_nan=False, sort_keys=True) + "\n")
        if self.flush:
            handle.flush()

    __call__ = write

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> InferenceDiagnosticsWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["InferenceDiagnosticsWriter"]
