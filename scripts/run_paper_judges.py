#!/usr/bin/env python3
"""Run the paper's Gemini preprocessing stages from a JSONL manifest.

SA-Co input rows require ``id``, ``image``, and ``noun_phrase``. ReasonSeg
rows require ``id``, ``image``, and ``query``. Relative image paths are
resolved from the manifest directory. Existing output IDs are skipped, making
the command safely resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.evaluation.gemini import (  # noqa: E402
    DEFAULT_PRESENCE_MODEL,
    DEFAULT_REASONING_MODEL,
    GeminiPaperJudge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["saco-presence", "reasonseg-rewrite"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["id"]))
    return completed


def main() -> int:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    judge = GeminiPaperJudge(api_key=api_key, timeout=args.timeout)
    completed = _load_completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8") as source, args.output.open(
        "a", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", line_number))
            if row_id in completed:
                continue
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = args.manifest.parent / image_path

            if args.mode == "saco-presence":
                model = args.model or DEFAULT_PRESENCE_MODEL
                result = {
                    "id": row_id,
                    "pred_present": judge.classify_noun_phrase(
                        image_path, row["noun_phrase"], model=model
                    ),
                    "model": model,
                }
            else:
                model = args.model or DEFAULT_REASONING_MODEL
                result = {
                    "id": row_id,
                    "description": judge.rewrite_reasoning_query(
                        image_path, row["query"], model=model
                    ),
                    "model": model,
                }
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
