#!/usr/bin/env python3
"""Compose Fig. 4 (a)/(b) multi-expression referring rows from a binary RES manifest.

Paper Fig. 4 (a)/(b) prompt several expressions of one image at once, each rendered
in its own colour ("The stretching cat is rendered in green, the cat that is cleaning
itself is in cyan."). No benchmark scores this, so it is an optional addition to the
training distribution; the single-expression white/black contract used for RefCOCOg
and ReasonSeg evaluation is untouched.

Input: a ``pairs.jsonl`` whose ``referring`` rows carry ``expression`` and a
white-on-black ``target`` (``scripts/build_res_manifest.py`` output, or Rea2Seg rows
with ``--expression-from-prompt``). Rows that share an ``input`` image are grouped;
``k`` of them (2..``--max-expressions``) are drawn, their masks are painted with
distinct named colours onto black (later masks on top, so overlapping regions keep
the last colour and the prompt order matches the paint order), and one new
``referring_multi`` row is written with the Fig. 4 prompt. Only the five tasks of
the paper are evaluated; this task name keeps the rows out of the single-expression
RES decoder unless a config lists it explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.manifest import resolve_manifest_path  # noqa: E402
from open_vision_banana.data.openvocab_semantic import NAMED_COLORS  # noqa: E402
from open_vision_banana.inference.prompt import PromptBuilder  # noqa: E402
from open_vision_banana.referring_contract import BINARY_REFERRING_TEMPLATE  # noqa: E402

TASK_NAME = "referring_multi"
# Named colours the model can both read in a prompt and render unambiguously on black.
MULTI_REFERRING_COLORS = ("white", "green", "cyan", "red", "yellow", "magenta", "blue", "orange")

_PREFIX, _SUFFIX = BINARY_REFERRING_TEMPLATE.split("{expression}")


def expression_from_binary_prompt(prompt: str) -> str | None:
    """Recover the expression from the canonical white/black prompt, else None."""
    if prompt.startswith(_PREFIX) and prompt.endswith(_SUFFIX):
        return prompt[len(_PREFIX) : len(prompt) - len(_SUFFIX)].strip() or None
    return None


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    return arr.mean(axis=-1) > 127


def build_multi_referring_manifest(
    manifest: Path,
    output: Path,
    targets_dir: Path,
    *,
    data_root: Path | None = None,
    max_expressions: int = 3,
    rows_per_image: int = 1,
    min_mask_fraction: float = 0.001,
    expression_from_prompt: bool = False,
    seed: int = 42,
) -> dict:
    """Write ``referring_multi`` rows and return an audit summary."""
    if max_expressions < 2 or max_expressions > len(MULTI_REFERRING_COLORS):
        raise ValueError(f"max_expressions must be in [2, {len(MULTI_REFERRING_COLORS)}]")
    rng = random.Random(seed)
    anchor = data_root.resolve() if data_root else manifest.resolve().parent
    targets_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    stats: dict[str, int] = defaultdict(int)
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "referring":
                continue
            expression = row.get("expression")
            if not expression and expression_from_prompt:
                expression = expression_from_binary_prompt(str(row.get("prompt", "")))
            if not expression:
                stats["skipped_no_expression"] += 1
                continue
            groups[(str(row.get("split", "train")), str(row["input"]))].append(
                dict(row, expression=expression)
            )

    out_rows: list[dict] = []
    for (split, image), rows in sorted(groups.items()):
        # distinct expressions with distinct masks only
        seen: dict[str, dict] = {}
        for row in rows:
            key = str(row["target"])
            if key not in seen and row["expression"] not in {r["expression"] for r in seen.values()}:
                seen[key] = row
        candidates = list(seen.values())
        if len(candidates) < 2:
            stats["skipped_single_expression_images"] += 1
            continue
        image_path = resolve_manifest_path(image, anchor)
        for repeat in range(rows_per_image):
            k = rng.randint(2, min(max_expressions, len(candidates)))
            chosen = rng.sample(candidates, k)
            colors = rng.sample(MULTI_REFERRING_COLORS, k)
            masks = [load_mask(resolve_manifest_path(str(r["target"]), anchor)) for r in chosen]
            height, width = masks[0].shape
            if any(m.shape != (height, width) for m in masks):
                stats["skipped_shape_mismatch"] += 1
                continue
            target = np.zeros((height, width, 3), dtype=np.uint8)
            for mask, color in zip(masks, colors, strict=True):
                target[mask] = NAMED_COLORS[color]
            visible = [
                float(np.all(target == np.asarray(NAMED_COLORS[c], np.uint8), axis=-1).mean())
                for c in colors
            ]
            if min(visible) < min_mask_fraction:
                stats["skipped_occluded_or_tiny"] += 1
                continue
            template_idx = rng.choice((0, 1))
            prompt = PromptBuilder.build_multi_referring_prompt(
                [(r["expression"], c) for r, c in zip(chosen, colors, strict=True)], template_idx
            )
            stem = f"{Path(image).stem}_{repeat}_{k}"
            target_path = targets_dir / f"{stem}.png"
            Image.fromarray(target).save(target_path, compress_level=1)

            def rel(path: Path) -> str:
                try:
                    return str(path.resolve().relative_to(anchor))
                except ValueError:
                    return str(path)

            out_rows.append(
                {
                    "task": TASK_NAME,
                    "split": split,
                    "input": rel(image_path),
                    "target": rel(target_path),
                    "prompt": prompt,
                    "expressions": [r["expression"] for r in chosen],
                    "colors": list(colors),
                    "template_idx": template_idx,
                    "source_rows": [r.get("sample_id") for r in chosen],
                }
            )
            stats[f"rows_with_{k}_expressions"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "output_manifest": str(output.resolve()),
        "rows_written": len(out_rows),
        "images_with_multiple_expressions": sum(1 for rows in groups.values() if len(rows) > 1),
        **{k: int(v) for k, v in sorted(stats.items())},
    }
    output.with_suffix(".multi_referring_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--max-expressions", type=int, default=3)
    parser.add_argument("--rows-per-image", type=int, default=1)
    parser.add_argument("--min-mask-fraction", type=float, default=0.001)
    parser.add_argument(
        "--expression-from-prompt",
        action="store_true",
        help="Recover the expression from the canonical prompt when rows lack an 'expression' field",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_multi_referring_manifest(
        args.manifest,
        args.output,
        args.targets_dir,
        data_root=args.data_root,
        max_expressions=args.max_expressions,
        rows_per_image=args.rows_per_image,
        min_mask_fraction=args.min_mask_fraction,
        expression_from_prompt=args.expression_from_prompt,
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
