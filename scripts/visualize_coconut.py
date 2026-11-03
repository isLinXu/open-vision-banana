#!/usr/bin/env python3
"""Render deterministic semantic/instance release samples with explicit legends."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pools = {"semantic": [], "instance": []}
    high = {}
    val = {}
    with (args.data_dir / "pairs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            task = r["task"]
            key = hashlib.sha256(("visual42:" + r["sample_id"]).encode()).hexdigest()
            pools[task].append((key, r))
            pools[task].sort(key=lambda x: x[0])
            del pools[task][6:]
            if task not in high or r["source_black_fraction"] > high[task]["source_black_fraction"]:
                high[task] = r
            if r["split"] == "val" and (task not in val or key < val[task][0]):
                val[task] = (key, r)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    selection = {}
    for task, pool in pools.items():
        selected = [r for _, r in pool] + ([val[task][1]] if task in val else []) + [high[task]]
        selected = list({r["sample_id"]: r for r in selected}.values())
        canvas = Image.new("RGB", (1120, 70 + len(selected) * 238), "#f3f4f6")
        d = ImageDraw.Draw(canvas)
        d.text(
            (20, 12),
            f"COCONut-B / OVB {task} training targets (not model predictions)",
            fill="#152333",
            font=font,
        )
        d.text(
            (20, 40),
            "6 deterministic hash samples + validation example + maximum source-black example",
            fill="#526272",
            font=small,
        )
        for i, r in enumerate(selected):
            y = 70 + i * 238
            for j, key in enumerate(("input", "target")):
                image = Image.open(args.data_dir / r[key]).convert("RGB")
                image.thumbnail(
                    (290, 205),
                    Image.Resampling.NEAREST if key == "target" else Image.Resampling.LANCZOS,
                )
                canvas.paste(image, (20 + j * 310, y + 22))
            d.text(
                (20, y),
                f"COCO {r['source_image_id']} | {r['split']} | source black {r['source_black_fraction']:.1%}",
                font=small,
                fill="#152333",
            )
            x = 650
            if task == "semantic":
                for k, (name, color) in enumerate(r["mapping"].items()):
                    dy = y + 5 + k * 16
                    d.rectangle((x, dy, x + 13, dy + 12), fill=tuple(color), outline="#555555")
                    d.text((x + 21, dy - 1), name, fill="#152333", font=small)
            else:
                d.text((x, y + 12), f"Category: {r['category']}", font=font, fill="#152333")
                d.text((x, y + 40), f"Instances: {r['num_instances']}", font=font, fill="#152333")
                d.text(
                    (x, y + 68),
                    "Different solid colors; background black.",
                    font=small,
                    fill="#152333",
                )
            d.line((20, y + 232, 1100, y + 232), fill="#d3d8df")
        canvas.save(args.output_dir / f"{task}_samples.png")
        selection[task] = selected
    (args.output_dir / "sample_records.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
