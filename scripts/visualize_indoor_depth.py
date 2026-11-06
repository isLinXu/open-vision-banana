#!/usr/bin/env python3
"""Deterministic RGB / observed metric depth / target / validity sample sheets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line) for line in (a.data_dir / "source_audit.jsonl").open(encoding="utf-8")
    ]
    for source in sorted({r["row"]["source_dataset"] for r in records}):
        pool = [r for r in records if r["row"]["source_dataset"] == source]
        selected = []
        seen = set()
        for r in sorted(
            pool,
            key=lambda r: hashlib.sha256(
                ("visual42:" + r["row"]["sample_id"]).encode()
            ).hexdigest(),
        ):
            if r["row"]["source_scene"] in seen:
                continue
            selected.append(r)
            seen.add(r["row"]["source_scene"])
            if len(selected) == 6:
                break
        selected.append(min(pool, key=lambda r: r["statistics"]["valid_fraction"]))
        val = [r for r in pool if r["row"]["split"] == "val"]
        if val:
            selected.append(val[0])
        selected = list({r["row"]["sample_id"]: r for r in selected}.values())
        canvas = Image.new("RGB", (1240, 90 + len(selected) * 248), "#f3f4f6")
        d = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        d.text(
            (15, 10),
            f"OVB {source} observed-depth training pairs | fixed 0.25-20 m scale | missing depth black",
            font=font,
            fill="#182636",
        )
        d.text(
            (15, 40),
            "RGB input                 Observed depth (log gray)     Cube-edge RGB target          Observation mask (white=valid)",
            font=font,
            fill="#182636",
        )
        for x in range(290):
            value = int(x / 289 * 255)
            d.line((320 + x, 59, 320 + x, 65), fill=(value, value, value))
        for value in (0.25, 1, 5, 20):
            x = 320 + int(np.log(value / 0.25) / np.log(80) * 289)
            d.text((x, 67), str(value) + "m", font=font, fill="#182636")
        for i, r in enumerate(selected):
            row = r["row"]
            y = 90 + i * 248
            d.text(
                (15, y),
                f"{row['sample_id']} | {row['split']} | observed {r['statistics']['valid_fraction']:.2%} | scene {row['source_scene'][-45:]}",
                font=font,
                fill="#182636",
            )
            depth = np.array(Image.open(a.data_dir / row["metric_depth"]), dtype=np.float32) / 1000
            mask = np.array(Image.open(a.data_dir / row["valid_mask"])) > 0
            gray = (np.log(np.clip(depth, 0.25, 20) / 0.25) / np.log(80) * 255).astype(np.uint8)
            gray[~mask] = 0
            images = [
                Image.open(a.data_dir / row["input"]),
                Image.fromarray(gray).convert("RGB"),
                Image.open(a.data_dir / row["target"]),
                Image.open(a.data_dir / row["valid_mask"]).convert("RGB"),
            ]
            for j, im in enumerate(images):
                im.thumbnail(
                    (290, 216), Image.Resampling.LANCZOS if j == 0 else Image.Resampling.NEAREST
                )
                canvas.paste(im, (15 + j * 305, y + 24))
            d.line((15, y + 243, 1220, y + 243), fill="#cad2db")
        canvas.save(a.output_dir / f"{source}_samples.png")
        (a.output_dir / f"{source}_sample_records.json").write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
