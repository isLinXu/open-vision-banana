#!/usr/bin/env python3
"""
Download FSC147 images from isentropic/FSC147 on HuggingFace.
JSONL references: FSC147/images_384_VarV2/1868.jpg
HF repo path:     images_384_VarV2/1868.jpg
"""
import json
import subprocess
import time
from pathlib import Path

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
FSC147_DIR = DATASETS_ROOT / "FSC147"
HF_BASE = "https://hf-mirror.com/datasets"
REPO_ID = "isentropic/FSC147"
JSONL_PATH = DATASETS_ROOT / "snvc50m" / "fsc147_bbox.jsonl"


def main():
    image_paths = set()
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img = entry.get("image", "")
            if isinstance(img, list):
                img = img[0] if img else ""
            if img:
                image_paths.add(img)

    print(f"Total unique image paths in JSONL: {len(image_paths)}")
    FSC147_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0
    skipped = 0

    for i, img_path in enumerate(sorted(image_paths)):
        # JSONL: FSC147/images_384_VarV2/1868.jpg
        # Local: /datasets/FSC147/images_384_VarV2/1868.jpg
        local_path = DATASETS_ROOT / img_path

        if local_path.exists() and local_path.stat().st_size > 100:
            skipped += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        # HF repo path: strip "FSC147/" prefix
        hf_path = img_path.replace("FSC147/", "", 1)
        url = f"{HF_BASE}/{REPO_ID}/resolve/main/{hf_path}"

        try:
            result = subprocess.run(
                ["curl", "-s", "-L", "-o", str(local_path), url, "--max-time", "30"],
                capture_output=True,
                timeout=35,
            )

            if local_path.exists() and local_path.stat().st_size > 100:
                downloaded += 1
            else:
                failed += 1
                try:
                    local_path.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    pass
        except Exception:
            failed += 1
            try:
                local_path.unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass

        if (i + 1) % 100 == 0:
            msg = f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(image_paths)}: downloaded={downloaded}, failed={failed}, skipped={skipped}"
            print(msg)
            with open("/tmp/fsc147_dl.log", "a") as log:
                log.write(msg + "\n")

    msg = f"[{time.strftime('%H:%M:%S')}] COMPLETE: downloaded={downloaded}, failed={failed}, skipped={skipped}, total={len(image_paths)}"
    print(msg)
    with open("/tmp/fsc147_dl.log", "a") as log:
        log.write(msg + "\n")


if __name__ == "__main__":
    main()
