#!/usr/bin/env python3
"""Fetch pinned COCONut-B source manifests and verified media TARs from ModelScope."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "ab56be89855bd1414a4be2aa98fef35602adc892"
BASE = "https://modelscope.cn/api/v1/datasets/isLinXu/open-vision-banana/repo"
RAW = (
    "training/cleaned_manifests/segmentation/coconut_b_total_unified_genseg_edit_regenerated.jsonl"
)
PAIRS = "image_text_image/manifests/coconut_b.jsonl"
ARCHIVES = "image_text_image/archives/coconut_b"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root, name, expected=None, size=None):
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (
        dest.exists()
        and (size is None or dest.stat().st_size == size)
        and (expected is None or sha256(dest) == expected)
    ):
        return dest
    url = BASE + "?" + urllib.parse.urlencode({"Revision": REVISION, "FilePath": name})
    partial = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, 4 * 1024**2)
            if size is not None and partial.stat().st_size != size:
                raise ValueError(f"Size mismatch: {name}")
            if expected is not None and sha256(partial) != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
            partial.replace(dest)
            print(json.dumps({"downloaded": name, "bytes": dest.stat().st_size}), flush=True)
            return dest
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def extract(path, root, expected_members):
    count = 0
    with tarfile.open(path) as tar:
        for member in tar:
            target = root / member.name
            if not member.isfile() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as inp, target.open("wb") as out:
                shutil.copyfileobj(inp, out)
            count += 1
    if count != expected_members:
        raise ValueError("Archive member count mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in [RAW, PAIRS, ARCHIVES + "/index.jsonl", "README.md"]:
        parent = str(Path(name).parent).replace("\\", "/")
        query = urllib.parse.urlencode(
            {"Revision": REVISION, "Root": "" if parent == "." else parent, "PageSize": 100}
        )
        with urllib.request.urlopen(BASE + "/tree?" + query, timeout=60) as response:
            entries = json.load(response)["Data"]["Files"]
        entry = next(e for e in entries if e["Path"] == name)
        path = fetch(root, name, entry["Sha256"], entry["Size"])
        records.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    shards = [
        json.loads(line) for line in (root / ARCHIVES / "index.jsonl").read_text().splitlines()
    ]

    def acquire(shard):
        if not re.fullmatch(r"shard-\d{5}\.tar", shard["archive"]):
            raise ValueError("Unsafe source shard name")
        path = fetch(root, ARCHIVES + "/" + shard["archive"], shard["sha256"], shard["size_bytes"])
        marker = path.with_suffix(".extracted.json")
        if not marker.exists():
            extract(path, root / "media", shard["files"])
            marker.write_text(json.dumps(shard))
        return shard

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(acquire, shards))
    (root / "source_inventory.json").write_text(
        json.dumps({"revision": REVISION, "files": records, "shards": shards}, indent=2)
    )
    print("SOURCE_READY", flush=True)


if __name__ == "__main__":
    main()
