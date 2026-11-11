#!/usr/bin/env python3
"""Restore licensed benchmark archives or verify separately acquired benchmark roots.

No ML dependencies required. Download the public HF snapshot first; see the guide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in value or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*path.parts)


def check_file(root: Path, record: dict) -> None:
    path = root / relative_path(record["path"])
    if not path.is_file() or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Missing or mismatched file: {path}")


def restore(snapshot: Path, dataset: str, destination: Path) -> None:
    manifest = json.loads((snapshot / "manifests" / f"{dataset}-archive.json").read_text())
    inventory = json.loads((snapshot / "manifests" / f"{dataset}-files.json").read_text())
    expected = {r["path"]: r for r in inventory["files"]}
    destination.mkdir(parents=True, exist_ok=True)
    # Join under the destination filesystem, and verify before extraction.
    with tempfile.TemporaryDirectory(prefix=".ovb-restore-", dir=destination) as temporary:
        temporary = Path(temporary)
        archive = temporary / "joined.tar.gz"
        with archive.open("wb") as output:
            for chunk in manifest["chunks"]:
                check_file(snapshot, chunk)
                with (snapshot / relative_path(chunk["path"])).open("rb") as source:
                    shutil.copyfileobj(source, output, 8 << 20)
        if archive.stat().st_size != manifest["archive_size"] or sha256(archive) != manifest["archive_sha256"]:
            raise ValueError("Joined archive checksum mismatch")
        seen = set()
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                relative = relative_path(member.name)
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                seen.add(member.name)
                record = expected[member.name]
                if member.size != record["size"]:
                    raise ValueError(f"Wrong archive member size: {member.name}")
                target = destination / relative
                # Refuse symlink parents and existing symlink files.
                if any(p.is_symlink() for p in [target, *target.parents]):
                    raise ValueError(f"Symlink destination: {target}")
                if target.exists():
                    check_file(destination, record)  # Never overwrite different data.
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                staged = temporary / "member"
                with handle.extractfile(member) as source, staged.open("wb") as output:
                    shutil.copyfileobj(source, output, 8 << 20)
                if sha256(staged) != record["sha256"]:
                    raise ValueError(f"Wrong archive member checksum: {member.name}")
                staged.replace(target)
        if seen != set(expected):
            raise ValueError("Archive does not contain every inventoried file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nyu", "eth3d", "diode", "cityscapes", "reasonseg", "refcocog"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="Exact adapter root, not its parent")
    parser.add_argument("--restore", action="store_true", help="Extract bundled ETH3D/DIODE; otherwise verify only")
    args = parser.parse_args()
    if args.restore:
        restore(args.snapshot, args.dataset, args.root)
    inventory = json.loads((args.snapshot / "manifests" / f"{args.dataset}-files.json").read_text())
    for record in inventory["files"]:
        check_file(args.root, record)
    print(json.dumps({"dataset": args.dataset, "verified_files": len(inventory["files"]),
                      "samples": inventory["samples"], "status": "verified"}))


if __name__ == "__main__":
    main()
