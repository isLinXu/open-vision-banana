"""Portable, checksum-verified RES bundles for hosting on the Hugging Face Hub."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath

FORMAT = "ovb_res_bundle_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Reject absolute, traversing and Windows-special paths, including links."""
    posix = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or posix.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in posix.parts
        or any(
            part in {"", "."}
            or part.endswith((".", " "))
            or any(ord(c) < 32 or c in '<>"|?*' for c in part)
            or part.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)],
            }
            for part in name.split("/")
        )
    ):
        raise ValueError(f"Unsafe bundle path: {name!r}")
    path = root.joinpath(*posix.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes root: {name!r}")
    return path


def package_dataset(data_dir: Path, output_dir: Path, shard_bytes: int) -> dict:
    """Package a validated portable dataset; never include unrelated files."""
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    if (data_dir / "BUILD_INCOMPLETE").exists():
        raise ValueError("Cannot package an incomplete RES build")
    build = json.loads((data_dir / "build_report.json").read_text(encoding="utf-8"))
    validation = json.loads((data_dir / "validation_report.json").read_text(encoding="utf-8"))
    if (
        build.get("status") != "complete"
        or validation.get("status") != "passed"
        or build.get("dataset_fingerprint") != validation.get("dataset_fingerprint")
        or build.get("counts") != validation.get("counts")
    ):
        raise ValueError("Package requires a matching successful validation report")
    fingerprint = hashlib.sha256(
        json.dumps(
            build["files"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != build["dataset_fingerprint"]:
        raise ValueError("Dataset metadata fingerprint mismatch")
    for name, entry in build["files"].items():
        path = safe_path(data_dir, name)
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Validated metadata changed: {name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Bundle output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: set[str] = set()
    with (data_dir / "pairs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                for key in ("input", "target"):
                    value = row[key]
                    if value not in assets:
                        if not safe_path(data_dir, value).is_file():
                            raise FileNotFoundError(value)
                        assets.add(value)
    entries = []
    # Reports and manifests only; never copy source pickles or machine config.
    metadata = {*build["files"], "build_report.json", "validation_report.json"}
    for name in sorted(metadata):
        source = safe_path(data_dir, name)
        if (
            source.is_file()
            and source.suffix in {".json", ".jsonl"}
            and source.parent == data_dir.resolve()
        ):
            target = output_dir / source.name
            shutil.copyfile(source, target)
            entries.append(
                {
                    "path": source.name,
                    "sha256": sha256_file(target),
                    "size": target.stat().st_size,
                    "kind": "file",
                }
            )
    shard = None
    tar = None
    used = count = 0
    with (data_dir / "assets.jsonl").open(encoding="utf-8") as handle:
        inventory = {item["path"]: item for item in map(json.loads, handle)}
    if set(inventory) != assets:
        raise ValueError("Manifest assets differ from validated inventory")

    def finish():
        if tar is not None:
            tar.close()
            entries.append(
                {
                    "path": shard.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(shard),
                    "size": shard.stat().st_size,
                    "kind": "tar",
                    "members": count,
                }
            )

    try:
        for name in sorted(assets):
            source = safe_path(data_dir, name)
            size = source.stat().st_size
            if size != inventory[name]["bytes"] or sha256_file(source) != inventory[name]["sha256"]:
                raise ValueError(f"Validated asset changed: {name}")
            if tar is None or (used + size > shard_bytes and count):
                finish()
                shard = (
                    output_dir
                    / "assets"
                    / f"part-{sum(e['kind'] == 'tar' for e in entries):05d}.tar"
                )
                shard.parent.mkdir(exist_ok=True)
                tar = tarfile.open(shard, "w")  # noqa: SIM115 - shards are closed on rollover/finally
                used = count = 0
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = size, 0o644, 0
            with source.open("rb") as asset_stream:
                tar.addfile(info, asset_stream)
            used += size
            count += 1
        finish()
        tar = None
    finally:
        if tar is not None:
            tar.close()
    index = {
        "format": FORMAT,
        "encoding": "binary_white_black_v1",
        "asset_files": len(assets),
        "dataset_fingerprint": build["dataset_fingerprint"],
        "files": entries,
    }
    (output_dir / "bundle.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def restore_bundle(bundle_dir: Path, output_dir: Path) -> dict:
    """Validate all downloads before restoring only regular, contained files."""
    index = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if index.get("format") != FORMAT:
        raise ValueError("Unsupported RES bundle format")
    names = set()
    for item in index["files"]:
        source = safe_path(bundle_dir, item["path"])
        if item["path"].casefold() in names or item.get("kind") not in {"file", "tar"}:
            raise ValueError("Duplicate or unsupported bundle entry")
        names.add(item["path"].casefold())
        if source.stat().st_size != item["size"] or sha256_file(source) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {item['path']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / "BUILD_INCOMPLETE"
    incomplete.write_text("RES bundle restore is incomplete; do not train.\n", encoding="utf-8")
    restored: set[str] = set()

    def copy_checked(name, stream):
        target = safe_path(output_dir, name)
        if name.casefold() in restored:
            raise ValueError(f"Duplicate extracted file: {name}")
        restored.add(name.casefold())
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".res-partial")
        if temporary.exists():
            raise FileExistsError(f"Remove unfinished temporary file before retrying: {temporary}")
        try:
            with temporary.open("xb") as handle:
                shutil.copyfileobj(stream, handle, 1024 * 1024)
            if target.exists():
                if sha256_file(target) != sha256_file(temporary):
                    raise FileExistsError(f"Refusing to replace different existing file: {target}")
                temporary.unlink()
            else:
                temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    assets = 0
    for item in index["files"]:
        source = safe_path(bundle_dir, item["path"])
        if item["kind"] == "file":
            with source.open("rb") as stream:
                copy_checked(item["path"], stream)
        else:
            count = 0
            with tarfile.open(source, "r:") as archive:
                for member in archive:
                    if not member.isfile():
                        raise ValueError(f"Non-regular archive member: {member.name}")
                    with archive.extractfile(member) as member_stream:
                        copy_checked(member.name, member_stream)
                    count += 1
            if count != item["members"]:
                raise ValueError(f"Archive member count mismatch: {item['path']}")
            assets += count
    if assets != index["asset_files"]:
        raise ValueError("Restored asset count does not match bundle")
    incomplete.unlink()
    return {"asset_files": assets, "files": len(restored), "format": FORMAT}
