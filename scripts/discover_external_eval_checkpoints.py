#!/usr/bin/env python3
"""Inventory all published I108 checkpoints at immutable HF revisions.

No step ceiling or _best-only filter. Reads repository metadata, never downloads
weights or launches inference. Discovery history is not evaluation acceptance.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

DEFAULT_REPOS = [
    "khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3",
    "khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3",
]


def inventory(repo, revision, files):
    """files maps repository paths to Git blob / LFS metadata, no tensor bytes."""
    checkpoints, unresolved, covered = [], [], set()
    roots = {str(PurePosixPath(*PurePosixPath(p).parts[:2])) for p in files
             if p.startswith("ckpt/") and len(PurePosixPath(p).parts) > 2}
    for path, config in sorted(files.items()):
        if not path.startswith("ckpt/") or not path.endswith("/adapter_config.json"):
            continue
        folder = PurePosixPath(path).parent
        root = str(PurePosixPath(*folder.parts[:2]))
        # The loader prefers safetensors if both serializations are published.
        weight_path = next((str(folder / name) for name in ("adapter_model.safetensors", "adapter_model.bin")
                            if str(folder / name) in files), None)
        if weight_path is None:
            unresolved.append(dict(checkpoint_path=root, adapter_dir=str(folder), reason="adapter weights not published"))
            continue
        weight = files[weight_path]
        if not weight.get("sha256") or not config.get("blob_id"):
            unresolved.append(dict(checkpoint_path=root, adapter_dir=str(folder), reason="missing authoritative weight/config hash"))
            continue
        metadata_path = next((str(candidate) for candidate in
                              (folder / "metadata.json", PurePosixPath(root) / "metadata.json")
                              if str(candidate) in files), None)
        metadata_blob = files[metadata_path].get("blob_id") if metadata_path else None
        identity = dict(repo=repo, weight_sha256=weight["sha256"],
                        config_blob_id=config["blob_id"], metadata_blob_id=metadata_blob)
        content_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        step = re.search(r"checkpoint-(\d+)", root)
        checkpoints.append(dict(repo=repo, revision=revision, checkpoint_path=root,
                                adapter_dir=str(folder), step=int(step[1]) if step else None,
                                weight_path=weight_path, weight_size=weight["size"],
                                adapter_sha256=weight["sha256"], config_blob_id=config["blob_id"],
                                metadata_path=metadata_path, metadata_blob_id=metadata_blob,
                                content_id=content_id,
                                checkpoint_url=f"https://huggingface.co/{repo}/tree/{revision}/{root}",
                                evaluation_status="requires_acceptance_check"))
        covered.add(root)
    for root in sorted(roots - covered - {row["checkpoint_path"] for row in unresolved}):
        unresolved.append(dict(checkpoint_path=root, reason="no complete PEFT adapter; check incomplete publication or another weight layout"))
    return checkpoints, unresolved


def scan(api, repos, previous=None):
    previous = previous or {}
    known = set(previous.get("history_content_ids", []))
    known.update(c["content_id"] for c in previous.get("checkpoints", []))
    old_paths = {(c["repo"], c["adapter_dir"]): c["content_id"] for c in previous.get("checkpoints", [])}
    checkpoints, unresolved, snapshots = [], [], []
    for repo in dict.fromkeys(repos):
        # Resolve HEAD first; every tree entry and resulting download URL is pinned.
        revision = api.model_info(repo).sha
        files = {}
        for entry in api.list_repo_tree(repo, path_in_repo="ckpt", recursive=True, revision=revision):
            if not hasattr(entry, "size"):
                continue
            lfs = getattr(entry, "lfs", None)
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            files[entry.path] = dict(size=entry.size, sha256=sha, blob_id=getattr(entry, "blob_id", None))
        rows, missing = inventory(repo, revision, files)
        for row in rows:
            old = old_paths.get((repo, row["adapter_dir"]))
            row["discovery_change"] = ("seen_content" if row["content_id"] in known else "updated_path" if old else "new_content")
        checkpoints.extend(rows)
        unresolved.extend(dict(repo=repo, revision=revision, **row) for row in missing)
        snapshots.append(dict(repo=repo, revision=revision))
    changes = sorted({c["content_id"] for c in checkpoints} - known)
    history = {}
    for row in [*previous.get("checkpoint_history", []), *previous.get("checkpoints", []), *checkpoints]:
        history.setdefault((row["repo"], row["adapter_dir"], row["content_id"]), row)
    return dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope="all published checkpoint directories, all steps and suffixes; no model inference",
                repositories=snapshots, checkpoints=checkpoints, unresolved_checkpoints=unresolved,
                new_or_updated_content_ids=changes,
                history_content_ids=sorted(known | {c["content_id"] for c in checkpoints}),
                checkpoint_history=list(history.values()),
                note="Seen content is not proof of completed evaluation. Match acceptance records by weights, base, dataset, resolution and scoring protocol before reusing results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="Default: previous contents of --output")
    parser.add_argument("--watch-seconds", type=int, default=0, help="Optional foreground discovery polling; 0 scans once")
    args = parser.parse_args()
    if args.watch_seconds and args.watch_seconds < 60:
        parser.error("--watch-seconds must be 0 or at least 60")
    from huggingface_hub import HfApi
    previous_path = args.previous or args.output
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    while True:
        try:
            result = scan(HfApi(), args.repos, previous)
        except Exception as exc:
            if not args.watch_seconds:
                raise
            print(json.dumps(dict(discovery_error=type(exc).__name__, previous_snapshot_preserved=True)), flush=True)
            time.sleep(args.watch_seconds)
            continue
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(dict(checkpoints=len(result["checkpoints"]),
                              new_or_updated=len(result["new_or_updated_content_ids"]),
                              unresolved=len(result["unresolved_checkpoints"]))), flush=True)
        if not args.watch_seconds:
            return
        previous = result
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
