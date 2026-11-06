#!/usr/bin/env python3
"""Import an immutable DIODE train mirror, verifying IDs against the official devkit."""

import argparse
import io
import json
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.hypersim_supplement import file_sha256, write_json
from open_vision_banana.data.indoor_depth import POLICY, read_png, save_pair, sha

MIRROR = "sayakpaul/diode-subset-train"
REVISION = "7b2a00c90ab7898d94ecca9300987379b1636fa7"
ARCHIVE_SHA = "393d93089aa47caa3e13e3f108077f4878ff0dbbeec7eea8bbc8f1d3af38a476"
DEVKIT = "8b1765b7d801a5f5e2877c434ffe164e62ce8c90"


def leaves(tree):
    if isinstance(tree, list):
        return set(tree)
    return set().union(*(leaves(v) for v in tree.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--official-metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    import torch

    torch.set_num_threads(1)
    if file_sha256(a.archive) != ARCHIVE_SHA:
        raise ValueError("Mirror archive checksum mismatch")
    metadata = json.loads(a.official_metadata.read_text(encoding="utf-8"))
    allowed = leaves(metadata["train"]["indoors"])
    heldout = leaves(metadata["val"])
    if allowed & heldout:
        raise ValueError("Official IDs overlap")
    with tarfile.open(a.archive, "r:") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
    jobs = []
    for name in members:
        path = PurePosixPath(name)
        if path.suffix != ".png" or path.stem not in allowed or "indoors" not in path.parts:
            continue
        depth = path.with_name(path.stem + "_depth.npy").as_posix()
        mask = path.with_name(path.stem + "_depth_mask.npy").as_posix()
        if depth not in members or mask not in members:
            raise ValueError("Incomplete source triplet")
        jobs.append((name, depth, mask))
    jobs.sort()
    results = defaultdict(list)
    rejects = defaultdict(list)

    def process(names):
        name, dn, mn = names
        stem = PurePosixPath(name).stem
        scene, scan = PurePosixPath(name).parts[-3:-1]
        with tarfile.open(a.archive, "r:") as archive:
            data = {
                key: archive.extractfile(members[n]).read()
                for key, n in zip(("rgb", "depth", "mask"), names, strict=True)
            }
        rgb = read_png(data["rgb"])
        depth = np.load(io.BytesIO(data["depth"]), allow_pickle=False).squeeze().astype(np.float32)
        valid = np.load(io.BytesIO(data["mask"]), allow_pickle=False).squeeze() > 0
        try:
            result = save_pair(
                a.output_dir,
                "diode_" + stem,
                rgb,
                depth,
                valid,
                dict(
                    split="train",
                    source_dataset="diode_indoor",
                    source_scene=scene,
                    source_video=scan,
                    source_split="official_train",
                    source_rgb=name,
                    source_depth=dn,
                    source_mask=mn,
                    source_revision=REVISION,
                    source_devkit_revision=DEVKIT,
                    policy=POLICY,
                    intrinsics=[886.81, 927.06, 512, 384],
                ),
            )
            result["source_sha256"] = {k: sha(v) for k, v in data.items()}
            return (scene, scan), result, None
        except ValueError as exc:
            return (scene, scan), None, dict(identity=stem, reason=str(exc))

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for key, result, rejected in pool.map(process, jobs):
            if result:
                results[key].append(result)
            else:
                rejects[key].append(rejected)
    for scene, scan in sorted(set(results) | set(rejects)):
        video = f"diode_{scene}_{scan}"
        write_json(
            a.output_dir / "records" / (video + ".json"),
            dict(
                video=video,
                scene=scene,
                fold="Training",
                archives=dict(
                    repo=MIRROR,
                    revision=REVISION,
                    sha256=ARCHIVE_SHA,
                    official_metadata_sha256=file_sha256(a.official_metadata),
                ),
                accepted=results[scene, scan],
                rejected=rejects[scene, scan],
            ),
        )
    report = dict(
        source=MIRROR,
        revision=REVISION,
        archive_sha256=ARCHIVE_SHA,
        official_train_ids_checked=True,
        candidate_pairs=len(jobs),
        accepted=sum(map(len, results.values())),
        rejected=sum(map(len, rejects.values())),
    )
    write_json(a.output_dir / "diode_source_report.json", report)
    (a.output_dir / "DIODE_IMPORT_COMPLETE").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
