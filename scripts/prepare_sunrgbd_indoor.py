#!/usr/bin/env python3
"""Select official SUN RGB-D training captures, excluding all NYU source images.

Only the original sensor depth/ stream counts as observed. depth_bfx/ is not
used to increase coverage: it contains source-provided depth refinement/filling.
"""

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.hypersim_supplement import file_sha256, write_json
from open_vision_banana.data.indoor_depth import POLICY, read_png, save_pair, sha

MIRROR = "ZHAO11564/SUNRGBD.zip"
REVISION = "4fafeae5f5bad5764cb8f3e89c38a5389ad2b633"
ARCHIVE_SHA = "1a6dbf2a1c9044c4805a35ee648d616ea39a231fd5bd6f77e84cd2b8287fe41c"


def decode_sensor_mm(raw):
    if raw.dtype != np.uint16:
        raise ValueError("Expected original uint16 SUN sensor depth")
    value = raw.astype(np.uint32)
    return ((value >> 3) | (value << 13)) & 65535


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--official-split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    import torch

    torch.set_num_threads(1)
    if file_sha256(a.archive) != ARCHIVE_SHA:
        raise ValueError("SUN source archive changed")
    splits = loadmat(a.official_split, squeeze_me=True)

    def names(values):
        return {"SUNRGBD/" + str(s).split("/SUNRGBD/", 1)[1].rstrip("/") for s in values}

    train = names(splits["alltrain"])
    test = names(splits["alltest"])
    if len(train) != 5285 or len(test) != 5050 or train & test:
        raise ValueError("Unexpected official SUN split")
    zip_reader = zipfile.ZipFile(a.archive)
    image_index = defaultdict(list)
    depth_index = defaultdict(list)
    for name in zip_reader.namelist():
        if "/image/" in name and name.lower().endswith((".jpg", ".png")):
            image_index[name.rsplit("/image/", 1)[0]].append(name)
        elif "/depth/" in name and name.lower().endswith(".png"):
            depth_index[name.rsplit("/depth/", 1)[0]].append(name)
    sources = {}
    rejected = []
    for scene in sorted(train):
        if "nyu" in scene.lower():
            rejected.append(dict(identity=scene, reason="nyu_source_excluded"))
            continue
        images = image_index[scene]
        depths = depth_index[scene]
        if len(images) != 1 or len(depths) != 1:
            rejected.append(dict(identity=scene, reason="ambiguous_or_missing_native_pair"))
            continue
        sources[scene] = (images[0], depths[0])

    def process(item):
        scene, (rn, dn) = item
        rgb_bytes = zip_reader.read(rn)
        depth_bytes = zip_reader.read(dn)
        rgb = read_png(rgb_bytes)
        raw = read_png(depth_bytes)
        depth = decode_sensor_mm(raw).astype(np.float32) / 1000
        identity = "sunrgbd_" + sha(scene.encode())[:24]
        try:
            record = save_pair(
                a.output_dir,
                identity,
                rgb,
                depth,
                depth > 0,
                dict(
                    split="train",
                    source_dataset="sunrgbd_nonnyu",
                    source_scene=scene,
                    source_video=scene,
                    source_split="official_train",
                    source_rgb=rn,
                    source_depth=dn,
                    source_revision=REVISION,
                    source_depth_kind="original_sensor_not_depth_bfx",
                    policy=POLICY,
                ),
            )
            record["source_sha256"] = dict(rgb=sha(rgb_bytes), depth=sha(depth_bytes))
            return scene, record, None
        except ValueError as exc:
            return scene, None, dict(identity=scene, reason=str(exc))

    accepted = []
    archive = dict(
        repo=MIRROR,
        revision=REVISION,
        sha256=ARCHIVE_SHA,
        official_split_sha256=file_sha256(a.official_split),
    )
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for scene, record, error in pool.map(process, sorted(sources.items())):
            if record:
                accepted.append(record)
                identity = record["row"]["sample_id"]
                write_json(
                    a.output_dir / "records" / (identity + ".json"),
                    dict(
                        video=identity,
                        scene=scene,
                        fold="Training",
                        archives=archive,
                        accepted=[record],
                        rejected=[],
                    ),
                )
            else:
                rejected.append(error)
    write_json(
        a.output_dir / "records" / "sunrgbd_exclusions.json",
        dict(
            video="sunrgbd_exclusions",
            scene="",
            fold="Training",
            archives=archive,
            accepted=[],
            rejected=rejected,
        ),
    )
    zip_reader.close()
    report = dict(
        source=MIRROR,
        revision=REVISION,
        archive_sha256=ARCHIVE_SHA,
        official_train_count=len(train),
        official_test_excluded=len(test),
        accepted=len(accepted),
        rejected=len(rejected),
        depth_bfx_used=False,
    )
    write_json(a.output_dir / "sun_source_report.json", report)
    (a.output_dir / "SUN_IMPORT_COMPLETE").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
