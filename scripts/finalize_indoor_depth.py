#!/usr/bin/env python3
"""Select diverse, >=98%-observed indoor frames without changing training code."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.hypersim_supplement import file_sha256, write_json
from open_vision_banana.data.indoor_depth import POLICY


def rank(s):
    return hashlib.sha256(("diversity42:" + s).encode()).hexdigest()


def visual_signature(path):
    with Image.open(path) as im:
        return signature_from_gray(im.convert("L"))


def signature_from_gray(gray):
    small = np.array(gray.resize((9, 8), Image.Resampling.BOX), dtype=np.int16)
    thumb = np.array(gray.resize((16, 16), Image.Resampling.BOX), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big"), thumb


def content_statistics(path):
    """Measure spatially distributed structure after suppressing sensor texture/noise."""
    with Image.open(path) as image:
        return content_from_gray(image.convert("L"))


def content_from_gray(gray):
    gray.thumbnail((256, 256), Image.Resampling.BOX)
    smooth = gaussian_filter(np.array(gray, dtype=np.float32), 1)
    gradient = np.hypot(sobel(smooth, axis=0) / 8, sobel(smooth, axis=1) / 8)
    edges = gradient > 5
    starts = []
    for size in edges.shape:
        indices = np.arange(8)
        offsets = indices * (size // 8) + np.minimum(indices, size % 8)
        starts.append(offsets[offsets < size])
    sums = np.add.reduceat(edges, starts[0], axis=0, dtype=np.int64)
    sums = np.add.reduceat(sums, starts[1], axis=1, dtype=np.int64)
    areas = np.outer(
        np.diff(np.r_[starts[0], edges.shape[0]]), np.diff(np.r_[starts[1], edges.shape[1]])
    )
    occupied = np.count_nonzero(sums / areas >= 0.01)
    return dict(edge_density_256=float(edges.mean()), edge_tile_fraction=float(occupied / 64))


def image_features(path):
    """Decode once; retain the same native signature and exact uneven 8x8 grid."""
    with Image.open(path) as image:
        gray = image.convert("L")
        return content_from_gray(gray.copy()), signature_from_gray(gray)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--max-per-scene", type=int, default=384)
    p.add_argument("--max-per-video", type=int, default=128)
    p.add_argument("--development-partial", action="store_true")
    a = p.parse_args()
    root = a.source
    if not a.development_partial and not (root / "ACQUISITION_COMPLETE").exists():
        raise ValueError("Acquisition not complete")
    if (root / "pairs.jsonl").exists():
        raise ValueError("Refuse to replace an existing finalized release")
    (root / "BUILD_INCOMPLETE").touch()
    plan = json.loads((root / "acquisition_plan.json").read_text(encoding="utf-8"))
    if plan["policy"] != POLICY:
        raise ValueError("Mixed cleaning policy")
    if not a.development_partial:
        if plan.get("partial"):
            raise ValueError("A partial acquisition plan cannot become a complete release")
        completed = sum(path.stem.isdigit() for path in (root / "records").glob("*.json"))
        if completed != plan["eligible_videos"]:
            raise ValueError("Incomplete ARKitScenes video inventory")
    groups = defaultdict(lambda: defaultdict(list))
    archives = []
    exclusions = []
    for path in sorted((root / "records").glob("*.json")):
        video = json.loads(path.read_text(encoding="utf-8"))
        archives.append({k: v for k, v in video.items() if k not in ("accepted", "rejected")})
        exclusions.extend(dict(video=video["video"], **r) for r in video["rejected"])
        for record in video["accepted"]:
            row = record["row"]
            if row["policy"] != POLICY or record["statistics"]["valid_fraction"] < 0.98:
                raise ValueError("A candidate violates the 98% policy")
            key = (row["split"], row["source_dataset"], row["source_scene"])
            groups[key][row["source_video"]].append(record)
    train_scenes = {key[1:] for key in groups if key[0] == "train"}
    val_scenes = {key[1:] for key in groups if key[0] == "val"}
    if train_scenes & val_scenes:
        raise ValueError("Physical scene crosses splits")
    seen = set()
    counts = Counter()
    scene_counts = {}
    records = []
    # Held-out rows win any exact-pixel collision; training never absorbs them.
    for key in sorted(groups, key=lambda k: (k[0] != "val", rank(str(k)))):
        videos = groups[key]
        signatures = []
        selected = 0
        per_video = Counter()
        queues = [
            sorted(videos[v], key=lambda r: rank(r["row"]["sample_id"]))
            for v in sorted(videos, key=rank)
        ]
        for wave in zip_longest(*queues):
            for r in wave:
                if r is None:
                    continue
                row = r["row"]
                identity = row["sample_id"]
                video = row["source_video"]
                reason = None
                if r["statistics"]["encoding_max_error"] > 0.05:
                    reason = "encoding_error_above_5cm"
                elif selected >= a.max_per_scene or per_video[video] >= a.max_per_video:
                    reason = "scene_or_video_diversity_cap"
                elif r["rgb_pixels_sha256"] in seen:
                    reason = "duplicate_decoded_rgb"
                else:
                    content, (dh, thumb) = image_features(root / row["input"])
                    r["statistics"].update(content)
                    if content["edge_density_256"] < 0.01 or content["edge_tile_fraction"] < 0.2:
                        exclusions.append(
                            dict(identity=identity, reason="low_visual_structure", **content)
                        )
                        continue
                    similar = [t for h, t in signatures if (dh ^ h).bit_count() <= 4]
                    if similar and (np.abs(np.stack(similar) - thumb).mean(axis=(1, 2)) < 5).any():
                        reason = "near_duplicate_within_scene"
                if reason:
                    exclusions.append(dict(identity=identity, reason=reason))
                    continue
                signatures.append((dh, thumb))
                seen.add(r["rgb_pixels_sha256"])
                selected += 1
                per_video[video] += 1
                records.append(r)
                counts[f"{row['source_dataset']}/{row['split']}"] += 1
        scene_counts["/".join(key)] = selected
        if len(scene_counts) % 100 == 0:
            print(
                json.dumps(dict(processed_scene_groups=len(scene_counts), retained=len(records))),
                flush=True,
            )
    records.sort(key=lambda r: r["row"]["sample_id"])
    if not records:
        raise ValueError("No eligible observations")
    outputs = {
        "pairs.jsonl": [r["row"] for r in records],
        "source_audit.jsonl": records,
        "archives.jsonl": archives,
        "rejected.jsonl": exclusions,
        "assets.jsonl": [
            x for r in records for x in r["assets"] if x["path"] != r["row"]["valid_mask"]
        ],
        "valid_mask_assets.jsonl": [
            x for r in records for x in r["assets"] if x["path"] == r["row"]["valid_mask"]
        ],
    }
    for name, values in outputs.items():
        with (root / name).open("w", encoding="utf-8", newline="\n") as f:
            for value in values:
                f.write(json.dumps(value, sort_keys=True) + "\n")
    policy = dict(
        version=POLICY,
        min_observed_fraction=0.98,
        min_depth_m=0.25,
        max_depth_m=20,
        max_per_scene=a.max_per_scene,
        max_per_video=a.max_per_video,
        near_duplicate_dhash_bits=4,
        near_duplicate_gray_mae=5,
        source_split="official",
        unknown_visits="excluded",
        frame_spacing_seconds=0.5,
        interpolation=False,
        min_edge_density_256=0.01,
        min_structured_tile_fraction=0.2,
        maximum_native_encoding_error_m=0.05,
        training_code_changes=False,
        missing_target="black zero placeholder, not an observed measurement",
        mask_usage="provenance only; unchanged training loader does not apply a loss mask",
    )
    write_json(root / "policy.json", policy)
    write_json(
        root / "provenance.json",
        dict(
            acquisition=plan,
            partial_build=a.development_partial,
            scene_counts=scene_counts,
            metadata_sha256=plan["metadata_sha256"],
            upstream="https://github.com/apple-aiml-research/ARKitScenes",
            upstream_revision="474aaa83d7b1c9080619d8485de2d7c885c01fc5",
        ),
    )
    names = [*outputs, "policy.json", "provenance.json", "acquisition_plan.json"]
    names += [
        n for n in ("diode_source_report.json", "sun_source_report.json") if (root / n).exists()
    ]
    files = {n: dict(bytes=(root / n).stat().st_size, sha256=file_sha256(root / n)) for n in names}
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = dict(
        status="complete",
        encoding=POLICY,
        counts=dict(counts),
        unique_images=len(seen),
        unique_scenes=sum(n > 0 for n in scene_counts.values()),
        asset_count=len(records) * 4,
        files=files,
        dataset_fingerprint=fingerprint,
        rejections=dict(Counter(x["reason"] for x in exclusions)),
    )
    write_json(root / "build_report.json", report)
    (root / "BUILD_INCOMPLETE").unlink()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
