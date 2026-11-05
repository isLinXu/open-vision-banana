#!/usr/bin/env python3
"""Resumable, bounded-disk ARKitScenes frame acquisition and metric preparation."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import itertools
import json
import os
import struct
import sys
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.indoor_depth import POLICY, orient, read_png, save_pair, sha

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
MAX_CONFIDENCE_ZIP_BYTES = 1024**2
_sessions = itertools.count()


def rank(s):
    return hashlib.sha256(("indoor42:" + s).encode()).hexdigest()


def stamp(name):
    return float(Path(name).stem.rsplit("_", 1)[1])


def nearest(times, t):
    i = min(bisect.bisect_left(times, t), len(times) - 1)
    return i - 1 if i and abs(times[i - 1] - t) <= abs(times[i] - t) else i


def session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    s.trust_env = False
    routes = [p.strip() for p in os.getenv("INDOOR_DATA_PROXIES", "").split(",") if p.strip()]
    route = routes[next(_sessions) % len(routes)] if routes else os.getenv("INDOOR_DATA_PROXY")
    if route:
        s.proxies = {"https": route}
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        ),
    )
    return s


def read_member(client, archive, info):
    """Fetch a complete local ZIP record once, checking source ETag and ZIP CRC."""
    start = info.header_offset
    end = min(
        archive["bytes"] - 1,
        start + 30 + len(info.filename.encode()) + len(info.extra) + 64 + info.compress_size,
    )
    response = client.get(
        archive["url"],
        headers={"Range": f"bytes={start}-{end}", "If-Match": archive["etag"]},
        timeout=90,
        stream=True,
    )
    response.raise_for_status()
    if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
        f"bytes {start}-"
    ):
        raise ValueError("Source ignored byte range")
    raw = response.raw.read(end - start + 2)
    response.close()
    if len(raw) != end - start + 1:
        raise ValueError("Truncated source byte range")
    fields = struct.unpack("<4s5H3I2H", raw[:30])
    if fields[0] != b"PK\x03\x04" or fields[2] & 1:
        raise ValueError("Invalid/encrypted local ZIP header")
    offset = 30 + fields[-2] + fields[-1]
    if offset + info.compress_size > len(raw):
        raise ValueError("Unexpected large local ZIP extra field")
    compressed = raw[offset : offset + info.compress_size]
    data = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
    if (
        info.compress_type not in (0, 8)
        or len(data) != info.file_size
        or zlib.crc32(data) != info.CRC
    ):
        raise ValueError("ZIP member size/CRC mismatch")
    return data


def load_small_archive(client, archive):
    """Read a bounded confidence ZIP once instead of hundreds of tiny requests."""
    if archive["bytes"] > MAX_CONFIDENCE_ZIP_BYTES:
        raise ValueError("Confidence ZIP exceeds bounded in-memory transport limit")
    response = client.get(
        archive["url"], headers={"If-Match": archive["etag"]}, timeout=90, stream=True
    )
    try:
        response.raise_for_status()
        if response.status_code != 200 or response.headers.get("ETag") != archive["etag"]:
            raise ValueError("Confidence ZIP source changed")
        blob = response.raw.read(archive["bytes"] + 1)
        if len(blob) != archive["bytes"]:
            raise ValueError("Confidence ZIP size mismatch")
    finally:
        response.close()
    return zipfile.ZipFile(io.BytesIO(blob))


def process_video(meta, root, quota):
    from remotezip import RemoteZip

    video, fold = meta["video_id"], meta["fold"]
    dest = root / "records" / f"{video}.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    with session() as s, ExitStack() as stack:
        archives, zips = {}, {}
        for asset in ("lowres_wide", "lowres_depth", "confidence"):
            url = f"{BASE}/{fold}/{video}/{asset}.zip"
            h = s.head(url, timeout=90)
            h.raise_for_status()
            etag = h.headers.get("ETag")
            if not etag:
                raise ValueError("Source archive has no ETag")
            archives[asset] = dict(url=url, etag=etag, bytes=int(h.headers["Content-Length"]))
            if (
                asset == "confidence"
                and quota > 16
                and archives[asset]["bytes"] <= MAX_CONFIDENCE_ZIP_BYTES
            ):
                zips[asset] = stack.enter_context(load_small_archive(s, archives[asset]))
            else:
                zips[asset] = stack.enter_context(
                    RemoteZip(url, session=s, headers={"If-Match": etag}, timeout=90)
                )
        names = {}
        for key, z in zips.items():
            names[key] = sorted((n for n in z.namelist() if n.endswith(".png")), key=stamp)
        if not all(names.values()):
            raise ValueError("Empty source stream")
        depths = [stamp(n) for n in names["lowres_depth"]]
        confidences = {Path(n).name: n for n in names["confidence"]}
        candidates, used = [], set()
        last = -float("inf")
        for name in names["lowres_wide"]:
            t = stamp(name)
            if t - last < 0.5:
                continue
            i = nearest(depths, t)
            dn = names["lowres_depth"][i]
            if abs(depths[i] - t) > 0.008 or dn in used or Path(dn).name not in confidences:
                continue
            candidates.append((name, dn, confidences[Path(dn).name], t, depths[i]))
            last = t
            used.add(dn)
        # Spread the per-video budget across the complete capture before hash ranking.
        order = sorted(range(len(candidates)), key=lambda i: rank(f"{video}:{i}"))
        accepted, rejected = [], []
        for i in order:
            if len(accepted) >= quota:
                break
            rn, dn, cn, rt, dt = candidates[i]
            identity = f"arkit_{video}_{rt:.3f}"
            confidence_zip = zips["confidence"]
            blobs = {
                "confidence": (
                    read_member(s, archives["confidence"], confidence_zip.getinfo(cn))
                    if isinstance(confidence_zip, RemoteZip)
                    else confidence_zip.read(cn)
                )
            }
            conf = orient(read_png(blobs["confidence"]), meta["sky_direction"])
            if (conf >= 1).mean() < 0.98:
                rejected.append(
                    dict(
                        identity=identity,
                        reason="insufficient_observed_depth",
                        confidence_fraction=float((conf >= 1).mean()),
                    )
                )
                continue
            blobs["lowres_depth"] = read_member(
                s, archives["lowres_depth"], zips["lowres_depth"].getinfo(dn)
            )
            raw = orient(read_png(blobs["lowres_depth"]), meta["sky_direction"])
            observed = (raw >= 250) & (raw <= 20000) & (conf >= 1)
            if observed.mean() < 0.98:
                rejected.append(
                    dict(
                        identity=identity,
                        reason="insufficient_observed_depth",
                        observed_fraction=float(observed.mean()),
                    )
                )
                continue
            blobs["lowres_wide"] = read_member(
                s, archives["lowres_wide"], zips["lowres_wide"].getinfo(rn)
            )
            # Network and ZIP CRC failures propagate; they are never cleaning exclusions.
            rgb = orient(read_png(blobs["lowres_wide"]), meta["sky_direction"])
            if raw.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("Unexpected native image type")
            try:
                row = save_pair(
                    root,
                    identity,
                    rgb,
                    raw.astype(np.float32) / 1000,
                    (raw > 0) & (conf >= 1),
                    dict(
                        split="train" if fold == "Training" else "val",
                        source_dataset="arkitscenes",
                        source_scene=meta["visit_id"],
                        source_video=video,
                        source_split=fold,
                        source_rgb=rn,
                        source_depth=dn,
                        source_confidence=cn,
                        source_rgb_timestamp=rt,
                        source_depth_timestamp=dt,
                        timestamp_error_seconds=abs(rt - dt),
                        sky_direction=meta["sky_direction"],
                        policy=POLICY,
                    ),
                )
                row["source_sha256"] = {k: sha(b) for k, b in blobs.items()}
                row["statistics"]["high_confidence_fraction"] = float((conf == 2).mean())
                accepted.append(row)
            except ValueError as exc:
                rejected.append(dict(identity=identity, reason=str(exc)))
        result = dict(
            video=video,
            scene=meta["visit_id"],
            fold=fold,
            archives=archives,
            candidate_count=len(candidates),
            accepted=accepted,
            rejected=rejected,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".tmp")
        temporary.write_text(json.dumps(result) + "\n", encoding="utf-8")
        temporary.replace(dest)
        return result


def main():
    import requests

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-videos", type=int)
    p.add_argument("--train-per-video", type=int, default=512)
    p.add_argument("--val-per-video", type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.metadata.open(encoding="utf-8")))
    heldout = {r["visit_id"] for r in rows if r["fold"] == "Validation"}
    eligible = [
        r
        for r in rows
        if r["visit_id"] != "NA" and (r["fold"] == "Validation" or r["visit_id"] not in heldout)
    ]
    groups = defaultdict(list)
    for row in eligible:
        groups[row["visit_id"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: rank(r["video_id"]))
    scenes = sorted(groups, key=rank)
    eligible = [
        groups[s][i]
        for i in range(max(map(len, groups.values())))
        for s in scenes
        if i < len(groups[s])
    ]
    if a.max_videos:
        eligible = eligible[: a.max_videos]
    plan = dict(
        policy=POLICY,
        metadata_sha256=sha(a.metadata.read_bytes()),
        eligible_videos=len(eligible),
        source_videos=len(rows),
        partial=bool(a.max_videos),
        train_quota=a.train_per_video,
        val_quota=a.val_per_video,
    )
    plan_path = a.output_dir / "acquisition_plan.json"
    if (a.output_dir / "pairs.jsonl").exists():
        raise ValueError("Refuse to acquire into a finalized release")
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Acquisition selection changed; use a new output directory")
    (a.output_dir / "acquisition_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    count = valid = errors = 0
    split_counts = defaultdict(int)
    selected_videos = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = {
            pool.submit(
                process_video,
                r,
                a.output_dir,
                a.train_per_video if r["fold"] == "Training" else a.val_per_video,
            ): r
            for r in eligible
        }
        # All planned videos are processed; the later selector uses fixed hash order.
        for f in as_completed(jobs):
            r = jobs[f]
            try:
                result = f.result()
                valid += len(result["accepted"])
                count += 1
                split_counts[r["fold"]] += len(result["accepted"])
                selected_videos.append(r["video_id"])
                print(
                    json.dumps(
                        dict(done=count, total=len(eligible), pairs=valid, video=r["video_id"])
                    ),
                    flush=True,
                )
            except Exception as exc:
                if isinstance(exc, requests.HTTPError) and exc.response.status_code == 404:
                    result = dict(
                        video=r["video_id"],
                        scene=r["visit_id"],
                        fold=r["fold"],
                        accepted=[],
                        rejected=[dict(reason="official_asset_http_404")],
                    )
                    dest = a.output_dir / "records" / f"{r['video_id']}.json"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(result), encoding="utf-8")
                    count += 1
                    print(
                        json.dumps(
                            dict(unavailable=r["video_id"], reason="official_asset_http_404")
                        ),
                        flush=True,
                    )
                    continue
                errors += 1
                print(
                    json.dumps(
                        dict(error=type(exc).__name__, video=r["video_id"], detail=str(exc)[:200])
                    ),
                    flush=True,
                )
    if errors:
        raise SystemExit(f"{errors} source acquisition failures; resume before publishing")
    (a.output_dir / "acquired_video_ids.json").write_text(
        json.dumps(selected_videos), encoding="utf-8"
    )
    (a.output_dir / "ACQUISITION_COMPLETE").touch()


if __name__ == "__main__":
    main()
