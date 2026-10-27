#!/usr/bin/env python3
"""Download only selected Hypersim ZIP members and build an audited depth supplement.

No whole-scene archives are downloaded. Per-frame records make interrupted runs
resumable; failed network reads are errors, never data-selection decisions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from remotezip import RemoteZip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_vision_banana.data.hypersim_supplement import (
    ARCHIVE_ROOT,
    METADATA_PATH,
    METADATA_SHA256,
    UPSTREAM_REVISION,
    file_sha256,
    frame_id,
    member_names,
    prepare_frame,
    select_candidates,
    sha256,
    write_json,
)
from open_vision_banana.data.manifest import fingerprint_records


def session() -> requests.Session:
    client = requests.Session()
    client.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        ),
    )
    return client


def process_scene(scene: str, rows: list[dict], output: Path) -> None:
    url = f"{ARCHIVE_ROOT}/{scene}.zip"
    with session() as client:
        head = client.head(url, timeout=(20, 60))
        head.raise_for_status()
        archive = {
            "url": url,
            "etag": head.headers["ETag"],
            "bytes": int(head.headers["Content-Length"]),
        }
        archive_path = output / "archives" / f"{scene}.json"
        if archive_path.exists() and json.loads(archive_path.read_text()) != archive:
            raise ValueError(f"Upstream archive changed: {scene}")
        write_json(archive_path, archive)
        with RemoteZip(
            url, session=client, headers={"If-Match": archive["etag"]}, timeout=(20, 90)
        ) as remote:
            for row in rows:
                identity = frame_id(row)
                record_path = output / "records" / f"{identity}.json"
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if "rejected" in record:
                        continue
                    if all(
                        (output / a["path"]).is_file()
                        and file_sha256(output / a["path"]) == a["sha256"]
                        for a in record["assets"]
                    ):
                        continue
                color_name, depth_name = member_names(row)
                if shutil.disk_usage(output).free < 512 * 1024**2:
                    raise OSError(
                        "Less than 512 MiB free; preserve records and resume after freeing space"
                    )
                # ZIP CRC is checked by ZipFile.read; source hashes are also retained.
                color_bytes, depth_bytes = remote.read(color_name), remote.read(depth_name)
                try:
                    record = prepare_frame(row, color_bytes, depth_bytes, output)
                except ValueError as exc:
                    record = {
                        "source": row,
                        "rejected": str(exc),
                        "source_rgb_sha256": sha256(color_bytes),
                        "source_depth_sha256": sha256(depth_bytes),
                    }
                record["archive"] = archive
                record["members"] = [color_name, depth_name]
                write_json(record_path, record)


def build(
    output: Path,
    *,
    count: int = 20000,
    workers: int = 8,
    metadata: Path | None = None,
    verify_release: Path | None = None,
) -> dict:
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/{UPSTREAM_REVISION}/{METADATA_PATH}"
    metadata_path = output / "upstream_metadata.csv"
    if metadata is not None:
        content = metadata.read_bytes()
    elif metadata_path.exists():
        content = metadata_path.read_bytes()
    else:
        with session() as client:
            response = client.get(metadata_url, timeout=(20, 90))
            response.raise_for_status()
            content = response.content
    if sha256(content) != METADATA_SHA256:
        raise ValueError("Metadata hash differs from the pinned official source")
    candidates = select_candidates(content.decode())
    if len(candidates) < count or count < 1:
        raise ValueError(f"Need {count} eligible unique frames; found {len(candidates)}")
    spec = {
        "schema_version": 1,
        "count": count,
        "metadata_url": metadata_url,
        "metadata_sha256": sha256(content),
        "seed": 42,
        "excluded_volumes": list(range(1, 11)),
        "frame_stride": 2,
        "split": "official_train",
        "eligible_frames": len(candidates),
    }
    spec_path = output / "selection_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Selection specification changed; use a new output directory")
    incomplete = output / "BUILD_INCOMPLETE"
    incomplete.write_text(
        "Preparing or verifying Hypersim assets\n", encoding="utf-8", newline="\n"
    )
    metadata_path.write_bytes(content)
    write_json(spec_path, spec)
    quality_policy = {
        "minimum_valid_depth_fraction": 0.5,
        "maximum_frame_encoded_mae_m": 0.05,
        "maximum_release_encoded_mae_m": 0.01,
        "duplicate_policy": "exact_decoded_rgb",
    }
    quality_path = output / "quality_policy.json"
    if quality_path.exists() and json.loads(quality_path.read_text()) != quality_policy:
        raise ValueError("Quality policy changed; use a new output directory")
    write_json(quality_path, quality_policy)
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_pixels: set[str] = set()
    cursor = 0
    started = time.monotonic()
    while len(accepted) < count:
        batch = candidates[cursor : cursor + count - len(accepted)]
        if not batch:
            raise ValueError(
                f"Only {len(accepted)} valid unique frames; refusing an incomplete release"
            )
        grouped = defaultdict(list)
        for row in batch:
            grouped[row["scene_name"]].append(row)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            failures = []
            futures = {
                pool.submit(process_scene, scene, rows, output): scene
                for scene, rows in grouped.items()
            }
            for number, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"scene": futures[future], "error": str(exc)})
                    print(f"Scene failed (resume required): {futures[future]}: {exc}", flush=True)
                if number % 5 == 0 or number == len(futures):
                    print(
                        f"Scenes {number}/{len(futures)}; elapsed {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
        if failures:
            write_json(output / "download_failures.json", failures)
            raise RuntimeError(f"{len(failures)} scene(s) failed; rerun the same command to resume")
        for row in batch:
            record = json.loads((output / "records" / f"{frame_id(row)}.json").read_text())
            if "rejected" in record:
                rejected.append(record)
            elif record["roundtrip_error_sum"] / record["valid_pixels"] > 0.05:
                rejected.append(
                    {
                        "source": row,
                        "rejected": "frame_encoded_mae_above_5cm",
                        "encoded_mae_m": record["roundtrip_error_sum"] / record["valid_pixels"],
                        "source_rgb_sha256": record["source_rgb_sha256"],
                        "source_depth_sha256": record["source_depth_sha256"],
                    }
                )
            elif record["rgb_pixels_sha256"] in seen_pixels:
                rejected.append(
                    {
                        "source": row,
                        "rejected": "duplicate_decoded_rgb",
                        "rgb_pixels_sha256": record["rgb_pixels_sha256"],
                    }
                )
            else:
                seen_pixels.add(record["rgb_pixels_sha256"])
                accepted.append(record)
        cursor += len(batch)
        print(f"Accepted {len(accepted)}/{count}; rejected {len(rejected)}", flush=True)
    records = sorted(accepted, key=lambda r: r["row"]["sample_id"])
    rows = [r["row"] for r in records]
    valid = sum(r["valid_pixels"] for r in records)
    if sum(r["roundtrip_error_sum"] for r in records) / valid >= 0.01:
        raise ValueError("Release fails the existing 1 cm encoded-depth MAE gate")
    source_fingerprint = sha256(
        json.dumps(
            [
                [r["row"]["sample_id"], r["source_rgb_sha256"], r["source_depth_sha256"]]
                for r in records
            ],
            separators=(",", ":"),
        ).encode()
    )
    if verify_release is not None:
        expected = json.loads(verify_release.read_text(encoding="utf-8"))["hypersim"]
        expected = expected.get("pre_filter", expected)
        if len(rows) != expected["rows"] or source_fingerprint != expected["source_fingerprint"]:
            raise ValueError("Prepared sources differ from the recorded release")
    for name, values in (
        ("pairs.jsonl", rows),
        ("provenance.jsonl", records),
        ("rejected.jsonl", rejected),
    ):
        (output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in values),
            encoding="utf-8",
            newline="\n",
        )
    report = {
        "schema_version": 2,
        "status": "prepared",
        "source": "hypersim",
        "source_revision": UPSTREAM_REVISION,
        "source_fingerprint": source_fingerprint,
        "selection": spec,
        "quality_policy": quality_policy,
        "pairs_written": len(rows),
        "pair_manifest": "pairs.jsonl",
        "manifest_sha256": file_sha256(output / "pairs.jsonl"),
        "provenance_sha256": file_sha256(output / "provenance.jsonl"),
        "dataset_fingerprint": fingerprint_records(rows, output),
        "path_policy": "manifest-relative",
        "output_resolution": "native",
        "unique_images": len(seen_pixels),
        "unique_scenes": len({r["source_scene"] for r in rows}),
        "unique_cameras": len({(r["source_scene"], r["source_camera"]) for r in rows}),
        "scene_counts": dict(sorted(Counter(r["source_scene"] for r in rows).items())),
        "rejected_frames": len(rejected),
        "asset_bytes": sum(a["bytes"] for r in records for a in r["assets"]),
        "mapper": {"colormap": "cube_edge", "lam": -3.0, "c": 10 / 3},
        "metric_depth": {
            "directory": "metric_depth",
            "dtype": "float32",
            "invalid_value": "NaN",
            "convention": "planar_z_depth",
        },
        "tasks": {
            "tasks": ["depth"],
            "rows_per_task": {"depth": len(rows)},
            "normal_backfacing_frames": 0,
        },
        "data_contract": {
            "valid_pixels": valid,
            "invalid_pixels": sum(r["invalid_pixels"] for r in records),
            "depth_meters": {
                "min": min(r["depth_min"] for r in records),
                "max": max(r["depth_max"] for r in records),
                "mean": sum(r["depth_sum"] for r in records) / valid,
            },
            "encoded_roundtrip_abs_error": {
                "mean": sum(r["roundtrip_error_sum"] for r in records) / valid,
                "max": max(r["roundtrip_error_max"] for r in records),
            },
        },
        "verification_scope": "All native RGB/depth pairs, complete encoded roundtrips, unique decoded RGB; no GPU training or accuracy claim",
    }
    write_json(output / "manifest.json", report)
    incomplete.unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--verify-release",
        type=Path,
        help="Check identities and raw payload hashes against a release JSON",
    )
    args = parser.parse_args()
    result = build(
        args.output,
        count=args.count,
        workers=args.workers,
        metadata=args.metadata,
        verify_release=args.verify_release,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "pairs_written",
                    "unique_scenes",
                    "asset_bytes",
                    "data_contract",
                )
            },
            indent=2,
        )
    )
