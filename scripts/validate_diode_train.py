#!/usr/bin/env python3
"""Independently audit native DIODE artifacts against their retained sources.

CPU only. A passing audit of a partial build does not imply source completion.
No training, uploads, packaging or writes to the immutable build artifacts occur.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image

ROLES = ("input", "target", "metric_depth", "valid_mask")
FOLDERS = ("images", "targets", "metric_depth", "valid_masks")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    require(isinstance(name, str) and name != "", "Empty/non-string relative path")
    p = PurePosixPath(name)
    require(
        not p.is_absolute()
        and "\\" not in name
        and ":" not in name
        and all(part not in ("", ".", "..") for part in name.split("/")),
        f"Unsafe relative path: {name}",
    )
    return p


def safe_path(root, name):
    p = safe_name(name)
    candidate = root.joinpath(*p.parts)
    current = root
    for part in p.parts:
        current /= part
        require(not current.is_symlink(), f"Symlink forbidden: {current}")
    require(candidate.resolve().is_relative_to(root.resolve()), "Path escapes build root")
    return candidate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def close_number(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: expected null")
    else:
        require(
            actual is not None and np.allclose(actual, expected, rtol=1e-10, atol=1e-10),
            f"{label}: {actual} != {expected}",
        )


def plane(a):
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    require(a.ndim == 2, "Source plane is not HxW")
    return a


def png(data, mode):
    with Image.open(io.BytesIO(data)) as im:
        require(im.format == "PNG" and im.mode == mode, f"Expected native PNG {mode}")
        return np.array(im)


def source_stats(rgb, depth, source_mask):
    require(source_mask.shape == depth.shape == rgb.shape[:2], "Source geometry mismatch")
    require(
        np.isfinite(source_mask).all() and np.isin(source_mask, [0, 1]).all(),
        "Source mask is not binary",
    )
    valid = (source_mask > 0) & np.isfinite(depth) & (depth > 0)
    metric = np.where(valid, depth, 0).astype(np.float32)
    values = metric[valid]
    stats = dict(
        valid_pixels=int(valid.sum()),
        total_pixels=int(valid.size),
        valid_fraction=float(valid.mean()),
        missing_fraction=float((~valid).mean()),
        source_nonfinite_pixels=int((~np.isfinite(depth)).sum()),
        source_mask_positive_pixels=int((source_mask > 0).sum()),
        depth_min_m=float(values.min()) if values.size else None,
        depth_max_m=float(values.max()) if values.size else None,
        depth_p5_p50_p95_m=np.quantile(values, [0.05, 0.5, 0.95]).tolist() if values.size else None,
        rgb_mean=float(rgb.mean()),
        rgb_std=float(rgb.std()),
    )
    return metric, valid, stats


class Source:
    """Independent source reader; never extract archive members."""

    def __init__(self, config, metadata):
        self.config = config
        self.path = Path(config["input"])
        self.tar = None
        self.members = {}
        self.kind = config["source"]
        if config.get("input_sha256"):
            require(digest(self.path) == config["input_sha256"], "Input file hash mismatch")
        if self.kind == "diode":
            require(metadata is not None, "--official-metadata is required for DIODE validation")
            self.metadata_hash = digest(metadata)
            require(
                self.metadata_hash == config["official_metadata_sha256"], "Metadata hash mismatch"
            )
            meta = read_json(metadata)

            def flatten(tree, prefix=()):
                if isinstance(tree, list):
                    for name in tree:
                        require(len(safe_name(name).parts) == 1, "Bad metadata ID")
                        yield (*prefix, name)
                else:
                    require(isinstance(tree, dict), "Bad metadata tree")
                    for key, value in tree.items():
                        require(len(safe_name(key).parts) == 1, "Bad metadata hierarchy")
                        yield from flatten(value, (*prefix, key))

            official = list(flatten(meta["train"]["indoors"]))
            allowed = {p[-1]: p for p in official}
            held = [p[-1] for p in flatten(meta["val"])]
            require(
                len(allowed) == len(official) == 8574, "Expected 8574 unique official DIODE IDs"
            )
            require(
                len(set(held)) == len(held) and not set(allowed) & set(held),
                "DIODE heldout overlap",
            )
            self.expected = 8574
            if self.path.is_dir():
                for item in self.path.rglob("*"):
                    require(not item.is_symlink(), "Source directory contains symlink")
                    if item.is_file():
                        self.members[item.relative_to(self.path).as_posix()] = item
            else:
                self.tar = tarfile.open(self.path, "r:")  # noqa: SIM115
                for item in self.tar:
                    name = safe_name(item.name).as_posix()
                    require(not item.issym() and not item.islnk(), "Archive link forbidden")
                    if item.isfile():
                        require(name not in self.members, "Duplicate source archive member")
                        self.members[name] = item
            self.ids = {}
            for name in sorted(self.members):
                p = PurePosixPath(name)
                if p.suffix != ".png" or "indoors" not in p.parts:
                    continue
                at = p.parts.index("indoors")
                if at == 0 or p.parts[at - 1] != "train":
                    require("val" in p.parts or "test" in p.parts, "Bad DIODE train hierarchy")
                    continue
                stem = p.stem
                require(
                    stem in allowed and p.parts[at + 1 :] == (*allowed[stem][:-1], stem + ".png"),
                    "DIODE ID/hierarchy is not official indoor train",
                )
                names = (
                    name,
                    str(p.with_name(stem + "_depth.npy")),
                    str(p.with_name(stem + "_depth_mask.npy")),
                )
                require(all(n in self.members for n in names), "Incomplete source triplet")
                identity = "diode_" + stem
                require(identity not in self.ids, "Duplicate source ID")
                self.ids[identity] = names
        else:
            raise ValueError("This auditor supports DIODE only")

    def read(self, name):
        safe_name(name)
        if self.tar:
            with self.tar.extractfile(self.members[name]) as f:
                return f.read()
        return self.members[name].read_bytes()

    def sample(self, identity, row=None):
        names = self.ids[identity]
        blobs = [self.read(n) for n in names]
        if row is not None:
            require(
                tuple(row[k] for k in ("source_rgb", "source_depth", "source_mask")) == names,
                "DIODE source paths mismatch",
            )
            require(row["source_id"] == identity.removeprefix("diode_"), "DIODE source ID mismatch")
            require(
                row["official_metadata_sha256"] == self.metadata_hash, "Row metadata hash mismatch"
            )
            require(
                row["source_sha256"]
                == dict(
                    zip(
                        ("rgb", "depth", "mask"),
                        (hashlib.sha256(b).hexdigest() for b in blobs),
                        strict=True,
                    )
                ),
                "Source triplet hash mismatch",
            )
            if self.tar:
                require(
                    row["source_archive_sha256"] == self.config["input_sha256"],
                    "Archive hash mismatch",
                )
        rgb = png(blobs[0], "RGB")
        require(rgb.shape == (768, 1024, 3), "DIODE native RGB geometry mismatch")
        return rgb, *(plane(np.load(io.BytesIO(b), allow_pickle=False)) for b in blobs[1:])

    def close(self):
        for item in (self.tar,):
            if item is not None:
                item.close()


def encoding_stats(metric, valid, mapper):
    import torch

    from open_vision_banana.data.hypersim_supplement import decode_quantized_rgb

    with torch.no_grad():
        encoded = mapper.depth_to_rgb(torch.from_numpy(metric.copy())[None, None])
        target = (encoded[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        target[~valid] = 0
        decoded = decode_quantized_rgb(target, mapper)
    error = np.abs(decoded[valid].astype(np.float64) - metric[valid])
    require(np.isfinite(error).all(), "Nonfinite encoding error")
    return dict(
        encoding_mae_m=float(error.mean()) if error.size else None,
        encoding_max_error_m=float(error.max()) if error.size else None,
        encoding_abs_error_sum_m=float(error.sum()),
    )


def audit_pair(root, row, source, mapper):
    import torch

    from open_vision_banana.data.hypersim_supplement import decode_quantized_rgb

    rgb = png(safe_path(root, row["input"]).read_bytes(), "RGB")
    target = png(safe_path(root, row["target"]).read_bytes(), "RGB")
    mask = png(safe_path(root, row["valid_mask"]).read_bytes(), "L")
    metric = np.load(safe_path(root, row["metric_depth"]), allow_pickle=False)
    shape = (row["native_height"], row["native_width"])
    require(
        rgb.shape == target.shape == (*shape, 3) and mask.shape == metric.shape == shape,
        "Asset geometry mismatch",
    )
    require(metric.dtype == np.float32, "Metric depth must be float32")
    require(np.isin(mask, [0, 255]).all(), "Mask values must be 0/255")
    valid = mask == 255
    require(
        np.isfinite(metric).all() and (metric[valid] > 0).all() and (metric[~valid] == 0).all(),
        "Invalid metric depth values",
    )
    require((target[~valid] == 0).all(), "Invalid pixels must have black target")
    raw_rgb, raw_depth, raw_mask = source.sample(row["sample_id"], row)
    expected, expected_valid, stats = source_stats(raw_rgb, raw_depth, raw_mask)
    require(np.array_equal(rgb, raw_rgb), "RGB differs from source")
    require(np.array_equal(metric, expected), "Metric depth differs from source")
    require(np.array_equal(valid, expected_valid), "Mask differs from source")
    with torch.no_grad():
        encoded = mapper.depth_to_rgb(torch.from_numpy(metric.copy())[None, None])
        expected_rgb = (encoded[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        expected_rgb[~valid] = 0
        require(
            np.array_equal(target, expected_rgb),
            "Target differs from deterministic existing mapper",
        )
        decoded = decode_quantized_rgb(target, mapper)
    error = np.abs(decoded[valid].astype(np.float64) - metric[valid])
    require(np.isfinite(error).all(), "Nonfinite encoding error")
    stats.update(
        encoding_mae_m=float(error.mean()) if error.size else None,
        encoding_max_error_m=float(error.max()) if error.size else None,
        encoding_abs_error_sum_m=float(error.sum()),
    )
    for key, value in stats.items():
        close_number(row[key], value, key)
    return stats


def audit(root, official_metadata=None):
    import torch

    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    torch.set_num_threads(1)
    result = dict(
        status="failed",
        build_dir=str(root.resolve()),
        failures=[],
        checks={},
        source_complete=None,
        build_complete_marker_verified=False,
        packaging="standalone native build; not res_bundle",
    )
    source = None
    try:
        report = read_json(root / "report.json")
        result["original_build_report"] = report
        result["source_complete"] = report["source_complete"]
        marker = read_json(root / "BUILD_COMPLETE.json")
        require(
            marker["report_sha256"] == digest(root / "report.json"),
            "Completion marker hash mismatch",
        )
        result["build_complete_marker_verified"] = True
        config = read_json(root / "build_config.json")
        require(
            report["build_config_sha256"] == digest(root / "build_config.json"),
            "Config hash mismatch",
        )
        for name, field in (("pairs.jsonl", "pairs_sha256"), ("assets.jsonl", "assets_sha256")):
            require(digest(root / name) == report[field], f"{name} hash mismatch")
        rows = read_jsonl(root / "pairs.jsonl")
        inventory = read_jsonl(root / "assets.jsonl")
        excluded = read_jsonl(root / "exclusions.jsonl")
        rejected = read_jsonl(root / "quality_rejections.jsonl")
        ids = [r["sample_id"] for r in rows]
        require(len(set(ids)) == len(ids), "Duplicate manifest IDs")
        require(all(re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids), "Unsafe sample ID")
        expected_assets = {}
        for row in rows:
            identity = row["sample_id"]
            require(
                row["task"] == "depth"
                and row["split"] == "train"
                and row["source_split"] == "official_train",
                "Non-train depth row",
            )
            require(
                row["depth_units"] == "metres"
                and row["metric_depth_dtype"] == "float32"
                and row["metric_depth_scale"] == 1
                and row["encoding"] == "cube_edge_metric_power"
                and row["encoding_lambda"] == -3
                and row["encoding_c"] == 10 / 3
                and row["loss_mask_applied"] is False,
                "Wrong depth/codec contract",
            )
            record = read_json(safe_path(root, f"records/{identity}.json"))
            require(
                record["row"] == row and set(record["assets"]) == set(ROLES),
                "Record differs from manifest",
            )
            for role, folder in zip(ROLES, FOLDERS, strict=True):
                asset = record["assets"][role]
                suffix = ".npy" if role == "metric_depth" else ".png"
                require(
                    asset["path"] == row[role] == f"{folder}/{identity}{suffix}",
                    "Asset role/path mismatch",
                )
                path = safe_path(root, asset["path"])
                require(
                    path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"],
                    f"Asset hash/size mismatch: {path}",
                )
                expected_assets[(identity, role)] = dict(sample_id=identity, role=role, **asset)
        actual_assets = {(a["sample_id"], a["role"]): a for a in inventory}
        require(
            len(actual_assets) == len(inventory) and actual_assets == expected_assets,
            "Asset inventory missing, duplicated or unexpected entries",
        )
        for folder in (*FOLDERS, "records"):
            actual = {
                p.relative_to(root).as_posix() for p in (root / folder).rglob("*") if p.is_file()
            }
            wanted = {a["path"] for a in inventory if PurePosixPath(a["path"]).parts[0] == folder}
            if folder == "records":
                wanted = {f"records/{i}.json" for i in ids}
            require(actual == wanted, f"Unexpected/missing files in {folder}")
        result["checks"].update(
            asset_hashes=len(inventory), inventory_exact=True, records=len(rows)
        )
        source = Source(config, official_metadata)
        groups = [set(ids), {r["sample_id"] for r in excluded}, {r["sample_id"] for r in rejected}]
        require(
            len(groups[1]) == len(excluded) and len(groups[2]) == len(rejected),
            "Duplicate exclusion/rejection IDs",
        )
        require(
            not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2],
            "Retained/excluded/rejected overlap",
        )
        require(set.union(*groups) == set(source.ids), "Source IDs not completely accounted for")
        require(
            report["source"] == config["source"]
            and report["expected_official_train_pairs"] == source.expected
            and report["source_available_pairs"] == len(source.ids)
            and report["source_complete"] is (len(source.ids) == source.expected),
            "Source completion/count mismatch",
        )
        require(report["source_complete"] or config["allow_partial"], "Unapproved partial build")
        excluded_ids = set()
        for item in config["exclude_manifests"]:
            require(digest(item["path"]) == item["sha256"], "Exclusion manifest changed")
            for row in read_jsonl(item["path"]):
                identity = row.get("sample_id", "")
                if identity.startswith("diode_"):
                    excluded_ids.add(identity)
                dataset = row.get("source_dataset", "")
                if dataset.startswith("diode"):
                    raw = row.get("source_id") or PurePosixPath(row.get("source_rgb", "")).stem
                    if raw:
                        excluded_ids.add("diode_" + raw)
        require(
            groups[1] == excluded_ids & set(source.ids),
            "Exclusions disagree with original manifests",
        )
        require(
            report["unmatched_exclusion_ids"] == sorted(excluded_ids - set(source.ids)),
            "Unmatched exclusions differ",
        )
        threshold = config["min_valid_fraction"]
        require(
            report["min_valid_fraction"] == threshold and 0.98 <= threshold <= 1,
            "Coverage threshold mismatch",
        )
        cap = config.get("max_encoding_error_m")
        require(report.get("max_encoding_error_m") == cap, "Encoding cap report mismatch")
        require(cap is not None and np.isfinite(cap) and 0 <= cap <= 0.05, "Invalid encoding cap")
        mapper = DepthRGBMapper()
        stats = []
        for row in rows:
            try:
                values = audit_pair(root, row, source, mapper)
                require(
                    values["valid_fraction"] >= threshold, "Retained row below coverage threshold"
                )
                require(
                    cap is None or (values["encoding_max_error_m"] or 0) <= cap,
                    "Retained row exceeds encoding error cap",
                )
                stats.append(values)
            except Exception as exc:
                result["failures"].append(f"{row['sample_id']}: {type(exc).__name__}: {exc}")
        for row in rejected:
            rgb, depth, mask = source.sample(row["sample_id"], row)
            metric, valid, values = source_stats(rgb, depth, mask)
            if row["reason"] == "explicit_min_valid_fraction":
                require(values["valid_fraction"] < threshold, "Incorrect coverage rejection")
            else:
                require(
                    row["reason"] == "explicit_max_encoding_error" and cap is not None,
                    "Unknown or unconfigured quality rejection",
                )
                require(
                    values["valid_fraction"] >= threshold,
                    "Coverage gate must precede encoding gate",
                )
                require(row["max_encoding_error_m"] == cap, "Rejection threshold mismatch")
                values.update(encoding_stats(metric, valid, mapper))
                require((values["encoding_max_error_m"] or 0) > cap, "Incorrect encoding rejection")
            for key, value in values.items():
                close_number(row[key], value, key)
        if "quality_rejected_by_reason" in report or "max_encoding_error_m" in config:
            reasons = {}
            for row in rejected:
                reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            require(
                report.get("quality_rejected_by_reason") == reasons,
                "Rejection reason counts mismatch",
            )
        for identity in sorted(groups[1]):
            rgb, depth, mask = source.sample(identity)
            source_stats(rgb, depth, mask)
        result["checks"].update(
            excluded_source_rows_verified=len(groups[1]),
            source_rows_verified=len(stats),
            rejected_source_rows_verified=len(rejected),
            official_ids_accounted=len(source.ids),
            expected_official_ids=source.expected,
        )
        require(not result["failures"], "One or more native pair comparisons failed")
        total = sum(s["valid_pixels"] for s in stats)
        pixels = sum(s["total_pixels"] for s in stats)
        summaries = dict(
            output_pairs=len(rows),
            output_assets=len(inventory),
            excluded_existing_pairs=len(excluded),
            quality_rejected_pairs=len(rejected),
            encoding_pixel_weighted_mae_m=sum(s["encoding_abs_error_sum_m"] for s in stats) / total
            if total
            else None,
            encoding_max_error_m=max((s["encoding_max_error_m"] or 0 for s in stats), default=None),
            missing_pixel_fraction=1 - total / pixels if pixels else None,
            minimum_valid_fraction=min((s["valid_fraction"] for s in stats), default=None),
        )
        for key, value in summaries.items():
            close_number(report[key], value, key)
        require(
            report["excluded_sample_ids"] == [r["sample_id"] for r in excluded],
            "Excluded IDs report differs",
        )
        require(
            report["official_test_or_val_pairs_included"] == 0
            and report["heldout_exclusion_verification"] == "verified",
            "Incorrect heldout report",
        )
        result["statistics"] = dict(summaries, valid_pixels=total, total_pixels=pixels)
        result["checks"]["native_asset_audit_passed"] = True
        result["status"] = "passed"
        result["scope"] = "complete_source" if report["source_complete"] else "partial_source_only"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if source is not None:
            source.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="Native component build directory")
    parser.add_argument("--official-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.build_dir, args.official_metadata)
    output = args.output or args.build_dir / "validation_report.json"
    require(
        output.resolve()
        not in {
            p.resolve()
            for p in args.build_dir.rglob("*")
            if p.is_file() and p.name != "validation_report.json"
        },
        "Refusing to overwrite build/source artifact",
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
