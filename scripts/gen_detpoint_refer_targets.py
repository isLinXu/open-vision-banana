#!/usr/bin/env python3
"""Generate RGB targets for the SNVC v5_full detection / keypoint / referring
tasks so the unified trainer can cover more SenseNova-Vision capabilities.

Three task families, all rendered to 512x512 RGB (Vision-Banana style, i.e.
every supervision signal is itself an image):

  object_detection (bbox)  : parse ``<p>cat</p><bbox>[x1,y1,x2,y2]</bbox>``
                             from the assistant turn, give every open-vocabulary
                             category a deterministic colour (md5 -> palette) and
                             draw coloured rectangle outlines via
                             ``ObjectDetectionRGBMapper``.
  keypoint       (point)   : parse ``<p>cat</p><point>[x,y]</point>`` and draw
                             coloured dots via ``KeypointRGBMapper`` (one colour
                             per point, chosen from the point's category).
  referring      (gcg)     : the sampled gcg annotations point at per-phrase mask
                             PNGs that only ship inside the raw
                             ``SenseNova-Vision-Corpus-50M`` pack.  For the
                             Cityscapes subset (the only gcg dataset whose masks
                             are present locally) we render the real ground-truth
                             label map with the PASCAL-VOC palette used by ``seg``.

The input-image location contract mirrors ``.tools/gen_seg_pseudo_labels.py``:

    flat_name = f"{dataset}_{md5(orig_image_path)[:8]}_{basename}"

Inputs are indexed from ``images/detection`` (bbox/point) and
``images/segmentation`` (gcg).  Outputs are absolute-pathed pair rows so the
trainer's ``resolve_manifest_path`` never has to guess a base directory.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path("data/raw/open-vision-banana-snvc-train-full")
RAW_CORPUS = Path(
    "data/raw/SenseNova-Vision-Corpus-50M/"
    "SenseNova-Vision-Corpus-50M/SenseNova-Vision-Corpus-50M"
)
ANN_DIR = ROOT / "annotations"
WORK_RES = 512

# Deterministic open-vocabulary colour palette (HSV rainbow, max separation).
NUM_COLORS = 64


def _build_palette(n: int) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        h, s, v = i / n, 0.85, 1.0 if i % 2 == 0 else 0.75
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        pal[i] = [r, g, b]
    return pal


PALETTE = _build_palette(NUM_COLORS)


def pascal_voc_palette(size: int) -> np.ndarray:
    palette = np.zeros((size, 3), dtype=np.uint8)
    for index in range(size):
        label, bit = index, 0
        while label:
            palette[index, 0] |= ((label >> 0) & 1) << (7 - bit)
            palette[index, 1] |= ((label >> 1) & 1) << (7 - bit)
            palette[index, 2] |= ((label >> 2) & 1) << (7 - bit)
            label >>= 3
            bit += 1
    return palette


CLASS_PALETTE = pascal_voc_palette(150 + 1)[1:]


def flat_name(orig: str, dataset: str) -> str:
    digest = hashlib.md5(orig.encode()).hexdigest()[:8]
    return f"{dataset}_{digest}_{Path(orig).name}"


def get_dataset_name(jsonl_name: str) -> str:
    name = jsonl_name.replace("_sampled.jsonl", "").replace(".jsonl", "")
    for suffix in (
        "_gcg", "_genseg", "_edit_regenerated", "_total_unified",
        "_point", "_bbox", "_visual", "_recon",
    ):
        name = name.replace(suffix, "")
    return name


def cat_color_id(cat: str) -> int:
    return int(hashlib.md5(cat.encode()).hexdigest()[:8], 16) % NUM_COLORS


def index_dir(base: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if not base.is_dir():
        return idx
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            idx.setdefault(p.name, p)
    return idx


# ── coordinate parsing ────────────────────────────────────────────────
_TOKEN_RE = re.compile(
    r"<p>(?P<cat>[^<]*)</p>|<bbox>\s*(?P<bbox>\[[^\]]*\])\s*</bbox>"
    r"|<point>\s*(?P<point>\[[^\]]*\])\s*</point>"
)


def _assistant_text(item: dict) -> str:
    for turn in reversed(item.get("conversations", [])):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def parse_coords(item: dict, kind: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return list of (category, (x1,y1,x2,y2)) normalised.

    For ``kind=="bbox"`` a token is an xyxy box; for ``kind=="point"`` the box
    degenerates to (x, y, x, y).
    """
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    cur = ""
    for m in _TOKEN_RE.finditer(_assistant_text(item)):
        if m.group("cat") is not None:
            cur = m.group("cat").strip()
        elif m.group("bbox") is not None and kind == "bbox":
            try:
                x1, y1, x2, y2 = [float(v) for v in json.loads(m.group("bbox"))][:4]
            except Exception:  # noqa: BLE001
                continue
            out.append((cur, (x1, y1, x2, y2)))
        elif m.group("point") is not None and kind == "point":
            try:
                x, y = [float(v) for v in json.loads(m.group("point"))][:2]
            except Exception:  # noqa: BLE001
                continue
            out.append((cur, (x, y, x, y)))
    return out


# ── rendering ─────────────────────────────────────────────────────────
def render_boxes(recs: list) -> np.ndarray:
    from open_vision_banana.core.object_detection_mapper import ObjectDetectionRGBMapper

    if not recs:
        return np.zeros((WORK_RES, WORK_RES, 3), dtype=np.uint8)
    bboxes, cls_ids = [], []
    for cat, (x1, y1, x2, y2) in recs:
        w = max(x2 - x1, 0.0)
        h = max(y2 - y1, 0.0)
        bboxes.append([x1, y1, w, h])
        cls_ids.append(cat_color_id(cat))
    boxes_t = torch.tensor([bboxes], dtype=torch.float32)
    cls_t = torch.tensor([cls_ids], dtype=torch.long)
    colors = torch.from_numpy(PALETTE)
    mapper = ObjectDetectionRGBMapper(
        num_classes=NUM_COLORS, line_width=3, colors=colors,
        output_height=WORK_RES, output_width=WORK_RES,
    )
    rgb = mapper.bboxes_to_rgb(boxes_t, cls_t, output_size=(WORK_RES, WORK_RES))
    return _to_uint8(rgb[0])


def render_points(recs: list) -> np.ndarray:
    from open_vision_banana.core.keypoint_mapper import KeypointRGBMapper

    if not recs:
        return np.zeros((WORK_RES, WORK_RES, 3), dtype=np.uint8)
    kps, cols = [], []
    for cat, (x, y, _x2, _y2) in recs:
        kps.append([x, y])
        cols.append(PALETTE[cat_color_id(cat)])
    kp_t = torch.tensor([kps], dtype=torch.float32)
    vis_t = torch.ones(1, len(kps), dtype=torch.bool)
    colors = torch.from_numpy(np.asarray(cols, dtype=np.float32))
    mapper = KeypointRGBMapper(
        num_keypoints=len(kps), dot_radius=4, colors=colors,
        output_height=WORK_RES, output_width=WORK_RES,
    )
    rgb = mapper.keypoints_to_rgb(kp_t, vis_t, output_size=(WORK_RES, WORK_RES))
    return _to_uint8(rgb[0])


def _to_uint8(chw: torch.Tensor) -> np.ndarray:
    arr = (chw.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return arr


def render_referring_labelmap(mask_path: Path) -> np.ndarray:
    """Render a Cityscapes gcg label map (single-channel ids) with the seg palette."""
    with Image.open(mask_path) as im:
        arr = np.array(im)
    label = arr[..., 0] if arr.ndim == 3 else arr  # ids stored in R channel
    label = label.astype(np.int64)
    lab_img = Image.fromarray((label % 256).astype(np.uint8), mode="L").resize(
        (WORK_RES, WORK_RES), Image.Resampling.NEAREST
    )
    lab = np.asarray(lab_img, dtype=np.int64)
    pal = CLASS_PALETTE.astype(np.uint8)
    rgb = pal[lab % len(pal)]
    return rgb


# ── pair writer ───────────────────────────────────────────────────────
def split_bucket(path: str) -> int:
    return int(hashlib.sha1(path.encode()).hexdigest()[:8], 16) % 100


def write_pairs(rows: list[dict], out_pairs: Path) -> None:
    rows.sort(key=lambda r: r["input"])
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with out_pairs.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_coord_task(kind: str, task: str, out_targets: Path, out_pairs: Path,
                       val_percent: int, limit: int) -> None:
    mapper_fn = render_boxes if kind == "bbox" else render_points
    img_idx = index_dir(ROOT / "images" / "detection")
    print(f"[{task}] indexed {len(img_idx)} detection images", flush=True)
    out_targets.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    stats: Counter = Counter()
    seen: set[str] = set()
    for ann in sorted(ANN_DIR.glob(f"*_{kind}_sampled.jsonl")):
        ds = get_dataset_name(ann.name)
        with ann.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = item.get("image")
                if not isinstance(src, str):
                    continue
                name = flat_name(src, ds)
                if name in seen:
                    continue
                inp = img_idx.get(name)
                if inp is None:
                    stats["input_missing"] += 1
                    continue
                recs = parse_coords(item, kind)
                if not recs:
                    stats["no_coords"] += 1
                    continue
                seen.add(name)
                tgt = out_targets / (Path(name).stem + ".png")
                if tgt.exists() and tgt.stat().st_size > 100:
                    stats["reused"] += 1
                else:
                    rgb = mapper_fn(recs)
                    tmp = tgt.with_name(f".{tgt.name}.tmp")
                    Image.fromarray(rgb, mode="RGB").save(tmp, format="PNG", compress_level=1)
                    tmp.replace(tgt)
                    stats["written"] += 1
                rows.append({
                    "task": task,
                    "input": str(inp.resolve()),
                    "target": str(tgt.resolve()),
                    "source": src,
                    "annotation": f"render:{kind}",
                    "split": "val" if split_bucket(str(inp)) < val_percent else "train",
                })
                if limit and len(rows) >= limit:
                    break
        if limit and len(rows) >= limit:
            break

    write_pairs(rows, out_pairs)
    report = {"task": task, "pairs": len(rows), "counters": dict(stats),
              "out_targets": str(out_targets.resolve()), "out_pairs": str(out_pairs.resolve())}
    out_pairs.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def process_referring(out_targets: Path, out_pairs: Path, val_percent: int, limit: int) -> None:
    """Render real Cityscapes gcg label maps as referring/grounded-seg targets."""
    out_targets.mkdir(parents=True, exist_ok=True)
    seg_idx = index_dir(ROOT / "images" / "segmentation")
    ann = ANN_DIR / "cityscapes_gcg_sampled.jsonl"
    if not ann.is_file():
        print("[referring] cityscapes_gcg_sampled.jsonl not found; skip", flush=True)
        return
    rows: list[dict] = []
    stats: Counter = Counter()
    seen: set[str] = set()
    with ann.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            src_img, src_seg = item.get("image"), item.get("seg")
            if not isinstance(src_img, str) or not isinstance(src_seg, str):
                continue
            name = flat_name(src_img, "cityscapes")
            if name in seen:
                continue
            inp = seg_idx.get(name)
            if inp is None:
                stats["input_missing"] += 1
                continue
            mask = RAW_CORPUS / src_seg
            if not mask.is_file():
                stats["mask_missing"] += 1
                continue
            seen.add(name)
            tgt = out_targets / (Path(name).stem + ".png")
            if tgt.exists() and tgt.stat().st_size > 100:
                stats["reused"] += 1
            else:
                try:
                    rgb = render_referring_labelmap(mask)
                except Exception as err:  # noqa: BLE001
                    stats["render_error"] += 1
                    if stats["render_error"] <= 5:
                        print(f"  [skip] {name}: {err}", flush=True)
                    continue
                tmp = tgt.with_name(f".{tgt.name}.tmp")
                Image.fromarray(rgb, mode="RGB").save(tmp, format="PNG", compress_level=1)
                tmp.replace(tgt)
                stats["written"] += 1
            rows.append({
                "task": "referring",
                "input": str(inp.resolve()),
                "target": str(tgt.resolve()),
                "source": src_img,
                "annotation": "gcg:cityscapes_gt_labelmap",
                "split": "val" if split_bucket(str(inp)) < val_percent else "train",
            })
            if limit and len(rows) >= limit:
                break

    write_pairs(rows, out_pairs)
    report = {"task": "referring", "pairs": len(rows), "counters": dict(stats),
              "out_targets": str(out_targets.resolve()), "out_pairs": str(out_pairs.resolve())}
    out_pairs.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _auto_canny(grey: np.ndarray) -> np.ndarray:
    """Median-seeded Canny (sigma=0.33) returning a 0/255 uint8 edge map."""
    import cv2

    v = np.median(grey)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    if lower == upper:
        upper = lower + 1
    return cv2.Canny(grey, lower, upper)


def _load_grey512(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        im = im.convert("L").resize((WORK_RES, WORK_RES), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def render_edge_from_grey(grey: np.ndarray) -> np.ndarray:
    """Canny an already-grey 512x512 map -> white-on-black RGB edge target."""
    from open_vision_banana.core.edge_mapper import EdgeRGBMapper

    edges = _auto_canny(grey)
    t = torch.from_numpy(edges.astype(np.float32) / 255.0).view(1, 1, WORK_RES, WORK_RES)
    rgb = EdgeRGBMapper().edge_to_rgb(t)
    return _to_uint8(rgb[0])


def process_edge_tasks(base_pairs: Path, out_edge_targets: Path, out_edge_pairs: Path,
                       out_de_targets: Path, out_de_pairs: Path,
                       val_percent: int, limit: int) -> None:
    """Derive `edge` (Canny on the RGB input) and `depth_edge` (Canny on the
    depth target) from the depth rows of an existing pair manifest."""
    out_edge_targets.mkdir(parents=True, exist_ok=True)
    out_de_targets.mkdir(parents=True, exist_ok=True)
    edge_rows: list[dict] = []
    de_rows: list[dict] = []
    stats: Counter = Counter()
    seen: set[str] = set()

    with base_pairs.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") != "depth":
                continue
            inp, dtarget = r.get("input", ""), r.get("target", "")
            if not (Path(inp).is_file() and Path(dtarget).is_file()):
                stats["src_missing"] += 1
                continue
            stem = Path(inp).stem
            if stem in seen:
                continue
            seen.add(stem)

            etgt = out_edge_targets / f"{stem}.png"
            if etgt.exists() and etgt.stat().st_size > 100:
                stats["edge_reused"] += 1
            else:
                etgt_rgb = render_edge_from_grey(_load_grey512(inp))
                tmp = etgt.with_name(f".{etgt.name}.tmp")
                Image.fromarray(etgt_rgb, mode="RGB").save(tmp, format="PNG", compress_level=1)
                tmp.replace(etgt)
                stats["edge_written"] += 1
            split = "val" if split_bucket(inp) < val_percent else "train"
            edge_rows.append({
                "task": "edge", "input": str(Path(inp).resolve()), "target": str(etgt.resolve()),
                "source": r.get("source", inp), "annotation": "derive:canny_rgb",
                "split": split,
            })

            detgt = out_de_targets / f"{stem}.png"
            if detgt.exists() and detgt.stat().st_size > 100:
                stats["depth_edge_reused"] += 1
            else:
                detgt_rgb = render_edge_from_grey(_load_grey512(dtarget))
                tmp = detgt.with_name(f".{detgt.name}.tmp")
                Image.fromarray(detgt_rgb, mode="RGB").save(tmp, format="PNG", compress_level=1)
                tmp.replace(detgt)
                stats["depth_edge_written"] += 1
            de_rows.append({
                "task": "depth_edge", "input": str(Path(inp).resolve()), "target": str(detgt.resolve()),
                "source": r.get("source", inp), "annotation": "derive:canny_depth",
                "split": split,
            })

            if limit and len(edge_rows) >= limit:
                break

    write_pairs(edge_rows, out_edge_pairs)
    write_pairs(de_rows, out_de_pairs)
    report = {"edge_pairs": len(edge_rows), "depth_edge_pairs": len(de_rows),
              "counters": dict(stats),
              "out_edge_pairs": str(out_edge_pairs.resolve()),
              "out_depth_edge_pairs": str(out_de_pairs.resolve())}
    out_edge_pairs.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _rgb_arr512(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        im = im.convert("RGB").resize((WORK_RES, WORK_RES), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def _degrade_colorize(arr: np.ndarray, rng) -> np.ndarray:
    grey = (arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)


def _degrade_sr(arr: np.ndarray, rng) -> np.ndarray:
    im = Image.fromarray(arr).resize((128, 128), Image.Resampling.BILINEAR)
    im = im.resize((WORK_RES, WORK_RES), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _degrade_deblur(arr: np.ndarray, rng) -> np.ndarray:
    from PIL import ImageFilter

    im = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=3))
    return np.asarray(im, dtype=np.uint8)


def _degrade_inpaint(arr: np.ndarray, rng) -> np.ndarray:
    out = arr.copy()
    for _ in range(rng.randint(3, 5)):
        x = rng.randint(0, WORK_RES - 160)
        y = rng.randint(0, WORK_RES - 160)
        w = rng.randint(80, 160)
        h = rng.randint(80, 160)
        out[y:y + h, x:x + w] = 128
    return out


LOWLEVEL_DEGRADERS = {
    "colorization": _degrade_colorize,
    "super_resolution": _degrade_sr,
    "deblurring": _degrade_deblur,
    "inpainting": _degrade_inpaint,
}


def _collect_source_photos(base_pairs: list[Path], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for bp in base_pairs:
        if not bp.is_file():
            continue
        with bp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                inp = r.get("input", "")
                if inp in seen or not Path(inp).is_file():
                    continue
                seen.add(inp)
                out.append(inp)
                if len(out) >= cap:
                    return out
    return out


def process_lowlevel(base_pairs: list[Path], out_root: Path, pairs_dir: Path,
                     cap: int, val_percent: int) -> None:
    """Synthesise low-level image-to-image restoration tasks by degrading real
    photos.  For each task the target is the clean photo and the input is a
    deterministic degradation; all four share one source photo pool."""
    import random

    photos = _collect_source_photos(base_pairs, cap)
    print(f"[lowlevel] source photo pool: {len(photos)}", flush=True)
    per_task: dict[str, list[dict]] = {t: [] for t in LOWLEVEL_DEGRADERS}
    stats: Counter = Counter()
    for task, deg in LOWLEVEL_DEGRADERS.items():
        (out_root / task).mkdir(parents=True, exist_ok=True)
    for path in photos:
        stem = Path(path).stem
        try:
            arr = _rgb_arr512(path)
        except Exception as err:  # noqa: BLE001
            stats["load_error"] += 1
            if stats["load_error"] <= 5:
                print(f"  [skip] {path}: {err}", flush=True)
            continue
        bucket = "val" if split_bucket(path) < val_percent else "train"
        for task, deg in LOWLEVEL_DEGRADERS.items():
            rng = random.Random(int(hashlib.md5((task + stem).encode()).hexdigest()[:8], 16))
            inp_png = out_root / task / f"{stem}.png"
            if not (inp_png.exists() and inp_png.stat().st_size > 100):
                deg_arr = deg(arr, rng)
                tmp = inp_png.with_name(f".{inp_png.name}.tmp")
                Image.fromarray(deg_arr, mode="RGB").save(tmp, format="PNG", compress_level=1)
                tmp.replace(inp_png)
                stats[f"{task}_written"] += 1
            else:
                stats[f"{task}_reused"] += 1
            per_task[task].append({
                "task": task, "input": str(inp_png.resolve()), "target": str(Path(path).resolve()),
                "source": stem, "annotation": f"derive:{task}", "split": bucket,
            })
    for task, rows in per_task.items():
        write_pairs(rows, pairs_dir / f"pairs_{task}.jsonl")
        print(f"[lowlevel:{task}] pairs={len(rows)}", flush=True)
    report = {"tasks": list(LOWLEVEL_DEGRADERS), "pool": len(photos), "counters": dict(stats),
              "out_root": str(out_root.resolve())}
    (pairs_dir / "pairs_lowlevel.audit.json").write_text(json.dumps(report, indent=2) + "\n")


def process_instance(seg_pairs: Path, out_targets: Path, out_pairs: Path, val_percent: int,
                     limit: int = 0) -> None:
    """Derive pseudo instance segmentation from the semantic pseudo-labels: every
    connected component (per class) is re-coloured so distinct instances of the
    same class get distinct solid colours (README: 'each dog colored differently')."""
    from scipy import ndimage

    out_targets.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stats: Counter = Counter()
    if not seg_pairs.is_file():
        print("[instance] seg pairs not found; skip", flush=True)
        return
    with seg_pairs.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task") != "seg":
                continue
            inp, sem = r.get("input", ""), r.get("target", "")
            if not (Path(inp).is_file() and Path(sem).is_file()):
                stats["src_missing"] += 1
                continue
            stem = Path(inp).stem
            tgt = out_targets / f"{stem}.png"
            if not (tgt.exists() and tgt.stat().st_size > 100):
                sem_arr = _rgb_arr512_nearest(sem)
                inst = _semantic_to_instance(sem_arr, ndimage)
                tmp = tgt.with_name(f".{tgt.name}.tmp")
                Image.fromarray(inst, mode="RGB").save(tmp, format="PNG", compress_level=1)
                tmp.replace(tgt)
                stats["written"] += 1
            else:
                stats["reused"] += 1
            rows.append({
                "task": "instance", "input": str(Path(inp).resolve()), "target": str(tgt.resolve()),
                "source": r.get("source", inp), "annotation": "derive:instance_cc",
                "split": "val" if split_bucket(inp) < val_percent else "train",
            })
            if limit and len(rows) >= limit:
                break
    write_pairs(rows, out_pairs)
    report = {"task": "instance", "pairs": len(rows), "counters": dict(stats),
              "out_pairs": str(out_pairs.resolve())}
    out_pairs.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _rgb_arr512_nearest(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        im = im.convert("RGB").resize((WORK_RES, WORK_RES), Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def _semantic_to_instance(sem_arr: np.ndarray, ndimage) -> np.ndarray:
    out = np.zeros_like(sem_arr)
    flat = sem_arr.reshape(-1, 3)
    # background = most common colour (near-black palette entry); keep it black
    codes, counts = np.unique(flat, axis=0, return_counts=True)
    bg = codes[int(np.argmax(counts))]
    for col in codes:
        if np.array_equal(col, bg):
            continue
        mask = (np.abs(sem_arr.astype(np.int16) - col.astype(np.int16)).sum(-1) <= 6)
        lab, n = ndimage.label(mask)
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() < 64:
                continue
            color = (PALETTE[int(hashlib.md5(f"{col.tobytes()}{k}".encode('latin1')).hexdigest()[:8], 16) % NUM_COLORS] * 255).astype(np.uint8)
            out[comp] = color
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+",
                        default=["object_detection", "keypoint", "referring", "edge", "depth_edge"],
                        choices=["object_detection", "keypoint", "referring", "edge", "depth_edge",
                                 "lowlevel", "instance"])
    parser.add_argument("--lowlevel-cap", type=int, default=3000,
                        help="source photos per low-level task")
    parser.add_argument("--base-pairs", type=Path,
                        default=Path("output/snvc_v5full_pairs/pairs_3task.jsonl"),
                        help="manifest with depth rows used to derive edge / depth_edge")
    parser.add_argument("--lowlevel-source", nargs="+", type=Path,
                        default=[Path("output/snvc_v5full_pairs/pairs_detection.jsonl"),
                                 Path("output/snvc_v5full_pairs/pairs_3task.jsonl")],
                        help="manifests whose input photos seed the low-level task pool")
    parser.add_argument("--out-root", type=Path, default=Path("output/snvc_v5full_pairs"))
    parser.add_argument("--target-root", type=Path, default=Path("output"))
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap pairs per task")
    args = parser.parse_args()

    pairs_dir = args.out_root
    if "object_detection" in args.tasks:
        process_coord_task(
            "bbox", "object_detection",
            args.target_root / "snvc_v5full_detection_targets", pairs_dir / "pairs_detection.jsonl",
            args.val_percent, args.limit,
        )
    if "keypoint" in args.tasks:
        process_coord_task(
            "point", "keypoint",
            args.target_root / "snvc_v5full_keypoint_targets", pairs_dir / "pairs_keypoint.jsonl",
            args.val_percent, args.limit,
        )
    if "referring" in args.tasks:
        process_referring(
            args.target_root / "snvc_v5full_referring_targets", pairs_dir / "pairs_refer.jsonl",
            args.val_percent, args.limit,
        )
    if "edge" in args.tasks or "depth_edge" in args.tasks:
        process_edge_tasks(
            args.base_pairs,
            args.target_root / "snvc_v5full_edge_targets", pairs_dir / "pairs_edge.jsonl",
            args.target_root / "snvc_v5full_depth_edge_targets", pairs_dir / "pairs_depth_edge.jsonl",
            args.val_percent, args.limit,
        )
    if "lowlevel" in args.tasks:
        process_lowlevel(
            args.lowlevel_source,
            args.target_root / "snvc_v5full_lowlevel", pairs_dir,
            args.lowlevel_cap, args.val_percent,
        )
    if "instance" in args.tasks:
        process_instance(
            pairs_dir / "pairs_seg.jsonl",
            args.target_root / "snvc_v5full_instance_targets", pairs_dir / "pairs_instance.jsonl",
            args.val_percent, args.limit,
        )


if __name__ == "__main__":
    main()
