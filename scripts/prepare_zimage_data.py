#!/usr/bin/env python3
"""Prepare expanded multi-task training data for Z-Image LoRA fine-tuning.

Sources:
  1. COCO2017 train (118K images) — sample 10K for generation + depth tasks
  2. Hypersim (8 scenes) — depth estimation pairs
  3. BSDS500 — edge detection pairs
  4. ADE20K — semantic segmentation pairs
  5. Existing vb_depth_pseudo_5k + vb_generation_5k — reuse

Output layout (matches VisionTaskDataset expectations):
  data/vb_zimage_train/
    depth/
      images/    ← RGB inputs (512×512)
      targets/   ← depth RGB visualizations
    generation/
      images/    ← COCO images for text-to-image
      prompts.jsonl
    segmentation/
      images/    ← RGB inputs
      targets/   ← segmentation color maps
    edge/
      images/    ← RGB inputs
      targets/   ← edge maps

Total target: ~20K samples across 4 tasks (9:1 base:vision ratio).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_zimage_data")

# ─── Config ──────────────────────────────────────────────────────────────
DATASETS_DIR = Path("/Users/gatilin/MyWork/datasets")
OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_zimage_train")
RESOLUTION = 512  # Z-Image native resolution
SEED = 42

# Sampling counts
N_COCO_GENERATION = 8000   # text-to-image from COCO captions
N_COCO_DEPTH = 5000        # depth from COCO (pseudo)
N_HYPERSIM_DEPTH = 2000    # depth from Hypersim (ground truth)
N_ADE20K_SEG = 3000        # semantic segmentation
N_BSDS_EDGE = 2000         # edge detection

# Depth model
DEPTH_MODEL = "LiheYoung/depth-anything-small-hf"


def resize_pair(img: Image.Image, target: Image.Image | None = None) -> tuple:
    """Resize image (and optional target) to RESOLUTION."""
    img = img.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    if target is not None:
        target = target.convert("RGB").resize((RESOLUTION, RESOLUTION), Image.NEAREST)
    return img, target


def load_depth_model():
    """Load Depth-Anything-Small for pseudo depth generation."""
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL, torch_dtype=torch.float32)
    processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info(f"Depth model loaded on {device}")
    return model, processor, device


def estimate_depth(image: Image.Image, model, processor, device) -> np.ndarray:
    """Run depth estimation, return [H, W] normalized to [0.1, 10.0]."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0) if depth.dim() == 3 else depth,
        size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False,
    ).squeeze()
    depth_np = depth.cpu().numpy()
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min > 1e-8:
        depth_np = (depth_np - d_min) / (d_max - d_min)
    else:
        depth_np = np.zeros_like(depth_np)
    return (0.1 + depth_np * 9.9).astype(np.float32)


def depth_to_rgb(depth_np: np.ndarray) -> Image.Image:
    """Convert depth to RGB visualization."""
    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    mapper = DepthRGBMapper(colormap="cube_edge", lam=-3.0, c=10.0 / 3.0)
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
    rgb_tensor = mapper.depth_to_rgb(depth_tensor)
    rgb_np = (rgb_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb_np)


def prepare_coco_generation(coco_dir: Path, output_dir: Path, n: int):
    """Prepare text-to-image generation data from COCO captions."""
    logger.info(f"Preparing {n} COCO generation samples...")
    img_dir = coco_dir / "train2017"
    # Try to find captions
    ann_files = list(coco_dir.glob("*captions*train*.json")) + list(coco_dir.glob("*caption*train*.json"))
    
    captions = {}
    if ann_files:
        import json as j
        with open(ann_files[0]) as f:
            data = j.load(f)
        # Build image_id -> captions mapping
        img_id_map = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            if iid not in captions:
                captions[iid] = []
            captions[iid].append(ann["caption"])
    
    if not captions:
        # Fallback: just use image filenames with generic prompts
        logger.warning("No COCO captions found, using generic prompts")
        img_paths = sorted(img_dir.glob("*.jpg"))
        random.seed(SEED)
        random.shuffle(img_paths)
        selected = img_paths[:n]
        prompts_data = []
        for i, p in enumerate(selected):
            img = Image.open(p).convert("RGB")
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            out_path = output_dir / f"coco_gen_{i:05d}.jpg"
            img.save(out_path, quality=95)
            prompt = "a photograph"
            prompts_data.append({"image": out_path.name, "prompt": prompt})
        return prompts_data
    
    # Sample from captions
    random.seed(SEED)
    img_ids = list(captions.keys())
    random.shuffle(img_ids)
    selected = img_ids[:n]
    
    prompts_data = []
    img_id_map_reversed = {v: k for k, v in img_id_map.items()}
    for i, iid in enumerate(selected):
        fname = img_id_map[iid]
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            out_path = output_dir / f"coco_gen_{i:05d}.jpg"
            img.save(out_path, quality=95)
            # Use first caption
            prompt = captions[iid][0] if captions[iid] else "a photograph"
            prompts_data.append({"image": out_path.name, "prompt": prompt})
        except Exception as e:
            logger.warning(f"Failed to process {fname}: {e}")
    
    logger.info(f"  Prepared {len(prompts_data)} generation samples")
    return prompts_data


def prepare_coco_depth(coco_dir: Path, output_dir: Path, n: int, depth_model):
    """Prepare pseudo depth data from COCO images."""
    model, processor, device = depth_model
    logger.info(f"Preparing {n} COCO depth samples...")
    img_dir = coco_dir / "train2017"
    img_paths = sorted(img_dir.glob("*.jpg"))
    random.seed(SEED + 1)
    random.shuffle(img_paths)
    selected = img_paths[:n]
    
    metadata = []
    for i, p in enumerate(selected):
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            depth_np = estimate_depth(img, model, processor, device)
            depth_rgb = depth_to_rgb(depth_np)
            stem = f"coco_depth_{i:05d}"
            img.save(output_dir / "images" / f"{stem}.jpg", quality=95)
            depth_rgb.save(output_dir / "targets" / f"{stem}.jpg", quality=95)
            metadata.append({"filename": f"{stem}.jpg"})
        except Exception as e:
            logger.warning(f"Failed {p}: {e}")
        if (i + 1) % 200 == 0:
            logger.info(f"  [{i+1}/{len(selected)}]")
    logger.info(f"  Prepared {len(metadata)} depth samples")
    return metadata


def prepare_hypersim_depth(hypersim_dir: Path, output_dir: Path, n: int):
    """Prepare depth data from Hypersim (ground truth depth)."""
    logger.info(f"Preparing {n} Hypersim depth samples...")
    scenes_dir = hypersim_dir / "scenes"
    if not scenes_dir.exists():
        scenes_dir = hypersim_dir
    
    metadata = []
    count = 0
    for scene in sorted(scenes_dir.iterdir()):
        if count >= n:
            break
        # Find image/depth pairs
        rgb_dir = scene / "images" / "final"
        depth_dir = scene / "images" / "depth"
        if not rgb_dir.exists():
            # Try alternate layout
            rgb_dir = scene / "rgb"
            depth_dir = scene / "depth"
        if not rgb_dir.exists():
            continue
        
        rgb_files = sorted(rgb_dir.glob("*.tonemapped.jpg")) + sorted(rgb_dir.glob("*.jpg"))
        depth_files = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.npy"))
        
        for rgb_f, depth_f in zip(rgb_files, depth_files):
            if count >= n:
                break
            try:
                img = Image.open(rgb_f).convert("RGB")
                # Load depth
                if depth_f.suffix == ".npy":
                    depth_np = np.load(depth_f)
                else:
                    # PNG depth (16-bit)
                    depth_img = Image.open(depth_f)
                    depth_np = np.array(depth_img, dtype=np.float32)
                    if depth_np.max() > 100:
                        depth_np = depth_np / 1000.0  # mm -> m
                
                depth_np = np.clip(depth_np, 0.1, 10.0)
                # Resize depth
                h, w = depth_np.shape[:2]
                if h != RESOLUTION or w != RESOLUTION:
                    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
                    depth_tensor = torch.nn.functional.interpolate(
                        depth_tensor, size=(RESOLUTION, RESOLUTION),
                        mode="bilinear", align_corners=False,
                    ).squeeze().numpy()
                    depth_np = depth_tensor
                
                img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
                depth_rgb = depth_to_rgb(depth_np)
                stem = f"hypersim_{scene.name}_{rgb_f.stem}"
                img.save(output_dir / "images" / f"{stem}.jpg", quality=95)
                depth_rgb.save(output_dir / "targets" / f"{stem}.jpg", quality=95)
                metadata.append({"filename": f"{stem}.jpg"})
                count += 1
            except Exception as e:
                logger.warning(f"Failed {rgb_f}: {e}")
    
    logger.info(f"  Prepared {len(metadata)} Hypersim depth samples")
    return metadata


def prepare_ade20k_seg(ade20k_dir: Path, output_dir: Path, n: int):
    """Prepare semantic segmentation data from ADE20K."""
    logger.info(f"Preparing {n} ADE20K segmentation samples...")
    # ADE20K structure: images/training/ + annotations/training/
    img_dir = ade20k_dir / "images" / "training"
    ann_dir = ade20k_dir / "annotations" / "training"
    if not img_dir.exists():
        # Try alternate
        img_dir = ade20k_dir / "training" / "images"
        ann_dir = ade20k_dir / "training" / "labels"
    
    if not img_dir.exists():
        logger.warning(f"ADE20K not found at {ade20k_dir}")
        return []
    
    img_paths = sorted(img_dir.glob("*.jpg"))
    random.seed(SEED + 2)
    random.shuffle(img_paths)
    selected = img_paths[:n]
    
    metadata = []
    for i, p in enumerate(selected):
        ann_p = ann_dir / (p.stem + ".png")
        if not ann_p.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            seg = Image.open(ann_p).convert("RGB")
            img, seg = resize_pair(img, seg)
            stem = f"ade20k_{i:05d}"
            img.save(output_dir / "images" / f"{stem}.jpg", quality=95)
            seg.save(output_dir / "targets" / f"{stem}.png")
            metadata.append({"filename": f"{stem}.jpg"})
        except Exception as e:
            logger.warning(f"Failed {p}: {e}")
    
    logger.info(f"  Prepared {len(metadata)} segmentation samples")
    return metadata


def prepare_bsds_edge(bsds_dir: Path, output_dir: Path, n: int):
    """Prepare edge detection data from BSDS500."""
    logger.info(f"Preparing {n} BSDS500 edge samples...")
    # BSDS500 structure: BSR/BSDS500/data/images/train/ + BSR/BSDS500/data/groundTruth/train/
    img_dir = bsds_dir / "BSR" / "BSDS500" / "data" / "images" / "train"
    edge_dir = bsds_dir / "BSR" / "BSDS500" / "data" / "groundTruth" / "train"
    if not img_dir.exists():
        img_dir = bsds_dir / "images" / "train"
        edge_dir = bsds_dir / "groundTruth" / "train"
    if not img_dir.exists():
        img_dir = bsds_dir / "train" / "images"
        edge_dir = bsds_dir / "train" / "edges"
    
    if not img_dir.exists():
        logger.warning(f"BSDS500 not found at {bsds_dir}")
        return []
    
    img_paths = sorted(img_dir.glob("*.jpg"))
    metadata = []
    for i, p in enumerate(img_paths[:n]):
        # BSDS edges are .mat files
        edge_p = edge_dir / (p.stem + ".mat")
        if not edge_p.exists():
            edge_p = edge_dir / (p.stem + ".png")
        if not edge_p.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            if edge_p.suffix == ".mat":
                import scipy.io as sio
                mat = sio.loadmat(edge_p)
                edge_data = mat["groundTruth"][0][0]["Boundaries"][0][0]
                edge_img = Image.fromarray((edge_data * 255).astype(np.uint8)).convert("RGB")
            else:
                edge_img = Image.open(edge_p).convert("RGB")
            img, edge_img = resize_pair(img, edge_img)
            stem = f"bsds_{i:05d}"
            img.save(output_dir / "images" / f"{stem}.jpg", quality=95)
            edge_img.save(output_dir / "targets" / f"{stem}.png")
            metadata.append({"filename": f"{stem}.jpg"})
        except Exception as e:
            logger.warning(f"Failed {p}: {e}")
    
    logger.info(f"  Prepared {len(metadata)} edge samples")
    return metadata


def copy_existing_data(src_dir: Path, dst_dir: Path, prefix: str):
    """Copy existing training data to new directory with prefix."""
    import shutil
    metadata = []
    img_dir = src_dir / "images"
    tgt_dir = src_dir / "targets"
    if not img_dir.exists():
        return metadata
    for p in sorted(img_dir.iterdir()):
        tgt_p = tgt_dir / p.name
        if not tgt_p.exists():
            continue
        dst_img = dst_dir / "images" / f"{prefix}_{p.name}"
        dst_tgt = dst_dir / "targets" / f"{prefix}_{p.name}"
        shutil.copy2(p, dst_img)
        shutil.copy2(tgt_p, dst_tgt)
        metadata.append({"filename": dst_img.name})
    return metadata


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    # Create output directories
    for task in ["depth", "generation", "segmentation", "edge"]:
        for sub in ["images", "targets"]:
            (OUTPUT_DIR / task / sub).mkdir(parents=True, exist_ok=True)
    
    all_metadata = {}
    
    # 1. Generation: COCO images + captions
    coco_dir = DATASETS_DIR / "coco2017"
    gen_prompts = prepare_coco_generation(coco_dir, OUTPUT_DIR / "generation" / "images", N_COCO_GENERATION)
    # Save prompts
    with open(OUTPUT_DIR / "generation" / "prompts.jsonl", "w") as f:
        for item in gen_prompts:
            f.write(json.dumps(item) + "\n")
    all_metadata["generation"] = gen_prompts
    
    # 2. Depth: COCO pseudo + Hypersim GT
    depth_model = load_depth_model()
    coco_depth_meta = prepare_coco_depth(coco_dir, OUTPUT_DIR / "depth", N_COCO_DEPTH, depth_model)
    hypersim_meta = prepare_hypersim_depth(DATASETS_DIR / "hypersim", OUTPUT_DIR / "depth", N_HYPERSIM_DEPTH)
    all_metadata["depth"] = coco_depth_meta + hypersim_meta
    
    # 3. Segmentation: ADE20K
    seg_meta = prepare_ade20k_seg(DATASETS_DIR / "ade20k", OUTPUT_DIR / "segmentation", N_ADE20K_SEG)
    all_metadata["segmentation"] = seg_meta
    
    # 4. Edge: BSDS500
    edge_meta = prepare_bsds_edge(DATASETS_DIR / "bsds500", OUTPUT_DIR / "edge", N_BSDS_EDGE)
    all_metadata["edge"] = edge_meta
    
    # 5. Copy existing data
    existing_depth = copy_existing_data(
        Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_depth_pseudo_5k"),
        OUTPUT_DIR / "depth", "existing"
    )
    all_metadata["depth"].extend(existing_depth)
    
    # Save summary
    summary = {k: len(v) for k, v in all_metadata.items()}
    summary["total"] = sum(summary.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset summary: {json.dumps(summary, indent=2)}")
    
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump({"summary": summary, "resolution": RESOLUTION, "seed": SEED, "tasks": all_metadata}, f)
    logger.info(f"Metadata saved to {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
