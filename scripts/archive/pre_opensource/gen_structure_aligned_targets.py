"""Generate aligned structure targets from structure images using Canny + LSD.

For each image in images/structure/, we create a structure map combining:
- Canny edges (general edge detection)
- LSD line segments (structural lines)
- Morphological thickening for visibility

Output: targets/structure_aligned/{stem}.png - 3-channel edge map.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

IMG_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full/images/structure")
OUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full/targets/structure_aligned")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_structure_map(image_path: Path, out_path: Path):
    """Generate aligned structure target from image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "load_failed"
    h, w = img.shape[:2]

    # Step 1: Convert to grayscale and slightly blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Canny edge detection (sparse but accurate)
    # Auto-threshold via median
    med = float(np.median(blurred))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    canny = cv2.Canny(blurred, lower, upper)

    # Step 3: LSD line segment detection for structural lines
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(blurred)[0]
    lsd_canvas = np.zeros((h, w), dtype=np.uint8)
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln
            cv2.line(lsd_canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                     255, thickness=2, lineType=cv2.LINE_AA)

    # Step 4: Dilate both to make them visible at lower res
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    canny_d = cv2.dilate(canny, kernel, iterations=1)
    lsd_d = cv2.dilate(lsd_canvas, kernel, iterations=1)

    # Step 5: Combine (max) - white where either has edge
    combined = cv2.max(canny_d, lsd_d)

    # Step 6: Subtle gradient: edges = 255, near-edges fade
    # Apply distance transform-based weighting
    inv = 255 - combined
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dist = np.clip(dist, 0, 8) / 8.0  # 0..1
    halo = (1.0 - dist) * 200  # near-edges have 200, far 0
    halo = halo.astype(np.uint8)
    combined = np.maximum(combined, halo)

    # Step 7: Convert to 3-channel (model expects RGB-like)
    combined_rgb = np.stack([combined, combined, combined], axis=-1)

    ok = cv2.imwrite(str(out_path), combined_rgb)
    return ok, "ok" if ok else "write_failed"


def main():
    img_files = sorted(IMG_DIR.glob("*.png"))
    print(f"Found {len(img_files)} structure images")

    # Skip already-done
    todo = [p for p in img_files if not (OUT_DIR / p.name).exists()]
    print(f"Need to process: {len(todo)} (skipped {len(img_files) - len(todo)} done)")

    if not todo:
        print("All done.")
        return

    # Parallel process
    n_workers = 8
    print(f"Processing with {n_workers} workers...")
    ok_count, fail_count = 0, 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(make_structure_map, p, OUT_DIR / p.name): p
            for p in todo
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            ok, msg = f.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1

    print(f"Done. ok={ok_count} fail={fail_count}")


if __name__ == "__main__":
    main()
