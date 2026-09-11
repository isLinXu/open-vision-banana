#!/usr/bin/env python3
"""Extract sayakpaul/nyu_depth_v2 webdataset shards (.h5) into the project's
NYUDepthDataset layout:  root/images/<id>.jpg  +  root/depths/<id>.npy (meters).
"""
import io
import sys
from pathlib import Path

import numpy as np
import tarfile
from PIL import Image

ROOT = Path("data/benchmarks/nyu")
RAW = ROOT / "raw"
OUT = ROOT / "extracted"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "images").mkdir(exist_ok=True)
(OUT / "depths").mkdir(exist_ok=True)


def main():
    import h5py
    files = sorted(RAW.glob("*.tar"))
    n = 0
    for tar_path in files:
        tf = tarfile.open(tar_path)
        for name in tf.getnames():
            if not name.endswith(".h5"):
                continue
            stem = Path(name).stem  # e.g. 00001
            ff = h5py.File(io.BytesIO(tf.extractfile(name).read()), "r")
            rgb = np.array(ff["rgb"]).astype(np.uint8)      # (3,480,640)
            depth = np.array(ff["depth"]).astype(np.float32)  # (480,640) meters
            img = Image.fromarray(rgb.transpose(1, 2, 0), "RGB")
            img.save(OUT / "images" / f"{stem}.jpg")
            np.save(OUT / "depths" / f"{stem}.npy", depth)
            n += 1
        print(f"done {tar_path.name}, total {n}", flush=True)
    # write a tiny index manifest
    ids = sorted(p.stem for p in (OUT / "images").glob("*.jpg"))
    (OUT / "manifest.json").write_text(
        "{\"split\": \"val\", \"n\": %d, \"ids\": %s}" % (len(ids), str(ids)))
    print(f"extracted {n} samples -> {OUT}")


if __name__ == "__main__":
    main()
