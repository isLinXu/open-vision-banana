"""End-to-end verification of Solution 2 (Selective Pipe Extraction) and
Solution 3 (indexed_gzip Virtual Stream Index O(1) Direct Reading) on split
SenseNova-Vision-Corpus-50M.tar.gz000~002 volumes.
"""

from __future__ import annotations

import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from open_vision_banana.data.tar_dataset import (
    IndexedTarReader,
    TarArchiveDataset,
    build_snvc_tar_index,
)
from open_vision_banana.training.trainer import ManifestVisionDataset
from scripts.build_snvc_tar_index import extract_manifest_files

ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _create_split_snvc_volumes(
    vol_dir: Path,
    entries: dict[str, bytes],
    num_volumes: int = 3,
) -> list[Path]:
    """Create a real gzipped tar archive and split its byte stream into .tar.gz000..00N."""
    vol_dir.mkdir(parents=True, exist_ok=True)
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for arcname, payload in entries.items():
            info = tarfile.TarInfo(name=arcname)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    gz_bytes = gzip.compress(tar_buf.getvalue(), compresslevel=6)

    chunk_size = (len(gz_bytes) + num_volumes - 1) // num_volumes
    volumes: list[Path] = []
    for i in range(num_volumes):
        part = gz_bytes[i * chunk_size : (i + 1) * chunk_size]
        if not part:
            break
        vol_path = vol_dir / f"SenseNova-Vision-Corpus-50M.tar.gz{i:03d}"
        vol_path.write_bytes(part)
        volumes.append(vol_path)

    # Add an incomplete .aria2 decoy volume that must be ignored by scripts
    decoy = vol_dir / f"SenseNova-Vision-Corpus-50M.tar.gz{len(volumes):03d}"
    decoy.write_bytes(b"corrupted_partial_download")
    (vol_dir / f"{decoy.name}.aria2").write_bytes(b"aria2_control")
    return volumes


def test_solution_2_selective_pipe_extraction_script(tmp_path: Path):
    """Verify Solution 2: scripts/extract_snvc50m.sh extracts only requested dirs
    from split .tar.gz000~002 volumes while skipping .aria2 incomplete volumes.
    """
    vol_dir = tmp_path / "volumes"
    extract_dir = tmp_path / "extracted"
    long_subdir = "Cityscapes/genseg/frankfurt_000000_000294_very_long_subdirectory_name_over_100_chars"
    entries = {
        f"SenseNova-Vision-Corpus-50M/{long_subdir}/mask.png": _png_bytes((128, 64, 128)),
        "SenseNova-Vision-Corpus-50M/scannetpp/normal/n001.png": _png_bytes((128, 128, 255)),
        "SenseNova-Vision-Corpus-50M/unwanted_ocr/doc001.png": _png_bytes((255, 255, 255)),
    }
    _create_split_snvc_volumes(vol_dir, entries, num_volumes=3)

    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "extract_snvc50m.sh"),
            "--dataset-root",
            str(vol_dir),
            "--extract-root",
            str(extract_dir),
            "--dirs",
            "Cityscapes",
            "scannetpp",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Skipping incomplete volume" in res.stdout
    cs_extracted = extract_dir / "SenseNova-Vision-Corpus-50M" / long_subdir / "mask.png"
    snpp_extracted = extract_dir / "SenseNova-Vision-Corpus-50M" / "scannetpp" / "normal" / "n001.png"
    unwanted = extract_dir / "SenseNova-Vision-Corpus-50M" / "unwanted_ocr" / "doc001.png"

    assert cs_extracted.is_file()
    assert snpp_extracted.is_file()
    assert not unwanted.exists()
    with Image.open(cs_extracted) as img:
        assert tuple(np.asarray(img)[0, 0]) == (128, 64, 128)


def test_solution_3_indexed_gzip_direct_read_and_hybrid_manifest(tmp_path: Path):
    """Verify Solution 3: single-pass build_snvc_tar_index (.gzidx + tar_index.json),
    O(1) random-order access via IndexedTarReader, TarArchiveDataset, and unified
    ManifestVisionDataset (including hybrid local RGB input + tar target).
    """
    vol_dir = tmp_path / "snvc_root"
    long_target_rel = (
        "coconut/coconut_xl/genseg/"
        "000000987654_extremely_long_semantic_segmentation_target_filename_exceeding_100_bytes.png"
    )
    entries = {
        "SenseNova-Vision-Corpus-50M/scannetpp/input/rgb_01.png": _png_bytes((10, 20, 30)),
        "SenseNova-Vision-Corpus-50M/scannetpp/normal/norm_01.png": _png_bytes((128, 128, 255)),
        f"SenseNova-Vision-Corpus-50M/{long_target_rel}": _png_bytes((244, 35, 232)),
    }
    volumes = _create_split_snvc_volumes(vol_dir, entries, num_volumes=3)

    # Also place a local input RGB image outside the tar archive (hybrid SNVC layout)
    local_inp = vol_dir / "coco2017" / "train2017" / "000000987654.png"
    local_inp.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (40, 50, 60)).save(local_inp)

    # Step 1: Single-pass index build (.gzidx + tar_index.json)
    tar_index_path = vol_dir / "tar_index.json"
    idx_payload = build_snvc_tar_index(volumes, tar_index_path, spacing_mb=1)
    assert tar_index_path.is_file()
    assert idx_payload["gzip_index"] is not None
    assert Path(idx_payload["gzip_index"]).is_file()
    assert f"SenseNova-Vision-Corpus-50M/{long_target_rel}" in idx_payload["files"]

    # Step 2: Verify O(1) reverse-order random access via IndexedTarReader
    reader = IndexedTarReader(volumes, Path(idx_payload["gzip_index"]))
    try:
        keys = list(idx_payload["files"].keys())
        # Read last entry first, then first entry (backward seek!)
        for key in reversed(keys):
            info = idx_payload["files"][key]
            raw = reader.read_at(info["offset"], info["size"])
            assert raw == entries[key]
    finally:
        reader.close()

    # Step 3: Verify unified ManifestVisionDataset reads hybrid local+tar and pure-tar pairs
    manifest_path = vol_dir / "pairs.jsonl"
    manifest_rows = [
        {
            "task": "normal",
            "input": "scannetpp/input/rgb_01.png",
            "target": "scannetpp/normal/norm_01.png",
            "split": "train",
            "prompt": "Generate a surface normal estimation visualization of this image.",
        },
        {
            "task": "semantic",
            "input": "coco2017/train2017/000000987654.png",
            "target": long_target_rel,
            "split": "train",
            "prompt": 'Generate a semantic segmentation visualization image, using this color mapping: {"sidewalk": <244, 35, 232>}.',
        },
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n", encoding="utf-8"
    )

    cache_dir = tmp_path / "tar_cache"
    ds_normal = ManifestVisionDataset(
        manifest_path,
        resolution=32,
        split="train",
        tasks={"normal"},
        data_root=vol_dir,
        disk_cache_dir=cache_dir,
    )
    ds_sem = ManifestVisionDataset(
        manifest_path,
        resolution=32,
        split="train",
        tasks={"semantic"},
        data_root=vol_dir,
        disk_cache_dir=cache_dir,
    )
    s_norm = ds_normal[0]
    s_sem = ds_sem[0]
    assert s_norm["target_rgb"].shape == (3, 32, 32)
    assert torch.allclose(
        s_sem["target_rgb"][:, 0, 0],
        torch.tensor([244.0, 35.0, 232.0]) / 255.0,
        atol=1e-3,
    )

    # Step 4: Also verify TarArchiveDataset uses IndexedTarReader and supports hybrid local+tar
    tar_ds = TarArchiveDataset(
        manifest_path=manifest_path,
        tar_index_path=tar_index_path,
        tar_root=vol_dir,
        resolution=32,
        split="train",
        tasks={"normal", "semantic"},
    )
    assert len(tar_ds) == 2
    # Access in reverse order to prove random access works without SequentialTarReader crash
    item_1 = tar_ds[1]
    item_0 = tar_ds[0]
    assert item_1["target_rgb"].shape == (3, 32, 32)
    assert item_0["target_rgb"].shape == (3, 32, 32)
    assert isinstance(tar_ds._reader, IndexedTarReader)
    tar_ds.close()

    # Step 5: Verify O(1) manifest subset extraction from tar_index.json
    subset_out = tmp_path / "subset_extracted"
    count = extract_manifest_files(idx_payload, manifest_path, subset_out)
    assert count == 3
    assert (subset_out / long_target_rel).is_file()
