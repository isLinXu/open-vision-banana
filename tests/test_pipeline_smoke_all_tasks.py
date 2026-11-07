"""Downloaded layouts -> prepare/build scripts -> merge -> zero-shot audit -> DataLoader batches.

Every source in configs/datasets/sources.tsv that feeds training is mocked in the
layout the downloader leaves on disk, pushed through the real preparation
scripts, merged with scripts/merge_multitask_manifest.py, audited with
scripts/build_qwen_baseline_manifest.py, and finally loaded through the Trainer's
ManifestVisionDataset + TaskWeightedSampler + _collate_fn. The test asserts that
all six paper_aligned tasks (depth, normal, semantic, instance, referring,
generation) come out as batches with the right contracts.
"""

from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

h5py = pytest.importorskip("h5py")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pycocotools")
pytest.importorskip("ijson")

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.core.normal_mapper import NormalRGBMapper  # noqa: E402
from open_vision_banana.core.referring import decode_referring_mask  # noqa: E402
from open_vision_banana.core.segmentation import SegmentationDecoder  # noqa: E402
from open_vision_banana.res_manifest import SOURCE_SPECS, build_res_manifest  # noqa: E402
from open_vision_banana.training.sampling import TaskWeightedSampler  # noqa: E402
from open_vision_banana.training.trainer import (  # noqa: E402
    ManifestVisionDataset,
    _collate_fn,
    parse_semantic_legend_from_prompt,
)
from scripts.build_generation_replay_manifest import build_generation_replay_manifest  # noqa: E402
from scripts.build_instance_manifest import build_instance_manifest  # noqa: E402
from scripts.build_qwen_baseline_manifest import build_qwen_baseline_manifest  # noqa: E402
from scripts.merge_multitask_manifest import merge_manifests  # noqa: E402
from scripts.prepare_hypersim import prepare_hypersim  # noqa: E402
from scripts.prepare_tartanair import prepare_tartanair  # noqa: E402

RES = 32


def _png_bytes(color: tuple[int, int, int], size: int = RES) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- fixtures in downloaded layouts
def _hypersim_scene(raw: Path) -> None:
    geo = raw / "hypersim" / "ai_001_001" / "images" / "scene_cam_00_geometry_hdf5"
    prev = raw / "hypersim" / "ai_001_001" / "images" / "scene_cam_00_final_preview"
    geo.mkdir(parents=True)
    prev.mkdir()
    for frame in range(2):
        Image.new("RGB", (64, 48), (90, 90, 90)).save(prev / f"frame.{frame:04d}.color.jpg")
        normal = np.zeros((48, 64, 3), np.float32)
        normal[..., 2] = 1.0
        semantic = np.full((48, 64), 1, np.int32)  # wall
        semantic[24:, :32] = 5  # chair A
        semantic[24:, 32:] = 5  # chair B
        instance = np.where(semantic == 1, 1, 0)
        instance[24:, :32] = 7
        instance[24:, 32:] = 8
        arrays = {
            "depth_meters": np.full((48, 64), 3.0, np.float32),
            "normal_cam": normal,
            "semantic": semantic,
            "semantic_instance": instance,
        }
        for name, arr in arrays.items():
            with h5py.File(geo / f"frame.{frame:04d}.{name}.hdf5", "w") as handle:
                handle.create_dataset("dataset", data=arr)


def _tartanair(raw: Path) -> None:
    traj = raw / "tartanair" / "abandonedfactory" / "Easy" / "P000"
    (traj / "image_left").mkdir(parents=True)
    (traj / "depth_left").mkdir()
    for frame in range(2):
        Image.new("RGB", (RES, RES), (70, 80, 90)).save(traj / "image_left" / f"{frame:06d}_left.png")
        np.save(traj / "depth_left" / f"{frame:06d}_left_depth.npy", np.full((RES, RES), 12.0, np.float32))


def _coco(raw: Path) -> tuple[Path, Path, Path]:
    images = raw / "coco2017" / "train2017"
    images.mkdir(parents=True)
    train2014 = raw / "coco2017" / "train2014"
    train2014.mkdir()
    for image_id in (7, 8, 9, 10, 11):
        path = images / f"{image_id:012d}.jpg"
        Image.new("RGB", (64, 48), (120, 100, 80)).save(path)
        os.link(path, train2014 / f"COCO_train2014_{image_id:012d}.jpg")  # downloader's coco2014_links (hard link)

    def rect(x0, y0, x1, y1):
        return [[x0, y0, x1, y0, x1, y1, x0, y1]]

    anns = []
    ann_id = 1
    for image_id in (7, 8, 9, 10, 11):
        for (x0, x1) in ((2, 28), (34, 60)):
            anns.append({"id": ann_id, "image_id": image_id, "category_id": 1, "segmentation": rect(x0, 4, x1, 44),
                         "area": 1000, "iscrowd": 0, "bbox": [x0, 4, x1 - x0, 40]})
            ann_id += 1
    payload = {
        "images": [{"id": i, "file_name": f"{i:012d}.jpg", "width": 64, "height": 48} for i in (7, 8, 9, 10, 11)],
        "annotations": anns,
        "categories": [{"id": 1, "name": "dog"}],
    }
    ann_dir = raw / "coco2017" / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train2017.json").write_text(json.dumps(payload))

    # REFER files for refcoco / refcoco+ / refcocog (what refer_unc / refer_umd write)
    refer = raw / "refer"
    refs_by_source = {
        "refcoco": [{"ref_id": 1, "image_id": 7, "ann_id": 1, "split": "train", "sentences": [{"sent_id": 1, "raw": "the left dog", "sent": "the left dog"}]},
                    # refcoco val lives on COCO train images too but is NOT a paper benchmark -> stays as referring/val
                    {"ref_id": 5, "image_id": 11, "ann_id": 9, "split": "val", "sentences": [{"sent_id": 9, "raw": "right dog", "sent": "right dog"}]}],
        "refcoco+": [{"ref_id": 1, "image_id": 8, "ann_id": 3, "split": "train", "sentences": [{"sent_id": 2, "raw": "dog on the left", "sent": "dog on the left"}]}],
        # refcocog val image 10 must be excluded from every train pool downstream
        "refcocog": [{"ref_id": 1, "image_id": 10, "ann_id": 7, "split": "val", "sentences": [{"sent_id": 3, "raw": "the dog near the wall", "sent": "the dog near the wall"}]},
                     {"ref_id": 2, "image_id": 9, "ann_id": 5, "split": "train", "sentences": [{"sent_id": 4, "raw": "left dog", "sent": "left dog"}]}],
    }
    for source, protocol in SOURCE_SPECS:
        folder = refer / source
        folder.mkdir(parents=True)
        inst = {
            "images": [{"id": i, "file_name": f"COCO_train2014_{i:012d}.jpg", "width": 64, "height": 48} for i in (7, 8, 9, 10, 11)],
            "annotations": anns,
            "categories": payload["categories"],
        }
        (folder / "instances.json").write_text(json.dumps(inst))
        (folder / f"refs({protocol}).p").write_bytes(pickle.dumps(refs_by_source[source], protocol=2))
    return images, ann_dir / "instances_train2017.json", refer


def _omniedit(raw: Path) -> Path:
    shards = raw / "omniedit" / "data"
    shards.mkdir(parents=True)
    rows = [
        {"omni_edit_id": f"oe_{i}", "task": "color",
         "src_img": {"bytes": _png_bytes((10, 20, 30)), "path": f"{i}_src.png"},
         "edited_img": {"bytes": _png_bytes((200, 20, 30)), "path": f"{i}_edit.png"},
         "edited_prompt_list": ["make the sky purple"], "width": RES, "height": RES}
        for i in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(rows), shards / "train-00000-of-00571.parquet")
    return shards


# ---------------------------------------------------------------- the pipeline
def test_all_six_tasks_flow_from_downloaded_layouts_to_batches(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _hypersim_scene(raw)
    _tartanair(raw)
    coco_images, coco_instances, refer_root = _coco(raw)
    omniedit_shards = _omniedit(raw)
    prepared = tmp_path / "prepared"

    # 1. preparation scripts, one per source, exactly as docs/DATASETS.md lists them
    assert prepare_hypersim(raw / "hypersim", prepared / "hypersim", tasks=("depth", "normal", "semantic", "instance")) == 2
    assert prepare_tartanair(raw / "tartanair", prepared / "tartanair", resolution=RES) == 2
    inst = build_instance_manifest(
        coco_instances, coco_images, prepared / "instance_coco" / "targets", prepared / "instance_coco" / "instance.jsonl",
        data_root=prepared / "instance_coco", exclude_image_ids=[refer_root / "refcocog" / "refs(umd).p"],
    )
    assert inst["rows_written"] == 4 and inst["rejected_excluded_image_id"] == 1  # image 10 is RefCOCOg val
    res = build_res_manifest(refer_root, raw / "coco2017" / "train2014", prepared / "res")
    assert res["counts"].get("train", 0) >= 2, res["counts"]
    gen = build_generation_replay_manifest(
        omniedit_shards, prepared / "replay" / "omniedit.jsonl", kind="edit", preset="omniedit",
        data_root=prepared / "replay", images_dir=prepared / "replay" / "images",
    )
    assert gen["rows_written"] == 3

    # 2. merge (paths rebased to the merged manifest; every task gets a val slice)
    merged = tmp_path / "merged" / "pairs.jsonl"
    summary = merge_manifests(
        [
            prepared / "hypersim" / "pairs.jsonl",
            prepared / "tartanair" / "pairs.jsonl",
            prepared / "instance_coco" / "instance.jsonl",
            prepared / "res" / "pairs.jsonl",
            prepared / "replay" / "omniedit.jsonl",
        ],
        merged,
        val_percent=34,
    )
    assert summary["per_task_dropped"] == {}, summary["per_task_dropped"]
    assert set(summary["per_task_emitted"]) >= {"metric_depth", "depth", "normal", "semantic", "instance", "referring", "generation"}

    # 3. zero-shot audit + metric-depth filter, as before training
    final = tmp_path / "final" / "pairs.jsonl"
    audit = build_qwen_baseline_manifest(
        merged, final, data_root=merged.parent, min_cityscapes_share=0.0,
        exclude_image_ids=[refer_root / "refcocog" / "refs(umd).p"],
    )
    assert audit["rejected_non_metric_depth_rows"] == 0
    final_rows = [json.loads(line) for line in final.read_text().splitlines() if line.strip()]
    for row in final_rows:
        if row["task"] == "depth":
            assert row.get("metric_depth"), "Native metric evidence must survive preparation and merge"
            assert (final.parent / row["metric_depth"]).is_file()
    # the RES builder emits the official RefCOCOg val row (image 10); the audit must drop it
    assert audit["rejected_excluded_image_id_rows"] == 1
    counts = audit["task_split_counts"]
    for task in ("depth", "normal", "semantic", "instance", "referring", "generation"):
        assert counts.get(f"{task}/train", 0) >= 1, (task, counts)
        assert counts.get(f"{task}/val", 0) >= 1, (task, counts)

    # 4. the Trainer's loading path: one dataset per task, weighted sampler, collate
    tasks = ["depth", "normal", "semantic", "instance", "referring", "generation"]
    datasets = [ManifestVisionDataset(final, RES, "train", {t}, data_root=final.parent) for t in tasks]
    assert all(len(ds) >= 1 for ds in datasets)
    groups, offset = {}, 0
    for task, ds in zip(tasks, datasets, strict=True):
        groups[task] = list(range(offset, offset + len(ds)))
        offset += len(ds)
    concat = ConcatDataset(datasets)
    weights = {t: 0.1 for t in tasks if t != "generation"}
    weights["generation"] = 0.5
    sampler = TaskWeightedSampler(groups, strategy="uniform", weights=weights, seed=0, num_samples=48)
    loader = DataLoader(concat, batch_size=4, sampler=sampler, collate_fn=_collate_fn)
    seen = set()
    for batch in loader:
        assert batch["image"].shape[1:] == (3, RES, RES) and batch["target_rgb"].shape[1:] == (3, RES, RES)
        assert batch["image"].min() >= -1.0 and batch["image"].max() <= 1.0
        assert batch["target_rgb"].min() >= 0.0 and batch["target_rgb"].max() <= 1.0
        assert len(batch["prompt"]) == len(batch["task"])
        seen.update(batch["task"])
    assert seen == set(tasks)

    # 5. per-task contract spot checks on real samples
    def first(task):
        return datasets[tasks.index(task)][0]

    depth = DepthRGBMapper().rgb_to_depth(first("depth")["target_rgb"].unsqueeze(0))
    assert 2.8 < float(depth.median()) < 12.5  # hypersim 3 m (planar at centre) or tartanair 12 m
    normal = NormalRGBMapper().rgb_to_normal(first("normal")["target_rgb"].unsqueeze(0))[0]
    assert float(normal[2].mean()) > 0.95
    sem = first("semantic")
    legend = sem["prompt"]
    assert "wall" in legend and "chair" in legend
    if parse_semantic_legend_from_prompt(legend):
        assert "wall" in parse_semantic_legend_from_prompt(legend)
    inst_sample = first("instance")
    assert inst_sample["prompt"].startswith("Generate an instance segmentation visualization")
    assert len(SegmentationDecoder.paper_reproduction().decode_instance(inst_sample["target_rgb"].unsqueeze(0))[0]) >= 1
    ref = first("referring")
    assert ref["prompt"].startswith("Identify the object or region referred to by the following query in the image.")
    mask = decode_referring_mask(ref["target_rgb"].unsqueeze(0))
    assert 0 < int(mask.sum()) < mask.numel()
    gen_sample = first("generation")
    assert gen_sample["prompt"] == "make the sky purple"
    assert float((gen_sample["image"] * 0.5 + 0.5 - gen_sample["target_rgb"]).abs().mean()) > 0.1  # edit pair, not identity
