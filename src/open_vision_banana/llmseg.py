"""Official LLM-Seg40K COCO train subset -> the existing binary RES contract.

Only offline preparation lives here. SAM proposals and answer strings are not
training targets; the published qa_pairs[].rle_seg is the native mask target.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .res_preparation import build_res_pairs

DATASET = "wangjunchi/LLMSeg"
UPSTREAM_REVISION = "65147a98cc49c2dac9b7a3633d9d3d336f1ac023"
TRAIN_SHA256 = "d925dabf2465f066c43f5d14b1360cf97de281744af15c1d063fe7cc61508754"
VALIDATION_SHA256 = "e25345607d94aea5f30b09ebdd6d9acf024f9a10ea3953d6300245a2cad27689"


def _annotations(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    data = json.loads(payload)
    if not isinstance(data, dict) or not data:
        raise ValueError("LLM-Seg annotations must be a nonempty image-keyed object")
    for name, item in data.items():
        if (
            not isinstance(item, dict)
            or item.get("from_dataset") not in {"coco", "ego_objects"}
            or not isinstance(item.get("qa_pairs"), list)
            or not item["qa_pairs"]
        ):
            raise ValueError(f"Invalid LLM-Seg image record: {name!r}")
        if item["from_dataset"] == "coco" and not re.fullmatch(r"\d{12}\.jpg", name):
            raise ValueError(f"Invalid COCO image name: {name!r}")
    return data, hashlib.sha256(payload).hexdigest()


def _image_path(root: Path, filename: str) -> Path:
    # COCO 2017 is a re-split of 2014; both naming conventions identify the same image.
    for name in (filename, f"COCO_train2014_{filename}", f"COCO_val2014_{filename}"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing COCO image {filename} under {root}")


def build_llmseg(
    annotations: str | Path,
    validation_annotations: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    *,
    max_samples: int = 5000,
    seed: int = 42,
    exclude_image_ids=(),
    exclude=(),
    merge=None,
    reject_invalid: bool = False,
    verify_release: bool = True,
) -> dict:
    """Select <=5k COCO training images, one deterministic question per image.

    Official validation images and caller-supplied COCO evaluation identities
    are removed before selection. No holdout is synthesized or relabeled.
    Release verification is enabled for the CLI; custom fixtures can opt out.
    """
    annotations, validation_annotations, image_root = map(
        Path, (annotations, validation_annotations, image_root)
    )
    train, train_hash = _annotations(annotations)
    val, val_hash = _annotations(validation_annotations)
    if verify_release and (train_hash != TRAIN_SHA256 or val_hash != VALIDATION_SHA256):
        raise ValueError(
            "Annotation SHA256 differs from the pinned official train/validation release"
        )
    blocked = set(exclude_image_ids)
    blocked.update(int(Path(k).stem) for k, v in val.items() if v["from_dataset"] == "coco")
    counts: Counter[str] = Counter()

    def rank(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    candidates = []
    for filename, item in train.items():
        counts[f"upstream_{item['from_dataset']}_images"] += 1
        counts[f"upstream_{item['from_dataset']}_questions"] += len(item["qa_pairs"])
        if item["from_dataset"] != "coco":
            continue
        if int(Path(filename).stem) in blocked:
            counts["excluded_coco_images"] += 1
            continue
        questions = item["qa_pairs"]
        for qa in questions:
            if (
                not isinstance(qa, dict)
                or not {"question_id", "question", "answer", "rle_seg"} <= qa.keys()
            ):
                raise ValueError(f"Invalid qa_pairs entry in {filename}")
        if len({str(q["question_id"]) for q in questions}) != len(questions):
            raise ValueError(f"Duplicate question_id in {filename}")
        qa = min(questions, key=lambda q: rank(f"{filename}:{q['question_id']}"))
        candidates.append((filename, qa))
    candidates.sort(key=lambda item: rank(item[0]))

    def rows():
        for filename, qa in candidates:
            yield (
                {
                    "image": {"path": str(_image_path(image_root, filename).resolve())},
                    "question": qa["question"],
                    "cot_answer": "",
                    "answer": qa["answer"],  # metadata only; never fed into the prompt
                    "mask": {"rle": qa["rle_seg"]},
                    "category": "coco",
                    "source_image": filename,
                    "image_id": int(Path(filename).stem),
                    "question_id": qa["question_id"],
                    "id": f"{filename}:{qa['question_id']}",
                },
                {
                    "source_shard": annotations.name,
                    "source_shard_sha256": train_hash,
                    "source_row": f"{filename}/qa_pairs/{qa['question_id']}",
                },
                image_root,
            )

    report = build_res_pairs(
        rows(),
        output_dir,
        source=DATASET,
        namespace="llmseg",
        source_label="LLM-Seg40K",
        source_protocol="official_train_coco_one_question_v1",
        available_categories=("coco",),
        categories=("coco",),
        revision=UPSTREAM_REVISION,
        max_samples=max_samples,
        exclude=exclude,
        merge=merge,
        exclude_image_ids=blocked,
        reject_invalid=reject_invalid,
        min_mask_fraction=0.005,
        max_mask_fraction=0.95,
    )
    report["selection"] = {
        "seed": seed,
        "policy": "COCO train only; SHA256-ranked images/questions; one question per image",
        "min_mask_fraction": 0.005,
        "max_mask_fraction": 0.95,
        "train_annotations_sha256": train_hash,
        "validation_annotations_sha256": val_hash,
        "official_release_verified": verify_release,
        "eligible_images_before_asset_checks": len(candidates),
        **counts,
    }
    Path(output_dir, "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
