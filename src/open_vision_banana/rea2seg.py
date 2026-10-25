"""Offline Rea²Seg adapter for the existing expression-conditioned pair loader.

No training or evaluation dispatch lives here. Optional readers are imported
only when used, and Hub access is cache-only.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from .res_preparation import build_res_pairs

DATASET = "snowball521/Rea2Seg-16K"
CATEGORIES = ("gqa-seg", "lisa-plus", "spatial", "reasonseg")
FIELDS = frozenset(("image", "question", "cot_answer", "mask", "category"))


def parquet_rows(paths: Iterable[Path]) -> Iterator[tuple[dict, dict, Path]]:
    """Yield rows, stable shard/row provenance, and the local image-path root."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Local Rea2Seg conversion requires pip install pyarrow") from exc
    # Keep snapshot symlink names: their targets in the Hub cache are blob hashes.
    paths = sorted(Path(path).absolute() for path in paths)
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Provide distinct local Parquet files")
    names: set[str] = set()
    for path in paths:
        if path.name.casefold() in names:
            raise ValueError(f"Conflicting source shard name: {path.name}")
        names.add(path.name.casefold())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        shard_hash = digest.hexdigest()
        parquet = pq.ParquetFile(path)
        if not FIELDS.issubset(parquet.schema_arrow.names):
            raise ValueError(f"{path.name}: required fields are {sorted(FIELDS)}")
        row_index = 0
        for batch in parquet.iter_batches(batch_size=1):
            for row in batch.to_pylist():
                # Short Visual Genome ids must not be interpreted as COCO ids.
                image = row.get("image")
                image_name = (
                    Path(str(image.get("path") or "")).name if isinstance(image, dict) else ""
                )
                match = re.fullmatch(
                    r"(?:COCO_(?:train|val)20\d{2}_)?(\d{12})\.(?:jpg|jpeg|png)",
                    image_name,
                    re.IGNORECASE,
                )
                if image_name:
                    row["source_image"] = image_name
                if match:
                    row["image_id"] = int(match.group(1))
                yield (
                    row,
                    {
                        "source_shard": path.name,
                        "source_shard_sha256": shard_hash,
                        "source_row": row_index,
                    },
                    path.parent,
                )
                row_index += 1


def cached_hf_shards(dataset: str, revision: str) -> tuple[list[Path], str]:
    """Resolve an explicitly requested, already downloaded Hub snapshot."""
    if not revision:
        raise ValueError("HF input requires --revision (prefer an immutable commit SHA)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("HF cache input requires pip install huggingface_hub") from exc
    snapshot = Path(
        snapshot_download(
            repo_id=dataset, repo_type="dataset", revision=revision, local_files_only=True
        )
    )
    paths = sorted((snapshot / "data").glob("train-*.parquet"))
    if not paths:
        raise ValueError(
            "Cached snapshot has no data/train-*.parquet; download it explicitly first"
        )
    # Official shards declare their expected count; do not quietly use a partial cache.
    expected = [f"train-{i:05d}-of-{len(paths):05d}.parquet" for i in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise ValueError("Incomplete or unexpected cached train shard set")
    return paths, snapshot.name


def build_rea2seg(
    rows: Iterable[tuple[dict, dict, Path]],
    output_dir: str | Path,
    *,
    source: str = DATASET,
    revision: str | None = None,
    categories: Iterable[str] = CATEGORIES,
    max_samples: int | None = None,
    source_split: str = "train",
    merge: tuple[Path, Path] | None = None,
    exclude: Iterable[tuple[Path, Path]] = (),
    reject_invalid: bool = False,
    exclude_image_ids: Iterable[int] = (),
) -> dict:
    """PR #26 entry point, using the shared native binary RES writer."""
    return build_res_pairs(
        rows,
        output_dir,
        source=source,
        namespace="rea2seg",
        source_label="Rea2Seg",
        source_protocol="original_train",
        available_categories=CATEGORIES,
        revision=revision,
        categories=categories,
        max_samples=max_samples,
        source_split=source_split,
        merge=merge,
        exclude=exclude,
        reject_invalid=reject_invalid,
        exclude_image_ids=exclude_image_ids,
    )
