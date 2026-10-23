"""NYU .mat -> split/images+depths and GenAI-Bench parquet -> prompts.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

h5py = pytest.importorskip("h5py")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
scipy_io = pytest.importorskip("scipy.io")

from open_vision_banana.data.datasets import NYUDepthDataset  # noqa: E402
from open_vision_banana.data.paper_eval_datasets import GenAIBenchDataset  # noqa: E402
from scripts.convert_eval_sets import convert_genai_bench, convert_nyu_mat  # noqa: E402


def test_nyu_mat_to_eigen_split_layout_loads_in_the_eval_dataset(tmp_path: Path):
    n, h, w = 6, 48, 64
    mat = tmp_path / "nyu_depth_v2_labeled.mat"
    with h5py.File(mat, "w") as handle:
        # MATLAB column-major: images [N, 3, W, H], depths [N, W, H]
        imgs = np.zeros((n, 3, w, h), np.uint8)
        imgs[:, 0] = 200  # red images
        handle.create_dataset("images", data=imgs)
        depths = np.stack([np.full((w, h), 1.0 + i, np.float32) for i in range(n)])
        depths[0, 0, 0] = np.nan
        handle.create_dataset("depths", data=depths)
    splits = tmp_path / "splits.mat"
    scipy_io.savemat(splits, {"trainNdxs": np.array([[1], [2], [3], [4]]), "testNdxs": np.array([[5], [6]])})

    summary = convert_nyu_mat(mat, tmp_path / "nyu", splits_path=splits)
    assert summary["frames_written"] == {"train": 4, "test": 2} and summary["eigen_test_split"]
    test_imgs = sorted((tmp_path / "nyu" / "test" / "images").glob("*.jpg"))
    assert [p.stem for p in test_imgs] == ["0004", "0005"]
    img = np.asarray(Image.open(test_imgs[0]))
    assert img.shape == (h, w, 3) and img[..., 0].mean() > 190  # transposed back to H x W
    depth_png = np.asarray(Image.open(tmp_path / "nyu" / "test" / "depths" / "0004.png"))
    assert depth_png.dtype == np.uint16 and depth_png.shape == (h, w) and int(depth_png[5, 5]) == 5000
    assert int(np.asarray(Image.open(tmp_path / "nyu" / "train" / "depths" / "0000.png"))[0, 0]) == 0  # NaN -> 0 (invalid)

    ds = NYUDepthDataset(tmp_path / "nyu", split="test", resolution=32)
    assert len(ds) == 2
    sample = ds[0]
    assert float(sample["gt_depth"].max()) == pytest.approx(5.0, abs=1e-3)

    # without splits.mat everything lands in all/ and the summary says so
    summary2 = convert_nyu_mat(mat, tmp_path / "nyu_all")
    assert summary2["frames_written"] == {"all": 6} and not summary2["eigen_test_split"]


def test_genai_bench_parquet_to_prompts_jsonl(tmp_path: Path):
    shards = tmp_path / "data"
    shards.mkdir()
    rows = [
        {"prompt": "a red cube on a table", "left_model": "x", "right_model": "y", "vote_type": "leftvote"},
        {"prompt": "a red cube on a table", "left_model": "x", "right_model": "z", "vote_type": "rightvote"},
        {"prompt": "two cats wearing hats", "left_model": "x", "right_model": "y", "vote_type": "tie"},
        {"prompt": "   ", "left_model": "x", "right_model": "y", "vote_type": "tie"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), shards / "train-00000-of-00001.parquet")
    summary = convert_genai_bench(shards, tmp_path / "gb")
    assert summary["prompts"] == 2
    written = [json.loads(line) for line in (tmp_path / "gb" / "prompts.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in written] == ["a red cube on a table", "two cats wearing hats"]
    assert written[0]["id"] == "genai_00000"

    # the eval dataset indexes the prompts (prompt-only samples deliberately refuse __getitem__
    # because the paper metric is a human pairwise study; prompts feed the generation proxy)
    ds = GenAIBenchDataset(tmp_path / "gb", split="test", resolution=32)
    assert len(ds) == 2 and ds.samples[1]["prompt"] == "two cats wearing hats"


def test_genai_bench_1600_capitalised_columns(tmp_path: Path):
    # the ModelScope / HF GenAI-Bench-1600 shards: Index, Prompt, Tags, HumanRatings, one image per model
    shards = tmp_path / "data"
    shards.mkdir()
    rows = [{"Index": 0, "Prompt": "A baker pulling bread out of an oven.", "HumanRatings": {"SDXL_Base": [3]}}]
    pq.write_table(pa.Table.from_pylist(rows), shards / "train-00000-of-00012.parquet")
    assert convert_genai_bench(shards, tmp_path / "gb")["prompts"] == 1
    assert GenAIBenchDataset(tmp_path / "gb", split="val", resolution=32).samples[0]["prompt"].startswith("A baker")


def test_genai_bench_without_a_prompt_column_fails(tmp_path: Path):
    shards = tmp_path / "data"
    shards.mkdir()
    pq.write_table(pa.Table.from_pylist([{"question": "q"}]), shards / "train-00000-of-00001.parquet")
    with pytest.raises(ValueError, match="no prompts"):
        convert_genai_bench(shards, tmp_path / "gb")
    assert not (tmp_path / "gb" / "prompts.jsonl").exists()
