"""scripts/download_datasets.sh + configs/datasets/sources.tsv (no network)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_datasets.sh"
REGISTRY = ROOT / "configs" / "datasets" / "sources.tsv"
COLUMNS = ["name", "group", "tasks", "source", "locator", "include", "dest", "post", "size_gb", "license", "note"]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _rows() -> list[dict[str, str]]:
    rows = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        assert len(fields) == len(COLUMNS), f"{fields[0]}: expected {len(COLUMNS)} tab-separated fields, got {len(fields)}"
        rows.append(dict(zip(COLUMNS, fields, strict=True)))
    return rows


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "PYTHON": sys.executable, **(env or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=full_env, timeout=120)


def test_registry_rows_are_well_formed_and_cover_the_paper_tasks():
    rows = _rows()
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names)), "duplicate dataset names"
    for row in rows:
        assert row["group"] in {"train", "eval", "replay"}, row["name"]
        assert row["source"] in {"ms", "hf", "url", "manual"}, row["name"]
        for step in row["post"].split("+"):
            assert step in {"none", "unzip", "untar", "unzip_each", "untar_each", "coco2014_links", "refer_umd", "refer_unc", "nyu_mat", "genai_prompts"}, row["name"]
        assert set(row["tasks"].split(",")) <= {"depth", "normal", "semantic", "instance", "referring", "generation"}, row["name"]
    train_tasks = {t for r in rows if r["group"] == "train" for t in r["tasks"].split(",")}
    assert {"depth", "normal", "semantic", "instance", "referring"} <= train_tasks
    assert any(r["group"] == "replay" for r in rows)
    # Paper benchmarks must never be in the train group
    for name in ("refcocog", "reasonseg", "nyu-depth-v2-labeled", "kitti-depth", "cityscapes", "genai-bench"):
        assert next(r for r in rows if r["name"] == name)["group"] == "eval"


def test_list_and_filters():
    out = _run("--list").stdout
    assert "hypersim" in out and "omniedit" in out
    only_depth = _run("--list", "--task", "depth").stdout
    assert "hypersim" in only_depth and "refcoco" not in only_depth
    only_eval = _run("--list", "--group", "eval").stdout
    assert "refcocog" in only_eval and "ade20k" not in only_eval


def test_dry_run_expands_macros_without_touching_the_network(tmp_path: Path):
    res = _run(
        "--only", "hypersim,tartanair,omniedit", "--dry-run", "--root", str(tmp_path),
        env={"HYPERSIM_SCENES": "ai_001_001 ai_009_003", "TARTANAIR_CHUNKS": "3", "OMNIEDIT_SHARDS": "1"},
    )
    assert res.returncode == 0, res.stderr
    # CLI form: "--include raw/a.zip raw/b.zip"; Python-API fallback form: "['raw/a.zip','raw/b.zip',]"
    assert "raw/ai_001_001.zip" in res.stdout and "raw/ai_009_003.zip" in res.stdout
    assert "chunk__part000002.tar.gz" in res.stdout and "chunk__part000003" not in res.stdout
    assert "data/train-00000-of-00571.parquet" in res.stdout
    assert not any(tmp_path.iterdir())  # nothing written in dry-run


def test_local_url_fetch_unzip_and_coco2014_links_are_idempotent(tmp_path: Path):
    src = tmp_path / "src" / "train2017"
    src.mkdir(parents=True)
    for image_id in (7, 581921):
        Image.new("RGB", (4, 4)).save(src / f"{image_id:012d}.jpg")
    archive = tmp_path / "train2017.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for img in src.iterdir():
            zf.write(img, f"train2017/{img.name}")
    registry = tmp_path / "registry.tsv"
    registry.write_text(
        "# name\tgroup\ttasks\tsource\tlocator\tinclude\tdest\tpost\tsize_gb\tlicense\tnote\n"
        f"mini-coco\ttrain\tinstance\turl\tfile://{archive}\t-\tcoco2017\tunzip+coco2014_links\t0\tCC\tfixture\n"
        "needs-login\teval\tdepth\tmanual\thttps://example.org\t-\tkitti\tnone\t1\tL\t\n",
        encoding="utf-8",
    )
    root = tmp_path / "root"
    first = _run("--registry", str(registry), "--root", str(root))
    assert first.returncode == 0, first.stderr
    assert "1 fetched, 1 need manual download, 0 failed" in first.stderr
    link = root / "coco2017" / "train2014" / "COCO_train2014_000000581921.jpg"
    assert link.is_symlink() and link.resolve().name == "000000581921.jpg"
    assert (root / "coco2017" / "train2017.zip.extracted").exists()

    second = _run("--registry", str(registry), "--root", str(root), "--only", "mini-coco")
    assert second.returncode == 0, second.stderr
    assert "exists:" in second.stderr and "0 new symlinks" in second.stdout
