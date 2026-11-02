"""website/: static project page renders from JSON; publish script dry-runs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"


def test_results_json_schema_and_paper_reference_values():
    data = json.loads((SITE / "data" / "results.json").read_text())
    assert data["inference_protocol"]["num_inference_steps"] == 50 and data["inference_protocol"]["guidance_scale"] == 4.0
    for table in data["tables"]:
        keys = {c["key"] for c in table["columns"]}
        kinds = [r["kind"] for r in table["rows"]]
        assert "ours" in kinds
        for row in table["rows"]:
            assert set(row["values"]) <= keys, (table["title"], row["label"])
            if row["kind"] == "paper":
                assert row.get("source") == "paper"  # paper numbers are always labelled as context
    paper_values = {}
    for t in data["tables"]:
        for r in t["rows"]:
            if r["kind"] == "paper":
                paper_values.update(r["values"])
    assert paper_values["cityscapes_miou"] == 69.9 and paper_values["cgf1"] == 47.5  # Tables 2 and 3 of the paper


def test_gallery_json_and_html_reference_existing_assets():
    gallery = json.loads((SITE / "data" / "gallery.json").read_text())
    assert set(gallery) == {"semantic", "instance", "referring", "depth", "normal"}
    html = (SITE / "index.html").read_text()
    for ref in re.findall(r'(?:src|href)="((?!https?://|#)[^"]+)"', html):
        assert (SITE / ref).exists(), ref
    assert "Not affiliated" in html and "Honest boundaries" in html


def test_publish_script_dry_run():
    res = subprocess.run(["bash", str(ROOT / "scripts" / "publish_site.sh"), "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert "json ok" in res.stdout and "index.html" in res.stdout and "results.json" in res.stdout
