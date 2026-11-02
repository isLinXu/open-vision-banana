"""scripts/run_e2_ablations.sh: E2 (vision_ratio x seeds) + E2b (quarter data), chained with gates."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_e2_ablations.sh"


def _dry(*args: str, env: dict | None = None) -> str:
    res = subprocess.run(["bash", str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **(env or {})})
    assert res.returncode == 0, res.stderr
    return re.sub(r"\x1b\[[0-9;]*m", "", res.stdout + res.stderr)


def test_default_runs_cover_two_ratios_two_seeds_and_e2b(tmp_path: Path):
    out = _dry(env={"OUT_ROOT": str(tmp_path / "e2")})
    for ratio, seed in (("0.2", "42"), ("0.2", "43"), ("0.8", "42"), ("0.8", "43")):
        assert f"--vision-ratio {ratio} --seed {seed} --steps 10000" in out
    assert "subsample_multitask_manifest.py --source data/qwen_baseline/pairs.jsonl --output data/qwen_baseline_e2b/pairs.jsonl --fraction 0.25 --seed 42" in out
    assert "merge_multitask_manifest.py" not in out
    assert out.count("check_experiment_gates.py run") == 5 and out.count("check_experiment_gates.py eval") == 5
    assert out.count("check_experiment_gates.py data") == 1  # only E2b rebuilds a manifest
    assert "--num-inference-steps 50 --guidance-scale 4.0 --seed 42 --resolution 512 --native-resolution" in out
    assert out.count("evaluate_generation_preservation.py") == 5
    assert out.count("build_generation_reference.py") == 5  # subsequent runs reuse the same reference
    assert out.count("check_experiment_gates.py generation") == 5
    assert "--profile e2" in out and "--saco-root data/raw/saco_gold" in out
    assert "checkpoint-10000_best" not in out and "checkpoint-10000_final" in out
    assert "--wandb" not in out
    assert not (tmp_path / "e2").exists()


def test_subset_and_resume(tmp_path: Path):
    run = tmp_path / "e2" / "vr0.8_s42"
    (run / "checkpoint-5000").mkdir(parents=True)
    (run / "checkpoint-5000_best").mkdir()
    out = _dry("--only", "vr0.8_s42", env={"OUT_ROOT": str(tmp_path / "e2")})
    assert "--vision-ratio 0.8 --seed 42" in out and "vr0.2_s" not in out and "E2b:" not in out
    assert f"--resume {run}/checkpoint-5000 " in out or f"--resume {run}/checkpoint-5000\n" in out or "resuming from checkpoint-5000" in out
    bad = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--only", "nope"], capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 2 and "unknown run name" in bad.stderr
