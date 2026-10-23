"""scripts/run_e0_smoke.sh: the E0 stages of exp_setting.md as one gated, resumable script."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_e0_smoke.sh"


def _dry(*args: str, env: dict | None = None) -> str:
    merged = {**os.environ, **(env or {})}
    res = subprocess.run(["bash", str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, cwd=ROOT, env=merged)
    assert res.returncode == 0, res.stderr
    return re.sub(r"\x1b\[[0-9;]*m", "", res.stdout)


def test_dry_run_lists_every_stage_in_order_with_gates():
    out = _dry(env={"RUN_DIR": "output/e0_test"})
    stages = [line for line in out.splitlines() if re.match(r"\[e0\] E0\.\d", line)]
    assert [s.split()[1] for s in stages] == ["E0.1", "E0.1", "E0.1", "E0.2", "E0.3", "E0.3"]
    # the three machine gates, with the strict protocol and the RefCOCOg exclusion in the audit
    assert "check_experiment_gates.py data" in out and "check_experiment_gates.py run" in out and "check_experiment_gates.py eval" in out
    assert "--protocol strict --exclude-image-ids data/raw/refer/refcocog/refs(umd).p" in out
    assert "--tasks depth,normal,semantic,instance" in out  # Hypersim four channels
    assert "--vision-ratio 0.5" in out and "--steps 500" in out
    assert "checkpoint-500_final" in out
    assert "--require AbsRel,cIoU" in out
    # inference protocol pinned (exp_setting E1.3), never the script's 1-step smoke default
    assert "--num-inference-steps 50 --guidance-scale 4.0 --seed 42 --resolution 512 --native-resolution" in out
    assert "eval_smoke_steps50.json" in out
    # knobs from exp_setting E0
    assert "HYPERSIM_SCENES='ai_001_001 ai_001_002', TARTANAIR_CHUNKS=1, OMNIEDIT_SHARDS=1" in out
    assert "nyu-depth-v2-labeled,omniedit" in out  # group replay is not in --group train
    # nothing was created
    assert not (ROOT / "output" / "e0_test").exists()


def test_from_stage_skips_earlier_stages_and_env_overrides_apply():
    out = _dry("--from", "train", env={"RUN_DIR": "output/e0_test", "DATA_ROOT": "/data/raw", "STEPS": "20", "MODEL_NAME": "sana-1600m", "INFER_STEPS": "20"})
    assert "--num-inference-steps 20" in out and "eval_smoke_steps20.json" in out
    assert "download_datasets.sh" not in out and "prepare_hypersim.py" not in out and "merge_multitask_manifest.py" not in out
    assert "--steps 20" in out and "checkpoint-20_final" in out
    assert "--model-name sana-1600m" in out and "/data/raw/nyu_depth_v2" in out


def test_unknown_stage_is_rejected():
    res = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--from", "nowhere"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 2 and "unknown stage" in res.stderr
