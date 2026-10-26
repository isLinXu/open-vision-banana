"""scripts/run_e1_full.sh: E1 as a gated, resumable, idle-free script."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_e1_full.sh"


def _dry(*args: str, env: dict | None = None) -> str:
    res = subprocess.run(["bash", str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **(env or {})})
    assert res.returncode == 0, res.stderr
    return re.sub(r"\x1b\[[0-9;]*m", "", res.stdout + res.stderr)


def test_full_dry_run_has_all_stages_gates_and_pinned_protocol(tmp_path: Path):
    out = _dry(env={"RUN_DIR": str(tmp_path / "run")})
    for marker in ("E1.1 download", "E1.1 prepare", "E1.1 merge", "E1.2 train", "E1.3 evaluate", "extra readings", "generation proxy"):
        assert marker in out, marker
    assert "check_experiment_gates.py data" in out and "check_experiment_gates.py run" in out and out.count("check_experiment_gates.py eval") >= 5
    assert "--protocol strict --exclude-image-ids data/raw/refer/refcocog/refs(umd).p" in out
    assert "--num-inference-steps 50 --guidance-scale 4.0 --seed 42 --resolution 512 --native-resolution" in out
    assert "--dtype bfloat16" in out
    assert "--cityscapes-root data/raw/cityscapes" in out and "--saco-root data/raw/saco_gold" in out
    assert "--profile e1" in out and "check_experiment_gates.py generation" in out
    assert "build_generation_reference.py" in out and "--dataset-root output/reference/genai_base_qwen-image-2.1_s42_n50" in out
    assert "checkpoint-30000_best" not in out
    assert "--guidance-scale 0.0" in out and "--seed 43" in out
    assert "--cap 40000 --val-percent 2" in out and "--semantic-variants 2" in out
    # run dir is created before anything is redirected into it; W&B off without a key
    mk, train = out.index("mkdir -p"), out.index("scripts/train.py")
    assert mk < train
    assert "W&B off" in out and "--wandb" not in out
    assert not (tmp_path / "run").exists()


def test_cpu_and_gpu_halves_and_watch_mode(tmp_path: Path):
    cpu = _dry("--to", "merge", env={"RUN_DIR": str(tmp_path / "run")})
    assert "scripts/train.py" not in cpu and "evaluate_paper_benchmarks" not in cpu and "data stages done" in cpu
    gpu = _dry("--from", "train", env={"RUN_DIR": str(tmp_path / "run"), "WANDB_API_KEY": "x"})
    assert "download_datasets.sh" not in gpu and "prepare_hypersim" not in gpu and "--wandb" in gpu
    watch = _dry("--from", "eval", "--watch", env={"RUN_DIR": str(tmp_path / "run"), "EVAL_GPU": "1", "MAX_EVAL_SAMPLES": "200", "EVAL_CKPTS": "2500 5000"})
    assert "watch mode" in watch and "CUDA_VISIBLE_DEVICES=1" in watch and "--max-samples 200" in watch
    assert "checkpoint-2500" in watch and "checkpoint-5000" in watch and "checkpoint-15000" not in watch


def test_resume_detects_latest_periodic_checkpoint(tmp_path: Path):
    run = tmp_path / "run"
    for name in ("checkpoint-2500", "checkpoint-5000", "checkpoint-5000_best", "checkpoint-7500"):
        (run / name).mkdir(parents=True)
    out = _dry("--from", "train", "--to", "train", env={"RUN_DIR": str(run)})
    assert "resuming full state" in out and f"--resume {run}/checkpoint-7500" in out
    assert "--print-effective-config" not in out  # only on a fresh start
