#!/usr/bin/env python3
"""Wait for ckpt-170 NYU eval, then start FLUX resume training with batch 4."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722")
OUT = ROOT / "output/flux_schnell_paper_gate_lr1e5_20260824"
LOG = OUT / "logs/wait_then_train_bs4.log"
NYU_JSON = OUT / "evals/nyu_delta1_n32_ckpt170.json"
EVAL_PID = 73171
OLD_RUNNER_PID = 73150
TRAIN_SCRIPT = ROOT / "scripts/run_flux_schnell_paper_gate_resume300_bs4_20260825.sh"


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {message}\n"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def mps_python_busy() -> list[str]:
    result = subprocess.run(
        ["ps", "aux"],
        check=False,
        capture_output=True,
        text=True,
    )
    busy = []
    for line in result.stdout.splitlines():
        if "python" not in line:
            continue
        if any(token in line for token in (
            "evaluate_paper_benchmarks",
            "train_sdxl_mps",
            "viz_flux_paper_gate",
            "train_sana",
        )):
            busy.append(line)
    return busy


def main() -> int:
    log(f"waiting for NYU eval pid={EVAL_PID}")
    while pid_alive(EVAL_PID) and not NYU_JSON.exists():
        time.sleep(15)
    if NYU_JSON.exists():
        log(f"NYU json ready: {NYU_JSON}")
    else:
        log("eval pid exited without json; continuing if GPU is free")

    deadline = time.time() + 180
    while pid_alive(EVAL_PID) and time.time() < deadline:
        time.sleep(5)
    if pid_alive(EVAL_PID):
        log(f"eval pid {EVAL_PID} still alive after json; waiting for GPU release")
        while pid_alive(EVAL_PID):
            time.sleep(5)

    if pid_alive(OLD_RUNNER_PID):
        os.kill(OLD_RUNNER_PID, 9)
        log(f"killed stopped runner {OLD_RUNNER_PID}")

    busy = mps_python_busy()
    if busy:
        log("GPU still busy, not starting train:\n" + "\n".join(busy))
        return 1

    train_log = open(OUT / "logs/runner_resume300_bs4.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [str(TRAIN_SCRIPT)],
        stdout=train_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(ROOT),
    )
    log(f"started bs4 runner pid={proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
