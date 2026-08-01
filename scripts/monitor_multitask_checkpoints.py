#!/usr/bin/env python3
"""Monitor v3 multi-task checkpoints and run fixed-sample inference analysis.

The trainer writes checkpoints before its validation pass. This monitor keeps
the training process untouched: it polls for new numeric checkpoints, waits
briefly for files to settle, then loads the SSD-1B LoRA and produces one or
more held-out visual comparisons for each supported task.

Outputs are kept outside checkpoint directories so CheckpointManager pruning
cannot remove the analysis history::

    output/multitask_lora_v3_full_256/checkpoint_monitoring/
      step_250/report.json
      step_250/report.md
      step_250/viz/depth_00.png

Use ``--once`` for a single pass or ``--watch --train-pid PID`` to monitor a
live run until its process exits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.decoder import VisualizationDecoder  # noqa: E402
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT  # noqa: E402

logger = logging.getLogger("multitask_monitor")

TASK_DIRS = {
    "depth": "depth",
    "normal": "normal",
    "semantic": "segmentation",
    "edge": "structure",
    "generation": "generation",
}
TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "semantic": SEMANTIC_SEGMENTATION_PROMPT,
    "edge": "Generate an edge detection visualization of this image.",
    "generation": "a photograph",
}
SUPPORTED_TASKS = tuple(TASK_DIRS)


def parse_task_names(value: str) -> list[str]:
    """Parse a comma-separated task selection while preserving order."""
    tasks = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not tasks:
        raise ValueError("At least one monitoring task is required")
    unsupported = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unsupported:
        raise ValueError(f"Unsupported monitoring tasks: {unsupported}")
    return tasks


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def numeric_checkpoints(output_dir: Path) -> list[Path]:
    """Return complete numeric checkpoints in ascending step order."""
    result = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir() or not path.name.split("-", 1)[-1].isdigit():
            continue
        if not (path / "adapter_config.json").exists():
            continue
        try:
            int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        result.append(path)
    return sorted(result, key=lambda path: int(path.name.rsplit("-", 1)[-1]))


def read_manifest(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "train_val_split.json").read_text(encoding="utf-8"))


def _find_target(task_dir: Path, name: str) -> Path | None:
    stem = Path(name).stem
    target_dir = task_dir / "targets"
    if not target_dir.is_dir():
        return None
    candidates = sorted(path for path in target_dir.iterdir() if path.is_file() and path.stem == stem)
    return candidates[0] if candidates else None


def load_validation_samples(data_dir: Path, tasks: list[str], samples_per_task: int) -> list[dict[str, Any]]:
    """Resolve deterministic manifest validation samples for inference."""
    manifest = read_manifest(data_dir)
    result: list[dict[str, Any]] = []
    for task in tasks:
        directory = data_dir / TASK_DIRS[task]
        names = sorted(manifest.get("val", {}).get("generation" if task == "generation" else {
            "semantic": "segmentation", "edge": "structure"
        }.get(task, task), []))
        for name in names[:samples_per_task]:
            image_path = directory / "images" / name
            target_path = image_path if task == "generation" else _find_target(directory, name)
            if not image_path.exists() or target_path is None or not target_path.exists():
                logger.warning("Skipping incomplete %s sample %s", task, name)
                continue
            prompt = TASK_PROMPTS[task]
            if task == "generation":
                prompt_file = directory / "prompts.jsonl"
                if prompt_file.exists():
                    for line in prompt_file.read_text(encoding="utf-8").splitlines():
                        item = json.loads(line)
                        if item.get("image") == name:
                            prompt = item.get("prompt", prompt)
                            break
            result.append({
                "task": task,
                "name": name,
                "image_path": image_path,
                "target_path": target_path,
                "prompt": prompt,
            })
    return result


def _pil_to_array(image: Image.Image, size: int) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    value = tensor.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.min() < 0:
        value = (value + 1.0) / 2.0
    array = (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _quick_metrics(pred: Image.Image, target: Image.Image, size: int = 256) -> dict[str, float]:
    pred_arr = _pil_to_array(pred, size).reshape(-1)
    target_arr = _pil_to_array(target, size).reshape(-1)
    centered_pred = pred_arr - pred_arr.mean()
    centered_target = target_arr - target_arr.mean()
    denominator = float(np.linalg.norm(centered_pred) * np.linalg.norm(centered_target))
    pearson = float(np.dot(centered_pred, centered_target) / denominator) if denominator > 1e-8 else 0.0
    return {
        "rgb_mae": float(np.mean(np.abs(pred_arr - target_arr))),
        "rgb_rmse": float(np.sqrt(np.mean((pred_arr - target_arr) ** 2))),
        "rgb_pearson": pearson,
    }


def _panel(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    width, height = 256, 288
    canvas = Image.new("RGB", (width * len(images), height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        panel = image.convert("RGB").resize((width, width), Image.Resampling.BILINEAR)
        x = index * width
        canvas.paste(panel, (x, 0))
        draw.text((x + 6, width + 8), label, fill="black")
    canvas.save(out_path)


def _quality_judgement(
    metadata: dict[str, Any],
    metrics: list[dict[str, Any]],
    previous: dict[str, Any] | None,
    expected_tasks: list[str],
) -> dict[str, Any]:
    losses = metadata.get("validation_losses", {})
    finite_losses = bool(losses) and all(_finite(value) for value in losses.values())
    complete_tasks = sorted({item["task"] for item in metrics})
    expected_tasks = sorted(expected_tasks)
    current_mean = float(np.mean([float(value) for value in losses.values()])) if finite_losses else None
    previous_mean = previous.get("quality", {}).get("mean_validation_loss") if previous else None
    if not finite_losses:
        verdict = "invalid: missing or non-finite per-task validation losses"
    elif complete_tasks != expected_tasks:
        verdict = f"incomplete: visual inference tasks={complete_tasks} expected={expected_tasks}"
    elif previous_mean is None:
        verdict = "baseline checkpoint: valid smoke quality, no trend comparison yet"
    elif current_mean < float(previous_mean) - 1e-6:
        verdict = "improving: mean held-out loss decreased"
    elif current_mean > float(previous_mean) + 1e-6:
        verdict = "regressing: mean held-out loss increased"
    else:
        verdict = "flat: mean held-out loss unchanged"
    return {
        "verdict": verdict,
        "mean_validation_loss": current_mean,
        "finite_validation_losses": finite_losses,
        "visual_inference_tasks": complete_tasks,
        "visual_inference_complete": complete_tasks == expected_tasks,
    }


def update_validation_metadata(output_dir: Path, step: int, losses: dict[str, float]) -> None:
    """Persist live trainer validation output when stdout has already exposed it."""
    checkpoint = output_dir / f"checkpoint-{step}" / "metadata.json"
    if not checkpoint.exists() or not losses:
        return
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not payload.get("validation_losses"):
        payload["validation_losses"] = losses
        checkpoint.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def analyze_checkpoint(
    checkpoint: Path,
    data_dir: Path,
    monitor_dir: Path,
    device: str,
    resolution: int,
    samples_per_task: int,
    inference_steps: int,
    previous: dict[str, Any] | None,
    tasks: list[str],
) -> dict[str, Any]:
    step = int(checkpoint.name.rsplit("-", 1)[-1])
    step_dir = monitor_dir / f"step_{step}"
    viz_dir = step_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    # A checkpoint can be written just before the trainer finishes validation.
    # Re-read metadata after inference so a concurrent post-validation update
    # is reflected in the report.
    samples = load_validation_samples(data_dir, tasks, samples_per_task)
    logger.info("Analyzing checkpoint step=%d with %d samples", step, len(samples))

    generator = BaseGenerator(
        model_name="segmind/SSD-1B",
        lora_path=str(checkpoint),
        device=device,
        dtype=torch.bfloat16,
    )
    conditioning_config = generator.source_conditioning_config
    if (
        "depth" in tasks
        and conditioning_config is not None
        and conditioning_config.recommended_prompt
    ):
        for sample in samples:
            if sample["task"] == "depth":
                sample["prompt"] = conditioning_config.recommended_prompt
    decoder = VisualizationDecoder()
    sample_reports: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        try:
            source = Image.open(sample["image_path"]).convert("RGB")
            target = Image.open(sample["target_path"]).convert("RGB")
            source_arg = None if sample["task"] == "generation" else source
            with torch.no_grad():
                output = generator.generate(
                    prompt=sample["prompt"],
                    image=source_arg,
                    num_inference_steps=inference_steps,
                    guidance_scale=1.0,
                    strength=0.85,
                    height=resolution,
                    width=resolution,
                    seed=1000 + index,
                )
            output_pil = _tensor_to_pil(output)
            panels = []
            if source_arg is not None:
                panels.append(("input", source))
            panels.extend([("generated", output_pil), ("target", target)])
            # A decoded depth panel makes the geometry output easier to inspect.
            if sample["task"] == "depth":
                decoded = decoder.decode_depth(output).get("visualization")
                if isinstance(decoded, torch.Tensor):
                    panels.insert(-1, ("decoded", _tensor_to_pil(decoded)))
            output_path = viz_dir / f"{sample['task']}_{index:02d}.png"
            _panel(panels, output_path)
            sample_reports.append({
                "task": sample["task"],
                "name": sample["name"],
                "panel": str(output_path),
                "metrics": _quick_metrics(output_pil, target),
            })
        except Exception as error:  # keep other task analyses alive
            logger.exception("Inference failed for %s/%s", sample["task"], sample["name"])
            sample_reports.append({"task": sample["task"], "name": sample["name"], "error": str(error)})
    with contextlib.suppress(Exception):
        generator.unload_lora()
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Metadata is still being written: %s", metadata_path)
    report = {
        "checkpoint": str(checkpoint),
        "step": step,
        "metadata": metadata,
        "samples": sample_reports,
    }
    report["quality"] = _quality_judgement(
        metadata,
        sample_reports,
        previous,
        tasks,
    )
    if not metadata.get("validation_losses"):
        report["quality"]["verdict"] = "visualized: validation metadata not yet available"
        report["quality"]["validation_pending"] = True
    (step_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# Checkpoint step {step}",
        "",
        f"Quality judgement: **{report['quality']['verdict']}**",
        f"Mean validation loss: `{report['quality']['mean_validation_loss']}`",
        "",
        "| Task | Sample | RGB MAE | RGB Pearson | Visualization |",
        "|---|---|---:|---:|---|",
    ]
    for item in sample_reports:
        metrics = item.get("metrics", {})
        lines.append(
            f"| {item['task']} | {item['name']} | {metrics.get('rgb_mae', 'error')} | "
            f"{metrics.get('rgb_pearson', 'error')} | `{item.get('panel', '')}` |"
        )
    (step_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def pause_process(pid: int | None) -> Iterator[None]:
    """Pause a live training process while a checkpoint uses its MPS device."""
    paused = pid is not None and _pid_alive(pid)
    if paused:
        logger.info("Pausing training PID %d for checkpoint inference", pid)
        os.kill(pid, signal.SIGSTOP)
    try:
        yield
    finally:
        if paused and pid is not None and _pid_alive(pid):
            os.kill(pid, signal.SIGCONT)
            logger.info("Resumed training PID %d", pid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument(
        "--tasks",
        default=",".join(SUPPORTED_TASKS),
        help="Comma-separated inference tasks to monitor.",
    )
    parser.add_argument("--settle-seconds", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--train-pid", type=int, default=None)
    parser.add_argument(
        "--pause-train-during-analysis",
        action="store_true",
        help="Pause train-pid while checkpoint inference uses the same device.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    if args.samples_per_task <= 0 or args.inference_steps <= 0 or args.poll_seconds <= 0:
        parser.error("samples-per-task, inference-steps and poll-seconds must be positive")
    if not args.once and not args.watch:
        args.watch = True
    try:
        tasks = parse_task_names(args.tasks)
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    monitor_dir = output_dir / "checkpoint_monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    processed: dict[int, dict[str, Any]] = {}
    while True:
        checkpoints = numeric_checkpoints(output_dir)
        for checkpoint in checkpoints:
            step = int(checkpoint.name.rsplit("-", 1)[-1])
            if step in processed or (monitor_dir / f"step_{step}" / "report.json").exists():
                if step not in processed:
                    processed[step] = json.loads((monitor_dir / f"step_{step}" / "report.json").read_text())
                continue
            logger.info("Detected checkpoint %s; waiting %ds for file settle", checkpoint, args.settle_seconds)
            if args.settle_seconds:
                time.sleep(args.settle_seconds)
            previous = processed[max(processed)] if processed else None
            pause_pid = args.train_pid if args.pause_train_during_analysis else None
            with pause_process(pause_pid):
                processed[step] = analyze_checkpoint(
                    checkpoint,
                    data_dir,
                    monitor_dir,
                    args.device,
                    args.resolution,
                    args.samples_per_task,
                    args.inference_steps,
                    previous,
                    tasks,
                )
            logger.info("Checkpoint step=%d judgement: %s", step, processed[step]["quality"]["verdict"])
        if args.once or (not _pid_alive(args.train_pid) and checkpoints):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
