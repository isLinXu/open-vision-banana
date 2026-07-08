#!/usr/bin/env python3
"""
Open-Vision-Banana 模型评估脚本 (evaluate_model.py)

功能:
    - 对训练后的 Vision Banana 模型在所有支持的 vision task 上进行端到端评估
    - 支持 resume: 评估中断后可从断点恢复, 避免重复推理
    - 支持 DDP 分布式评估 (torchrun 启动)
    - 支持逐 batch error handling, 可配置遇到异常时跳过继续
    - 支持可视化结果导出、SOTA baseline 对比、W&B 自动上传

Usage:
    # 单卡评估 depth
    python scripts/evaluate_model.py --task depth --data-path /data/nyu --lora-path output/checkpoint-best

    # 多卡分布式评估 (DDP)
    torchrun --nproc_per_node=4 scripts/evaluate_model.py --task semantic --data-path /data/cityscapes --ddp

    # 断点续评 (resume)
    python scripts/evaluate_model.py --task normal --data-path /data/scannet --resume-state output/eval_normal_state.json

    # 限制 batch 数快速 smoke test
    python scripts/evaluate_model.py --task depth --data-path /data/nyu --max-batches 10 --compare-sota
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

# ------------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Task 常量与映射
# ------------------------------------------------------------------
SUPPORTED_TASKS: list[str] = [
    "depth",
    "normal",
    "semantic",
    "instance",
    "referring",
    "edge",
    "flow",
    "keypoint",
    "object_detection",
    "depth_edge",
    "generation_preservation",
]

# CLI task name → build_dataset registry key 的映射
TASK_TO_DATASET_KEY: dict[str, str] = {
    "object_detection": "coco_detection",
    "flow": "sintel",
    "edge": "coco_stuff",
    "keypoint": "coco_detection",
    "depth_edge": "depth",
    "generation_preservation": "depth",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回 Namespace."""
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Banana model on vision tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --task depth --data-path ./data/nyu --lora-path ./output/checkpoint-best
  %(prog)s --task semantic --data-path ./data/cityscapes --num-classes 19 --ddp
  %(prog)s --task normal --data-path ./data/scannet --resume-state ./eval_state.json
        """,
    )

    # 基础配置
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=SUPPORTED_TASKS,
        help="评估任务类型",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="评估数据集根目录路径",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="训练好的 LoRA checkpoint 路径 (目录或 hf:repo_id)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Base FLUX.1 model name (默认 FLUX.1-schnell)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=19,
        help="Semantic segmentation 的类别数 (默认 19, Cityscapes)",
    )

    # 推理超参
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="推理 batch size (默认 4)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="输入图像分辨率 (默认 512)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1,
        help="Diffusion denoising steps (默认 1, schnell 推荐)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="img2img noise strength, 越低越接近输入图 (默认 0.85)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="限制最大评估 batch 数量 (用于 smoke test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cuda / mps / cpu / auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="模型 weight dtype (默认 bfloat16)",
    )

    # 输出配置
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="评估 metrics 输出 JSON 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/evaluation",
        help="可视化结果与 state 文件保存目录 (默认 output/evaluation)",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="是否保存 prediction vs ground-truth 可视化图像",
    )
    parser.add_argument(
        "--viz-freq",
        type=int,
        default=50,
        help="每 N 个 batch 保存一次可视化 (默认 50)",
    )
    parser.add_argument(
        "--compare-sota",
        action="store_true",
        help="是否打印与 SOTA baseline 的对比表",
    )

    # Resume 与容错
    parser.add_argument(
        "--resume-state",
        type=str,
        default=None,
        help="Resume state JSON 路径; 若存在则跳过已评估 batch",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单个 batch 出错时跳过继续, 而非直接退出",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="允许的最大连续错误次数 (默认 10)",
    )

    # 分布式
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="启用 DDP 分布式评估 (需用 torchrun 启动)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (默认 0, MPS/CPU 建议保持 0)",
    )

    # W&B
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="启用 Weights & Biases 自动记录 metrics",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="open-vision-banana",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (默认自动生成)",
    )

    return parser.parse_args()


def _resolve_device(device_arg: str | None) -> str:
    """根据参数与环境自动选择最优 device."""
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """将字符串 dtype 映射为 torch.dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def _load_state(state_path: str | None) -> dict[str, Any]:
    """加载 resume state; 若文件不存在则返回空状态."""
    if state_path is None:
        return {}
    path = Path(state_path)
    if not path.exists():
        logger.info("Resume state 文件不存在, 将从头开始评估: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            "Loaded resume state: 已完成 %d / %d batches",
            state.get("completed_batches", 0),
            state.get("total_batches", 0),
        )
        return state
    except Exception as exc:
        logger.warning("读取 resume state 失败 (%s), 将从头开始", exc)
        return {}


def _save_state(state: dict[str, Any], state_path: str | None) -> None:
    """将当前评估状态持久化到磁盘, 用于 resume."""
    if state_path is None:
        return
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _advance_dataloader(dataloader: DataLoader, skip_batches: int, device: str) -> None:
    """通过空转 iterator 跳过已评估的 batch (resume 场景)."""
    if skip_batches <= 0:
        return
    logger.info("Resuming: 跳过前 %d 个已评估 batch …", skip_batches)
    iterator = iter(dataloader)
    skipped = 0
    for _ in range(skip_batches):
        try:
            next(iterator)
            skipped += 1
        except StopIteration:
            logger.warning("DataLoader 在 skip 过程中提前结束 (skipped %d / %d)", skipped, skip_batches)
            break
    logger.info("实际跳过 %d 个 batch", skipped)


def _save_visualization(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    batch_idx: int,
    output_dir: Path,
    task: str,
    sample_idx: int = 0,
) -> None:
    """将单张 prediction 与 ground truth 并排保存为 PNG."""
    try:
        import numpy as np
        from PIL import Image

        pred = pred_rgb[sample_idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        gt = gt_rgb[sample_idx].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        # 拼接: pred | gt
        canvas = np.concatenate([pred, gt], axis=1)
        canvas = (canvas * 255).astype(np.uint8)
        img = Image.fromarray(canvas)

        viz_dir = output_dir / "visualizations" / task
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_path = viz_dir / f"batch_{batch_idx:06d}_sample_{sample_idx}.png"
        img.save(save_path)
    except Exception as exc:
        logger.warning("可视化保存失败 (batch %d): %s", batch_idx, exc)


def _init_wandb(args: argparse.Namespace, is_main: bool) -> Any:
    """初始化 W&B logger (仅在主进程启用)."""
    if not args.wandb or not is_main:
        return None
    try:
        import wandb

        run_name = args.wandb_run_name or f"eval-{args.task}-{int(time.time())}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "task": args.task,
                "model_name": args.model_name,
                "lora_path": args.lora_path,
                "batch_size": args.batch_size,
                "resolution": args.resolution,
                "num_inference_steps": args.num_inference_steps,
                "strength": args.strength,
                "max_batches": args.max_batches,
            },
        )
        logger.info("W&B run initialized: %s", run.name)
        return run
    except ImportError:
        logger.warning("wandb 未安装, 跳过 W&B 记录")
        return None
    except Exception as exc:
        logger.warning("W&B 初始化失败: %s", exc)
        return None


def _log_to_wandb(wandb_run: Any, metrics: dict[str, float], step: int | None = None) -> None:
    """向 W&B 上传 scalar metrics."""
    if wandb_run is None:
        return
    try:
        import wandb

        # 过滤非 scalar 值
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb.log(scalar_metrics, step=step)
    except Exception as exc:
        logger.debug("W&B log 失败 (非致命): %s", exc)


def _print_metrics(metrics: dict[str, Any], task: str) -> None:
    """在终端打印格式化后的 metrics 表格."""
    logger.info("=" * 60)
    logger.info("Evaluation Results — Task: %s", task)
    logger.info("=" * 60)
    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info("  %-30s : %.4f", name, value)
        elif isinstance(value, list):
            formatted = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in value]
            logger.info("  %-30s : %s", name, formatted)
        elif isinstance(value, dict):
            logger.info("  %-30s :", name)
            for sub_k, sub_v in sorted(value.items()):
                if isinstance(sub_v, float):
                    logger.info("    %-28s : %.4f", sub_k, sub_v)
                else:
                    logger.info("    %-28s : %s", sub_k, sub_v)
        else:
            logger.info("  %-30s : %s", name, value)


def main() -> int:
    """主入口函数, 返回进程退出码 (0 为成功)."""
    args = parse_args()

    # ── DDP 初始化 ────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp = args.ddp and torch.cuda.is_available()

    if use_ddp:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = _resolve_device(args.device)

    is_main = (not use_ddp) or (local_rank == 0)
    world_size = torch.distributed.get_world_size() if use_ddp else 1

    if is_main:
        logger.info("=" * 60)
        logger.info("Vision Banana — Evaluate Model")
        logger.info("=" * 60)
        logger.info("Task        : %s", args.task)
        logger.info("Model       : %s", args.model_name)
        logger.info("LoRA path   : %s", args.lora_path or "(none — base model)")
        logger.info("Device      : %s", device)
        logger.info("Dtype       : %s", args.dtype)
        logger.info("Batch size  : %d", args.batch_size)
        logger.info("Resolution  : %d", args.resolution)
        logger.info("DDP         : %s", "enabled (world_size=%d)" % world_size if use_ddp else "disabled")
        logger.info("Resume      : %s", args.resume_state or "no")

    # ── 加载模型 ──────────────────────────────────────────────────────
    dtype = _resolve_dtype(args.dtype)

    if args.lora_path:
        if is_main:
            logger.info("Loading VisionBanana (FLUX.1 + LoRA) …")
        from open_vision_banana.inference.pipeline import VisionBanana

        try:
            model = VisionBanana.from_pretrained(
                model_name=args.model_name,
                lora_path=args.lora_path,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            logger.error("模型加载失败: %s", exc)
            traceback.print_exc()
            return 1
    else:
        if is_main:
            logger.warning("未提供 --lora-path, 使用 base FLUX.1 进行 zero-shot 评估")
        from open_vision_banana.inference.pipeline import VisionBanana

        try:
            model = VisionBanana.from_pretrained(
                model_name=args.model_name,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            logger.error("模型加载失败: %s", exc)
            traceback.print_exc()
            return 1

    # ── 构建 Dataset & DataLoader ─────────────────────────────────────
    from open_vision_banana.data import build_dataset

    dataset_task = TASK_TO_DATASET_KEY.get(args.task, args.task)
    try:
        dataset = build_dataset(
            dataset_task,
            args.data_path,
            split="val",
            resolution=args.resolution,
        )
    except Exception as exc:
        logger.error("Dataset 构建失败: %s", exc)
        traceback.print_exc()
        return 1

    if len(dataset) == 0:
        if is_main:
            logger.error("Dataset 为空, 请检查 --data-path: %s", args.data_path)
        return 1

    if is_main:
        logger.info("Dataset loaded: %d samples", len(dataset))

    sampler = None
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=False)

    num_workers = args.num_workers
    if device in ("mps", "cpu") and num_workers > 0:
        if is_main:
            logger.warning("%s 设备下 num_workers>0 可能导致 hang, 强制设为 0", device)
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    total_batches = len(dataloader)

    # ── Resume 状态恢复 ───────────────────────────────────────────────
    state: dict[str, Any] = {}
    state_path = args.resume_state
    if state_path is None:
        # 自动生成 state 文件路径
        state_path = str(Path(args.output_dir) / f"eval_{args.task}_state.json")

    if is_main:
        state = _load_state(state_path)
    if use_ddp:
        # 广播 state 到所有 rank
        state_list = [state] if is_main else [None]
        torch.distributed.broadcast_object_list(state_list, src=0)
        state = state_list[0] if state_list[0] is not None else {}

    completed_batches: set[int] = set(state.get("completed_batch_indices", []))
    skip_batches = max(completed_batches) + 1 if completed_batches else 0

    if skip_batches > 0:
        _advance_dataloader(dataloader, skip_batches, device)

    # ── Evaluator 初始化 ──────────────────────────────────────────────
    from open_vision_banana.evaluation.evaluator import VisionEvaluator

    class_color_map = None
    if args.task == "semantic":
        from open_vision_banana.data.datasets import CityscapesSegDataset
        class_color_map = CityscapesSegDataset.CITYSCAPES_COLORS

    evaluator = VisionEvaluator(model, device=device)

    # ── W&B 初始化 ────────────────────────────────────────────────────
    wandb_run = _init_wandb(args, is_main)

    # ── 主评估循环 ────────────────────────────────────────────────────
    if is_main:
        logger.info("开始评估, total_batches=%d, 已跳过 %d", total_batches, skip_batches)

    all_metrics: dict[str, Any] = {}
    n_batches_evaluated = 0
    n_errors = 0
    start_time = time.time()

    # 用于 resume 的实时状态
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader, start=skip_batches):
        # max_batches 限制
        if args.max_batches is not None and batch_idx >= (skip_batches + args.max_batches):
            if is_main:
                logger.info("达到 --max-batches 限制, 提前结束")
            break

        # DDP 下确保各 rank 进度一致 (barrier 可选, 这里用 iterator 同步)
        if batch_idx in completed_batches:
            continue

        try:
            image = batch["image"].to(device)
            target_rgb = batch["target_rgb"].to(device)

            # 构造额外 kwargs
            eval_kwargs: dict[str, Any] = {
                "num_inference_steps": args.num_inference_steps,
                "strength": args.strength,
            }
            if class_color_map is not None:
                eval_kwargs["class_color_map"] = class_color_map
            if "expression" in batch:
                eval_kwargs["expression"] = batch["expression"]
            if "object_name" in batch:
                eval_kwargs["object_name"] = batch["object_name"]

            # 运行 evaluate_batch (单 batch 模式, 更细粒度)
            batch_metrics = evaluator.evaluate_batch(
                image=image,
                target_rgb=target_rgb,
                task=args.task,
                **eval_kwargs,
            )

            # 累加 metrics
            for k, v in batch_metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
                elif isinstance(v, list):
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
                elif isinstance(v, dict):
                    if k not in all_metrics:
                        all_metrics[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            all_metrics[k][sub_k] = all_metrics[k].get(sub_k, 0.0) + sub_v

            n_batches_evaluated += 1
            n_errors = 0  # 重置错误计数
            completed_batches.add(batch_idx)

            # 可视化导出
            if args.save_visualizations and is_main and (batch_idx % args.viz_freq == 0):
                # 需要重新推理获取 pred_rgb (evaluate_batch 未返回 rgb)
                # 这里用 predict 做一次快速前向, 仅用于可视化
                with torch.no_grad():
                    predict_kwargs = {"task": args.task, **eval_kwargs}
                    # 移除 evaluate_batch 不接受的 key
                    predict_kwargs.pop("class_color_map", None)
                    if args.task == "semantic" and class_color_map is not None:
                        predict_kwargs["class_color_map"] = class_color_map
                    result = model.predict(image=image, **predict_kwargs)
                    pred_rgb = result.get("rgb_output", image)
                _save_visualization(pred_rgb, target_rgb, batch_idx, output_dir, args.task, sample_idx=0)

            # 每 10 个 batch 保存一次 state (resume 粒度)
            if is_main and (batch_idx % 10 == 0):
                state["completed_batch_indices"] = sorted(completed_batches)
                state["total_batches"] = total_batches
                state["task"] = args.task
                state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                _save_state(state, state_path)

            # 进度打印
            if is_main and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
                elapsed = time.time() - start_time
                throughput = (batch_idx - skip_batches + 1) / (elapsed + 1e-6)
                logger.info(
                    "Progress: batch %d/%d | throughput %.2f batch/s | elapsed %.1fs",
                    batch_idx,
                    total_batches,
                    throughput,
                    elapsed,
                )

        except Exception as exc:
            n_errors += 1
            logger.error("Batch %d 评估失败 (%d/%d 次错误): %s", batch_idx, n_errors, args.max_errors, exc)
            if not args.continue_on_error:
                traceback.print_exc()
                if is_main:
                    logger.error("使用 --continue-on-error 可跳过错误 batch 继续")
                return 1
            if n_errors >= args.max_errors:
                logger.error("连续错误次数达到上限 %d, 终止评估", args.max_errors)
                return 1
            continue

    # ── Metrics 归一化 ────────────────────────────────────────────────
    if n_batches_evaluated == 0:
        if is_main:
            logger.error("没有成功评估任何 batch")
        return 1

    final_metrics: dict[str, Any] = {}
    for k, v in all_metrics.items():
        if isinstance(v, float):
            final_metrics[k] = v / n_batches_evaluated
        elif isinstance(v, list) and v and isinstance(v[0], list):
            # list of lists → per-element mean
            elem_len = len(v[0])
            result = []
            import math

            for i in range(elem_len):
                vals = [
                    batch[i]
                    for batch in v
                    if not (isinstance(batch[i], float) and math.isnan(batch[i]))
                ]
                result.append(sum(vals) / len(vals) if vals else float("nan"))
            final_metrics[k] = result
        elif isinstance(v, dict):
            final_metrics[k] = {}
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, float):
                    final_metrics[k][sub_k] = sub_v / n_batches_evaluated
                else:
                    final_metrics[k][sub_k] = sub_v
        else:
            final_metrics[k] = v

    # ── DDP 聚合 ──────────────────────────────────────────────────────
    if use_ddp:
        final_metrics = VisionEvaluator.reduce_metrics(final_metrics, device=device)
        torch.distributed.barrier()

    # 非主进程直接退出
    if not is_main:
        if use_ddp:
            torch.distributed.destroy_process_group()
        return 0

    # ── 结果输出 ──────────────────────────────────────────────────────
    total_time = time.time() - start_time
    final_metrics["eval_time_seconds"] = round(total_time, 2)
    final_metrics["batches_evaluated"] = n_batches_evaluated
    final_metrics["samples_evaluated"] = n_batches_evaluated * args.batch_size
    final_metrics["world_size"] = world_size

    _print_metrics(final_metrics, args.task)

    # JSON 输出
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 将 float 四舍五入以便人类阅读
        json_safe = {}
        for k, v in final_metrics.items():
            if isinstance(v, float):
                json_safe[k] = round(v, 6)
            elif isinstance(v, list):
                json_safe[k] = [round(x, 6) if isinstance(x, float) else x for x in v]
            else:
                json_safe[k] = v
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_safe, f, indent=2, ensure_ascii=False)
        logger.info("Metrics 已保存至: %s", out_path)

    # 对比 SOTA
    if args.compare_sota:
        from open_vision_banana.evaluation.baselines import format_sota_comparison

        print()  # 空行分隔
        print(format_sota_comparison(args.task, final_metrics))

    # W&B 上传
    _log_to_wandb(wandb_run, final_metrics)
    if wandb_run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    # 清理 resume state (评估成功完成后)
    if state_path and Path(state_path).exists():
        # 可选择保留或删除; 这里重命名为 .completed 作为归档
        completed_path = str(Path(state_path).with_suffix(".completed.json"))
        Path(state_path).rename(completed_path)
        logger.info("Resume state 已归档: %s", completed_path)

    if use_ddp:
        torch.distributed.destroy_process_group()

    logger.info("评估完成, 总耗时 %.2f 秒", total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
