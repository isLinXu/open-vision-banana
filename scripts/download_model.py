#!/usr/bin/env python3
"""下载 Vision Banana 支持的 base diffusion 模型权重到本地 HuggingFace Cache。

该脚本基于 model_registry 中注册的模型配置，统一处理 23 种 model family 的
pretrained weight 下载，支持断点续传 (resume)、dry-run、内存预算筛选等功能。

Usage:
    # 列出所有可用模型及其内存需求
    python scripts/download_model.py --list

    # 下载指定模型（如 FLUX.1-dev）
    python scripts/download_model.py --model flux-dev

    # 下载并指定本地 cache 目录
    python scripts/download_model.py --model ssd-1b --cache-dir /mnt/hf-cache

    # 按内存预算自动筛选可下载的模型
    python scripts/download_model.py --list --max-memory 8

    # Dry-run：查看将要下载的内容而不实际下载
    python scripts/download_model.py --model flux-schnell --dry-run

    # 强制重新下载（忽略已有 cache）
    python scripts/download_model.py --model sd15 --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 日志配置 —— 统一格式，避免重复 handler
# ---------------------------------------------------------------------------
_logger_name = "download_model"
logger = logging.getLogger(_logger_name)


def _setup_logging(level: int = logging.INFO) -> None:
    """配置 root logger，避免与已存在的 handler 冲突。"""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# HF Mirror 与 Token 自动注入
# ---------------------------------------------------------------------------
def _ensure_hf_env() -> None:
    """自动设置 HF 镜像与 Token（如果存在本地配置文件）。"""
    # 优先使用 hf-mirror 加速国内下载
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 如果用户已经登录了 huggingface-cli，则复用其 token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists() and "HF_TOKEN" not in os.environ:
        try:
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                os.environ.setdefault("HF_TOKEN", token)
                os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        except OSError:
            pass  # 无权限读取时静默忽略


# ---------------------------------------------------------------------------
# 核心下载逻辑
# ---------------------------------------------------------------------------
def _get_registry() -> dict[str, dict]:
    """从 model_registry 提取模型元数据，避免直接 import torch。"""
    # 使用延迟 import，避免在 --list / --dry-run 时加载 torch
    try:
        from open_vision_banana.model_registry import REGISTRY, ModelConfig
    except ImportError as exc:
        logger.error(
            "无法导入 open_vision_banana.model_registry: %s\n"
            "请确保在项目根目录运行，或已安装 editable 模式: pip install -e .",
            exc,
        )
        sys.exit(1)

    entries: dict[str, dict] = {}
    for name in REGISTRY.keys():
        cfg: ModelConfig = REGISTRY[name]
        entries[name] = {
            "model_id": cfg.model_id,
            "family": cfg.family,
            "backbone_type": cfg.backbone_type,
            "memory_budget_gb": cfg.memory_budget_gb,
            "default_resolution": cfg.default_resolution,
            "description": cfg.description,
            "pipeline_class": cfg.pipeline_class,
        }
    return entries


def _download_with_resume(
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    resume: bool = True,
    force: bool = False,
    local_files_only: bool = False,
) -> Path:
    """使用 huggingface_hub.snapshot_download 执行带断点续传的下载。

    Args:
        repo_id: HuggingFace Hub 上的模型 repo ID，如 "black-forest-labs/FLUX.1-dev"。
        cache_dir: 本地 cache 目录，默认使用 HF 默认路径 (~/.cache/huggingface/hub)。
        resume: 是否启用断点续传（默认 True）。
        force: 为 True 时忽略本地 cache，强制重新下载。
        local_files_only: 仅使用本地 cache，不发起网络请求。

    Returns:
        下载完成的本地 snapshot 路径（Path）。

    Raises:
        RuntimeError: 当 snapshot_download 返回 None 或出现网络错误时抛出。
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError
    except ImportError as exc:
        logger.error(
            "缺少 huggingface_hub 依赖: %s\n"
            "请安装 inference 依赖组: pip install -e '.[inference]'",
            exc,
        )
        raise

    logger.info("开始下载: %s", repo_id)
    if cache_dir:
        logger.info("Cache 目录: %s", cache_dir)
    logger.info("断点续传: %s", "启用" if resume else "禁用")

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=resume,
            local_files_only=local_files_only,
            force_download=force,
        )
    except RepositoryNotFoundError as exc:
        logger.error("模型 repo 不存在或无权访问: %s", repo_id)
        logger.error(
            "提示: 若该模型需要 HuggingFace 登录，请先运行:\n"
            "  huggingface-cli login"
        )
        raise RuntimeError(f"RepositoryNotFoundError: {exc}") from exc
    except LocalEntryNotFoundError as exc:
        logger.error("本地 cache 中未找到该模型且 local_files_only=True: %s", repo_id)
        raise RuntimeError(f"LocalEntryNotFoundError: {exc}") from exc
    except Exception as exc:
        logger.error("下载过程中发生未知错误: %s", exc)
        raise RuntimeError(f"Download failed: {exc}") from exc

    if local_path is None:
        raise RuntimeError("snapshot_download 返回 None，下载失败。")

    return Path(local_path)


def _calculate_dir_size(path: Path) -> float:
    """递归计算目录总大小（GB）。"""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError as exc:
        logger.warning("计算目录大小时出错: %s", exc)
    return total / 1e9


def _print_model_list(entries: dict[str, dict], max_memory: Optional[float] = None) -> None:
    """打印模型列表，可选按内存预算过滤。"""
    if max_memory is not None:
        filtered = {
            k: v for k, v in entries.items()
            if v["memory_budget_gb"] <= max_memory
        }
        if not filtered:
            print(f"⚠️  没有模型满足内存预算 ≤ {max_memory} GB")
            return
        entries_to_show = filtered
        title_suffix = f"（≤ {max_memory} GB）"
    else:
        entries_to_show = entries
        title_suffix = ""

    print(f"\n可用模型列表 {title_suffix}:")
    print("-" * 95)
    print(
        f"  {'Name':<18} {'Family':<12} {'Backbone':<11} "
        f"{'VRAM(GB)':<9} {'Resolution':<11} {'Model ID'}"
    )
    print("-" * 95)

    for name, info in sorted(entries_to_show.items(), key=lambda x: x[1]["memory_budget_gb"]):
        print(
            f"  {name:<18} {info['family']:<12} {info['backbone_type']:<11} "
            f"{info['memory_budget_gb']:<9.1f} {info['default_resolution']:<11d} "
            f"{info['model_id']}"
        )

    print("-" * 95)
    print(f"共 {len(entries_to_show)} 个模型")
    print("\n下载示例:")
    print("  python scripts/download_model.py --model flux-dev")
    print("  python scripts/download_model.py --model ssd-1b --cache-dir /mnt/cache")


def _print_model_detail(name: str, info: dict) -> None:
    """打印单个模型的详细信息。"""
    print(f"\n{'='*60}")
    print(f"模型名称:    {name}")
    print(f"Model ID:    {info['model_id']}")
    print(f"Family:      {info['family']}")
    print(f"Backbone:    {info['backbone_type']}")
    print(f"VRAM 预算:   {info['memory_budget_gb']:.1f} GB")
    print(f"默认分辨率:  {info['default_resolution']} px")
    print(f"Pipeline:    {info['pipeline_class']}")
    if info["description"]:
        print(f"描述:        {info['description']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """构建并返回解析后的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="下载 Vision Banana 所需的 base diffusion 模型权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有模型
  %(prog)s --list

  # 下载 FLUX.1-schnell（Apache-2.0，适合快速实验）
  %(prog)s --model flux-schnell

  # 下载 SSD-1B（显存友好，质量/效率最佳平衡点）
  %(prog)s --model ssd-1b --cache-dir /mnt/hf-cache

  # 按 8GB 显存预算筛选可用模型
  %(prog)s --list --max-memory 8

  # 仅验证本地 cache 是否完整，不发起网络请求
  %(prog)s --model sd15 --local-files-only
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有在 model_registry 中注册的模型",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="要下载的模型名称（如 flux-dev, ssd-1b, sd15 等）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace Hub 本地 cache 目录（默认: ~/.cache/huggingface/hub）",
    )
    parser.add_argument(
        "--max-memory",
        type=float,
        default=None,
        help="与 --list 配合使用时，仅显示显存需求不超过该值的模型（单位: GB）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要下载的信息，不执行实际下载",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，忽略已有的本地 cache",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="仅使用本地 cache，不发起网络请求（用于验证 cache 完整性）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传（默认启用）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="输出 DEBUG 级别日志",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """脚本主入口。"""
    args = parse_args()
    _setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    _ensure_hf_env()

    # 提前解析 model_registry，获取模型列表
    try:
        entries = _get_registry()
    except SystemExit:
        return 1

    # ── List 模式 ──────────────────────────────────────────────────────────
    if args.list:
        _print_model_list(entries, max_memory=args.max_memory)
        return 0

    # ── 参数校验 ───────────────────────────────────────────────────────────
    if not args.model:
        logger.error("必须指定 --model <name> 或使用 --list 查看可用模型")
        return 1

    model_name = args.model.strip().lower()
    if model_name not in entries:
        logger.error("未知模型: %r", model_name)
        logger.error("可用模型: %s", ", ".join(sorted(entries.keys())))
        return 1

    info = entries[model_name]
    _print_model_detail(model_name, info)

    # ── Dry-run 模式 ───────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] 将要下载的模型信息:")
        print(f"  Model name: {model_name}")
        print(f"  Repo ID:    {info['model_id']}")
        print(f"  Cache dir:  {args.cache_dir or '（默认）'}")
        print(f"  Resume:     {not args.no_resume}")
        print(f"  Force:      {args.force}")
        print("  未执行实际下载。")
        return 0

    # ── 实际下载 ───────────────────────────────────────────────────────────
    if args.local_files_only:
        logger.info("local-files-only 模式：仅验证本地 cache")

    logger.info("开始下载模型: %s (%s)", model_name, info["model_id"])

    try:
        local_path = _download_with_resume(
            repo_id=info["model_id"],
            cache_dir=args.cache_dir,
            resume=not args.no_resume,
            force=args.force,
            local_files_only=args.local_files_only,
        )
    except RuntimeError as exc:
        logger.error("下载失败: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断下载。已下载的部分保存在 cache 中，下次运行时自动续传。")
        return 130  # 标准 Unix 中断退出码

    # ── 下载后统计 ─────────────────────────────────────────────────────────
    total_gb = _calculate_dir_size(local_path)
    logger.info("✅ 下载完成")
    logger.info("  本地路径: %s", local_path)
    logger.info("  总大小:   %.2f GB", total_gb)
    logger.info("  模型:     %s (%s)", model_name, info["model_id"])

    # 输出纯路径（方便 shell 脚本捕获）
    print(str(local_path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
