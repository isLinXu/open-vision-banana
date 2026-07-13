#!/usr/bin/env python3
"""一键下载 Vision Banana 支持的全部 base model 权重。

基于 model_registry 注册表，自动下载所有支持的 diffusion model 权重到本地缓存。
支持 HuggingFace + ModelScope 双源下载，自动镜像加速。

模型清单
==========
Core Models (论文核心):
  - Z-Image 6B (Tongyi-MAI/Z-Image)       — 当前主模型
  - FLUX.1-schnell (black-forest-labs)     — 论文原始基准
  - SSD-1B (segmind)                       — 轻量验证模型
  - SD 1.5 (runwayml)                      — 经典基准

Extended Models (扩展支持):
  - SD 2.1, SDXL, SDXL-Turbo, SD3
  - Qwen-Image, SANA 600M/1600M, Lumina-2.0
  - Chroma1-HD, FLUX.2-dev, CogView4-6B
  - AuraFlow, PixArt, Kolors, HunyuanDiT, Kandinsky 3

Usage:
    # 列出所有可用模型
    python scripts/download_all_models.py --list

    # 下载全部模型（需要大量磁盘空间 ~500GB+）
    python scripts/download_all_models.py

    # 只下载核心模型（Z-Image + FLUX + SSD-1B + SD15）
    python scripts/download_all_models.py --core

    # 下载指定模型
    python scripts/download_all_models.py --models z-image flux-schnell

    # 指定缓存目录
    python scripts/download_all_models.py --core --cache-dir /data/models

    # 按磁盘预算筛选
    python scripts/download_all_models.py --list --max-disk 50

    # 使用 ModelScope（国内推荐）
    python scripts/download_all_models.py --core --source modelscope

    # Dry-run
    python scripts/download_all_models.py --core --dry-run

Requires: huggingface_hub or modelscope
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ── HF mirror for faster downloads in China ──────────────────────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_models")


# ─── Model specifications ────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Specification for a downloadable model."""
    name: str           # short name for CLI
    model_id: str       # HuggingFace repo ID
    family: str         # model family
    size_gb: float      # estimated download size
    is_core: bool       # core model (paper-relevant)
    modelscope_id: str | None = None  # ModelScope alternative
    description: str = ""
    license: str = "unknown"
    gated: bool = False  # requires HF token / acceptance


MODELS: list[ModelSpec] = [
    # ════════════════════════════════════════════════════════════════════
    # CORE MODELS (论文核心)
    # ════════════════════════════════════════════════════════════════════

    ModelSpec(
        name="z-image",
        model_id="Tongyi-MAI/Z-Image",
        family="z-image",
        size_gb=19.0,
        is_core=True,
        modelscope_id="Tongyi-MAI/Z-Image",
        description="Z-Image 6B — current primary model (Qwen3 + 16ch VAE)",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="flux-schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        size_gb=23.0,
        is_core=True,
        modelscope_id="AI-ModelScope/FLUX.1-schnell",
        description="FLUX.1-schnell — paper original baseline (4-step generation)",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="ssd-1b",
        model_id="segmind/SSD-1B",
        family="sdxl",
        size_gb=6.5,
        is_core=True,
        modelscope_id="AI-ModelScope/SSD-1B",
        description="SSD-1B — lightweight verification model",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="sd15",
        model_id="runwayml/stable-diffusion-v1-5",
        family="sd15",
        size_gb=4.0,
        is_core=True,
        modelscope_id="AI-ModelScope/stable-diffusion-v1-5",
        description="SD 1.5 — classic baseline (860M params)",
        license="OpenRAIL++",
    ),

    # ════════════════════════════════════════════════════════════════════
    # EXTENDED MODELS
    # ════════════════════════════════════════════════════════════════════

    ModelSpec(
        name="sd21",
        model_id="stabilityai/stable-diffusion-2-1",
        family="sd2",
        size_gb=5.0,
        is_core=False,
        modelscope_id="AI-ModelScope/stable-diffusion-2-1",
        description="SD 2.1 — improved text understanding",
        license="OpenRAIL++",
    ),
    ModelSpec(
        name="sdxl",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        family="sdxl",
        size_gb=7.0,
        is_core=False,
        modelscope_id="AI-ModelScope/stable-diffusion-xl-base-1.0",
        description="SDXL — high-resolution generation",
        license="OpenRAIL++",
    ),
    ModelSpec(
        name="sdxl-turbo",
        model_id="stabilityai/sdxl-turbo",
        family="sdxl",
        size_gb=7.0,
        is_core=False,
        modelscope_id="AI-ModelScope/sdxl-turbo",
        description="SDXL Turbo — 1-4 step generation",
        license="SAIIL",
    ),
    ModelSpec(
        name="qwen-image",
        model_id="Qwen/Qwen-Image",
        family="qwen-image",
        size_gb=20.0,
        is_core=False,
        modelscope_id="Qwen/Qwen-Image",
        description="Qwen-Image — multimodal generation",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="sana-600m",
        model_id="Efficient-Large-Model/SANA_600M_1024px_diffusers",
        family="sana",
        size_gb=2.5,
        is_core=False,
        modelscope_id=None,
        description="SANA 600M — efficient high-res generation",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="sana-1600m",
        model_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        family="sana",
        size_gb=6.5,
        is_core=False,
        modelscope_id=None,
        description="SANA 1600M — larger variant",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="lumina2",
        model_id="Alpha-VLLM/Lumina-Image-2.0",
        family="lumina2",
        size_gb=7.0,
        is_core=False,
        modelscope_id=None,
        description="Lumina-Image 2.0 — flag model",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="chroma",
        model_id="lodestones/Chroma1-HD",
        family="chroma",
        size_gb=12.0,
        is_core=False,
        modelscope_id=None,
        description="Chroma1-HD — high-definition generation",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="flux2-dev",
        model_id="black-forest-labs/FLUX.2-dev",
        family="flux2",
        size_gb=24.0,
        is_core=False,
        modelscope_id=None,
        description="FLUX.2-dev — next-gen FLUX",
        license="FLUX.2 License",
        gated=True,
    ),
    ModelSpec(
        name="cogview4",
        model_id="THUDM/CogView4-6B",
        family="cogview4",
        size_gb=24.0,
        is_core=False,
        modelscope_id="ZhipuAI/CogView4-6B",
        description="CogView4-6B — Tsinghua generation model",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="aura-flow",
        model_id="fal/AuraFlow",
        family="aura-flow",
        size_gb=8.0,
        is_core=False,
        modelscope_id=None,
        description="AuraFlow — rectified flow generation",
        license="Apache-2.0",
    ),
    ModelSpec(
        name="sd3",
        model_id="stabilityai/stable-diffusion-3-medium",
        family="sd3",
        size_gb=10.0,
        is_core=False,
        modelscope_id="AI-ModelScope/stable-diffusion-3-medium",
        description="SD3 Medium — MMDiT architecture",
        license="SAIIL",
    ),
]


# ─── Download implementation ─────────────────────────────────────────────

def _get_cache_dir(cache_dir: str | None) -> Path:
    """Resolve cache directory."""
    if cache_dir:
        return Path(cache_dir).resolve()
    # Default: HF cache or local models/ dir
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def download_from_hf(
    model_id: str,
    cache_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Download model from HuggingFace Hub."""
    if dry_run:
        logger.info(f"  [DRY-RUN] Would download from HF: {model_id}")
        return True

    try:
        from huggingface_hub import snapshot_download
        logger.info(f"  Downloading from HuggingFace: {model_id}")

        # Check if already cached
        local_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
        if local_dir.exists() and not force:
            logger.info(f"  Already cached: {local_dir}")
            return True

        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False,
        )
        logger.info(f"  ✅ Downloaded to: {path}")
        return True
    except Exception as e:
        logger.error(f"  ❌ HF download failed: {e}")
        return False


def download_from_modelscope(
    model_id: str,
    cache_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Download model from ModelScope (faster in China)."""
    if dry_run:
        logger.info(f"  [DRY-RUN] Would download from ModelScope: {model_id}")
        return True

    try:
        from modelscope import snapshot_download
        logger.info(f"  Downloading from ModelScope: {model_id}")

        path = snapshot_download(
            model_id=model_id,
            cache_dir=str(cache_dir),
            resume_download=True,
        )
        logger.info(f"  ✅ Downloaded to: {path}")
        return True
    except ImportError:
        logger.warning("  modelscope not installed, falling back to HuggingFace")
        return download_from_hf(model_id, cache_dir, dry_run, force)
    except Exception as e:
        logger.error(f"  ❌ ModelScope download failed: {e}")
        logger.info("  Falling back to HuggingFace...")
        return download_from_hf(model_id, cache_dir, dry_run, force)


def download_model(
    spec: ModelSpec,
    cache_dir: Path,
    source: str = "auto",
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Download a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {spec.name} ({spec.family})")
    logger.info(f"  Size: ~{spec.size_gb:.1f} GB")
    logger.info(f"  HF ID: {spec.model_id}")
    if spec.modelscope_id:
        logger.info(f"  ModelScope ID: {spec.modelscope_id}")
    logger.info(f"  Description: {spec.description}")
    if spec.gated:
        logger.info(f"  ⚠️  Gated model — requires HF token + license acceptance")

    # Choose source
    use_modelscope = False
    if source == "modelscope":
        use_modelscope = True
    elif source == "auto":
        # Auto-detect: prefer ModelScope if available (faster in China)
        try:
            import modelscope  # noqa: F401
            use_modelscope = spec.modelscope_id is not None
        except ImportError:
            use_modelscope = False

    if use_modelscope and spec.modelscope_id:
        return download_from_modelscope(spec.modelscope_id, cache_dir, dry_run, force)
    else:
        return download_from_hf(spec.model_id, cache_dir, dry_run, force)


def check_disk_space(required_gb: float, path: Path) -> bool:
    """Check if enough disk space is available."""
    try:
        stat = os.statvfs(str(path))
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if available_gb < required_gb:
            logger.warning(
                f"  ⚠️  Low disk space: {available_gb:.1f} GB available, "
                f"~{required_gb:.1f} GB needed"
            )
            return False
        return True
    except Exception:
        return True  # Can't check, proceed anyway


# ─── Listing ─────────────────────────────────────────────────────────────

def list_models(max_disk: float | None = None) -> None:
    """List all registered models in a table."""
    print(f"\n{'='*100}")
    print(f"Vision Banana — Model Registry")
    print(f"{'='*100}")
    print(f"{'Name':<16} {'Family':<14} {'Core':<5} {'Size(GB)':<10} {'Gated':<7} {'ModelScope':<12} Description")
    print(f"{'-'*100}")

    total = 0.0
    for m in MODELS:
        core = "✓" if m.is_core else ""
        gated = "✓" if m.gated else ""
        ms = "✓" if m.modelscope_id else ""
        if max_disk is None or total + m.size_gb <= max_disk:
            print(f"{m.name:<16} {m.family:<14} {core:<5} {m.size_gb:<10.1f} {gated:<7} {ms:<12} {m.description}")
            total += m.size_gb

    core_total = sum(m.size_gb for m in MODELS if m.is_core)
    all_total = sum(m.size_gb for m in MODELS)
    print(f"{'-'*100}")
    print(f"{'CORE total':<16} {'':<14} {'':<5} {core_total:<10.1f}")
    print(f"{'ALL total':<16} {'':<14} {'':<5} {all_total:<10.1f}")
    if max_disk:
        print(f"{'Selected':<16} {'':<14} {'':<5} {total:<10.1f} (within {max_disk}GB budget)")
    print(f"{'='*100}\n")


# ─── Main pipeline ───────────────────────────────────────────────────────

def run_download(
    cache_dir: Path,
    core_only: bool = False,
    models: list[str] | None = None,
    source: str = "auto",
    dry_run: bool = False,
    force: bool = False,
    skip_gated: bool = False,
) -> None:
    """Download selected models."""
    specs = MODELS

    if core_only:
        specs = [m for m in specs if m.is_core]
    elif models:
        specs = [m for m in specs if m.name in models]

    if skip_gated:
        specs = [m for m in specs if not m.gated]

    total_size = sum(m.size_gb for m in specs)
    logger.info(f"\n{'='*60}")
    logger.info(f"Vision Banana — Model Download Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"Models to download: {len(specs)}")
    logger.info(f"Estimated total size: ~{total_size:.1f} GB")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Source: {source} | Dry-run: {dry_run}")
    logger.info(f"{'='*60}\n")

    # Check disk space
    if not dry_run:
        check_disk_space(total_size, cache_dir)

    success_count = 0
    fail_count = 0

    for spec in specs:
        try:
            ok = download_model(spec, cache_dir, source, dry_run, force)
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"  Failed: {spec.name} — {e}")
            fail_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failed:  {fail_count}")
    logger.info(f"  Cache: {cache_dir}")

    # List cached models
    logger.info(f"\nCached models:")
    for spec in specs:
        local_dir = cache_dir / f"models--{spec.model_id.replace('/', '--')}"
        exists = "✅" if local_dir.exists() else "❌"
        logger.info(f"  {exists} {spec.name:<16} {spec.model_id}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="一键下载 Vision Banana 全部 base model 权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 列出所有模型
    python scripts/download_all_models.py --list

    # 下载核心模型（Z-Image + FLUX + SSD-1B + SD15, ~52GB）
    python scripts/download_all_models.py --core

    # 只下载 Z-Image
    python scripts/download_all_models.py --models z-image

    # 使用 ModelScope（国内推荐）
    python scripts/download_all_models.py --core --source modelscope

    # 指定缓存目录
    python scripts/download_all_models.py --core --cache-dir /data/models

    # 按磁盘预算筛选
    python scripts/download_all_models.py --list --max-disk 50

    # Dry-run
    python scripts/download_all_models.py --core --dry-run
        """,
    )
    parser.add_argument("--list", action="store_true",
                        help="List all available models")
    parser.add_argument("--core", action="store_true",
                        help="Download only core models (Z-Image + FLUX + SSD-1B + SD15)")
    parser.add_argument("--models", nargs="+", type=str, default=None,
                        help="Download specific models by name")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory (default: ~/.cache/huggingface/hub)")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "huggingface", "modelscope"],
                        help="Download source (default: auto-detect)")
    parser.add_argument("--max-disk", type=float, default=None,
                        help="Filter models by max disk budget (GB, for --list)")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip gated models (requires HF token)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without downloading")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cached")

    args = parser.parse_args()

    if args.list:
        list_models(max_disk=args.max_disk)
        return

    cache_dir = _get_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.core and not args.models:
        logger.info("No selection specified. Use --core, --models, or --list.")
        logger.info("Example: python scripts/download_all_models.py --core")
        return

    run_download(
        cache_dir=cache_dir,
        core_only=args.core,
        models=args.models,
        source=args.source,
        dry_run=args.dry_run,
        force=args.force,
        skip_gated=args.skip_gated,
    )


if __name__ == "__main__":
    main()
