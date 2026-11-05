#!/usr/bin/env python3
"""Evaluate decoded-task metrics on the paper-aligned benchmarks.

Examples:
    python scripts/evaluate_paper_benchmarks.py \
        --model-name black-forest-labs/FLUX.1-schnell \
        --lora-path output/checkpoint-best \
        --depth-root /data/benchmarks \
        --normal-root /data/benchmarks \
        --cityscapes-root /data/cityscapes \
        --benchmark all \
        --output-json output/eval/paper_metrics.json

    python scripts/evaluate_paper_benchmarks.py \
        --benchmark semantic --cityscapes-root /data/cityscapes \
        --output-json output/eval/preflight.json --preflight-only

Evaluation requires local benchmark data and model weights. Preflight-only
mode requires benchmark data but never loads model weights. Neither mode falls
back to synthetic data or RGB reconstruction metrics.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import platform
import sys
from collections.abc import Container
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.segmentation import (  # noqa: E402
    PAPER_INSTANCE_DECODING_CONFIG,
    InstanceDecodingConfig,
)
from open_vision_banana.data import build_dataset  # noqa: E402
from open_vision_banana.data.manifest import fingerprint_records  # noqa: E402
from open_vision_banana.evaluation.evaluator import VisionEvaluator  # noqa: E402
from open_vision_banana.evaluation.paper_capabilities import (  # noqa: E402
    CAPABILITY_TO_BENCHMARKS,
    PAPER_CAPABILITIES,
    capability_for_benchmark,
    paper_reference,
)
from open_vision_banana.evaluation.results_io import write_metrics_csv, write_results  # noqa: E402
from open_vision_banana.evaluation.shards import SHARDABLE_BENCHMARKS, shard_range  # noqa: E402
from open_vision_banana.inference.checkpoint import read_training_prompts  # noqa: E402
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT  # noqa: E402

ADE150_SEMANTIC_MARKERS = (
    "150-class semantic segmentation",
    "fixed training palette",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paper_benchmarks")


DEPTH_DATASETS = {
    "nyu": "nyu_depth",
    "ibims": "ibims",
    "eth3d": "eth3d",
    "diode": "diode",
    "kitti": "kitti",
    "nuscenes": "nuscenes_depth",
}
# Dataset adapters that can return unresized GT (``native_gt=True``).
NATIVE_GT_DATASETS = {
    "nyu_depth",
    "ibims",
    "diode",
    "kitti",
    "eth3d",
    "nyu_normal",
    "diode_normal",
    "normal",
    "vkitti_normal",
    "referring",
    "reasonseg",
    "nuscenes_depth",
    "semantic",  # the Cityscapes job's adapter key (build_jobs: semantic/cityscapes)
}
# Extra adapter kwargs that make a benchmark match the paper's definition.
# The paper evaluates DIODE-Indoor, not the indoor+outdoor union, and RefCOCOg
# on the UMD split.
BENCHMARK_DATASET_KWARGS: dict[str, dict[str, Any]] = {
    "diode": {"scene_type": "indoors"},
    "diode_normal": {"scene_type": "indoors"},
    "referring": {"protocol": "umd", "strict": True},
}
NORMAL_DATASETS = {
    "nyu": "nyu_normal",
    "diode": "diode_normal",
    "scannet": "normal",
    "vkitti": "vkitti_normal",
}


def parse_instance_decoding(spec: str | None) -> InstanceDecodingConfig:
    """Paper Appendix A defaults, with ``key=value`` overrides for ablations."""
    base = PAPER_INSTANCE_DECODING_CONFIG.to_dict()
    if spec:
        for item in spec.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            key = key.strip()
            if key not in base:
                raise ValueError(f"Unknown instance decoding field {key!r}; choose from {sorted(base)}")
            current = base[key]
            base[key] = (
                value.strip().lower() in {"1", "true", "yes"} if isinstance(current, bool)
                else type(current)(value.strip())
            )
    return InstanceDecodingConfig(**base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=[
            "depth",
            "normal",
            "semantic",
            "referring",
            "instance",
            "generation",
            "2d",
            "3d",
            "all",
        ],
        default="all",
        help=(
            "Paper capability groups: 2d = Cityscapes + RefCOCOg + ReasonSeg + "
            "SA-Co/Gold; 3d = NYU/iBims/ETH3D/DIODE-Indoor/KITTI/nuScenes depth "
            "plus normals; generation = GenAI-Bench / ImgEdit proxies; all = 2d+3d+generation."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model identifier. Required for evaluation, omitted for --preflight-only.",
    )
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Sequential CPU offload. Use this for FLUX.1-dev on a 24 GB GPU.",
    )
    parser.add_argument(
        "--quantize",
        choices=["4bit"],
        default=None,
        help="Load a FLUX transformer in 4-bit NF4 (and T5 in 8-bit) so a 24 GB GPU can evaluate without sequential offload.",
    )
    # SANA's MPS kernels can produce non-finite img2img samples in fp16/bf16.
    # Keep evaluation reproducible and finite by defaulting to fp32; users may
    # still request reduced precision on CUDA/CPU explicitly.
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--allow-mps-reduced-dtype",
        action="store_true",
        help=(
            "Explicitly allow fp16/bf16 on MPS for an audited compatibility run. "
            "Non-finite generator output still hard-fails."
        ),
    )
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--normal-root", type=Path, default=None)
    parser.add_argument("--cityscapes-root", type=Path, default=None)
    parser.add_argument("--nyu-root", type=Path, default=None)
    parser.add_argument("--ibims-root", type=Path, default=None)
    parser.add_argument("--eth3d-root", type=Path, default=None)
    parser.add_argument("--diode-root", type=Path, default=None)
    parser.add_argument("--kitti-root", type=Path, default=None)
    parser.add_argument("--nuscenes-root", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=None)
    parser.add_argument("--vkitti-root", type=Path, default=None)
    parser.add_argument("--refcocog-root", type=Path, default=None)
    parser.add_argument("--reasonseg-root", type=Path, default=None)
    parser.add_argument(
        "--reasonseg-query-mode", choices=("all", "first"), default="all",
        help="ReasonSeg: score every query (default), or only the first for historical comparisons.",
    )
    parser.add_argument("--saco-root", type=Path, default=None)
    parser.add_argument(
        "--instance-decoding",
        default=None,
        help=(
            "Override Appendix A decoder fields as key=value,key=value, e.g. "
            "'seed_merge_rule=pixel_and_seed,spatial_merge_area_basis=pixel' for the 2x2 "
            "ablation of the two readings the paper leaves open. Default: PAPER_INSTANCE_DECODING_CONFIG."
        ),
    )
    parser.add_argument("--genai-bench-root", type=Path, default=None)
    parser.add_argument("--imgedit-root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--native-resolution",
        action="store_true",
        help=(
            "Score at the original GT resolution. Depth: NYU, iBims, DIODE, "
            "KITTI, ETH3D, nuScenes. Normals: NYU, DIODE-Indoor, ScanNet, VKITTI. "
            "Referring: RefCOCOg and ReasonSeg masks. Semantic jobs ignore the flag. "
            "Use --batch-size 1 for datasets with mixed image sizes."
        ),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    # Inference protocol. The paper states none of these; exp_setting.md pins the values a run
    # must use, and every value is written to results["metadata"]["inference_protocol"].
    parser.add_argument("--num-inference-steps", type=int, default=1,
                        help="denoising steps (default 1 is a smoke setting, not an evaluation setting)")
    parser.add_argument("--guidance-scale", type=float, default=0.0, help="classifier-free guidance scale (0 = off)")
    parser.add_argument("--strength", type=float, default=0.85,
                        help="img2img noise strength; ignored by instruction-editor layouts that start from noise")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--depth-colormap",
        default="cube_edge",
        help="Colormap used to invert predicted depth RGB. Default matches the paper cube-edge.",
    )
    parser.add_argument(
        "--depth-curve",
        choices=["metric_power", "log_relative"],
        default="metric_power",
        help=(
            "1-D depth curve before the colormap. metric_power is the Vision Banana "
            "paper protocol. log_relative is an opt-in median+log ablation; decode "
            "units are relative and must not be cited as paper AbsRel."
        ),
    )
    parser.add_argument(
        "--depth-log-alpha",
        type=float,
        default=0.4,
        help="α for --depth-curve log_relative (clip(α ln(d_rel+1))). Ignored otherwise.",
    )
    parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Decode bright pixels as near (1 - normalized) before the power inverse.",
    )
    parser.add_argument(
        "--depth-prompt",
        default=None,
        help="Override the depth prompt. Use the checkpoint's training prompt when metadata.json has none.",
    )
    parser.add_argument(
        "--relative-luminance",
        action="store_true",
        help="Also score a closer=brighter luminance map after median scale (relative_delta_1.25_aligned).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Score one contiguous block of each benchmark (with --shard-index) and store the raw "
            "accumulator sums; scripts/merge_eval_shards.py adds the blocks back into one result. "
            "Seeds stay those of the unsharded run. Depth, semantic and referring jobs only."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--allow-semantic-palette-mismatch",
        action="store_true",
        help=(
            "Allow Cityscapes eval of a LoRA whose recorded semantic prompt is the "
            "ADE-150 / SNVC training palette. That score is not paper mIoU."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate benchmark roots and splits, then write the audit without loading a model.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--referring-prompt-version", default="query_v2",
                        choices=("query_v2", "fig4_v1", "segment_v0"))
    return parser.parse_args()


def resolve_device(value: str | None) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(value: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def resolve_runtime_dtype(
    value: str,
    device: str,
    *,
    allow_mps_reduced_dtype: bool = False,
) -> torch.dtype:
    """Resolve an evaluation dtype and reject known-unstable MPS modes."""
    dtype = resolve_dtype(value)
    if device == "mps" and dtype != torch.float32 and not allow_mps_reduced_dtype:
        raise ValueError(
            "SANA evaluation on MPS requires --dtype float32; fp16/bf16 can "
            "produce non-finite scheduler outputs. Pass "
            "--allow-mps-reduced-dtype only for an explicit compatibility run."
        )
    return dtype


def resolve_inference_steps(model_name: str | None, requested: int) -> int:
    """Keep FLUX-schnell's 1-step default, but use multi-step schedules for SANA/Qwen/editors."""
    if requested != 1 or not model_name:
        return requested
    from open_vision_banana.model_registry import resolve_model_config

    try:
        config = resolve_model_config(model_name)
    except Exception:
        return requested
    if (
        config.family in {"sana", "qwen-image", "qwen-image-2.1"}
        or getattr(config, "instruction_editing", False)
    ):
        return int(config.default_num_inference_steps or 20)
    return requested


def evaluate_compatible(evaluator: Any, dataloader, **kwargs: Any) -> dict[str, Any]:
    """Call ``VisionEvaluator.evaluate`` without kwargs the deploy copy rejects.

    Older deploy trees lack ``depth_curve`` / ``depth_log_alpha``. Passing them
    raises TypeError and drops the whole benchmark run.
    """
    params = inspect.signature(evaluator.evaluate).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return evaluator.evaluate(dataloader, **kwargs)
    dropped = sorted(key for key in kwargs if key not in params)
    if dropped:
        logger.warning("VisionEvaluator.evaluate ignored unsupported kwargs: %s", dropped)
    return evaluator.evaluate(
        dataloader,
        **{key: value for key, value in kwargs.items() if key in params},
    )


def fingerprint_dataset(dataset: Any) -> str:
    """Hash dataset records with paths normalized to the dataset root."""
    root = getattr(dataset, "root_dir", None)
    records: list[dict[str, Any]] = []
    for sample in getattr(dataset, "samples", []):
        record: dict[str, Any] = {}
        for key in ("image", "target", "expression", "query", "object_name", "annotation"):
            value = sample.get(key)
            if value is not None:
                record[key] = str(value)
        records.append(record)
    return fingerprint_records(records, root)


def select_dataset(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def selected_benchmarks(choice: str) -> set[str]:
    """Expand ``2d`` / ``3d`` / ``all`` onto concrete eval tasks."""
    if choice == "all":
        return {"depth", "normal", "semantic", "referring", "instance", "generation"}
    if choice in CAPABILITY_TO_BENCHMARKS:
        return set(CAPABILITY_TO_BENCHMARKS[choice])
    return {choice}


def _resolve_collection_root(root: Path | None, name: str) -> Path | None:
    """Resolve either a per-dataset root or a collection with named children."""
    if root is None:
        return None
    nested = root / name
    if nested.is_dir():
        return nested
    # A collection root must not be passed to every adapter: doing so turns a
    # missing dataset into a misleading "empty layout" error for another one.
    if root.name.casefold() == name.casefold():
        return root
    return root / name if root.is_dir() else None


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    jobs: list[tuple[str, str, Path, str]] = []
    wanted = selected_benchmarks(args.benchmark)
    if "depth" in wanted:
        depth_roots = {
            "nyu": args.nyu_root or _resolve_collection_root(args.depth_root, "nyu"),
            "ibims": args.ibims_root or _resolve_collection_root(args.depth_root, "ibims"),
            "eth3d": args.eth3d_root or _resolve_collection_root(args.depth_root, "eth3d"),
            "diode": args.diode_root or _resolve_collection_root(args.depth_root, "diode"),
            "kitti": args.kitti_root or _resolve_collection_root(args.depth_root, "kitti"),
            "nuscenes": args.nuscenes_root or _resolve_collection_root(args.depth_root, "nuscenes"),
        }
        for name, task in DEPTH_DATASETS.items():
            if depth_roots[name] is not None:
                jobs.append(("depth", name, depth_roots[name], task))
    if "normal" in wanted:
        normal_roots = {
            "nyu": args.nyu_root or _resolve_collection_root(args.normal_root, "nyu"),
            "diode": args.diode_root or _resolve_collection_root(args.normal_root, "diode"),
            "scannet": args.scannet_root or _resolve_collection_root(args.normal_root, "scannet"),
            "vkitti": args.vkitti_root or _resolve_collection_root(args.normal_root, "vkitti"),
        }
        for name, task in NORMAL_DATASETS.items():
            if normal_roots[name] is not None:
                jobs.append(("normal", name, normal_roots[name], task))
    if "semantic" in wanted and args.cityscapes_root is not None:
        jobs.append(("semantic", "cityscapes", args.cityscapes_root, "semantic"))
    if "referring" in wanted and args.refcocog_root is not None:
        jobs.append(("referring", "refcocog", args.refcocog_root, "referring"))
    if "referring" in wanted and args.reasonseg_root is not None:
        jobs.append(("referring", "reasonseg", args.reasonseg_root, "reasonseg"))
    if "instance" in wanted and args.saco_root is not None:
        jobs.append(("instance", "saco", args.saco_root, "instance"))
    if "generation" in wanted and args.genai_bench_root is not None:
        jobs.append(("generation", "genai_bench", args.genai_bench_root, "genai_bench"))
    if "generation" in wanted and args.imgedit_root is not None:
        jobs.append(("generation", "imgedit", args.imgedit_root, "imgedit"))
    return jobs


def optional_normal_jobs(args: argparse.Namespace) -> set[tuple[str, str]]:
    """Normal jobs that ``--benchmark all`` derives from ``--nyu-root`` / ``--diode-root``.

    The official NYU and DIODE depth downloads carry no normal maps, so such a job may come up
    empty and is then skipped. Normals stay required when asked for: ``--benchmark normal`` or
    ``3d``, or any ``--normal-root``.
    """
    if args.benchmark != "all" or args.normal_root is not None:
        return set()
    depth_roots = {"nyu": args.nyu_root, "diode": args.diode_root}
    return {("normal", name) for name, root in depth_roots.items() if root is not None}


def preflight_jobs(
    jobs: list[tuple[str, str, Path, str]],
    *,
    split: str,
    resolution: int,
    optional: Container[tuple[str, str]] = frozenset(),
    reasonseg_query_mode: str = "all",
) -> dict[str, Any]:
    """Check requested benchmark adapters before allocating model resources.

    An ``optional`` job whose adapter finds no pairs goes to ``skipped`` instead of failing.
    """
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for benchmark, name, root, dataset_key in jobs:
        dataset_split = split
        root = root.expanduser()
        record: dict[str, Any] = {
            "benchmark": benchmark,
            "dataset": name,
            "dataset_key": dataset_key,
            "root": str(root),
            "split": dataset_split,
            "ok": False,
        }
        if not root.is_dir():
            record["error"] = f"Benchmark root does not exist or is not a directory: {root}"
        else:
            try:
                dataset = build_dataset(
                    dataset_key,
                    str(root),
                    split=dataset_split,
                    resolution=resolution,
                    **BENCHMARK_DATASET_KWARGS.get(dataset_key, {}),
                    **({"query_mode": reasonseg_query_mode} if dataset_key == "reasonseg" else {}),
                )
                record["samples"] = len(dataset)
                if len(dataset) == 0 and (benchmark, name) in optional:
                    record["skipped"] = f"no {benchmark} pairs under {root}"
                    skipped.append(record)
                    continue
                if len(dataset) == 0:
                    record["error"] = (
                        "Adapter found no usable pairs. Verify the dataset layout and requested split."
                    )
                else:
                    record["ok"] = True
                    record["dataset_fingerprint"] = fingerprint_dataset(dataset)
            except Exception as exc:
                record["error"] = f"Adapter initialization failed: {type(exc).__name__}: {exc}"
        records.append(record)
    return {"ok": all(record["ok"] for record in records), "jobs": records, "skipped": skipped}


def main() -> int:
    args = parse_args()
    if not args.preflight_only and not args.model_name:
        raise SystemExit("--model-name is required unless --preflight-only is set")
    args.num_inference_steps = resolve_inference_steps(args.model_name, args.num_inference_steps)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("No benchmark roots supplied")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"--shard-index must be in [0, --num-shards={args.num_shards})")
    unshardable = sorted({job[0] for job in jobs} - SHARDABLE_BENCHMARKS)
    if args.num_shards > 1 and unshardable:
        raise SystemExit(
            f"--num-shards supports {sorted(SHARDABLE_BENCHMARKS)} jobs, not {unshardable}"
        )

    preflight = preflight_jobs(
        jobs, split=args.split, resolution=args.resolution, optional=optional_normal_jobs(args),
        reasonseg_query_mode=args.reasonseg_query_mode,
    )
    skipped = {(record["benchmark"], record["dataset"]): record for record in preflight["skipped"]}
    for (benchmark, name), record in skipped.items():
        logger.warning("%s/%s skipped: %s", benchmark, name, record["skipped"])
    jobs = [job for job in jobs if (job[0], job[1]) not in skipped]
    results: dict[str, Any] = {
        "metadata": {
            "evaluation_mode": "decoded_task_metrics",
            "model_name": args.model_name,
            "lora_path": str(args.lora_path) if args.lora_path else None,
            "cpu_offload": bool(args.cpu_offload),
            "quantize": args.quantize,
            "device": device,
            "dtype": args.dtype,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "reasonseg_query_mode": args.reasonseg_query_mode,
            "reasonseg_protocol": f"{args.reasonseg_query_mode}_queries_v1",
            "num_inference_steps": args.num_inference_steps,
            "strength": args.strength,
            "seed": args.seed,
            "skipped_benchmarks": [
                f"{record['benchmark']}/{record['dataset']}" for record in preflight["skipped"]
            ],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "capability": args.benchmark,
            "paper_capabilities": PAPER_CAPABILITIES,
            "depth_colormap": args.depth_colormap,
            "depth_curve": args.depth_curve,
            "depth_log_alpha": args.depth_log_alpha,
            "invert_depth": bool(args.invert_depth),
        },
        "preflight": preflight,
        "benchmarks": {},
    }
    if args.preflight_only or not preflight["ok"]:
        write_results(args.output_json, results)
        for record in preflight["jobs"]:
            if not record["ok"]:
                logger.error("%s/%s: %s", record["benchmark"], record["dataset"], record["error"])
        if not preflight["ok"]:
            return 2
        return 0

    if any(job[0] == "semantic" for job in jobs) and args.lora_path and not args.allow_semantic_palette_mismatch:
        recorded = read_training_prompts(args.lora_path).get("semantic", "")
        if recorded and any(marker in recorded for marker in ADE150_SEMANTIC_MARKERS):
            raise SystemExit(
                "LoRA semantic prompt is the ADE-150 / SNVC palette; official "
                "Cityscapes val needs the 19-color specialist "
                "(configs/experiments/sana_1600m_cityscapes19.yaml). Pass "
                "--allow-semantic-palette-mismatch only for a cross-palette diagnostic."
            )
        if recorded and recorded != SEMANTIC_SEGMENTATION_PROMPT:
            logger.warning(
                "LoRA semantic prompt is not the official Cityscapes 19-color legend; "
                "decoded mIoU may not be paper-protocol."
            )

    from open_vision_banana.inference.pipeline import VisionBanana

    runtime_dtype = resolve_runtime_dtype(
        args.dtype,
        device,
        allow_mps_reduced_dtype=args.allow_mps_reduced_dtype,
    )
    results["metadata"]["dtype"] = str(runtime_dtype).replace("torch.", "")
    instance_decoding = parse_instance_decoding(args.instance_decoding)
    results["metadata"]["instance_decoding"] = instance_decoding.to_dict()
    model = VisionBanana.from_pretrained(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device=device,
        dtype=runtime_dtype,
        enable_sequential_cpu_offload=args.cpu_offload,
        quantize=args.quantize,
        instance_decoding_config=instance_decoding,
    )
    model.referring_prompt_version = args.referring_prompt_version
    if args.depth_prompt:
        model.training_prompts["depth"] = args.depth_prompt
    evaluator = VisionEvaluator(model, device=device)
    results["metadata"]["depth_colormap"] = args.depth_colormap
    results["metadata"]["depth_curve"] = args.depth_curve
    results["metadata"]["depth_log_alpha"] = args.depth_log_alpha
    results["metadata"]["invert_depth"] = bool(args.invert_depth)
    results["metadata"]["referring_prompt_version"] = args.referring_prompt_version
    results["metadata"]["reasonseg_expression_selection"] = args.reasonseg_query_mode
    results["metadata"]["depth_prompt"] = args.depth_prompt
    results["metadata"]["relative_luminance"] = bool(args.relative_luminance)
    results["metadata"]["inference_protocol"] = {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "seed": args.seed,
        "canvas_resolution": args.resolution,
        "native_resolution_scoring": bool(args.native_resolution),
        "note": "The paper does not state inference settings; these are the repository values for this run "
                "(exp_setting.md E1.3). num_inference_steps=1 is a smoke setting and must not be reported as a result.",
    }
    results["metadata"]["protocol"] = {
        "depth": "decoded RGB to metric depth with curve=metric_power (paper Eq.1); log_relative is an opt-in relative ablation and must not be cited as paper AbsRel; valid GT pixels only; NYU Eigen crop + 10m cap / KITTI Garg crop + 80m cap / ETH3D 80m cap / nuScenes 80m cap (no paper crop) when those datasets are named; AbsRel_aligned/delta_1.25_aligned are per-image median-scale diagnostics, not paper protocol; *_per_image keys are means of per-image metrics (headline keys are pixel-weighted); crop/cap/aggregation are conventions, not paper-disclosed",
        "normal": "decoded RGB to unit camera-space normals; valid pixels are finite, non-zero GT (plus DIODE mask when present); mean_angle_deg_per_image is the mean of per-image means; NYU/ScanNet require community normal maps; VKITTI2 ships no normals, so VirtualKITTINormalDataset derives them from its centimetre depth + intrinsic.txt in the paper frame (used only to build GT, never as model input); DIODE is indoors",
        "semantic": "decoded RGB to class labels; Cityscapes labelIds mapped to trainIds; global confusion matrix; official gtFine only, never cityscapes8; with --native-resolution the canvas is decoded first and the label map nearest-upsampled to the 2048x1024 GT (interpolating RGB would blend boundary colours into phantom classes)",
        "referring": "RefCOCOg requires refs(umd).p with no protocol fallback and validates images, expressions, and nonempty native COCO polygon/RLE masks using pycocotools; ReasonSeg val uses LISA JSON-next-to-image; one sample per expression, ReasonSeg expands all queries by default (first is an explicit historical option); gIoU/IoU is the equal-weight mean over queries, cIoU is summed intersection / summed union over queries; no image-level averaging; paper ReasonSeg gIoU includes a Gemini 2.5 Pro rewrite that this command does not run unless rewritten_text is already in the JSON",
        "instance": "SA-Co/Gold query JSONL when present, else color/instance pairs; cgF1 = 100 * pmF1 * IL_MCC; paper IL_MCC uses Gemini 3.1 Flash-Lite presence filtering, which this command does not invent",
        "generation": "paper GenAI-Bench / ImgEdit numbers are human pairwise win rates; this command only scores an optional CLIP/FID proxy when images exist and never writes those proxies into the paper win-rate column",
        "aggregation": "task-specific global accumulators over the selected dataset; referring gIoU is query-weighted and cIoU is cumulative intersection / union, not batch mean",
    }

    for benchmark, name, root, dataset_key in jobs:
        # Preflight and execution must use the same requested split.  The
        # default is ``val`` for the paper gate; callers may explicitly run
        # the official train split when that data is present.
        split = args.split
        native = args.native_resolution and benchmark in {"depth", "normal", "referring", "semantic"}
        if native and dataset_key not in NATIVE_GT_DATASETS:
            raise SystemExit(
                f"--native-resolution is not implemented for dataset {dataset_key!r}; "
                f"supported: {sorted(NATIVE_GT_DATASETS)}"
            )
        dataset_kwargs = dict(BENCHMARK_DATASET_KWARGS.get(dataset_key, {}))
        if dataset_key == "reasonseg":
            dataset_kwargs["query_mode"] = args.reasonseg_query_mode
        dataset = build_dataset(
            dataset_key,
            str(root),
            split=split,
            resolution=args.resolution,
            **dataset_kwargs,
            **({"native_gt": True} if native else {}),
        )
        if len(dataset) == 0:
            raise RuntimeError(f"Empty benchmark dataset: {name} ({root})")
        key = f"{benchmark}/{name}"
        record = {
            "capability": capability_for_benchmark(benchmark, name),
            "dataset_key": dataset_key,
            "root": str(root),
            "split": split,
            "native_resolution": native,
            "dataset_kwargs": dataset_kwargs,
            "samples": len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples),
            "source_samples": len(dataset),
            "dataset_fingerprint": fingerprint_dataset(dataset),
            "paper_reference": paper_reference(benchmark, name),
        }
        if benchmark == "generation" and not getattr(dataset, "has_images", False):
            record["decoded_metrics"] = None
            record["blocked"] = (
                "prompt-only set; paper metric is a human pairwise win rate. "
                "Use scripts/evaluate_generation_preservation.py after generating images "
                "for a CLIP/FID proxy — never cite that proxy as GenAI-Bench or ImgEdit."
            )
            results["benchmarks"][key] = record
            logger.info("%s: %s", key, record["blocked"])
            continue
        selected = select_dataset(dataset, args.max_samples)
        job_seed = args.seed
        if args.num_shards > 1:
            block = shard_range(len(selected), args.num_shards, args.shard_index, args.batch_size)
            record["shard"] = {
                "index": args.shard_index,
                "count": args.num_shards,
                "start": block.start,
                "stop": block.stop,
                "of_samples": len(selected),
            }
            selected = Subset(selected, block)
            # the evaluator seeds batch i with seed + i, so the block keeps the unsharded seeds
            job_seed = args.seed + block.start // args.batch_size
        loader = DataLoader(
            selected,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if device == "cuda" else 0,
        )
        class_color_map = None
        if benchmark == "semantic":
            from open_vision_banana.data.datasets import CityscapesSegDataset
            class_color_map = CityscapesSegDataset.CITYSCAPES_COLORS
        eval_task = "generation_preservation" if benchmark == "generation" else benchmark
        metrics = evaluate_compatible(
            evaluator,
            loader,
            task=eval_task,
            num_classes=19 if benchmark == "semantic" else None,
            class_color_map=class_color_map,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            seed=job_seed,
            guidance_scale=args.guidance_scale,
            dataset_name=name,
            native_resolution=native,
            depth_colormap=args.depth_colormap,
            depth_curve=args.depth_curve,
            depth_log_alpha=args.depth_log_alpha,
            invert_depth=args.invert_depth,
            report_relative_luminance=args.relative_luminance,
        )
        if benchmark == "generation":
            metrics = dict(metrics)
            metrics["formal_genai_bench"] = False
            metrics["formal_imgedit"] = False
        record["samples"] = len(selected)
        record["decoded_metrics"] = metrics
        if args.num_shards > 1:
            record["accumulator_states"] = getattr(evaluator, "last_accumulator_states", None)
        results["benchmarks"][key] = record
        logger.info("%s: %s", key, metrics)

    write_results(args.output_json, results)
    if args.output_csv is not None:
        write_metrics_csv(args.output_csv, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
