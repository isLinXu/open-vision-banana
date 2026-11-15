"""Dataset format validation for Vision Banana instruction tuning.

Checks paired ``images/`` + ``targets/`` directories for common issues
before training or fine-tuning experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ValidationIssue:
    """A single validation finding."""

    level: str  # "error" | "warning"
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    """Aggregated validation results for a dataset directory."""

    data_path: str
    pair_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.data_path}",
            f"Valid pairs: {self.pair_count}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Status: {'PASS' if self.ok else 'FAIL'}",
        ]
        for issue in self.issues:
            prefix = "ERROR" if issue.level == "error" else "WARN"
            loc = f" ({issue.path})" if issue.path else ""
            lines.append(f"  [{prefix}]{loc} {issue.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "pair_count": self.pair_count,
            "ok": self.ok,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
            "stats": self.stats,
        }


def _find_target(tgt_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        candidate = tgt_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_rgb_size(path: Path) -> tuple[tuple[int, int], tuple[float, float]]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        extrema = rgb.getextrema()
        assert len(extrema) == 3, "Expected RGB image with 3 bands"
        mins = (extrema[0][0], extrema[1][0], extrema[2][0])
        maxs = (extrema[0][1], extrema[1][1], extrema[2][1])
        return rgb.size, (min(mins), max(maxs))


def validate_vision_dataset(
    data_path: str | Path,
    *,
    min_pairs: int = 1,
    require_square: bool = False,
    min_resolution: int = 64,
    max_resolution: int | None = None,
    check_value_range: bool = True,
    task: str | None = None,
    depth_colormap: str = "cube_edge",
) -> ValidationReport:
    """Validate a flat ``images/`` + ``targets/`` dataset directory.

    Expected layout::

        <data_path>/
          images/    input RGB images
          targets/   matching RGB visualisations (same stem)

    Args:
        data_path: Root directory containing ``images/`` and ``targets/``.
        min_pairs: Minimum number of valid pairs required.
        require_square: Warn when width != height.
        min_resolution: Minimum allowed width/height in pixels.
        max_resolution: Optional maximum allowed width/height.
        check_value_range: Warn when pixel values look out of [0, 255].
        task: Optional task name enabling semantic target validation.
        depth_colormap: Expected depth RGB colormap when ``task='depth'``.

    Returns:
        ValidationReport with errors, warnings, and pair count.
    """
    root = Path(data_path)
    report = ValidationReport(data_path=str(root.resolve()))

    img_dir = root / "images"
    tgt_dir = root / "targets"

    if not root.exists():
        report.issues.append(ValidationIssue("error", "Data path does not exist"))
        return report

    if not img_dir.is_dir():
        report.issues.append(
            ValidationIssue("error", "Missing images/ directory", str(img_dir))
        )
        return report

    if not tgt_dir.is_dir():
        report.issues.append(
            ValidationIssue("error", "Missing targets/ directory", str(tgt_dir))
        )
        return report

    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    if not image_files:
        report.issues.append(
            ValidationIssue("error", "No image files found in images/", str(img_dir))
        )
        return report

    matched_targets: set[str] = set()
    seen_sizes: set[tuple[int, int]] = set()
    semantic_stats: dict[str, list[float]] = {}

    for img_path in image_files:
        tgt_path = _find_target(tgt_dir, img_path.stem)
        if tgt_path is None:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"No matching target for {img_path.name}",
                    str(img_path),
                )
            )
            continue

        matched_targets.add(tgt_path.stem)
        report.pair_count += 1

        try:
            img_size, (img_lo, img_hi) = _load_rgb_size(img_path)
            tgt_size, (tgt_lo, tgt_hi) = _load_rgb_size(tgt_path)
        except OSError as exc:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"Failed to read image pair: {exc}",
                    str(img_path),
                )
            )
            continue

        if img_size != tgt_size:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"Size mismatch: image {img_size} vs target {tgt_size}",
                    str(img_path),
                )
            )

        w, h = img_size
        if w < min_resolution or h < min_resolution:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"Resolution {w}x{h} below minimum {min_resolution}",
                    str(img_path),
                )
            )

        if max_resolution is not None and (w > max_resolution or h > max_resolution):
            report.issues.append(
                ValidationIssue(
                    "warning",
                    f"Resolution {w}x{h} exceeds recommended max {max_resolution}",
                    str(img_path),
                )
            )

        if require_square and w != h:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    f"Non-square image {w}x{h}",
                    str(img_path),
                )
            )

        seen_sizes.add(img_size)

        if check_value_range:
            if img_lo < 0 or img_hi > 255:
                report.issues.append(
                    ValidationIssue(
                        "warning",
                        f"Image pixel range [{img_lo:.0f}, {img_hi:.0f}] outside [0, 255]",
                        str(img_path),
                    )
                )
            if tgt_lo < 0 or tgt_hi > 255:
                report.issues.append(
                    ValidationIssue(
                        "warning",
                        f"Target pixel range [{tgt_lo:.0f}, {tgt_hi:.0f}] outside [0, 255]",
                        str(tgt_path),
                    )
                )

        task_issues, task_stats = _validate_target_semantics(
            tgt_path, task=task, depth_colormap=depth_colormap
        )
        report.issues.extend(task_issues)
        for name, value in task_stats.items():
            semantic_stats.setdefault(name, []).append(value)

    # Orphan targets without matching images
    for tgt_path in sorted(tgt_dir.iterdir()):
        if not tgt_path.is_file() or tgt_path.suffix.lower() not in IMG_EXTS:
            continue
        if tgt_path.stem not in {p.stem for p in image_files}:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    f"Target without matching image: {tgt_path.name}",
                    str(tgt_path),
                )
            )

    if len(seen_sizes) > 1:
        sizes_str = ", ".join(f"{w}x{h}" for w, h in sorted(seen_sizes))
        report.issues.append(
            ValidationIssue(
                "warning",
                f"Inconsistent resolutions across pairs: {sizes_str}",
            )
        )

    if report.pair_count < min_pairs:
        report.issues.append(
            ValidationIssue(
                "error",
                f"Only {report.pair_count} valid pair(s); need at least {min_pairs}",
            )
        )

    for name, values in semantic_stats.items():
        if values:
            report.stats[name] = {
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }

    return report


def _validate_target_semantics(
    target_path: Path,
    *,
    task: str | None,
    depth_colormap: str,
) -> tuple[list[ValidationIssue], dict[str, float]]:
    """Validate task-specific semantics for an RGB training target."""
    if task is None:
        return [], {}

    import numpy as np
    import torch

    issues: list[ValidationIssue] = []
    stats: dict[str, float] = {}
    structured_tasks = {"depth", "normal", "semantic", "segmentation", "instance", "referring"}
    if task in structured_tasks and target_path.suffix.lower() != ".png":
        issues.append(
            ValidationIssue(
                "error",
                f"{task} targets must use lossless PNG",
                str(target_path),
            )
        )

    try:
        with Image.open(target_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except OSError:
        return issues, stats
    rgb = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    if task == "depth":
        from ..core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper(colormap=depth_colormap)
        decoded = mapper.rgb_to_depth(rgb)
        reconstructed = mapper.depth_to_rgb(decoded)
        mae = float((reconstructed - rgb).abs().mean().item())
        stats["depth_rgb_roundtrip_mae"] = mae
        if float(decoded.max().item()) == 0.0:
            issues.append(
                ValidationIssue("error", "Depth target contains no positive depth", str(target_path))
            )
        if mae > 2.0 / 255.0:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Depth RGB is off the {depth_colormap} path (roundtrip MAE={mae:.6f})",
                    str(target_path),
                )
            )
    elif task == "normal":
        x = 1.0 - 2.0 * rgb[:, 0]
        y = 2.0 * rgb[:, 1] - 1.0
        z = 2.0 * rgb[:, 2] - 1.0
        raw_norm = torch.sqrt(x.square() + y.square() + z.square())
        valid = rgb.sum(dim=1) > 0
        if not valid.any():
            issues.append(
                ValidationIssue("error", "Normal target contains no valid vectors", str(target_path))
            )
        else:
            unit_error = (raw_norm[valid] - 1.0).abs()
            p95 = float(torch.quantile(unit_error, 0.95).item())
            stats["normal_unit_error_p95"] = p95
            if p95 > 0.05:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Decoded normals are not unit length (P95 error={p95:.6f})",
                        str(target_path),
                    )
                )
    elif task == "referring":
        channel_delta = (rgb - rgb[:, :1]).abs().amax(dim=1)
        binary_distance = torch.minimum(rgb[:, 0], 1.0 - rgb[:, 0])
        invalid_fraction = float(
            ((channel_delta > 1.0 / 255.0) | (binary_distance > 1.0 / 255.0))
            .float()
            .mean()
            .item()
        )
        stats["referring_nonbinary_fraction"] = invalid_fraction
        if invalid_fraction > 0.001:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Referring target is not a binary grayscale mask ({invalid_fraction:.2%} invalid)",
                    str(target_path),
                )
            )
    elif task in {"semantic", "segmentation", "instance"}:
        quantized = (rgb[0].permute(1, 2, 0) * 255.0).round().to(torch.uint8)
        unique_colors = int(torch.unique(quantized.reshape(-1, 3), dim=0).shape[0])
        stats["unique_target_colors"] = float(unique_colors)
        if unique_colors > 1024:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"Target has {unique_colors} colors; check anti-aliasing or lossy compression",
                    str(target_path),
                )
            )

    return issues, stats


_ATTENTION_LORA_MARKERS = (
    "to_q",
    "to_k",
    "to_v",
    "q_proj",
    "k_proj",
    "v_proj",
    "in_proj",
    "img_attn_qkv",
    "txt_attn_qkv",
)


def validate_manifest_contract(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    model_name: str = "sana-1600m",
    lora_targets: list[str] | None = None,
    depth_colormap: str = "cube_edge",
    depth_curve: str = "metric_power",
    max_target_samples_per_task: int = 16,
    split: str = "train",
    relative_depth: bool = False,
    relative_depth_samples_per_source: int = 8,
) -> ValidationReport:
    """Audit a JSONL pair manifest and model config for Vision-Banana contract bugs.

    Specifically catches the failure modes observed in the SNVC shared-LoRA runs:
    1. Model LoRA targets missing attention Q/K/V projections (e.g. SANA 42-layer
       FFN-only LoRA cannot learn open-vocabulary prompt-to-color cross-attention).
    2. Text encoder ``max_length`` (e.g. 77) truncating semantic RGB legends.
    3. ``depth_curve='metric_power'`` paired with relative-depth prompts or
       non-cube-edge / grayscale depth targets whose decoded metres explode.
    4. Multi-prompt tasks that prevent ``uniform_task_prompts`` from recording
       the training prompt into ``checkpoint/metadata.json``.
    5. With ``relative_depth`` (the ``depth_rel`` task), per depth source: the
       ``depth_rel`` encode/decode round trip, sentinel colour == invalid pixels
       exactly, and the invalid-pixel fraction (``stats["depth_rel_audit"]``).
    """
    import json
    import re

    import numpy as np
    import torch

    from ..core.depth_mapper import DepthRGBMapper
    from ..model_registry import resolve_model_config
    from ..task_prompts import SEMANTIC_SEGMENTATION_PROMPT
    from .manifest import resolve_manifest_path

    manifest = Path(manifest_path)
    report = ValidationReport(data_path=str(manifest.resolve()))
    if not manifest.is_file():
        report.issues.append(ValidationIssue("error", f"Manifest not found: {manifest}"))
        return report

    model_cfg = resolve_model_config(model_name)
    effective_targets = list(lora_targets) if lora_targets is not None else list(model_cfg.lora_target_modules)
    has_attn_lora = any(
        any(marker in target for marker in _ATTENTION_LORA_MARKERS)
        for target in effective_targets
    )
    if not has_attn_lora:
        report.issues.append(
            ValidationIssue(
                "error",
                f"LoRA target_modules {effective_targets} for '{model_name}' contain no "
                "attention Q/K/V projections; cross-attention cannot bind prompt color legends",
            )
        )

    max_tokens = int(getattr(model_cfg, "text_encoder_max_length", 77) or 77)
    anchor = Path(data_root).resolve() if data_root else manifest
    task_aliases = {"segmentation": "semantic", "seg": "semantic", "metric_depth": "depth"}

    task_counts: dict[str, int] = {}
    task_prompts: dict[str, set[str]] = {}
    task_targets: dict[str, list[Path]] = {}
    truncated_legend_rows = 0
    cityscapes_semantic_rows = 0
    ade150_semantic_rows = 0
    relative_depth_prompt_rows = 0

    relative_rows: dict[str, list[tuple[Path, Path | None]]] = {}
    legend_entry_re = re.compile(r'"[^"]+"\s*:\s*<\s*\d+\s*,\s*\d+\s*,\s*\d+\s*>')

    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "train") != split:
                continue
            raw_task = row.get("task", "")
            task = task_aliases.get(raw_task, raw_task)
            prompt = str(row.get("prompt") or "")
            task_counts[task] = task_counts.get(task, 0) + 1
            if prompt:
                task_prompts.setdefault(task, set()).add(prompt)

            tgt_rel = row.get("target", "")
            if tgt_rel:
                tgt_path = resolve_manifest_path(tgt_rel, anchor)
                bucket = task_targets.setdefault(task, [])
                if len(bucket) < max_target_samples_per_task and tgt_path.is_file():
                    bucket.append(tgt_path)

            if task == "semantic":
                if prompt == SEMANTIC_SEGMENTATION_PROMPT:
                    cityscapes_semantic_rows += 1
                if "150-class" in prompt or "fixed training palette" in prompt:
                    ade150_semantic_rows += 1
                entries = legend_entry_re.findall(prompt)
                # Each '"cls": <R, G, B>' entry takes ~13 subword tokens in CLIP/Gemma.
                estimated_tokens = 12 + 13 * len(entries) if entries else len(prompt.split()) * 2
                if estimated_tokens > max_tokens:
                    truncated_legend_rows += 1

            if relative_depth and task == "depth" and tgt_rel:
                source = str(row.get("source_dataset") or row.get("target_kind") or "unknown")
                bucket = relative_rows.setdefault(source, [])
                if len(bucket) < relative_depth_samples_per_source:
                    mask_rel = row.get("valid_mask")
                    bucket.append((
                        resolve_manifest_path(tgt_rel, anchor),
                        resolve_manifest_path(mask_rel, anchor) if mask_rel else None,
                    ))

            if task == "depth" and "relative depth" in prompt.lower():
                relative_depth_prompt_rows += 1

            report.pair_count += 1

    if truncated_legend_rows > 0:
        report.issues.append(
            ValidationIssue(
                "error",
                f"{truncated_legend_rows} semantic row(s) exceed {model_name} "
                f"text_encoder_max_length={max_tokens} and will have class RGB legends truncated",
                str(manifest),
            )
        )

    if ade150_semantic_rows > 0:
        report.issues.append(
            ValidationIssue(
                "error",
                f"{ade150_semantic_rows} semantic row(s) use the fixed ADE-150 palette prompt "
                "instead of an explicit class-to-RGB legend",
                str(manifest),
            )
        )

    if depth_curve == "metric_power" and relative_depth_prompt_rows > 0:
        report.issues.append(
            ValidationIssue(
                "error",
                f"{relative_depth_prompt_rows} depth row(s) use a 'relative depth' prompt while "
                "depth_curve='metric_power' expects metric depth supervision and prompt",
                str(manifest),
            )
        )

    if "depth" in task_prompts and len(task_prompts["depth"]) > 1:
        report.issues.append(
            ValidationIssue(
                "warning",
                f"Depth task has {len(task_prompts['depth'])} distinct prompts; "
                "uniform_task_prompts() will not record a depth prompt in metadata.json",
                str(manifest),
            )
        )

    # Audit sampled depth target pixels against metric_power + depth_colormap
    depth_paths = task_targets.get("depth", [])
    depth_stats: dict[str, float] = {}
    if depth_paths:
        mapper = DepthRGBMapper(colormap=depth_colormap, curve=depth_curve)
        maes: list[float] = []
        gray_fracs: list[float] = []
        medians: list[float] = []
        p95s: list[float] = []
        for path in depth_paths:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            decoded = mapper.rgb_to_depth(rgb)
            recon = mapper.depth_to_rgb(decoded)
            maes.append(float((recon - rgb).abs().mean().item()))
            channel_span = (rgb[:, 0] - rgb[:, 1]).abs() + (rgb[:, 1] - rgb[:, 2]).abs()
            gray_fracs.append(float((channel_span < 1.0 / 255.0).float().mean().item()))
            medians.append(float(decoded.median().item()))
            p95s.append(float(torch.quantile(decoded.reshape(-1), 0.95).item()))

        mean_mae = sum(maes) / len(maes)
        mean_gray = sum(gray_fracs) / len(gray_fracs)
        mean_median = sum(medians) / len(medians)
        mean_p95 = sum(p95s) / len(p95s)
        depth_stats = {
            "sampled_targets": len(depth_paths),
            "mean_roundtrip_mae": mean_mae,
            "mean_grayscale_fraction": mean_gray,
            "mean_decoded_median_m": mean_median,
            "mean_decoded_p95_m": mean_p95,
        }
        if depth_colormap != "grayscale" and mean_gray > 0.90:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"Depth targets are grayscale ({mean_gray:.1%} achromatic pixels) "
                    f"but depth_colormap={depth_colormap!r}; metric_power decoding will invert/explode",
                    str(depth_paths[0]),
                )
            )
        elif mean_mae > 2.0 / 255.0:
            report.issues.append(
                ValidationIssue(
                    "error",
                    f"Depth targets do not lie on the {depth_colormap!r} colormap curve "
                    f"(mean roundtrip MAE={mean_mae:.4f} > {2.0 / 255.0:.4f})",
                    str(depth_paths[0]),
                )
            )

    relative_stats = _audit_relative_depth(relative_rows, report) if relative_depth else {}

    semantic_total = task_counts.get("semantic", 0)
    report.stats = {
        "model_name": model_name,
        "text_encoder_max_length": max_tokens,
        "lora_targets": effective_targets,
        "has_attention_lora": has_attn_lora,
        "task_counts": task_counts,
        "cityscapes_semantic_rows": cityscapes_semantic_rows,
        "cityscapes_semantic_share": (
            cityscapes_semantic_rows / semantic_total if semantic_total > 0 else 0.0
        ),
        "truncated_legend_rows": truncated_legend_rows,
        "relative_depth_prompt_rows": relative_depth_prompt_rows,
        "depth_target_audit": depth_stats,
        **({"depth_rel_audit": relative_stats} if relative_depth else {}),
    }
    return report


def _audit_relative_depth(
    rows: dict[str, list[tuple[Path, Path | None]]], report: ValidationReport
) -> dict[str, dict]:
    """Encode sampled depth rows as depth_rel targets and check the stored contract."""
    import numpy as np
    import torch

    from ..core.relative_depth_mapper import RelativeDepthMapper, metric_target_inverse_depth

    mapper = RelativeDepthMapper()
    stats: dict[str, dict] = {}
    for source, items in sorted(rows.items()):
        invalid, roundtrip, collisions, mismatches, checked = [], [], 0, 0, 0
        for target_path, mask_path in items:
            if not target_path.is_file() or (mask_path is not None and not mask_path.is_file()):
                report.issues.append(ValidationIssue("error", f"depth_rel audit: missing file for {target_path}"))
                continue
            with Image.open(target_path) as img:
                rgb = np.asarray(img.convert("RGB"))
            mask = None
            if mask_path is not None:
                with Image.open(mask_path) as img:
                    mask = np.asarray(img)
                mask = mask[..., 0] if mask.ndim == 3 else mask
                if mask.shape != rgb.shape[:2]:
                    report.issues.append(ValidationIssue("error", f"depth_rel audit: mask/target size differ for {target_path}"))
                    continue
            inverse, valid = metric_target_inverse_depth(rgb, mask)
            valid_t = torch.from_numpy(valid)
            normalized = mapper.normalize_inverse_depth(torch.from_numpy(inverse), valid_t)
            encoded = (mapper.encode(normalized, valid_t) * 255.0).round().to(torch.uint8)
            sentinel = mapper.sentinel_mask(encoded)
            mismatches += int((sentinel != ~valid_t).sum())
            collisions += int((sentinel & valid_t).sum())
            if valid.any():
                decoded = mapper.decode(encoded.float() / 255.0)
                roundtrip.append(float((decoded - normalized)[valid_t].abs().max()))
            invalid.append(float(1.0 - valid.mean()))
            checked += 1
        entry = {
            "sampled_rows": checked,
            "invalid_fraction_mean": float(np.mean(invalid)) if invalid else None,
            "invalid_fraction_max": float(np.max(invalid)) if invalid else None,
            "roundtrip_max_abs": float(np.max(roundtrip)) if roundtrip else None,
            "sentinel_mismatch_pixels": mismatches,
            "sentinel_collision_pixels": collisions,
        }
        stats[source] = entry
        if mismatches or collisions:
            report.issues.append(ValidationIssue(
                "error", f"depth_rel audit ({source}): sentinel pixels differ from invalid pixels "
                f"({mismatches} mismatched, {collisions} on valid pixels)"))
        if roundtrip and max(roundtrip) > 0.5 / 255.0 + 1e-6:
            report.issues.append(ValidationIssue(
                "error", f"depth_rel audit ({source}): round trip error {max(roundtrip):.5f} exceeds 0.5/255"))
    return stats


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_manifest_contract",
    "validate_vision_dataset",
]
