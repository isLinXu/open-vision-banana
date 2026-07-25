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


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_vision_dataset",
]
