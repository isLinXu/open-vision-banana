"""Shared contracts for OPD / GRPO post-training.

SFT remains the default ``objective``. OPD and GRPO are opt-in and refuse to
start unless the config (and, for GRPO, the SFT gate) says they should.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SFT_GATE_PROTOCOL = "post_sft_gate_v1"
COLORMAP_OK_MIN_SHARE = 0.5
VISION_GT_TASKS = frozenset({"depth", "normal", "semantic", "instance", "referring"})
FORBIDDEN_GRPO_REWARDS = frozenset(
    {"aesthetic", "alignment", "portrait", "face_id", "hpsv2", "pickscore", "image_reward"}
)


def parse_teacher_map(value: Any, name: str = "opd_teachers") -> dict[str, str]:
    """Parse ``task=path`` strings or YAML mappings into a path map."""
    if isinstance(value, str):
        parsed: dict[str, str] = {}
        if not value.strip():
            return parsed
        for pair in value.split(","):
            if pair.count("=") != 1:
                raise ValueError(f"{name}: expected task=path pairs")
            key, path = (part.strip() for part in pair.split("="))
            if not key or key in parsed:
                raise ValueError(f"{name}: duplicate or empty task {key!r}")
            parsed[key] = path
        value = parsed
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a task-to-path mapping")
    result: dict[str, str] = {}
    for key, path in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name}: invalid task {key!r}")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{name}: teacher path for {key} must be a non-empty string")
        result[key.strip()] = path.strip()
    return result


def _nyu_record(eval_results: dict[str, Any]) -> dict[str, Any]:
    benches = eval_results.get("benchmarks")
    if not isinstance(benches, dict):
        benches = {}
    direct = benches.get("depth/nyu")
    if isinstance(direct, dict):
        return direct
    for key, rec in benches.items():
        if not isinstance(rec, dict):
            continue
        if key == "nyu" or str(key).endswith("/nyu") or rec.get("dataset") == "nyu":
            return rec
    raise ValueError("eval JSON has no depth/nyu benchmark; score NYU before writing the SFT gate")


def colormap_ok_from_hist(
    hist: dict[str, Any] | None,
    trained: str,
    *,
    min_share: float = COLORMAP_OK_MIN_SHARE,
) -> tuple[bool, str]:
    """Palette binding: trained colormap must be the oracle majority with enough share."""
    if not hist:
        return False, "oracle_colormap_hist missing; rerun eval with --depth-oracle-colormaps"
    counts: dict[str, float] = {}
    for name, count in hist.items():
        try:
            counts[str(name)] = float(count)
        except (TypeError, ValueError):
            continue
    total = sum(counts.values())
    if total <= 0:
        return False, "empty oracle colormap histogram"
    share = counts.get(trained, 0.0) / total
    majority = max(counts, key=counts.get)
    detail = f"trained={trained} majority={majority} share={share:.2f}"
    if majority != trained or share < min_share:
        return False, detail
    return True, detail


def sft_gate_from_eval(
    eval_results: dict[str, Any],
    *,
    trained_colormap: str = "cube_edge",
    min_share: float = COLORMAP_OK_MIN_SHARE,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the GRPO SFT-gate object from ``evaluate_paper_benchmarks`` NYU metrics."""
    record = _nyu_record(eval_results)
    metrics = record.get("decoded_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    hist = metrics.get("oracle_colormap_hist")
    if not isinstance(hist, dict):
        hist = {}
    ok, detail = colormap_ok_from_hist(hist, trained_colormap, min_share=min_share)
    metadata = eval_results.get("metadata") if isinstance(eval_results.get("metadata"), dict) else {}
    return {
        "protocol": SFT_GATE_PROTOCOL,
        "colormap_ok": ok,
        "colormap_detail": detail,
        "trained_colormap": trained_colormap,
        "raw_delta_1.25": metrics.get("delta_1.25"),
        "aligned_delta_1.25": metrics.get("delta_1.25_aligned"),
        "AbsRel": metrics.get("AbsRel"),
        "oracle_colormap_hist": hist,
        "source": source,
        "model_name": metadata.get("model_name"),
        "lora_path": metadata.get("lora_path"),
    }


def write_sft_gate(path: str | Path, gate: dict[str, Any]) -> Path:
    """Write ``post_sft_gate.json`` (creates parents)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(gate)
    payload.setdefault("protocol", SFT_GATE_PROTOCOL)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_sft_gate(path: str | Path) -> dict[str, Any]:
    """Load the JSON written after an SFT eval that decides whether GRPO may run."""
    gate_path = Path(path)
    if not gate_path.is_file():
        raise FileNotFoundError(f"SFT gate not found: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise ValueError("SFT gate must be a JSON object")
    return gate


def grpo_sft_gate_status(gate: dict[str, Any]) -> tuple[bool, str]:
    """Allow depth GRPO only when the palette is right and metric scale still drifts.

    ``colormap_ok`` means a probe set decodes with the trained palette (not a
    random other colormap). Scale drift is ``raw δ1.25`` worse than ``aligned``:
    the model has structure but not metric scale — the failure mode SFT latent
    MSE does not correct.
    """
    protocol = gate.get("protocol")
    if protocol not in {None, SFT_GATE_PROTOCOL}:
        return False, f"unknown SFT gate protocol {protocol!r}"
    if not gate.get("colormap_ok"):
        return False, "colormap_ok is not true; fix SFT palette binding before GRPO"
    try:
        raw = float(gate["raw_delta_1.25"])
        aligned = float(gate["aligned_delta_1.25"])
    except (KeyError, TypeError, ValueError):
        return False, "SFT gate must include finite raw_delta_1.25 and aligned_delta_1.25"
    if not (raw == raw and aligned == aligned):  # NaN check without math import at call sites
        return False, "SFT gate δ1.25 values must be finite"
    if raw >= aligned:
        return False, (
            f"raw δ1.25 {raw:.3f} is not below aligned {aligned:.3f}; "
            "scale-drift GRPO is not indicated"
        )
    return True, f"raw_delta_1.25={raw:.3f} aligned_delta_1.25={aligned:.3f}"
