"""Pre-registered pass/fail gates for depth and surface-normal experiments.

A depth number is only evidence of learned geometry relative to what trivial
predictors achieve on the same protocol (``trivial_baselines``). The thresholds
here are fixed *before* running an experiment so results cannot be rationalized
afterwards.

Two gates:

* **metric gate**: delta_1.25 must beat the best *non-oracle* trivial baseline
  (fixed constants, training-set mean depth map) by ``margin``, and AbsRel must
  not be worse than that baseline's.
* **structure gate**: after per-image median-scale alignment the model's
  delta_1.25 must beat ``oracle_image_median`` (perfect per-image scale, zero
  spatial structure). Passing means the prediction carries spatial structure
  beyond a constant, i.e. it is not just a correct-scale blob.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

# Baselines that use the evaluation GT and therefore cannot serve as the bar a
# legitimate zero-shot method must clear.
ORACLE_BASELINES = frozenset({"oracle_constant", "oracle_image_median"})


@dataclass(frozen=True)
class DepthGateResult:
    passed: bool
    metric_gate: bool
    structure_gate: bool
    model_delta1: float
    model_absrel: float
    reference_baseline: str
    reference_delta1: float
    reference_absrel: float
    required_delta1: float
    model_aligned_delta1: float
    structure_reference_delta1: float
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_depth_gate(
    model_metrics: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, float]],
    *,
    margin: float = 0.10,
) -> DepthGateResult:
    """Apply the metric and structure gates to ``model_metrics``.

    ``model_metrics`` is a ``decoded_metrics`` dict from
    ``evaluate_paper_benchmarks.py`` (needs ``delta_1.25``, ``AbsRel`` and
    ``delta_1.25_aligned``); ``baselines`` is the ``baselines`` dict written by
    ``scripts/depth_trivial_baselines.py``.
    """
    candidates = {k: v for k, v in baselines.items() if k not in ORACLE_BASELINES}
    if not candidates:
        raise ValueError("No non-oracle baselines available to define the metric gate")
    if "oracle_image_median" not in baselines:
        raise ValueError("baselines must include 'oracle_image_median' for the structure gate")
    for key in ("delta_1.25", "AbsRel", "delta_1.25_aligned"):
        if key not in model_metrics:
            raise KeyError(f"model metrics missing {key!r}")

    ref_name = max(candidates, key=lambda name: candidates[name]["delta_1.25"])
    ref = candidates[ref_name]
    best_absrel = min(v["AbsRel"] for v in candidates.values())
    required = ref["delta_1.25"] + margin
    metric_gate = model_metrics["delta_1.25"] >= required and model_metrics["AbsRel"] <= best_absrel
    structure_ref = baselines["oracle_image_median"]["delta_1.25"]
    structure_gate = model_metrics["delta_1.25_aligned"] > structure_ref
    return DepthGateResult(
        passed=bool(metric_gate and structure_gate),
        metric_gate=bool(metric_gate),
        structure_gate=bool(structure_gate),
        model_delta1=float(model_metrics["delta_1.25"]),
        model_absrel=float(model_metrics["AbsRel"]),
        reference_baseline=ref_name,
        reference_delta1=float(ref["delta_1.25"]),
        reference_absrel=float(best_absrel),
        required_delta1=float(required),
        model_aligned_delta1=float(model_metrics["delta_1.25_aligned"]),
        structure_reference_delta1=float(structure_ref),
        margin=float(margin),
    )


# Normal oracles use the evaluation GT. A single global direction must not be
# allowed to set the bar, and the per-image mean direction is the structure
# ceiling for "one direction, no spatial field".
ORACLE_NORMAL_BASELINES = frozenset({"oracle_image_mean", "oracle_dataset_mean"})
NORMAL_ERROR_KEY = "Mean Angular Error (°)"


@dataclass(frozen=True)
class NormalGateResult:
    passed: bool
    metric_gate: bool
    structure_gate: bool
    model_mean_deg: float
    reference_baseline: str
    reference_mean_deg: float
    required_mean_deg: float
    structure_reference_deg: float
    margin_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_normal_gate(
    model_metrics: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, float]],
    *,
    margin_deg: float = 5.0,
) -> NormalGateResult:
    """Apply the normal metric and structure gates.

    Lower angular error is better.

    * **metric gate**: mean angular error must be at least ``margin_deg``
      below the best non-oracle trivial baseline (camera-facing, up-axis,
      training-set mean direction).
    * **structure gate**: mean angular error must be strictly below
      ``oracle_image_mean``, the best single direction per image. Passing
      means the prediction carries a spatial normal field.
    """
    candidates = {k: v for k, v in baselines.items() if k not in ORACLE_NORMAL_BASELINES}
    if not candidates:
        raise ValueError("No non-oracle normal baselines available to define the metric gate")
    if "oracle_image_mean" not in baselines:
        raise ValueError("baselines must include 'oracle_image_mean' for the structure gate")
    if NORMAL_ERROR_KEY not in model_metrics:
        raise KeyError(f"model metrics missing {NORMAL_ERROR_KEY!r}")

    ref_name = min(candidates, key=lambda name: candidates[name][NORMAL_ERROR_KEY])
    ref_error = float(candidates[ref_name][NORMAL_ERROR_KEY])
    required = ref_error - margin_deg
    model_error = float(model_metrics[NORMAL_ERROR_KEY])
    structure_ref = float(baselines["oracle_image_mean"][NORMAL_ERROR_KEY])
    metric_gate = model_error <= required
    structure_gate = model_error < structure_ref
    return NormalGateResult(
        passed=bool(metric_gate and structure_gate),
        metric_gate=bool(metric_gate),
        structure_gate=bool(structure_gate),
        model_mean_deg=model_error,
        reference_baseline=ref_name,
        reference_mean_deg=ref_error,
        required_mean_deg=float(required),
        structure_reference_deg=structure_ref,
        margin_deg=float(margin_deg),
    )


__all__ = [
    "DepthGateResult",
    "NormalGateResult",
    "ORACLE_BASELINES",
    "ORACLE_NORMAL_BASELINES",
    "evaluate_depth_gate",
    "evaluate_normal_gate",
]
