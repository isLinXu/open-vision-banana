#!/usr/bin/env python3
"""Machine-check the gates of ``exp_setting.md`` before / after a run.

Every gate in the E0 / E1 tables is encoded here so an operator (or an agent)
cannot "read the audit JSON" wrong. Each check prints PASS / FAIL / SKIP with the
observed value; the exit code is 1 if any required gate fails.

    # after data preparation (E0.1 / E1.1)
    python scripts/check_experiment_gates.py data \\
        --merged-audit data/merged/pairs.audit.json \\
        --final-manifest data/qwen_baseline/pairs.jsonl \\
        --hypersim-manifest data/prep/hypersim/manifest.json \\
        --tasks depth,normal,semantic,instance,referring,generation \\
        --model-name qwen-image-2.1

    # after training (E0.2 / E1.2)
    python scripts/check_experiment_gates.py run --run-dir output/<run> \\
        --tasks depth,normal,semantic,instance,referring,generation --vision-ratio 0.5

    # after evaluation (E0.3 / E1.3)
    python scripts/check_experiment_gates.py eval --eval-json output/<run>/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class GateResult:
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    def add(self, name: str, ok: bool | None, detail: str = "", *, warn_only: bool = False) -> None:
        if ok is None:
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "WARN" if warn_only else "FAIL"
        self.results.append(GateResult(name, status, detail))

    @property
    def failed(self) -> list[GateResult]:
        return [r for r in self.results if r.status == "FAIL"]

    def render(self) -> str:
        width = max((len(r.name) for r in self.results), default=10)
        lines = [f"{r.status:4s} {r.name.ljust(width)}  {r.detail}" for r in self.results]
        warned = sum(1 for r in self.results if r.status == "WARN")
        lines.append(f"---- {len(self.results) - len(self.failed) - warned}/{len(self.results)} gates passed, {warned} warning(s), {len(self.failed)} failed")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"gates": [r.__dict__ for r in self.results], "failed": len(self.failed)}


def _load(path: Path | None) -> dict | None:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------- data gates
def check_data(
    *,
    merged_audit: Path | None,
    final_manifest: Path | None,
    hypersim_manifest: Path | None,
    tasks: list[str],
    model_name: str,
    require_strict: bool = True,
) -> GateReport:
    report = GateReport()
    merged = _load(merged_audit)
    if merged is None:
        report.add("merge.audit_present", None, f"{merged_audit} not found")
    else:
        dropped = merged.get("per_task_dropped", {})
        report.add("merge.per_task_dropped_empty", dropped == {}, json.dumps(dropped))
        split = merged.get("per_task_split", {})
        for task in tasks:
            key = {"depth": ("depth", "metric_depth")}.get(task, (task,))
            counts = {}
            for k in key:
                for s, n in split.get(k, {}).items():
                    counts[s] = counts.get(s, 0) + n
            ok = counts.get("train", 0) >= 1 and counts.get("val", 0) >= 1
            report.add(f"merge.{task}.train_and_val", ok, json.dumps(counts))

    final_audit = _load(final_manifest.with_suffix(".audit.json")) if final_manifest else None
    if final_audit is None:
        report.add("final.audit_present", None, "final audit json not found")
    else:
        if require_strict:
            report.add("final.protocol_strict", final_audit.get("protocol") == "strict", str(final_audit.get("protocol")))
            report.add(
                "final.zero_shot_claim_allowed",
                bool(final_audit.get("zero_shot_claim_allowed")),
                str(final_audit.get("zero_shot_claim_allowed")),
            )
        report.add(
            "final.no_non_metric_depth",
            final_audit.get("rejected_non_metric_depth_rows", 0) == 0,
            f"rejected_non_metric_depth_rows={final_audit.get('rejected_non_metric_depth_rows')}",
        )
        overlap = final_audit.get("evaluation_image_id_overlap_by_source", {})
        report.add("final.no_eval_image_id_overlap", all(v == 0 for v in overlap.values()), json.dumps(overlap))
        # Images shared between this manifest's own train and val slices (across tasks, e.g. a
        # RefCOCO official val image that is also a COCO instance training image). Not a benchmark
        # leak - the paper only forbids *evaluated* data - but it makes val loss optimistic, so WARN.
        report.add(
            "final.train_val_image_overlap",
            final_audit.get("train_val_coco_image_id_overlap", 0) == 0,
            f"{final_audit.get('train_val_coco_image_id_overlap')} COCO image(s) in both internal splits",
            warn_only=True,
        )
        counts = final_audit.get("task_split_counts", {})
        for task in tasks:
            ok = counts.get(f"{task}/train", 0) >= 1 and counts.get(f"{task}/val", 0) >= 1
            report.add(f"final.{task}.train_and_val", ok, f"train={counts.get(f'{task}/train', 0)} val={counts.get(f'{task}/val', 0)}")
        report.add("final.manifest_sha256", len(str(final_audit.get("manifest_sha256", ""))) == 64, str(final_audit.get("manifest_sha256", ""))[:12])

    hyp = _load(hypersim_manifest)
    if hyp is None:
        report.add("hypersim.manifest_present", None, "not provided / not found")
    else:
        report.add(
            "hypersim.planar_z_depth",
            hyp.get("metric_depth", {}).get("convention") == "planar_z_depth",
            str(hyp.get("metric_depth", {}).get("convention")),
        )
        backfacing = hyp.get("tasks", {}).get("normal_backfacing_frames")
        report.add("hypersim.no_backfacing_normals", backfacing in (0, None), f"normal_backfacing_frames={backfacing}")
        err = hyp.get("data_contract", {}).get("encoded_roundtrip_abs_error", {}).get("mean")
        report.add("hypersim.depth_roundtrip_mean_lt_1cm", err is not None and err < 0.01, f"mean={err}")

    if final_manifest and Path(final_manifest).is_file():
        from open_vision_banana.data.validation import validate_manifest_contract

        contract = validate_manifest_contract(final_manifest, data_root=Path(final_manifest).parent, model_name=model_name)
        report.add(
            "contract.no_errors",
            not contract.errors,
            "; ".join(i.message for i in contract.errors)[:200] or f"{len(contract.warnings)} warning(s)",
        )
    else:
        report.add("contract.no_errors", None, "final manifest not found")
    return report


# ----------------------------------------------------------------------------- run gates
def check_run(*, run_dir: Path, tasks: list[str], vision_ratio: float | None) -> GateReport:
    report = GateReport()
    effective = _load(run_dir / "effective_config.json")
    if effective is None:
        report.add("run.effective_config_present", False, str(run_dir / "effective_config.json"))
        return report
    report.add("run.tasks_match", sorted(effective.get("tasks", [])) == sorted(tasks), str(effective.get("tasks")))
    if vision_ratio is not None:
        report.add("run.vision_ratio", abs(float(effective.get("vision_ratio", -1)) - vision_ratio) < 1e-9, str(effective.get("vision_ratio")))
        probs = effective.get("sampling_probabilities", {})
        if "generation" in tasks and probs:
            gen = probs.get("generation", 0.0)
            others = [probs[t] for t in tasks if t != "generation" and t in probs]
            ok = abs(gen - (1 - vision_ratio)) < 1e-6 and all(abs(p - vision_ratio / max(len(others), 1)) < 1e-6 for p in others)
            report.add("run.sampling_probabilities", ok, json.dumps({k: round(v, 4) for k, v in probs.items()}))
    preflight = effective.get("preflight", {})
    report.add("run.preflight_not_skipped", not preflight.get("skipped", False), f"skipped={preflight.get('skipped')} errors={len(preflight.get('errors', []))}")
    report.add("run.preflight_no_errors", not preflight.get("errors"), "; ".join(preflight.get("errors", []))[:200])
    report.add("run.optimizer_recorded", all(k in effective for k in ("weight_decay", "adam_beta1", "adam_beta2", "adam_eps")), "")
    checkpoints = sorted(p for p in run_dir.glob("checkpoint-*") if p.is_dir())
    report.add("run.checkpoints_present", bool(checkpoints), ", ".join(p.name for p in checkpoints)[:120])
    report.add("run.best_checkpoint", any(p.name.endswith("_best") for p in checkpoints), "")
    report.add("run.final_checkpoint", any(p.name.endswith("_final") for p in checkpoints), "")
    finite_losses = True
    for ckpt in checkpoints:
        meta = _load(ckpt / "metadata.json")
        if meta and meta.get("val_loss") is not None and not _finite(meta["val_loss"]):
            finite_losses = False
    report.add("run.finite_val_losses", finite_losses, "")
    return report


# ----------------------------------------------------------------------------- eval gates
def check_eval(*, eval_json: Path, required_metrics: list[str] | None = None) -> GateReport:
    report = GateReport()
    results = _load(eval_json)
    if results is None:
        report.add("eval.json_present", False, str(eval_json))
        return report
    numeric = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            numeric[prefix] = node

    walk("", results)
    report.add("eval.has_numeric_metrics", bool(numeric), f"{len(numeric)} numeric entries")
    report.add("eval.all_finite", all(_finite(v) for v in numeric.values()), "")
    for metric in required_metrics or []:
        hits = {k: v for k, v in numeric.items() if k.endswith(metric)}
        report.add(f"eval.{metric}", bool(hits) and all(_finite(v) for v in hits.values()), json.dumps({k: round(v, 4) for k, v in list(hits.items())[:3]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("data")
    d.add_argument("--merged-audit", type=Path, default=None)
    d.add_argument("--final-manifest", type=Path, required=True)
    d.add_argument("--hypersim-manifest", type=Path, default=None)
    d.add_argument("--tasks", default="depth,normal,semantic,instance,referring,generation")
    d.add_argument("--model-name", default="qwen-image-2.1")
    d.add_argument("--allow-in-domain", action="store_true", help="do not require protocol=strict")
    r = sub.add_parser("run")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--tasks", default="depth,normal,semantic,instance,referring,generation")
    r.add_argument("--vision-ratio", type=float, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--eval-json", type=Path, required=True)
    e.add_argument("--require", default="", help="comma-separated metric suffixes that must be present and finite, e.g. AbsRel,cIoU")
    for p in (d, r, e):
        p.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.stage == "data":
        report = check_data(
            merged_audit=args.merged_audit, final_manifest=args.final_manifest, hypersim_manifest=args.hypersim_manifest,
            tasks=[t for t in args.tasks.split(",") if t], model_name=args.model_name, require_strict=not args.allow_in_domain,
        )
    elif args.stage == "run":
        report = check_run(run_dir=args.run_dir, tasks=[t for t in args.tasks.split(",") if t], vision_ratio=args.vision_ratio)
    else:
        report = check_eval(eval_json=args.eval_json, required_metrics=[m for m in args.require.split(",") if m])
    print(report.render())
    if args.json_out:
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
