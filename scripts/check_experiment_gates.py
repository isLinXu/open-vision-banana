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
    python scripts/check_experiment_gates.py eval --eval-json output/<run>/eval.json --profile e1

    # after NYU eval: write the GRPO SFT gate (or use --write-sft-gate on evaluate_paper_benchmarks)
    python scripts/write_sft_gate.py output/<run>/eval.json \\
        --output output/sft_gate/post_sft_gate.json

    # before OPD / GRPO (refuses aesthetic RL and GRPO without an SFT gate)
    python scripts/check_experiment_gates.py post \\
        --config configs/experiments/qwen_image_21_grpo_depth.yaml \\
        --sft-gate output/sft_gate/post_sft_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

E1_E2_BENCHMARKS: dict[str, tuple[str, ...]] = {
    "depth/nyu": ("AbsRel", "delta_1.25"),
    "depth/diode": ("AbsRel", "delta_1.25"),
    "depth/eth3d": ("AbsRel", "delta_1.25"),
    "normal/diode": ("mean_angle_deg",),
    "semantic/cityscapes": ("mIoU",),
    "referring/refcocog": ("cIoU",),
    "referring/reasonseg": ("gIoU",),
    "instance/saco": ("cgF1",),
}


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
        passed = sum(1 for r in self.results if r.status == "PASS")
        warned = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        lines.append(f"---- {passed}/{len(self.results)} gates passed, {warned} warning(s), {skipped} skipped, {len(self.failed)} failed")
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
        report.add("merge.audit_present", False, f"{merged_audit} not found")
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
        report.add("final.audit_present", False, "final audit json not found")
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
        expected_hash = final_audit.get("manifest_sha256")
        actual_hash = hashlib.sha256(final_manifest.read_bytes()).hexdigest() if final_manifest and final_manifest.is_file() else None
        report.add(
            "final.manifest_sha256",
            actual_hash is not None and isinstance(expected_hash, str) and expected_hash == actual_hash,
            f"audit={str(expected_hash)[:12]} actual={str(actual_hash)[:12]}",
        )

    hyp = _load(hypersim_manifest)
    if hyp is None:
        report.add("hypersim.manifest_present", False, "not provided / not found")
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
        report.add("contract.no_errors", False, "final manifest not found")
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
    objective = effective.get("objective", "sft")
    report.add("run.objective", objective in {"sft", "opd", "grpo"}, str(objective))
    tune_mode = effective.get("tune_mode", "lora")
    report.add("run.tune_mode", tune_mode in {"lora", "full"}, str(tune_mode))
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
def check_eval(
    *,
    eval_json: Path,
    required_metrics: list[str] | None = None,
    profile: str | None = None,
) -> GateReport:
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
    if profile in {"e1", "e2"}:
        benchmarks = results.get("benchmarks", {})
        for key, metrics in E1_E2_BENCHMARKS.items():
            record = benchmarks.get(key, {}) if isinstance(benchmarks, dict) else {}
            decoded = record.get("decoded_metrics") if isinstance(record, dict) else None
            report.add(
                f"eval.{key}.coverage",
                isinstance(decoded, dict) and _finite(record.get("samples")) and float(record["samples"]) > 0
                and record.get("samples") == record.get("source_samples")
                and record.get("split") == "val",
                f"samples={record.get('samples')} source_samples={record.get('source_samples')} split={record.get('split')}"
                if isinstance(record, dict) else "invalid record",
            )
            for metric in metrics:
                value = decoded.get(metric) if isinstance(decoded, dict) else None
                report.add(f"eval.{key}.{metric}", _finite(value), str(value))
    return report


def check_generation(*, eval_json: Path, reference_dir: Path) -> GateReport:
    report = GateReport()
    spec = _load(reference_dir / "reference_config.json")
    results = _load(eval_json)
    if not isinstance(spec, dict):
        report.add("generation.reference_present", False, str(reference_dir / "reference_config.json"))
        return report
    if not isinstance(results, dict):
        report.add("generation.eval_present", False, str(eval_json))
        return report
    manifest = reference_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()] if manifest.is_file() else []
    prompts = [row.get("prompt") for row in rows if isinstance(row, dict)]
    prompt_hash = hashlib.sha256(json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    sample_count = spec.get("samples")
    report.add(
        "generation.reference_complete",
        isinstance(sample_count, int) and sample_count > 0 and len(rows) == sample_count
        and all(isinstance(row, dict) and isinstance(row.get("prompt"), str)
                and isinstance(row.get("image"), str) and (reference_dir / "images" / row["image"]).is_file()
                for row in rows)
        and spec.get("prompts_sha256") == prompt_hash,
        f"manifest={len(rows)} expected={sample_count}",
    )
    metadata = results.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    report.add(
        "generation.reference_match",
        isinstance(metadata, dict) and Path(str(metadata.get("dataset_root", ""))).resolve() == reference_dir.resolve()
        and metadata.get("model_name") == spec.get("model_name")
        and metadata.get("seed") == spec.get("seed")
        and metadata.get("num_inference_steps") == spec.get("num_inference_steps")
        and metadata.get("guidance_scale") == spec.get("guidance_scale")
        and metadata.get("dtype") == spec.get("dtype"),
        str(metadata.get("dataset_root")) if isinstance(metadata, dict) else "invalid metadata",
    )
    report.add(
        "generation.all_samples_scored",
        metadata.get("requested_samples") == sample_count
        and metadata.get("successful_samples") == sample_count
        and metadata.get("failed_samples") == 0,
        f"requested={metadata.get('requested_samples')} successful={metadata.get('successful_samples')} failed={metadata.get('failed_samples')}",
    )
    metrics = results.get("metrics", {})
    for key in ("FID_proxy", "diversity_score", "perceptual_similarity"):
        value = metrics.get(key) if isinstance(metrics, dict) else None
        report.add(f"generation.{key}", _finite(value), str(value))
    return report


def check_post_training(*, config_path: Path, sft_gate: Path | None = None) -> GateReport:
    """Refuse OPD/GRPO launches that skip teachers, mix generation into GRPO, or lack the SFT gate."""
    from open_vision_banana.training.config import load_config
    from open_vision_banana.training.post_training import grpo_sft_gate_status, load_sft_gate

    report = GateReport()
    try:
        config = load_config(config_path=config_path)
    except Exception as exc:
        report.add("post.config_loads", False, str(exc))
        return report
    report.add("post.config_loads", True, f"objective={config.objective} tune_mode={config.tune_mode}")
    if config.objective == "sft":
        report.add("post.sft_default", True, "no post-training requested")
        return report
    if config.objective == "opd":
        report.add("post.teachers_present", bool(config.opd_teachers), json.dumps(config.opd_teachers))
        covered = set(config.opd_teachers)
        missing = [task for task in config.tasks if task not in covered and "*" not in covered]
        report.add("post.teachers_cover_tasks", not missing, f"missing={missing}")
        for task, path in config.opd_teachers.items():
            candidate = Path(path)
            exists = path == "self" or candidate.is_file() or (candidate / "lora.pt").is_file()
            report.add(f"post.teacher.{task}.exists", exists, path)
        report.add(
            "post.rollout_steps",
            config.opd_rollout_steps >= 2 and config.opd_train_timesteps <= config.opd_rollout_steps,
            f"steps={config.opd_rollout_steps} train={config.opd_train_timesteps}",
        )
        return report
    report.add("post.reward_decoded_metric", config.grpo_reward == "decoded_metric", config.grpo_reward)
    report.add("post.no_generation_task", "generation" not in config.tasks, str(config.tasks))
    report.add("post.group_size", config.grpo_group_size >= 2, str(config.grpo_group_size))
    report.add("post.sde_sigma", config.grpo_sde_sigma > 0, str(config.grpo_sde_sigma))
    gate_path = sft_gate or (Path(config.sft_gate_path) if config.sft_gate_path else None)
    if config.allow_grpo_without_sft_gate and gate_path is None:
        report.add("post.sft_gate", False, "bypassed via allow_without_sft_gate", warn_only=True)
        return report
    if gate_path is None:
        report.add("post.sft_gate", False, "sft_gate path not set")
        return report
    try:
        ok, detail = grpo_sft_gate_status(load_sft_gate(gate_path))
        report.add("post.sft_gate", ok, detail)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.add("post.sft_gate", False, str(exc))
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
    e.add_argument("--profile", choices=("e1", "e2"), default=None,
                   help="require all E1/E2 paper benchmark datasets, val splits, sample counts, and headline metrics")
    g = sub.add_parser("generation")
    g.add_argument("--eval-json", type=Path, required=True)
    g.add_argument("--reference-dir", type=Path, required=True)
    p_post = sub.add_parser("post")
    p_post.add_argument("--config", type=Path, required=True)
    p_post.add_argument("--sft-gate", type=Path, default=None)
    for p in (d, r, e, g, p_post):
        p.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.stage == "data":
        report = check_data(
            merged_audit=args.merged_audit, final_manifest=args.final_manifest, hypersim_manifest=args.hypersim_manifest,
            tasks=[t for t in args.tasks.split(",") if t], model_name=args.model_name, require_strict=not args.allow_in_domain,
        )
    elif args.stage == "run":
        report = check_run(run_dir=args.run_dir, tasks=[t for t in args.tasks.split(",") if t], vision_ratio=args.vision_ratio)
    elif args.stage == "generation":
        report = check_generation(eval_json=args.eval_json, reference_dir=args.reference_dir)
    elif args.stage == "post":
        report = check_post_training(config_path=args.config, sft_gate=args.sft_gate)
    else:
        report = check_eval(eval_json=args.eval_json, required_metrics=[m for m in args.require.split(",") if m], profile=args.profile)
    print(report.render())
    if args.json_out:
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
