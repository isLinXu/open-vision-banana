#!/usr/bin/env python3
"""Portable six-benchmark Qwen suite. Print the plan by default; --execute uses idle GPUs.

Linux GPU execution; one process per GPU, shared advisory GPU locks. Existing
outputs are reused only within an identical content-addressed run plan.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))
from open_vision_banana.evaluation.shards import shard_range

SPECS = {
    "nyu": ("depth", "test", 654, 4), "eth3d": ("depth", "val", 454, 4),
    "diode": ("depth", "val", 325, 4), "cityscapes": ("semantic", "val", 500, 4),
    "reasonseg": ("referring", "val", 344, 2), "refcocog": ("referring", "val", 4896, 32),
}


def finite_metrics(value):
    if isinstance(value, dict):
        return all(finite_metrics(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_metrics(v) for v in value)
    return value is not None and (not isinstance(value, (float, int)) or math.isfinite(value))


def command(config, model, dataset, output):
    task, split, _, _ = SPECS[dataset]
    args = [sys.executable, str(SOURCE / "scripts/evaluate_external_audited.py"),
            "--model-name", model.get("model_name", "qwen-image-2.1"), "--benchmark", task,
            f"--{dataset}-root", config["data_roots"][dataset], "--split", split,
            "--device", "cuda", "--dtype", "bfloat16", "--batch-size", "1",
            "--resolution", str(config.get("resolution", 512)), "--native-resolution",
            "--num-inference-steps", "50", "--guidance-scale", "4", "--strength", "0.85",
            "--seed", "42", "--depth-colormap", "cube_edge", "--depth-curve", "metric_power",
            "--reasonseg-query-mode", "all", "--reasonseg-annotation-variants",
            "--reasonseg-annotation-mode", config["reasonseg_annotation_mode"],
            "--referring-prompt-version", "query_v2", "--output-json", str(output)]
    if model.get("lora_path"):
        args += ["--lora-path", model["lora_path"]]
    return args


def validate(path, dataset, count, *, fingerprint=None, config=None):
    data = json.loads(path.read_text())
    record = data["benchmarks"][f"{SPECS[dataset][0]}/{dataset}"]
    metrics = record["decoded_metrics"]
    checked_metrics = metrics
    if dataset == "cityscapes":
        # A one-image smoke legitimately lacks some classes. The core writer
        # serializes undefined class IoUs as null; headline mIoU must be finite.
        classes = metrics.get("per_class_iou", [])
        if len(classes) != 19 or not all(v is None or isinstance(v, (float, int)) and math.isfinite(v) for v in classes):
            raise ValueError(f"Invalid Cityscapes class diagnostics: {path}")
        checked_metrics = {k: v for k, v in metrics.items() if k != "per_class_iou"}
    if record["samples"] != count or metrics.get("skipped_samples", 0) or not finite_metrics(checked_metrics):
        raise ValueError(f"Incomplete or invalid metrics: {path}")
    if dataset in {"reasonseg", "refcocog"} and metrics["samples"] != count:
        raise ValueError(f"Missing referring predictions: {path}")
    if dataset == "reasonseg" and config is not None:
        variants = metrics.get("annotation_variants", {})
        if set(variants) != {"legacy_union", "lisa_v1"} or any(
            v.get("samples") != count or not v.get("valid_pixels") for v in variants.values()
        ):
            raise ValueError(f"Missing ReasonSeg annotation diagnostics: {path}")
    if fingerprint is not None and record["dataset_fingerprint"] != fingerprint:
        raise ValueError(f"Dataset changed after preflight: {path}")
    if config is not None:
        meta = data["metadata"]
        resolution = config.get("resolution", 512)
        verified = [dict(raw_pipeline_output=raw, every_prediction_checked=True) for raw in ([resolution, resolution], "pipeline_native")]
        if meta.get("resolution_verification") not in verified:
            raise ValueError(f"Raw pipeline resolution not verified: {path}")
        if dataset in {"reasonseg", "refcocog"} and meta.get("referring_prompt_verification") != dict(version="query_v2", every_prediction_checked=True):
            raise ValueError(f"Actual referring prompt not verified: {path}")
        if meta["reasonseg_annotation_mode"] != config["reasonseg_annotation_mode"]:
            raise ValueError(f"Wrong ReasonSeg annotation mode: {path}")


def run_process(args, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(args) + "\n")
        handle.flush()
        subprocess.run(args, cwd=SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--lock-dir", type=Path, default=Path("/tmp/ovb-eval-gpu-locks"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    config = json.loads(args.config.read_text())
    if config["reasonseg_annotation_mode"] not in {"legacy_union", "lisa_v1"}:
        raise ValueError("Choose an explicit ReasonSeg annotation mode")
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be nonempty and unique")
    models = config["models"]
    if not models:
        raise ValueError("At least one model is required")
    if len({m["name"] for m in models}) != len(models):
        raise ValueError("Model names must be unique")
    for model in models:
        if Path(model["name"]).name != model["name"] or model["name"] in {".", ".."}:
            raise ValueError("Model name must be a safe directory name")
    datasets = config.get("datasets", list(SPECS))
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in SPECS for d in datasets):
        raise ValueError("Invalid dataset selection")
    plan = dict(config=config, source_sha256={
        p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in (SOURCE / "src", SOURCE / "scripts") for p in sorted(folder.rglob("*.py"))})
    encoded = json.dumps(plan, sort_keys=True).encode()
    print(json.dumps({"plan_sha256": hashlib.sha256(encoded).hexdigest(),
                      "models": [m["name"] for m in models], "datasets": datasets,
                      "samples_per_model": sum(SPECS[d][2] for d in datasets), "gpus": gpu_ids}))
    if not args.execute:
        return
    import fcntl
    from restore_external_eval_data import sha256

    args.output.mkdir(parents=True, exist_ok=True)
    run_lock = (args.output / "suite.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output belongs to a different plan; use a new output directory")
    plan_path.write_text(json.dumps(plan, indent=2))
    for model in models:
        if model.get("lora_path"):
            for record in model["weight_files"]:
                if sha256(Path(record["path"])) != record["sha256"]:
                    raise ValueError(f"Weight mismatch: {record['path']}")
            if not model["weight_files"]:
                raise ValueError("Adapter SHA256 inventory is required")
    # Verify counts before loading any model.
    fingerprints = {}
    for dataset in datasets:
        path = args.output / "preflight" / f"{dataset}.json"
        run_process(command(config, models[0], dataset, path) + ["--preflight-only"],
                    os.environ.copy(), path.with_suffix(".log"))
        data = json.loads(path.read_text())["preflight"]
        if not data["ok"] or data["jobs"][0]["samples"] != SPECS[dataset][2]:
            raise ValueError(f"Unexpected benchmark inventory: {dataset}")
        fingerprints[dataset] = data["jobs"][0]["dataset_fingerprint"]
    args.lock_dir.mkdir(parents=True, exist_ok=True)
    leases, available = [], queue.Queue()
    for gpu in gpu_ids:
        lease = (args.lock_dir / f"gpu-{gpu}.lock").open("a")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
                                           "--format=csv,noheader,nounits"], text=True).strip())
        if used >= 1000:
            raise RuntimeError(f"GPU {gpu} is occupied ({used} MiB); choose idle GPUs")
        uuid = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
        leases.append(lease)
        available.put(uuid)

    def evaluate(job):
        model, dataset, index, smoke = job
        uuid = available.get()
        try:
            directory = args.output / model["name"] / dataset
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("smoke.json" if smoke else f"shard{index}.json")
            shards = SPECS[dataset][3]
            count = 1 if smoke else len(shard_range(SPECS[dataset][2], shards, index))
            if path.exists():
                validate(path, dataset, count, fingerprint=fingerprints[dataset], config=config)
                return
            extra = ["--max-samples", "1"] if smoke else ["--num-shards", str(shards), "--shard-index", str(index)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, OPEN_VISION_BANANA_MODEL_PATH=model["base_path"], OMP_NUM_THREADS="8")
            run_process(command(config, model, dataset, path) + extra, env, path.with_suffix(".log"))
            result = json.loads(path.read_text())
            result["metadata"]["external_run_identity"] = {**model, "model": model["name"]}
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
            validate(path, dataset, count, fingerprint=fingerprints[dataset], config=config)
        finally:
            available.put(uuid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        # Complete every non-RefCOCOg group before starting RefCOCOg.
        for phase in ([d for d in datasets if d != "refcocog"], [d for d in datasets if d == "refcocog"]):
            list(pool.map(evaluate, [(m, d, 0, True) for d in phase for m in models]))
            list(pool.map(evaluate, [(m, d, i, False) for d in phase for m in models for i in range(SPECS[d][3])]))
            for dataset in phase:
                for model in models:
                    directory = args.output / model["name"] / dataset
                    full = directory / "full.json"
                    files = [str(directory / f"shard{i}.json") for i in range(SPECS[dataset][3])]
                    run_process([sys.executable, str(SOURCE / "scripts/merge_eval_shards.py"), *files,
                                 "--output-json", str(full), "--output-csv", str(directory / "full.csv")],
                                os.environ.copy(), directory / "merge.log")
                    validate(full, dataset, SPECS[dataset][2], fingerprint=fingerprints[dataset], config=config)
            if phase:
                run_process([sys.executable, str(SOURCE / "scripts/export_external_eval_comparison.py"),
                             "--run-root", str(args.output), "--output", str(args.output / "comparison")],
                            os.environ.copy(), args.output / "comparison-export.log")
    print("All selected benchmark groups verified")


if __name__ == "__main__":
    main()
