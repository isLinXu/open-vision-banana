"""One orchestration loop for all registered models, tasks and resolutions."""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from . import PROJECT_ROOT, SCHEMA_VERSION
from .data import load_image, load_target, load_valid_mask, resized_hw, select_samples, sha256_file
from .reporting import PortableReport


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def plan_benchmark(config):
    """No torch/model/network imports, including on a machine without CUDA."""
    return {"schema_version": SCHEMA_VERSION, "name": config["name"],
            "device": config["device"], "max_samples": config["dataset"]["max_samples"],
            "status": "plan_only_not_verified", "matrix": [
                {"model": model["id"], "backend": model["backend"], "size": size,
                 "task": task["name"], "weights_variant": model.get("weights_variant"),
                 "contract_verified": model.get("options", {}).get("inference_contract_verified")}
                for model in config["models"] for size in config["sizes"] for task in config["tasks"]]}


def _git(args):
    try:
        return subprocess.check_output(["git"] + args, cwd=str(PROJECT_ROOT),
                                       stderr=subprocess.DEVNULL, encoding="utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stats(values):
    import numpy as np
    return {"count": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values)),
            "min_ms": float(min(values)), "max_ms": float(max(values))}


def run_benchmark(config, output, allow_download=False, allow_legacy=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .backends import build_backend
    from .tasks import build_task, clean_json
    from .visualization import save_preview, save_raw

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; choose a new run directory: {}".format(output))
    samples = select_samples(config["dataset"])
    # Partial GT is easy to mistake for a full dataset score; reject it up front.
    for task in config["tasks"]:
        coverage = [task["name"] in row["gt"] for row in samples]
        if any(coverage) and not all(coverage):
            raise ValueError("Partial GT coverage for {}; provide a separate manifest".format(task["name"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no implicit CPU fallback")
    torch.set_num_threads(config["timing"]["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    portable = PortableReport(config, PROJECT_ROOT)
    config_hash = hashlib.sha256(json.dumps(portable.clean(config), sort_keys=True).encode("utf-8")).hexdigest()
    evidence = []
    for row in samples:
        item = {"id": row["id"], "synthetic": row.get("synthetic", False), "files": {}}
        if row.get("image"):
            paths = {"image": row["image"]}
            paths.update({"gt:" + key: val for key, val in row["gt"].items()})
            paths.update({"valid:" + key: val for key, val in row.get("valid_masks", {}).items()})
            item["files"] = {key: {"path": val, "sha256": sha256_file(val)} for key, val in paths.items()}
        evidence.append(item)
    cuda = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cuda = {"name": props.name, "total_memory_bytes": props.total_memory,
                "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    report = {"schema_version": SCHEMA_VERSION, "name": config["name"],
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "running", "config_sha256": config_hash, "config": config,
              "source": {"commit": _git(["rev-parse", "HEAD"]),
                         "dirty": bool(_git(["status", "--porcelain"])),
                         "harness_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                                            for p in sorted(Path(__file__).parent.glob("*.py"))}},
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "torch": torch.__version__, "device": str(device), "cuda": cuda,
                          "cpu_threads": torch.get_num_threads(), "allow_legacy": allow_legacy},
              "protocol": {"batch_size": 1, "timing": "synchronized_backend_predict_only",
                           "paired_predict_decode_timing": "one_last_prediction_per_sample; excludes_input_resize_disk_metrics_and_visualization",
                           "excluded_from_timing": ["load_weights", "disk_io", "input_resize", "task_decode", "metrics", "visualization"],
                           "resize": "max_edge_aspect_then_floor_alignment_no_padding_no_crop",
                           "metric_grid": "EXIF-oriented_original_GT_grid",
                           "semantic_resize": "nearest", "continuous_resize": "bilinear_align_corners_false",
                           "depth_prediction_floor": 1e-6, "normal_re_normalized_after_resize": True,
                           "memory": "PyTorch_peak_allocated_and_reserved_includes_resident_model_excludes_driver_context",
                           "allocator_cache": "released_blocks_cleared_before_each_sample_warmup; baseline_after_warmup",
                           "synthetic_quality_is_not_model_accuracy": config["dataset"]["kind"] == "synthetic"},
              "samples": evidence, "models": [], "jobs": [], "groups": [], "errors": []}
    def persist():
        write_json(output / "results.json", portable.clean(report))

    persist()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for spec in config["models"]:
        backend = None
        try:
            backend = build_backend(spec, str(device), allow_download, allow_legacy)
            prepare_start = time.perf_counter()
            backend.prepare()
            synchronize()
            description = backend.describe()
            description["prepare_seconds"] = time.perf_counter() - prepare_start
            report["models"].append(description)
            for size in config["sizes"]:
                for task_spec in config["tasks"]:
                    name = task_spec["name"]
                    if name not in description["tasks"]:
                        raise ValueError("{} cannot evaluate {}; fix the model/task matrix".format(spec["id"], name))
                    task = build_task(name, description["task_contracts"].get(name),
                                      config["dataset"], task_spec["options"])
                    has_gt = name in samples[0]["gt"]
                    if has_gt:
                        task.start_metrics()
                    group_times = []
                    group_decode_times = []
                    group_paired_times = []
                    peaks = []
                    complete_peaks = []
                    for sample in samples:
                        original, image = load_image(sample)
                        height, width = resized_hw(original.height, original.width, size)
                        image = F.interpolate(image, size=(height, width), mode="bilinear",
                                              align_corners=False).to(device)
                        gt = load_target(sample["gt"][name], name) if has_gt else None
                        mask = (load_valid_mask(sample.get("valid_masks", {}).get(name), gt.shape[-2:])
                                if gt is not None else None)
                        if gt is not None:
                            task.validate_gt(gt, (original.height, original.width))
                        with torch.inference_mode():
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            for _ in range(config["timing"]["warmup"]):
                                backend.predict(image, name, size)
                            synchronize()
                            memory_before = None
                            if device.type == "cuda":
                                memory_before = {"baseline_allocated_bytes": torch.cuda.memory_allocated(device),
                                                 "baseline_reserved_bytes": torch.cuda.memory_reserved(device)}
                                torch.cuda.reset_peak_memory_stats(device)
                            times = []
                            prediction = None
                            for _ in range(config["timing"]["repeats"]):
                                # Release the previous prediction before measuring so
                                # peak memory does not count two inference outputs.
                                prediction = None
                                synchronize()
                                start = time.perf_counter()
                                prediction = backend.predict(image, name, size)
                                synchronize()
                                times.append((time.perf_counter() - start) * 1000)
                            memory = None
                            if device.type == "cuda":
                                memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                                memory.update(memory_before)
                                memory["incremental_peak_allocated_bytes"] = memory["peak_allocated_bytes"] - memory_before["baseline_allocated_bytes"]
                                peaks.append(memory)
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(device)
                            synchronize()
                            decode_start = time.perf_counter()
                            decoded = task.decode(prediction)
                            synchronize()
                            decode_ms = (time.perf_counter() - decode_start) * 1000
                            complete_memory = None
                            if device.type == "cuda":
                                complete_memory = {"peak_allocated_bytes": max(memory["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device)),
                                                   "peak_reserved_bytes": max(memory["peak_reserved_bytes"], torch.cuda.max_memory_reserved(device))}
                                complete_peaks.append(complete_memory)
                            if gt is not None:
                                task.update(decoded.values, gt, mask)
                        directory = output / "artifacts" / spec["id"] / size["id"] / name / sample["id"]
                        artifacts = {}
                        if config["output"]["raw_predictions"]:
                            save_raw(decoded, directory, name)
                            artifacts["raw"] = str((directory / "prediction.npy").relative_to(output))
                        if config["output"]["visualizations"]:
                            save_preview(original, decoded, directory,
                                         "{} / {} / {} / {}".format(spec["id"], size["id"], name, sample["id"]))
                            artifacts["preview"] = str((directory / "preview.png").relative_to(output))
                        report["jobs"].append({"model": spec["id"], "size": size["id"], "task": name,
                                               "sample": sample["id"], "original_hw": [original.height, original.width],
                                               "input_hw": [height, width], "output_hw": list(decoded.values.shape[-2:]),
                                               "has_gt": has_gt, "latency": _stats(times), "latency_samples_ms": times,
                                               "decode_ms": decode_ms, "paired_predict_decode_ms": times[-1] + decode_ms,
                                               "memory": memory, "artifacts": artifacts,
                                               "predict_decode_memory": complete_memory,
                                               "auxiliary": clean_json(prediction.auxiliary)})
                        group_times.extend(times)
                        group_decode_times.append(decode_ms)
                        group_paired_times.append(times[-1] + decode_ms)
                        # The next job starts with no tensors from this inference.
                        del image, prediction, decoded
                    group = {"model": spec["id"], "size": size["id"], "task": name,
                             "samples": len(samples), "latency": _stats(group_times),
                             "decode_latency": _stats(group_decode_times), "paired_predict_decode_latency": _stats(group_paired_times),
                             "quality": task.compute(), "memory": None}
                    if peaks:
                        group["memory"] = {key: max(item[key] for item in peaks) for key in peaks[0]}
                        group["predict_decode_memory"] = {key: max(item[key] for item in complete_peaks) for key in complete_peaks[0]}
                    report["groups"].append(group)
                    print("{} / {} / {}: {:.2f} ms p50; quality={}".format(
                        spec["id"], size["id"], name, group["latency"]["p50_ms"], group["quality"]["status"]), flush=True)
                    persist()
        except Exception as error:
            report["errors"].append({"model": spec["id"], "type": type(error).__name__, "message": str(error)})
            report["status"] = "failed"
            persist()
            raise
        finally:
            if backend is not None:
                backend.close()
    report["status"] = "completed"
    report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    persist()
    return portable.clean(report)
