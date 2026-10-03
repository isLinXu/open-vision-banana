"""Small JSON schema with explicit paths and no model imports."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def checked_id(value, context="id"):
    reserved = {"CON", "PRN", "AUX", "NUL"} | {"COM" + str(n) for n in range(1, 10)} | {"LPT" + str(n) for n in range(1, 10)}
    if (not isinstance(value, str) or not ID_PATTERN.fullmatch(value) or value.endswith(".")
            or value.split(".")[0].upper() in reserved):
        raise ValueError("{} must be a safe, nonempty identifier".format(context))
    return value


def _keys(obj, allowed, context):
    if not isinstance(obj, dict):
        raise ValueError("{} must be an object".format(context))
    unknown = set(obj) - set(allowed)
    if unknown:
        raise ValueError("Unknown {} fields: {}".format(context, sorted(unknown)))


def _integer(value, low, high, context):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("{} must be an integer in [{}, {}]".format(context, low, high))


def validate_config(raw):
    cfg = copy.deepcopy(raw)
    _keys(cfg, {"schema_version", "name", "author", "description", "device", "seed",
                "dataset", "models", "sizes", "tasks", "timing", "output"}, "config")
    if cfg.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    checked_id(cfg.get("name"), "name")
    # Older local profiles may contain an author; identity is not a run parameter.
    cfg.pop("author", None)
    cfg.setdefault("device", "cpu")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", cfg["device"]):
        raise ValueError("device must be cpu, cuda or cuda:N")
    cfg.setdefault("seed", 42)
    _integer(cfg["seed"], 0, 2**32 - 1, "seed")
    for field in ("models", "sizes", "tasks"):
        if not isinstance(cfg.get(field), list) or not cfg[field]:
            raise ValueError("{} must be a nonempty array".format(field))
    for field in ("models", "sizes"):
        if any(not isinstance(obj, dict) for obj in cfg[field]):
            raise ValueError("{} entries must be objects".format(field))
        ids = [checked_id(obj.get("id"), field + ".id") for obj in cfg[field]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate {} IDs".format(field))
    for model in cfg["models"]:
        _keys(model, {"id", "backend", "checkpoint", "base_model_id", "revision",
                      "weights_variant", "options"}, "model")
        checked_id(model.get("backend"), "backend")
        model.setdefault("options", {})
        if not isinstance(model["options"], dict):
            raise ValueError("model.options must be an object")
    for size in cfg["sizes"]:
        _keys(size, {"id", "max_edge", "alignment"}, "size")
        _integer(size.get("max_edge"), 8, 8192, "max_edge")
        size.setdefault("alignment", 1)
        _integer(size["alignment"], 1, size["max_edge"], "alignment")
        if size["max_edge"] % size["alignment"]:
            raise ValueError("max_edge must be divisible by alignment")
    tasks = []
    for task in cfg["tasks"]:
        task = {"name": task} if isinstance(task, str) else task
        _keys(task, {"name", "options"}, "task")
        checked_id(task.get("name"), "task.name")
        task.setdefault("options", {})
        tasks.append(task)
    if len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task names")
    cfg["tasks"] = tasks
    ds = cfg.get("dataset", {})
    _keys(ds, {"kind", "root", "include", "max_samples", "manifest", "label_space",
               "ignore_index", "depth_units", "depth_protocol", "synthetic_count"}, "dataset")
    if ds.get("kind") not in {"images", "manifest", "synthetic"}:
        raise ValueError("dataset.kind must be images, manifest or synthetic")
    if ds["kind"] == "images" and not ds.get("root"):
        raise ValueError("images dataset requires root (or --dataset-root override)")
    if ds["kind"] == "manifest" and not ds.get("manifest"):
        raise ValueError("manifest dataset requires manifest path")
    ds.setdefault("max_samples", 6)
    _integer(ds["max_samples"], 1, 1000000, "max_samples")
    ds.setdefault("ignore_index", 255)
    _integer(ds["ignore_index"], 0, 65535, "ignore_index")
    if ds["kind"] == "synthetic":
        ds.setdefault("synthetic_count", 2)
        _integer(ds["synthetic_count"], 1, 10000, "synthetic_count")
    if "include" in ds and (not isinstance(ds["include"], list) or not ds["include"]
                           or not all(isinstance(p, str) and p for p in ds["include"])):
        raise ValueError("dataset.include must be a nonempty list of patterns")
    if ds.get("depth_protocol", "full") not in {"full", "nyu", "kitti", "eth3d"}:
        raise ValueError("Unsupported depth_protocol")
    cfg["dataset"] = ds
    timing = cfg.setdefault("timing", {})
    _keys(timing, {"warmup", "repeats", "cpu_threads"}, "timing")
    for field, default, low, high in (("warmup", 1, 0, 100), ("repeats", 3, 1, 1000),
                                    ("cpu_threads", 1, 1, 256)):
        timing.setdefault(field, default)
        _integer(timing[field], low, high, "timing." + field)
    output = cfg.setdefault("output", {})
    _keys(output, {"visualizations", "raw_predictions"}, "output")
    for key in ("visualizations", "raw_predictions"):
        output.setdefault(key, True)
        if not isinstance(output[key], bool):
            raise ValueError("output.{} must be boolean".format(key))
    return cfg


def load_config(path, dataset_root=None, device=None, checkpoints=None):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if dataset_root is not None:
        raw["dataset"]["root"] = str(Path(dataset_root).resolve())
    if device is not None:
        raw["device"] = device
    overrides = checkpoints or {}
    for model in raw.get("models", []):
        if model["id"] in overrides:
            model["checkpoint"] = str(Path(overrides[model["id"]]).resolve())
        for key in ("checkpoint",):
            if model.get(key):
                model[key] = str((path.parent / model[key]).resolve())
        for key in ("palette_path", "weights_provenance_path"):
            if model.get("options", {}).get(key):
                model["options"][key] = str((path.parent / model["options"][key]).resolve())
        base = model.get("base_model_id")
        if base and (base.startswith(".") or Path(base).is_absolute()):
            model["base_model_id"] = str((path.parent / base).resolve())
    unknown = set(overrides) - {m["id"] for m in raw.get("models", [])}
    if unknown:
        raise ValueError("Unknown model checkpoint overrides: {}".format(sorted(unknown)))
    for key in ("root", "manifest"):
        if raw.get("dataset", {}).get(key):
            raw["dataset"][key] = str((path.parent / raw["dataset"][key]).resolve())
    return validate_config(raw)
