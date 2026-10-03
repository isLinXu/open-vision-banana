"""Registered prediction backends; dataset handling and metric decoding live outside.

The common boundary is RGB [0,1] input at the requested size. Backends return
RGB task encodings or class IDs on the input device and declare their contracts.
Adding a model requires registering one class, without changing the runner.
"""

import copy
import gc
import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .compatibility import load_tensor_state, require_runtime, sha256_file, strict_zip


SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PALETTE = SOURCE_ROOT / "configs" / "semantic_palette.json"
BACKEND_REGISTRY = {}
SPEC_FIELDS = {"id", "backend", "checkpoint", "base_model_id", "revision", "weights_variant", "options"}


@dataclass
class Prediction:
    rgb_output: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    auxiliary: dict = field(default_factory=dict)


def register_backend(name):
    def decorate(backend_class):
        if name in BACKEND_REGISTRY:
            raise ValueError("Backend already registered: %s" % name)
        BACKEND_REGISTRY[name] = backend_class
        return backend_class
    return decorate


def build_backend(spec, device, allow_download=False, allow_legacy=False):
    if not isinstance(spec, dict):
        raise TypeError("Model spec must be an object")
    unknown = set(spec) - SPEC_FIELDS
    if unknown:
        raise ValueError("Unknown model fields: %s" % sorted(unknown))
    name = spec.get("backend")
    if name not in BACKEND_REGISTRY:
        raise ValueError("Unknown backend %r; available: %s" % (name, sorted(BACKEND_REGISTRY)))
    if not isinstance(spec.get("id"), str) or not spec["id"]:
        raise ValueError("Model spec requires a non-empty id")
    return BACKEND_REGISTRY[name](spec, device, allow_download, allow_legacy)


def _add_project_path():
    source = str(SOURCE_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _palette(path):
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    values = document.get("palette", document)
    if not isinstance(values, dict) or not values:
        raise ValueError("Palette must be a non-empty ordered class-name -> RGB object")
    result = {}
    for name, color in values.items():
        if not isinstance(name, str) or not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ValueError("Invalid palette entry: %r" % name)
        if any(type(value) is not int or not 0 <= value <= 255 for value in color):
            raise ValueError("Palette colors must contain three integer bytes")
        result[name] = tuple(color)
    if len(set(result.values())) != len(result):
        raise ValueError("Palette contains duplicate colors and cannot decode class IDs uniquely")
    return result


def _contracts(contract, defaults=None):
    if contract is None:
        return copy.deepcopy(defaults or {})
    if not isinstance(contract, dict):
        raise ValueError("encoding_contract must be a task -> contract object")
    result = copy.deepcopy(defaults or {})
    for task, value in contract.items():
        if not isinstance(value, dict):
            raise ValueError("Task contract must be an object: %s" % task)
        merged = result.get(task, {})
        merged.update(copy.deepcopy(value))
        result[task] = merged
    return result


def _verify_weight_provenance(path, adapter, variant):
    """Verify this benchmark's prepared-checkpoint manifest against actual files.

    EMA is established by comparing the exported tensor values with the cached
    checkpoint's ema_state_dict, before loading any model backbone. A filename
    or a caller-provided 'ema' label alone is never accepted as proof.
    """
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    # Full-pipeline verification runs in a modern runtime. Keep this import lazy
    # so the Tiny/LR-ASPP component backends remain importable on Python 3.7.
    from .checkpoints import resolve_prepared_paths
    record = resolve_prepared_paths(record, manifest_path)
    if record.get("schema_version") != 1 or record.get("weights_variant") != variant:
        raise ValueError("Expected a schema_version=1 prepared-checkpoint provenance with matching weights_variant")
    if not record.get("adapter_dir") or Path(record["adapter_dir"]).resolve() != adapter.resolve():
        raise ValueError("Provenance adapter_dir does not match selected adapter directory")
    provenance = record.get("provenance", {})
    if provenance.get("weights_variant") != variant:
        raise ValueError("Nested provenance weights_variant disagrees with selected adapter")
    exports = provenance.get("export_files", [])
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        target = (adapter / name).resolve()
        matches = [item for item in exports if item.get("path") and Path(item["path"]).resolve() == target]
        if len(matches) != 1 or matches[0].get("sha256") != sha256_file(target):
            raise ValueError("Provenance export path/hash does not match selected %s" % name)
    verified = {"schema_version": 1, "export_hashes_verified": True, "ema_tensors_equal": None}
    if variant == "ema":
        if record.get("validation", {}).get("safe_weights_only") is not True:
            raise ValueError("EMA provenance does not record safe checkpoint export")
        revision = record.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("EMA provenance requires a pinned full source revision")
        expected_remote = record.get("checkpoint_subdir", "") + "/lora.pt"
        sources = [item for item in provenance.get("source_files", [])
                   if item.get("remote_path") == expected_remote and item.get("revision") == revision]
        if len(sources) != 1 or not sources[0].get("local_path"):
            raise ValueError("EMA provenance lacks the pinned source lora.pt")
        source = Path(sources[0]["local_path"])
        if not source.is_file() or sha256_file(source) != sources[0].get("sha256"):
            raise ValueError("EMA source checkpoint is missing or differs from provenance SHA256")
        state = torch.load(str(source), map_location="cpu", weights_only=True)
        ema = state.get("ema_state_dict") if isinstance(state, dict) else None
        if not isinstance(ema, dict) or not ema:
            raise ValueError("Source checkpoint does not contain ema_state_dict")
        normalized = {}
        for key, tensor in ema.items():
            if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError("EMA checkpoint state must contain named tensors")
            canonical = key.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
            if canonical in normalized:
                raise ValueError("EMA key normalization collided")
            normalized[canonical] = tensor
        from safetensors.torch import load_file
        exported = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
        if exported.keys() != normalized.keys() or any(
            exported[key].dtype != normalized[key].dtype
            or not torch.equal(exported[key], normalized[key]) for key in exported
        ):
            raise ValueError("Selected adapter tensors do not equal the source checkpoint EMA")
        verified.update(ema_tensors_equal=True, source_checkpoint=str(source.resolve()), source_revision=revision)
    return verified


class Backend:
    OPTION_FIELDS = set()

    def __init__(self, spec, device, allow_download=False, allow_legacy=False):
        self.spec = copy.deepcopy(spec)
        self.options = self.spec.get("options", {})
        if not isinstance(self.options, dict):
            raise ValueError("options must be an object")
        unknown = set(self.options) - self.OPTION_FIELDS
        if unknown:
            raise ValueError("Unsupported %s options: %s" % (spec["backend"], sorted(unknown)))
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but this runtime has no available CUDA device")
        self.allow_download = bool(allow_download)
        self.allow_legacy = bool(allow_legacy)
        self.model = None
        self.ready = False
        self.metadata = {
            "id": spec["id"], "backend": spec["backend"], "device": str(self.device),
            "weights_variant": spec.get("weights_variant"), "task_contracts": {},
            "checkpoint_revision": spec.get("revision"),
        }

    def prepare(self):
        raise NotImplementedError

    def predict(self, image_01, task, size):
        raise NotImplementedError

    def describe(self):
        return copy.deepcopy(self.metadata)

    def _validate_input(self, image_01, task):
        if not self.ready:
            raise RuntimeError("Call prepare() before predict()")
        if task not in self.metadata["tasks"]:
            raise ValueError("Unsupported task %r for %s" % (task, self.spec["id"]))
        if image_01.ndim != 4 or tuple(image_01.shape[:2]) != (1, 3):
            raise ValueError("Expected RGB input [1,3,H,W]")
        # CUDA device('cuda') and device('cuda:0') denote the same current device.
        expected_index = self.device.index
        if expected_index is None and self.device.type == "cuda":
            expected_index = torch.cuda.current_device()
        if image_01.device.type != self.device.type or (
            self.device.type == "cuda" and image_01.device.index != expected_index
        ):
            raise ValueError("Input must be on backend device %s" % self.device)
        if not image_01.is_floating_point() or not torch.isfinite(image_01).all().item():
            raise ValueError("Input must contain finite floating-point RGB values")
        if image_01.min().item() < 0 or image_01.max().item() > 1:
            raise ValueError("Input RGB range must be [0,1]")

    def close(self):
        self.model = None
        self.ready = False
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@register_backend("tiny")
class TinyBackend(Backend):
    OPTION_FIELDS = {"config", "tasks", "seed", "dtype", "palette_path", "encoding_contract", "trusted_checkpoint"}

    def prepare(self):
        if self.ready:
            return
        self.metadata["runtime"] = require_runtime(self.allow_legacy)
        variant = self.spec.get("weights_variant", "random")
        if variant not in {"random", "pretrained"}:
            raise ValueError("Tiny accepts weights_variant=random or pretrained; provide an explicit exported state_dict")
        if variant == "pretrained" and not self.spec.get("checkpoint"):
            raise ValueError("Pretrained Tiny requires a local checkpoint")
        if self.spec.get("base_model_id"):
            raise ValueError("Standalone Tiny does not use base_model_id")
        _add_project_path()
        tiny = importlib.import_module("open_vision_banana.tiny_model")
        if sys.version_info < (3, 10):
            tiny.zip = strict_zip
            self.metadata["adaptations"] = ["module-scoped tiny_model.zip(strict=True) backport"]
        config = {"base_width": 8, "num_stages": 2, "block_depth": 1, "task_embed_dim": 1}
        config.update(self.options.get("config", {}))
        if config.get("in_channels", 3) != 3 or config.get("out_channels", 3) != 3:
            raise ValueError("Benchmark Tiny boundary requires three-channel RGB input/output")
        if config.get("output_activation", "sigmoid") != "sigmoid":
            raise ValueError("Benchmark Tiny RGB contract requires output_activation=sigmoid")
        tasks = self.options.get("tasks", list(tiny.TINY_TASKS))
        if not tasks or len(set(tasks)) != len(tasks) or not all(isinstance(task, str) for task in tasks):
            raise ValueError("Tiny tasks must be non-empty, unique task names")
        dtype_name = self.options.get("dtype", "float32")
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported Tiny dtype: %s" % dtype_name)
        if dtype_name != "float32" and self.device.type != "cuda":
            raise ValueError("Tiny reduced precision is restricted to CUDA in this harness")
        # Do not change the runner's random stream when constructing a model.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.options.get("seed", 42)))
            self.model = tiny.TinyVisionBanana(tiny.TinyVisionBananaConfig(**config), tasks=tasks)
        if self.spec.get("checkpoint"):
            path = Path(self.spec["checkpoint"])
            if path.is_dir():
                raise ValueError("Tiny checkpoint must be a specific tensor state file; configure architecture explicitly")
            state, provenance = load_tensor_state(
                path, self.allow_legacy, self.options.get("trusted_checkpoint", False)
            )
            self.model.load_state_dict(state, strict=True)
            self.metadata["checkpoint"] = provenance
        self.model.to(device=self.device, dtype=getattr(torch, dtype_name)).eval()
        palette_path = Path(self.options.get("palette_path", DEFAULT_PALETTE))
        palette = _palette(palette_path)
        defaults = {
            "semantic": {"label_space": "ade20k150" if palette_path.resolve() == DEFAULT_PALETTE.resolve() else "custom",
                         "palette": palette, "num_classes": len(palette)},
            "depth": {"representation": "metric_power", "colormap": "cube_edge", "lam": -3.0, "c": 10.0 / 3.0},
            "normal": {"encoding": "paper_camera"},
            "edge": {"encoding": "rgb_edge_visualization"},
            "generation": {"encoding": "rgb"},
        }
        contract = _contracts(self.options.get("encoding_contract"), defaults)
        self.metadata.update(
            weights_variant=variant, trained=variant != "random", tasks=list(tasks),
            config=dict(vars(self.model.config)), parameters=self.model.num_parameters(),
            dtype=dtype_name, seed=int(self.options.get("seed", 42)),
            task_contracts={task: contract[task] for task in tasks if task in contract},
            palette_path=str(palette_path.resolve()), palette_sha256=sha256_file(palette_path),
            source="src/open_vision_banana/tiny_model.py",
            source_sha256=sha256_file(Path(tiny.__file__)),
            accuracy_scope="component smoke test only" if variant == "random" else "requires matching ground truth",
        )
        self.ready = True

    @torch.no_grad()
    def predict(self, image_01, task, size):
        self._validate_input(image_01, task)
        dtype = next(self.model.parameters()).dtype
        prediction = self.model(image_01.to(dtype=dtype) * 2 - 1, task)
        return Prediction(rgb_output=prediction.float(), auxiliary={"trained": self.metadata["trained"]})


def _voc_palette(categories):
    result = {}
    for index, category in enumerate(categories):
        color, bits = [0, 0, 0], index
        for shift in range(8):
            for channel in range(3):
                color[channel] |= ((bits >> channel) & 1) << (7 - shift)
            bits >>= 3
        result[category] = tuple(color)
    return result


@register_backend("torchvision_lraspp")
class LRASPPBackend(Backend):
    OPTION_FIELDS = {"dtype", "trusted_checkpoint", "expected_sha256"}

    def prepare(self):
        if self.ready:
            return
        self.metadata["runtime"] = require_runtime(self.allow_legacy)
        if self.spec.get("weights_variant", "pretrained") != "pretrained":
            raise ValueError("The LRASPP reference backend requires weights_variant=pretrained")
        if self.spec.get("base_model_id"):
            raise ValueError("LRASPP architecture is fixed; do not provide base_model_id")
        if self.options.get("dtype", "float32") != "float32":
            raise ValueError("LRASPP reference uses float32; reduced precision needs a separate validated backend")
        path = self.spec.get("checkpoint")
        if not path and not self.allow_download:
            raise RuntimeError("LRASPP needs a local checkpoint or explicit --allow-download")
        import torchvision
        from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

        weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        if path:
            self.model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
            expected = self.options.get("expected_sha256")
            if expected and sha256_file(path) != expected:
                raise ValueError("LRASPP checkpoint SHA256 does not match expected_sha256")
            state, provenance = load_tensor_state(path, self.allow_legacy, self.options.get("trusted_checkpoint", False))
            self.model.load_state_dict(state, strict=True)
            self.metadata["checkpoint"] = provenance
        elif self.allow_download:
            self.model = lraspp_mobilenet_v3_large(weights=weights)
            # torchvision validates its official file's hash prefix; record the full file hash too.
            cached = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
            self.metadata["checkpoint"] = {"path": str(cached), "sha256": sha256_file(cached),
                                           "load_method": "torchvision official weights; download explicitly enabled"}
        else:
            raise RuntimeError("LRASPP needs a local checkpoint or explicit --allow-download")
        self.model.to(self.device).eval()
        categories = weights.meta["categories"]
        self.metadata.update(
            weights_variant="pretrained", trained=True, tasks=["semantic"], dtype="float32",
            parameters=sum(parameter.numel() for parameter in self.model.parameters()),
            weights_name=weights.name, weights_url=weights.url, torchvision=torchvision.__version__,
            task_contracts={"semantic": {"label_space": "voc21", "palette": _voc_palette(categories),
                                         "num_classes": len(categories)}},
            normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            accuracy_scope="input sizing differs from official 520px short-edge evaluation unless explicitly configured",
        )
        self.ready = True

    @torch.no_grad()
    def predict(self, image_01, task, size):
        self._validate_input(image_01, task)
        image = image_01.float()
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        logits = self.model((image - mean) / std)["out"]
        if not torch.isfinite(logits).all().item():
            raise ValueError("Nonfinite LRASPP logits; refusing argmax that would hide model failures")
        return Prediction(labels=logits.argmax(dim=1), auxiliary={"label_space": "voc21"})


@register_backend("vision_banana")
class VisionBananaBackend(Backend):
    OPTION_FIELDS = {
        "dtype", "inference_options", "inference_contract_verified", "model_time_convention",
        "task_prompts", "encoding_contract", "palette_path", "num_inference_steps", "strength",
        "seed", "weights_provenance_path",
    }

    def prepare(self):
        if self.ready:
            return
        self.metadata["runtime"] = require_runtime(full_pipeline=True)
        if self.options.get("inference_contract_verified") is not True:
            raise ValueError("Checkpoint inference contract is unverified; refusing diffusion inference")
        convention = self.options.get("model_time_convention")
        if convention not in {"native_sana", "native_flux", "diffusers_native"}:
            raise ValueError("Explicit model_time_convention must match a supported native main-branch path")
        variant = self.spec.get("weights_variant")
        if variant not in {"online", "ema"}:
            raise ValueError("Trained VisionBanana requires explicit weights_variant=online or ema")
        checkpoint = Path(self.spec.get("checkpoint") or "")
        if not self.spec.get("checkpoint") or not checkpoint.is_dir():
            raise ValueError("Select a specific local adapter/checkpoint directory; HF repository roots are not accepted")
        adapter = checkpoint
        if not (adapter / "adapter_model.safetensors").is_file():
            candidates = [checkpoint / "lora_adapter", checkpoint / "adapter"]
            adapter = next((candidate for candidate in candidates if (candidate / "adapter_model.safetensors").is_file()), checkpoint)
        weight_file = adapter / "adapter_model.safetensors"
        config_file = adapter / "adapter_config.json"
        if not weight_file.is_file() or not config_file.is_file():
            raise ValueError("Local adapter must contain adapter_model.safetensors and adapter_config.json")
        weight_sha = sha256_file(weight_file)
        provenance = self.options.get("weights_provenance_path")
        if variant == "ema" and not provenance:
            raise ValueError("EMA requires a provenance JSON for an explicitly exported EMA adapter; default online adapter is not EMA")
        if provenance:
            provenance_verification = _verify_weight_provenance(provenance, adapter, variant)
        else:
            provenance_verification = {"export_hashes_verified": False, "ema_tensors_equal": None}
        base = self.spec.get("base_model_id")
        if not base:
            raise ValueError("VisionBanana requires explicit base_model_id")
        if not self.allow_download and not Path(base).is_dir():
            raise RuntimeError("Without --allow-download, base_model_id must be a complete local diffusers model directory")
        contract = _contracts(self.options.get("encoding_contract"))
        if not contract:
            raise ValueError("Trained model requires an explicit encoding_contract per evaluated task")
        if "semantic" in contract and self.options.get("palette_path"):
            contract["semantic"]["palette"] = _palette(self.options["palette_path"])
            contract["semantic"]["num_classes"] = len(contract["semantic"]["palette"])
        if "semantic" in contract and (
            not contract["semantic"].get("label_space") or not contract["semantic"].get("palette")
        ):
            raise ValueError("Semantic contract requires label_space and the exact training palette")
        if "depth" in contract and contract["depth"].get("representation") not in {"metric_power", "normalized_scalar"}:
            raise ValueError("Depth contract must distinguish metric_power from normalized_scalar representation")
        if "normal" in contract and contract["normal"].get("encoding") not in {"paper_camera", "standard_xyz_to_rgb"}:
            raise ValueError("Normal contract must select paper_camera or standard_xyz_to_rgb")
        prompts = self.options.get("task_prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError("task_prompts must be an object")
        steps = self.options.get("num_inference_steps", 1)
        strength = self.options.get("strength", 0.85)
        if type(steps) is not int or steps < 1 or not 0 < strength <= 1:
            raise ValueError("num_inference_steps must be positive and strength in (0,1]")
        dtype_name = self.options.get("dtype", "bfloat16")
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("VisionBanana dtype must be float32, float16 or bfloat16")
        _add_project_path()
        from open_vision_banana.inference.optimization import InferenceOptions
        from open_vision_banana.inference.pipeline import VisionBanana
        from open_vision_banana.model_registry import resolve_model_config

        family = resolve_model_config(base).family
        if (family == "sana" and convention != "native_sana") or (family == "flux" and convention != "native_flux"):
            raise ValueError("Time convention %s does not match base-model family %s" % (convention, family))
        if family not in {"sana", "flux"} and convention != "diffusers_native":
            raise ValueError("Other model families require model_time_convention=diffusers_native")
        optimization = dict(self.options.get("inference_options", {}))
        optimization.setdefault("dtype", dtype_name)
        if optimization["dtype"] != dtype_name:
            raise ValueError("inference_options.dtype must match backend dtype")
        unknown = set(optimization) - set(InferenceOptions.__dataclass_fields__)
        if unknown:
            raise ValueError("Unsupported inference_options: %s" % sorted(unknown))
        # Current BaseGenerator moves pipe.to(device) unless sequential offload is set;
        # refuse the separate model-offload option until that combination is validated.
        if optimization.get("enable_model_cpu_offload"):
            raise ValueError("model_cpu_offload is not validated for main's generator; use sequential offload with a separate smoke test")
        self.model = VisionBanana.from_pretrained(
            model_name=base, lora_path=str(checkpoint.resolve()), device=str(self.device),
            dtype=getattr(torch, dtype_name), inference_options=InferenceOptions(**optimization),
        )
        generator = self.model.generator
        if not generator._resolved_lora_path:
            raise RuntimeError("LoRA was not resolved; refusing unadapted base-model inference")
        active_adapter = Path(generator._resolved_lora_path).resolve()
        if active_adapter != checkpoint.resolve():
            raise RuntimeError("Generator resolved a different checkpoint directory")
        self.prompts = dict(self.model.training_prompts)
        self.prompts.update(prompts)
        missing = [task for task in contract if not isinstance(self.prompts.get(task), str) or not self.prompts[task].strip()]
        if missing:
            self.close()
            raise ValueError("Exact training task_prompts are required for: %s" % missing)
        self.metadata.update(
            weights_variant=variant, trained=True, tasks=list(contract), dtype=dtype_name,
            task_contracts=contract, base_model_id=base, model_family=family,
            checkpoint={"path": str(adapter.resolve()), "sha256": weight_sha,
                        "adapter_config_sha256": sha256_file(config_file), "weights_provenance_path": provenance},
            weights_provenance_verification=provenance_verification,
            model_time_convention=convention, inference_contract_verified=True,
            task_prompts=dict(self.prompts), num_inference_steps=steps, strength=strength,
            inference_options=optimization, parameters=None,
            load_verification="main generator rejects zero attached/merged LoRA layers",
            size_scope="requested tensor dimensions; generator may apply family-specific latent alignment",
        )
        self.ready = True

    @torch.no_grad()
    def predict(self, image_01, task, size):
        self._validate_input(image_01, task)
        alignment = int(size.get("alignment", 1))
        if alignment < 1:
            raise ValueError("Size alignment must be positive")
        if image_01.shape[-2] % alignment or image_01.shape[-1] % alignment:
            raise ValueError("Preprocessed dimensions do not satisfy configured alignment")
        output = self.model.generator.generate(
            prompt=self.prompts[task], image=image_01 * 2 - 1,
            num_inference_steps=self.metadata["num_inference_steps"], strength=self.metadata["strength"],
            height=image_01.shape[-2], width=image_01.shape[-1], seed=int(self.options.get("seed", 42)),
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError("Generator must return an RGB tensor")
        return Prediction(rgb_output=output.to(device=image_01.device, dtype=torch.float32),
                          auxiliary={"prompt": self.prompts[task], "model_time_convention": self.metadata["model_time_convention"]})
