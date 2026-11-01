"""Unified multi-task, multi-model Trainer class.

Assembles ModelBackend, TrainingLosses, and TrainingCallbacks into a
complete training loop that replaces both train_sana_512.py and train_flux.py.

Usage:
    config = load_config(args)
    trainer = Trainer(config)
    trainer.train()
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Subset

from ..model_registry import resolve_model_config
from . import full_finetune, qwen_image21
from .callbacks import TrainingCallbacks
from .config import TrainingConfig
from .distributed import (
    all_reduce_gradients_,
    barrier,
    broadcast_float,
    broadcast_parameters_,
    cleanup,
    clip_grad_norm,
    ensure_process_group,
    init_distributed,
)
from .edit_forward import edit_latent_layout
from .losses import TrainingLosses
from .model_backend import ModelBackend
from .optimization import SchedulerFactory
from .sampling import TaskWeightedSampler
from .utils import EMAModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def drop_oversized_gradients(grad_norm: float, drop_grad_norm: float) -> bool:
    """Return True when this update should be discarded instead of clipped.

    Paper-aligned training still clips at ``max_grad_norm``. A positive
    ``drop_grad_norm`` (must be larger than the clip) throws away exploding
    batches. Disabled when ``drop_grad_norm <= 0``.
    """
    return drop_grad_norm > 0.0 and grad_norm > drop_grad_norm


# ── Task aliases and default prompts ──────────────────────────────────────

TASK_ALIASES = {
    "segmentation": "semantic",
    "seg": "semantic",
    # Hypersim/KITTI manifests label cube-edge metric targets as metric_depth.
    # They train as depth and keep the prompt stored on each row.
    "metric_depth": "depth",
}

TASK_PROMPTS = {
    # Paper Section 3.2: the depth task is metric. A row without a prompt must not fall
    # back to a relative-depth instruction while its target is cube-edge metric depth.
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
    "semantic": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
    "structure": "Generate a binary structural edge map. Use a black background and thin white edges aligned to visible scene boundaries.",
}


# ── Shared dataset ────────────────────────────────────────────────────────


def uniform_task_prompts(dataset: Dataset) -> dict[str, str]:
    """Return the prompt of each task whose training samples all share one prompt.

    That prompt is what inference has to use for the task. Tasks with per-sample prompts
    (e.g. a color legend per image) have no single prompt to record and are left out.
    """
    prompts: dict[str, set[str]] = {}
    for part in getattr(dataset, "datasets", [dataset]):
        for sample in part.samples:
            # A single-object pilot still must not bake its expression into
            # the checkpoint as the prompt for every future RES query.
            if sample["task"] == "referring":
                continue
            prompts.setdefault(sample["task"], set()).add(sample["prompt"])
    return {
        task: next(iter(found))
        for task, found in prompts.items()
        if len(found) == 1 and next(iter(found))
    }


def _reject_identity_generation_replay(dataset, model_name: str) -> None:
    """Editors condition on the source image, so a T2I-style generation row whose
    input is its own target would train identity reconstruction, not generation.
    Generation-preservation data for instruction editors must be edit pairs
    (source, instruction, edited target), e.g. from
    ``scripts/build_generation_replay_manifest.py --kind edit``.
    """
    identity = 0
    for part in getattr(dataset, "datasets", [dataset]):
        for sample in getattr(part, "samples", []):
            if sample["task"] == "generation" and sample["image_path"] == sample["target_path"]:
                identity += 1
    if identity:
        raise ValueError(
            f"{identity} 'generation' rows use the same file as input and target, but "
            f"'{model_name}' is an instruction editor that conditions on the input image. "
            "Provide edit pairs (input != target) for generation replay, or drop the "
            "'generation' task for this backbone."
        )


_SEMANTIC_RGB_TUPLE_RE = re.compile(
    r'"([^"]+)"\s*:\s*<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>'
)


def parse_semantic_legend_from_prompt(prompt: str) -> dict[str, tuple[int, int, int]]:
    """Extract a ``class_name -> (R, G, B)`` mapping from a semantic legend prompt.

    Understands all four Fig. 2 styles (RGB-tuple JSON, named-colour JSON, natural
    language, hex); previously only ``<R, G, B>`` tuples were recognised, so
    natural/hex rows were silently never restyled or colour-remapped.
    """
    matches = _SEMANTIC_RGB_TUPLE_RE.findall(prompt)
    if matches:
        return {name: (int(r), int(g), int(b)) for name, r, g, b in matches}
    from ..data.openvocab_semantic import parse_prompt_mapping

    return parse_prompt_mapping(prompt)


class ManifestVisionDataset(Dataset):
    """Load input/target pairs from a JSONL manifest for multi-task training.

    Works for both SANA and FLUX training — the dataset is model-agnostic.
    """

    def __init__(
        self,
        manifest_path: Path,
        resolution: int,
        split: str,
        tasks: set[str],
        max_samples: int | None = None,
        data_root: Path | None = None,
        depth_curve: str = "metric_power",
        depth_log_alpha: float = 0.4,
        depth_colormap: str = "cube_edge",
        depth_colormaps: list[str] | None = None,
        semantic_prompt_styles: list[str] | None = None,
        semantic_color_remap_prob: float = 0.0,
        aspect_preserving_crop: bool = False,
        seed: int = 42,
        tar_index_path: Path | None = None,
        inner_prefix: str = "SenseNova-Vision-Corpus-50M",
        disk_cache_dir: Path | None = None,
    ):
        self.resolution = resolution
        self.split = split
        self.depth_curve = depth_curve
        self.depth_log_alpha = depth_log_alpha
        self.depth_colormap = depth_colormap
        self.depth_colormaps = list(depth_colormaps) if depth_colormaps else None
        self.semantic_prompt_styles = (
            list(semantic_prompt_styles) if semantic_prompt_styles else None
        )
        self.semantic_color_remap_prob = float(semantic_color_remap_prob)
        self.aspect_preserving_crop = bool(aspect_preserving_crop)
        self.rng = random.Random(seed)
        self.inner_prefix = inner_prefix
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.samples: list[dict[str, str]] = []
        path_anchor = data_root.resolve() if data_root else manifest_path
        if "referring" in tasks:
            roots = [Path(manifest_path).parent]
            if data_root is not None:
                roots.append(Path(data_root))
            if any((root / "BUILD_INCOMPLETE").exists() for root in roots):
                raise ValueError("RES dataset build or restore is incomplete; validate before training")

        # Optional indexed_gzip tar stream support (Solution 3)
        self._tar_files: dict[str, dict[str, int]] = {}
        self._tar_volumes: list[Path] = []
        self._gzip_index_path: Path | None = None
        self._tar_reader = None
        resolved_tar_idx = Path(tar_index_path) if tar_index_path else None
        if resolved_tar_idx is None:
            for cand in (
                (data_root / "tar_index.json") if data_root else None,
                Path(manifest_path).parent / "tar_index.json",
            ):
                if cand is not None and cand.is_file():
                    resolved_tar_idx = cand
                    break
        if resolved_tar_idx is not None and resolved_tar_idx.is_file():
            raw_idx = json.loads(resolved_tar_idx.read_text(encoding="utf-8"))
            self._tar_files = raw_idx.get("files", {})
            self._tar_volumes = [Path(v["path"]) for v in raw_idx.get("volumes", [])]
            if raw_idx.get("gzip_index"):
                gz_cand = Path(raw_idx["gzip_index"])
                if not gz_cand.is_file():
                    gz_cand = resolved_tar_idx.parent / gz_cand.name
                if gz_cand.is_file():
                    self._gzip_index_path = gz_cand
            elif resolved_tar_idx.with_suffix(".gzidx").is_file():
                self._gzip_index_path = resolved_tar_idx.with_suffix(".gzidx")

        from open_vision_banana.data.manifest import resolve_manifest_path

        with Path(manifest_path).open(encoding="utf-8") as f:
            for _line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_task = row.get("task")
                mapped = TASK_ALIASES.get(raw_task, raw_task)
                if mapped not in tasks:
                    continue
                if row.get("split") != split:
                    continue
                raw_inp = row.get("input", "")
                raw_tgt = row.get("target", "")
                # Pure T2I generation rows may provide only "target" (or only "input"/"image")
                if mapped == "generation":
                    if not raw_tgt:
                        raw_tgt = raw_inp or row.get("image", "")
                    if not raw_inp:
                        raw_inp = raw_tgt
                img_path = resolve_manifest_path(raw_inp, path_anchor)
                tgt_path = resolve_manifest_path(raw_tgt, path_anchor)
                inp_tar_key = self._resolve_tar_key(raw_inp)
                tgt_tar_key = self._resolve_tar_key(raw_tgt)
                if not (img_path.is_file() or inp_tar_key) or not (tgt_path.is_file() or tgt_tar_key):
                    raise FileNotFoundError(
                        f"{manifest_path}:{_line_no}: missing input/target: {img_path}, {tgt_path}"
                    )
                prompt = row.get("prompt") or TASK_PROMPTS.get(
                    mapped, TASK_PROMPTS.get(raw_task, "")
                )
                if mapped == "referring":
                    from ..data.res_contract import resolve_referring_prompt

                    try:
                        prompt = resolve_referring_prompt(row)
                    except ValueError as exc:
                        raise ValueError(f"{manifest_path}:{_line_no}: {exc}") from exc
                mapping = row.get("mapping") if mapped == "semantic" else None
                background = row.get("background_color") if mapped == "instance" else None
                self.samples.append(
                    {
                        "image_path": str(img_path),
                        "target_path": str(tgt_path),
                        "input_tar_key": inp_tar_key or "",
                        "target_tar_key": tgt_tar_key or "",
                        "task": mapped,
                        "prompt": prompt,
                        # Builders record the exact legend; it beats re-parsing the prompt.
                        **({"mapping": {k: tuple(int(c) for c in v) for k, v in mapping.items()}}
                           if isinstance(mapping, dict) and mapping else {}),
                        # Fig. 3 instance targets may use a non-black background; the Appendix A
                        # decoder must be told, so it rides along to the batch (review item 6b).
                        **({"background_color": tuple(int(c) for c in background)}
                           if isinstance(background, (list, tuple)) and len(background) == 3 else {}),
                        **{
                            key: row[key]
                            for key in ("expression", "encoding", "sample_id", "source_dataset")
                            if key in row and mapped == "referring"
                        },
                    }
                )

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def _resolve_tar_key(self, rel_path: str) -> str | None:
        if not self._tar_files or not rel_path:
            return None
        clean = rel_path.lstrip("./")
        if clean in self._tar_files:
            return clean
        if self.inner_prefix:
            prefixed = f"{self.inner_prefix.rstrip('/')}/{clean}"
            if prefixed in self._tar_files:
                return prefixed
            prefix_slash = f"{self.inner_prefix.rstrip('/')}/"
            if clean.startswith(prefix_slash):
                stripped = clean[len(prefix_slash) :]
                if stripped in self._tar_files:
                    return stripped
        return None

    def _get_tar_reader(self):
        if self._tar_reader is None:
            from open_vision_banana.data.tar_dataset import (
                IndexedTarReader,
                SequentialTarReader,
            )

            if self._gzip_index_path and self._gzip_index_path.is_file():
                try:
                    self._tar_reader = IndexedTarReader(self._tar_volumes, self._gzip_index_path)
                    return self._tar_reader
                except ImportError:
                    pass
            self._tar_reader = SequentialTarReader(self._tar_volumes)
        return self._tar_reader

    def _open_pil_rgb(self, disk_path: str, tar_key: str) -> Image.Image:
        import io
        import os

        p = Path(disk_path)
        if p.is_file():
            with Image.open(p) as img:
                img.load()
                return img.convert("RGB")
        if tar_key and tar_key in self._tar_files:
            if self.disk_cache_dir is not None:
                cache_file = self.disk_cache_dir / tar_key.replace("/", os.sep)
                if cache_file.is_file():
                    with Image.open(cache_file) as img:
                        img.load()
                        return img.convert("RGB")
            info = self._tar_files[tar_key]
            raw_bytes = self._get_tar_reader().read_at(info["offset"], info["size"])
            if self.disk_cache_dir is not None:
                cache_file = self.disk_cache_dir / tar_key.replace("/", os.sep)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(raw_bytes)
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.load()
                return img.convert("RGB")
        raise FileNotFoundError(f"Missing image on disk ({disk_path}) and in tar index ({tar_key})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load(idx)
        except Exception as exc:
            raise RuntimeError(f"Unreadable manifest sample idx={idx}: {self.samples[idx]}") from exc

    def _resize_and_crop_pair(
        self,
        inp_img: Image.Image,
        tgt_img: Image.Image,
        target_resample: Image.Resampling,
    ) -> tuple[Image.Image, Image.Image]:
        """Resize (inp, tgt) to (resolution, resolution), optionally preserving aspect ratio."""
        r = self.resolution
        if not self.aspect_preserving_crop or inp_img.size == (r, r):
            return (
                inp_img.resize((r, r), Image.Resampling.LANCZOS),
                tgt_img.resize((r, r), target_resample),
            )
        w, h = inp_img.size
        if tgt_img.size != (w, h):
            tgt_img = tgt_img.resize((w, h), target_resample)
        short = min(w, h)
        if short <= 0:
            return (
                inp_img.resize((r, r), Image.Resampling.LANCZOS),
                tgt_img.resize((r, r), target_resample),
            )
        scale = r / float(short)
        new_w = max(r, int(round(w * scale)))
        new_h = max(r, int(round(h * scale)))
        inp_scaled = inp_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tgt_scaled = tgt_img.resize((new_w, new_h), target_resample)
        if new_w == r and new_h == r:
            return inp_scaled, tgt_scaled
        if self.split == "train":
            left = self.rng.randint(0, new_w - r)
            top = self.rng.randint(0, new_h - r)
        else:
            left = (new_w - r) // 2
            top = (new_h - r) // 2
        box = (left, top, left + r, top + r)
        return inp_scaled.crop(box), tgt_scaled.crop(box)

    def _load(self, idx: int) -> dict:
        s = self.samples[idx]
        resample = (
            Image.Resampling.NEAREST
            if s["task"] in {"semantic", "instance", "referring", "referring_multi"}
            else Image.Resampling.LANCZOS
        )
        raw_inp = self._open_pil_rgb(s["image_path"], s.get("input_tar_key", ""))
        raw_tgt = self._open_pil_rgb(s["target_path"], s.get("target_tar_key", ""))
        img, tgt_img = self._resize_and_crop_pair(raw_inp, raw_tgt, resample)
        inp = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        tgt = torch.from_numpy(np.asarray(tgt_img, np.float32) / 255.0).permute(2, 0, 1)
        prompt = s["prompt"]
        if s["task"] == "depth":
            target_cmap = self.depth_colormap
            if self.split == "train" and self.depth_colormaps:
                target_cmap = self.rng.choice(self.depth_colormaps)
            if self.depth_curve != "metric_power" or target_cmap != self.depth_colormap:
                from open_vision_banana.core.depth_mapper import DepthRGBMapper

                paper = DepthRGBMapper(colormap=self.depth_colormap)
                alt = DepthRGBMapper(
                    colormap=target_cmap,
                    curve=self.depth_curve,
                    log_alpha=self.depth_log_alpha,
                )
                metric = paper.rgb_to_depth(tgt.unsqueeze(0))
                tgt = alt.depth_to_rgb(metric).squeeze(0)
            if self.split == "train" and self.depth_colormaps and target_cmap != "cube_edge":
                prompt = (
                    f"Generate a metric depth visualization of this image using the {target_cmap} colormap."
                    if self.depth_curve == "metric_power"
                    else f"Generate a relative depth map of this image using the {target_cmap} RGB palette."
                )
        elif s["task"] == "semantic" and self.split == "train" and (
            self.semantic_prompt_styles or self.semantic_color_remap_prob > 0.0
        ):
            tgt, prompt = self._augment_semantic(tgt, prompt, s.get("mapping"))
        item = {
            "prompt": prompt,
            "image": inp * 2.0 - 1.0,
            "target_rgb": tgt,
            "task": s["task"],
        }
        if s["task"] == "referring":
            item.update({key: s[key] for key in ("expression", "encoding", "sample_id", "source_dataset") if key in s})
        if s["task"] == "instance" and "background_color" in s:
            item["background_color"] = s["background_color"]
        return item

    def _augment_semantic(
        self, tgt: torch.Tensor, prompt: str, mapping: dict | None = None
    ) -> tuple[torch.Tensor, str]:
        """Apply dynamic open-vocabulary prompt style and optional color remapping."""
        from open_vision_banana.data.openvocab_semantic import (
            NAMED_COLORS,
            build_prompt,
            sample_legend_colors,
        )

        mapping = dict(mapping) if mapping else parse_semantic_legend_from_prompt(prompt)
        if not mapping:
            return tgt, prompt

        styles = self.semantic_prompt_styles or ["json_tuple"]
        chosen_style = self.rng.choice(styles)
        do_remap = (
            self.semantic_color_remap_prob > 0.0
            and self.rng.random() < self.semantic_color_remap_prob
        )
        if chosen_style == "json_named" and not do_remap:
            named_pool = set(NAMED_COLORS.values())
            if not all(tuple(c) in named_pool for c in mapping.values()):
                do_remap = True

        if do_remap:
            chosen_style, new_colors = sample_legend_colors(
                len(mapping), self.rng, chosen_style, min_distance=64.0
            )
            quantized = (tgt.permute(1, 2, 0) * 255.0).round().to(torch.int32)
            remapped = quantized.clone()
            new_mapping: dict[str, tuple[int, int, int]] = {}
            for (cls_name, old_rgb), new_rgb in zip(mapping.items(), new_colors, strict=True):
                old_t = torch.tensor(old_rgb, dtype=torch.int32)
                mask = (quantized == old_t).all(dim=-1)
                if mask.any():
                    remapped[mask] = torch.tensor(new_rgb, dtype=torch.int32)
                new_mapping[cls_name] = new_rgb
            tgt = (remapped.float() / 255.0).permute(2, 0, 1).clamp(0.0, 1.0)
            mapping = new_mapping

        return tgt, build_prompt(chosen_style, mapping)


def _collate_fn(batch: list[dict]) -> dict:
    result = {
        "prompt": [b["prompt"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "target_rgb": torch.stack([b["target_rgb"] for b in batch]),
        "task": [b["task"] for b in batch],
    }
    if any(b["task"] == "referring" for b in batch):
        for key in ("expression", "encoding", "sample_id", "source_dataset"):
            if any(key in b for b in batch):
                result[key] = [b.get(key) for b in batch]
    if any("background_color" in b for b in batch):
        # instance rows with a recorded Fig. 3 background; rows without one default to black
        # so evaluators can index per sample. Batches without any such row keep the legacy keys.
        result["background_color"] = [tuple(b.get("background_color", (0, 0, 0))) for b in batch]
    return result


# ── Trainer ───────────────────────────────────────────────────────────────


class Trainer:
    """Unified multi-task, multi-model training trainer.

    Replaces both train_sana_512.py and train_flux.py main() functions.
    """

    @property
    def full(self) -> bool:
        """Full-parameter (FSDP2) training instead of a LoRA adapter."""
        return self.config.train_mode == "full"

    def __init__(self, config: TrainingConfig):
        config.validate()
        self.config = config
        self.dist = init_distributed()
        self.device = config.get_device()
        if self.dist.enabled and self.device.type == "cuda":
            self.device = torch.device("cuda", self.dist.local_rank)
        self.dtype = config.get_dtype()

        # Seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        logger.info("=" * 70)
        logger.info("Vision Banana — Unified Training")
        logger.info("=" * 70)
        logger.info(
            "Model: %s | Device: %s | dtype: %s", config.model_name, self.device, config.dtype
        )
        logger.info(
            "LoRA: rank=%d, alpha=%d, rsLoRA=%s", config.lora_rank, config.lora_alpha, config.rslora
        )
        logger.info(
            "Steps: %d | LR: %s | eff_batch: %d (%d rank(s))",
            config.steps,
            config.lr,
            config.batch_size * config.grad_accum * self.dist.world_size,
            self.dist.world_size,
        )
        logger.info("Output: %s", config.output_dir)

        # Data errors must fail before downloading/loading a large backbone: structural
        # errors first (missing manifest, identity replay), then the contract preflight.
        self.train_loader, self.val_loader = self._build_datasets()
        model_layout = edit_latent_layout(resolve_model_config(config.model_name))
        if model_layout == "qwen21":
            qwen_image21.check_training_config(config)
        if model_layout is not None:
            _reject_identity_generation_replay(self.train_loader.dataset, config.model_name)
        self.preflight_report = self._run_preflight()

        # ── Model backend ──────────────────────────────────────────────
        self.backend = ModelBackend(
            model_name=config.model_name,
            device=self.device,
            dtype=self.dtype,
            quantize=config.quantize,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            use_rslora=config.rslora,
            lora_targets=config.lora_targets,
            enable_gradient_checkpointing=config.gradient_checkpointing,
            train_mode=config.train_mode,
        )
        # Full-parameter mode shards the backbone (FSDP2); every rank loaded the same weights.
        if self.full:
            ensure_process_group()
            self.full_record = full_finetune.shard_backbone(self.backend.peft_model)
        self.trainable_params = [p for p in self.backend.peft_model.parameters() if p.requires_grad]
        if not self.full:
            broadcast_parameters_(self.trainable_params, self.dist)
        if self.dist.enabled:
            # Same seed above keeps LoRA init identical; from here on each rank needs its
            # own noise / timestep / CFG-dropout draws, or every rank samples the same ones.
            torch.manual_seed(config.seed + self.dist.rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed + self.dist.rank)

        # ── Losses ─────────────────────────────────────────────────────
        task_weights = config.parse_task_weights()
        self.losses = TrainingLosses(
            task_weights=task_weights,
            depth_edge_aux_weight=config.depth_edge_aux_weight,
            normal_angular_loss_weight=config.normal_angular_loss_weight,
            depth_edge_aux_gradient_weight=config.depth_edge_aux_gradient_weight,
            depth_edge_aux_edge_weight=config.depth_edge_aux_edge_weight,
            depth_edge_aux_edge_threshold=config.depth_edge_aux_edge_threshold,
            depth_edge_aux_sigma_floor=config.depth_edge_aux_sigma_floor,
            min_snr=config.min_snr,
            min_snr_gamma=config.min_snr_gamma,
            family=self.backend.family,
        )

        # ── Callbacks ──────────────────────────────────────────────────
        self.callbacks = TrainingCallbacks(
            output_dir=config.output_dir,
            wandb_enabled=config.wandb and self.dist.is_main,
            wandb_project=config.wandb_project,
            wandb_run_name=config.wandb_run_name,
            wandb_config={k: str(v) for k, v in config.__dict__.items() if not k.startswith("_")},
            log_every=config.log_every,
            save_every=config.save_every,
            eval_every=config.eval_every,
        )

        # ── EMA ────────────────────────────────────────────────────────
        self.ema = EMAModel(
            self.backend.peft_model,
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup,
        ) if config.ema_enabled else None

        # ── Datasets ───────────────────────────────────────────────────
        self.callbacks.task_prompts = uniform_task_prompts(self.train_loader.dataset)
        effective = config.to_dict()
        effective["task_sample_counts"] = {
            ds.samples[0]["task"]: len(ds)
            for ds in getattr(self.train_loader.dataset, "datasets", [self.train_loader.dataset])
        }
        effective["trainable_parameter_names"] = [
            n for n, p in self.backend.peft_model.named_parameters() if p.requires_grad
        ]
        effective["preflight"] = {
            "skipped": bool(config.skip_preflight),
            "errors": [i.message for i in self.preflight_report.errors] if self.preflight_report else [],
            "warnings": [i.message for i in self.preflight_report.warnings] if self.preflight_report else [],
        }
        effective["sampling_rank"] = self.dist.rank
        effective["sampling_world_size"] = self.dist.world_size
        effective["sampling_probabilities"] = getattr(
            self.train_loader.sampler, "task_probabilities", {
                t: n / sum(effective["task_sample_counts"].values())
                for t, n in effective["task_sample_counts"].items()
            }
        )
        effective["sampling_with_replacement"] = isinstance(
            self.train_loader.sampler, TaskWeightedSampler
        )
        if self.dist.is_main:
            (self.callbacks.output_dir / "effective_config.json").write_text(
                json.dumps(effective, indent=2), encoding="utf-8"
            )

        # ── Optimizer & Scheduler ──────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
        total_steps = config.steps
        warmup_steps = (config.warmup_steps if config.warmup_steps is not None
                        else int(total_steps * config.warmup_ratio))
        self.lr_scheduler = SchedulerFactory.create(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            scheduler_type=config.scheduler,
        )

        # ── Resume ─────────────────────────────────────────────────────
        self.start_step = self._resume(config.resume) if config.resume else 0

        logger.info("Setup complete. Starting training...")

    def _resume(self, resume: str) -> int:
        """Restore training state from a checkpoint; return the step to continue from.

        ``resume`` is a checkpoint directory or its ``lora.pt``. Every rank loads it, so
        ranks stay identical. With ``resume_weights_only`` only the LoRA weights are taken
        (the EMA shadow restarts from them) and training begins at step 0.
        Full-parameter runs resume from their sharded ``train_state`` directory (or the
        run directory holding it): weights, optimizer, EMA, schedule and step.
        """
        if self.full:
            path = Path(resume)
            if (path / "train_state").is_dir():
                path = path / "train_state"
            extra = full_finetune.load_train_state(
                path, self.backend.peft_model, self.optimizer, self.lr_scheduler, self.ema
            )
            return self._resume_progress(path, int(extra["step"]), extra)
        path = Path(resume)
        if path.is_dir():
            path = path / "lora.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location="cpu")
        lora_state = ckpt.get("lora_state_dict", ckpt)

        params = {n: p for n, p in self.backend.peft_model.named_parameters() if p.requires_grad}
        missing = sorted(params.keys() - lora_state.keys())
        unexpected = sorted(lora_state.keys() - params.keys())
        reshaped = sorted(
            n for n in params.keys() & lora_state.keys() if lora_state[n].shape != params[n].shape
        )
        if missing or unexpected or reshaped:
            raise ValueError(
                f"{path} does not match this run's LoRA (model / rank / target modules): "
                f"{len(missing)} missing, e.g. {missing[:2]}; "
                f"{len(unexpected)} unexpected, e.g. {unexpected[:2]}; "
                f"{len(reshaped)} with another shape, e.g. {reshaped[:2]}"
            )
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(lora_state[name])
        # The shadow was cloned from the fresh LoRA init; start it from the loaded weights
        # so a checkpoint without (or with a partial) EMA does not average towards the init.
        for name, shadow in (self.ema.shadow_params if self.ema else {}).items():
            shadow.copy_(params[name].data)

        if self.config.resume_weights_only:
            logger.info("Loaded LoRA weights from %s; optimizer, schedule and step reset", path)
            return 0

        step = int(ckpt.get("step", 0))
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        for name, tensor in (ckpt.get("ema_state_dict") or {}).items():
            if self.ema and name in self.ema.shadow_params:
                self.ema.shadow_params[name].copy_(tensor)
        if self.ema:
            self.ema.step = step
        return self._resume_progress(path, step, ckpt)

    def _resume_progress(self, path: Path, step: int, saved: dict) -> int:
        """Reseed for ``step`` and take the saved best score if it is still comparable."""
        # RNG state is not checkpointed. Without a new seed the resumed run would replay
        # the shuffles, noise and timesteps of the run's first steps.
        torch.manual_seed(self.config.seed + self.dist.rank + step)  # seeds every CUDA device too
        best = saved.get("best_val_loss")
        if best is not None and saved.get("selection_protocol") == self._selection_protocol():
            self.callbacks.best_val_loss = best
        elif best is not None:
            logger.warning("Reset best score: saved validation protocol differs from current run")
        logger.info("Resumed from %s at step %d (best val_loss=%s)", path, step, best)
        return step

    def _save_full(self, step: int, suffix: str = "", val_loss: float | None = None) -> None:
        """Full-parameter checkpoint (collective): ``transformer/``, metadata, then train state.

        Plain and ``_final`` checkpoints hold the online weights and refresh the
        resumable ``train_state``; ``_best`` holds the EMA weights (when EMA is on) and
        replaces the previous best directory (a full backbone is 14 GB).
        """
        model = self.backend.peft_model
        out = self.callbacks.output_dir
        ckpt_dir = out / f"checkpoint-{step}{suffix}"
        is_best = suffix == "_best"
        source = "ema" if is_best and self.ema is not None else "online"
        with self.ema.applied(model) if source == "ema" else contextlib.nullcontext():
            full_finetune.save_transformer(
                model, ckpt_dir / "transformer", self.full_record, is_main=self.dist.is_main
            )
        protocol = self._selection_protocol()
        validated = ("ema" if self.ema else "online") if val_loss is not None else None
        if self.dist.is_main:
            meta = {
                "step": step,
                "val_loss": val_loss,
                "best_val_loss": val_loss if is_best else self.callbacks.best_val_loss,
                "train_mode": "full",
                "model_name": self.config.model_name,
                "checkpoint_weight_source": "online",
                "transformer_weight_source": source,
                "validation_weight_source": validated,
                "selection_protocol": protocol,
            }
            self.callbacks.write_metadata(ckpt_dir, meta)
            if is_best:
                for old in out.glob("checkpoint-*_best"):
                    if old != ckpt_dir:
                        shutil.rmtree(old, ignore_errors=True)
            logger.info(
                "  Saved full checkpoint at step %d (%s weights) → %s", step, source, ckpt_dir
            )
        if not is_best:
            progress = {
                "best_val_loss": self.callbacks.best_val_loss,
                "selection_protocol": protocol,
            }
            full_finetune.save_train_state(
                out / "train_state",
                model,
                self.optimizer,
                self.lr_scheduler,
                self.ema,
                step,
                progress,
                is_main=self.dist.is_main,
            )

    def _ema_weights(self, model: torch.nn.Module):
        """Context with the EMA weights in ``model`` (the online ones when EMA is off)."""
        return self.ema.applied(model) if self.ema is not None else contextlib.nullcontext()

    def _validate_full(self, step: int) -> None:
        """EMA validation on every rank (the backbone is sharded), best checkpoint if improved."""
        model = self.backend.peft_model
        with self._ema_weights(model):
            try:
                val_loss = self.validate()
            finally:
                full_finetune.reshard(model)  # before the online weights return (EMA swaps shards)
        val_loss = broadcast_float(val_loss)
        self.callbacks.on_validation_loss(
            step, val_loss, lambda: self._save_full(step, "_best", val_loss)
        )
        if self.dist.is_main:
            logger.info("Validation details: %s", self.validation_metrics)

    def _selection_protocol(self) -> str:
        source = "ema" if self.ema is not None else "online"
        # Loss settings and data must agree before a saved best score is comparable.
        settings = self.config.to_dict()
        if self.config.pair_manifest and Path(self.config.pair_manifest).is_file():
            settings["manifest_sha256"] = hashlib.sha256(
                Path(self.config.pair_manifest).read_bytes()
            ).hexdigest()
        for key in ("steps", "output_dir", "resume", "resume_weights_only", "lr",
                    "save_every", "log_every", "eval_every", "wandb", "wandb_run_name"):
            settings.pop(key, None)
        fingerprint = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"macro_flow_loss_v1:{source}:{fingerprint}"

    def _build_datasets(self) -> tuple[DataLoader, DataLoader | None]:
        """Build train and val data loaders from manifest."""
        config = self.config
        data_root = Path(config.data_dir) if config.data_dir else None
        manifest_path = Path(config.pair_manifest) if config.pair_manifest else None

        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(f"Pair manifest not found: {manifest_path}")

        # Resolve tasks
        all_tasks = {"depth", "normal", "semantic"}
        tasks_to_load = set()
        if config.tasks:
            for t in config.tasks:
                tasks_to_load.add(TASK_ALIASES.get(t, t))
        else:
            tasks_to_load = all_tasks

        # Build per-task datasets
        train_datasets = [
            ManifestVisionDataset(
                manifest_path,
                config.resolution,
                "train",
                {t},
                config.max_samples_per_task,
                data_root=data_root,
                depth_curve=config.depth_curve,
                depth_log_alpha=config.depth_log_alpha,
                depth_colormaps=config.depth_colormaps,
                semantic_prompt_styles=config.semantic_prompt_styles,
                semantic_color_remap_prob=config.semantic_color_remap_prob,
                aspect_preserving_crop=config.aspect_preserving_crop,
                seed=config.seed,
            )
            for t in sorted(tasks_to_load)
        ]
        empty_tasks = [t for t, ds in zip(sorted(tasks_to_load), train_datasets, strict=True)
                       if len(ds) == 0]
        if empty_tasks:
            raise ValueError(f"Requested training tasks have no samples: {empty_tasks}")

        val_datasets = [
            ManifestVisionDataset(
                manifest_path,
                config.resolution,
                "val",
                {t},
                data_root=data_root,
                depth_curve=config.depth_curve,
                depth_log_alpha=config.depth_log_alpha,
                aspect_preserving_crop=config.aspect_preserving_crop,
                seed=config.seed,
            )
            for t in sorted(tasks_to_load)
        ]
        val_datasets = [ds for ds in val_datasets if len(ds) > 0]

        train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_ds = ConcatDataset(val_datasets) if val_datasets else None
        if val_datasets and len(val_datasets) != len(tasks_to_load):
            raise ValueError("Validation must cover every requested training task")

        # Under torchrun each rank reads its own shard; set_epoch() in train() reshuffles it.
        train_sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
                seed=config.seed,
                drop_last=True,
            )
            if self.dist.enabled
            else None
        )
        if (
            config.task_sampling != "proportional"
            or config.task_sampling_weights
            or config.vision_ratio < 1.0
        ):
            groups = {}
            offset = 0
            for task, ds in zip(sorted(tasks_to_load), train_datasets, strict=True):
                groups[task] = list(range(offset, offset + len(ds)))
                offset += len(ds)
            sampling_weights = {}
            for task, weight in config.task_sampling_weights.items():
                canonical = TASK_ALIASES.get(task, task)
                if canonical in sampling_weights:
                    raise ValueError(f"Duplicate sampling task alias: {canonical}")
                sampling_weights[canonical] = weight
            if config.vision_ratio < 1.0 and not sampling_weights and "generation" in groups:
                vision_tasks = [t for t in groups if t != "generation"]
                if vision_tasks:
                    per_vision = config.vision_ratio / len(vision_tasks)
                    for t in vision_tasks:
                        sampling_weights[t] = per_vision
                    sampling_weights["generation"] = 1.0 - config.vision_ratio
            train_sampler = TaskWeightedSampler(
                groups,
                strategy="uniform" if (config.vision_ratio < 1.0 and not config.task_sampling_weights) else config.task_sampling,
                weights=sampling_weights,
                seed=config.seed,
                rank=self.dist.rank,
                world_size=self.dist.world_size,
            )
            logger.info("Task sampling probabilities: %s", train_sampler.task_probabilities)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            collate_fn=_collate_fn,
            drop_last=True,
        )
        if not len(train_loader):
            raise ValueError("Training loader is empty after rank sharding and drop_last")
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_fn,
            )

        for ds in train_datasets:
            task_name = ds.samples[0]["task"] if ds.samples else "?"
            logger.info("  train/%s: %d samples", task_name, len(ds))
        logger.info("  Total train: %d | Val: %d", len(train_ds), len(val_ds) if val_ds else 0)

        return train_loader, val_loader

    def train(self) -> None:
        """Main training loop."""
        config = self.config
        peft_model = self.backend.peft_model
        total_steps = config.steps

        train_metrics = {
            "loss": 0.0,
            "flow_loss": 0.0,
            "depth_edge_aux_loss": 0.0,
            "normal_angular_aux_loss": 0.0,
            "steps": 0,
        }
        task_loss_acc: dict[str, list[float]] = {}
        task_sample_counts: dict[str, int] = {}
        global_step = self.start_step
        micro_step = 0
        # full passes over the train loader; seeds DistributedSampler shuffles. A resumed
        # run continues the count (from the start of the pass it stopped in).
        data_pass = global_step * config.grad_accum // max(1, len(self.train_loader))
        self.optimizer.zero_grad(set_to_none=True)

        while global_step < total_steps:
            updates_per_epoch = max(
                1, (len(self.train_loader) + config.grad_accum - 1) // config.grad_accum
            )
            epoch = global_step // updates_per_epoch
            if hasattr(self.train_loader.sampler, "set_epoch"):
                # not `epoch`: that lags the real pass count when grad_accum does not divide
                # len(train_loader), which would replay the same shuffle.
                self.train_loader.sampler.set_epoch(data_pass)
            data_pass += 1
            peft_model.train()

            for batch in self.train_loader:
                if global_step >= total_steps:
                    break
                for task in batch.get("task", []) if isinstance(batch, dict) else []:
                    task_sample_counts[task] = task_sample_counts.get(task, 0) + 1

                # ── Forward ────────────────────────────────────────────
                loss_dict = self._train_step(batch, global_step)

                # ── Accumulate metrics ─────────────────────────────────
                train_metrics["loss"] += loss_dict["total_loss"]
                train_metrics["flow_loss"] += loss_dict["flow_loss"]
                train_metrics["depth_edge_aux_loss"] += loss_dict["depth_aux"]
                train_metrics["normal_angular_aux_loss"] += loss_dict["normal_aux"]
                train_metrics["steps"] += 1

                for t, vals in loss_dict.get("task_loss_acc", {}).items():
                    task_loss_acc.setdefault(t, []).extend(vals)

                # ── Backward ───────────────────────────────────────────
                loss = loss_dict["loss"] / config.grad_accum
                loss.backward()

                micro_step += 1
                if micro_step % config.grad_accum != 0:
                    continue

                # ── Optimizer step ─────────────────────────────────────
                if not self.full:  # FSDP already reduce-scattered the gradients in backward
                    all_reduce_gradients_(self.trainable_params, self.dist)
                # The pre-clip norm decides a drop; a dropped update discards the clipped
                # gradients, so clipping first leaves both outcomes unchanged.
                grad_norm = clip_grad_norm(self.trainable_params, config.max_grad_norm)
                if drop_oversized_gradients(grad_norm, config.drop_grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.dist.is_main:
                        logger.warning(
                            "Dropped update at pending step %d: grad_norm=%.3f > drop_grad_norm=%.3f",
                            global_step,
                            grad_norm,
                            config.drop_grad_norm,
                        )
                    continue
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(peft_model)
                global_step += 1

                # ── Logging ────────────────────────────────────────────
                if self.callbacks.should_log(global_step):
                    n = max(1, train_metrics["steps"])
                    metrics = {
                        k: train_metrics[k] / n
                        for k in (
                            "loss",
                            "flow_loss",
                            "depth_edge_aux_loss",
                            "normal_angular_aux_loss",
                        )
                    }
                    lr = self.lr_scheduler.get_last_lr()[0]
                    if self.dist.is_main:
                        metrics.update({f"sample_count/{t}": n for t, n in task_sample_counts.items()})
                        count = sum(task_sample_counts.values())
                        metrics.update({
                            f"sample_ratio/{t}": n / count
                            for t, n in task_sample_counts.items()
                        })
                        self.callbacks.on_log(
                            global_step, total_steps, metrics, lr, epoch, task_loss_acc
                        )
                    train_metrics = {
                        "loss": 0.0,
                        "flow_loss": 0.0,
                        "depth_edge_aux_loss": 0.0,
                        "normal_angular_aux_loss": 0.0,
                        "steps": 0,
                    }
                    task_loss_acc = {}
                    task_sample_counts = {}

                if self.full:
                    # The backbone is sharded: every rank takes part in saving and validating.
                    if self.callbacks.should_checkpoint(global_step):
                        self._save_full(global_step)
                    if self.val_loader is not None and self.callbacks.should_validate(global_step):
                        self._validate_full(global_step)
                    continue

                # Ranks hold identical weights, so rank 0 alone saves and validates;
                # the others wait for it in the next step's gradient all-reduce.
                # ── Checkpoint ─────────────────────────────────────────
                if self.dist.is_main and self.callbacks.should_checkpoint(global_step):
                    self.callbacks.on_checkpoint(
                        global_step,
                        peft_model,
                        self.optimizer,
                        self.lr_scheduler,
                        ema=self.ema,
                        model_name=config.model_name,
                        lora_rank=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        selection_protocol=self._selection_protocol(),
                    )

                # ── Validation ─────────────────────────────────────────
                if (
                    self.dist.is_main
                    and self.val_loader is not None
                    and self.callbacks.should_validate(global_step)
                ):
                    with self._ema_weights(peft_model):
                        val_loss = self.validate()
                    self.callbacks.on_validate(
                        global_step,
                        val_loss,
                        peft_model,
                        self.optimizer,
                        self.lr_scheduler,
                        ema=self.ema,
                        model_name=config.model_name,
                        lora_rank=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                        selection_protocol=self._selection_protocol(),
                    )
                    logger.info("Validation details: %s", self.validation_metrics)

        # ── Final save ─────────────────────────────────────────────────
        if self.full:
            self._save_full(global_step, "_final")
        elif self.dist.is_main:
            self.callbacks.on_checkpoint(
                global_step,
                peft_model,
                self.optimizer,
                self.lr_scheduler,
                ema=self.ema,
                model_name=config.model_name,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                suffix="_final",
                selection_protocol=self._selection_protocol(),
            )
        if self.dist.is_main:
            self.callbacks.on_finish(global_step)
        barrier(self.dist)  # keep the process group alive until rank 0 has saved
        cleanup(self.dist)

    def _in_context_conditioning(self) -> bool:
        """Whether the backbone takes the source image as in-context tokens (Kontext-style)."""
        return self._edit_layout() == "kontext"

    def _encode_edit_text(self, prompts, images, cfg_dropout_rate: float = 0.0):
        if hasattr(self.backend, "encode_edit_text"):
            return self.backend.encode_edit_text(
                prompts, images, cfg_dropout_rate=cfg_dropout_rate
            )
        if self._edit_layout() == "hunyuan":
            return self.backend.encode_hunyuan_edit(prompts, images)
        text_cond = self.backend.encode_text(prompts, self.device, self.dtype)
        if cfg_dropout_rate > 0 and hasattr(self.backend, "apply_cfg_dropout"):
            return self.backend.apply_cfg_dropout(text_cond, cfg_dropout_rate, len(prompts))
        return text_cond

    def _run_preflight(self):
        """Manifest contract audit (review item 2b): refuse to train on contract errors.

        Runs ``validate_manifest_contract`` on the training manifest with this run's
        model, LoRA targets and depth settings. Errors abort unless
        ``skip_preflight`` is set, in which case they are still written to
        ``effective_config.json`` so the run is auditable.
        """
        config = self.config
        if not config.pair_manifest:
            return None
        from ..data.validation import validate_manifest_contract

        report = validate_manifest_contract(
            config.pair_manifest,
            data_root=config.data_dir,
            model_name=config.model_name,
            lora_targets=config.lora_targets,
            # Stored depth targets are always cube-edge metric PNGs; ``depth_colormaps`` is the
            # training-time re-encoding list and must not change what the storage contract is.
            depth_colormap="cube_edge",
            depth_curve=config.depth_curve,
        )
        for issue in report.warnings:
            logger.warning("preflight: %s", issue.message)
        if report.errors:
            summary = "\n".join(f"  - {i.message}" for i in report.errors)
            if config.skip_preflight:
                logger.error("preflight found %d error(s) but skip_preflight is set:\n%s", len(report.errors), summary)
            else:
                raise ValueError(
                    f"Manifest preflight failed with {len(report.errors)} error(s); fix the data/config or set "
                    f"training.skip_preflight (recorded in effective_config.json):\n{summary}"
                )
        return report

    def _img2img_mask(self, batch_tasks: list[str], edit_layout: str | None, device) -> torch.Tensor:
        """Per-sample flow source: True = start from input latents, False = from noise.

        Shared by training and validation so both see the same objective.
        """
        if edit_layout is not None or not self.config.img2img_flow:
            return torch.zeros(len(batch_tasks), dtype=torch.bool, device=device)
        return torch.tensor([t != "generation" for t in batch_tasks], dtype=torch.bool, device=device)

    def _edit_layout(self) -> str | None:
        """Source-image layout of the training forward, or None for a text-to-image backbone."""
        from .edit_forward import edit_latent_layout

        model_config = getattr(self.backend, "model_config", None)
        if model_config is None:
            return None
        return edit_latent_layout(model_config)

    def _train_step(self, batch: dict, global_step: int) -> dict:
        """Execute a single training step (forward + loss)."""
        config = self.config
        prompts = batch["prompt"]
        batch_size = len(prompts)
        target_rgb = batch["target_rgb"].to(device=self.device, dtype=self.dtype)
        input_images = batch["image"].to(device=self.device, dtype=self.dtype)

        # ── VAE encode ─────────────────────────────────────────────────
        with torch.no_grad():
            target_pixels = target_rgb * 2.0 - 1.0  # [0,1] → [-1,1]
            target_latents = self.backend.encode_image(target_pixels)
            input_pixels = input_images  # already [-1,1]
            input_latents = self.backend.encode_image(input_pixels)
            if input_latents.shape != target_latents.shape:
                input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

        # ── Flow matching ──────────────────────────────────────────────
        edit_layout = self._edit_layout()
        batch_tasks = batch.get("task", ["depth"] * batch_size)
        flow_batch = self.backend.prepare_flow_batch(
            target_latents,
            input_latents,
            batch_size,
            # Per sample: editors see the source as conditioning and generate the target
            # from noise; T2I generation-replay rows always start from Gaussian noise
            # (their input is the target image itself); vision rows on a T2I backbone
            # start from the input latents when img2img_flow is on.
            img2img_flow=self._img2img_mask(batch_tasks, edit_layout, self.device),
            flow_sigma=config.flow_sigma,
            logit_normal=config.logit_normal,
            flow_shift=config.flow_shift,
        )
        timesteps = flow_batch["timesteps"]
        noisy_latents = flow_batch["noisy_latents"]
        velocity_target = flow_batch["velocity_target"]

        # ── Text encoding ──────────────────────────────────────────────
        with torch.no_grad():
            text_cond = self._encode_edit_text(
                prompts, input_images, cfg_dropout_rate=config.cfg_dropout_rate
            )

        # ── Forward pass ───────────────────────────────────────────────
        predicted = self.backend.forward(
            noisy_latents,
            timesteps,
            text_cond,
            batch_size,
            # Only passed for editors so text-to-image backends are called exactly as before.
            **({"context_latents": input_latents} if edit_layout else {}),
        )

        # ── Loss computation ───────────────────────────────────────────
        batch_tasks = batch.get("task", ["depth"] * batch_size)
        flow_loss, per_sample_loss, task_loss_acc = self.losses.compute_flow_loss(
            predicted,
            velocity_target,
            timesteps,
            batch_tasks,
            self.device,
        )

        depth_aux_loss, normal_aux_loss = self.losses.compute_aux_losses(
            predicted,
            noisy_latents,
            target_latents,
            target_rgb,
            timesteps,
            batch_tasks,
            vae_decode_fn=self.backend.decode_latents,
            device=self.device,
        )

        total = self.losses.total_loss(flow_loss, depth_aux_loss, normal_aux_loss)

        if not torch.isfinite(total).all():
            raise FloatingPointError(f"Non-finite loss at step {global_step}")

        return {
            "loss": total,
            "total_loss": total.item(),
            "flow_loss": flow_loss.item(),
            "depth_aux": depth_aux_loss.item(),
            "normal_aux": normal_aux_loss.item(),
            "task_loss_acc": task_loss_acc,
        }

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss.

        Every call draws the same noise and timesteps, so losses are comparable across
        evaluations and best-checkpoint selection does not depend on the draw. The training
        RNG stream is left as it was.
        """
        model = self.backend.peft_model
        was_training = model.training
        try:
            with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                torch.manual_seed(self.config.seed)
                return self._validate()
        finally:
            model.train(was_training)

    def _validation_batches(self):
        """Deterministic per-task sample budget, independent of concatenation order."""
        counts = {}
        budget = getattr(self.config, "val_max_samples_per_task", 40)
        loader = self.val_loader
        if isinstance(loader, DataLoader):
            dataset = loader.dataset
            indices = []
            offset = 0
            for part in getattr(dataset, "datasets", [dataset]):
                if not hasattr(part, "samples"):
                    raise ValueError("Validation dataset needs task metadata for bounded reads")
                for index, sample in enumerate(part.samples):
                    task = sample["task"]
                    if counts.get(task, 0) < budget:
                        indices.append(offset + index)
                        counts[task] = counts.get(task, 0) + 1
                offset += len(part)
            loader = DataLoader(
                Subset(dataset, indices), batch_size=loader.batch_size,
                shuffle=False, num_workers=0, collate_fn=loader.collate_fn,
            )
            counts = {}
        for batch in loader:
            tasks = batch.get("task", ["depth"] * len(batch["prompt"]))
            for task in sorted(set(tasks)):
                available = budget - counts.get(task, 0)
                indices = [i for i, t in enumerate(tasks) if t == task][:max(0, available)]
                if not indices:
                    continue
                counts[task] = counts.get(task, 0) + len(indices)
                yield {
                    key: value[indices] if isinstance(value, torch.Tensor)
                    else [value[i] for i in indices]
                    for key, value in batch.items()
                }

    def _validate(self) -> float:
        peft_model = self.backend.peft_model
        peft_model.eval()

        totals = {}
        counts = {}
        for batch in self._validation_batches():

            prompts = batch["prompt"]
            batch_size = len(prompts)
            target_rgb = batch["target_rgb"].to(device=self.device, dtype=self.dtype)
            input_images = batch["image"].to(device=self.device, dtype=self.dtype)

            # VAE encode
            target_pixels = target_rgb * 2.0 - 1.0
            target_latents = self.backend.encode_image(target_pixels)
            input_latents = self.backend.encode_image(input_images)
            if input_latents.shape != target_latents.shape:
                input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

            # Flow matching (same per-sample source rule as training)
            edit_layout = self._edit_layout()
            flow_batch = self.backend.prepare_flow_batch(
                target_latents,
                input_latents,
                batch_size,
                img2img_flow=self._img2img_mask(
                    batch.get("task", ["depth"] * batch_size), edit_layout, self.device
                ),
                flow_sigma=self.config.flow_sigma,
                logit_normal=self.config.logit_normal,
                flow_shift=self.config.flow_shift,
            )

            # Text encoding
            text_cond = self.backend.encode_edit_text(prompts, input_images)

            # Forward
            predicted = self.backend.forward(
                flow_batch["noisy_latents"],
                flow_batch["timesteps"],
                text_cond,
                batch_size,
                **({"context_latents": input_latents} if edit_layout else {}),
            )

            # Loss
            batch_tasks = batch.get("task", ["depth"] * batch_size)
            flow_loss, _, _ = self.losses.compute_flow_loss(
                predicted,
                flow_batch["velocity_target"],
                flow_batch["timesteps"],
                batch_tasks,
                self.device,
            )
            if not torch.isfinite(flow_loss):
                raise FloatingPointError("Non-finite validation flow loss")
            task = batch_tasks[0]
            totals[task] = totals.get(task, 0.0) + flow_loss.item() * batch_size
            counts[task] = counts.get(task, 0) + batch_size

        if not counts:
            raise ValueError("Validation has no samples")
        per_task = {task: totals[task] / counts[task] for task in counts}
        self.validation_metrics = {
            "per_task": per_task, "sample_counts": counts,
            "protocol": "macro_flow_loss_v1",
        }
        return sum(per_task.values()) / len(per_task)
