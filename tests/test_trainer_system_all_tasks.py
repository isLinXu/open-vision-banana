"""System test: the real Trainer over a six-task manifest with a fake (CPU, tiny) backend.

Builds the six-task manifest with the same pipeline as test_pipeline_smoke_all_tasks
(prepare/build scripts -> merge -> audit) and then constructs the *real* Trainer
with ModelBackend swapped for a small CPU stand-in that honours the interface the
Trainer uses (encode_image, prepare_flow_batch, encode_edit_text/encode_text,
apply_cfg_dropout, forward, decode_latents, peft_model). It runs real training
steps (dataset build, TaskWeightedSampler with vision_ratio, collate, flow batch,
loss, optimizer, EMA), validation and a checkpoint, and asserts every task was
seen with the contracts the tasks require. It also audits ManifestVisionDataset
behaviours per task (NEAREST for discrete targets, semantic restyle on
natural/hex rows, aspect-preserving crop sync, T2I identity row for generation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))  # share fixtures with the smoke test

import pytest
import torch
from torch import nn

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyarrow")
pytest.importorskip("pycocotools")
pytest.importorskip("ijson")

from test_pipeline_smoke_all_tasks import RES, _coco, _hypersim_scene, _omniedit, _tartanair  # noqa: E402

import open_vision_banana.training.trainer as trainer_module  # noqa: E402
from open_vision_banana.data.openvocab_semantic import parse_prompt_mapping  # noqa: E402
from open_vision_banana.model_registry import get_model_config  # noqa: E402
from open_vision_banana.res_manifest import build_res_manifest  # noqa: E402
from open_vision_banana.training.config import TrainingConfig  # noqa: E402
from open_vision_banana.training.flow_matching import FlowScheduler  # noqa: E402
from open_vision_banana.training.model_backend import ModelBackend  # noqa: E402
from open_vision_banana.training.trainer import ManifestVisionDataset, Trainer  # noqa: E402
from scripts.build_generation_replay_manifest import build_generation_replay_manifest  # noqa: E402
from scripts.build_instance_manifest import build_instance_manifest  # noqa: E402
from scripts.build_qwen_baseline_manifest import build_qwen_baseline_manifest  # noqa: E402
from scripts.merge_multitask_manifest import merge_manifests  # noqa: E402
from scripts.prepare_hypersim import prepare_hypersim  # noqa: E402
from scripts.prepare_tartanair import prepare_tartanair  # noqa: E402

LATENT_C = 4


class _FakePeftModel(nn.Sequential):
    """Tiny conv net exposing the PEFT ``save_pretrained`` the checkpoint callback calls."""

    def __init__(self) -> None:
        super().__init__(nn.Conv2d(LATENT_C + 8, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, LATENT_C, 3, padding=1))

    def save_pretrained(self, directory, **_: object) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(directory) / "adapter_model.bin")


class FakeBackend:
    """CPU stand-in for ModelBackend: 8x 'VAE', text-conditioned conv 'transformer'."""

    def __init__(self, model_name: str, **_: object) -> None:
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.family = self.model_config.family
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.flow_scheduler = FlowScheduler()
        torch.manual_seed(0)
        self.peft_model = _FakePeftModel()
        self.seen_prompts: list[str] = []
        self.seen_tasks: list[str] = []
        self.forward_calls = 0
        self.img2img_masks: list[torch.Tensor] = []

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(images, 8)
        return torch.cat([pooled, pooled[:, :1]], dim=1)  # [B, 4, H/8, W/8]

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(latents[:, :3], scale_factor=8, mode="nearest")

    def prepare_flow_batch(self, target_latents, input_latents, batch_size, img2img_flow=True, flow_sigma=0.0, logit_normal=True, flow_shift=1.0):
        # Same per-sample contract as ModelBackend.prepare_flow_batch (bool or [B] mask).
        out = ModelBackend.prepare_flow_batch(
            self, target_latents, input_latents, batch_size, img2img_flow=img2img_flow,
            flow_sigma=flow_sigma, logit_normal=logit_normal, flow_shift=flow_shift,
        )
        self.img2img_masks.append(out["img2img_mask"].clone())
        return out

    def encode_text(self, prompts, device, dtype):
        self.seen_prompts.extend(prompts)
        emb = torch.stack([torch.tensor([float(ord(c)) for c in p[:8].ljust(8)]) / 128.0 for p in prompts])
        return {"encoder_hidden_states": emb}

    def encode_edit_text(self, prompts, images, cfg_dropout_rate=0.0):
        return self.encode_text(prompts, self.device, self.dtype)

    def apply_cfg_dropout(self, cond, rate, n):
        return cond

    def forward(self, latents, timesteps, text_cond, batch_size, guidance=None, context_latents=None):
        self.forward_calls += 1
        emb = text_cond["encoder_hidden_states"].view(batch_size, 8, 1, 1).expand(-1, -1, *latents.shape[-2:])
        return self.peft_model(torch.cat([latents, emb], dim=1))


def _six_task_manifest(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    _hypersim_scene(raw)
    _tartanair(raw)
    coco_images, coco_instances, refer_root = _coco(raw)
    shards = _omniedit(raw)
    prep = tmp_path / "prep"
    prepare_hypersim(raw / "hypersim", prep / "hypersim", tasks=("depth", "normal", "semantic", "instance"), semantic_variants=3)
    prepare_tartanair(raw / "tartanair", prep / "tartanair", resolution=RES)
    build_instance_manifest(coco_instances, coco_images, prep / "inst" / "targets", prep / "inst" / "instance.jsonl",
                            data_root=prep / "inst", exclude_image_ids=[refer_root / "refcocog" / "refs(umd).p"])
    build_res_manifest(refer_root, raw / "coco2017" / "train2014", prep / "res")
    build_generation_replay_manifest(shards, prep / "replay" / "omniedit.jsonl", kind="edit", preset="omniedit",
                                     data_root=prep / "replay", images_dir=prep / "replay" / "images")
    merged = tmp_path / "merged" / "pairs.jsonl"
    merge_manifests([prep / "hypersim" / "pairs.jsonl", prep / "tartanair" / "pairs.jsonl", prep / "inst" / "instance.jsonl",
                     prep / "res" / "pairs.jsonl", prep / "replay" / "omniedit.jsonl"], merged, val_percent=34)
    final = tmp_path / "final" / "pairs.jsonl"
    build_qwen_baseline_manifest(merged, final, data_root=merged.parent, min_cityscapes_share=0.0,
                                 exclude_image_ids=[refer_root / "refcocog" / "refs(umd).p"])
    return final


def test_real_trainer_runs_all_six_tasks_with_a_fake_backend(tmp_path: Path, monkeypatch):
    manifest = _six_task_manifest(tmp_path)
    backends: list[FakeBackend] = []

    def make_backend(**kwargs):
        backend = FakeBackend(**kwargs)
        backends.append(backend)
        return backend

    monkeypatch.setattr(trainer_module, "ModelBackend", make_backend)
    mask_calls: list[tuple[list[str], list[bool]]] = []
    original_mask = Trainer._img2img_mask

    def recording_mask(self, batch_tasks, edit_layout, device):
        mask = original_mask(self, batch_tasks, edit_layout, device)
        mask_calls.append((list(batch_tasks), mask.tolist()))
        return mask

    monkeypatch.setattr(Trainer, "_img2img_mask", recording_mask)
    config = TrainingConfig(
        model_name="sana-1600m",  # text-to-image layout: T2I generation rows are allowed
        device="cpu", dtype="float32", resolution=RES, batch_size=2, grad_accum=1, steps=12, lr=1e-3,
        output_dir=str(tmp_path / "out"), data_dir=str(manifest.parent), pair_manifest=str(manifest),
        tasks=["depth", "normal", "semantic", "instance", "referring", "generation"],
        vision_ratio=0.5, task_sampling="uniform",
        depth_colormaps=["cube_edge", "viridis"], semantic_prompt_styles=["json_named", "json_tuple", "natural", "hex"],
        semantic_color_remap_prob=0.5, aspect_preserving_crop=True,
        num_workers=0, save_every=6, eval_every=6, log_every=1, warmup_steps=2, ema_enabled=True, ema_warmup=1,
        min_snr=False, cfg_dropout_rate=0.1, wandb=False,
    )
    config.validate()
    trainer = Trainer(config)
    backend = backends[0]

    # dataset build: every task has train and val rows, sampler weights follow vision_ratio
    assert sorted(trainer.train_loader.sampler.task_probabilities) == sorted(config.tasks)
    probs = trainer.train_loader.sampler.task_probabilities
    assert probs["generation"] == pytest.approx(0.5, abs=1e-6)
    assert all(probs[t] == pytest.approx(0.1, abs=1e-6) for t in config.tasks if t != "generation")
    assert trainer.val_loader is not None

    effective = json.loads((tmp_path / "out" / "effective_config.json").read_text())
    assert effective["vision_ratio"] == 0.5 and set(effective["task_sample_counts"]) >= set(config.tasks)

    trainer.train()

    # all six tasks were trained on and the prompts the model saw carry their contracts
    assert backend.forward_calls >= 12
    # review item 6a: on a T2I backbone (sana) with img2img_flow=True, generation rows must start
    # from noise *per sample*; with batch 2 and vision_ratio 0.5 some batches are mixed.
    assert mask_calls and len(backend.img2img_masks) == len(mask_calls)  # training + validation batches
    for tasks_in_batch, mask in mask_calls:
        assert mask == [t != "generation" for t in tasks_in_batch], (tasks_in_batch, mask)
    assert any(len(set(t)) > 1 for t, _ in mask_calls), "fixture should produce at least one mixed batch"
    prompts = backend.seen_prompts
    assert any("metric depth" in p for p in prompts)
    assert any("surface normal" in p for p in prompts)
    assert any("instance segmentation" in p for p in prompts)
    assert any(p.startswith("This image shows segmentation masks from the given image.") for p in prompts)
    assert any(p == "make the sky purple" for p in prompts)
    semantic_prompts = [p for p in prompts if parse_prompt_mapping(p)]
    assert semantic_prompts and any(("wall" in p or "chair" in p) for p in semantic_prompts)
    # depth colormap augmentation reached the prompt
    assert any("viridis" in p for p in prompts)
    # loss was finite and a checkpoint + metadata were produced
    checkpoints = sorted((tmp_path / "out").glob("checkpoint-*"))
    assert checkpoints, list((tmp_path / "out").iterdir())
    with_meta = next(c for c in checkpoints if (c / "metadata.json").exists())
    metadata = json.loads((with_meta / "metadata.json").read_text())
    assert metadata["step"] >= 6
    assert metadata.get("val_loss") is None or torch.isfinite(torch.tensor(float(metadata["val_loss"])))


def test_manifest_dataset_per_task_behaviours(tmp_path: Path):
    manifest = _six_task_manifest(tmp_path)
    root = manifest.parent

    # 1. discrete targets keep exact palette colours through resize (NEAREST), continuous ones do not need to
    for task in ("semantic", "instance", "referring"):
        ds = ManifestVisionDataset(manifest, 24, "train", {task}, data_root=root)  # 24 forces resampling of 32/48/64 px sources
        sample = ds[0]
        colors = {tuple((sample["target_rgb"][:, y, x] * 255).round().int().tolist()) for y in range(24) for x in range(24)}
        assert len(colors) <= 12, (task, len(colors))  # no interpolated intermediate colours

    # 2. semantic restyle + remap now reaches natural/hex rows too (mapping comes from the manifest row)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    sem_rows = [r for r in rows if r["task"] == "semantic"]
    assert any(r.get("style") in ("natural", "hex") for r in sem_rows), "fixture should include non-tuple styles"
    ds = ManifestVisionDataset(manifest, RES, "train", {"semantic"}, data_root=root,
                               semantic_prompt_styles=["json_tuple"], semantic_color_remap_prob=1.0, seed=3)
    for index, sample_meta in enumerate(ds.samples):
        if parse_prompt_mapping(sample_meta["prompt"]) and ("hex" in sample_meta["prompt"] or "#" in sample_meta["prompt"] or "represented by (" in sample_meta["prompt"]):
            sample = ds[index]
            new_mapping = parse_prompt_mapping(sample["prompt"])
            assert new_mapping and "<" in sample["prompt"], sample["prompt"]  # restyled to json_tuple
            # target pixels follow the new legend
            px = tuple((sample["target_rgb"][:, 2, 2] * 255).round().int().tolist())
            assert px in set(new_mapping.values()), (px, new_mapping)
            break
    else:
        pytest.fail("no natural/hex semantic row was restyled")

    # 3. aspect-preserving crop keeps the pair in sync (input and target cropped identically)
    ds = ManifestVisionDataset(manifest, RES, "val", {"instance"}, data_root=root, aspect_preserving_crop=True)
    sample = ds[0]
    assert sample["image"].shape == (3, RES, RES) and sample["target_rgb"].shape == (3, RES, RES)

    # 4. generation rows: an edit pair has distinct input/target; the trainer's identity guard is for editors
    gen = ManifestVisionDataset(manifest, RES, "train", {"generation"}, data_root=root)
    sample = gen[0]
    assert float((sample["image"] * 0.5 + 0.5 - sample["target_rgb"]).abs().mean()) > 0.1
    with pytest.raises(ValueError, match="instruction editor"):
        trainer_module._reject_identity_generation_replay(
            SimpleNamespace(datasets=[SimpleNamespace(samples=[{"task": "generation", "image_path": "a", "target_path": "a"}])]),
            "qwen-image-edit-2509",
        )

    # 5. val split carries no augmentation: a val depth row stays cube_edge with the metric prompt
    val_depth = ManifestVisionDataset(manifest, RES, "val", {"depth"}, data_root=root, depth_colormaps=["viridis"])
    assert "viridis" not in val_depth[0]["prompt"] and "metric depth" in val_depth[0]["prompt"]
