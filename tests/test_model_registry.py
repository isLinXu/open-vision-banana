#!/usr/bin/env python3
"""Tests for open_vision_banana.model_registry.

Covers:
- ModelConfig dataclass and defaults
- Lazy registry instantiation and public API
- PipelineComponents helper property
- Text encoding helpers (SD1.5, SDXL, FLUX, SD3, HunyuanDiT, Kolors, etc.)
- Backbone forward dispatch for all model families
- build_added_cond_kwargs, vae_encode/decode, prepare_position_ids
- auto_select_model memory-budget logic
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from open_vision_banana.model_registry import (
    _REGISTRY_FACTORIES,
    REGISTRY,
    ModelConfig,
    PipelineComponents,
    apply_lora,
    auto_select_model,
    backbone_forward,
    build_added_cond_kwargs,
    encode_text,
    get_model_config,
    get_models_by_family,
    list_models,
    prepare_position_ids,
    unet_forward,
    vae_decode,
    vae_encode,
)


class TestModelConfig:
    """ModelConfig dataclass creation and field coverage."""

    def test_default_creation(self):
        cfg = ModelConfig(name="test", model_id="test/model", family="test", num_text_encoders=1)
        assert cfg.name == "test"
        assert cfg.model_id == "test/model"
        assert cfg.family == "test"
        assert cfg.num_text_encoders == 1
        assert cfg.backbone_type == "unet"
        assert cfg.default_lora_rank == 16
        assert cfg.default_lora_alpha == 32
        assert cfg.default_resolution == 256
        assert cfg.memory_budget_gb == 7.0
        assert cfg.vae_scaling_factor == 0.18215
        assert cfg.vae_downsample_factor == 8
        assert cfg.cross_attention_dim == 768
        assert cfg.text_encoder_max_length == 77
        assert cfg.pipeline_class == "StableDiffusionXLPipeline"
        assert cfg.default_guidance_scale == 2.0
        assert cfg.default_num_inference_steps == 30
        assert not cfg.needs_added_cond_kwargs
        assert cfg.vae_has_latent_dist
        assert cfg.description == ""

    def test_custom_values(self):
        cfg = ModelConfig(
            name="custom",
            model_id="custom/model",
            family="sdxl",
            num_text_encoders=2,
            backbone_type="transformer",
            cross_attention_dim=2048,
            needs_added_cond_kwargs=True,
            vae_scaling_factor=0.3611,
            vae_downsample_factor=16,
            memory_budget_gb=24.0,
            pipeline_class="FluxPipeline",
            default_resolution=512,
            default_guidance_scale=3.5,
            default_lora_rank=32,
            default_lora_alpha=64,
            needs_guidance_in_forward=True,
            needs_attention_mask=True,
            needs_position_ids=True,
            needs_size_kwargs=True,
            needs_pooled_projections=True,
            needs_image_meta_size=True,
            needs_t5_and_clip=True,
            needs_added_cond_kwargs_dit=True,
            vae_has_latent_dist=False,
            description="test desc",
        )
        assert cfg.backbone_type == "transformer"
        assert cfg.cross_attention_dim == 2048
        assert cfg.needs_added_cond_kwargs
        assert cfg.vae_scaling_factor == 0.3611
        assert cfg.memory_budget_gb == 24.0
        assert cfg.default_lora_rank == 32
        assert cfg.needs_guidance_in_forward
        assert cfg.needs_attention_mask
        assert cfg.needs_position_ids
        assert cfg.needs_size_kwargs
        assert cfg.needs_pooled_projections
        assert cfg.needs_image_meta_size
        assert cfg.needs_t5_and_clip
        assert cfg.needs_added_cond_kwargs_dit
        assert not cfg.vae_has_latent_dist
        assert cfg.description == "test desc"

    def test_lora_target_modules_default(self):
        cfg = ModelConfig(name="t", model_id="t", family="t", num_text_encoders=1)
        assert "to_q" in cfg.lora_target_modules
        assert "ff.net.0.proj" in cfg.lora_target_modules


class TestLazyRegistry:
    """_LazyRegistry lazy-instantiation behaviour."""

    def test_lazy_instantiation_on_getitem(self):
        """Accessing a key should instantiate the ModelConfig once."""
        # Use the real registry with a known key
        obj = REGISTRY["sd15"]
        assert isinstance(obj, ModelConfig)
        assert obj.name == "sd15"
        # Second access should return cached instance
        assert REGISTRY["sd15"] is obj

    def test_contains(self):
        assert "sd15" in REGISTRY
        assert "nonexistent" not in REGISTRY

    def test_keys(self):
        keys = list(REGISTRY.keys())
        assert "sd15" in keys
        assert "sdxl" in keys
        assert "flux-dev" in keys

    def test_values(self):
        values = list(REGISTRY.values())
        assert all(isinstance(v, ModelConfig) for v in values)
        names = {v.name for v in values}
        assert "sd15" in names

    def test_items(self):
        items = list(REGISTRY.items())
        assert all(isinstance(k, str) and isinstance(v, ModelConfig) for k, v in items)

    def test_len(self):
        assert len(REGISTRY) == len(_REGISTRY_FACTORIES)

    def test_get_default(self):
        assert REGISTRY.get("nonexistent") is None
        assert REGISTRY.get("nonexistent", "fallback") == "fallback"
        assert isinstance(REGISTRY.get("sd15"), ModelConfig)

    def test_iter(self):
        assert "sd15" in set(REGISTRY)


class TestPublicAPI:
    """get_model_config, list_models, get_models_by_family, auto_select_model."""

    def test_get_model_config_known(self):
        cfg = get_model_config("sd15")
        assert cfg.name == "sd15"
        assert cfg.family == "sd15"
        assert cfg.num_text_encoders == 1

    def test_get_model_config_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_config("not-a-model")

    def test_list_models(self):
        models = list_models()
        assert "sd15" in models
        assert "sdxl" in models
        assert "flux-dev" in models
        assert models == sorted(models)

    def test_get_models_by_family(self):
        sdxl_models = get_models_by_family("sdxl")
        assert "sdxl" in sdxl_models
        assert "ssd-1b" in sdxl_models
        assert "sdxl-turbo" in sdxl_models

        flux_models = get_models_by_family("flux")
        assert "flux-dev" in flux_models
        assert "flux-schnell" in flux_models

    def test_auto_select_model(self):
        assert auto_select_model(24.0) == "sdxl"
        assert auto_select_model(7.0) == "sdxl"
        assert auto_select_model(6.0) == "ssd-1b"  # 6.0 >= 5.5, ssd-1b checked before sana-600m
        assert auto_select_model(5.5) == "ssd-1b"
        assert auto_select_model(5.0) == "sd2"
        assert auto_select_model(4.0) == "sd15"
        assert auto_select_model(3.0) == "sd15"

    def test_new_models_registered(self):
        """Verify 5 new model families are in the registry."""
        models = list_models()
        for name in ("longcat-image", "lumina", "nucleusmoe-image", "ernie-image", "hidream-image"):
            assert name in models, f"{name} not found in registry"

    def test_new_model_configs(self):
        """Verify ModelConfig fields for new models."""
        for name, family, backbone_type in [
            ("longcat-image", "longcat-image", "transformer"),
            ("lumina", "lumina", "transformer"),
            ("nucleusmoe-image", "nucleusmoe-image", "transformer"),
            ("ernie-image", "ernie-image", "transformer"),
            ("hidream-image", "hidream-image", "transformer"),
        ]:
            cfg = get_model_config(name)
            assert cfg.family == family
            assert cfg.backbone_type == backbone_type
            assert cfg.pipeline_class.endswith("Pipeline")

    def test_resolve_new_model_paths(self):
        """Verify path resolution for new model families."""
        from open_vision_banana.model_registry import resolve_model_config
        for basename, expected_family in [
            ("LongCat-Image", "longcat-image"),
            ("Lumina-Next-SFT-diffusers", "lumina"),
            ("NucleusMoE-Image", "nucleusmoe-image"),
            ("Ernie-Image", "ernie-image"),
            ("HiDream-I1-Full", "hidream-image"),
        ]:
            cfg = resolve_model_config(f"/fake/path/{basename}")
            assert cfg.family == expected_family, (
                f"Path '{basename}' resolved to '{cfg.family}', expected '{expected_family}'"
            )


class TestPipelineComponents:
    """PipelineComponents dataclass and transformer alias."""

    def test_transformer_alias(self):
        mock_backbone = MagicMock()
        comps = PipelineComponents(
            unet=mock_backbone,
            vae=MagicMock(),
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
        )
        assert comps.transformer is mock_backbone


class TestBuildAddedCondKwargs:
    """build_added_cond_kwargs for SDXL / Kolors."""

    def test_none_when_not_needed(self):
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1)
        assert build_added_cond_kwargs(cfg, None, (256, 256), torch.device("cpu"), torch.float32) is None

    def test_none_when_pooled_embeds_missing(self):
        cfg = ModelConfig(name="t", model_id="t", family="sdxl", num_text_encoders=2, needs_added_cond_kwargs=True)
        assert build_added_cond_kwargs(cfg, None, (256, 256), torch.device("cpu"), torch.float32) is None

    def test_builds_correct_dict(self):
        cfg = ModelConfig(name="t", model_id="t", family="sdxl", num_text_encoders=2, needs_added_cond_kwargs=True)
        pooled = torch.randn(2, 1280)
        result = build_added_cond_kwargs(cfg, pooled, (512, 768), torch.device("cpu"), torch.float32, batch_size=2)
        assert result is not None
        assert "text_embeds" in result
        assert "time_ids" in result
        assert result["time_ids"].shape == (2, 6)
        assert torch.allclose(result["time_ids"][0], torch.tensor([512, 768, 0, 0, 512, 768], dtype=torch.float32))


class TestVaeHelpers:
    """vae_encode and vae_decode."""

    def test_vae_encode_with_scaling(self):
        vae = MagicMock()
        vae.encode.return_value.latent_dist.mode.return_value = torch.ones(1, 4, 32, 32)
        vae.config = MagicMock(spec=[])  # no attributes → getattr returns default
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1, vae_scaling_factor=0.5)
        latents = vae_encode(vae, torch.zeros(1, 3, 256, 256), cfg)
        assert torch.allclose(latents, torch.ones(1, 4, 32, 32) * 0.5)

    def test_vae_encode_without_scaling(self):
        vae = MagicMock()
        vae.encode.return_value.latent_dist.mode.return_value = torch.ones(1, 4, 32, 32)
        vae.config = MagicMock(spec=["scaling_factor"])
        vae.config.scaling_factor = 0.25
        cfg = ModelConfig(name="t", model_id="t", family="sana", num_text_encoders=1, vae_scaling_factor=None)
        latents = vae_encode(vae, torch.zeros(1, 3, 256, 256), cfg)
        assert torch.allclose(latents, torch.ones(1, 4, 32, 32) * 0.25)

    def test_vae_encode_no_latent_dist(self):
        vae = MagicMock()
        vae.encode.return_value = MagicMock(spec=[])
        vae.encode.return_value.sample = torch.ones(1, 4, 32, 32)
        vae.config = MagicMock(spec=["scaling_factor"])
        vae.config.scaling_factor = 1.0
        cfg = ModelConfig(name="t", model_id="t", family="sana", num_text_encoders=1, vae_has_latent_dist=False, vae_scaling_factor=None)
        latents = vae_encode(vae, torch.zeros(1, 3, 256, 256), cfg)
        assert isinstance(latents, torch.Tensor)
        assert torch.allclose(latents, torch.ones(1, 4, 32, 32))

    def test_vae_encode_with_shift(self):
        """Z-Image/FLUX VAE: latents = (encoded - shift) * scaling"""
        vae = MagicMock()
        vae.encode.return_value.latent_dist.mode.return_value = torch.ones(1, 4, 32, 32)
        vae.config = MagicMock(spec=[])
        cfg = ModelConfig(
            name="t", model_id="t", family="z-image", num_text_encoders=1,
            vae_scaling_factor=0.3611, vae_shift_factor=0.1159,
        )
        latents = vae_encode(vae, torch.zeros(1, 3, 256, 256), cfg)
        expected = (torch.ones(1, 4, 32, 32) - 0.1159) * 0.3611
        assert torch.allclose(latents, expected, atol=1e-6)

    def test_vae_decode_with_scaling(self):
        vae = MagicMock()
        vae.decode.return_value = (torch.ones(1, 3, 256, 256),)
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1, vae_scaling_factor=0.5)
        decoded = vae_decode(vae, torch.ones(1, 4, 32, 32), cfg)
        assert torch.allclose(decoded, torch.ones(1, 3, 256, 256))
        vae.decode.assert_called_once()

    def test_vae_decode_with_shift(self):
        vae = MagicMock()
        vae.decode.return_value = (torch.ones(1, 3, 256, 256),)
        vae.config.scaling_factor = 0.5
        vae.config.shift_factor = 0.1
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1, vae_scaling_factor=None)
        decoded = vae_decode(vae, torch.ones(1, 4, 32, 32), cfg)
        assert torch.allclose(decoded, torch.ones(1, 3, 256, 256))


class TestPreparePositionIds:
    """prepare_position_ids for FLUX / Chroma / FLUX.2."""

    def test_returns_none_when_not_needed(self):
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1)
        txt_ids, img_ids = prepare_position_ids(
            torch.randn(1, 4, 32, 32),
            torch.randn(1, 77, 768),
            cfg,
            torch.device("cpu"),
            torch.float32,
        )
        assert txt_ids is None
        assert img_ids is None

    def test_flux_position_ids(self):
        cfg = ModelConfig(name="t", model_id="t", family="flux", num_text_encoders=2, needs_position_ids=True)
        latents = torch.randn(1, 4096, 64)  # 3D packed
        enc = torch.randn(1, 512, 4096)
        txt_ids, img_ids = prepare_position_ids(latents, enc, cfg, torch.device("cpu"), torch.float32)
        assert txt_ids is not None
        assert img_ids is not None
        assert txt_ids.shape == (1, 512, 3)
        assert img_ids.shape == (1, 4096, 3)

    def test_flux2_position_ids(self):
        cfg = ModelConfig(name="t", model_id="t", family="flux2", num_text_encoders=1, needs_position_ids=True)
        latents = torch.randn(1, 4, 32, 32)
        enc = torch.randn(1, 512, 4096)
        txt_ids, img_ids = prepare_position_ids(latents, enc, cfg, torch.device("cpu"), torch.float32)
        assert txt_ids is not None
        assert img_ids is not None
        assert txt_ids.shape == (1, 512, 4)
        assert img_ids.shape == (1, 1024, 4)  # 32*32 = 1024

    def test_chroma_position_ids(self):
        cfg = ModelConfig(name="t", model_id="t", family="chroma", num_text_encoders=1, needs_position_ids=True)
        latents = torch.randn(1, 4, 32, 32)
        enc = torch.randn(1, 512, 4096)
        txt_ids, img_ids = prepare_position_ids(latents, enc, cfg, torch.device("cpu"), torch.float32)
        assert txt_ids is not None
        assert img_ids is not None
        assert txt_ids.shape == (1, 512, 3)
        assert img_ids.shape == (1, 1024, 3)


class TestEncodeText:
    """encode_text dispatch for different model families."""

    def _make_te_and_tokenizer(self, hidden_shape=(1, 77, 768), has_text_embeds=False):
        te = MagicMock()
        output = MagicMock()
        hidden = torch.randn(*hidden_shape)
        output.__getitem__.return_value = hidden
        output.last_hidden_state = hidden
        if has_text_embeds:
            output.text_embeds = torch.randn(hidden_shape[0], 1280)
        te.return_value = output
        tokenizer = MagicMock()
        tok_output = MagicMock()
        tok_output.input_ids = torch.randint(0, 1000, hidden_shape[:2])
        tok_output.attention_mask = torch.ones(hidden_shape[:2])
        tokenizer.return_value = tok_output
        return te, tokenizer

    def test_sd15(self):
        te, tokenizer = self._make_te_and_tokenizer()
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1)
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te, text_encoder_2=None, tokenizer=tokenizer,
            config=cfg,
        )
        hidden, pooled = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 77, 768)
        assert pooled is None

    def test_sdxl(self):
        te1, tok1 = self._make_te_and_tokenizer((1, 77, 768), has_text_embeds=True)
        te2, tok2 = self._make_te_and_tokenizer((1, 77, 1280), has_text_embeds=True)
        cfg = ModelConfig(name="t", model_id="t", family="sdxl", num_text_encoders=2, cross_attention_dim=2048)
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te1, text_encoder_2=te2, tokenizer=tok1,
            tokenizer_2=tok2, config=cfg,
        )
        hidden, pooled = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 77, 2048)  # 768 + 1280
        assert pooled is not None
        assert pooled.shape == (1, 1280)

    def test_flux(self):
        te1, tok1 = self._make_te_and_tokenizer((1, 77, 768))
        te2, tok2 = self._make_te_and_tokenizer((1, 77, 4096))
        cfg = ModelConfig(name="t", model_id="t", family="flux", num_text_encoders=2, cross_attention_dim=3072)
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te1, text_encoder_2=te2, tokenizer=tok1,
            tokenizer_2=tok2, config=cfg,
        )
        hidden, pooled = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 77, 4096)
        assert pooled is None

    def test_sd3(self):
        te1, tok1 = self._make_te_and_tokenizer((1, 256, 768), has_text_embeds=True)
        te2, tok2 = self._make_te_and_tokenizer((1, 256, 1280))
        te3, tok3 = self._make_te_and_tokenizer((1, 256, 2048))  # match concatenated CLIP dim
        cfg = ModelConfig(
            name="t", model_id="t", family="sd3", num_text_encoders=3,
            cross_attention_dim=4096, text_encoder_max_length=256,
            needs_pooled_projections=True, needs_attention_mask=True,
        )
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te1, text_encoder_2=te2, text_encoder_3=te3,
            tokenizer=tok1, tokenizer_2=tok2, tokenizer_3=tok3,
            config=cfg,
        )
        hidden, pooled = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 512, 2048)  # 256+256 along seq, 2048 feature dim
        assert pooled is not None

    def test_hunyuandit_with_t5(self):
        te1 = MagicMock()
        te1.return_value = [torch.randn(1, 77, 1024)]
        te2 = MagicMock()
        te2.return_value = [torch.randn(1, 77, 1024)]
        tok1 = MagicMock()
        tok1.return_value = MagicMock(
            input_ids=torch.randint(0, 1000, (1, 77)),
            attention_mask=torch.ones(1, 77),
        )
        tok2 = MagicMock()
        tok2.return_value = MagicMock(
            input_ids=torch.randint(0, 1000, (1, 77)),
            attention_mask=torch.ones(1, 77),
        )
        cfg = ModelConfig(
            name="t", model_id="t", family="hunyuandit", num_text_encoders=2,
            cross_attention_dim=1024, text_encoder_max_length=77,
        )
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te1, text_encoder_2=te2, tokenizer=tok1,
            tokenizer_2=tok2, config=cfg,
        )
        hidden, aux = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 77, 1024)
        assert isinstance(aux, tuple)
        assert len(aux) == 3

    def test_kolors(self):
        te = MagicMock()
        te.return_value = [torch.randn(1, 256, 4096)]
        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock(
            input_ids=torch.randint(0, 1000, (1, 256)),
            attention_mask=torch.ones(1, 256),
        )
        cfg = ModelConfig(
            name="t", model_id="t", family="kolors", num_text_encoders=1,
            cross_attention_dim=4096, text_encoder_max_length=256,
        )
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te, text_encoder_2=None, tokenizer=tokenizer,
            config=cfg,
        )
        hidden, mask = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 256, 4096)
        assert mask is not None
        assert mask.shape == (1, 256)

    def test_attention_mask_for_dit(self):
        te = MagicMock()
        te.return_value = [torch.randn(1, 77, 768)]
        tokenizer = MagicMock()
        tokenizer.return_value = MagicMock(
            input_ids=torch.randint(0, 1000, (1, 77)),
            attention_mask=torch.ones(1, 77),
        )
        cfg = ModelConfig(
            name="t", model_id="t", family="sana", num_text_encoders=1,
            needs_attention_mask=True,
        )
        comps = PipelineComponents(
            unet=MagicMock(), vae=MagicMock(),
            text_encoder=te, text_encoder_2=None, tokenizer=tokenizer,
            config=cfg,
        )
        hidden, mask = encode_text(comps, ["a cat"], torch.device("cpu"), torch.float32)
        assert hidden.shape == (1, 77, 768)
        assert mask is not None
        assert mask.shape == (1, 77)


class TestBackboneForward:
    """backbone_forward dispatch for all families."""

    def _make_config(self, family, backbone_type="unet", **kwargs):
        defaults = dict(name="t", model_id="t", family=family, num_text_encoders=1)
        defaults.update(kwargs)
        return ModelConfig(**defaults, backbone_type=backbone_type)

    def _make_unet_backbone(self):
        """Return a mock UNet backbone that has .sample on output."""
        backbone = MagicMock()
        output = MagicMock()
        output.sample = torch.randn(1, 4, 32, 32)
        backbone.return_value = output
        return backbone

    def _make_transformer_backbone(self):
        """Return a mock transformer backbone that returns a tuple."""
        backbone = MagicMock()
        backbone.return_value = (torch.randn(1, 4, 32, 32),)
        return backbone

    def test_sd15_unet(self):
        cfg = self._make_config("sd15")
        backbone = self._make_unet_backbone()
        latents = torch.randn(1, 4, 32, 32)
        t = torch.tensor([1.0])
        enc = torch.randn(1, 77, 768)
        backbone_forward(backbone, latents, t, enc, cfg)
        backbone.assert_called_once()

    def test_sdxl_with_added_cond(self):
        cfg = self._make_config("sdxl", needs_added_cond_kwargs=True, num_text_encoders=2)
        backbone = self._make_unet_backbone()
        added_cond = {"text_embeds": torch.randn(1, 1280), "time_ids": torch.randn(1, 6)}
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, added_cond_kwargs=added_cond
        )
        backbone.assert_called_once()
        _, kwargs = backbone.call_args
        assert kwargs["added_cond_kwargs"] is added_cond

    def test_kandinsky3_with_attention_mask(self):
        cfg = self._make_config("kandinsky3", needs_attention_mask=True)
        backbone = self._make_unet_backbone()
        mask = torch.ones(1, 77)
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, attention_mask=mask
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_attention_mask"] is mask

    def test_flux_transformer(self):
        cfg = self._make_config("flux", "transformer", num_text_encoders=2)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg,
            pooled_embeds=torch.randn(1, 1280),
            txt_ids=torch.randn(1, 77, 3), img_ids=torch.randn(1, 1024, 3),
        )
        _, kwargs = backbone.call_args
        assert kwargs["return_dict"] is False

    def test_qwen_image(self):
        cfg = self._make_config("qwen-image", "transformer", needs_guidance_in_forward=True, needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, guidance=4.0, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert kwargs["guidance"] == 4.0

    def test_z_image(self):
        cfg = self._make_config("z-image", "transformer")
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg
        )
        args = backbone.call_args
        assert len(args[0]) == 3  # positional args

    def test_sana(self):
        cfg = self._make_config("sana", "transformer", needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_attention_mask"] is not None

    def test_lumina2(self):
        cfg = self._make_config("lumina2", "transformer", needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_attention_mask"] is not None

    def test_chroma(self):
        cfg = self._make_config("chroma", "transformer", needs_position_ids=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg,
            txt_ids=torch.randn(1, 77, 3), img_ids=torch.randn(1, 1024, 3)
        )
        _, kwargs = backbone.call_args
        assert kwargs["txt_ids"] is not None
        assert kwargs["img_ids"] is not None

    def test_flux2(self):
        cfg = self._make_config("flux2", "transformer", needs_guidance_in_forward=True, needs_position_ids=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg,
            guidance=3.5, txt_ids=torch.randn(1, 77, 4), img_ids=torch.randn(1, 1024, 4),
        )
        _, kwargs = backbone.call_args
        assert kwargs["guidance"] == 3.5

    def test_cogview4(self):
        cfg = self._make_config("cogview4", "transformer", needs_size_kwargs=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg,
            original_size=(1024, 1024), target_size=(512, 512), crop_coords=(0, 0)
        )
        _, kwargs = backbone.call_args
        assert kwargs["original_size"] == (1024, 1024)
        assert kwargs["target_size"] == (512, 512)
        assert kwargs["crop_coords"] == (0, 0)

    def test_aura_flow(self):
        cfg = self._make_config("aura-flow", "transformer")
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg
        )
        _, kwargs = backbone.call_args
        assert kwargs["return_dict"] is False

    def test_sd3(self):
        cfg = self._make_config("sd3", "transformer", num_text_encoders=3, needs_pooled_projections=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, pooled_embeds=torch.randn(1, 2048)
        )
        _, kwargs = backbone.call_args
        assert kwargs["pooled_projections"] is not None

    def test_pixart(self):
        cfg = self._make_config("pixart", "transformer", needs_attention_mask=True, needs_added_cond_kwargs_dit=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_attention_mask"] is not None
        assert "added_cond_kwargs" in kwargs

    def test_hunyuandit(self):
        cfg = self._make_config("hunyuandit", "transformer", needs_t5_and_clip=True, needs_image_meta_size=True, needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        t5_hidden = torch.randn(1, 77, 1024)
        clip_mask = torch.ones(1, 77)
        t5_mask = torch.ones(1, 77)
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, attention_mask=(t5_hidden, clip_mask, t5_mask)
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_hidden_states_t5"] is t5_hidden
        assert kwargs["text_embedding_mask"] is clip_mask
        assert kwargs["text_embedding_mask_t5"] is t5_mask
        assert kwargs["image_meta_size"] is not None
        assert kwargs["style"] is not None

    def test_sana_sprint(self):
        cfg = self._make_config("sana-sprint", "transformer", needs_guidance_in_forward=True, needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 768), cfg, guidance=0.0, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert kwargs["guidance"] == 0.0
        assert kwargs["encoder_attention_mask"] is not None

    def test_unsupported_family(self):
        cfg = self._make_config("unsupported-family", "transformer")
        backbone = self._make_transformer_backbone()
        with pytest.raises(ValueError, match="Unsupported family"):
            backbone_forward(
                backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
                torch.randn(1, 77, 768), cfg
            )

    # ── New model families (2026-07-12) ──

    def test_longcat_image(self):
        cfg = self._make_config("longcat-image", "transformer",
                                needs_guidance_in_forward=True, needs_position_ids=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 2048), cfg, guidance=4.0,
        )
        _, kwargs = backbone.call_args
        assert kwargs["guidance"] == 4.0
        assert "txt_ids" in kwargs
        assert "img_ids" in kwargs

    def test_lumina(self):
        cfg = self._make_config("lumina", "transformer", needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 2048), cfg, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_mask"] is not None

    def test_nucleusmoe_image(self):
        cfg = self._make_config("nucleusmoe-image", "transformer",
                                needs_attention_mask=True, vae_downsample_factor=16)
        backbone = self._make_transformer_backbone()
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 2048), cfg, attention_mask=torch.ones(1, 77)
        )
        _, kwargs = backbone.call_args
        assert "img_shapes" in kwargs
        assert kwargs["encoder_hidden_states_mask"] is not None

    def test_ernie_image(self):
        cfg = self._make_config("ernie-image", "transformer", needs_attention_mask=True)
        backbone = self._make_transformer_backbone()
        mask = torch.ones(1, 77)
        text_lens = mask.sum(dim=1).to(dtype=torch.long)
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 2048), cfg, attention_mask=(mask, text_lens)
        )
        _, kwargs = backbone.call_args
        assert kwargs["text_bth"] is not None
        assert kwargs["text_lens"] is not None

    def test_hidream_image(self):
        cfg = self._make_config("hidream-image", "transformer",
                                needs_t5_and_clip=True, vae_downsample_factor=16)
        backbone = self._make_transformer_backbone()
        pooled = torch.randn(1, 1024)
        llama_hidden = torch.randn(1, 77, 4096)
        llama_mask = torch.ones(1, 77)
        backbone_forward(
            backbone, torch.randn(1, 4, 32, 32), torch.tensor([1.0]),
            torch.randn(1, 77, 4096), cfg,
            attention_mask=(pooled, llama_hidden, llama_mask)
        )
        _, kwargs = backbone.call_args
        assert kwargs["encoder_hidden_states_t5"] is not None
        assert "img_ids" in kwargs
        assert "img_sizes" in kwargs


class TestApplyLora:
    """apply_lora integration with peft."""

    def test_applies_lora(self):
        mock_unet = MagicMock()
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1, lora_target_modules=["to_q"])
        with patch("peft.LoraConfig") as mock_lora_cfg, \
             patch("peft.get_peft_model") as mock_get_peft:
            mock_get_peft.return_value = mock_unet
            result = apply_lora(mock_unet, cfg, rank=8, alpha=16, use_rslora=False)
            mock_lora_cfg.assert_called_once()
            call_kwargs = mock_lora_cfg.call_args.kwargs
            assert call_kwargs["r"] == 8
            assert call_kwargs["lora_alpha"] == 16
            assert call_kwargs["target_modules"] == ["to_q"]
            assert call_kwargs["use_rslora"] is False
            mock_get_peft.assert_called_once_with(mock_unet, mock_lora_cfg.return_value)
            assert result is mock_unet

    def test_defaults_from_config(self):
        mock_unet = MagicMock()
        cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1, default_lora_rank=32, default_lora_alpha=64)
        with patch("peft.LoraConfig") as mock_lora_cfg, \
             patch("peft.get_peft_model") as mock_get_peft:
            mock_get_peft.return_value = mock_unet
            apply_lora(mock_unet, cfg)
            call_kwargs = mock_lora_cfg.call_args.kwargs
            assert call_kwargs["r"] == 32
            assert call_kwargs["lora_alpha"] == 64


class TestUnetForwardAlias:
    """Backward-compatible unet_forward alias."""

    def test_calls_backbone_forward(self):
        with patch("open_vision_banana.model_registry.backbone_forward") as mock_bfw:
            mock_bfw.return_value = torch.randn(1, 4, 32, 32)
            cfg = ModelConfig(name="t", model_id="t", family="sd15", num_text_encoders=1)
            unet = MagicMock()
            latents = torch.randn(1, 4, 32, 32)
            t = torch.tensor([1.0])
            enc = torch.randn(1, 77, 768)
            added = {"text_embeds": torch.randn(1, 1280)}
            result = unet_forward(
                unet, latents, t, enc, cfg, added_cond_kwargs=added
            )
            mock_bfw.assert_called_once()
            assert result is mock_bfw.return_value
