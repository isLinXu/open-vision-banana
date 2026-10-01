"""Tests for checkpoint resolution utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from open_vision_banana.inference.checkpoint import (
    CheckpointNotAvailableError,
    is_local_path,
    is_registry_placeholder,
    load_checkpoint_registry,
    peft_adapter_state_to_diffusers,
    resolve_lora_path,
    resolve_lora_repo_id,
)


class TestCheckpointRegistry:
    def test_registry_loads(self):
        registry = load_checkpoint_registry()
        assert "vision-banana" in registry
        assert "repo_id" in registry["vision-banana"]

    def test_resolve_registry_alias(self):
        repo_id = resolve_lora_repo_id("registry:vision-banana", allow_placeholder=True)
        assert repo_id == "open-vision-banana/vision-banana-lora"

    def test_resolve_hf_prefix(self):
        repo_id = resolve_lora_repo_id("hf:org/custom-lora")
        assert repo_id == "org/custom-lora"

    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown checkpoint alias"):
            resolve_lora_repo_id("registry:does-not-exist")

    def test_placeholder_raises_by_default(self):
        with pytest.raises(CheckpointNotAvailableError, match="not yet published"):
            resolve_lora_repo_id("registry:vision-banana")

    def test_placeholder_allowed_when_flag_set(self):
        repo_id = resolve_lora_repo_id("registry:vision-banana", allow_placeholder=True)
        assert repo_id == "open-vision-banana/vision-banana-lora"

    def test_is_registry_placeholder(self):
        assert is_registry_placeholder("vision-banana")
        assert not is_registry_placeholder("sana-depth-relative-v1")

    @patch("open_vision_banana.inference.checkpoint.download_lora_from_hub")
    def test_local_registry_alias_resolves_without_hub(self, mock_download, tmp_path: Path):
        # A registry of its own: the shipped sana-depth-relative-v1 entry points at an
        # untracked output/ directory that only exists on the machine that trained it.
        registry = tmp_path / "configs" / "checkpoint_registry.yaml"
        registry.parent.mkdir()
        registry.write_text(
            "checkpoints:\n"
            "  local-adapter:\n"
            "    repo_id: org/local-adapter\n"
            "    status: placeholder\n"
            "    local_path: output/run/lora_adapter\n",
            encoding="utf-8",
        )
        adapter = tmp_path / "output" / "run" / "lora_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

        resolved = resolve_lora_path("registry:local-adapter", registry_path=registry)

        assert resolved == str(adapter.resolve())
        assert not is_registry_placeholder("local-adapter", registry_path=registry)
        mock_download.assert_not_called()


class TestResolveLoraPath:
    def test_local_path(self, tmp_path: Path):
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        resolved = resolve_lora_path(str(lora_dir))
        assert resolved == str(lora_dir.resolve())

    def test_is_local_path(self, tmp_path: Path):
        p = tmp_path / "weights"
        p.mkdir()
        assert is_local_path(str(p))
        assert not is_local_path("hf:org/repo")
        assert not is_local_path("registry:vision-banana")

    @patch("open_vision_banana.inference.checkpoint.download_lora_from_hub")
    def test_remote_download(self, mock_download, tmp_path: Path):
        mock_download.return_value = tmp_path / "cached"
        (tmp_path / "cached").mkdir()

        resolved = resolve_lora_path("hf:org/test-lora")
        assert resolved == str((tmp_path / "cached").resolve())
        mock_download.assert_called_once()

    def test_none_returns_none(self):
        assert resolve_lora_path(None) is None


class TestPeftAdapterToDiffusers:
    def test_flux_peft_prefix_becomes_transformer_and_drops_base_model(self):
        raw = {
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": "A",
            "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": "B",
            "base_model.model.single_transformer_blocks.0.proj_out.lora_A.weight": "C",
        }
        converted = peft_adapter_state_to_diffusers(raw, "transformer")
        assert converted == {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": "A",
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": "B",
            "transformer.single_transformer_blocks.0.proj_out.lora_A.weight": "C",
        }
        assert not any("base_model" in key for key in converted)

    def test_raw_peft_keys_match_fal_kontext_heuristic(self):
        raw_keys = [
            "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight",
        ]
        assert any("base_model" in key for key in raw_keys)

    def test_empty_adapter_raises(self):
        with pytest.raises(RuntimeError, match="no LoRA"):
            peft_adapter_state_to_diffusers({"not_lora": 1}, "transformer")

    def test_prepare_lora_load_converts_peft_adapter(self, tmp_path: Path):
        from types import SimpleNamespace

        from safetensors.torch import save_file
        import torch

        from open_vision_banana.inference.generator import BaseGenerator

        adapter = tmp_path / "adapter_model.safetensors"
        save_file(
            {
                "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
                "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
            },
            adapter,
        )
        generator = BaseGenerator.__new__(BaseGenerator)
        generator.model_config = SimpleNamespace(family="flux")
        pipe = SimpleNamespace(transformer=object(), unet=None)
        payload, kwargs = generator._prepare_lora_load(tmp_path, pipe)
        assert kwargs == {}
        assert isinstance(payload, dict)
        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in payload
        assert not any("base_model" in key for key in payload)

    def test_prepare_lora_load_skips_peft_conversion_for_sana(self, tmp_path: Path):
        from types import SimpleNamespace

        from safetensors.torch import save_file
        import torch

        from open_vision_banana.inference.generator import BaseGenerator

        adapter = tmp_path / "adapter_model.safetensors"
        save_file(
            {
                "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 4),
                "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(4, 2),
            },
            adapter,
        )
        generator = BaseGenerator.__new__(BaseGenerator)
        generator.model_config = SimpleNamespace(family="sana")
        pipe = SimpleNamespace(transformer=object(), unet=None)
        payload, kwargs = generator._prepare_lora_load(tmp_path, pipe)
        assert kwargs == {}
        assert payload == str(tmp_path)
