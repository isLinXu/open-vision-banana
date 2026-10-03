"""Prediction contracts and fail-closed checkpoint handling, without downloads."""

import builtins
import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark.backends import _verify_weight_provenance, build_backend
from scripts.benchmark.compatibility import load_tensor_state, sha256_file, strict_zip


def test_registry_rejects_unknown_backend_and_options():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_backend({"id": "bad", "backend": "not_registered"}, "cpu")
    with pytest.raises(ValueError, match="Unsupported tiny options"):
        build_backend({"id": "bad", "backend": "tiny", "options": {"typo": True}}, "cpu")


def test_tiny_actual_forward_contract_and_seed_preserves_rng():
    torch.manual_seed(913)
    before = torch.random.get_rng_state().clone()
    model_spec = {"id": "tiny", "backend": "tiny", "weights_variant": "random", "options": {"seed": 42}}
    first = build_backend(model_spec, "cpu")
    second = build_backend(model_spec, "cpu")
    try:
        first.prepare()
        second.prepare()
        assert torch.equal(before, torch.random.get_rng_state())
        assert all(torch.equal(value, second.model.state_dict()[name])
                   for name, value in first.model.state_dict().items())
        image = torch.linspace(0, 1, 3 * 32 * 48).reshape(1, 3, 32, 48)
        prediction = first.predict(image, "semantic", {"id": "small", "max_edge": 48})
        assert prediction.rgb_output.shape == image.shape
        assert prediction.rgb_output.device == image.device
        assert torch.isfinite(prediction.rgb_output).all()
        assert 0 <= prediction.rgb_output.min() <= prediction.rgb_output.max() <= 1
        descriptor = first.describe()
        assert descriptor["parameters"] == 42492
        assert descriptor["trained"] is False
        assert descriptor["task_contracts"]["semantic"]["num_classes"] == 150
        assert descriptor["task_contracts"]["normal"]["encoding"] == "paper_camera"
        assert descriptor["task_contracts"]["depth"]["representation"] == "metric_power"
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("variant", ["pretrained", "online", "ema"])
def test_tiny_never_silently_relabels_random_weights_as_trained(variant):
    backend = build_backend({"id": "tiny", "backend": "tiny", "weights_variant": variant}, "cpu")
    with pytest.raises(ValueError):
        backend.prepare()
    assert backend.ready is False
    assert backend.model is None


def test_lraspp_default_forbids_download_and_random_fallback():
    backend = build_backend({"id": "voc", "backend": "torchvision_lraspp", "weights_variant": "pretrained"}, "cpu")
    with pytest.raises(RuntimeError, match="local checkpoint"):
        backend.prepare()
    assert backend.model is None
    invalid = build_backend({"id": "voc", "backend": "torchvision_lraspp", "weights_variant": "random"}, "cpu")
    with pytest.raises(ValueError, match="pretrained"):
        invalid.prepare()


def test_lraspp_nonfinite_logits_cannot_be_hidden_by_argmax():
    backend = build_backend({"id": "voc", "backend": "torchvision_lraspp", "weights_variant": "pretrained"}, "cpu")
    backend.ready = True
    backend.metadata["tasks"] = ["semantic"]
    backend.model = lambda image: {"out": torch.full((1, 21, 32, 32), float("nan"))}
    with pytest.raises(ValueError, match="Nonfinite LRASPP logits"):
        backend.predict(torch.zeros(1, 3, 32, 32), "semantic", {"id": "small", "max_edge": 32})


def test_tensor_loading_detects_misnamed_torch_save_and_is_safe(tmp_path):
    path = tmp_path / "model.safetensors"
    torch.save({"weights": torch.ones(2)}, path)
    state, descriptor = load_tensor_state(path)
    assert torch.equal(state["weights"], torch.ones(2))
    assert descriptor["load_method"] == "torch.load(weights_only=True)"
    assert descriptor["sha256"] == sha256_file(path)


def test_old_torch_pickle_requires_separate_trust_declaration(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    path.write_bytes(b"non-safetensors test container")
    monkeypatch.setattr(inspect, "signature", lambda _: inspect.Signature())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Untrusted pickle load attempted"))
    with pytest.raises(RuntimeError, match="BOTH"):
        load_tensor_state(path, allow_legacy=True, trusted_checkpoint=False)


def test_strict_zip_keeps_length_guard():
    assert list(strict_zip([1], [2], strict=True)) == [(1, 2)]
    with pytest.raises(ValueError, match="different lengths"):
        list(strict_zip([1], [], strict=True))


@pytest.fixture
def adapter_fixture(tmp_path):
    from safetensors.torch import save_file

    adapter = tmp_path / "ema"
    adapter.mkdir()
    state = {"base_model.model.layer.lora_A.weight": torch.ones(2, 3),
             "base_model.model.layer.lora_B.weight": torch.ones(4, 2)}
    save_file(state, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text('{"r":2,"lora_alpha":4}', encoding="utf-8")
    source = tmp_path / "lora.pt"
    torch.save({"ema_state_dict": state, "lora_state_dict": {name: tensor * 2 for name, tensor in state.items()}}, source)
    revision = "a" * 40
    record = {
        "schema_version": 1, "weights_variant": "ema", "adapter_dir": str(adapter),
        "revision": revision, "checkpoint_subdir": "ckpt/checkpoint-100_final",
        "validation": {"safe_weights_only": True},
        "provenance": {
            "weights_variant": "ema",
            "export_files": [{"path": str(adapter / name), "sha256": sha256_file(adapter / name)}
                             for name in ("adapter_model.safetensors", "adapter_config.json")],
            "source_files": [{"remote_path": "ckpt/checkpoint-100_final/lora.pt", "revision": revision,
                              "local_path": str(source), "sha256": sha256_file(source)}],
        },
    }
    manifest = tmp_path / "prepared-checkpoint.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return adapter, source, manifest, record, state


def test_prepared_ema_provenance_matches_actual_source_tensors(adapter_fixture):
    adapter, _, manifest, _, _ = adapter_fixture
    result = _verify_weight_provenance(manifest, adapter, "ema")
    assert result["export_hashes_verified"] is True
    assert result["ema_tensors_equal"] is True


def test_portable_ema_cache_verifies_after_directory_move(adapter_fixture, tmp_path_factory):
    adapter, _, manifest, record, _ = adapter_fixture
    original = manifest.parent
    record["path_base"] = "prepared_checkpoint_directory"
    record["adapter_dir"] = adapter.relative_to(original).as_posix()
    for item in record["provenance"]["export_files"]:
        item["path"] = Path(item["path"]).relative_to(original).as_posix()
    for item in record["provenance"]["source_files"]:
        item["local_path"] = Path(item["local_path"]).relative_to(original).as_posix()
    manifest.write_text(json.dumps(record), encoding="utf-8")

    relocated = tmp_path_factory.mktemp("relocated") / "checkpoint-cache"
    shutil.move(str(original), str(relocated))
    assert not original.exists()
    result = _verify_weight_provenance(
        relocated / manifest.name, relocated / record["adapter_dir"], "ema"
    )
    assert result["export_hashes_verified"] is True
    assert result["ema_tensors_equal"] is True
    assert Path(result["source_checkpoint"]).is_relative_to(relocated)


def test_portable_ema_source_tampering_still_rejected(adapter_fixture):
    adapter, source, manifest, record, _ = adapter_fixture
    record["path_base"] = "prepared_checkpoint_directory"
    record["adapter_dir"] = adapter.relative_to(manifest.parent).as_posix()
    for item in record["provenance"]["export_files"]:
        item["path"] = Path(item["path"]).relative_to(manifest.parent).as_posix()
    record["provenance"]["source_files"][0]["local_path"] = source.name
    manifest.write_text(json.dumps(record), encoding="utf-8")
    source.write_bytes(b"modified checkpoint")
    with pytest.raises(ValueError, match="source checkpoint is missing or differs"):
        _verify_weight_provenance(manifest, adapter, "ema")


def test_provenance_cannot_relabel_online_adapter_as_ema(adapter_fixture):
    from safetensors.torch import save_file

    adapter, _, manifest, record, state = adapter_fixture
    save_file({name: tensor * 2 for name, tensor in state.items()}, str(adapter / "adapter_model.safetensors"))
    # Even a manifest updated to the online file's real hash is not EMA evidence.
    record["provenance"]["export_files"][0]["sha256"] = sha256_file(adapter / "adapter_model.safetensors")
    manifest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="do not equal.*EMA"):
        _verify_weight_provenance(manifest, adapter, "ema")


@pytest.fixture
def forbid_backbone_import(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("open_vision_banana.inference"):
            pytest.fail("Invalid spec attempted to load model-backbone inference modules")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_unverified_trained_contract_stops_before_backbone(forbid_backbone_import):
    backend = build_backend({"id": "trained", "backend": "vision_banana", "weights_variant": "ema"}, "cpu")
    with pytest.raises(ValueError, match="contract is unverified"):
        backend.prepare()


def _trained_spec(adapter, **option_overrides):
    options = {"inference_contract_verified": True, "model_time_convention": "native_sana"}
    options.update(option_overrides)
    return {"id": "trained", "backend": "vision_banana", "checkpoint": str(adapter),
            "base_model_id": str(adapter.parent), "weights_variant": "ema", "options": options}


def test_ema_without_export_evidence_stops_before_backbone(adapter_fixture, forbid_backbone_import):
    adapter, _, _, _, _ = adapter_fixture
    backend = build_backend(_trained_spec(adapter), "cpu")
    with pytest.raises(ValueError, match="EMA requires a provenance"):
        backend.prepare()


def test_ema_hash_mismatch_stops_before_backbone(adapter_fixture, forbid_backbone_import):
    adapter, _, manifest, record, _ = adapter_fixture
    record["provenance"]["export_files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(record), encoding="utf-8")
    backend = build_backend(_trained_spec(adapter, weights_provenance_path=str(manifest)), "cpu")
    with pytest.raises(ValueError, match="export path/hash"):
        backend.prepare()


def test_missing_encoding_contract_stops_before_backbone(adapter_fixture, forbid_backbone_import):
    adapter, _, manifest, _, _ = adapter_fixture
    backend = build_backend(_trained_spec(adapter, weights_provenance_path=str(manifest)), "cpu")
    with pytest.raises(ValueError, match="explicit encoding_contract"):
        backend.prepare()


@pytest.mark.parametrize("task,contract,match", [
    ("depth", {"colormap": "turbo"}, "metric_power"),
    ("normal", {"encoding": "guess"}, "paper_camera"),
    ("semantic", {"label_space": "ade20k150"}, "exact training palette"),
])
def test_ambiguous_trained_decoder_contract_stops_before_backbone(
    adapter_fixture, forbid_backbone_import, task, contract, match
):
    adapter, _, manifest, _, _ = adapter_fixture
    backend = build_backend(_trained_spec(adapter, weights_provenance_path=str(manifest),
                            encoding_contract={task: contract}), "cpu")
    with pytest.raises(ValueError, match=match):
        backend.prepare()
