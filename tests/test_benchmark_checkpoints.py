"""Offline tests for pinned source integrity and distinct EMA/online adapters."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import shutil

import pytest
import torch
from safetensors.torch import load_file, save

from scripts.benchmark import checkpoints as cp


REVISION = "1" * 40
SUBDIR = "ckpt/checkpoint-10_final"


@pytest.fixture
def hub(monkeypatch):
    """Emulate manifest + binary HTTP responses; never access a real service."""
    tensors = {
        "base_model.model.linear.lora_A.weight": torch.arange(6, dtype=torch.float32).view(2, 3),
        "base_model.model.linear.lora_B.weight": torch.arange(8, dtype=torch.float32).view(4, 2),
    }
    config = {"r": 2, "lora_alpha": 4, "use_rslora": True, "peft_type": "LORA",
              "rank_pattern": {}, "alpha_pattern": {}, "base_model_name_or_path": None}
    state = {
        "step": 10,
        "lora_state_dict": {k.replace(".lora_A.", ".lora_A.default.").replace(
            ".lora_B.", ".lora_B.default."): value.clone() for k, value in tensors.items()},
        "ema_state_dict": {k.replace(".lora_A.", ".lora_A.default.").replace(
            ".lora_B.", ".lora_B.default."): value.clone() + 1 for k, value in tensors.items()},
    }
    payloads = {}
    manifest = {"id": "owner/model", "sha": REVISION, "siblings": []}
    calls = []

    def update():
        buffer = io.BytesIO()
        torch.save(state, buffer)
        payloads.clear()
        payloads.update({
            f"{SUBDIR}/metadata.json": json.dumps({"step": 10, "lora_rank": 2, "lora_alpha": 4}).encode(),
            f"{SUBDIR}/lora_adapter/adapter_config.json": json.dumps(config).encode(),
            f"{SUBDIR}/lora_adapter/adapter_model.safetensors": save(tensors),
            f"{SUBDIR}/lora.pt": buffer.getvalue(),
            "unrelated/huge-backbone.bin": b"must-not-download",
        })
        manifest["siblings"] = []
        for name, data in payloads.items():
            record = {"rfilename": name, "size": len(data),
                      "blobId": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
            if not name.endswith(".json"):
                record["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            manifest["siblings"].append(record)

    update()

    def fake_open(url, timeout):
        calls.append(url)
        if "/api/models/" in url:
            return io.BytesIO(json.dumps(manifest).encode())
        assert f"/resolve/{REVISION}/" in url
        name = url.split(f"/resolve/{REVISION}/", 1)[1]
        return io.BytesIO(payloads[name])

    monkeypatch.setattr(cp, "urlopen", fake_open)
    return {"tensors": tensors, "state": state, "config": config,
            "payloads": payloads, "manifest": manifest, "calls": calls, "update": update}


def spec(variant="ema"):
    return {"repo_id": "owner/model", "revision": REVISION,
            "checkpoint_subdir": SUBDIR, "weights": variant,
            "base_model_id": "owner/backbone", "selection_note": "annotations allowed"}


def runtime(record, root):
    return cp.resolve_prepared_paths(record, root / "prepared-checkpoint.json")


@pytest.mark.parametrize("field,value", [
    ("revision", "main"), ("revision", "1" * 39), ("revision", "g" * 40),
    ("checkpoint_subdir", "../ckpt"), ("checkpoint_subdir", "ckpt/../../escape"),
    ("checkpoint_subdir", "C:\\outside"), ("checkpoint_subdir", "/tmp/ckpt"),
    ("checkpoint_subdir", "ckpt//checkpoint-10"), ("checkpoint_subdir", "ckpt/NUL"),
    ("repo_id", "../model"), ("weights", "best"), ("weights", None),
])
def test_invalid_spec_rejected_before_network(tmp_path, hub, field, value):
    value_spec = spec()
    value_spec[field] = value
    with pytest.raises(cp.CheckpointValidationError):
        cp.prepare_checkpoint(value_spec, tmp_path, allow_download=True)
    assert not hub["calls"]


def test_requires_download_opt_in(tmp_path, hub):
    with pytest.raises(FileNotFoundError, match="allow_download"):
        cp.prepare_checkpoint(spec(), tmp_path)
    assert not hub["calls"]


def test_online_and_ema_are_separate_pinned_exports(tmp_path, hub, monkeypatch):
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("must not initialize CUDA"))
    original_load = torch.load
    safe_calls = []

    def safe_load(*args, **kwargs):
        safe_calls.append(kwargs)
        assert kwargs["weights_only"] is True
        assert kwargs["map_location"] == "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    online = cp.prepare_checkpoint(spec("online"), tmp_path, True)
    assert len(hub["calls"]) == 4  # one manifest plus three selected checkpoint files
    assert not any("/lora.pt" in call or "huge-backbone" in call for call in hub["calls"])
    assert online["weights_variant"] == "online"
    exported_online = load_file(str(Path(runtime(online, tmp_path)["adapter_dir"]) / "adapter_model.safetensors"))
    assert all(torch.equal(exported_online[k], value) for k, value in hub["tensors"].items())
    ema = cp.prepare_checkpoint(spec(), tmp_path, True)
    assert ema["adapter_dir"] != online["adapter_dir"]
    assert ema["adapter_dir"] == ema["lora_path"]
    assert ema["source_checkpoint"].endswith("lora.pt")
    assert ema["adapter_sha256"] == cp._sha256(Path(runtime(ema, tmp_path)["adapter_dir"]) / "adapter_model.safetensors")
    exported_ema = load_file(str(Path(runtime(ema, tmp_path)["adapter_dir"]) / "adapter_model.safetensors"))
    assert all(torch.equal(exported_ema[k], value + 1) for k, value in hub["tensors"].items())
    assert ema["adapter_configuration"] == hub["config"]
    assert ema["validated_lora_pairs"] == 1
    assert ema["lora_parameter_count"] == 14
    assert ema["provenance"]["source_revision"] == REVISION
    assert len(ema["provenance"]["source_files"]) == 4
    assert all(record["sha256"] and record["revision"] == REVISION
               for record in ema["provenance"]["source_files"])
    assert ema["validation"]["safe_weights_only"] is True
    assert ema["validation"]["base_architecture_shape_verified"] is False
    hub["calls"].clear()
    repeated = cp.prepare_checkpoint(spec(), tmp_path)
    assert repeated["adapter_dir"] == ema["adapter_dir"]
    assert not hub["calls"]
    assert safe_calls


@pytest.mark.parametrize("filename,error", [
    ("lora_adapter/adapter_model.safetensors", "LFS SHA256"),
    ("metadata.json", "Git blob checksum"),
])
def test_same_size_cache_tamper_rejected_or_atomically_redownloaded(tmp_path, hub, filename, error):
    result = cp.prepare_checkpoint(spec(), tmp_path, True)
    record = next(r for r in result["provenance"]["source_files"]
                  if r["remote_path"] == f"{SUBDIR}/{filename}")
    target = tmp_path / record["local_path"]
    damaged = bytearray(target.read_bytes())
    damaged[0] ^= 1
    target.write_bytes(damaged)
    with pytest.raises(cp.CheckpointValidationError, match=error):
        cp.prepare_checkpoint(spec(), tmp_path)
    cp.prepare_checkpoint(spec(), tmp_path, True)
    assert target.read_bytes() == hub["payloads"][f"{SUBDIR}/{filename}"]
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize("case", ["missing_ema", "shape", "nonfinite", "collision", "online_mismatch"])
def test_invalid_ema_never_falls_back_to_online(tmp_path, hub, case):
    state = hub["state"]
    first = next(iter(state["ema_state_dict"]))
    if case == "missing_ema":
        state.pop("ema_state_dict")
    elif case == "shape":
        state["ema_state_dict"][first] = torch.zeros(3, 3)
    elif case == "nonfinite":
        state["ema_state_dict"][first].flatten()[0] = float("nan")
    elif case == "collision":
        state["ema_state_dict"][first.replace(".default.", ".")] = state["ema_state_dict"][first]
    else:
        state["lora_state_dict"][first] = state["lora_state_dict"][first] + 2
    hub["update"]()
    with pytest.raises(cp.CheckpointValidationError):
        cp.prepare_checkpoint(spec(), tmp_path, True)
    assert not any(path.name == "adapter_model.safetensors" for path in (tmp_path / "adapters").rglob("*"))


def test_rejects_manifest_pin_mismatch(tmp_path, hub):
    hub["manifest"]["sha"] = "2" * 40
    with pytest.raises(cp.CheckpointValidationError, match="revision"):
        cp.prepare_checkpoint(spec(), tmp_path, True)
    assert len(hub["calls"]) == 1


def test_failed_download_checksum_keeps_no_partial_file(tmp_path, hub):
    key = f"{SUBDIR}/lora_adapter/adapter_model.safetensors"
    hub["payloads"][key] = b"x" * len(hub["payloads"][key])
    with pytest.raises(cp.CheckpointValidationError, match="LFS SHA256"):
        cp.prepare_checkpoint(spec(), tmp_path, True)
    assert not list(tmp_path.rglob("*.partial"))
    assert not list(tmp_path.rglob("adapter_model.safetensors"))


def test_interrupted_download_removes_partial_without_masking_original_error(tmp_path, hub, monkeypatch):
    original_open = cp.urlopen

    def interrupted(url, timeout):
        if url.endswith("adapter_model.safetensors"):
            raise TimeoutError("synthetic network timeout")
        return original_open(url, timeout)

    monkeypatch.setattr(cp, "urlopen", interrupted)
    with pytest.raises(TimeoutError, match="synthetic network timeout"):
        cp.prepare_checkpoint(spec(), tmp_path, True)
    assert not list(tmp_path.rglob("*.partial"))


def test_atomic_json_failure_cleans_up_and_preserves_existing_file(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("prior content", encoding="utf-8")
    with pytest.raises(ValueError):
        cp._atomic_json(target, {"invalid": float("nan")})
    assert target.read_text(encoding="utf-8") == "prior content"
    assert not list(tmp_path.glob("*.partial"))


def test_metadata_step_must_match_directory(tmp_path, hub):
    key = f"{SUBDIR}/metadata.json"
    data = json.dumps({"step": 11}).encode()
    hub["payloads"][key] = data
    record = next(r for r in hub["manifest"]["siblings"] if r["rfilename"] == key)
    record["size"] = len(data)
    record["blobId"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    with pytest.raises(cp.CheckpointValidationError, match="step"):
        cp.prepare_checkpoint(spec(), tmp_path, True)


def test_cli_candidate_and_provenance(tmp_path, hub, capsys):
    document = {"selected": "trained", "candidates": [dict(spec(), id="trained")]}
    input_path = tmp_path / "spec.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "cache"
    assert cp.main(["--spec", str(input_path), "--output", str(output), "--allow-download"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["weights_variant"] == "ema"
    assert printed["path_base"] == "prepared_checkpoint_directory"
    assert printed["provenance_path"] == "prepared-checkpoint.json"
    assert not Path(printed["adapter_dir"]).is_absolute()
    provenance = json.loads((output / "prepared-checkpoint.json").read_text(encoding="utf-8"))
    assert provenance["revision"] == REVISION
    assert provenance["adapter_configuration"]["use_rslora"] is True
    with pytest.raises(cp.CheckpointValidationError, match="uniquely"):
        cp._select_spec(document, "unknown")


def test_rank_shape_mismatch_rejected_for_online(tmp_path, hub):
    hub["config"]["rank_pattern"] = {"linear": 3}
    hub["update"]()
    with pytest.raises(cp.CheckpointValidationError, match="rank shape"):
        cp.prepare_checkpoint(spec("online"), tmp_path, True)


def test_existing_export_tamper_rejected(tmp_path, hub):
    result = cp.prepare_checkpoint(spec(), tmp_path, True)
    target = tmp_path / result["adapter_dir"] / "adapter_model.safetensors"
    altered = {key: value + 9 for key, value in hub["tensors"].items()}
    target.write_bytes(save(altered))
    with pytest.raises(cp.CheckpointValidationError, match="Cached export"):
        cp.prepare_checkpoint(spec(), tmp_path)


def test_destination_symlink_escape_rejected(tmp_path, hub):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    tmp_path.mkdir(exist_ok=True)
    try:
        (tmp_path / "sources").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows host")
    with pytest.raises(cp.CheckpointValidationError, match="escapes"):
        cp.prepare_checkpoint(spec(), tmp_path, True)
    assert not hub["calls"]


def assert_no_local_roots(value):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_local_roots(key)
            assert_no_local_roots(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_local_roots(item)
    elif isinstance(value, str):
        assert not Path(value).is_absolute(), value
        assert not PureWindowsPath(value).drive, value
        assert "\\" not in value, value


def test_portable_record_and_offline_prepare_after_cache_relocation(tmp_path, hub):
    original = tmp_path / "a"
    result = cp.prepare_checkpoint(spec(), original, True)
    cp._atomic_json(original / "prepared-checkpoint.json", result)
    assert result["path_base"] == "prepared_checkpoint_directory"
    assert_no_local_roots(result)
    assert str(tmp_path) not in json.dumps(result)
    relocated = tmp_path / "b"
    shutil.move(str(original), str(relocated))
    record_path = relocated / "prepared-checkpoint.json"
    portable = json.loads(record_path.read_text(encoding="utf-8"))
    resolved = cp.resolve_prepared_paths(portable, record_path)
    assert Path(resolved["adapter_dir"]).is_relative_to(relocated)
    assert Path(resolved["source_checkpoint"]).is_file()
    exported = load_file(str(Path(resolved["adapter_dir"]) / "adapter_model.safetensors"))
    assert all(torch.equal(exported[key], value + 1) for key, value in hub["tensors"].items())
    for item in resolved["provenance"]["source_files"]:
        assert cp._sha256(Path(item["local_path"])) == item["sha256"]
    for item in resolved["provenance"]["export_files"]:
        assert cp._sha256(Path(item["path"])) == item["sha256"]
    assert portable == result  # Resolver does not mutate the shareable JSON.
    hub["calls"].clear()
    repeated = cp.prepare_checkpoint(spec(), relocated)
    assert repeated == result
    assert not hub["calls"]


@pytest.mark.parametrize("path", ["../outside", "/outside", "C:/Users/Private/weights", "sub\\weights"])
def test_portable_resolver_rejects_absolute_and_escaping_paths(tmp_path, path):
    record = {"path_base": "prepared_checkpoint_directory", "adapter_dir": path}
    with pytest.raises(cp.CheckpointValidationError):
        cp.resolve_prepared_paths(record, tmp_path / "prepared-checkpoint.json")


def test_resolver_reads_legacy_absolute_records_without_mutation(tmp_path):
    record = {"adapter_dir": str(tmp_path / "adapter"), "lora_path": str(tmp_path / "adapter"),
              "source_checkpoint": str(tmp_path / "lora.pt"),
              "provenance": {"source_files": [{"local_path": str(tmp_path / "lora.pt")}],
                             "export_files": [{"path": str(tmp_path / "adapter" / "adapter_config.json")} ]}}
    resolved = cp.resolve_prepared_paths(record, tmp_path / "prepared-checkpoint.json")
    assert resolved == record
    assert "path_base" not in record


@pytest.mark.parametrize("private_base", ["C:/Users/Private Owner/model-cache/base", "/home/private-owner/model-cache/base"])
def test_absolute_base_config_redacted_only_in_record_snapshot(tmp_path, hub, private_base):
    hub["config"]["base_model_name_or_path"] = private_base
    hub["update"]()
    value_spec = spec()
    value_spec["base_model_id"] = private_base
    record = cp.prepare_checkpoint(value_spec, tmp_path, True)
    assert_no_local_roots(record)
    assert private_base not in json.dumps(record)
    assert record["base_model_id"] == "<local-base-model>"
    exported = tmp_path / record["adapter_dir"] / "adapter_config.json"
    assert json.loads(exported.read_text())["base_model_name_or_path"] == private_base
    exported_record = next(item for item in record["provenance"]["export_files"]
                           if item["path"].endswith("adapter_config.json"))
    assert exported_record["sha256"] == cp._sha256(exported)
