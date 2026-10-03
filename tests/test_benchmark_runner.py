"""Analytical/failure cases for the benchmark protocol, rather than model scores."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.benchmark import PROJECT_ROOT
from scripts.benchmark.backends import Prediction
from scripts.benchmark.config import checked_id, load_config, validate_config
from scripts.benchmark.data import resized_hw, select_samples
from scripts.benchmark.runner import run_benchmark
from scripts.benchmark.reporting import PortableReport
from scripts.benchmark.tasks import build_task


def profile():
    return load_config(PROJECT_ROOT / "scripts/benchmark/configs/synthetic-smoke.json")


@pytest.mark.parametrize("bad", ["../escape", "C:escape", "", "CON", "lpt1.txt", "normal."])
def test_ids_cannot_escape_output_or_use_windows_devices(bad):
    with pytest.raises(ValueError):
        checked_id(bad)


def test_schema_rejects_typos_and_ambiguous_matrix():
    cfg = profile()
    cfg["sizes"][0]["max_edeg"] = 32
    with pytest.raises(ValueError, match="Unknown size"):
        validate_config(cfg)
    cfg = profile()
    cfg["models"].append(copy.deepcopy(cfg["models"][0]))
    with pytest.raises(ValueError, match="Duplicate models"):
        validate_config(cfg)


def test_profile_paths_are_relative_to_config_and_cli_paths_to_cwd(tmp_path):
    cfg = profile()
    cfg["models"][0]["checkpoint"] = "model.pt"
    cfg["models"][0]["options"]["weights_provenance_path"] = "prepared.json"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(cfg))
    loaded = load_config(path)
    assert loaded["models"][0]["checkpoint"] == str(tmp_path / "model.pt")
    assert loaded["models"][0]["options"]["weights_provenance_path"] == str(tmp_path / "prepared.json")


def test_legacy_profile_author_is_not_a_run_parameter():
    cfg = profile()
    cfg["author"] = "private-contributor"
    assert "author" not in validate_config(cfg)


def test_report_paths_do_not_expose_machine_identity_and_are_relocation_stable(tmp_path):
    def make_report(location):
        data_root = location / "private-user-home" / "photos"
        weights = location / "private-user-home" / "weights" / "model.pt"
        cfg = {"dataset": {"root": str(data_root)},
               "models": [{"id": "model", "checkpoint": str(weights)}]}
        serializer = PortableReport(cfg, PROJECT_ROOT)
        return serializer.clean({"author": "private-contributor", "config": cfg,
                                 "image": str(data_root / "input.jpg"),
                                 "error": "Missing image: " + str(data_root / "missing.jpg"),
                                 "source": str(PROJECT_ROOT / "src/open_vision_banana/tiny_model.py")})
    result = make_report(tmp_path / "original")
    assert result == make_report(tmp_path / "relocated")
    encoded = json.dumps(result)
    assert "private-user-home" not in encoded
    assert "private-contributor" not in encoded
    assert str(tmp_path) not in encoded
    assert result["image"] == "dataset/input.jpg"
    assert result["config"]["models"][0]["checkpoint"] == "checkpoints/model/model.pt"
    assert result["source"] == "src/open_vision_banana/tiny_model.py"


def test_unbound_external_file_reports_only_basename(tmp_path):
    serializer = PortableReport({}, PROJECT_ROOT)
    assert serializer.clean(str(tmp_path / "private-user" / "weights.pt")) == "external/weights.pt"


@pytest.mark.parametrize("error", [
    "Cannot open 'C:/Users/Private Owner/weights.pt'",
    "Cannot open '/home/private-owner/weights.pt'",
    r"Cannot open C:\Users\Private Owner\weights.pt",
])
def test_external_paths_embedded_in_errors_are_redacted(error):
    serializer = PortableReport({}, PROJECT_ROOT)
    result = serializer.clean(error)
    assert "external/weights.pt" in result
    assert "Private Owner" not in result
    assert "private-owner" not in result
    url = "See https://huggingface.co/owner/model/blob/revision/config.json"
    assert serializer.clean(url) == url


def test_plan_does_not_import_torch_or_access_weights():
    code = ("import sys; from scripts.benchmark.config import load_config; "
            "from scripts.benchmark.runner import plan_benchmark; "
            "c=load_config('scripts/benchmark/configs/trained-sana.json'); "
            "p=plan_benchmark(c); assert len(p['matrix'])==6; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), check=True)


def semantic_task(label_space="toy2"):
    return build_task("semantic", {"label_space": "toy2", "num_classes": 2,
                                  "palette": {"red": (255, 0, 0), "blue": (0, 0, 255)}},
                      {"label_space": label_space, "ignore_index": 255})


def test_semantic_score_is_global_confusion_not_image_mean():
    task = semantic_task()
    task.start_metrics()
    task.update(torch.tensor([[[[0]]]]), torch.tensor([[[[0]]]]))
    task.update(torch.zeros(1, 1, 3, 3, dtype=torch.long), torch.ones(1, 1, 3, 3, dtype=torch.long))
    result = task.compute()["metrics"]
    assert result["valid_pixels"] == 10
    assert result["mIoU"] == pytest.approx(0.05)
    assert result["pixel_accuracy"] == pytest.approx(0.1)
    assert result["mIoU"] != pytest.approx(0.5)  # Mean of the two image scores.


def test_semantic_ignore_masks_and_empty_valid_are_explicit():
    task = semantic_task()
    task.start_metrics()
    task.update(torch.tensor([[[[1, 0]]]]), torch.tensor([[[[0, 0]]]]),
                torch.tensor([[[[False, True]]]]))
    result = task.compute()["metrics"]
    assert result["mIoU"] == 1.0
    assert result["per_class_iou"] == [1.0, None]
    task = semantic_task()
    task.start_metrics()
    task.update(torch.zeros(1, 1, 1, 1, dtype=torch.long), torch.full((1, 1, 1, 1), 255))
    assert task.compute()["reason"] == "zero_valid_pixels"


def test_different_label_spaces_cannot_be_compared():
    with pytest.raises(ValueError, match="label_space"):
        semantic_task("voc21").start_metrics()


def test_invalid_labels_are_failures_not_dropped_pixels():
    task = semantic_task()
    task.start_metrics()
    with pytest.raises(ValueError, match="class ID"):
        task.decode(Prediction(labels=torch.tensor([[[2]]])))
    with pytest.raises(ValueError, match="GT class IDs"):
        task.update(torch.zeros(1, 1, 1, 1), torch.tensor([[[[4]]]]))


def test_turbo_contract_roundtrip_and_wrong_colormap_counterexample():
    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    mapper = DepthRGBMapper(colormap="turbo")
    truth = torch.linspace(0.5, 20, 100).view(1, 1, 10, 10)
    rgb = mapper.depth_to_rgb(truth)
    task = build_task("depth", {"representation": "metric_power", "colormap": "turbo", "lam": -3, "c": 10 / 3}, {"depth_units": "meters"})
    correct = task.decode(Prediction(rgb_output=rgb)).values
    wrong = build_task("depth", {"representation": "metric_power", "colormap": "cube_edge", "lam": -3, "c": 10 / 3}, {"depth_units": "meters"}).decode(Prediction(rgb_output=rgb)).values
    assert ((correct - truth).abs() / truth).mean() < 1e-5
    assert ((wrong - truth).abs() / truth).mean() > 0.5
    with pytest.raises(ValueError, match="units"):
        build_task("depth", {"representation": "normalized_scalar", "colormap": "turbo"}, {"depth_units": "meters"}).start_metrics()


def test_zero_depth_predictions_remain_in_metric_denominator():
    task = build_task("depth", {"representation": "metric_power", "colormap": "cube_edge", "lam": -3, "c": 10 / 3}, {"depth_units": "meters"})
    task.start_metrics()
    task.update(torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
    metrics = task.compute()["metrics"]
    assert metrics["valid_pixels"] == 4
    assert metrics["AbsRel"] > 0.99


def test_depth_contract_does_not_fill_in_unverified_encoding():
    with pytest.raises(ValueError, match="explicit fields"):
        build_task("depth", {"representation": "metric_power"}, {"depth_units": "meters"})


def test_nyu_crop_top_error_is_not_hidden_by_kitti_garg_crop():
    task = build_task("depth", {"representation": "metric_power", "colormap": "cube_edge", "lam": -3, "c": 10 / 3},
                      {"depth_units": "meters", "depth_protocol": "nyu"})
    task.start_metrics()
    gt = torch.ones(1, 1, 480, 640)
    pred = gt.clone()
    pred[..., :100, :] = 2
    task.update(pred, gt)
    quality = task.compute()
    assert quality["metrics"]["AbsRel"] == pytest.approx(55 / 426)
    assert quality["metrics"]["valid_pixels"] == 426 * 560
    assert quality["protocol"]["crop"] == [45, 471, 41, 601]
    with pytest.raises(ValueError, match="480x640"):
        task.update(torch.ones(1, 1, 32, 32), torch.ones(1, 1, 32, 32))


def test_kitti_garg_exact_ratios_and_cap_mask():
    from scripts.benchmark.protocols import depth_protocol
    gt = torch.ones(1, 1, 352, 1216)
    pred = gt * 100
    clipped, _, mask, metadata = depth_protocol("kitti", pred, gt)
    assert metadata["crop"] == [143, 349, 43, 1172]
    assert mask.sum() == 206 * 1129
    assert clipped.max() == 80


def test_normal_camera_x_sign_is_not_silently_changed():
    rgb = torch.tensor([0.0, 0.5, 0.5]).reshape(1, 3, 1, 1)
    paper = build_task("normal", {"encoding": "paper_camera"}, {}).decode(Prediction(rgb_output=rgb)).values
    standard = build_task("normal", {"encoding": "standard_xyz_to_rgb"}, {}).decode(Prediction(rgb_output=rgb)).values
    assert paper[0, 0, 0, 0] == 1
    assert standard[0, 0, 0, 0] == -1


def test_alignment_and_missing_explicit_samples(tmp_path):
    assert resized_hw(333, 500, {"max_edge": 256, "alignment": 8}) == (168, 256)
    assert resized_hw(12, 12, {"max_edge": 32, "alignment": 8}) == (32, 32)
    with pytest.raises(FileNotFoundError):
        select_samples({"kind": "images", "root": str(tmp_path), "include": ["missing.jpg"], "max_samples": 6})


def test_partial_gt_does_not_produce_ambiguous_full_dataset_score(tmp_path):
    image = tmp_path / "image.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    gt = tmp_path / "gt.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(gt)
    manifest = tmp_path / "data.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "image": str(image), "gt": {"semantic": str(gt)}},
        {"id": "b", "image": str(image)}]))
    cfg = profile()
    cfg["dataset"] = {"kind": "manifest", "manifest": str(manifest), "max_samples": 2}
    with pytest.raises(ValueError, match="Partial GT"):
        run_benchmark(cfg, tmp_path / "out")


def test_minimal_run_artifacts_provenance_and_no_gt_quality(tmp_path):
    cfg = profile()
    cfg["sizes"] = cfg["sizes"][:1]
    cfg["tasks"] = cfg["tasks"][:1]
    cfg["timing"] = {"warmup": 0, "repeats": 1, "cpu_threads": 1}
    cfg["dataset"]["synthetic_count"] = 1
    result = run_benchmark(cfg, tmp_path / "run")
    assert result["status"] == "completed"
    assert result["groups"][0]["quality"]["metrics"] is None
    assert result["models"][0]["trained"] is False
    assert result["source"]["harness_sha256"]
    assert result["groups"][0]["decode_latency"]["count"] == 1
    assert result["jobs"][0]["paired_predict_decode_ms"] >= result["jobs"][0]["latency_samples_ms"][-1]
    artifact = result["jobs"][0]["artifacts"]
    assert (tmp_path / "run" / artifact["preview"]).is_file()
    assert np.load(tmp_path / "run" / artifact["raw"], allow_pickle=False).shape == (1, 1, 24, 32)
    with pytest.raises(ValueError, match="Output must be empty"):
        run_benchmark(cfg, tmp_path / "run")


def test_gt_manifest_full_runner_restores_original_grid(tmp_path, monkeypatch):
    from scripts.benchmark.backends import BACKEND_REGISTRY
    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    contract = {
        "semantic": {"label_space": "toy2", "num_classes": 2,
                     "palette": {"red": (255, 0, 0), "blue": (0, 0, 255)}},
        "depth": {"representation": "metric_power", "colormap": "turbo", "lam": -3, "c": 10 / 3},
        "normal": {"encoding": "paper_camera"}}

    class AnalyticalBackend:
        def __init__(self, *args):
            pass

        def prepare(self):
            pass

        def describe(self):
            return {"tasks": list(contract), "task_contracts": contract, "trained": False}

        def predict(self, image, task, size):
            height, width = image.shape[-2:]
            if task == "semantic":
                return Prediction(labels=torch.zeros(1, height, width, dtype=torch.long))
            if task == "depth":
                return Prediction(rgb_output=DepthRGBMapper(colormap="turbo").depth_to_rgb(torch.full((1, 1, height, width), 2.0)))
            rgb = torch.full((1, 3, height, width), 0.5)
            rgb[:, 2] = 1
            return Prediction(rgb_output=rgb)

        def close(self):
            pass

    monkeypatch.setitem(BACKEND_REGISTRY, "analytical", AnalyticalBackend)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "image.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "semantic.png")
    np.save(tmp_path / "depth.npy", np.full((8, 8), 2.0, dtype=np.float32))
    normal = np.zeros((8, 8, 3), dtype=np.float32)
    normal[..., 2] = 1
    np.save(tmp_path / "normal.npy", normal)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "known", "image": "image.png", "gt": {
        "semantic": "semantic.png", "depth": "depth.npy", "normal": "normal.npy"}}))
    cfg = profile()
    cfg["models"] = [{"id": "analytical", "backend": "analytical"}]
    cfg["sizes"] = cfg["sizes"][:1]
    cfg["dataset"] = {"kind": "manifest", "manifest": str(manifest), "max_samples": 1,
                      "label_space": "toy2", "ignore_index": 255, "depth_units": "meters"}
    cfg["timing"] = {"warmup": 0, "repeats": 1, "cpu_threads": 1}
    result = run_benchmark(cfg, tmp_path / "result")
    groups = {r["task"]: r["quality"]["metrics"] for r in result["groups"]}
    assert groups["semantic"]["mIoU"] == 1
    assert groups["depth"]["AbsRel"] < 1e-5
    assert groups["normal"]["mean_angle_deg"] == 0
    assert all(group["valid_pixels"] == 64 for group in groups.values())
