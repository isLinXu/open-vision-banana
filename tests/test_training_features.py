"""Tests for training early stopping, logging, and generation datasets."""

from pathlib import Path

import pytest
import torch.nn as nn
from PIL import Image

from open_vision_banana.data.generation import GenerationDataset, build_generation_dataset
from open_vision_banana.data.split import split_dataset
from open_vision_banana.training.early_stopping import EarlyStopping
from open_vision_banana.training.logging import TrainingLogger
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset


class TestEarlyStopping:
    def test_improves_resets_counter(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        assert es.step(1.0) is True
        assert es.counter == 0
        assert es.step(0.5) is True
        assert es.counter == 0

    def test_no_improvement_triggers_stop(self):
        es = EarlyStopping(patience=2, min_delta=0.0)
        es.step(1.0)
        es.step(1.1)
        assert not es.should_stop
        es.step(1.2)
        assert es.should_stop

    def test_min_delta_required(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)
        es.step(0.95)  # improvement 0.05 < min_delta
        assert es.counter == 1
        es.step(0.84)
        assert es.counter == 0


class TestTrainingLogger:
    def test_disabled_is_noop(self):
        logger = TrainingLogger()
        logger.log_metrics(1, {"train/loss": 0.5})
        logger.finish()

    def test_tensorboard_only(self, tmp_path: Path):
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            pytest.skip("tensorboard not installed")
        logger = TrainingLogger(
            tensorboard_enabled=True,
            tensorboard_dir=str(tmp_path / "tb"),
            run_name="test-run",
        )
        logger.log_metrics(10, {"train/loss": 0.42, "train/lr": 1e-4})
        logger.finish()
        assert (tmp_path / "tb" / "test-run").exists()


class TestGenerationDataset:
    def _make_gen_data(self, tmp_path: Path) -> Path:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (64, 64), (100, 150, 200)).save(img_dir / "a.jpg")
        (tmp_path / "prompts.json").write_text('{"a.jpg": "a scenic mountain"}')
        return tmp_path

    def test_loads_with_prompts_json(self, tmp_path: Path):
        root = self._make_gen_data(tmp_path)
        ds = build_generation_dataset(root, resolution=64)
        assert len(ds) == 1
        sample = ds[0]
        assert sample["prompt"] == "a scenic mountain"
        assert sample["target_rgb"].shape == (3, 64, 64)

    def test_default_prompt_without_json(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (32, 32), (0, 128, 255)).save(img_dir / "b.png")
        ds = GenerationDataset(tmp_path, resolution=32)
        assert len(ds) == 1
        assert "Generate" in ds[0]["prompt"]

    def test_require_captions_rejects_unannotated_images(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (32, 32), (0, 128, 255)).save(img_dir / "uncaptioned.png")
        with pytest.raises(ValueError, match="Missing caption"):
            GenerationDataset(tmp_path, resolution=32, require_captions=True)


class TestSplitDataset:
    def test_split_disabled(self):
        ds = VisionTaskDataset()
        train, val = split_dataset(ds, val_ratio=0.0)
        assert train is ds
        assert val is None


class TestInstructionTunerEarlyStopping:
    def _make_tuner(self):
        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_k = nn.Conv2d(3, 8, 1)
                self.to_v = nn.Conv2d(3, 8, 1)
                self.out = nn.Conv2d(8, 3, 1)

            def forward(self, hidden_states, timestep=None, **kwargs):
                return self.out(self.to_k(hidden_states) + self.to_v(hidden_states))

        class FakeGenerator:
            transformer = FakeTransformer()

        gen = FakeGenerator()
        gen.transformer = FakeTransformer()
        return InstructionTuner(
            model=gen,
            lora_rank=2,
            lora_alpha=4,
            allow_component_fallback=True,
            use_ema=False,
            use_gradient_checkpointing=False,
        )

    def test_train_returns_summary(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        tgt_dir = tmp_path / "targets"
        img_dir.mkdir()
        tgt_dir.mkdir()
        Image.new("RGB", (16, 16), (50, 50, 50)).save(img_dir / "x.jpg")
        Image.new("RGB", (16, 16), (200, 100, 50)).save(tgt_dir / "x.jpg")

        train_ds = VisionTaskDataset(str(tmp_path), resolution=16)
        val_ds = VisionTaskDataset(str(tmp_path), resolution=16)

        tuner = self._make_tuner()
        tb_logger = TrainingLogger()

        summary = tuner.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            epochs=1,
            batch_size=1,
            eval_every=1,
            save_every=99999,
            output_dir=str(tmp_path / "ckpts"),
            training_logger=tb_logger,
            log_every=1,
            early_stopping_patience=1,
        )

        assert summary["global_step"] >= 1
        assert "stopped_early" in summary
        tb_logger.finish()

    def test_validation_does_not_overwrite_last_training_metrics(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        tgt_dir = tmp_path / "targets"
        img_dir.mkdir()
        tgt_dir.mkdir()
        Image.new("RGB", (8, 8), (50, 50, 50)).save(img_dir / "x.png")
        Image.new("RGB", (8, 8), (200, 100, 50)).save(tgt_dir / "x.png")
        dataset = VisionTaskDataset(str(tmp_path), resolution=8)
        tuner = self._make_tuner()
        training_step = tuner._training_step

        def training_step_with_metrics(batch):
            loss = training_step(batch)
            tuner._last_step_metrics = {"train/raw_prediction_mse": 1.25}
            return loss

        def evaluate_with_different_metrics(_dataset):
            tuner._last_step_metrics = {"train/raw_prediction_mse": 9.0}
            return 0.5

        tuner._training_step = training_step_with_metrics
        tuner._evaluate = evaluate_with_different_metrics

        summary = tuner.train(
            train_dataset=dataset,
            val_datasets={"depth": dataset},
            epochs=1,
            batch_size=1,
            eval_every=1,
            output_dir=None,
            log_every=1,
        )

        assert summary["last_training_metrics"] == {
            "train/raw_prediction_mse": 1.25
        }

    def test_vision_sample_contains_input_and_target(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        tgt_dir = tmp_path / "targets"
        img_dir.mkdir()
        tgt_dir.mkdir()
        Image.new("RGB", (8, 8), (0, 0, 0)).save(img_dir / "x.png")
        Image.new("RGB", (8, 8), (255, 255, 255)).save(tgt_dir / "x.png")
        sample = VisionTaskDataset(str(tmp_path), resolution=8, task="depth")[0]
        assert sample["image"].shape == sample["target_rgb"].shape == (3, 8, 8)
        assert not sample["image"].equal(sample["target_rgb"])
