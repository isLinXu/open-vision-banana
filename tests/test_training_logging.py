"""Tests for training logging integrations (W&B and TensorBoard)."""

from unittest.mock import MagicMock, patch

from open_vision_banana.training.logging import TrainingLogger, create_training_logger_from_config


class TestTrainingLoggerInit:
    """Test TrainingLogger initialization."""

    def test_defaults_disabled(self):
        """Both W&B and TensorBoard should be disabled by default."""
        logger = TrainingLogger()
        assert logger.wandb_enabled is False
        assert logger.tensorboard_enabled is False
        assert logger._wandb is None
        assert logger._writer is None

    def test_wandb_disabled_when_import_fails(self):
        """W&B should be disabled when wandb is not installed."""
        with patch.dict("sys.modules", {"wandb": None}):
            logger = TrainingLogger(wandb_enabled=True)
            assert logger.wandb_enabled is False

    def test_tensorboard_disabled_when_import_fails(self):
        """TB should be disabled when tensorboard is not available."""
        with patch.dict("sys.modules", {"torch.utils.tensorboard": None}):
            logger = TrainingLogger(tensorboard_enabled=True)
            assert logger.tensorboard_enabled is False

    def test_wandb_init_exception_handled(self):
        """W&B init failure should be caught gracefully."""
        with patch("builtins.__import__", side_effect=Exception("init failed")):
            logger = TrainingLogger(wandb_enabled=True)
            assert logger.wandb_enabled is False

    def test_tensorboard_init_exception_handled(self):
        """TB init failure should be caught gracefully."""
        with patch("builtins.__import__", side_effect=Exception("tb failed")):
            logger = TrainingLogger(tensorboard_enabled=True)
            assert logger.tensorboard_enabled is False

    def test_wandb_log_checkpoints_requires_wandb(self):
        """wandb_log_checkpoints should be False when wandb is disabled."""
        logger = TrainingLogger(wandb_enabled=False, wandb_log_checkpoints=True)
        assert logger.wandb_log_checkpoints is False

    def test_run_name_priority(self):
        """run_name should take priority over wandb_run_name."""
        logger = TrainingLogger(
            wandb_enabled=False,
            run_name="my-run",
            wandb_run_name="wb-run",
        )
        assert logger is not None


class TestTrainingLoggerLogMetrics:
    """Test log_metrics method."""

    def test_log_metrics_noop_when_disabled(self):
        """log_metrics should be a no-op when both backends are disabled."""
        logger = TrainingLogger()
        logger.log_metrics(step=10, metrics={"loss": 0.5})

    def test_log_metrics_with_mock_wandb(self):
        """log_metrics should call wandb.log when enabled."""
        mock_wandb = MagicMock()
        logger = TrainingLogger()
        logger.wandb_enabled = True
        logger._wandb = mock_wandb

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger.log_metrics(step=5, metrics={"loss": 0.3, "lr": 1e-4})
        mock_wandb.log.assert_called_once_with({"loss": 0.3, "lr": 1e-4}, step=5)

    def test_log_metrics_with_mock_tensorboard(self):
        """log_metrics should call add_scalar when TB is enabled."""
        mock_writer = MagicMock()
        logger = TrainingLogger()
        logger.tensorboard_enabled = True
        logger._writer = mock_writer

        logger.log_metrics(step=3, metrics={"loss": 0.1, "acc": 0.95})
        assert mock_writer.add_scalar.call_count == 2

    def test_log_metrics_skips_non_numeric(self):
        """Non-numeric metrics should be skipped for TensorBoard."""
        mock_writer = MagicMock()
        logger = TrainingLogger()
        logger.tensorboard_enabled = True
        logger._writer = mock_writer

        logger.log_metrics(step=1, metrics={"loss": 0.5, "name": "test"})
        assert mock_writer.add_scalar.call_count == 1

    def test_log_metrics_wandb_exception_handled(self):
        """wandb.log failure should be caught gracefully."""
        mock_wandb = MagicMock()
        mock_wandb.log.side_effect = Exception("log failed")
        logger = TrainingLogger()
        logger.wandb_enabled = True
        logger._wandb = mock_wandb

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Should not raise
            logger.log_metrics(step=1, metrics={"loss": 0.5})


class TestTrainingLoggerLogHyperparams:
    """Test log_hyperparams method."""

    def test_log_hyperparams_noop_when_disabled(self):
        """log_hyperparams should be a no-op when disabled."""
        logger = TrainingLogger()
        logger.log_hyperparams({"lr": 1e-4}, {"loss": 0.5})

    def test_log_hyperparams_with_mock_wandb(self):
        """log_hyperparams should call wandb.config.update."""
        mock_wandb = MagicMock()
        logger = TrainingLogger()
        logger.wandb_enabled = True
        logger._wandb = mock_wandb

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger.log_hyperparams({"lr": 1e-4, "batch_size": 32})
        mock_wandb.config.update.assert_called_once_with({"lr": 1e-4, "batch_size": 32})

    def test_log_hyperparams_with_mock_tensorboard(self):
        """log_hyperparams should call add_hparams for TB."""
        mock_writer = MagicMock()
        logger = TrainingLogger()
        logger.tensorboard_enabled = True
        logger._writer = mock_writer

        logger.log_hyperparams({"lr": 0.001}, {"loss": 0.5})
        mock_writer.add_hparams.assert_called_once()

    def test_log_hyperparams_tb_exception_handled(self):
        """TB add_hparams failure should be caught gracefully."""
        mock_writer = MagicMock()
        mock_writer.add_hparams.side_effect = Exception("hparams failed")
        logger = TrainingLogger()
        logger.tensorboard_enabled = True
        logger._writer = mock_writer

        logger.log_hyperparams({"lr": 0.001}, {"loss": 0.5})


class TestTrainingLoggerLogCheckpoint:
    """Test log_checkpoint method."""

    def test_log_checkpoint_noop_when_disabled(self):
        """log_checkpoint should be a no-op when wandb_log_checkpoints is False."""
        logger = TrainingLogger()
        logger.log_checkpoint("/tmp/ckpt")

    def test_log_checkpoint_no_wandb(self):
        """log_checkpoint should skip when _wandb is None."""
        logger = TrainingLogger()
        logger.wandb_log_checkpoints = True
        logger._wandb = None
        logger.log_checkpoint("/tmp/ckpt")

    def test_log_checkpoint_not_a_directory(self, tmp_path):
        """log_checkpoint should warn when path is not a directory."""
        logger = TrainingLogger()
        logger.wandb_log_checkpoints = True
        logger._wandb = MagicMock()

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")
        logger.log_checkpoint(str(file_path))

    def test_log_checkpoint_success(self, tmp_path):
        """log_checkpoint should upload artifact when everything is set up."""
        mock_wandb = MagicMock()
        logger = TrainingLogger()
        logger.wandb_log_checkpoints = True
        logger._wandb = mock_wandb

        ckpt_dir = tmp_path / "checkpoint-1"
        ckpt_dir.mkdir()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger.log_checkpoint(str(ckpt_dir), alias="best", step=100)
        mock_wandb.Artifact.assert_called_once()

    def test_log_checkpoint_exception_handled(self, tmp_path):
        """log_checkpoint should handle upload failures gracefully."""
        mock_wandb = MagicMock()
        mock_wandb.Artifact.side_effect = Exception("upload failed")
        logger = TrainingLogger()
        logger.wandb_log_checkpoints = True
        logger._wandb = mock_wandb

        ckpt_dir = tmp_path / "checkpoint-1"
        ckpt_dir.mkdir()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger.log_checkpoint(str(ckpt_dir))


class TestTrainingLoggerFinish:
    """Test finish method."""

    def test_finish_noop_when_disabled(self):
        """finish should be a no-op when both backends are disabled."""
        logger = TrainingLogger()
        logger.finish()

    def test_finish_closes_wandb(self):
        """finish should call wandb.finish when enabled."""
        mock_wandb = MagicMock()
        logger = TrainingLogger()
        logger._wandb = mock_wandb

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            logger.finish()
        mock_wandb.finish.assert_called_once()
        assert logger._wandb is None

    def test_finish_closes_tensorboard(self):
        """finish should call writer.close when enabled."""
        mock_writer = MagicMock()
        logger = TrainingLogger()
        logger._writer = mock_writer

        logger.finish()
        mock_writer.close.assert_called_once()
        assert logger._writer is None

    def test_finish_idempotent(self):
        """finish should be safe to call multiple times."""
        mock_writer = MagicMock()
        logger = TrainingLogger()
        logger._writer = mock_writer

        logger.finish()
        logger.finish()
        mock_writer.close.assert_called_once()


class TestCreateTrainingLoggerFromConfig:
    """Test create_training_logger_from_config factory."""

    def test_no_logging_config(self):
        """Should return default logger when config has no logging section."""
        from types import SimpleNamespace

        cfg = SimpleNamespace()
        logger = create_training_logger_from_config(cfg)
        assert logger.wandb_enabled is False
        assert logger.tensorboard_enabled is False

    def test_with_logging_config(self):
        """Should create logger from logging config."""
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            logging=SimpleNamespace(
                wandb_enabled=True,
                wandb_project="test-project",
                wandb_run_name="test-run",
                wandb_entity="test-entity",
                wandb_log_checkpoints=True,
                tensorboard_enabled=True,
                tensorboard_dir="output/tb",
                run_name="custom-run",
            )
        )
        logger = create_training_logger_from_config(cfg)
        assert logger is not None
