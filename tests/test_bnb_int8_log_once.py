"""The 8-bit T5 under FLUX must not flood the log with one bitsandbytes warning per step."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import torch

from open_vision_banana.training import model_backend
from open_vision_banana.training.model_backend import (
    _BNB_INT8_LOGGER,
    ModelBackend,
    _CastWarningOnceFilter,
    _log_bnb_int8_cast_warning_once,
)

CAST_MSG = "MatMul8bitLt: inputs will be cast from %s to float16 during quantization"


@pytest.fixture
def bnb_logger():
    bnb = logging.getLogger(_BNB_INT8_LOGGER)
    saved = list(bnb.filters)
    yield bnb
    bnb.filters[:] = saved


def _record(msg, *args):
    return logging.LogRecord("x", logging.WARNING, __file__, 1, msg, args, None)


def test_filter_passes_cast_warning_once_and_other_records_always():
    f = _CastWarningOnceFilter()
    assert f.filter(_record(CAST_MSG, torch.bfloat16))
    assert not f.filter(_record(CAST_MSG, torch.bfloat16))
    assert not f.filter(_record(CAST_MSG, torch.float32))
    assert f.filter(_record("some other bitsandbytes warning"))
    assert f.filter(_record("some other bitsandbytes warning"))


def test_repeated_cast_warning_logged_once(bnb_logger, caplog):
    _log_bnb_int8_cast_warning_once()
    with caplog.at_level(logging.WARNING, logger=_BNB_INT8_LOGGER):
        for _ in range(5):
            bnb_logger.warning(CAST_MSG, torch.bfloat16)
    assert [r.getMessage() for r in caplog.records] == [CAST_MSG % torch.bfloat16]


def test_install_is_idempotent(bnb_logger):
    _log_bnb_int8_cast_warning_once()
    _log_bnb_int8_cast_warning_once()
    assert sum(isinstance(f, _CastWarningOnceFilter) for f in bnb_logger.filters) == 1


def test_quantized_flux_loader_installs_filter(bnb_logger, monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    accelerate_hooks = pytest.importorskip("accelerate.hooks")

    pipe = MagicMock()
    pipe.to.return_value = pipe
    pipe.transformer.parameters.return_value = []
    monkeypatch.setattr(diffusers.FluxPipeline, "from_pretrained", MagicMock(return_value=pipe))
    monkeypatch.setattr(diffusers.FluxTransformer2DModel, "from_pretrained", MagicMock())
    monkeypatch.setattr(transformers.T5EncoderModel, "from_pretrained", MagicMock())
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", MagicMock())
    monkeypatch.setattr(accelerate_hooks, "remove_hook_from_module", MagicMock())
    monkeypatch.setattr(model_backend.torch.cuda, "memory_allocated", lambda: 0)

    backend = object.__new__(ModelBackend)
    backend.device = torch.device("cpu")
    backend.dtype = torch.bfloat16
    backend.model_config = MagicMock()
    backend._load_flux_quantized("/unused")

    assert any(isinstance(f, _CastWarningOnceFilter) for f in bnb_logger.filters)
