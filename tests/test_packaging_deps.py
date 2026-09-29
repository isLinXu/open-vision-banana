"""The T5 tokenizer deps (see the comment on the pyproject ``inference`` extra) stay declared."""

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
T5_TOKENIZER_DEPS = {"sentencepiece", "protobuf"}


def _extra(name: str) -> list[Requirement]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(rf"^{name} = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert block, f"extra {name!r} not found in pyproject.toml"
    return [Requirement(dep) for dep in re.findall(r'"([^"]+)"', block.group(1))]


def _requirements() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {Requirement(line).name for line in lines if line.strip() and not line.startswith("#")}


def test_t5_tokenizer_deps_declared() -> None:
    assert {r.name for r in _extra("inference")} >= T5_TOKENIZER_DEPS
    assert any(
        r.name == "open-vision-banana" and "inference" in r.extras for r in _extra("training")
    ), "training extra must pull in open-vision-banana[inference]"
    assert _requirements() >= T5_TOKENIZER_DEPS
