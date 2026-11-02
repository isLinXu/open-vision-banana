"""Tests for external paper-protocol judges without network access."""

from __future__ import annotations

import json

import pytest
import torch

from open_vision_banana.evaluation.gemini import GeminiPaperJudge
from open_vision_banana.evaluation.generation_preservation import (
    evaluate_pairwise_preferences,
)


def _transport_with_text(text: str, captured: list):
    def transport(request, timeout):
        captured.append((request, timeout, json.loads(request.data)))
        return json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()

    return transport


def test_saco_presence_uses_paper_prompt_and_header_key() -> None:
    captured = []
    judge = GeminiPaperJudge(
        api_key="secret",
        transport=_transport_with_text("(A): Yes", captured),
    )
    present = judge.classify_noun_phrase(torch.zeros(3, 2, 2), "red cup")

    assert present is True
    request, timeout, payload = captured[0]
    assert timeout == 60.0
    assert request.get_header("X-goog-api-key") == "secret"
    prompt = payload["contents"][0]["parts"][1]["text"]
    assert "Is there an instance of <red cup> visible" in prompt
    assert "(A): Yes, (B): No" in prompt


def test_saco_presence_parses_negative_response() -> None:
    judge = GeminiPaperJudge(
        api_key="secret",
        transport=_transport_with_text("The answer is B: No", []),
    )
    assert judge.classify_noun_phrase(torch.zeros(3, 2, 2), "cat") is False


def test_reasonseg_rewrite_returns_descriptive_reference() -> None:
    judge = GeminiPaperJudge(
        api_key="secret",
        transport=_transport_with_text("the person holding the umbrella", []),
    )
    result = judge.rewrite_reasoning_query(torch.zeros(3, 2, 2), "Who stays dry?")
    assert result == "the person holding the umbrella"


def test_gemini_judge_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key required"):
        GeminiPaperJudge()


def test_pairwise_win_rate_counts_ties_as_half() -> None:
    metrics = evaluate_pairwise_preferences(["candidate", "candidate", "baseline", "tie"])
    assert metrics == {
        "wins": 2,
        "losses": 1,
        "ties": 1,
        "total": 4,
        "win_rate": pytest.approx(0.625),
    }


def test_pairwise_win_rate_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown pairwise decisions"):
        evaluate_pairwise_preferences(["candidate", "maybe"])
