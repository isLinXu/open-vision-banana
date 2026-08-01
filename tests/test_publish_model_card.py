"""Regression tests for truthful checkpoint model-card generation."""

from scripts.publish_to_hub import _build_model_card


def test_generated_model_card_uses_checkpoint_specific_claims() -> None:
    card = _build_model_card(
        task="depth",
        description="SSD depth adapter",
        repo_id="example/depth-adapter",
        base_model="segmind/SSD-1B",
        metrics={"relative_depth_pearson": 0.42},
        training_config={"rank": 64, "generation_vision_ratio": "9:1 (experimental)"},
    )

    assert "segmind/SSD-1B" in card
    assert "relative_depth_pearson" in card
    assert "Nano Banana Pro" in card
    assert "Paper metrics do not apply" in card
    assert "model.predict(input_tensor" in card
    assert "arXiv:2604.20329" in card
    assert "Vision Banana Team" not in card
    assert "year={2024}" not in card
    assert "MMDiT-XXL" not in card
    assert "12B base + ~60M" not in card


def test_generated_model_card_does_not_invent_metrics() -> None:
    card = _build_model_card(
        task="normal",
        description="Unvalidated adapter",
        repo_id="example/normal-adapter",
    )

    assert "Metrics not yet reported" in card
    assert "69.9" not in card
    assert "0.929" not in card
