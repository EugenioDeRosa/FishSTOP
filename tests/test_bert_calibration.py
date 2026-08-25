import pytest
import torch

from src.bert_calibration import (
    calibrated_probabilities,
    classify,
    fit_calibration,
    optimal_f1_threshold,
)
from src.bert_inference import aggregate_chunk_logits


def test_chunk_aggregation_selects_highest_phishing_margin():
    logits = torch.tensor([[4.0, 1.0], [0.5, 2.5], [1.0, 1.2]])

    aggregated = aggregate_chunk_logits(logits, positive_label_id=1)

    assert aggregated.shape == (1, 2)
    assert aggregated.tolist() == [[0.5, 2.5]]


def test_chunk_aggregation_rejects_non_binary_logits():
    with pytest.raises(ValueError, match="binary"):
        aggregate_chunk_logits(torch.zeros((2, 3)))


def test_fit_calibration_produces_valid_runtime_parameters():
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.5],
            [1.0, 1.5],  # intentionally difficult legitimate sample
            [0.5, 2.0],
            [0.0, 3.0],
            [0.2, 2.4],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    calibration = fit_calibration(logits, labels, minimum_coverage=0.5)
    probabilities = calibrated_probabilities(logits, calibration["temperature"])

    assert calibration["method"] == "temperature_scaling"
    assert calibration["positive_label_id"] == 1
    assert calibration["temperature"] > 0
    assert 0 <= calibration["threshold"] <= 1
    assert 0 <= calibration["band"] <= 2
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(labels)))
    assert classify(1.0, calibration["threshold"], calibration["band"]) == "phishing"


def test_optimal_threshold_is_not_hardcoded_to_half():
    threshold, f1 = optimal_f1_threshold([0.10, 0.20, 0.30, 0.40], [0, 0, 1, 1])

    assert threshold == pytest.approx(0.30)
    assert f1 == pytest.approx(1.0)
