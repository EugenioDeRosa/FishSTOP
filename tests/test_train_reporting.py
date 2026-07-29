import json

import pytest
import torch

from src.train import (
    evaluate_prevalence_scenarios,
    latest_checkpoint,
    prepare_training_run,
)


def test_latest_checkpoint_ignores_incomplete_directories(tmp_path):
    incomplete = tmp_path / "checkpoint-300"
    incomplete.mkdir()
    complete = tmp_path / "checkpoint-200"
    complete.mkdir()
    complete.joinpath("trainer_state.json").write_text("{}", encoding="utf-8")

    assert latest_checkpoint(tmp_path) == complete


def test_prepare_training_run_resumes_only_matching_recipe(tmp_path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    checkpoint.joinpath("trainer_state.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("training_run_config.json").write_text(
        json.dumps(
            {
                "dataset_sha256": "same-dataset",
                "base_model": "base-model",
                "epochs": 4,
            }
        ),
        encoding="utf-8",
    )

    resolved = prepare_training_run(
        tmp_path,
        dataset_sha256="same-dataset",
        base_model="base-model",
        epochs=4,
        resume_from_checkpoint="auto",
    )
    assert resolved == checkpoint

    with pytest.raises(ValueError, match="does not match"):
        prepare_training_run(
            tmp_path,
            dataset_sha256="changed-dataset",
            base_model="base-model",
            epochs=4,
            resume_from_checkpoint="auto",
        )


def test_prevalence_scenarios_use_calibrated_threshold():
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [0.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    calibration = {"temperature": 1.0, "threshold": 0.7}

    results = evaluate_prevalence_scenarios(
        logits,
        labels,
        calibration,
        prevalence_scenarios=(0.50,),
    )

    assert results[0]["achieved_phishing_prevalence"] == 0.5
    assert results[0]["precision"] == 1.0
    assert results[0]["recall"] == 1.0
    assert results[0]["confusion_matrix"] == [[2, 0], [0, 2]]
