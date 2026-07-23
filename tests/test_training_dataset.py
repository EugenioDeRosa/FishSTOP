import pandas as pd
import numpy as np
import pytest

from src.train import DistilBERTPhishingTrainer, audit_training_dataframe, load_training_dataframe


def _valid_rows():
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            [
                {"text": f"Legitimate unique message for {split}", "label": 0, "split": split},
                {"text": f"Phishing unique message for {split}", "label": 1, "split": split},
            ]
        )
    return rows


def test_chunk_weights_sum_to_one_per_email():
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": [[1], [2], [3]],
                "attention_mask": [[1], [1], [1]],
                "overflow_to_sample_mapping": [0, 0, 1],
            }

    trainer = object.__new__(DistilBERTPhishingTrainer)
    trainer.tokenizer = FakeTokenizer()
    tokenized = trainer._tokenize({
        "text": ["first", "second"],
        "label": [0, 1],
        "email_id": [10, 11],
    })

    weights_by_email = {}
    for owner, weight in zip(tokenized["email_id"], tokenized["sample_weight"]):
        weights_by_email[owner] = weights_by_email.get(owner, 0.0) + weight
    assert weights_by_email == {10: 1.0, 11: 1.0}


def test_validation_metrics_aggregate_chunks_at_email_level():
    metrics = DistilBERTPhishingTrainer.compute_email_metrics(
        (np.array([[3.0, 0.0], [0.0, 4.0], [4.0, 0.0]]), np.array([1, 1, 0])),
        [0, 0, 1],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_training_loader_preserves_preassigned_splits_and_labels(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame(_valid_rows()).to_csv(path, index=False)

    loaded = load_training_dataframe(path)

    assert loaded.groupby(["split", "label"]).size().to_dict() == {
        ("test", 0): 1,
        ("test", 1): 1,
        ("train", 0): 1,
        ("train", 1): 1,
        ("validation", 0): 1,
        ("validation", 1): 1,
    }


def test_training_audit_flags_single_real_source_per_class():
    rows = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            rows.append(
                {
                    "text": f"{split} class {label} sufficiently long unique email content",
                    "label": label,
                    "split": split,
                    "source": f"only-real-source-{label}",
                }
            )

    audit = audit_training_dataframe(pd.DataFrame(rows))

    assert len(audit["real_train_sources_per_label"]["0"]) == 1
    assert len(audit["real_train_sources_per_label"]["1"]) == 1
    assert any("corpus style" in warning for warning in audit["warnings"])
    assert any("language column" in warning for warning in audit["warnings"])


def test_training_loader_blocks_normalized_cross_split_leakage(tmp_path):
    rows = _valid_rows()
    rows[0]["text"] = "SAME   EMAIL"
    rows[2]["text"] = "same email"
    path = tmp_path / "leaked.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Data leakage"):
        load_training_dataframe(path)


def test_training_loader_allows_real_sources_to_be_mixed_across_splits(tmp_path):
    rows = _valid_rows()
    for row in rows:
        row["source"] = "mixed-public-source"
        row["campaign_id"] = f"{row['split']}:{row['label']}"
    path = tmp_path / "mixed-sources.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    loaded = load_training_dataframe(path)

    assert loaded["source"].nunique() == 1
    assert set(loaded["split"]) == {"train", "validation", "test"}


def test_training_loader_blocks_campaign_cross_split_leakage(tmp_path):
    rows = _valid_rows()
    for index, row in enumerate(rows):
        row["campaign_id"] = f"campaign-{index}"
    rows[0]["campaign_id"] = "leaked-campaign"
    rows[2]["campaign_id"] = "leaked-campaign"
    path = tmp_path / "campaign-leak.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Campaign leakage"):
        load_training_dataframe(path)


def test_training_loader_blocks_synthetic_validation_or_test_rows(tmp_path):
    rows = _valid_rows()
    for row in rows:
        row["source"] = "public_real"
    rows[2]["source"] = "synthetic_modern_v2"
    path = tmp_path / "synthetic_leak.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="only in the train split"):
        load_training_dataframe(path)


def test_training_loader_rejects_ubuntu_sources(tmp_path):
    rows = _valid_rows()
    for index, row in enumerate(rows):
        row["source"] = f"real_source_{index}"
    rows[0]["source"] = "ubuntu_users_2024"
    path = tmp_path / "ubuntu.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Ubuntu emails are not allowed"):
        load_training_dataframe(path)
