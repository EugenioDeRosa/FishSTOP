import pandas as pd
import pytest

from src.train import load_training_dataframe


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


def test_training_loader_blocks_normalized_cross_split_leakage(tmp_path):
    rows = _valid_rows()
    rows[0]["text"] = "SAME   EMAIL"
    rows[2]["text"] = "same email"
    path = tmp_path / "leaked.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Data leakage"):
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
