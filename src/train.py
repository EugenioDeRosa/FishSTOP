"""Reproducible DistilBERT training pipeline for FishSTOP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.bert_calibration import calibrated_probabilities, fit_calibration, save_calibration
from src.bert_inference import (
    DEFAULT_CHUNK_STRIDE,
    MAX_BERT_TOKENS,
    MAX_EMAIL_CHUNKS,
    predict_email_logits,
)
from src.bert_input import normalize_bert_text


DEFAULT_BASE_MODEL = "distilbert-base-uncased"
DEFAULT_DATASET = Path("data/processed/fishstop_train_complete.csv")
DEFAULT_OUTPUT_DIR = Path("models/fishstop-distilbert")
ID2LABEL = {0: "LEGITIMATE", 1: "MALICIOUS"}
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
REQUIRED_SPLITS = {"train", "validation", "test"}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgressCallback(TrainerCallback):
    """Print stable newline-based progress that remains visible in Colab subprocesses."""

    def __init__(self):
        self.started_at = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        print(
            f"Training loop started: {state.max_steps} optimizer steps across "
            f"{args.num_train_epochs:g} epochs.",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or state.global_step <= 0:
            return
        logs = logs or {}
        elapsed = time.monotonic() - self.started_at
        progress = state.global_step / max(1, state.max_steps)
        remaining = elapsed * (1 - progress) / progress if progress > 0 else 0
        details = []
        for key in ("loss", "eval_loss", "eval_f1", "learning_rate", "grad_norm"):
            if key in logs:
                value = logs[key]
                details.append(f"{key}={value:.6g}" if isinstance(value, (int, float)) else f"{key}={value}")
        print(
            f"[TRAIN] {progress * 100:6.2f}% | epoch {float(state.epoch or 0):.2f}/{args.num_train_epochs:g} "
            f"| step {state.global_step}/{state.max_steps} | elapsed {_format_duration(elapsed)} "
            f"| ETA {_format_duration(remaining)}"
            + (" | " + " | ".join(details) if details else ""),
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.monotonic() - self.started_at
        print(f"Training loop completed in {_format_duration(elapsed)}.", flush=True)


class EmailWeightedTrainer(Trainer):
    """Give every email total weight 1 even when it produces several chunks."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        sample_weight = inputs.pop("sample_weight", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        losses = F.cross_entropy(outputs.logits, labels, reduction="none")
        if sample_weight is None:
            loss = losses.mean()
        else:
            weights = sample_weight.to(losses.device, dtype=losses.dtype)
            loss = (losses * weights).sum() / weights.sum().clamp_min(1e-12)
        return (loss, outputs) if return_outputs else loss


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_training_dataframe(path: str | Path) -> pd.DataFrame:
    """Load and validate the immutable train/validation/test CSV."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    missing = {"text", "label", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label", "split"]).copy()
    df["text"] = df["text"].map(normalize_bert_text)
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"]).issubset({0, 1}):
        raise ValueError("Only labels 0=LEGITIMATE and 1=PHISHING are supported")

    df["split"] = df["split"].astype(str).str.lower().replace({"val": "validation"})
    unknown_splits = set(df["split"]) - REQUIRED_SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown_splits)}")
    missing_splits = REQUIRED_SPLITS - set(df["split"])
    if missing_splits:
        raise ValueError(f"Dataset is missing splits: {sorted(missing_splits)}")

    if "source" in df.columns:
        sources = df["source"].fillna("").astype(str).str.lower()
        if sources.str.startswith("ubuntu_").any():
            raise ValueError("Ubuntu emails are not allowed in the FishSTOP training dataset")
        synthetic = sources.eq("kaggle_phishing_and_legitimate_emails") | sources.str.startswith("synthetic_")
        if (synthetic & df["split"].ne("train")).any():
            raise ValueError("Synthetic emails are allowed only in the train split")
        real_source_splits = df.loc[~synthetic].groupby("source")["split"].nunique()
        leaked_sources = real_source_splits[real_source_splits > 1]
        if not leaked_sources.empty:
            raise ValueError(
                "Source leakage detected: real sources occur in multiple splits: "
                + ", ".join(map(str, leaked_sources.index[:10]))
            )

    df["_normalized_hash"] = df["text"].map(_text_hash)
    split_counts = df.groupby("_normalized_hash")["split"].nunique()
    leaked_hashes = set(split_counts[split_counts > 1].index)
    if leaked_hashes:
        raise ValueError(
            f"Data leakage detected: {len(leaked_hashes)} normalized emails occur in multiple splits"
        )
    df = df.drop_duplicates(subset=["_normalized_hash"], keep="first")

    for split in REQUIRED_SPLITS:
        labels = set(df.loc[df["split"] == split, "label"])
        if labels != {0, 1}:
            raise ValueError(f"Split '{split}' must contain both classes; found {sorted(labels)}")
    return df.drop(columns=["_normalized_hash"]).reset_index(drop=True)


class DistilBERTPhishingTrainer:
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL):
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def _tokenize(self, examples):
        encoded = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_BERT_TOKENS,
            stride=DEFAULT_CHUNK_STRIDE,
            return_overflowing_tokens=True,
        )
        mapping = list(encoded.pop("overflow_to_sample_mapping"))
        selected_positions: list[int] = []
        for sample_index in range(len(examples["text"])):
            positions = [position for position, owner in enumerate(mapping) if owner == sample_index]
            if len(positions) > MAX_EMAIL_CHUNKS:
                offsets = np.linspace(0, len(positions) - 1, MAX_EMAIL_CHUNKS, dtype=int)
                positions = [positions[offset] for offset in offsets]
            selected_positions.extend(positions)
        tokenized = {
            key: [values[position] for position in selected_positions]
            for key, values in encoded.items()
        }
        tokenized["label"] = [examples["label"][mapping[position]] for position in selected_positions]
        selected_counts = {
            sample_index: sum(mapping[position] == sample_index for position in selected_positions)
            for sample_index in range(len(examples["text"]))
        }
        tokenized["sample_weight"] = [
            1.0 / selected_counts[mapping[position]] for position in selected_positions
        ]
        if "email_id" in examples:
            tokenized["email_id"] = [examples["email_id"][mapping[position]] for position in selected_positions]
        return tokenized

    @staticmethod
    def compute_email_metrics(eval_prediction, email_owners):
        logits, labels = eval_prediction
        owners = np.asarray(email_owners, dtype=int)
        labels = np.asarray(labels, dtype=int)
        email_logits = []
        email_labels = []
        for owner in np.unique(owners):
            positions = np.flatnonzero(owners == owner)
            owner_logits = logits[positions]
            best = int(np.argmax(owner_logits[:, 1] - owner_logits[:, 0]))
            email_logits.append(owner_logits[best])
            email_labels.append(labels[positions[0]])
        predictions = np.argmax(np.asarray(email_logits), axis=1)
        email_labels = np.asarray(email_labels)
        return {
            "accuracy": float(accuracy_score(email_labels, predictions)),
            "precision": float(precision_score(email_labels, predictions, zero_division=0)),
            "recall": float(recall_score(email_labels, predictions, zero_division=0)),
            "f1": float(f1_score(email_labels, predictions, zero_division=0)),
        }

    @staticmethod
    def prepare_datasets(df: pd.DataFrame) -> tuple[Dataset, Dataset]:
        train_df = df[df["split"] == "train"][["text", "label"]].reset_index(drop=True)
        validation_df = df[df["split"] == "validation"][["text", "label"]].reset_index(drop=True)
        train_df["email_id"] = np.arange(len(train_df), dtype=int)
        validation_df["email_id"] = np.arange(len(validation_df), dtype=int)
        return Dataset.from_pandas(train_df), Dataset.from_pandas(validation_df)

    def train(
        self,
        df: pd.DataFrame,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        epochs: int = 5,
    ) -> Trainer:
        output_dir = Path(output_dir)
        print("Preparing train and validation datasets...", flush=True)
        train_dataset, validation_dataset = self.prepare_datasets(df)
        print(
            f"Email rows: train={len(train_dataset)}, validation={len(validation_dataset)}. "
            "Tokenizing long messages into weighted chunks...",
            flush=True,
        )
        train_tokenized = train_dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        validation_tokenized = validation_dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=validation_dataset.column_names,
        )
        validation_owners = validation_tokenized["email_id"]
        print(
            f"Tokenized chunks: train={len(train_tokenized)}, validation={len(validation_tokenized)}.",
            flush=True,
        )
        train_tokenized = train_tokenized.remove_columns("email_id")
        validation_tokenized = validation_tokenized.remove_columns("email_id")

        if self.device.type == "cuda":
            batch_size, gradient_accumulation = 16, 1
        elif self.device.type == "mps":
            batch_size, gradient_accumulation = 8, 2
        else:
            batch_size, gradient_accumulation = 4, 4

        args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=2e-5,
            num_train_epochs=epochs,
            weight_decay=0.01,
            optim="adamw_torch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            dataloader_pin_memory=self.device.type == "cuda",
            dataloader_num_workers=0,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=50,
            logging_first_step=True,
            disable_tqdm=False,
            report_to="none",
            remove_unused_columns=False,
            seed=42,
            data_seed=42,
        )
        trainer = EmailWeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=train_tokenized,
            eval_dataset=validation_tokenized,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=lambda prediction: self.compute_email_metrics(prediction, validation_owners),
            callbacks=[TrainingProgressCallback()],
        )
        trainer.train()

        self.model.config.id2label = ID2LABEL
        self.model.config.label2id = LABEL2ID
        self.model.config.fishstop_positive_label_id = 1
        self.model.config.fishstop_preprocessing = "src.bert_input.normalize_bert_text"
        self.model.config.fishstop_chunk_aggregation = "maximum_positive_logit_margin"
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer

    def collect_email_logits(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        logits = []
        chunk_counts = []
        total = len(frame)
        for position, text in enumerate(frame["text"].tolist(), start=1):
            email_logits, chunks = predict_email_logits(
                self.model,
                self.tokenizer,
                text,
                positive_label_id=1,
            )
            logits.append(email_logits.squeeze(0))
            chunk_counts.append(chunks)
            if position % 100 == 0 or position == total:
                print(f"Email-level inference: {position}/{total}")
        return torch.stack(logits), torch.tensor(frame["label"].to_numpy()), chunk_counts


def evaluate_calibrated(
    logits: torch.Tensor,
    labels: torch.Tensor,
    calibration: dict,
) -> dict:
    probabilities = calibrated_probabilities(logits, calibration["temperature"])[:, 1].numpy()
    labels_np = labels.numpy()
    threshold = float(calibration["threshold"])
    predictions = (probabilities >= threshold).astype(int)
    half_band = float(calibration["band"]) / 2
    decided = np.abs(probabilities - threshold) >= half_band
    return {
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "precision": float(precision_score(labels_np, predictions, zero_division=0)),
        "recall": float(recall_score(labels_np, predictions, zero_division=0)),
        "f1": float(f1_score(labels_np, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels_np, predictions).tolist(),
        "selective_coverage": float(decided.mean()),
        "selective_accuracy": float((predictions[decided] == labels_np[decided]).mean()) if decided.any() else None,
    }


def write_model_card(output_dir: Path, metadata: dict) -> None:
    metrics = metadata["test_metrics"]
    output_dir.joinpath("README.md").write_text(
        f"""---
language: en
library_name: transformers
pipeline_tag: text-classification
tags:
- phishing
- email-security
---

# FishSTOP DistilBERT

Binary content classifier for legitimate vs malicious (phishing or spam) email, used as one
signal in FishSTOP. It does not inspect SPF, DKIM,
DMARC, sender reputation, links or attachments and must not be used as a standalone verdict.

- Base model: `{metadata['base_model']}`
- Labels: `0=LEGITIMATE`, `1=MALICIOUS` (phishing, scam or spam)
- Input: normalized email subject plus body
- Long emails: up to {MAX_EMAIL_CHUNKS} evenly spaced overlapping {MAX_BERT_TOKENS}-token windows,
  stride {DEFAULT_CHUNK_STRIDE}, maximum malicious-margin aggregation
- Calibration: temperature scaling on the validation split
- Test F1: {metrics['f1']:.4f}
- Test precision: {metrics['precision']:.4f}
- Test recall: {metrics['recall']:.4f}
- Test selective coverage: {metrics['selective_coverage']:.4f}

The reported probability is meaningful only for data sufficiently similar to the validation
distribution. Performance must be rechecked on recent, external and multilingual email sets.
""",
        encoding="utf-8",
    )


def run_training(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 5,
) -> dict:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    print(f"Loading training dataset: {dataset_path}", flush=True)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    df = load_training_dataframe(dataset_path)
    split_counts = df["split"].value_counts().to_dict()
    print(f"Validated {len(df)} emails. Split counts: {split_counts}", flush=True)
    print(f"Loading base model: {base_model}", flush=True)
    trainer = DistilBERTPhishingTrainer(base_model)
    trainer.train(df, output_dir=output_dir, epochs=epochs)
    trainer.model.config.fishstop_dataset_sha256 = dataset_sha256
    trainer.model.config.fishstop_split_strategy = "source_holdout"
    trainer.model.save_pretrained(output_dir)

    validation_df = df[df["split"] == "validation"]
    print(f"Calibrating probabilities on {len(validation_df)} validation emails...", flush=True)
    validation_logits, validation_labels, validation_chunks = trainer.collect_email_logits(validation_df)
    calibration = fit_calibration(validation_logits, validation_labels, positive_label_id=1)
    calibration.update(
        {
            "model_type": "distilbert",
            "aggregation": "maximum_positive_logit_margin",
            "max_length": MAX_BERT_TOKENS,
            "stride": DEFAULT_CHUNK_STRIDE,
            "max_chunks": MAX_EMAIL_CHUNKS,
            "dataset_sha256": dataset_sha256,
        }
    )
    save_calibration(calibration, output_dir / "calibration.json")

    test_df = df[df["split"] == "test"]
    print(f"Running final evaluation on {len(test_df)} held-out test emails...", flush=True)
    test_logits, test_labels, test_chunks = trainer.collect_email_logits(test_df)
    test_metrics = evaluate_calibrated(test_logits, test_labels, calibration)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "rows": {split: int((df["split"] == split).sum()) for split in sorted(REQUIRED_SPLITS)},
        "labels": ID2LABEL,
        "epochs": int(epochs),
        "best_validation_email_f1": float(trainer.state.best_metric or 0.0),
        "sources_by_split": {
            split: sorted(df.loc[df["split"].eq(split), "source"].dropna().astype(str).unique().tolist())
            if "source" in df else []
            for split in sorted(REQUIRED_SPLITS)
        },
        "validation_multi_chunk_emails": int(sum(count > 1 for count in validation_chunks)),
        "test_multi_chunk_emails": int(sum(count > 1 for count in test_chunks)),
        "calibration": calibration,
        "test_metrics": test_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("training_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_model_card(output_dir, metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train and calibrate FishSTOP DistilBERT")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    arguments = parse_args()
    run_training(
        dataset_path=arguments.dataset,
        output_dir=arguments.output_dir,
        base_model=arguments.base_model,
        epochs=arguments.epochs,
    )
