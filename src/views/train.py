import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.eml_dataset_builder import EmlDatasetBuilder


PUBLIC_DATASET = Path("data/processed/fishstop_train_balanced.csv")
PUBLIC_DATASET_FALLBACK = Path("data/processed/fishstop_train.csv")
COMPANY_MODEL_DIR = Path("models/company_model")


def _normalize_training_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["text", "label", "source"])

    text_col = "text" if "text" in df.columns else "xt_combined"
    if text_col not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=["text", "label", "source"])

    out = df[[text_col, "label"]].rename(columns={text_col: "text"}).copy()
    out["text"] = out["text"].astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["text", "label"])
    out["label"] = out["label"].astype(int)
    out = out[out["text"].str.len() > 30]
    out = out[out["label"].isin([0, 1])]
    out["source"] = source_name
    return out[["text", "label", "source"]]


def _build_export_df(include_public: bool, include_custom: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if include_public:
        public_path = PUBLIC_DATASET if PUBLIC_DATASET.exists() else PUBLIC_DATASET_FALLBACK
        if public_path.exists():
            frames.append(_normalize_training_df(pd.read_csv(public_path), public_path.stem))

    if include_custom:
        custom_df = EmlDatasetBuilder().load_for_training()
        frames.append(_normalize_training_df(custom_df, "custom_eml"))

    if not frames:
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _colab_notebook_bytes(csv_name: str = "fishstop_train.csv") -> bytes:
    code = f'''!pip install -q transformers datasets evaluate accelerate scikit-learn pandas matplotlib seaborn

from google.colab import drive
drive.mount("/content/drive")

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

CSV_PATH = "/content/drive/MyDrive/{csv_name}"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "/content/fishstop_bert_model"

df = pd.read_csv(CSV_PATH)
df = df[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.len() > 30].reset_index(drop=True)

print("Shape:", df.shape)
print("\\nLabel distribution:")
print(df["label"].value_counts())

train_val_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.125,
    random_state=42,
    stratify=train_val_df["label"],
)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

train_tok = train_ds.map(tokenize, batched=True).remove_columns(["text"])
val_tok = val_ds.map(tokenize, batched=True).remove_columns(["text"])
test_tok = test_ds.map(tokenize, batched=True).remove_columns(["text"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)
    return {{"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate(test_tok)
print("\\nTest metrics:", metrics)

pred_out = trainer.predict(test_tok)
y_true = pred_out.label_ids
y_pred = np.argmax(pred_out.predictions, axis=1)
print("\\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["Legittima", "Phishing"], zero_division=0))
print("\\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "training_meta.json"), "w", encoding="utf-8") as f:
    import json
    json.dump({{
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "base_model": MODEL_NAME,
        "csv_path": CSV_PATH,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "metrics": metrics,
    }}, f, indent=2)

!cd /content && zip -r fishstop_bert_model.zip fishstop_bert_model
!cp /content/fishstop_bert_model.zip /content/drive/MyDrive/fishstop_bert_model.zip

print("\\nFatto. Scarica da Drive: fishstop_bert_model.zip")
'''

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# FishStop Colab Training\\n",
                    "Carica `fishstop_train.csv` in `MyDrive`, esegui tutto, poi scarica `fishstop_bert_model.zip` e importalo in FishStop.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2).encode("utf-8")


def _safe_extract_model_zip(uploaded_zip) -> tuple[bool, str]:
    raw = uploaded_zip.read()
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        if not names:
            return False, "Zip vuoto."

        root_prefix = ""
        if all(name.startswith("fishstop_bert_model/") for name in names if name.strip()):
            root_prefix = "fishstop_bert_model/"

        required = {"config.json", "tokenizer_config.json"}
        archive_basenames = {Path(name.removeprefix(root_prefix)).name for name in names}
        if not required.issubset(archive_basenames):
            return False, "Zip non valido: mancano config.json o tokenizer_config.json."

        temp_dir = COMPANY_MODEL_DIR.parent / "_company_model_import"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for member in archive.infolist():
            rel = member.filename.removeprefix(root_prefix)
            if not rel or rel.endswith("/"):
                continue
            dest = (temp_dir / rel).resolve()
            if temp_dir.resolve() not in dest.parents:
                shutil.rmtree(temp_dir)
                return False, "Zip non valido: percorso interno non sicuro."
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    if COMPANY_MODEL_DIR.exists():
        backup = COMPANY_MODEL_DIR.with_name(
            f"company_model_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        COMPANY_MODEL_DIR.rename(backup)
    temp_dir.rename(COMPANY_MODEL_DIR)
    return True, f"Modello importato in {COMPANY_MODEL_DIR}."


def render():
    st.header("Colab Training")
    st.markdown(
        "Prepara il CSV, apri il notebook in Google Colab con GPU, poi importa qui lo zip del modello allenato."
    )

    builder = EmlDatasetBuilder()
    stats = builder.stats()
    public_path = PUBLIC_DATASET if PUBLIC_DATASET.exists() else PUBLIC_DATASET_FALLBACK

    st.subheader("1. Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Custom totali", stats["total"])
    c2.metric("Legittime custom", stats["legitimate"])
    c3.metric("Phishing custom", stats["phishing"])

    include_public = st.checkbox(
        "Includi dataset pubblico bilanciato",
        value=public_path.exists(),
        disabled=not public_path.exists(),
    )
    if public_path.exists():
        st.caption(f"Dataset pubblico trovato: `{public_path}`")
    else:
        st.caption("Dataset pubblico non trovato. Puoi crearlo dalla sezione Public Datasets.")

    include_custom = st.checkbox("Includi dataset custom EML", value=stats["total"] > 0)
    export_df = _build_export_df(include_public=include_public, include_custom=include_custom)

    if export_df.empty:
        st.warning("Nessun dato disponibile per esportare il training CSV.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Righe export", len(export_df))
        d2.metric("Legittime", int((export_df["label"] == 0).sum()))
        d3.metric("Phishing", int((export_df["label"] == 1).sum()))

        st.download_button(
            "Scarica fishstop_train.csv",
            data=_csv_bytes(export_df),
            file_name="fishstop_train.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Anteprima export", expanded=False):
            st.dataframe(export_df.head(200), hide_index=True, width="stretch")

    st.divider()
    st.subheader("2. Notebook Colab")
    st.download_button(
        "Scarica notebook Colab",
        data=_colab_notebook_bytes(),
        file_name="fishstop_colab_training.ipynb",
        mime="application/x-ipynb+json",
        use_container_width=True,
    )
    st.markdown(
        "Carica `fishstop_train.csv` in Google Drive, cartella `MyDrive`. "
        "Apri il notebook in Colab, attiva GPU, esegui tutto e scarica `fishstop_bert_model.zip`."
    )

    st.divider()
    st.subheader("3. Import modello")
    uploaded_model = st.file_uploader("Carica fishstop_bert_model.zip", type=["zip"])
    if uploaded_model and st.button("Importa modello aziendale", type="primary", use_container_width=True):
        try:
            ok, message = _safe_extract_model_zip(uploaded_model)
        except zipfile.BadZipFile:
            ok, message = False, "File zip non valido."
        except Exception as exc:
            ok, message = False, f"Import fallito: {exc}"

        if ok:
            st.success(message)
            st.info("Riavvia l'app per caricare il modello aziendale.")
        else:
            st.error(message)
