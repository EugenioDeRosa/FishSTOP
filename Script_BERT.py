# -*- coding: utf-8 -*-
"""FishSTOP: EDA e addestramento DistilBERT multilingua su Google Colab.

Prima di eseguire:
1. abilita una GPU in Colab;
2. assicurati che FishSTOP sia in /content/drive/MyDrive/FishSTOP;
3. rigenera fishstop_train_complete.csv e il relativo manifest dal progetto.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Configurazione riproducibile
# ---------------------------------------------------------------------------

SEED = 42
BASE_MODEL = "distilbert/distilbert-base-multilingual-cased"
EPOCHS = 4

PROJECT_DIR = Path("/content/drive/MyDrive/FishSTOP")
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "fishstop_train_complete.csv"
MANIFEST_PATH = DATASET_PATH.with_suffix(".manifest.json")

# Il training avviene sul disco locale di Colab, più veloce di Google Drive.
MODEL_DIR = Path("/content/fishstop-distilbert-multilingual")
EDA_DIR = MODEL_DIR / "eda"
DRIVE_MODEL_DIR = PROJECT_DIR / "models" / "fishstop-distilbert-multilingual"

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(SEED)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_dependencies() -> None:
    """Installa soltanto le dipendenze necessarie a training ed EDA."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.46.0",
            "datasets>=3.0.0",
            "accelerate>=1.0.0",
            "scikit-learn>=1.2.0",
            "pandas>=2.0.0",
            "seaborn>=0.13.0",
            "matplotlib>=3.7.0",
            "beautifulsoup4>=4.12.0",
            "langdetect>=1.0.9",
        ],
        check=True,
    )


def mount_and_validate_drive() -> None:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    print("Progetto:", PROJECT_DIR)
    print("Dataset:", DATASET_PATH)
    print("Manifest:", MANIFEST_PATH)

    if not PROJECT_DIR.exists():
        raise FileNotFoundError(f"Progetto FishSTOP non trovato: {PROJECT_DIR}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset di training non trovato: {DATASET_PATH}")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest del dataset non trovato: {MANIFEST_PATH}")

    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)
    actual_hash = sha256_file(DATASET_PATH)
    expected_hash = manifest.get("dataset_sha256")
    print("SHA-256 dataset :", actual_hash)
    print("SHA-256 manifest:", expected_hash)
    if actual_hash != expected_hash:
        raise ValueError("Il CSV non corrisponde al manifest: rigenera il dataset completo")


def configure_runtime() -> None:
    import numpy as np
    import torch

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    else:
        raise RuntimeError(
            "GPU non disponibile. In Colab seleziona Runtime > Change runtime type > GPU."
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------------------------
# 2. EDA: dati e controlli di qualità
# ---------------------------------------------------------------------------

def infer_language(text: str) -> str:
    """Stima esplorativa della lingua; non viene usata come feature del modello."""
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = SEED
    sample = re.sub(r"https?://\S+|www\.\S+", " ", str(text))[:5000].strip()
    if len(sample) < 20:
        return "unknown"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def transformer_token_lengths(texts: list[str], tokenizer, batch_size: int = 64) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
    return lengths


def save_figure(figure, name: str) -> None:
    import matplotlib.pyplot as plt

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(EDA_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(EDA_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(figure)


def save_table(frame, name: str, index: bool = False) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EDA_DIR / f"{name}.csv", index=index)
    try:
        (EDA_DIR / f"{name}.tex").write_text(
            frame.to_latex(index=index, escape=True),
            encoding="utf-8",
        )
    except Exception as error:
        print(f"Tabella LaTeX {name} non generata: {error}")


def run_eda():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from transformers import AutoTokenizer

    from src.train import load_training_dataframe

    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    # Usa la stessa validazione e normalizzazione applicata dal training.
    df = load_training_dataframe(DATASET_PATH)
    df["label_name"] = df["label"].map({0: "Legittima", 1: "Malevola"})
    df["characters"] = df["text"].str.len()
    df["words"] = df["text"].str.findall(r"\b\w+\b").str.len()
    df["urls"] = df["text"].str.count(r"(?i)\b(?:https?://|www\.)")
    df["language_inferred"] = df["text"].map(infer_language)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    df["tokens"] = transformer_token_lengths(df["text"].tolist(), tokenizer)
    df["requires_multiple_windows"] = df["tokens"] > 512

    normalized_hash = df["text"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    exact_duplicates = int(normalized_hash.duplicated(keep=False).sum())
    split_leaks = int(
        df.assign(_hash=normalized_hash).groupby("_hash")["split"].nunique().gt(1).sum()
    )
    label_conflicts = int(
        df.assign(_hash=normalized_hash).groupby("_hash")["label"].nunique().gt(1).sum()
    )
    source_leaks = (
        int(df.groupby("source")["split"].nunique().gt(1).sum())
        if "source" in df.columns
        else 0
    )
    campaign_leaks = (
        int(df.groupby("campaign_id")["split"].nunique().gt(1).sum())
        if "campaign_id" in df.columns
        else 0
    )

    quality = {
        "rows": int(len(df)),
        "classes": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "splits": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "sources": int(df["source"].nunique()) if "source" in df.columns else None,
        "inferred_languages": int(df["language_inferred"].nunique()),
        "exact_duplicate_rows": exact_duplicates,
        "exact_hashes_across_splits": split_leaks,
        "exact_label_conflicts": label_conflicts,
        "sources_across_splits": source_leaks,
        "campaigns_across_splits": campaign_leaks,
        "emails_over_512_tokens": int(df["requires_multiple_windows"].sum()),
        "emails_over_512_tokens_percent": round(
            100 * float(df["requires_multiple_windows"].mean()), 2
        ),
        "token_length_median": float(df["tokens"].median()),
        "token_length_p95": float(df["tokens"].quantile(0.95)),
        "token_length_max": int(df["tokens"].max()),
        "language_note": (
            "Lingua stimata automaticamente solo per EDA; non usata dal modello "
            "e soggetta a errore su testi brevi o misti."
        ),
    }
    (EDA_DIR / "eda_summary.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if split_leaks or label_conflicts or source_leaks or campaign_leaks:
        raise ValueError(f"EDA interrotta: possibile leakage o conflitto rilevato: {quality}")

    class_split = (
        df.groupby(["split", "label_name"])
        .size()
        .rename("emails")
        .reset_index()
    )
    save_table(class_split, "tabella_classi_split")

    source_labels = (
        df.groupby(["source", "label_name"])
        .size()
        .rename("emails")
        .reset_index()
        .sort_values("emails", ascending=False)
    )
    save_table(source_labels, "tabella_fonti_classi")

    language_counts = (
        df.groupby(["language_inferred", "label_name"])
        .size()
        .rename("emails")
        .reset_index()
        .sort_values("emails", ascending=False)
    )
    save_table(language_counts, "tabella_lingue_stimate")

    length_stats = (
        df.groupby(["split", "label_name"])[["characters", "words", "tokens", "urls"]]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
        .reset_index()
    )
    length_stats.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in length_stats.columns
    ]
    save_table(length_stats, "tabella_statistiche_lunghezza")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    sns.barplot(data=class_split, x="split", y="emails", hue="label_name", ax=axis)
    axis.set(title="Distribuzione delle classi per split", xlabel="Split", ylabel="Email")
    axis.legend(title="Classe")
    save_figure(figure, "figura_classi_per_split")

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=source_labels,
        y="source",
        x="emails",
        hue="label_name",
        ax=axis,
    )
    axis.set(title="Composizione del dataset per fonte", xlabel="Email", ylabel="Fonte")
    axis.legend(title="Classe")
    save_figure(figure, "figura_fonti_per_classe")

    top_languages = (
        df["language_inferred"].value_counts().head(10).index
    )
    language_plot = language_counts[
        language_counts["language_inferred"].isin(top_languages)
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=language_plot,
        x="language_inferred",
        y="emails",
        hue="label_name",
        ax=axis,
    )
    axis.set(
        title="Prime 10 lingue stimate",
        xlabel="Codice lingua stimato",
        ylabel="Email",
    )
    axis.legend(title="Classe")
    save_figure(figure, "figura_lingue_stimate")

    plot_limit = max(513, float(df["tokens"].quantile(0.99)))
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df[df["tokens"] <= plot_limit],
        x="tokens",
        hue="label_name",
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
        ax=axis,
    )
    axis.axvline(512, color="black", linestyle="--", linewidth=1.2, label="512 token")
    axis.set(
        title="Distribuzione della lunghezza delle email (fino al 99° percentile)",
        xlabel="Token DistilBERT",
        ylabel="Densità",
    )
    save_figure(figure, "figura_distribuzione_token")

    figure, axis = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="split",
        y=np.log1p(df["tokens"]),
        hue="label_name",
        showfliers=False,
        ax=axis,
    )
    axis.set(
        title="Lunghezza delle email per split e classe",
        xlabel="Split",
        ylabel="log(1 + token)",
    )
    axis.legend(title="Classe")
    save_figure(figure, "figura_token_per_split")

    # Versione arricchita utile per analisi successive, non usata dal training.
    df.to_csv(EDA_DIR / "dataset_eda_arricchito.csv", index=False)
    print("\nRiepilogo EDA:")
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    return df, quality


# ---------------------------------------------------------------------------
# 3. Addestramento, calibrazione e risultati finali
# ---------------------------------------------------------------------------

def save_training_figures(metadata: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    metrics = metadata["test_metrics"]
    confusion = np.asarray(metrics["confusion_matrix"])

    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legittima", "Malevola"],
        yticklabels=["Legittima", "Malevola"],
        ax=axis,
    )
    axis.set(
        title="Matrice di confusione sul test set",
        xlabel="Classe predetta",
        ylabel="Classe reale",
    )
    save_figure(figure, "figura_matrice_confusione_test")

    metric_names = ["accuracy", "precision", "recall", "f1"]
    metric_frame = pd.DataFrame(
        {
            "metrica": metric_names,
            "valore": [float(metrics[name]) for name in metric_names],
        }
    )
    save_table(metric_frame, "tabella_metriche_test")
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=metric_frame, x="metrica", y="valore", color="#2878B5", ax=axis)
    axis.set(title="Prestazioni sul test set", xlabel="Metrica", ylabel="Valore", ylim=(0, 1))
    for patch, value in zip(axis.patches, metric_frame["valore"]):
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
        )
    save_figure(figure, "figura_metriche_test")


def run_training_and_export() -> dict:
    from src.train import run_training

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = run_training(
        dataset_path=DATASET_PATH,
        output_dir=MODEL_DIR,
        base_model=BASE_MODEL,
        epochs=EPOCHS,
    )
    save_training_figures(metadata)

    # I checkpoint intermedi non servono dopo che il modello migliore è stato salvato.
    for checkpoint in MODEL_DIR.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)

    DRIVE_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MODEL_DIR, DRIVE_MODEL_DIR, dirs_exist_ok=True)
    print("\nModello ed EDA salvati in:", DRIVE_MODEL_DIR)
    print("\nMetriche finali sul test:")
    print(json.dumps(metadata["test_metrics"], indent=2, ensure_ascii=False))
    print("\nCalibrazione:")
    print(json.dumps(metadata["calibration"], indent=2, ensure_ascii=False))
    return metadata


def main() -> None:
    mount_and_validate_drive()
    install_dependencies()
    os.chdir(PROJECT_DIR)
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    configure_runtime()

    # Pulisce soltanto l'area temporanea locale della nuova esecuzione.
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    run_eda()
    run_training_and_export()


if __name__ == "__main__":
    main()
