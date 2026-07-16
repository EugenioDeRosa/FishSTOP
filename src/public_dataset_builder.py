"""
public_dataset_builder.py - Costruzione dataset pubblico per training FishStop.

Produces a clean CSV with columns:
  text,label,source,source_file,text_hash

Label:
  0 = legitimate
  1 = phishing
"""

from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import mailbox
import os
import re
import tarfile
import tempfile
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback intentionally simple
    BeautifulSoup = None


ROOT = Path("data")
SOURCES_DIR = ROOT / "training_sources"
PROCESSED_DIR = ROOT / "processed"
DEFAULT_OUTPUT_CSV = PROCESSED_DIR / "fishstop_train.csv"
FINAL_COLUMNS = ["text", "label", "source", "source_file", "text_hash"]
MIN_TEXT_CHARS = 40
MAX_TEXT_CHARS = 200_000
INVALID_TEXT_VALUES = {
    "",
    "empty",
    "nan",
    "none",
    "null",
    "no content",
    "no text",
}

SPAMASSASSIN_URLS = {
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "hard_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
}

NAZARIO_URLS = [
    "https://monkey.org/~jose/phishing/phishing-2025",
    "https://monkey.org/~jose/phishing/phishing-2024",
    "https://monkey.org/~jose/phishing/phishing-2023",
    "https://monkey.org/~jose/phishing/phishing-2022",
    "https://monkey.org/~jose/phishing/phishing-2021",
    "https://monkey.org/~jose/phishing/phishing-2020",
]

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
KAGGLE_DATASET = "naserabdullahalam/phishing-email-dataset"
KAGGLE_PHISHING_LEGITIMATE_DATASET = "kuladeep19/phishing-and-legitimate-emails-dataset"
KAGGLE_SUBHAJOURNAL_PHISHING_EMAILS_DATASET = "subhajournal/phishingemails"
KAGGLE_COMBINED_OVERLAP_SOURCES = {"enron", "nazario", "spamassassin"}
PHISHING_POT_ZIP_URL = "https://github.com/rf-peixoto/phishing_pot/archive/refs/heads/main.zip"

KAGGLE_SCHEMAS = {
    KAGGLE_DATASET: {
        "filename": "phishing_email.csv",
        "text_column": "text_combined",
        "label_column": "label",
    },
    KAGGLE_PHISHING_LEGITIMATE_DATASET: {
        "filename": "phishing_legit_dataset_KD_10000.csv",
        "text_column": "text",
        "label_column": "label",
    },
    KAGGLE_SUBHAJOURNAL_PHISHING_EMAILS_DATASET: {
        "filename": "Phishing_Email.csv",
        "text_column": "Email Text",
        "label_column": "Email Type",
    },
}


@dataclass
class BuildResult:
    source: str
    rows: int
    added: int
    skipped: int
    errors: int
    message: str


def _ensure_dirs() -> None:
    for path in [SOURCES_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _strip_html(value: str) -> str:
    if not value:
        return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(value, "lxml")
        except Exception:
            soup = BeautifulSoup(value, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        return soup.get_text(separator=" ")
    value = re.sub(r"<[^>]+>", " ", value)
    return (
        value.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )


def normalize_text(text: str) -> str:
    if text is None or pd.isna(text):
        return ""
    text = str(text)
    if re.search(r"<[a-zA-Z][\s>/]", text):
        text = _strip_html(text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        char if char in "\n\t" or unicodedata.category(char) != "Cc" else " "
        for char in text
    )
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def template_hash(text: str) -> str:
    """
    Fingerprint aggressiva per template con URL, email, numeri o tracking id diversi.
    Preserva lettere Unicode per non far collidere tutti i testi non inglesi.
    """
    normalized = normalize_text(text)
    template = re.sub(r"https?://\S+|www\.\S+", " URL ", normalized)
    template = re.sub(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+", " EMAIL ", template)
    template = re.sub(r"\b[0-9a-f]{16,}\b", " ID ", template)
    template = re.sub(r"\b\d+\b", " NUM ", template)
    template = re.sub(r"[^\w]+", " ", template, flags=re.UNICODE)
    template = re.sub(r"\s+", " ", template.replace("_", " ")).strip()
    fingerprint = template or normalized
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest() if fingerprint else ""

def _extract_email_text(raw: bytes) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    subject = str(msg.get("Subject") or "")
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        parts = msg.walk()
    else:
        parts = [msg]

    for part in parts:
        content_disposition = str(part.get_content_disposition() or "")
        if "attachment" in content_disposition.lower():
            continue

        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue

        try:
            payload = part.get_content()
        except Exception:
            raw_payload = part.get_payload(decode=True)
            if not raw_payload:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                payload = raw_payload.decode(charset, errors="ignore")
            except LookupError:
                try:
                    payload = raw_payload.decode("utf-8", errors="ignore")
                except Exception:
                    payload = raw_payload.decode("latin-1", errors="ignore")

        if content_type == "text/plain":
            plain_parts.append(str(payload))
        else:
            html_parts.append(str(payload))

    body = "\n".join(plain_parts) if plain_parts else _strip_html("\n".join(html_parts))
    return normalize_text(f"{subject} {body}")


def _download(url: str, dest: Path, progress: Callable[[str], None] | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if progress:
            progress(f"Already present: {dest.name}")
        return dest
    if progress:
        progress(f"Download: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _rows_from_tar_emails(archive_path: Path, label: int, source: str) -> Iterable[dict]:
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            text = _extract_email_text(raw)
            yield _row(text, label, source, member.name)


def _rows_from_phishing_pot_zip(archive_path: Path) -> Iterable[dict]:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if "/email/" not in normalized or not normalized.lower().endswith(".eml"):
                continue
            raw = archive.read(name)
            text = _extract_email_text(raw)
            yield _row(text, 1, "github_phishing_pot", normalized)


def _rows_from_mbox(path: Path, label: int, source: str) -> Iterable[dict]:
    try:
        mbox = mailbox.mbox(str(path), create=False)
        for idx, msg in enumerate(mbox):
            raw = msg.as_bytes(policy=policy.default)
            text = _extract_email_text(raw)
            yield _row(text, label, source, f"{path.name}#{idx}")
    except Exception:
        raw = path.read_bytes()
        for idx, chunk in enumerate(re.split(rb"\nFrom .*\n", raw)):
            if not chunk.strip():
                continue
            text = _extract_email_text(chunk)
            yield _row(text, label, source, f"{path.name}#{idx}")


def _parse_binary_label(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return int(numeric)
        return None
    except (TypeError, ValueError):
        pass

    normalized = re.sub(r"[\s_\-]+", " ", str(value).strip().lower())
    legitimate = {
        "0", "benign", "ham", "legit", "legitimate", "non phishing",
        "not phishing", "safe", "safe email",
    }
    phishing = {
        "1", "fraud", "fraudulent", "malicious", "phish", "phishing",
        "phishing email", "scam", "spam",
    }
    if normalized in legitimate:
        return 0
    if normalized in phishing:
        return 1
    return None


def _normalize_label(value) -> int:
    label = _parse_binary_label(value)
    if label is None:
        raise ValueError(f"Unsupported binary label: {value!r}")
    return label


def _row(text: str, label: int, source: str, source_file: str) -> dict:
    normalized_text = normalize_text(text)
    normalized_label = _normalize_label(label)
    return {
        "text": normalized_text,
        "label": normalized_label,
        "source": source,
        "source_file": source_file,
        "text_hash": text_hash(normalized_text) if normalized_text else "",
    }


def _clean_dataset_frame(
    df: pd.DataFrame,
    min_chars: int = MIN_TEXT_CHARS,
    max_chars: int = MAX_TEXT_CHARS,
) -> tuple[pd.DataFrame, dict]:
    """Normalizza e filtra il dataset senza nascondere conflitti di etichetta."""
    working = df.copy()
    for column, default in {
        "text": "",
        "label": None,
        "source": "unknown",
        "source_file": "",
    }.items():
        if column not in working:
            working[column] = default

    working["text"] = working["text"].map(normalize_text)
    parsed_labels = working["label"].map(_parse_binary_label)
    lengths = working["text"].str.len()
    invalid_text = working["text"].isin(INVALID_TEXT_VALUES)
    invalid_label = parsed_labels.isna()
    too_short = ~invalid_text & lengths.lt(min_chars)
    too_long = lengths.gt(max_chars)
    valid = ~(invalid_text | invalid_label | too_short | too_long)

    stats = {
        "invalid_text": int(invalid_text.sum()),
        "invalid_label": int(invalid_label.sum()),
        "too_short": int(too_short.sum()),
        "too_long": int(too_long.sum()),
        "exact_duplicates": 0,
        "exact_label_conflicts": 0,
    }
    working = working.loc[valid].copy()
    working["label"] = parsed_labels.loc[valid].astype(int)
    working["source"] = working["source"].fillna("unknown").astype(str)
    working["source_file"] = working["source_file"].fillna("").astype(str)
    working["text_hash"] = working["text"].apply(text_hash)

    label_counts = working.groupby("text_hash")["label"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index)
    if conflict_hashes:
        conflict_mask = working["text_hash"].isin(conflict_hashes)
        stats["exact_label_conflicts"] = int(conflict_mask.sum())
        working = working.loc[~conflict_mask]

    before_dedup = len(working)
    working = working.drop_duplicates(subset=["text_hash"], keep="first")
    stats["exact_duplicates"] = int(before_dedup - len(working))
    return working[FINAL_COLUMNS].reset_index(drop=True), stats


def _normalize_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    return _clean_dataset_frame(df)[0]


def _dedupe_templates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Deduplica quasi-duplicati/template entro la stessa label.
    Se la stessa fingerprint appare con label diverse, scarta tutte le righe
    coinvolte: meglio perdere campioni ambigui che insegnare segnali contraddittori.
    """
    working, clean_info = _clean_dataset_frame(df)
    if working.empty:
        return working, {
            "template_duplicates": 0,
            "label_conflicts": 0,
            **clean_info,
        }

    working["_template_hash"] = working["text"].apply(template_hash)
    before = len(working)
    label_counts = working.groupby("_template_hash")["label"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index)
    if conflict_hashes:
        working = working[~working["_template_hash"].isin(conflict_hashes)]

    after_conflicts = len(working)
    working = working.drop_duplicates(subset=["label", "_template_hash"], keep="first")
    stats = {
        "template_duplicates": int(after_conflicts - len(working)),
        "label_conflicts": int(before - after_conflicts),
        **clean_info,
    }
    return working[FINAL_COLUMNS].reset_index(drop=True), stats


def _append_rows(
    rows: Iterable[dict],
    existing_labels: dict[str, int | None],
    min_chars: int,
    max_chars: int = MAX_TEXT_CHARS,
) -> tuple[list[dict], int, int]:
    added_rows = []
    skipped = 0
    errors = 0
    for row in rows:
        try:
            text = normalize_text(row.get("text", ""))
            label = _parse_binary_label(row.get("label"))
            if (
                label is None
                or text in INVALID_TEXT_VALUES
                or len(text) < min_chars
                or len(text) > max_chars
            ):
                skipped += 1
                continue

            h = text_hash(text)
            if h in existing_labels:
                if existing_labels[h] != label:
                    existing_labels[h] = None
                    row = {**row, "text": text, "label": label, "text_hash": h}
                    added_rows.append(row)
                else:
                    skipped += 1
                continue

            existing_labels[h] = label
            row = {**row, "text": text, "label": label, "text_hash": h}
            added_rows.append(row)
        except Exception:
            errors += 1
    return added_rows, skipped, errors


def _load_existing(output_csv: Path) -> tuple[list[dict], dict[str, int | None]]:
    if not output_csv.exists():
        return [], {}
    df = _normalize_dataset_frame(pd.read_csv(output_csv))
    rows = df.to_dict("records")
    labels = dict(zip(df["text_hash"].astype(str), df["label"].astype(int)))
    return rows, labels


def _save_rows(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = _normalize_dataset_frame(pd.DataFrame(rows, columns=FINAL_COLUMNS))
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

def add_spamassassin(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    include_hard_ham: bool = True,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    added_total = skipped_total = errors_total = 0
    sources = ["easy_ham"] + (["hard_ham"] if include_hard_ham else [])

    for name in sources:
        archive = _download(
            SPAMASSASSIN_URLS[name],
            SOURCES_DIR / "legitimate" / "spamassassin" / f"{name}.tar.bz2",
            progress,
        )
        rows, skipped, errors = _append_rows(
            _rows_from_tar_emails(archive, label=0, source=f"spamassassin_{name}"),
            hashes,
            min_chars,
        )
        all_rows.extend(rows)
        added_total += len(rows)
        skipped_total += skipped
        errors_total += errors

    _save_rows(all_rows, output_csv)
    return BuildResult("spamassassin", len(all_rows), added_total, skipped_total, errors_total, "SpamAssassin ham importato")


def add_nazario(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    urls: list[str] | None = None,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    added_total = skipped_total = errors_total = 0
    urls = urls or NAZARIO_URLS

    for url in urls:
        name = url.rstrip("/").split("/")[-1]
        dest = SOURCES_DIR / "phishing" / "nazario" / name
        try:
            path = _download(url, dest, progress)
            rows, skipped, errors = _append_rows(
                _rows_from_mbox(path, label=1, source=f"nazario_{name}"),
                hashes,
                min_chars,
            )
            all_rows.extend(rows)
            added_total += len(rows)
            skipped_total += skipped
            errors_total += errors
        except Exception as exc:
            errors_total += 1
            if progress:
                progress(f"Error Nazario {name}: {exc}")

    _save_rows(all_rows, output_csv)
    return BuildResult("nazario", len(all_rows), added_total, skipped_total, errors_total, "Nazario phishing importato")


def add_kaggle(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    return add_kaggle_dataset(
        dataset_slug=KAGGLE_DATASET,
        source_name="kaggle_phishing_email_dataset",
        result_source="kaggle",
        output_csv=output_csv,
        progress=progress,
        min_chars=min_chars,
    )


def add_kaggle_dataset(
    dataset_slug: str,
    source_name: str,
    result_source: str,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = MIN_TEXT_CHARS,
) -> BuildResult:
    _ensure_dirs()
    try:
        import kagglehub
    except ImportError as exc:
        return BuildResult(result_source, 0, 0, 0, 1, f"kagglehub non installato: {exc}")

    schema = KAGGLE_SCHEMAS.get(dataset_slug)
    if schema is None:
        return BuildResult(result_source, 0, 0, 0, 1, "Schema Kaggle non configurato")

    all_rows, hashes = _load_existing(output_csv)
    if progress:
        progress(f"Download Kaggle: {dataset_slug}")
    dataset_dir = Path(kagglehub.dataset_download(dataset_slug))
    matches = [
        path for path in dataset_dir.rglob("*.csv")
        if path.name.casefold() == schema["filename"].casefold()
    ]
    if not matches:
        return BuildResult(
            result_source, len(all_rows), 0, 0, 1,
            f"File atteso non trovato: {schema['filename']}",
        )

    csv_path = matches[0]
    df = pd.read_csv(csv_path)
    text_col = schema["text_column"]
    label_col = schema["label_column"]
    missing_columns = [column for column in (text_col, label_col) if column not in df.columns]
    if missing_columns:
        return BuildResult(
            result_source, len(all_rows), 0, 0, 1,
            f"Colonne attese mancanti: {', '.join(missing_columns)}",
        )

    rows = []
    invalid_labels = 0
    for idx, record in df.iterrows():
        label = _parse_binary_label(record[label_col])
        if label is None:
            invalid_labels += 1
            continue
        rows.append(_row(record[text_col], label, source_name, f"{csv_path.name}#{idx}"))

    added_rows, skipped, errors = _append_rows(rows, hashes, min_chars)
    skipped += invalid_labels
    all_rows.extend(added_rows)
    _save_rows(all_rows, output_csv)
    final_rows = len(_normalize_dataset_frame(pd.DataFrame(all_rows, columns=FINAL_COLUMNS)))
    return BuildResult(
        result_source,
        final_rows,
        len(added_rows),
        skipped,
        errors,
        "Kaggle importato con schema e label validati",
    )

def add_kaggle_phishing_legitimate(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    return add_kaggle_dataset(
        dataset_slug=KAGGLE_PHISHING_LEGITIMATE_DATASET,
        source_name="kaggle_phishing_and_legitimate_emails",
        result_source="kaggle_phishing_legitimate",
        output_csv=output_csv,
        progress=progress,
        min_chars=min_chars,
    )


def add_kaggle_subhajournal_phishingemails(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    return add_kaggle_dataset(
        dataset_slug=KAGGLE_SUBHAJOURNAL_PHISHING_EMAILS_DATASET,
        source_name="kaggle_subhajournal_phishingemails",
        result_source="kaggle_subhajournal_phishingemails",
        output_csv=output_csv,
        progress=progress,
        min_chars=min_chars,
    )


def add_github_phishing_pot(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    archive = _download(
        PHISHING_POT_ZIP_URL,
        SOURCES_DIR / "phishing" / "phishing_pot" / "phishing_pot_main.zip",
        progress,
    )
    rows, skipped, errors = _append_rows(
        _rows_from_phishing_pot_zip(archive),
        hashes,
        min_chars,
    )
    all_rows.extend(rows)
    _save_rows(all_rows, output_csv)
    return BuildResult(
        "github_phishing_pot",
        len(all_rows),
        len(rows),
        skipped,
        errors,
        "GitHub Phishing Pot importato come phishing.",
    )


def add_enron_sample(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    max_messages: int = 10000,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    archive = _download(ENRON_URL, SOURCES_DIR / "legitimate" / "enron" / "enron_mail_20150507.tar.gz", progress)
    added = skipped = errors = seen = 0
    added_rows = []

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if seen >= max_messages:
                break
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            seen += 1
            try:
                text = _extract_email_text(extracted.read())
                rows, sk, er = _append_rows([_row(text, 0, "enron", member.name)], hashes, min_chars)
                added_rows.extend(rows)
                added += len(rows)
                skipped += sk
                errors += er
            except Exception:
                errors += 1

    all_rows.extend(added_rows)
    _save_rows(all_rows, output_csv)
    return BuildResult("enron", len(all_rows), added, skipped, errors, f"Enron importato, limite {max_messages} email")


def _assign_splits(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Crea split 70/10/20 dopo la deduplica; i dati sintetici restano nel train."""
    result = df.copy()
    result["split"] = "train"
    synthetic_sources = {"kaggle_phishing_and_legitimate_emails"}
    real_mask = ~result["source"].isin(synthetic_sources)
    real_counts = result.loc[real_mask, "label"].value_counts()
    candidate_mask = real_mask if all(real_counts.get(label, 0) >= 10 for label in (0, 1)) else pd.Series(True, index=result.index)

    for label in (0, 1):
        indices = result.index[candidate_mask & result["label"].eq(label)].tolist()
        indices.sort(
            key=lambda index: hashlib.sha256(
                f"{random_state}:{result.at[index, 'text_hash']}".encode("utf-8")
            ).hexdigest()
        )
        n_test = max(1, round(len(indices) * 0.20)) if len(indices) >= 5 else 0
        n_val = max(1, round(len(indices) * 0.10)) if len(indices) >= 10 else 0
        result.loc[indices[:n_test], "split"] = "test"
        result.loc[indices[n_test:n_test + n_val], "split"] = "validation"

    return result


def balance_dataset(
    input_csv: Path = DEFAULT_OUTPUT_CSV,
    output_csv: Path | None = None,
    per_class: int | None = None,
    random_state: int = 42,
) -> dict:
    output_csv = output_csv or input_csv.with_name(f"{input_csv.stem}_balanced.csv")
    df, dedupe_info = _dedupe_templates(pd.read_csv(input_csv))
    counts = df["label"].value_counts() if not df.empty else pd.Series(dtype=int)
    if df.empty or any(counts.get(label, 0) == 0 for label in (0, 1)):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=FINAL_COLUMNS + ["split"]).to_csv(output_csv, index=False)
        return {"rows": 0, "per_class": 0, "output": str(output_csv), **dedupe_info}

    target = min(per_class or int(counts.min()), int(counts.min()))
    sampled_parts = [
        df[df["label"] == label].sample(n=target, random_state=random_state + label)
        for label in (0, 1)
    ]
    balanced = pd.concat(sampled_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    balanced = _assign_splits(balanced, random_state=random_state)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_csv, index=False)
    return {
        "rows": len(balanced),
        "per_class": target,
        "output": str(output_csv),
        **dedupe_info,
        "splits": balanced["split"].value_counts().to_dict(),
    }

def build_balanced_public_dataset(
    selected_sources: list[str],
    output_csv: Path = PROCESSED_DIR / "fishstop_train_balanced.csv",
    staging_csv: Path = DEFAULT_OUTPUT_CSV,
    include_hard_ham: bool = True,
    max_enron: int = 10000,
    progress: Callable[[str], None] | None = None,
) -> dict:
    _ensure_dirs()
    if not selected_sources:
        return {"status": "error", "message": "Select at least one source.", "results": []}

    selected_sources = list(dict.fromkeys(selected_sources))
    skipped_overlap: list[str] = []
    if "kaggle" in selected_sources:
        skipped_overlap = [source for source in selected_sources if source in KAGGLE_COMBINED_OVERLAP_SOURCES]
        selected_sources = [source for source in selected_sources if source not in KAGGLE_COMBINED_OVERLAP_SOURCES]

    for csv_path in {staging_csv, output_csv}:
        if csv_path.exists():
            csv_path.unlink()

    source_steps = {
        "kaggle": lambda: add_kaggle(output_csv=staging_csv, progress=progress),
        "kaggle_phishing_legitimate": lambda: add_kaggle_phishing_legitimate(output_csv=staging_csv, progress=progress),
        "kaggle_subhajournal_phishingemails": lambda: add_kaggle_subhajournal_phishingemails(output_csv=staging_csv, progress=progress),
        "github_phishing_pot": lambda: add_github_phishing_pot(output_csv=staging_csv, progress=progress),
        "nazario": lambda: add_nazario(output_csv=staging_csv, progress=progress),
        "spamassassin": lambda: add_spamassassin(output_csv=staging_csv, include_hard_ham=include_hard_ham, progress=progress),
        "enron": lambda: add_enron_sample(output_csv=staging_csv, max_messages=max_enron, progress=progress),
    }

    results: list[BuildResult] = []
    for source in skipped_overlap:
        results.append(
            BuildResult(
                source,
                0,
                0,
                0,
                0,
                "Source saltata: gia inclusa nel Kaggle Phishing Email Dataset combinato.",
            )
        )

    for source in selected_sources:
        step = source_steps.get(source)
        if step is None:
            results.append(BuildResult(source, 0, 0, 0, 1, "Source non riconosciuta"))
            continue
        if progress:
            progress(f"Import source: {source}")
        results.append(step())

    stats = dataset_stats(staging_csv)
    if stats["legitimate"] == 0 or stats["phishing"] == 0:
        return {
            "status": "error",
            "message": "At least one legitimate source and one phishing source are required to create a balanced 50/50 dataset.",
            "results": results,
            "stats": stats,
        }

    balanced = balance_dataset(staging_csv, output_csv=output_csv)
    return {
        "status": "ok",
        "message": (
            f"Creato dataset bilanciato 50/50 con {balanced['per_class']} email per classe. "
            f"Quasi-duplicati rimossi: {balanced.get('template_duplicates', 0)}; "
            f"conflitti label rimossi: {balanced.get('label_conflicts', 0)}."
        ),
        "results": results,
        "stats": dataset_stats(output_csv),
        "output": balanced["output"],
    }


def dataset_stats(csv_path: Path = DEFAULT_OUTPUT_CSV) -> dict:
    empty_stats = {
        "exists": False,
        "rows": 0,
        "legitimate": 0,
        "phishing": 0,
        "duplicates": 0,
        "template_duplicates": 0,
        "label_conflicts": 0,
        "exact_label_conflicts": 0,
        "invalid_text": 0,
        "invalid_label": 0,
        "too_short": 0,
        "too_long": 0,
        "missing_label": False,
        "sources": {},
        "splits": {},
    }
    if not csv_path.exists():
        return empty_stats

    raw_df = pd.read_csv(csv_path)
    if "label" not in raw_df:
        return {
            **empty_stats,
            "exists": True,
            "rows": len(raw_df),
            "missing_label": True,
            "sources": raw_df["source"].value_counts().to_dict() if "source" in raw_df else {},
        }

    df, clean_info = _clean_dataset_frame(raw_df)
    deduped_df, dedupe_info = _dedupe_templates(raw_df)
    return {
        **empty_stats,
        "exists": True,
        "rows": len(df),
        "legitimate": int((df["label"] == 0).sum()),
        "phishing": int((df["label"] == 1).sum()),
        "duplicates": clean_info["exact_duplicates"],
        "template_duplicates": dedupe_info["template_duplicates"],
        "label_conflicts": dedupe_info["label_conflicts"] + clean_info["exact_label_conflicts"],
        "exact_label_conflicts": clean_info["exact_label_conflicts"],
        "invalid_text": clean_info["invalid_text"],
        "invalid_label": clean_info["invalid_label"],
        "too_short": clean_info["too_short"],
        "too_long": clean_info["too_long"],
        "rows_after_template_dedupe": len(deduped_df),
        "missing_label": False,
        "sources": df["source"].value_counts().to_dict(),
        "splits": raw_df["split"].value_counts().to_dict() if "split" in raw_df else {},
    }
