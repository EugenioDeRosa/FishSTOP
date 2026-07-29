"""
public_dataset_builder.py - Costruzione dataset pubblico per training FishStop.

Produces a clean CSV with columns:
  text,label,source,source_file,text_hash

Label:
  0 = legitimate
  1 = phishing
"""

from __future__ import annotations

import calendar
import csv
import gzip
import hashlib
import json
import mailbox
import re
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.bert_input import normalize_bert_text

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback intentionally simple
    BeautifulSoup = None


ROOT = Path("data")
SOURCES_DIR = ROOT / "training_sources"
PROCESSED_DIR = ROOT / "processed"
DEFAULT_OUTPUT_CSV = PROCESSED_DIR / "fishstop_train.csv"
DEFAULT_BALANCED_OUTPUT_CSV = PROCESSED_DIR / "fishstop_train_balanced.csv"
DEFAULT_SYNTHETIC_CSV = PROCESSED_DIR / "fishstop_synthetic_modern_v2.csv"
DEFAULT_LEGITIMATE_HARD_NEGATIVE_CSV = (
    PROCESSED_DIR / "fishstop_synthetic_legitimate_hard_negatives_v1.csv"
)
DEFAULT_COMPLETE_OUTPUT_CSV = PROCESSED_DIR / "fishstop_train_complete.csv"
FINAL_COLUMNS = ["text", "label", "source", "source_file", "text_hash"]
OUTPUT_COLUMNS = FINAL_COLUMNS + ["campaign_id", "split"]
SYNTHETIC_SOURCE_PREFIXES = ("synthetic_",)
SYNTHETIC_SOURCE_NAMES = {"kaggle_phishing_and_legitimate_emails"}
DISALLOWED_SOURCE_NAMES = {"ubuntu_modern_ham"}
MAX_SYNTHETIC_TRAIN_FRACTION = 0.10
CAMPAIGN_SIMILARITY_THRESHOLD = 0.83
MODERN_START_YEAR = 2022
MODERN_END_YEAR = 2025
MIN_TEXT_CHARS = 40
MIN_TEXT_WORDS = 5
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
    f"https://monkey.org/~jose/phishing/phishing-{year}"
    for year in range(MODERN_END_YEAR, MODERN_START_YEAR - 1, -1)
]

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
ZENODO_VALIDATION_URL = (
    "https://zenodo.org/api/records/13474746/files/"
    "Phishing_validation_emails.csv/content"
)
ZENODO_VALIDATION_SHA256 = "ad15f63cb8db2caaee33c442f1ff4488b9444530a4c42ab63bb580016b160bd3"
SPAPHISH_VERSION = 5
SPAPHISH_URL = (
    "https://data.mendeley.com/public-files/datasets/hz2d6gz7pc/files/"
    "f796c8e2-3768-4c2d-8b73-48f0d7771de5/file_downloaded"
)
SPAPHISH_SHA256 = "656b2245d58da72d640680e5c2a168673a130b38607f2a427c773bbb167e995e"
KAGGLE_DATASET = "naserabdullahalam/phishing-email-dataset"
KAGGLE_PHISHING_LEGITIMATE_DATASET = "kuladeep19/phishing-and-legitimate-emails-dataset"
KAGGLE_SUBHAJOURNAL_PHISHING_EMAILS_DATASET = "subhajournal/phishingemails"
KAGGLE_COMBINED_OVERLAP_SOURCES = {"enron", "nazario", "spamassassin"}
PHISHING_POT_COMMIT = "80685cbfe69a1f905707be92e144ba5b71f9ee37"
PHISHING_POT_ZIP_URL = f"https://github.com/rf-peixoto/phishing_pot/archive/{PHISHING_POT_COMMIT}.zip"
UBUNTU_LISTS = ("ubuntu-users", "ubuntu-security-announce")
LEGACY_SOURCE_NAMES = {"kaggle", "kaggle_subhajournal_phishingemails"}
HISTORICAL_SOURCE_NAMES = {"enron", "spamassassin"}
FORCED_SOURCE_SPLITS = {
    "spaphish_train": "train",
    "spaphish_validation": "validation",
    "spaphish_test": "test",
    "zenodo_validation_2024": "validation",
}

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
    return normalize_bert_text(text)

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
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FishSTOP-dataset-builder/1.0"},
    )
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(dest)
    return dest


def _download_verified(
    url: str,
    dest: Path,
    expected_sha256: str,
    progress: Callable[[str], None] | None = None,
) -> Path:
    path = _download(url, dest, progress)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checksum non valido per {path.name}: atteso {expected_sha256}, ottenuto {actual}"
        )
    return path


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


def _rows_from_phishing_pot_zip(archive_path: Path) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if "/email/" not in normalized or not normalized.lower().endswith(".eml"):
                continue
            raw = archive.read(name)
            text = _extract_email_text(raw)
            rows.append(_row(text, 1, "github_phishing_pot", normalized))
    return rows


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


def _rows_from_gzip_mbox(path: Path, label: int, source: str) -> Iterable[dict]:
    raw = gzip.open(path, "rb").read()
    for idx, chunk in enumerate(re.split(rb"(?m)^From .*(?:\r?\n)", raw)):
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
    min_words: int = MIN_TEXT_WORDS,
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
    word_counts = working["text"].str.split().str.len()
    invalid_text = working["text"].isin(INVALID_TEXT_VALUES)
    invalid_label = parsed_labels.isna()
    too_short = ~invalid_text & lengths.lt(min_chars)
    too_few_words = ~invalid_text & word_counts.lt(min_words)
    too_long = lengths.gt(max_chars)
    valid = ~(invalid_text | invalid_label | too_short | too_few_words | too_long)

    stats = {
        "invalid_text": int(invalid_text.sum()),
        "invalid_label": int(invalid_label.sum()),
        "too_short": int(too_short.sum()),
        "too_few_words": int(too_few_words.sum()),
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


def _rows_from_zenodo_validation_frame(frame: pd.DataFrame) -> Iterable[dict]:
    missing = {"Email Text", "Email Type"} - set(frame.columns)
    if missing:
        raise ValueError(f"Schema Zenodo non valido, colonne mancanti: {sorted(missing)}")
    for index, row in frame.iterrows():
        yield _row(
            row.get("Email Text", ""),
            row.get("Email Type"),
            "zenodo_2024",
            f"10.5281/zenodo.13474746#{index}",
        )


def add_zenodo_validation(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = MIN_TEXT_CHARS,
) -> BuildResult:
    """Import the versioned Zenodo corpus; exact duplicates are discarded."""
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    csv_path = _download_verified(
        ZENODO_VALIDATION_URL,
        SOURCES_DIR / "mixed" / "zenodo_validation" / "Phishing_validation_emails.csv",
        ZENODO_VALIDATION_SHA256,
        progress,
    )
    frame = pd.read_csv(csv_path)
    rows, skipped, errors = _append_rows(
        _rows_from_zenodo_validation_frame(frame),
        hashes,
        min_chars,
    )
    all_rows.extend(rows)
    _save_rows(all_rows, output_csv)
    return BuildResult(
        "zenodo_validation",
        len(all_rows),
        len(rows),
        skipped,
        errors,
        "Zenodo importato nel dataset misto; duplicati esatti rimossi.",
    )


def _rows_from_spaphish_frame(frame: pd.DataFrame) -> list[dict]:
    missing = {"subject", "body", "Label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Schema SpaPhish non valido, colonne mancanti: {sorted(missing)}")
    rows: list[dict] = []
    for index, row in frame.iterrows():
        subject = "" if pd.isna(row.get("subject")) else str(row.get("subject"))
        body = "" if pd.isna(row.get("body")) else str(row.get("body"))
        published_hash = row.get("hash")
        if pd.isna(published_hash) or not str(published_hash).strip():
            source_file = f"row-{index}-{text_hash(f'{subject} {body}')[:16]}"
        else:
            source_file = str(published_hash).strip()
        rows.append(
            _row(
                f"{subject} {body}",
                row.get("Label"),
                "spaphish",
                source_file,
            )
        )
    return rows


def add_spaphish(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = MIN_TEXT_CHARS,
) -> BuildResult:
    """Import all valid Spanish emails from the fixed SpaPhish v5 release."""
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    csv_path = _download_verified(
        SPAPHISH_URL,
        SOURCES_DIR / "mixed" / "spaphish" / f"spaphish_v{SPAPHISH_VERSION}.csv",
        SPAPHISH_SHA256,
        progress,
    )
    frame = pd.read_csv(csv_path)
    candidate_rows = _rows_from_spaphish_frame(frame)
    rows, skipped, errors = _append_rows(candidate_rows, hashes, min_chars)
    all_rows.extend(rows)
    _save_rows(all_rows, output_csv)
    return BuildResult(
        "spaphish",
        len(all_rows),
        len(rows),
        skipped,
        errors,
        f"SpaPhish v{SPAPHISH_VERSION} importato senza filtrare l'header data; "
        "lo split viene assegnato successivamente per campagna.",
    )


def add_ubuntu_modern_ham(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    mailing_lists: tuple[str, ...] = UBUNTU_LISTS,
    start_year: int = MODERN_START_YEAR,
    end_year: int = MODERN_END_YEAR,
    progress: Callable[[str], None] | None = None,
    min_chars: int = MIN_TEXT_CHARS,
) -> BuildResult:
    """Importa messaggi recenti da mailing list pubbliche Ubuntu come ham reale."""
    _ensure_dirs()
    all_rows, hashes = _load_existing(output_csv)
    added_total = skipped_total = errors_total = 0

    for list_name in mailing_lists:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                period = f"{year}-{calendar.month_name[month]}"
                url = f"https://lists.ubuntu.com/archives/{list_name}/{period}.txt.gz"
                dest = SOURCES_DIR / "legitimate" / "ubuntu" / list_name / f"{period}.txt.gz"
                try:
                    archive = _download(url, dest, progress)
                    rows, skipped, errors = _append_rows(
                        _rows_from_gzip_mbox(
                            archive,
                            label=0,
                            source=f"ubuntu_{list_name}_{year}",
                        ),
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
                        progress(f"Errore archivio Ubuntu {list_name} {period}: {exc}")

    _save_rows(all_rows, output_csv)
    return BuildResult(
        "ubuntu_modern_ham",
        len(all_rows),
        added_total,
        skipped_total,
        errors_total,
        f"Ubuntu ham pubblico {start_year}-{end_year} importato",
    )


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
        SOURCES_DIR / "phishing" / "phishing_pot" / f"phishing_pot_{PHISHING_POT_COMMIT}.zip",
        progress,
    )
    candidate_rows = _rows_from_phishing_pot_zip(archive)
    rows, skipped, errors = _append_rows(
        candidate_rows,
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
        "GitHub Phishing Pot importato come phishing senza filtrare l'header data.",
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


def _synthetic_source_mask(df: pd.DataFrame) -> pd.Series:
    """Identifica tutte le fonti sintetiche che non devono entrare in val/test."""
    sources = df["source"].fillna("").astype(str).str.lower()
    return sources.isin(SYNTHETIC_SOURCE_NAMES) | sources.str.startswith(SYNTHETIC_SOURCE_PREFIXES)


def _campaign_groups(
    df: pd.DataFrame,
    similarity_threshold: float = CAMPAIGN_SIMILARITY_THRESHOLD,
) -> pd.Series:
    """Raggruppa varianti quasi identiche affinche restino nello stesso split."""
    if df.empty:
        return pd.Series(dtype=str, index=df.index)
    if len(df) == 1:
        return pd.Series([f"campaign:{df.iloc[0]['text_hash']}"], index=df.index, dtype=str)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(4, 5),
        min_df=1 if len(df) < 20 else 2,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(df["text"])
    neighbor_count = min(64, len(df))
    distances, neighbors = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
    ).fit(matrix).kneighbors(matrix)

    parents = list(range(len(df)))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for row_position in range(len(df)):
        for distance, neighbor_position in zip(distances[row_position], neighbors[row_position]):
            if row_position != int(neighbor_position) and 1.0 - float(distance) >= similarity_threshold:
                union(row_position, int(neighbor_position))

    members: dict[int, list[int]] = {}
    for position in range(len(df)):
        members.setdefault(find(position), []).append(position)
    group_names = {
        root: "campaign:" + min(str(df.iloc[position]["text_hash"]) for position in positions)
        for root, positions in members.items()
    }
    return pd.Series(
        [group_names[find(position)] for position in range(len(df))],
        index=df.index,
        dtype=str,
    )


def _assign_splits(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Crea split casuali 70/10/20 stratificati per fonte, classe e campagna."""
    result = df.copy()
    result["split"] = "train"
    synthetic_mask = _synthetic_source_mask(result)
    real_mask = ~synthetic_mask
    result["campaign_id"] = "synthetic:" + result["text_hash"].astype(str)
    if real_mask.any():
        result.loc[real_mask, "campaign_id"] = _campaign_groups(result.loc[real_mask])

    campaign_labels = result.loc[real_mask].groupby("campaign_id")["label"].nunique()
    conflicts = campaign_labels[campaign_labels > 1]
    if not conflicts.empty:
        result = result.loc[
            ~(real_mask & result["campaign_id"].isin(conflicts.index))
        ].copy()
        synthetic_mask = _synthetic_source_mask(result)
        real_mask = ~synthetic_mask

    # If the same near-duplicate campaign occurs in multiple corpora, keep one
    # deterministic representative. This prevents both corpus duplication and
    # contradictory stratum assignments.
    campaign_source_counts = result.loc[real_mask].groupby("campaign_id")["source"].nunique()
    cross_source_campaigns = set(campaign_source_counts[campaign_source_counts > 1].index)
    if cross_source_campaigns:
        cross_source_rows = result.loc[
            real_mask & result["campaign_id"].isin(cross_source_campaigns)
        ]
        keep_indices = set(
            cross_source_rows.sort_values(["campaign_id", "text_hash", "source"])
            .drop_duplicates("campaign_id", keep="first")
            .index
        )
        result = result.loc[
            ~(
                real_mask
                & result["campaign_id"].isin(cross_source_campaigns)
                & ~result.index.isin(keep_indices)
            )
        ].copy()
        synthetic_mask = _synthetic_source_mask(result)
        real_mask = ~synthetic_mask

    real_rows = result.loc[real_mask]
    for (label, source), stratum_rows in real_rows.groupby(["label", "source"], dropna=False):
        groups = stratum_rows.groupby("campaign_id").size().to_dict()
        ordered_groups = sorted(
            groups,
            key=lambda group: (
                -groups[group],
                hashlib.sha256(
                    f"{random_state}:{label}:{source}:{group}".encode("utf-8")
                ).hexdigest(),
            ),
        )
        row_count = len(stratum_rows)
        targets = {
            "test": round(row_count * 0.20),
            "validation": round(row_count * 0.10),
        }
        targets["train"] = row_count - targets["test"] - targets["validation"]
        assigned = {split: 0 for split in targets}
        for group in ordered_groups:
            size = groups[group]
            fitting = [split for split in targets if assigned[split] + size <= targets[split]]
            candidates = fitting or list(targets)
            split = max(
                candidates,
                key=lambda candidate: (
                    (targets[candidate] - assigned[candidate]) / max(targets[candidate], 1),
                    hashlib.sha256(f"{group}:{candidate}".encode("utf-8")).hexdigest(),
                ),
            )
            assigned[split] += size
            result.loc[result["campaign_id"].eq(group), "split"] = split

    campaign_split_counts = result.loc[real_mask].groupby("campaign_id")["split"].nunique()
    if (campaign_split_counts > 1).any():
        raise ValueError("Campaign leakage detected after stratified random splitting")
    return result


def _assign_source_holdout_splits(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Reserve entire real sources for validation/test to measure source generalization."""
    result = df.copy()
    result["split"] = "train"
    synthetic_mask = _synthetic_source_mask(result)
    real_mask = ~synthetic_mask
    result["campaign_id"] = "synthetic:" + result["text_hash"].astype(str)
    if real_mask.any():
        result.loc[real_mask, "campaign_id"] = _campaign_groups(result.loc[real_mask])
    campaign_labels = result.loc[real_mask].groupby("campaign_id")["label"].nunique()
    conflicts = campaign_labels[campaign_labels > 1]
    if not conflicts.empty:
        # Contradictory near-duplicates are ambiguous supervision. Drop the
        # entire campaign rather than arbitrarily trusting either annotation.
        result = result.loc[
            ~(real_mask & result["campaign_id"].isin(conflicts.index))
        ].copy()
        synthetic_mask = _synthetic_source_mask(result)
        real_mask = ~synthetic_mask

    # A near-duplicate campaign imported by more than one corpus would force
    # those sources into the same split. Keep one deterministic representative
    # instead, preserving both source isolation and campaign isolation.
    campaign_source_counts = result.loc[real_mask].groupby("campaign_id")["source"].nunique()
    cross_source_campaigns = set(campaign_source_counts[campaign_source_counts > 1].index)
    if cross_source_campaigns:
        cross_source_rows = result.loc[real_mask & result["campaign_id"].isin(cross_source_campaigns)]
        keep_indices = set(
            cross_source_rows.sort_values(["campaign_id", "text_hash"])
            .drop_duplicates("campaign_id", keep="first")
            .index
        )
        drop_mask = (
            real_mask
            & result["campaign_id"].isin(cross_source_campaigns)
            & ~result.index.isin(keep_indices)
        )
        result = result.loc[~drop_mask].copy()
        synthetic_mask = _synthetic_source_mask(result)
        real_mask = ~synthetic_mask

    forced_mask = real_mask & result["source"].isin(FORCED_SOURCE_SPLITS)
    for source, split in FORCED_SOURCE_SPLITS.items():
        result.loc[real_mask & result["source"].eq(source), "split"] = split

    for label in (0, 1):
        label_rows = result.loc[real_mask & result["label"].eq(label)]
        source_sizes = label_rows.groupby("source").size().to_dict()
        if len(source_sizes) < 3:
            raise ValueError(
                f"Source-held-out splitting requires at least three real sources for label {label}; "
                f"found {sorted(source_sizes)}"
            )
        targets = {
            "train": round(len(label_rows) * 0.70),
            "validation": round(len(label_rows) * 0.10),
            "test": round(len(label_rows) * 0.20),
        }
        assigned = {
            split: int(
                result.loc[
                    forced_mask
                    & result["label"].eq(label)
                    & result["split"].eq(split)
                ].shape[0]
            )
            for split in targets
        }
        unforced_source_sizes = {
            source: size
            for source, size in source_sizes.items()
            if source not in FORCED_SOURCE_SPLITS
        }
        ordered_sources = sorted(
            unforced_source_sizes,
            key=lambda source: (
                -unforced_source_sizes[source],
                hashlib.sha256(f"{random_state}:{label}:{source}".encode("utf-8")).hexdigest(),
            ),
        )
        for source in ordered_sources:
            size = unforced_source_sizes[source]
            split = max(
                targets,
                key=lambda candidate: (
                    (targets[candidate] - assigned[candidate]) / max(targets[candidate], 1),
                    hashlib.sha256(f"{source}:{candidate}".encode("utf-8")).hexdigest(),
                ),
            )
            assigned[split] += size
            mask = real_mask & result["label"].eq(label) & result["source"].eq(source)
            result.loc[mask, "split"] = split

    source_split_counts = result.loc[real_mask].groupby("source")["split"].nunique()
    if (source_split_counts > 1).any():
        raise ValueError("Source leakage detected: a real source occurs in multiple splits")
    campaign_split_counts = result.loc[real_mask].groupby("campaign_id")["split"].nunique()
    if (campaign_split_counts > 1).any():
        raise ValueError("Campaign leakage detected after source-held-out splitting")
    return result


def balance_dataset(
    input_csv: Path = DEFAULT_OUTPUT_CSV,
    output_csv: Path | None = None,
    per_class: int | None = None,
    random_state: int = 42,
    split_strategy: str = "campaign",
    keep_all: bool = False,
) -> dict:
    output_csv = output_csv or input_csv.with_name(f"{input_csv.stem}_balanced.csv")
    df, dedupe_info = _dedupe_templates(pd.read_csv(input_csv))
    counts = df["label"].value_counts() if not df.empty else pd.Series(dtype=int)
    if df.empty or any(counts.get(label, 0) == 0 for label in (0, 1)):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        return {"rows": 0, "per_class": 0, "output": str(output_csv), **dedupe_info}

    if keep_all:
        balanced = _assign_splits(df, random_state=random_state)
        target = None
    elif split_strategy == "source_holdout":
        assigned = _assign_source_holdout_splits(df, random_state=random_state)
        sampled_parts = []
        for split in ("train", "validation", "test"):
            split_rows = assigned[assigned["split"].eq(split)]
            split_counts = split_rows["label"].value_counts()
            if any(split_counts.get(label, 0) == 0 for label in (0, 1)):
                raise ValueError(f"Source-held-out split '{split}' does not contain both labels")
            split_target = min(int(split_counts.get(0, 0)), int(split_counts.get(1, 0)))
            if per_class is not None:
                split_target = min(split_target, int(per_class))
            sampled_parts.extend(
                split_rows[split_rows["label"].eq(label)].sample(
                    n=split_target,
                    random_state=random_state + label + {"train": 0, "validation": 10, "test": 20}[split],
                )
                for label in (0, 1)
            )
        balanced = pd.concat(sampled_parts, ignore_index=True)
        target = int(balanced.groupby("label").size().min())
    else:
        target = min(per_class or int(counts.min()), int(counts.min()))
        sampled_parts = [
            df[df["label"] == label].sample(n=target, random_state=random_state + label)
            for label in (0, 1)
        ]
        balanced = pd.concat(sampled_parts, ignore_index=True)
        balanced = _assign_splits(balanced, random_state=random_state)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_csv, index=False)
    return {
        "rows": len(balanced),
        "per_class": target,
        "class_counts": {
            str(int(label)): int(count)
            for label, count in balanced["label"].value_counts().sort_index().items()
        },
        "output": str(output_csv),
        **dedupe_info,
        "splits": balanced["split"].value_counts().to_dict(),
    }

def build_balanced_public_dataset(
    selected_sources: list[str],
    output_csv: Path = DEFAULT_BALANCED_OUTPUT_CSV,
    staging_csv: Path = DEFAULT_OUTPUT_CSV,
    include_hard_ham: bool = True,
    max_enron: int = 10000,
    progress: Callable[[str], None] | None = None,
) -> dict:
    _ensure_dirs()
    if not selected_sources:
        return {"status": "error", "message": "Select at least one source.", "results": []}

    selected_sources = list(dict.fromkeys(selected_sources))
    disallowed_sources = sorted(set(selected_sources) & DISALLOWED_SOURCE_NAMES)
    if disallowed_sources:
        return {
            "status": "error",
            "message": "Fonti non consentite: " + ", ".join(disallowed_sources),
            "results": [],
        }
    legacy_sources = sorted(set(selected_sources) & LEGACY_SOURCE_NAMES)
    if legacy_sources:
        return {
            "status": "error",
            "message": (
                "Fonti legacy escluse dalla composizione corrente: "
                + ", ".join(legacy_sources)
            ),
            "results": [],
        }
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
        "zenodo_validation": lambda: add_zenodo_validation(output_csv=staging_csv, progress=progress),
        "spaphish": lambda: add_spaphish(output_csv=staging_csv, progress=progress),
        "ubuntu_modern_ham": lambda: add_ubuntu_modern_ham(output_csv=staging_csv, progress=progress),
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
        try:
            results.append(step())
        except Exception as exc:
            results.append(BuildResult(source, 0, 0, 0, 1, f"Import fallito: {exc}"))

    failed_sources = [result for result in results if result.errors]
    if failed_sources:
        return {
            "status": "error",
            "message": (
                "Generazione interrotta: una o piu fonti richieste non sono state importate integralmente: "
                + ", ".join(result.source for result in failed_sources)
            ),
            "results": results,
            "stats": dataset_stats(staging_csv),
        }

    stats = dataset_stats(staging_csv)
    if stats["legitimate"] == 0 or stats["phishing"] == 0:
        return {
            "status": "error",
            "message": "At least one legitimate source and one phishing source are required.",
            "results": results,
            "stats": stats,
        }

    try:
        balanced = balance_dataset(
            staging_csv,
            output_csv=output_csv,
            split_strategy="campaign",
            keep_all=True,
        )
    except ValueError as exc:
        return {
            "status": "error",
            "message": f"Controllo campagne fallito: {exc}",
            "results": results,
            "stats": stats,
        }
    return {
        "status": "ok",
        "message": (
            f"Creato dataset misto con {balanced['rows']} email pubbliche, senza scartare "
            "la classe maggioritaria. Split casuale riproducibile 70/10/20 per fonte, classe "
            "e campagna. "
            f"Quasi-duplicati rimossi: {balanced.get('template_duplicates', 0)}; "
            f"conflitti label rimossi: {balanced.get('label_conflicts', 0)}."
        ),
        "results": results,
        "stats": dataset_stats(output_csv),
        "output": balanced["output"],
    }


def combine_public_and_synthetic_datasets(
    public_csv: Path = DEFAULT_BALANCED_OUTPUT_CSV,
    synthetic_csv: Path = DEFAULT_SYNTHETIC_CSV,
    legitimate_hard_negative_csv: Path | None = None,
    output_csv: Path = DEFAULT_COMPLETE_OUTPUT_CSV,
    max_synthetic_train_fraction: float = MAX_SYNTHETIC_TRAIN_FRACTION,
) -> dict:
    """Unisce dati pubblici e sintetici senza contaminare validation e test."""
    if not public_csv.exists():
        return {"status": "error", "message": f"Dataset pubblico non trovato: {public_csv}"}
    if not synthetic_csv.exists():
        return {"status": "error", "message": f"Dataset sintetico non trovato: {synthetic_csv}"}
    if not 0 < max_synthetic_train_fraction < 1:
        return {"status": "error", "message": "La quota sintetica massima deve essere compresa tra 0 e 1."}

    public_raw = pd.read_csv(public_csv)
    if "split" not in public_raw:
        return {"status": "error", "message": "Il dataset pubblico non contiene la colonna split."}

    public, public_quality = _dedupe_templates(public_raw)
    split_lookup = (
        public_raw.assign(_normalized_text=public_raw["text"].fillna("").astype(str).map(normalize_text))
        .assign(_normalized_hash=lambda frame: frame["_normalized_text"].map(text_hash))
        .drop_duplicates("_normalized_hash", keep="first")
        .set_index("_normalized_hash")["split"]
        .astype(str)
        .str.lower()
        .replace({"val": "validation"})
        .to_dict()
    )
    public["split"] = public["text_hash"].map(split_lookup)
    if public["split"].isna().any() or not set(public["split"]).issubset({"train", "validation", "test"}):
        return {"status": "error", "message": "Gli split del dataset pubblico non sono validi."}
    if "campaign_id" in public_raw:
        campaign_lookup = (
            public_raw.drop_duplicates("text_hash", keep="first")
            .set_index("text_hash")["campaign_id"]
            .astype(str)
            .to_dict()
        )
        public["campaign_id"] = public["text_hash"].map(campaign_lookup)
    else:
        public["campaign_id"] = "campaign:" + public["text_hash"].astype(str)

    synthetic_raw = pd.read_csv(synthetic_csv)
    synthetic, synthetic_quality = _dedupe_templates(synthetic_raw)
    if synthetic.empty:
        return {"status": "error", "message": "Il dataset sintetico non contiene righe valide."}
    if not _synthetic_source_mask(synthetic).all():
        return {
            "status": "error",
            "message": "Il CSV sintetico contiene fonti non marcate con il prefisso synthetic_.",
        }
    synthetic_counts = synthetic["label"].value_counts().to_dict()
    if synthetic_counts.get(0, 0) != synthetic_counts.get(1, 0):
        return {
            "status": "error",
            "message": (
                "Il dataset sintetico deve essere bilanciato: "
                f"legitimate={synthetic_counts.get(0, 0)}, phishing={synthetic_counts.get(1, 0)}."
            ),
        }

    hard_negative = pd.DataFrame(columns=FINAL_COLUMNS)
    hard_negative_quality = {
        "template_duplicates": 0,
        "label_conflicts": 0,
        "invalid_text": 0,
        "invalid_label": 0,
        "too_short": 0,
        "too_few_words": 0,
        "too_long": 0,
        "exact_duplicates": 0,
        "exact_label_conflicts": 0,
    }
    if legitimate_hard_negative_csv is not None:
        if not legitimate_hard_negative_csv.exists():
            return {
                "status": "error",
                "message": (
                    "Dataset legitimate hard-negative non trovato: "
                    f"{legitimate_hard_negative_csv}"
                ),
            }
        hard_negative_raw = pd.read_csv(legitimate_hard_negative_csv)
        hard_negative, hard_negative_quality = _dedupe_templates(hard_negative_raw)
        if hard_negative.empty:
            return {
                "status": "error",
                "message": "Il dataset legitimate hard-negative non contiene righe valide.",
            }
        if not _synthetic_source_mask(hard_negative).all():
            return {
                "status": "error",
                "message": "Le fonti hard-negative devono usare il prefisso synthetic_.",
            }
        if set(hard_negative["label"]) != {0}:
            return {
                "status": "error",
                "message": "Il dataset legitimate hard-negative deve contenere solo label 0.",
            }

    public_train_rows = int(public["split"].eq("train").sum())
    max_synthetic_rows = int(
        np.floor(
            max_synthetic_train_fraction
            * public_train_rows
            / (1.0 - max_synthetic_train_fraction)
        )
    )
    max_synthetic_rows -= max_synthetic_rows % 2
    available_balanced_synthetic_rows = len(synthetic)
    available_hard_negative_rows = len(hard_negative)
    available_synthetic_rows = (
        available_balanced_synthetic_rows + available_hard_negative_rows
    )
    if available_synthetic_rows > max_synthetic_rows:
        hard_negative_limit = min(
            available_hard_negative_rows,
            max_synthetic_rows // 2,
        )
        if len(hard_negative) > hard_negative_limit:
            hard_negative = hard_negative.sample(
                n=hard_negative_limit,
                random_state=84,
            )
        remaining_rows = max_synthetic_rows - len(hard_negative)
        remaining_rows -= remaining_rows % 2
        per_class = remaining_rows // 2
        if per_class == 0:
            return {
                "status": "error",
                "message": "Il train pubblico è troppo piccolo per aggiungere augmentation sintetica.",
            }
        synthetic = pd.concat(
            [
                synthetic[synthetic["label"].eq(label)].sample(
                    n=per_class,
                    random_state=42 + label,
                )
                for label in (0, 1)
            ],
            ignore_index=True,
        )
    augmentation = pd.concat([synthetic, hard_negative], ignore_index=True)
    augmentation["split"] = "train"
    augmentation["campaign_id"] = "synthetic:" + augmentation["text_hash"].astype(str)

    combined_with_split = pd.concat(
        [public[OUTPUT_COLUMNS], augmentation[OUTPUT_COLUMNS]],
        ignore_index=True,
    )
    combined, combined_quality = _dedupe_templates(combined_with_split)
    combined_split_lookup = (
        combined_with_split.drop_duplicates("text_hash", keep="first")
        .set_index("text_hash")["split"]
        .to_dict()
    )
    combined["split"] = combined["text_hash"].map(combined_split_lookup)
    combined_campaign_lookup = (
        combined_with_split.drop_duplicates("text_hash", keep="first")
        .set_index("text_hash")["campaign_id"]
        .to_dict()
    )
    combined["campaign_id"] = combined["text_hash"].map(combined_campaign_lookup)

    synthetic_mask = _synthetic_source_mask(combined)
    if set(combined.loc[synthetic_mask, "split"]) - {"train"}:
        return {"status": "error", "message": "Dati sintetici rilevati fuori dallo split train."}
    campaign_split_counts = combined.groupby("campaign_id")["split"].nunique()
    if (campaign_split_counts > 1).any():
        return {
            "status": "error",
            "message": "Leakage rilevato: una campagna quasi duplicata compare in piu split.",
        }

    for split in ("train", "validation", "test"):
        labels = set(combined.loc[combined["split"] == split, "label"])
        if labels != {0, 1}:
            return {
                "status": "error",
                "message": f"Lo split {split} deve contenere entrambe le classi; trovate {sorted(labels)}.",
            }

    train_rows = int(combined["split"].eq("train").sum())
    synthetic_rows = int(synthetic_mask.sum())
    synthetic_fraction = synthetic_rows / train_rows if train_rows else 0.0
    if synthetic_fraction > max_synthetic_train_fraction:
        return {
            "status": "error",
            "message": (
                f"Quota sintetica nel train troppo alta: {synthetic_fraction:.1%}. "
                f"Limite di qualita: {max_synthetic_train_fraction:.0%}. Aggiungere piu email pubbliche reali."
            ),
        }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined[OUTPUT_COLUMNS].to_csv(output_csv, index=False)
    return {
        "status": "ok",
        "message": (
            f"Creato dataset completo con {len(combined)} email; {synthetic_rows} sintetiche "
            f"selezionate su {available_synthetic_rows} disponibili "
            f"({synthetic_fraction:.1%} del train), tutte escluse da validation e test."
        ),
        "output": str(output_csv),
        "stats": dataset_stats(output_csv),
        "synthetic_rows": synthetic_rows,
        "synthetic_rows_available": available_synthetic_rows,
        "legitimate_hard_negative_rows": int(len(hard_negative)),
        "legitimate_hard_negative_rows_available": available_hard_negative_rows,
        "synthetic_train_fraction": synthetic_fraction,
        "quality": {
            "public": public_quality,
            "synthetic": synthetic_quality,
            "legitimate_hard_negative": hard_negative_quality,
            "combined": combined_quality,
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset_manifest(
    output_csv: Path,
    synthetic_csv: Path,
    legitimate_hard_negative_csv: Path | None,
    selected_sources: list[str],
    source_results: list[BuildResult],
    stats: dict,
) -> Path:
    artifact_paths: list[Path] = []
    if "github_phishing_pot" in selected_sources:
        artifact_paths.append(
            SOURCES_DIR / "phishing" / "phishing_pot" / f"phishing_pot_{PHISHING_POT_COMMIT}.zip"
        )
    if "nazario" in selected_sources:
        artifact_paths.extend(
            SOURCES_DIR / "phishing" / "nazario" / f"phishing-{year}"
            for year in range(MODERN_START_YEAR, MODERN_END_YEAR + 1)
        )
    if "spamassassin" in selected_sources:
        artifact_paths.extend((SOURCES_DIR / "legitimate" / "spamassassin").glob("*.tar.bz2"))
    if "enron" in selected_sources:
        artifact_paths.append(SOURCES_DIR / "legitimate" / "enron" / "enron_mail_20150507.tar.gz")
    if "zenodo_validation" in selected_sources:
        artifact_paths.append(
            SOURCES_DIR / "mixed" / "zenodo_validation" / "Phishing_validation_emails.csv"
        )
    if "spaphish" in selected_sources:
        artifact_paths.append(
            SOURCES_DIR / "mixed" / "spaphish" / f"spaphish_v{SPAPHISH_VERSION}.csv"
        )
    artifact_paths.append(synthetic_csv)
    if legitimate_hard_negative_csv is not None:
        artifact_paths.append(legitimate_hard_negative_csv)

    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(output_csv),
        "dataset_sha256": _file_sha256(output_csv),
        "label_semantics": {"0": "LEGITIMATE", "1": "MALICIOUS_PHISHING_OR_SPAM"},
        "source_policy": {
            "excluded_sources": sorted(
                DISALLOWED_SOURCE_NAMES | LEGACY_SOURCE_NAMES
            ),
            "split_strategy": "campaign_grouped_random_stratified_70_10_20",
            "historical_sources_included_when_selected": sorted(HISTORICAL_SOURCE_NAMES),
            "message_date_filtering": False,
            "nazario_release_years": [MODERN_START_YEAR, MODERN_END_YEAR],
            "class_policy": "retain_all_valid_deduplicated_public_rows",
        },
        "campaign_similarity_threshold": CAMPAIGN_SIMILARITY_THRESHOLD,
        "selected_sources": selected_sources,
        "source_results": [asdict(result) for result in source_results],
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
            for path in sorted(set(artifact_paths))
            if path.exists()
        ],
        "stats": stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_complete_training_dataset(
    selected_sources: list[str],
    output_csv: Path = DEFAULT_COMPLETE_OUTPUT_CSV,
    public_output_csv: Path = DEFAULT_BALANCED_OUTPUT_CSV,
    synthetic_csv: Path = DEFAULT_SYNTHETIC_CSV,
    legitimate_hard_negative_csv: Path | None = DEFAULT_LEGITIMATE_HARD_NEGATIVE_CSV,
    staging_csv: Path = DEFAULT_OUTPUT_CSV,
    include_hard_ham: bool = True,
    max_enron: int = 10000,
    max_synthetic_train_fraction: float = MAX_SYNTHETIC_TRAIN_FRACTION,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Pipeline one-click: fonti pubbliche complete + augmentation sintetica controllata."""
    public_result = build_balanced_public_dataset(
        selected_sources=selected_sources,
        output_csv=public_output_csv,
        staging_csv=staging_csv,
        include_hard_ham=include_hard_ham,
        max_enron=max_enron,
        progress=progress,
    )
    if public_result["status"] != "ok":
        return public_result
    if progress:
        progress("Controllo qualita e aggiunta del dataset sintetico v2")
    complete_result = combine_public_and_synthetic_datasets(
        public_csv=public_output_csv,
        synthetic_csv=synthetic_csv,
        legitimate_hard_negative_csv=legitimate_hard_negative_csv,
        output_csv=output_csv,
        max_synthetic_train_fraction=max_synthetic_train_fraction,
    )
    complete_result["results"] = public_result.get("results", [])
    if complete_result.get("status") == "ok":
        manifest_path = write_dataset_manifest(
            output_csv=output_csv,
            synthetic_csv=synthetic_csv,
            legitimate_hard_negative_csv=legitimate_hard_negative_csv,
            selected_sources=selected_sources,
            source_results=public_result.get("results", []),
            stats=complete_result.get("stats", {}),
        )
        complete_result["manifest"] = str(manifest_path)
    return complete_result


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
        "too_few_words": 0,
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
        "too_few_words": clean_info["too_few_words"],
        "too_long": clean_info["too_long"],
        "rows_after_template_dedupe": len(deduped_df),
        "missing_label": False,
        "sources": df["source"].value_counts().to_dict(),
        "splits": raw_df["split"].value_counts().to_dict() if "split" in raw_df else {},
    }
