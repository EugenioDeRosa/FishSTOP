"""
public_dataset_builder.py - Costruzione dataset pubblico per training FishStop.

Produce un CSV pulito con colonne:
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
import urllib.request
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
    if not text:
        return ""
    if re.search(r"<[a-zA-Z][\s>/]", text):
        text = _strip_html(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            payload = raw_payload.decode(charset, errors="ignore")

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
            progress(f"Già presente: {dest.name}")
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


def _row(text: str, label: int, source: str, source_file: str) -> dict:
    return {
        "text": text,
        "label": int(label),
        "source": source,
        "source_file": source_file,
        "text_hash": text_hash(text) if text else "",
    }


def _append_rows(
    rows: Iterable[dict],
    existing_hashes: set[str],
    min_chars: int,
) -> tuple[list[dict], int, int]:
    added_rows = []
    skipped = 0
    errors = 0
    for row in rows:
        try:
            text = row.get("text", "")
            h = row.get("text_hash", "")
            if len(text) < min_chars or not h or h in existing_hashes:
                skipped += 1
                continue
            existing_hashes.add(h)
            added_rows.append(row)
        except Exception:
            errors += 1
    return added_rows, skipped, errors


def _load_existing(output_csv: Path) -> tuple[list[dict], set[str]]:
    if not output_csv.exists():
        return [], set()
    df = pd.read_csv(output_csv)
    rows = df.to_dict("records")
    hashes = set(df.get("text_hash", pd.Series(dtype=str)).dropna().astype(str))
    return rows, hashes


def _save_rows(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["text", "label", "source", "source_file", "text_hash"]
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


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
                progress(f"Errore Nazario {name}: {exc}")

    _save_rows(all_rows, output_csv)
    return BuildResult("nazario", len(all_rows), added_total, skipped_total, errors_total, "Nazario phishing importato")


def add_kaggle(
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    progress: Callable[[str], None] | None = None,
    min_chars: int = 40,
) -> BuildResult:
    _ensure_dirs()
    try:
        import kagglehub
    except ImportError as exc:
        return BuildResult("kaggle", 0, 0, 0, 1, f"kagglehub non installato: {exc}")

    all_rows, hashes = _load_existing(output_csv)
    if progress:
        progress(f"Download Kaggle: {KAGGLE_DATASET}")
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        return BuildResult("kaggle", len(all_rows), 0, 0, 1, "Nessun CSV trovato nel dataset Kaggle")

    df = pd.read_csv(csv_files[0])
    df.columns = [c.lower().strip() for c in df.columns]
    text_col = next((c for c in df.columns if any(k in c for k in ["text", "body", "email"])), None)
    label_col = next((c for c in df.columns if any(k in c for k in ["label", "class", "target"])), None)
    if not text_col or not label_col:
        return BuildResult("kaggle", len(all_rows), 0, 0, 1, "Colonne text/label non riconosciute nel CSV Kaggle")

    rows = []
    for idx, record in df.iterrows():
        text = normalize_text(str(record[text_col]))
        try:
            label = int(record[label_col])
        except Exception:
            raw_label = str(record[label_col]).lower()
            label = 1 if "phish" in raw_label or "spam" in raw_label else 0
        rows.append(_row(text, label, "kaggle_phishing_email_dataset", f"kaggle#{idx}"))

    added_rows, skipped, errors = _append_rows(rows, hashes, min_chars)
    all_rows.extend(added_rows)
    _save_rows(all_rows, output_csv)
    return BuildResult("kaggle", len(all_rows), len(added_rows), skipped, errors, "Kaggle importato")


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


def balance_dataset(
    input_csv: Path = DEFAULT_OUTPUT_CSV,
    output_csv: Path | None = None,
    per_class: int | None = None,
    random_state: int = 42,
) -> dict:
    output_csv = output_csv or input_csv.with_name(f"{input_csv.stem}_balanced.csv")
    df = pd.read_csv(input_csv)
    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return {"rows": 0, "per_class": 0, "output": str(output_csv)}

    counts = df["label"].value_counts()
    target = per_class or int(counts.min())
    balanced = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), target), random_state=random_state))
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_csv, index=False)
    return {"rows": len(balanced), "per_class": target, "output": str(output_csv)}


def dataset_stats(csv_path: Path = DEFAULT_OUTPUT_CSV) -> dict:
    if not csv_path.exists():
        return {"exists": False, "rows": 0, "legitimate": 0, "phishing": 0, "sources": {}}
    df = pd.read_csv(csv_path)
    sources = df["source"].value_counts().to_dict() if "source" in df else {}
    return {
        "exists": True,
        "rows": len(df),
        "legitimate": int((df["label"] == 0).sum()) if "label" in df else 0,
        "phishing": int((df["label"] == 1).sum()) if "label" in df else 0,
        "sources": sources,
    }
