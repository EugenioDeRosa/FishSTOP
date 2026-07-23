"""
eml_dataset_builder.py - Costruisce e gestisce un dataset custom da file .eml

The produced text is aligned with the preprocessing applied in train.py to ALL
le fonti (Kaggle, personal_emails, custom_legitimate/custom_phishing):
  - corpo dell'email soltanto (l'oggetto viene ignorato)
  - HTML stripping (BeautifulSoup se disponibile, fallback regex)
  - URL e dati personali sostituiti da segnaposto informativi
  - lowercase
  - spazi multipli collassati a uno solo
  (la punteggiatura viene preservata)

Il CSV prodotto (data/custom_dataset.csv) viene poi letto da train.py
as a third data source, concatenated with the Kaggle + personal_emails pool, with
dedup globale per hash prima dello split train/val/test.

Colonne del CSV:
  xt_combined  - preprocessed text (aligned with the other sources)
  label        - 0 (legittima) | 1 (phishing)
  source_file  - nome del file .eml originale
  text_hash    - SHA-256 of the text for deduplication
  added_at     - timestamp ISO 8601
"""

import os
import re
import csv
import email
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email import policy
from pathlib import Path
from typing import Optional

import pandas as pd

from src.bert_input import prepare_bert_input

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Percorsi di default
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH        = os.path.join("data", "custom_dataset.csv")
DEFAULT_LEGIT_FOLDER    = os.path.join("data", "raw", "custom_legitimate")
DEFAULT_PHISHING_FOLDER = os.path.join("data", "raw", "custom_phishing")

CSV_COLUMNS = ["xt_combined", "label", "source_file", "text_hash", "added_at"]

# ---------------------------------------------------------------------------
# HTML stripping (standalone, non dipende da analyzer.py)
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


def _strip_html(html: str) -> str:
    if not html or not html.strip():
        return ""
    if _BS4_AVAILABLE:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = re.sub(r"<[^>]+>", " ", html)
        text = (text
                .replace("&amp;",  "&")
                .replace("&lt;",   "<")
                .replace("&gt;",   ">")
                .replace("&nbsp;", " ")
                .replace("&quot;", '"')
                .replace("&#39;",  "'"))
    return text


# ---------------------------------------------------------------------------
# Preprocessing identico al dataset Kaggle
# ---------------------------------------------------------------------------

def _preprocess(subject: str, body: str) -> str:
    """Preprocessing shared with runtime BERT inference."""
    return prepare_bert_input(subject, body)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Body extraction from EML
# ---------------------------------------------------------------------------

def _extract_from_eml(eml_bytes: bytes) -> dict:
    """
    Estrae subject e body da bytes .eml.
    Restituisce {"subject": str, "body": str, "sender": str}.
    """
    msg = email.message_from_bytes(eml_bytes, policy=policy.compat32)

    subject = str(msg.get("Subject") or "").strip()
    sender  = str(msg.get("From")    or "").strip()

    body_parts = []
    html_parts = []

    for part in msg.walk():
        ct   = part.get_content_type()
        disp = str(part.get("Content-Disposition") or "")
        if "attachment" in disp.lower():
            continue
        if ct == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="ignore"))
        elif ct == "text/html":
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                html_parts.append(payload.decode(charset, errors="ignore"))

    body = "\n".join(body_parts) if body_parts else "\n".join(html_parts)
    return {"subject": subject, "body": body.strip(), "sender": sender}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmlDatasetBuilder:
    """
    Gestisce il dataset custom:
      - add_eml()        : aggiunge un singolo file .eml con la label scelta
      - add_batch()      : aggiunge una lista di (eml_bytes, filename, label)
                           con scrittura CSV in un solo flush e salvataggio
                           file .eml in parallelo via ThreadPoolExecutor
      - remove_by_hash() : rimuove una riga dal CSV tramite text_hash
      - load_df()        : legge il CSV e restituisce un DataFrame
      - stats()          : statistiche sul dataset corrente
    """

    def __init__(
        self,
        csv_path:        str = DEFAULT_CSV_PATH,
        legit_folder:    str = DEFAULT_LEGIT_FOLDER,
        phishing_folder: str = DEFAULT_PHISHING_FOLDER,
    ):
        self.csv_path        = csv_path
        self.legit_folder    = legit_folder
        self.phishing_folder = phishing_folder

        for folder in [legit_folder, phishing_folder, os.path.dirname(csv_path)]:
            if folder:
                os.makedirs(folder, exist_ok=True)

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------ #
    # Aggiunta singola
    # ------------------------------------------------------------------ #

    def add_eml(
        self,
        eml_bytes: bytes,
        filename:  str,
        label:     int,
        overwrite: bool = False,
        _existing_hashes: set | None = None,  # optional: reuse already loaded set
    ) -> dict:
        """
        Processa un singolo .eml e lo aggiunge al dataset.

        Returns
        -------
        {
          "status"  : "added" | "duplicate" | "error",
          "hash"    : str,
          "text"    : str,
          "message" : str,
          "_raw"    : bytes | None,   # usato internamente da add_batch
          "_dest"   : str | None,     # percorso di destinazione del file .eml
        }
        """
        try:
            parsed = _extract_from_eml(eml_bytes)
            text   = _preprocess(parsed["subject"], parsed["body"])

            if not text or len(text) < 10:
                return {
                    "status":  "error",
                    "hash":    "",
                    "text":    text,
                    "message": f"Text too short after preprocessing ({len(text)} car.) - file ignored",
                    "_raw":    None,
                    "_dest":   None,
                }

            h = _text_hash(text)

            existing = _existing_hashes if _existing_hashes is not None else self._load_hashes()

            if h in existing and not overwrite:
                return {
                    "status":  "duplicate",
                    "hash":    h,
                    "text":    text,
                    "message": f"'{filename}' already present in the dataset (hash: {h[:12]}...)",
                    "_raw":    None,
                    "_dest":   None,
                }

            if h in existing and overwrite:
                self.remove_by_hash(h)

            row = {
                "xt_combined": text,
                "label":       label,
                "source_file": filename,
                "text_hash":   h,
                "added_at":    datetime.now(timezone.utc).isoformat(),
            }

            # Calcola il percorso di destinazione del file .eml.
            # Usiamo sempre il suffisso hash per evitare collisioni nel batch parallelo
            # (os.path.exists is not reliable when files have not been written yet).
            dest_folder = self.phishing_folder if label == 1 else self.legit_folder
            stem        = Path(filename).stem
            ext         = Path(filename).suffix or ".eml"
            dest_path   = os.path.join(dest_folder, f"{stem}_{h[:8]}{ext}")

            label_str = "Phishing" if label == 1 else "Legitimate"
            return {
                "status":  "added",
                "hash":    h,
                "text":    text,
                "message": f"'{filename}' added as {label_str} (hash: {h[:12]}...)",
                "_row":    row,
                "_raw":    eml_bytes,
                "_dest":   dest_path,
            }

        except Exception as exc:
            logger.exception("Error in add_eml per %s", filename)
            return {
                "status":  "error",
                "hash":    "",
                "text":    "",
                "message": f"Error durante il processing di '{filename}': {exc}",
                "_raw":    None,
                "_dest":   None,
            }

    # ------------------------------------------------------------------ #
    # Aggiunta batch  ← riscritta per efficienza
    # ------------------------------------------------------------------ #

    def add_batch(
        self,
        items:             list[tuple[bytes, str, int]],
        overwrite:         bool = False,
        max_file_workers:  int  = 8,
        progress_callback: Optional[callable] = None,
    ) -> list[dict]:
        """
        Processa una lista di EML in modo ottimizzato per batch grandi (500-600+).

        Ottimizzazioni rispetto al loop singolo:
          - _load_hashes() chiamata UNA VOLTA sola per tutto il batch
          - CSV aperto in append UNA VOLTA sola per tutte le rows "added"
          - Salvataggio file .eml su disco parallelizzato (ThreadPoolExecutor)

        Parameters
        ----------
        items             : lista di (eml_bytes, filename, label)
        overwrite         : se True sovrascrive i duplicati
        max_file_workers  : thread per il salvataggio parallelo dei file .eml
        progress_callback : callable(done: int, total: int) - notifica avanzamento
        """
        total   = len(items)
        results = []

        # ── 1. Load existing hashes ONCE ───────────────────
        existing_hashes: set = self._load_hashes()

        rows_to_write:  list[dict]               = []  # rows CSV da appendere
        files_to_write: list[tuple[bytes, str]]  = []  # (raw_bytes, dest_path)

        # ── 2. Processing in-memory (CPU-bound, non parallelizzato per
        #       evitare lock sul set condiviso e sul CSV) ──────────────────
        for i, (eml_bytes, filename, label) in enumerate(items, start=1):
            res = self.add_eml(
                eml_bytes,
                filename,
                label,
                overwrite=overwrite,
                _existing_hashes=existing_hashes,
            )

            if res["status"] == "added":
                existing_hashes.add(res["hash"])   # aggiorna set in-memory
                rows_to_write.append(res["_row"])
                if res["_raw"] and res["_dest"]:
                    files_to_write.append((res["_raw"], res["_dest"]))

            # Rimuovi campi interni prima di restituire
            res.pop("_row",  None)
            res.pop("_raw",  None)
            res.pop("_dest", None)
            results.append(res)

            if progress_callback:
                progress_callback(i, total)

        # ── 3. Scrittura CSV in un UNICO flush ────────────────────────────
        if rows_to_write:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerows(rows_to_write)

        # ── 4. Salvataggio file .eml in parallelo ─────────────────────────
        if files_to_write:
            def _write_file(args: tuple[bytes, str]) -> None:
                raw, dest = args
                with open(dest, "wb") as f:
                    f.write(raw)

            with ThreadPoolExecutor(max_workers=max_file_workers) as pool:
                futures = {pool.submit(_write_file, fw): fw[1] for fw in files_to_write}
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        logger.warning("Error salvataggio file %s: %s", futures[fut], exc)

        return results

    # ------------------------------------------------------------------ #
    # Rimozione
    # ------------------------------------------------------------------ #

    def remove_by_hash(self, text_hash: str) -> bool:
        """
        Rimuove la riga con il dato text_hash dal CSV.
        Restituisce True se trovata e rimossa.
        """
        if not os.path.exists(self.csv_path):
            return False
        df = pd.read_csv(self.csv_path, dtype=str)
        before = len(df)
        df = df[df["text_hash"] != text_hash]
        if len(df) == before:
            return False
        df.to_csv(self.csv_path, index=False)
        return True

    # ------------------------------------------------------------------ #
    # Lettura
    # ------------------------------------------------------------------ #

    def load_df(self) -> pd.DataFrame:
        """
        Legge il CSV e restituisce un DataFrame.
        """
        if not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=CSV_COLUMNS)
        df = pd.read_csv(self.csv_path, dtype={"label": int})
        return df

    def load_for_training(self) -> pd.DataFrame:
        """
        Restituisce solo le colonne necessarie per il training,
        con 'text' al posto di 'xt_combined' - pronto per essere
        concatenato al pool Kaggle in train.py.
        """
        df = self.load_df()
        if df.empty:
            return pd.DataFrame(columns=["text", "label"])
        return df[["xt_combined", "label"]].rename(
            columns={"xt_combined": "text"}
        ).dropna(subset=["text"])

    # ------------------------------------------------------------------ #
    # Statistiche
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        df = self.load_df()
        if df.empty:
            return {"total": 0, "legitimate": 0, "phishing": 0, "last_added": None}
        return {
            "total":      len(df),
            "legitimate": int((df["label"] == 0).sum()),
            "phishing":   int((df["label"] == 1).sum()),
            "last_added": df["added_at"].iloc[-1] if "added_at" in df.columns else None,
        }

    # ------------------------------------------------------------------ #
    # Helpers interni
    # ------------------------------------------------------------------ #

    def _load_hashes(self) -> set:
        if not os.path.exists(self.csv_path):
            return set()
        try:
            df = pd.read_csv(self.csv_path, usecols=["text_hash"], dtype=str)
            return set(df["text_hash"].dropna().tolist())
        except Exception:
            return set()


# ---------------------------------------------------------------------------
# Script standalone - uso da riga di comando
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python eml_dataset_builder.py <cartella_eml> <label: 0|1>")
        print("  Es: python eml_dataset_builder.py data/raw/nuove_phishing 1")
        sys.exit(1)

    folder = sys.argv[1]
    label  = int(sys.argv[2])

    if label not in (0, 1):
        print("[!] Label must be 0 (legitimate) or 1 (phishing)")
        sys.exit(1)

    builder = EmlDatasetBuilder()

    batch_items = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".eml"):
            continue
        fpath = os.path.join(folder, fname)
        with open(fpath, "rb") as f:
            batch_items.append((f.read(), fname, label))

    def _cli_progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  ... {done}/{total} processed", flush=True)

    results = builder.add_batch(batch_items, progress_callback=_cli_progress)

    added = skipped = errors = 0
    for res in results:
        status = res["status"]
        print(f"  [{status.upper():9s}] {res['message']}")
        if status == "added":       added   += 1
        elif status == "duplicate": skipped += 1
        else:                       errors  += 1

    print(f"\n✅ Completed: {added} added | {skipped} duplicate | {errors} errors")
    s = builder.stats()
    print(f"📊 Total dataset: {s['total']} rows ({s['legitimate']} legitimate, {s['phishing']} phishing)")
