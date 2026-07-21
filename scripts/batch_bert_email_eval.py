"""Batch evaluation of FishSTOP's deployed DistilBERT on an EML corpus.

This runner intentionally reuses the same SOC body extraction, BERT input
normalization, long-email chunking, chunk aggregation, calibration, and
three-way decision function used by the Streamlit application.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer import EmlSOCAnalyzer
from src.bert_calibration import (
    DEFAULT_BAND,
    DEFAULT_POSITIVE_LABEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD,
    calibrated_probabilities,
    classify,
)
from src.bert_inference import aggregate_chunk_logits, encode_email_chunks
from src.bert_input import prepare_bert_input
from src.config import get_secret


DEFAULT_MODEL_ID = "eugenioderodev/fishstop-bert"
CSV_FIELDS = [
    "relative_path",
    "absolute_path",
    "size_bytes",
    "sha256",
    "parse_status",
    "inference_status",
    "error_stage",
    "error_type",
    "error_message",
    "subject",
    "sender",
    "date",
    "body_source",
    "body_clean_chars",
    "body_for_ai_chars",
    "body_plain_noise_removed_lines",
    "body_plain_noise_removed_chars",
    "bert_input_chars",
    "chunk_count",
    "logit_legitimate",
    "logit_malicious",
    "probability_legitimate",
    "probability_malicious",
    "probability_malicious_percent",
    "classification",
    "classified_phishing",
    "false_negative_expected_phishing",
    "elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the deployed FishSTOP BERT model on all .eml files under a folder."
    )
    parser.add_argument("email_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "bert_email_eval")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default=os.getenv("FISHSTOP_HF_MODEL_REVISION", "").strip(),
        help="Hugging Face revision. Empty reproduces the app's current unpinned main revision.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Model chunks per forward pass.")
    parser.add_argument("--flush-chunks", type=int, default=64, help="Pending chunks before inference.")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Optional test limit; 0 means all EML files.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from email_results.csv, skipping rows already present.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, retry rows whose parsing or inference did not finish successfully.",
    )
    return parser.parse_args()


def compact_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:1000]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(model_id: str, revision: str, token: str) -> dict[str, Any]:
    auth: dict[str, Any] = {"repo_id": model_id, "filename": "calibration.json"}
    if revision:
        auth["revision"] = revision
    if token:
        auth["token"] = token
    try:
        calibration_path = hf_hub_download(**auth)
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        source = "huggingface"
    except (HfHubHTTPError, FileNotFoundError, OSError, json.JSONDecodeError):
        calibration = {}
        source = "default (nessun calibration.json pubblicato)"
    calibration.setdefault("temperature", DEFAULT_TEMPERATURE)
    calibration.setdefault("threshold", DEFAULT_THRESHOLD)
    calibration.setdefault("band", DEFAULT_BAND)
    calibration.setdefault("positive_label_id", DEFAULT_POSITIVE_LABEL_ID)
    calibration["source"] = source
    return calibration


def resolved_model_revision(model_id: str, revision: str, token: str) -> str:
    try:
        return str(HfApi(token=token or None).model_info(model_id, revision=revision or None).sha or "")
    except Exception:
        return "unavailable"


def base_row(path: Path, root: Path) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "absolute_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "parse_status": "pending",
            "inference_status": "pending",
            "classified_phishing": "",
            "false_negative_expected_phishing": "",
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pad_chunk_batch(chunks: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    names = sorted({name for chunk in chunks for name in chunk})
    width = max(int(tensor.shape[-1]) for chunk in chunks for tensor in chunk.values())
    batch: dict[str, torch.Tensor] = {}
    for name in names:
        padded = []
        for chunk in chunks:
            tensor = chunk[name]
            pad_value = 0
            padded.append(F.pad(tensor, (0, width - int(tensor.shape[-1])), value=pad_value))
        batch[name] = torch.stack(padded).to(device)
    return batch


def run_pending(
    pending: list[dict[str, Any]],
    model,
    calibration: dict[str, Any],
    batch_size: int,
) -> None:
    if not pending:
        return
    flat_chunks: list[dict[str, torch.Tensor]] = []
    spans: list[tuple[int, int, dict[str, Any]]] = []
    for item in pending:
        start = len(flat_chunks)
        encoded = item.pop("encoded")
        count = int(encoded["input_ids"].shape[0])
        for index in range(count):
            flat_chunks.append(
                {
                    name: tensor[index].cpu()
                    for name, tensor in encoded.items()
                    if name != "overflow_to_sample_mapping" and isinstance(tensor, torch.Tensor)
                }
            )
        spans.append((start, start + count, item))

    device = next(model.parameters()).device
    logits_parts = []
    try:
        for start in range(0, len(flat_chunks), batch_size):
            batch = pad_chunk_batch(flat_chunks[start : start + batch_size], device)
            with torch.inference_mode():
                logits_parts.append(model(**batch).logits.detach().cpu())
        all_logits = torch.cat(logits_parts, dim=0)
    except Exception as exc:
        for _, _, item in spans:
            row = item["row"]
            row.update(
                {
                    "inference_status": "error",
                    "error_stage": "inference",
                    "error_type": type(exc).__name__,
                    "error_message": compact_error(exc),
                    "elapsed_seconds": round(time.perf_counter() - item["started"], 6),
                }
            )
        return

    positive_id = int(calibration["positive_label_id"])
    negative_id = 1 - positive_id
    for start, end, item in spans:
        row = item["row"]
        try:
            email_logits = aggregate_chunk_logits(all_logits[start:end], positive_id)
            probabilities = calibrated_probabilities(email_logits, calibration["temperature"])[0]
            probability_malicious = float(probabilities[positive_id])
            probability_legitimate = float(probabilities[negative_id])
            classification = classify(
                probability_malicious,
                threshold=float(calibration["threshold"]),
                band=float(calibration["band"]),
            )
            row.update(
                {
                    "inference_status": "ok",
                    "chunk_count": end - start,
                    "logit_legitimate": float(email_logits[0, negative_id]),
                    "logit_malicious": float(email_logits[0, positive_id]),
                    "probability_legitimate": probability_legitimate,
                    "probability_malicious": probability_malicious,
                    "probability_malicious_percent": probability_malicious * 100.0,
                    "classification": classification,
                    "classified_phishing": classification == "phishing",
                    "false_negative_expected_phishing": classification != "phishing",
                    "elapsed_seconds": round(time.perf_counter() - item["started"], 6),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "inference_status": "error",
                    "error_stage": "postprocessing",
                    "error_type": type(exc).__name__,
                    "error_message": compact_error(exc),
                    "elapsed_seconds": round(time.perf_counter() - item["started"], 6),
                }
            )


def percentile_map(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    points = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{point:02d}": float(np.percentile(values, point)) for point in points}


def build_summary(
    rows: list[dict[str, Any]],
    files_found: int,
    args: argparse.Namespace,
    calibration: dict[str, Any],
    model_meta: dict[str, Any],
    started_utc: str,
    elapsed: float,
) -> dict[str, Any]:
    parsed = [row for row in rows if row["parse_status"] == "ok"]
    inferred = [row for row in rows if row["inference_status"] == "ok"]
    classifications = Counter(str(row["classification"]) for row in inferred)
    phishing = classifications.get("phishing", 0)
    non_phishing = len(inferred) - phishing
    probs = [float(row["probability_malicious"]) for row in inferred]
    bins = Counter()
    for value in probs:
        lower = min(int(value * 10) * 10, 90)
        bins[f"{lower:02d}-{lower + 10:02d}%"] += 1
    denominator = len(inferred)
    noise_rows = [
        row for row in rows if int(row.get("body_plain_noise_removed_lines") or 0) > 0
    ]
    threshold = float(calibration["threshold"])
    band = float(calibration["band"])
    return {
        "run": {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "email_dir": str(args.email_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "model": model_meta,
        "decision": {
            "positive_label": "malicious (phishing/spam)",
            "positive_label_id": int(calibration["positive_label_id"]),
            "temperature": float(calibration["temperature"]),
            "threshold_center": threshold,
            "uncertainty_band_total_width": band,
            "legitimate_at_or_below": threshold - band / 2,
            "phishing_at_or_above": threshold + band / 2,
            "calibration_source": calibration["source"],
        },
        "counts": {
            "eml_files_found": files_found,
            "rows_written": len(rows),
            "valid_parsed_files": len(parsed),
            "parse_errors": sum(row["parse_status"] == "error" for row in rows),
            "successful_inferences": denominator,
            "inference_or_postprocessing_errors": sum(
                row["parse_status"] == "ok" and row["inference_status"] == "error" for row in rows
            ),
            "classified_phishing": phishing,
            "classified_phishing_percent": (100.0 * phishing / denominator) if denominator else 0.0,
            "not_classified_phishing_false_negatives": non_phishing,
            "not_classified_phishing_false_negatives_percent": (100.0 * non_phishing / denominator)
            if denominator
            else 0.0,
            "classification_distribution": dict(sorted(classifications.items())),
            "plaintext_noise_files": len(noise_rows),
            "plaintext_noise_removed_lines": sum(
                int(row.get("body_plain_noise_removed_lines") or 0) for row in noise_rows
            ),
            "plaintext_noise_removed_chars": sum(
                int(row.get("body_plain_noise_removed_chars") or 0) for row in noise_rows
            ),
        },
        "probability_malicious_distribution": {
            "percentiles_0_to_1": percentile_map(probs),
            "histogram_10_percentage_point_bins": dict(sorted(bins.items())),
            "mean": float(np.mean(probs)) if probs else None,
            "standard_deviation": float(np.std(probs)) if probs else None,
        },
        "error_distribution": dict(
            sorted(Counter(str(row["error_type"]) for row in rows if row["error_type"]).items())
        ),
    }


def human_summary(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    decision = summary["decision"]
    dist = summary["probability_malicious_distribution"]
    lines = [
        "FishSTOP - valutazione batch DistilBERT",
        "=" * 43,
        f"Cartella email: {summary['run']['email_dir']}",
        f"Modello: {summary['model']['model_id']} @ {summary['model']['resolved_revision']}",
        f"Dispositivo: {summary['run']['device']}",
        "",
        f"File .eml trovati: {counts['eml_files_found']}",
        f"File validi (parsing OK): {counts['valid_parsed_files']}",
        f"Errori parsing: {counts['parse_errors']}",
        f"Inferenze riuscite: {counts['successful_inferences']}",
        f"Errori inferenza/post-processing: {counts['inference_or_postprocessing_errors']}",
        "",
        f"Classificati phishing: {counts['classified_phishing']} "
        f"({counts['classified_phishing_percent']:.4f}%)",
        f"Non classificati phishing / falsi negativi: "
        f"{counts['not_classified_phishing_false_negatives']} "
        f"({counts['not_classified_phishing_false_negatives_percent']:.4f}%)",
        f"Distribuzione etichette: {json.dumps(counts['classification_distribution'], ensure_ascii=False)}",
        f"Email con padding plain-text rimosso: {counts['plaintext_noise_files']}",
        f"Righe di padding rimosse: {counts['plaintext_noise_removed_lines']}",
        f"Caratteri di padding rimossi: {counts['plaintext_noise_removed_chars']}",
        "",
        f"Temperatura: {decision['temperature']}",
        f"Soglia centrale: {decision['threshold_center']}",
        f"Legitimate <= {decision['legitimate_at_or_below']}",
        f"Phishing >= {decision['phishing_at_or_above']}",
        f"Uncertain tra le due soglie",
        f"Calibrazione: {decision['calibration_source']}",
        "",
        "Distribuzione probabilita malicious (0-1):",
        f"  media={dist['mean']} deviazione_standard={dist['standard_deviation']}",
        f"  percentili={json.dumps(dist['percentiles_0_to_1'], ensure_ascii=False)}",
        f"  istogramma={json.dumps(dist['histogram_10_percentage_point_bins'], ensure_ascii=False)}",
        "",
        f"Durata: {summary['run']['elapsed_seconds']:.2f} secondi",
        f"CSV dettagliato: {Path(summary['run']['output_dir']) / 'email_results.csv'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.email_dir = args.email_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.email_dir.is_dir():
        raise SystemExit(f"Email directory not found: {args.email_dir}")
    if args.batch_size < 1 or args.flush_chunks < 1:
        raise SystemExit("--batch-size and --flush-chunks must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        (path for path in args.email_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".eml"),
        key=lambda path: path.as_posix().lower(),
    )
    files_found = len(files)
    if args.limit:
        files = files[: args.limit]

    token = get_secret("HF_TOKEN")
    hf_args: dict[str, Any] = {"token": token} if token else {}
    if args.revision:
        hf_args["revision"] = args.revision
    resolved_revision = resolved_model_revision(args.model_id, args.revision, token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **hf_args)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **hf_args)
    model.eval()
    calibration = load_calibration(args.model_id, args.revision, token)
    positive_id = int(calibration["positive_label_id"])
    model_meta = {
        "model_id": args.model_id,
        "requested_revision": args.revision or "main (not pinned)",
        "resolved_revision": resolved_revision,
        "model_type": str(model.config.model_type),
        "id2label": {str(key): str(value) for key, value in model.config.id2label.items()},
        "positive_label_id": positive_id,
    }

    analyzer = EmlSOCAnalyzer()
    csv_path = args.output_dir / "email_results.csv"
    rows: list[dict[str, Any]] = read_csv(csv_path) if args.resume and csv_path.exists() else []
    if args.retry_errors and not args.resume:
        raise SystemExit("--retry-errors requires --resume")
    if args.retry_errors:
        rows = [
            row
            for row in rows
            if row.get("parse_status") == "ok" and row.get("inference_status") == "ok"
        ]
    completed_paths = {str(row["relative_path"]) for row in rows}
    files = [path for path in files if path.relative_to(args.email_dir).as_posix() not in completed_paths]
    pending: list[dict[str, Any]] = []
    pending_chunks = 0
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    def flush() -> None:
        nonlocal pending, pending_chunks
        run_pending(pending, model, calibration, args.batch_size)
        pending = []
        pending_chunks = 0

    initial_count = len(rows)
    for index, path in enumerate(files, start=1):
        item_started = time.perf_counter()
        row = base_row(path, args.email_dir)
        rows.append(row)
        try:
            row["sha256"] = file_sha256(path)
            soc = analyzer.analyze(str(path))
            body = soc.get("body_for_ai") or soc.get("body_ai") or soc.get("body_clean") or ""
            bert_text = prepare_bert_input(soc.get("subject") or "", body)
            row.update(
                {
                    "parse_status": "ok",
                    "subject": soc.get("subject") or "",
                    "sender": soc.get("from_") or "",
                    "date": soc.get("date") or "",
                    "body_source": soc.get("body_source") or "",
                    "body_clean_chars": len(soc.get("body_clean") or ""),
                    "body_for_ai_chars": len(body),
                    "body_plain_noise_removed_lines": int(
                        soc.get("body_plain_noise_removed_lines") or 0
                    ),
                    "body_plain_noise_removed_chars": int(
                        soc.get("body_plain_noise_removed_chars") or 0
                    ),
                    "bert_input_chars": len(bert_text),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "parse_status": "error",
                    "inference_status": "not_run",
                    "error_stage": "parsing",
                    "error_type": type(exc).__name__,
                    "error_message": compact_error(exc),
                    "elapsed_seconds": round(time.perf_counter() - item_started, 6),
                }
            )
            continue

        try:
            encoded = encode_email_chunks(tokenizer, bert_text)
            chunk_count = int(encoded["input_ids"].shape[0])
            row["chunk_count"] = chunk_count
            pending.append({"row": row, "encoded": encoded, "started": item_started})
            pending_chunks += chunk_count
        except Exception as exc:
            row.update(
                {
                    "inference_status": "error",
                    "error_stage": "tokenization",
                    "error_type": type(exc).__name__,
                    "error_message": compact_error(exc),
                    "elapsed_seconds": round(time.perf_counter() - item_started, 6),
                }
            )

        if pending_chunks >= args.flush_chunks:
            flush()
        total_done = initial_count + index
        if total_done % args.checkpoint_every == 0:
            flush()
            write_csv(csv_path, rows)
            ok = sum(row["inference_status"] == "ok" for row in rows)
            print(f"[{total_done}/{initial_count + len(files)}] completed; successful inference={ok}", flush=True)

    flush()
    write_csv(csv_path, rows)
    elapsed = time.perf_counter() - started
    summary = build_summary(rows, files_found, args, calibration, model_meta, started_utc, elapsed)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text = human_summary(summary)
    (args.output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
