"""Shared BERT text preprocessing for training and inference."""

import re


def normalize_bert_text(text: str) -> str:
    if not text:
        return ""
    if re.search(r"<[a-zA-Z][^>]*>", text):
        try:
            from src.analyzer.html_utils import strip_html
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)
        else:
            text = strip_html(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_bert_input(subject: str, body: str) -> str:
    return normalize_bert_text(f"{subject or ''} {body or ''}")
