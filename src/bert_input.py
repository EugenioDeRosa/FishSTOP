"""Shared BERT text preprocessing for training and inference."""

import re
import unicodedata


def normalize_bert_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(
        char if char in "\n\t" or unicodedata.category(char) != "Cc" else " "
        for char in text
    )
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
