from email.message import EmailMessage

import pytest
import torch

from src.analysis_limits import (
    EmailAnalysisLimitError,
    MAX_AI_BODY_CHARS,
    MAX_ATTACHMENTS,
    MAX_EML_BYTES,
    MAX_LINKS,
    MAX_MIME_DEPTH,
    MAX_MIME_PARTS,
)
from src.analyzer import llm_context_analyzer as llm
from src.analyzer.link_extractor import extract_links
from src.analyzer.soc_analyzer import EmlSOCAnalyzer, _validate_mime_structure
from src.bert_inference import MAX_EMAIL_CHUNKS, encode_email_chunks
from src.views.analyzer import _validate_eml_size


def _analyze_message(tmp_path, message: EmailMessage) -> dict:
    path = tmp_path / "limited.eml"
    path.write_bytes(message.as_bytes())
    return EmlSOCAnalyzer().analyze(str(path))


def test_normal_email_remains_fully_supported(tmp_path):
    message = EmailMessage()
    message["From"] = "sender@example.test"
    message["To"] = "recipient@example.test"
    message["Subject"] = "Normal message"
    message.set_content("Read the complete message at https://example.test/news")

    report = _analyze_message(tmp_path, message)

    assert report["ai_analysis_supported"] is True
    assert report["body_for_ai"]
    assert len(report["links"]) == 1


def test_oversized_upload_is_rejected_before_parsing():
    with pytest.raises(EmailAnalysisLimitError, match="maximum supported size"):
        _validate_eml_size(b"x" * (MAX_EML_BYTES + 1))


def test_body_over_ai_limit_keeps_static_analysis_and_marks_ai_unavailable(tmp_path):
    message = EmailMessage()
    message["Subject"] = "Large body"
    body = "\n".join(
        f"Line {index}: preserve this complete context with value {index}."
        for index in range(3_000)
    )
    assert len(body) > MAX_AI_BODY_CHARS
    message.set_content(body)

    report = _analyze_message(tmp_path, message)

    assert report["ai_analysis_supported"] is False
    assert "Static checks remain available" in report["ai_analysis_limit_message"]
    assert len(report["body_for_ai"]) > MAX_AI_BODY_CHARS


def test_excessive_attachment_count_is_rejected(tmp_path):
    message = EmailMessage()
    message.set_content("Body")
    for index in range(MAX_ATTACHMENTS + 1):
        message.add_attachment(
            b"x",
            maintype="application",
            subtype="octet-stream",
            filename=f"file-{index}.bin",
        )

    with pytest.raises(EmailAnalysisLimitError, match="attachments"):
        _analyze_message(tmp_path, message)


def test_excessive_mime_depth_is_rejected():
    root = EmailMessage()
    current = root
    for _ in range(MAX_MIME_DEPTH):
        current.make_mixed()
        child = EmailMessage()
        current.attach(child)
        current = child

    with pytest.raises(EmailAnalysisLimitError, match="nesting"):
        _validate_mime_structure(root)


def test_excessive_mime_part_count_is_rejected():
    root = EmailMessage()
    root.make_mixed()
    for index in range(MAX_MIME_PARTS):
        child = EmailMessage()
        child.set_content(f"Part {index}")
        root.attach(child)

    with pytest.raises(EmailAnalysisLimitError, match="MIME parts"):
        _validate_mime_structure(root)


def test_excessive_unique_links_are_rejected():
    body = "\n".join(
        f"https://host-{index}.example.test/path"
        for index in range(MAX_LINKS + 1)
    )

    with pytest.raises(EmailAnalysisLimitError, match="unique links"):
        extract_links(body, "")


def test_phi4_refuses_more_than_the_supported_number_of_sections(monkeypatch):
    monkeypatch.setattr(llm, "MAX_PHI4_SECTIONS", 1)
    body = "\n\n".join(
        f"Paragraph {index}: ordinary contextual words remain available "
        f"throughout this complete email section number {index}."
        for index in range(200)
    )

    with pytest.raises(EmailAnalysisLimitError, match="Phi-4 analysis sections"):
        llm._build_complete_email_prompts(
            {"subject": "Long", "body_for_ai": body, "links": [], "attachments": []},
            anonymize=False,
        )


class _CountingTokenizer:
    def __init__(self):
        self.prepared_count = 0

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        return {"input_ids": list(range(len(text)))}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, token_ids, **kwargs):
        self.prepared_count += 1
        return {
            "input_ids": [101, *token_ids, 102],
            "attention_mask": [1] * (len(token_ids) + 2),
        }

    def pad(self, prepared, **kwargs):
        width = max(len(item["input_ids"]) for item in prepared)
        return {
            "input_ids": torch.tensor([
                item["input_ids"] + [0] * (width - len(item["input_ids"]))
                for item in prepared
            ]),
            "attention_mask": torch.tensor([
                item["attention_mask"] + [0] * (width - len(item["attention_mask"]))
                for item in prepared
            ]),
        }


class _TransformersFiveStyleTokenizer(_CountingTokenizer):
    cls_token_id = 101
    sep_token_id = 102
    prepare_for_model = None


def test_bert_materializes_only_the_selected_windows():
    tokenizer = _CountingTokenizer()

    encoded = encode_email_chunks(
        tokenizer,
        "x" * 20_000,
        max_length=64,
        stride=16,
        max_chunks=MAX_EMAIL_CHUNKS,
    )

    assert tokenizer.prepared_count == MAX_EMAIL_CHUNKS
    assert encoded["input_ids"].shape[0] == MAX_EMAIL_CHUNKS


def test_bert_supports_tokenizer_without_prepare_for_model():
    tokenizer = _TransformersFiveStyleTokenizer()

    encoded = encode_email_chunks(
        tokenizer,
        "x" * 2_000,
        max_length=64,
        stride=16,
        max_chunks=3,
    )

    assert tokenizer.prepared_count == 0
    assert encoded["input_ids"].shape == (3, 64)
    assert encoded["input_ids"][:, 0].tolist() == [101, 101, 101]
    assert encoded["input_ids"][:, -1].tolist() == [102, 102, 102]
