from src.bert_input import prepare_bert_input
from src.eml_dataset_builder import _preprocess


def test_runtime_bert_input_matches_training_preprocess_and_uses_body_only():
    subject = "Invoice Payment REQUIRED"
    body = "<html><body>Please   PAY the invoice.</body></html>"

    assert prepare_bert_input(subject, body) == _preprocess(subject, body)
    assert prepare_bert_input(subject, body) == "please pay the invoice."
    assert "invoice payment required" not in prepare_bert_input(subject, body)


def test_runtime_bert_input_ignores_subject():
    text = prepare_bert_input("Hello", "World")

    assert text == "world"
    assert "subject:" not in text


def test_runtime_bert_input_compacts_links_and_personal_values():
    text = prepare_bert_input(
        "Ignored subject",
        "Open https://example.test/a/very/long/path or email user@example.test.",
    )

    assert text == "open [url link] or email [email address]."
    assert "example.test/a/" not in text
