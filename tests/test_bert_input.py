from src.bert_input import prepare_bert_input
from src.eml_dataset_builder import _preprocess


def test_runtime_bert_input_matches_training_preprocess():
    subject = "Invoice Payment REQUIRED"
    body = "<html><body>Please   PAY the invoice.</body></html>"

    assert prepare_bert_input(subject, body) == _preprocess(subject, body)
    assert prepare_bert_input(subject, body) == "invoice payment required please pay the invoice."


def test_runtime_bert_input_has_no_subject_prefix():
    text = prepare_bert_input("Hello", "World")

    assert text == "hello world"
    assert "subject:" not in text
