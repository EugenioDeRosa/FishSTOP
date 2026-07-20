import json

import os

import streamlit as st

from src.bert_calibration import (
    DEFAULT_BAND,
    DEFAULT_POSITIVE_LABEL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD,
)
from src.config import get_secret

HF_MODEL_ID = "eugenioderodev/fishstop-bert"
CALIBRATION_FILENAME = "calibration.json"
HF_MODEL_REVISION = os.getenv("FISHSTOP_HF_MODEL_REVISION", "").strip()


@st.cache_resource
def init_core_backend():
    from src.analyzer import EmlSOCAnalyzer
    from src.parser import EmailParserPipeline
    from src.validators import EmailSecurityValidator

    parser = EmailParserPipeline()
    validator = EmailSecurityValidator()
    analyzer = EmlSOCAnalyzer()
    return parser, validator, analyzer


@st.cache_resource
def init_content_model(hf_token: str = ""):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_auth = {"token": hf_token} if hf_token else {}
    if HF_MODEL_REVISION:
        hf_auth["revision"] = HF_MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, **hf_auth)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID, **hf_auth)
    model.eval()

    return tokenizer, model, "huggingface"


@st.cache_resource
def init_calibration(hf_token: str = "") -> dict:
    """
    Scarica calibration.json (temperature scaling + soglia calcolati sul
    validation set, vedi notebooks/Phishing_detection.ipynb) dallo stesso
    repo Hugging Face del modello. Se il file non è ancora stato
    pubblicato (es. modello non ancora ri-addestrato con la pipeline di
    calibrazione), usa i default legacy - stesso comportamento di prima
    (softmax grezzo, banda 35-65) senza rompere nulla.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    hf_auth = {"token": hf_token} if hf_token else {}
    if HF_MODEL_REVISION:
        hf_auth["revision"] = HF_MODEL_REVISION
    try:
        path = hf_hub_download(repo_id=HF_MODEL_ID, filename=CALIBRATION_FILENAME, **hf_auth)
        with open(path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        calibration.setdefault("temperature", DEFAULT_TEMPERATURE)
        calibration.setdefault("threshold", DEFAULT_THRESHOLD)
        calibration.setdefault("band", DEFAULT_BAND)
        calibration.setdefault("positive_label_id", DEFAULT_POSITIVE_LABEL_ID)
        calibration["source"] = "huggingface"
        return calibration
    except (HfHubHTTPError, FileNotFoundError, OSError):
        return {
            "temperature": DEFAULT_TEMPERATURE,
            "threshold": DEFAULT_THRESHOLD,
            "band": DEFAULT_BAND,
            "positive_label_id": DEFAULT_POSITIVE_LABEL_ID,
            "source": "default (nessun calibration.json pubblicato)",
        }


def get_core_backend():
    return init_core_backend()


def get_backend():
    parser, validator, analyzer = init_core_backend()
    tokenizer, model, model_source = get_content_model()
    calibration = get_calibration()
    return parser, validator, analyzer, tokenizer, model, model_source, calibration


def get_content_model():
    return init_content_model(get_secret("HF_TOKEN"))


def get_calibration() -> dict:
    return init_calibration(get_secret("HF_TOKEN"))


def get_model_source() -> str:
    return f"huggingface ({HF_MODEL_ID}@{HF_MODEL_REVISION or 'main (not pinned)'})"


def warm_up_backend(preload_content_model: bool = True):
    init_core_backend()
    if preload_content_model:
        get_content_model()
