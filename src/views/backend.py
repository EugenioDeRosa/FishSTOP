import os

import streamlit as st

from src.analyzer import EmlSOCAnalyzer
from src.parser import EmailParserPipeline
from src.validators import EmailSecurityValidator

HF_MODEL_ID = "eugenioderodev/fishstop-bert"


@st.cache_resource
def init_core_backend():
    parser = EmailParserPipeline()
    validator = EmailSecurityValidator()
    analyzer = EmlSOCAnalyzer()
    return parser, validator, analyzer


@st.cache_resource
def init_content_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    company_path = os.path.join("models", "company_model")
    base_path = os.path.join("models", "saved_models")

    if os.path.isdir(company_path) and os.path.exists(os.path.join(company_path, "config.json")):
        model_path = company_path
        model_source = "company"
    elif os.path.isdir(base_path) and os.path.exists(os.path.join(base_path, "config.json")):
        model_path = base_path
        model_source = "base"
    else:
        model_path = HF_MODEL_ID
        model_source = "base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model, model_source


def get_core_backend():
    return init_core_backend()


def get_backend():
    parser, validator, analyzer = init_core_backend()
    tokenizer, model, model_source = init_content_model()
    return parser, validator, analyzer, tokenizer, model, model_source


def get_content_model():
    return init_content_model()


def get_model_source() -> str:
    return init_content_model()[2]
