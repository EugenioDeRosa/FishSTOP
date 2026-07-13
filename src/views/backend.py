import streamlit as st

from src.config import HF_TOKEN

HF_MODEL_ID = "eugenioderodev/fishstop-bert"


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
def init_content_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_auth = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, **hf_auth)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID, **hf_auth)

    return tokenizer, model, "huggingface"


def get_core_backend():
    return init_core_backend()


def get_backend():
    parser, validator, analyzer = init_core_backend()
    tokenizer, model, model_source = init_content_model()
    return parser, validator, analyzer, tokenizer, model, model_source


def get_content_model():
    return init_content_model()


def get_model_source() -> str:
    return f"huggingface ({HF_MODEL_ID})"


def warm_up_backend():
    init_core_backend()
