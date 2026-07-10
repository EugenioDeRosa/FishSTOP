import streamlit as st

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

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)

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
