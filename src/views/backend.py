import os

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.analyzer import EmlSOCAnalyzer
from src.parser import EmailParserPipeline
from src.validators import EmailSecurityValidator

HF_MODEL_ID = "eugenioderodev/fishstop-bert"


@st.cache_resource
def init_backend():
    parser = EmailParserPipeline()
    validator = EmailSecurityValidator()
    analyzer = EmlSOCAnalyzer()

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

    return parser, validator, analyzer, tokenizer, model, model_source


def get_backend():
    return init_backend()


def get_model_source() -> str:
    return init_backend()[5]
