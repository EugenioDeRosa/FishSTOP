import os

import streamlit as st

from src.config import ABUSEIPDB_API_KEY, VIRUSTOTAL_API_KEY, URLHAUS_API_KEY
from src.views.backend import get_model_source


def render():
    st.header("⚙️ Settings")

    model_source = get_model_source()
    if model_source == "company":
        st.success("Modello **aziendale** attivo (`models/company_model`)")
    else:
        st.info("Modello **base** attivo (Kaggle-BERT / HuggingFace)")

    st.divider()
    st.subheader("API Keys")

    c1, c2, c3 = st.columns(3)
    with c1:
        if ABUSEIPDB_API_KEY:
            st.success("AbuseIPDB — configurata")
        else:
            st.error("AbuseIPDB — mancante")
    with c2:
        if VIRUSTOTAL_API_KEY:
            st.success("VirusTotal — configurata")
        else:
            st.error("VirusTotal — mancante")
    with c3:
        if URLHAUS_API_KEY:
            st.success("URLhaus — configurata")
        else:
            st.info("URLhaus — lookup pubblico")

    st.markdown(
        "Imposta le chiavi in un file `.env` nella root del progetto "
        "oppure in **Streamlit Cloud → Settings → Secrets**."
    )

    st.divider()
    st.subheader("Paths")
    st.code(
        "\n".join(
            [
                f"company_model : {os.path.abspath('models/company_model')}",
                f"base_model    : {os.path.abspath('models/saved_models')}",
                f"custom_dataset: {os.path.abspath('data/custom_dataset.csv')}",
            ]
        ),
        language="text",
    )
