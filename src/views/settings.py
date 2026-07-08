import streamlit as st

from src.config import ABUSEIPDB_API_KEY, VIRUSTOTAL_API_KEY
from src.views.backend import get_model_source


def render():
    st.header("Settings")

    model_source = get_model_source()
    st.info(f"Modello BERT attivo da Hugging Face (`{model_source}`).")

    st.divider()
    st.subheader("API Keys")

    c1, c2 = st.columns(2)
    with c1:
        if ABUSEIPDB_API_KEY:
            st.success("AbuseIPDB configurata")
        else:
            st.error("AbuseIPDB mancante")
    with c2:
        if VIRUSTOTAL_API_KEY:
            st.success("VirusTotal configurata per URL e file")
        else:
            st.error("VirusTotal mancante")

    st.markdown(
        "Imposta le chiavi in un file `.env` nella root del progetto "
        "oppure in Streamlit Cloud -> Settings -> Secrets."
    )

    st.divider()
    st.subheader("Modelli online")
    st.code(
        "\n".join(
            [
                "bert_model : https://huggingface.co/eugenioderodev/fishstop-bert",
                "phi4_mini  : GitHub Models / Azure AI endpoint via GITHUB_MODELS_TOKEN",
            ]
        ),
        language="text",
    )
