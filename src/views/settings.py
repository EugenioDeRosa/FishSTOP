import streamlit as st

from src.config import MANAGED_API_KEYS, get_secret, get_secret_source, set_user_secret
from src.views import backend
from src.views.backend import get_model_source
from src.analyzer.llm_context_analyzer import active_llm_backend


KEY_LABELS = {
    "VIRUSTOTAL_API_KEY": "VirusTotal API key",
    "ABUSEIPDB_API_KEY": "AbuseIPDB API key",
    "GITHUB_MODELS_TOKEN": "GitHub Models token",
    "HF_TOKEN": "Hugging Face token",
}

KEY_HELP = {
    "VIRUSTOTAL_API_KEY": "Used for URL and attachment hash reputation checks.",
    "ABUSEIPDB_API_KEY": "Used for sender IP and resolved-domain reputation checks.",
    "GITHUB_MODELS_TOKEN": "Used by Phi-4 mini hosted analysis.",
    "HF_TOKEN": "Used when loading the Hugging Face BERT model, especially private/gated access.",
}


def _masked(value: str) -> str:
    value = str(value or "")
    if not value:
        return "missing"
    if len(value) <= 8:
        return "configured"
    return f"{value[:4]}...{value[-4:]}"


def render():
    st.header("Settings")

    model_source = get_model_source()
    st.info(f"Active BERT model from Hugging Face (`{model_source}`).")

    st.divider()
    st.subheader("API Keys")
    st.caption("Keys entered here are used first for API requests in this Streamlit session. If a field is empty, FishSTOP falls back to `.env`, environment variables, or Streamlit secrets.")

    with st.form("api_keys_form"):
        pending_values = {}
        for key in MANAGED_API_KEYS:
            current = get_secret(key)
            source = get_secret_source(key)
            st.caption(f"{KEY_LABELS[key]}: {_masked(current)} - source: {source}")
            pending_values[key] = st.text_input(
                KEY_LABELS[key],
                value="",
                type="password",
                placeholder="Leave empty to keep current/fallback value",
                help=KEY_HELP[key],
                key=f"settings_input_{key}",
            )

        col_save, col_clear = st.columns(2)
        save = col_save.form_submit_button("Save keys", use_container_width=True)
        clear = col_clear.form_submit_button("Clear session keys", use_container_width=True)

    if save:
        for key, value in pending_values.items():
            if value.strip():
                set_user_secret(key, value)
        backend.init_content_model.clear()
        st.success("API keys saved for this session. New requests will use them first.")
        st.rerun()

    if clear:
        for key in MANAGED_API_KEYS:
            set_user_secret(key, "")
        backend.init_content_model.clear()
        st.success("Session API keys cleared. FishSTOP will use .env/environment/secrets fallbacks.")
        st.rerun()

    st.divider()
    st.subheader("Status")
    c1, c2, c3, c4 = st.columns(4)
    status_cols = {
        "VIRUSTOTAL_API_KEY": c1,
        "ABUSEIPDB_API_KEY": c2,
        "GITHUB_MODELS_TOKEN": c3,
        "HF_TOKEN": c4,
    }
    for key, col in status_cols.items():
        with col:
            if get_secret(key):
                st.success(KEY_LABELS[key])
            else:
                st.error(KEY_LABELS[key])
            st.caption(get_secret_source(key))

    st.divider()
    st.subheader("Online models")
    st.code(
        "\n".join(
            [
                "bert_model : https://huggingface.co/eugenioderodev/fishstop-bert",
                f"phi4_mini  : {active_llm_backend()}",
                "local dev  : Ollama auto-selected when reachable at OLLAMA_CHAT_ENDPOINT",
                "hosted     : GitHub Models fallback via GITHUB_MODELS_TOKEN",
            ]
        ),
        language="text",
    )

