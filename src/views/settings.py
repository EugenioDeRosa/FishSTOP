import streamlit as st

from src.config import MANAGED_API_KEYS, get_secret, get_secret_source, set_user_secret
from src.views.backend import get_model_source
from src.analyzer.llm_context_analyzer import active_llm_backend
from src.ui import page_intro, status_card


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
    "HF_TOKEN": "Used when loading the Hugging Face DistilBERT model, especially private/gated access.",
}


def _masked(value: str) -> str:
    value = str(value or "")
    if not value:
        return "missing"
    if len(value) <= 8:
        return "configured"
    return f"{value[:4]}...{value[-4:]}"


def render():
    page_intro(
        "Connections",
        "Analysis services",
        "Manage optional reputation and AI services used during email analysis.",
    )

    model_source = get_model_source()
    status_card(
        "Content classifier",
        f"Active model: {model_source}",
        status="success",
        badge="READY",
    )

    st.subheader("Service credentials")
    st.caption("Credentials entered here are kept for this session. Leave a field empty to keep its current value.")

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
        st.success("API keys saved for this session. New requests will use them first.")
        st.rerun()

    if clear:
        for key in MANAGED_API_KEYS:
            set_user_secret(key, "")
        st.success("Session API keys cleared. FishSTOP will use .env/environment/secrets fallbacks.")
        st.rerun()

    st.subheader("Connection status")
    c1, c2, c3, c4 = st.columns(4)
    status_cols = {
        "VIRUSTOTAL_API_KEY": c1,
        "ABUSEIPDB_API_KEY": c2,
        "GITHUB_MODELS_TOKEN": c3,
        "HF_TOKEN": c4,
    }
    for key, col in status_cols.items():
        with col:
            configured = bool(get_secret(key))
            status_card(
                KEY_LABELS[key],
                f"Source: {get_secret_source(key)}",
                status="success" if configured else "warning",
                badge="READY" if configured else "MISSING",
            )

    with st.expander("Model details", expanded=False):
        st.write("**DistilBERT:** Hugging Face content classification")
        st.write(f"**Phi-4 mini:** {active_llm_backend()}")
        st.caption("A reachable local Ollama service is preferred; GitHub Models is used as fallback when configured.")
