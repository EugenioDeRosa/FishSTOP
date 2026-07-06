import sys
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="FishStop - Triage & Phishing Detection",
    page_icon="shield",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.version import APP_VERSION
from src.views import analyzer, dataset_sources, settings, train


PAGES = {
    "settings": settings.render,
    "train": train.render,
    "dataset_sources": dataset_sources.render,
    "analyze": analyzer.render,
}

if st.session_state.get("page") not in {"home", *PAGES}:
    st.session_state.page = "home"


def render_home():
    st.title("FishStop")
    st.markdown("Email Security Platform")
    st.divider()
    st.markdown("### Seleziona una sezione")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Settings", use_container_width=True, type="primary"):
            st.session_state.page = "settings"
            st.rerun()
    with c2:
        if st.button("Colab Training", use_container_width=True, type="primary"):
            st.session_state.page = "train"
            st.rerun()
    with c3:
        if st.button("Analyze EML", use_container_width=True, type="primary"):
            st.session_state.page = "analyze"
            st.rerun()
    with c4:
        if st.button("Public Datasets", use_container_width=True, type="primary"):
            st.session_state.page = "dataset_sources"
            st.rerun()


with st.sidebar:
    st.markdown("## FishStop")
    st.caption(f"Build: `{APP_VERSION}`")

    if st.session_state.page != "home":
        if st.button("Menu principale", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        st.divider()

    if st.session_state.page == "analyze":
        if st.session_state.get("raw_eml_debug_data"):
            st.markdown("### Raw EML Debugger (Cleaned)")
            current_file_name = st.session_state.get("current_eml_name", "default")
            st.text_area(
                label="Contenuto MIME (senza blocchi encoded)",
                value=st.session_state["raw_eml_debug_data"],
                height=500,
                disabled=True,
                key=f"sidebar_debug_{current_file_name}",
            )
            st.divider()

    st.caption("FishStop - Email Security Platform")


if st.session_state.page == "home":
    render_home()
else:
    PAGES[st.session_state.page]()
