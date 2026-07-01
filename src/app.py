import os
import sys

import streamlit as st

st.set_page_config(
    page_title="FishStop - Triage & Phishing Detection",
    page_icon="🛡️",
    layout="wide",
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.views import analyzer, dataset_sources, settings, train


if "page" not in st.session_state:
    st.session_state.page = "home"


def render_home():
    st.title("🛡️ FishStop")
    st.markdown("Email Security Platform")
    st.divider()
    st.markdown("### Seleziona una sezione")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("⚙️ Settings", use_container_width=True, type="primary"):
            st.session_state.page = "settings"
            st.rerun()
    with c2:
        if st.button("🗃️ Train Dataset", use_container_width=True, type="primary"):
            st.session_state.page = "train"
            st.rerun()
    with c3:
        if st.button("🔍 Analyze EML", use_container_width=True, type="primary"):
            st.session_state.page = "analyze"
            st.rerun()
    with c4:
        if st.button("🌐 Public Datasets", use_container_width=True, type="primary"):
            st.session_state.page = "dataset_sources"
            st.rerun()


with st.sidebar:
    st.markdown("## 🛡️ FishStop")

    if st.session_state.page != "home":
        if st.button("← Menu principale", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        st.divider()

    if st.session_state.page == "analyze":
        if st.session_state.get("raw_eml_debug_data"):
            st.markdown("### 🪲 Raw EML Debugger (Cleaned)")
            current_file_name = st.session_state.get("current_eml_name", "default")
            st.text_area(
                label="Contenuto MIME (Senza blocchi Encode)",
                value=st.session_state["raw_eml_debug_data"],
                height=500,
                disabled=True,
                key=f"sidebar_debug_{current_file_name}",
            )
            st.divider()

    st.caption("FishStop — Email Security Platform")


page = st.session_state.page
if page == "home":
    render_home()
elif page == "settings":
    settings.render()
elif page == "train":
    train.render()
elif page == "dataset_sources":
    dataset_sources.render()
elif page == "analyze":
    analyzer.render()
