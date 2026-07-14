import re
import sys
from pathlib import Path

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="FishStop - Triage & Phishing Detection",
        page_icon="shield",
        layout="wide",
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_app_version() -> str:
    try:
        version_text = (PROJECT_ROOT / "src" / "version.py").read_text(encoding="utf-8-sig")
        match = re.search(r'APP_VERSION\s*=\s*["\']([^"\']+)["\']', version_text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


APP_VERSION = _load_app_version()


PAGES = {
    "settings": "src.views.settings",
    "train": "src.views.train",
    "dataset_sources": "src.views.dataset_sources",
    "analyze": "src.views.analyzer",
}


def initialize_session_state() -> None:
    if st.session_state.get("page") not in {"home", *PAGES}:
        st.session_state.page = "home"


def _render_startup_splash():
    st.markdown(
        f"""
        <style>
            .fishstop-splash {{
                min-height: 72vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 4rem 1rem;
                background:
                    radial-gradient(circle at top left, rgba(36, 126, 255, 0.24), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(0, 209, 178, 0.18), transparent 24%),
                    linear-gradient(135deg, #07111f 0%, #0b1a33 52%, #081018 100%);
                border-radius: 28px;
                overflow: hidden;
                position: relative;
            }}
            .fishstop-splash::before {{
                content: "";
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
                background-size: 44px 44px;
                mask-image: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent 92%);
                pointer-events: none;
            }}
            .fishstop-card {{
                position: relative;
                z-index: 1;
                width: min(720px, 100%);
                padding: 2.2rem 2rem;
                border-radius: 24px;
                background: rgba(5, 13, 24, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
                backdrop-filter: blur(18px);
            }}
            .fishstop-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.08);
                color: #b6d7ff;
                font-size: 0.82rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}
            .fishstop-dot {{
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 999px;
                background: linear-gradient(135deg, #4cf3ff, #7c8dff);
                box-shadow: 0 0 16px rgba(76, 243, 255, 0.9);
                animation: fishstopPulse 1.6s ease-in-out infinite;
            }}
            .fishstop-title {{
                margin: 1rem 0 0.4rem;
                font-size: clamp(2rem, 4vw, 3.5rem);
                line-height: 1.05;
                font-weight: 800;
                color: #f5fbff;
                letter-spacing: -0.04em;
            }}
            .fishstop-subtitle {{
                margin: 0;
                font-size: 1.02rem;
                line-height: 1.6;
                color: rgba(231, 240, 255, 0.8);
                max-width: 58ch;
            }}
            .fishstop-loader {{
                margin-top: 1.7rem;
                height: 0.8rem;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}
            .fishstop-loader > span {{
                display: block;
                width: 45%;
                height: 100%;
                border-radius: inherit;
                background: linear-gradient(90deg, #4cf3ff, #7c8dff, #00d1b2);
                animation: fishstopSweep 1.2s ease-in-out infinite;
            }}
            .fishstop-meta {{
                margin-top: 1rem;
                color: rgba(231, 240, 255, 0.62);
                font-size: 0.92rem;
            }}
            @keyframes fishstopPulse {{
                0%, 100% {{ transform: scale(0.95); opacity: 0.72; }}
                50% {{ transform: scale(1.08); opacity: 1; }}
            }}
            @keyframes fishstopSweep {{
                0% {{ transform: translateX(-120%); }}
                100% {{ transform: translateX(260%); }}
            }}
        </style>
        <div class="fishstop-splash">
            <div class="fishstop-card">
                <div class="fishstop-badge"><span class="fishstop-dot"></span> FishStop startup</div>
                <div class="fishstop-title">Warming up the triage engine</div>
                <p class="fishstop-subtitle">
                    Loading the core parser, SOC backend and BERT model so EML analysis starts fast once the dashboard appears.
                </p>
                <div class="fishstop-loader"><span></span></div>
                <div class="fishstop-meta">Build {APP_VERSION} · Preparing secure email analysis workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_startup_warmup() -> bool:
    if st.session_state.get("startup_warmup_done"):
        return True

    _render_startup_splash()
    try:
        from src.views.backend import warm_up_backend

        warm_up_backend(preload_content_model=True)
        st.session_state["startup_warmup_error"] = None
    except Exception as exc:
        st.session_state["startup_warmup_error"] = str(exc)
    st.session_state["startup_warmup_done"] = True
    st.rerun()
    return False



def set_page(page_name: str) -> None:
    st.session_state.page = page_name


def render_home() -> None:
    st.title("FishStop")
    st.markdown("Email Security Platform")
    st.divider()
    st.markdown("### Select a section")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.button(
            "Settings",
            use_container_width=True,
            type="primary",
            on_click=set_page,
            args=("settings",),
        )
    with c2:
        st.button(
            "Colab Training",
            use_container_width=True,
            type="primary",
            on_click=set_page,
            args=("train",),
        )
    with c3:
        st.button(
            "Analyze EML",
            use_container_width=True,
            type="primary",
            on_click=set_page,
            args=("analyze",),
        )
    with c4:
        st.button(
            "Public Datasets",
            use_container_width=True,
            type="primary",
            on_click=set_page,
            args=("dataset_sources",),
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## FishStop")
        st.caption(f"Build: `{APP_VERSION}`")

        if st.session_state.page != "home":
            st.button(
                "Main menu",
                use_container_width=True,
                on_click=set_page,
                args=("home",),
            )
            st.divider()

        if st.session_state.page == "analyze":
            if st.session_state.get("raw_eml_debug_data"):
                st.markdown("### Raw EML Debugger (Cleaned)")
                current_file_name = st.session_state.get("current_eml_name", "default")
                st.text_area(
                    label="MIME content (without encoded blocks)",
                    value=st.session_state["raw_eml_debug_data"],
                    height=500,
                    disabled=True,
                    key=f"sidebar_debug_{current_file_name}",
                )
                st.divider()

        st.caption("FishStop - Email Security Platform")


def render_selected_page(page_name: str) -> None:
    import importlib
    import traceback

    module_name = PAGES.get(page_name)
    if not module_name:
        st.session_state.page = "home"
        render_home()
        return

    try:
        importlib.invalidate_caches()
        page_module = importlib.import_module(module_name)
        render = getattr(page_module, "render")
        render()
    except Exception as exc:
        st.error(f"Unable to render page `{page_name}`.")
        st.caption("The app caught the error instead of leaving a blank screen.")
        with st.expander("Error details", expanded=True):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), language="text")
        if st.button("Back to main menu", use_container_width=True):
            st.session_state.page = "home"


def main() -> None:
    configure_page()
    initialize_session_state()
    if not run_startup_warmup():
        return
    render_sidebar()

    if st.session_state.page == "home":
        render_home()
    else:
        render_selected_page(st.session_state.page)
