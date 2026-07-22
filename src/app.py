import re
import sys
from pathlib import Path

import streamlit as st

from src.ui import inject_global_styles


def configure_page() -> None:
    st.set_page_config(
        page_title="FishStop | Email analysis",
        page_icon="🛡️",
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
    "dataset_sources": "src.views.dataset_sources",
    "analyze": "src.views.analyzer",
}


def initialize_session_state() -> None:
    if st.session_state.get("page") not in PAGES:
        st.session_state.page = "analyze"


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
    set_page("analyze")
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="fs-brand"><div class="fs-brand-mark">F</div><div>'
            '<div class="fs-brand-name">FishStop</div><div class="fs-brand-note">EMAIL SECURITY</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("ANALYSIS")
        st.button("Analyze email", use_container_width=True, type="primary" if st.session_state.page == "analyze" else "secondary", on_click=set_page, args=("analyze",))
        st.button("Connections", use_container_width=True, type="primary" if st.session_state.page == "settings" else "secondary", on_click=set_page, args=("settings",))

        with st.expander("Project tools", expanded=False):
            st.caption("Dataset preparation is separate from day-to-day analysis.")
            st.button("Training datasets", use_container_width=True, on_click=set_page, args=("dataset_sources",))

        st.markdown("---")
        st.caption("Local-first analysis")
        st.caption(f"Version {APP_VERSION}")


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
    inject_global_styles()
    initialize_session_state()
    if not run_startup_warmup():
        return
    render_sidebar()

    render_selected_page(st.session_state.page)
