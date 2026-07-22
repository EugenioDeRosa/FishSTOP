"""Shared visual language for the FishStop Streamlit interface."""

import streamlit as st


PALETTE = {
    "ink": "#0B1220",
    "muted": "#526077",
    "primary": "#2563EB",
    "accent": "#0F9F8F",
    "surface": "#FFFFFF",
    "canvas": "#F4F7FB",
    "border": "#DCE3EC",
}


def inject_global_styles() -> None:
    """Apply the compact, analysis-first FishStop theme."""
    st.markdown(
        """
        <style>
        :root {
            --fs-ink: #0B1220;
            --fs-muted: #526077;
            --fs-primary: #2563EB;
            --fs-accent: #0F9F8F;
            --fs-canvas: #F4F7FB;
            --fs-surface: #FFFFFF;
            --fs-border: #DCE3EC;
        }
        .stApp { background: var(--fs-canvas); }
        [data-testid="stHeader"] {
            min-height: 3rem;
            background: rgba(244, 247, 251, .94);
            backdrop-filter: blur(10px);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1240px;
            padding-top: 4.25rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: #0B1220;
            border-right: 1px solid #1D2939;
        }
        [data-testid="stSidebar"] * { color: #D7E0EC; }
        [data-testid="stSidebar"] .stButton > button {
            background: transparent;
            border: 1px solid transparent;
            color: #D7E0EC;
            text-align: left;
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: #162236;
            border-color: #2A3A52;
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary p,
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary span {
            color: #D7E0EC !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary {
            background: transparent !important;
            border-color: #2A3A52 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover {
            background: #162236 !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary svg {
            fill: #D7E0EC !important;
            color: #D7E0EC !important;
        }
        h1, h2, h3 { color: var(--fs-ink); letter-spacing: -.025em; }
        h1 { font-size: clamp(2rem, 4vw, 3rem) !important; }
        p, label { color: var(--fs-muted); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--fs-surface);
            border-color: var(--fs-border) !important;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, .045);
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--fs-border);
            border-radius: 12px;
            padding: .85rem 1rem;
        }
        [data-testid="stMetricLabel"] { color: var(--fs-muted); }
        [data-testid="stMetricValue"] { color: var(--fs-ink); font-weight: 720; }
        [data-testid="stFileUploaderDropzone"] {
            background: #F8FAFD;
            border: 1.5px dashed #AAB8CB;
            border-radius: 12px;
            padding: 1.1rem;
        }
        .stButton > button[kind="primary"] {
            background: var(--fs-primary);
            border-color: var(--fs-primary);
            border-radius: 9px;
            font-weight: 650;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.2rem;
            border-bottom: 1px solid var(--fs-border);
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding: 0 .1rem;
            color: var(--fs-muted);
            white-space: nowrap;
        }
        .stTabs [aria-selected="true"] { color: var(--fs-primary); font-weight: 650; }
        .fs-eyebrow {
            color: var(--fs-primary);
            font-size: .75rem;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .4rem;
        }
        .fs-page-subtitle { color: var(--fs-muted); max-width: 690px; margin: -.35rem 0 1.6rem; }
        .fs-brand { display: flex; align-items: center; gap: .75rem; margin: .25rem 0 1.8rem; }
        .fs-brand-mark {
            display: grid; place-items: center; width: 2rem; height: 2rem;
            border-radius: 9px; background: linear-gradient(135deg, #2F6FED, #0F9F8F);
            color: white !important; font-weight: 800;
        }
        .fs-brand-name { color: #FFFFFF !important; font-size: 1.05rem; font-weight: 750; }
        .fs-brand-note { color: #8FA0B7 !important; font-size: .73rem; }
        .fs-risk {
            display: flex; align-items: center; justify-content: space-between; gap: 1rem;
            padding: 1.05rem 1.2rem; border-radius: 12px; margin: .15rem 0 1rem;
            border: 1px solid;
        }
        .fs-risk strong { display: block; color: inherit; font-size: 1.02rem; }
        .fs-risk span { font-size: .86rem; opacity: .82; }
        .fs-risk-critical { color: #991B1B; background: #FEF2F2; border-color: #FECACA; }
        .fs-risk-suspicious { color: #92400E; background: #FFFBEB; border-color: #FDE68A; }
        .fs-risk-watch { color: #1E40AF; background: #EFF6FF; border-color: #BFDBFE; }
        .fs-risk-low { color: #065F46; background: #ECFDF5; border-color: #A7F3D0; }
        .fs-section-label { color: var(--fs-ink); font-weight: 720; margin-bottom: .3rem; }
        footer { visibility: hidden; }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] { padding-top: 3.75rem; }
            .fs-risk { align-items: flex-start; flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="fs-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="fs-page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def risk_banner(severity: str, caption: str) -> None:
    css_class = severity.lower() if severity.lower() in {"critical", "suspicious", "watch", "low"} else "watch"
    label = {
        "CRITICAL": "High risk",
        "SUSPICIOUS": "Needs review",
        "WATCH": "Monitor",
        "LOW": "Low risk",
    }.get(severity, severity.title())
    st.markdown(
        f'<div class="fs-risk fs-risk-{css_class}"><div><strong>{label}</strong><span>{caption}</span></div>'
        f'<strong>{severity}</strong></div>',
        unsafe_allow_html=True,
    )
