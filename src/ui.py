"""Shared visual language for the FishStop Streamlit interface."""

from html import escape

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
            --fs-red: #D92D20;
            --fs-red-bg: #FEF3F2;
            --fs-amber: #B54708;
            --fs-amber-bg: #FFFAEB;
            --fs-blue: #175CD3;
            --fs-blue-bg: #EFF8FF;
            --fs-green: #067647;
            --fs-green-bg: #ECFDF3;
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
        h1, h2, h3, h4 { color: var(--fs-ink); letter-spacing: -.025em; }
        h1 { font-size: clamp(2rem, 4vw, 3rem) !important; }
        h3 { font-size: 1.28rem !important; margin: 1rem 0 .45rem !important; }
        h4 { font-size: 1rem !important; margin: .75rem 0 .35rem !important; }
        p, label { color: var(--fs-muted); }
        [data-testid="stVerticalBlock"] { gap: .65rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--fs-surface);
            border-color: var(--fs-border) !important;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, .035);
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: .75rem .85rem !important;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--fs-border);
            border-radius: 9px;
            padding: .55rem .7rem;
        }
        [data-testid="stMetricLabel"] { color: var(--fs-muted); font-size: .72rem; }
        [data-testid="stMetricValue"] {
            color: var(--fs-ink); font-size: 1.35rem; font-weight: 720;
        }
        [data-testid="stAlert"] {
            border-radius: 8px;
            padding: .5rem .65rem;
        }
        [data-testid="stAlert"] p { font-size: .82rem; line-height: 1.35; }
        [data-testid="stExpander"] details {
            border-color: var(--fs-border) !important;
            border-radius: 9px !important;
            background: #FFFFFF;
        }
        [data-testid="stExpander"] details summary {
            min-height: 2.5rem;
            padding: .45rem .7rem !important;
        }
        [data-testid="stExpander"] details summary p { font-size: .82rem; font-weight: 650; }
        iframe { border: 0 !important; }
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
            gap: .85rem;
            border-bottom: 1px solid var(--fs-border);
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.55rem;
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
        .fs-card {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: .8rem;
            padding: .68rem .78rem;
            margin: .25rem 0;
            border: 1px solid var(--fs-border);
            border-left-width: 5px;
            border-radius: 9px;
            background: #FFFFFF;
        }
        .fs-card__body { min-width: 0; }
        .fs-card__title {
            color: var(--fs-ink); font-size: .86rem; font-weight: 750;
            line-height: 1.25;
        }
        .fs-card__detail {
            color: var(--fs-muted); font-size: .78rem; line-height: 1.35;
            margin-top: .18rem;
        }
        .fs-card__meta { color: #667085; font-size: .7rem; margin-top: .24rem; }
        .fs-card__badge {
            flex: none; border-radius: 999px; padding: .2rem .48rem;
            font-size: .64rem; font-weight: 800; letter-spacing: .03em;
        }
        .fs-card-critical { border-color: var(--fs-red); background: var(--fs-red-bg); }
        .fs-card-critical .fs-card__badge { color: var(--fs-red); background: #FEE4E2; }
        .fs-card-warning { border-color: #F79009; background: var(--fs-amber-bg); }
        .fs-card-warning .fs-card__badge { color: var(--fs-amber); background: #FEF0C7; }
        .fs-card-info { border-color: #2E90FA; background: var(--fs-blue-bg); }
        .fs-card-info .fs-card__badge { color: var(--fs-blue); background: #D1E9FF; }
        .fs-card-success { border-color: #12B76A; background: var(--fs-green-bg); }
        .fs-card-success .fs-card__badge { color: var(--fs-green); background: #D1FADF; }
        .fs-card-neutral { border-color: var(--fs-border); background: #FFFFFF; }
        .fs-card-neutral .fs-card__badge { color: #475467; background: #F2F4F7; }
        .fs-metric-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
            gap: 1px;
            overflow: hidden;
            margin: .35rem 0 .8rem;
            border: 1px solid var(--fs-border);
            border-radius: 9px;
            background: var(--fs-border);
        }
        .fs-metric-strip__item { min-width: 0; padding: .52rem .65rem; background: #FFFFFF; }
        .fs-metric-strip__label {
            color: var(--fs-muted); font-size: .67rem; font-weight: 650;
            text-transform: uppercase; letter-spacing: .04em;
        }
        .fs-metric-strip__value {
            color: var(--fs-ink); font-size: 1.08rem; font-weight: 760;
            margin-top: .08rem; overflow: hidden; text-overflow: ellipsis;
        }
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


def status_card(
    title: str,
    detail: str = "",
    *,
    status: str = "neutral",
    badge: str = "",
    meta: str = "",
    target=None,
) -> None:
    """Render a compact semantic card shared by every FishStop view."""
    normalized = status.lower()
    if normalized not in {"critical", "warning", "info", "success", "neutral"}:
        normalized = "neutral"
    badge_html = (
        f'<span class="fs-card__badge">{escape(badge)}</span>'
        if badge
        else ""
    )
    detail_html = (
        f'<div class="fs-card__detail">{escape(detail)}</div>'
        if detail
        else ""
    )
    meta_html = (
        f'<div class="fs-card__meta">{escape(meta)}</div>'
        if meta
        else ""
    )
    renderer = st if target is None else target
    renderer.markdown(
        f'<div class="fs-card fs-card-{normalized}">'
        f'<div class="fs-card__body"><div class="fs-card__title">{escape(title)}</div>'
        f'{detail_html}{meta_html}</div>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, object]]) -> None:
    """Render dense, visually grouped metrics without separate large cards."""
    cells = "".join(
        '<div class="fs-metric-strip__item">'
        f'<div class="fs-metric-strip__label">{escape(str(label))}</div>'
        f'<div class="fs-metric-strip__value">{escape(str(value))}</div>'
        '</div>'
        for label, value in items
    )
    st.markdown(
        f'<div class="fs-metric-strip">{cells}</div>',
        unsafe_allow_html=True,
    )
