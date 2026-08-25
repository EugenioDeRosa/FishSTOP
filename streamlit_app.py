"""Public Streamlit landing page for FishSTOP desktop installers."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DESKTOP_REPOSITORY = "EugenioDeRosa/fishstop-desktop-email-security"
RELEASES_URL = f"https://github.com/{DESKTOP_REPOSITORY}/releases"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{DESKTOP_REPOSITORY}/releases/latest"


@st.cache_data(ttl=900, show_spinner=False)
def latest_release() -> dict[str, Any] | None:
    """Return the latest public desktop release, or None until one is published."""
    try:
        response = requests.get(
            LATEST_RELEASE_URL,
            headers={"Accept": "application/vnd.github+json"},
            timeout=8,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def installer_for(release: dict[str, Any] | None, platform: str) -> dict[str, Any] | None:
    if not release:
        return None
    extensions = (".dmg",) if platform == "macos" else (".msi", ".exe")
    return next(
        (
            asset
            for asset in release.get("assets", [])
            if str(asset.get("name", "")).lower().endswith(extensions)
        ),
        None,
    )


def download_card(platform: str, title: str, description: str, asset: dict[str, Any] | None, version: str | None) -> None:
    st.markdown(f"<p class='platform-label'>{platform}</p><h2>{title}</h2><p class='platform-copy'>{description}</p>", unsafe_allow_html=True)
    if asset:
        st.link_button("Scarica ora  →", asset["browser_download_url"], use_container_width=True)
        st.caption(f"{version} · {asset['name']}")
    else:
        st.link_button("Vedi le release  →", RELEASES_URL, use_container_width=True)
        st.caption("Installer non ancora pubblicato")


st.set_page_config(page_title="FishSTOP — Download", page_icon="🎣", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .stApp { background: #07120f; color: #e5f4e9; }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { max-width: 1140px; padding-top: 2rem; padding-bottom: 4.5rem; }
      .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -.05em; margin: 0 0 5.7rem; }
      .brand span { display:inline-grid; place-items:center; width:27px; height:27px; margin-right:8px; border-radius:50% 50% 50% 4px; background:#bdff60; color:#102317; font-size:.8rem; }
      .eyebrow, .platform-label { color:#bdff60; font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
      .eyebrow::before { content:""; display:inline-block; width:8px; height:8px; margin-right:8px; border-radius:50%; background:#bdff60; box-shadow:0 0 16px #bdff60; }
      .hero-title { max-width:830px; margin:0; font-size:clamp(3.4rem,7vw,6.7rem); font-weight:760; line-height:.93; letter-spacing:-.07em; }
      .hero-title em { color:#bdff60; font-style:normal; }
      .intro { max-width:690px; margin:1.8rem 0 4.5rem; color:#9eb5a5; font-size:1.15rem; line-height:1.65; }
      .download-title { margin:0 0 .8rem; font-size:clamp(2.2rem,4vw,3.6rem); letter-spacing:-.06em; }
      .release-status { color:#9eb5a5; margin-bottom:1.5rem; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { background:#0d1d17; border:1px solid #294238; padding:1.8rem; }
      h2 { margin:.8rem 0 .55rem; font-size:1.85rem; letter-spacing:-.055em; } .platform-copy { min-height:50px; color:#9eb5a5; line-height:1.55; }
      [data-testid="stLinkButton"] a { border:0; border-radius:0; background:#bdff60; color:#112016; font-weight:800; } [data-testid="stLinkButton"] a:hover { background:#d1ff8b; color:#112016; }
      [data-testid="stCaptionContainer"] { color:#9eb5a5; margin-top:.55rem; }
      .rule { border:0; border-top:1px solid #294238; margin:5.3rem 0 2rem; }
      .feature { color:#9eb5a5; font-size:.92rem; line-height:1.55; } .feature strong { display:block; color:#e5f4e9; font-size:1.06rem; margin:.65rem 0; }
      @media (max-width: 740px) { .brand { margin-bottom:3.5rem; }.hero-title { font-size:3.35rem; }.intro { margin-bottom:3rem; } [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { margin-bottom:1rem; } }
    </style>
    <p class="brand"><span>F</span>FishSTOP</p>
    <p class="eyebrow">Analisi email locale</p>
    <h1 class="hero-title">Le email sospette meritano un’analisi che resta <em>con te.</em></h1>
    <p class="intro">FishSTOP è l’app desktop per analizzare file <code>.eml</code>, verificare gli indicatori di rischio e supportare il triage SOC senza caricare il contenuto delle email su un server.</p>
    <p class="eyebrow">Desktop app</p>
    <h1 class="download-title">Scegli il tuo sistema.</h1>
    """,
    unsafe_allow_html=True,
)

release = latest_release()
version = release.get("name") or release.get("tag_name") if release else None
if release:
    st.markdown(f"<p class='release-status'>Ultima versione disponibile: <strong>{version}</strong></p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='release-status'>Gli installer appariranno qui appena sarà pubblicata la prima release desktop.</p>", unsafe_allow_html=True)

mac, windows = st.columns(2, gap="medium")
with mac:
    download_card("macOS", "FishSTOP per Mac", "Per Mac con Apple Silicon e processore Intel.", installer_for(release, "macos"), version)
with windows:
    download_card("Windows", "FishSTOP per Windows", "Installer per PC Windows a 64 bit.", installer_for(release, "windows"), version)

st.markdown("<hr class='rule'>", unsafe_allow_html=True)
one, two, three = st.columns(3, gap="medium")
for column, number, title, copy in (
    (one, "01", "File sotto la lente.", "Parsing di struttura MIME, intestazioni, corpo, link e allegati."),
    (two, "02", "Indicatori verificabili.", "SPF, DKIM, DMARC, dominio, percorso e reputazione in un’unica analisi."),
    (three, "03", "Privacy per impostazione.", "L’elaborazione avviene sul dispositivo. Il risultato non sostituisce la verifica umana."),
):
    with column:
        st.markdown(f"<p class='feature'><span class='platform-label'>{number}</span><strong>{title}</strong>{copy}</p>", unsafe_allow_html=True)

st.markdown(f"<p class='release-status' style='margin-top:4rem'>FishSTOP · Progetto di tesi · <a href='{RELEASES_URL}' target='_blank'>Tutte le release ↗</a></p>", unsafe_allow_html=True)
