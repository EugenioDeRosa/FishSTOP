"""Public Streamlit landing page for FishStop desktop installers."""

from __future__ import annotations

from html import escape
from typing import Any

import requests
import streamlit as st


DESKTOP_REPOSITORY = "EugenioDeRosa/fishstop-desktop-email-security"
RELEASES_URL = f"https://github.com/{DESKTOP_REPOSITORY}/releases"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{DESKTOP_REPOSITORY}/releases/latest"


@st.cache_data(ttl=900, show_spinner=False)
def latest_release() -> dict[str, Any] | None:
    try:
        response = requests.get(LATEST_RELEASE_URL, headers={"Accept": "application/vnd.github+json"}, timeout=8)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def installer_for(release: dict[str, Any] | None, platform: str, architecture: str | None = None) -> dict[str, Any] | None:
    if not release:
        return None
    extensions = (".dmg",) if platform == "macos" else (".msi", ".exe")
    candidates = [asset for asset in release.get("assets", []) if str(asset.get("name", "")).lower().endswith(extensions)]
    if platform != "macos" or architecture is None:
        return candidates[0] if candidates else None
    markers = ("aarch64", "arm64") if architecture == "apple-silicon" else ("x86_64", "x64", "intel")
    matching = next((asset for asset in candidates if any(marker in str(asset.get("name", "")).lower() for marker in markers)), None)
    return matching or (candidates[0] if len(candidates) == 1 else None)


def download_card(icon: str, platform: str, title: str, description: str, asset: dict[str, Any] | None, version: str | None) -> None:
    st.markdown(
        f"<div class='download-card-head'><span class='platform-icon'>{icon}</span><p class='platform-label'>{escape(platform)}</p></div>"
        f"<h2>{escape(title)}</h2><p class='platform-copy'>{escape(description)}</p>",
        unsafe_allow_html=True,
    )
    if asset:
        st.link_button("Scarica l’installer", asset["browser_download_url"], use_container_width=True)
        st.caption(f"{version} · {asset['name']}")
    else:
        st.link_button("Vedi tutte le release", RELEASES_URL, use_container_width=True)
        st.caption("Installer in arrivo")


st.set_page_config(page_title="FishStop — Download", page_icon="◒", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      :root { --ink:#0b1220; --muted:#64748b; --line:#e5eaf1; --canvas:#f7f9fc; --blue:#2563eb; --teal:#0f9f8f; --aqua:#dff8f3; }
      html { scroll-behavior:smooth; } html,body,[class*="css"] { font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Inter",sans-serif; }
      .stApp { background:var(--canvas); color:var(--ink); } [data-testid="stHeader"] { height:0; background:transparent; }
      #MainMenu,footer,[data-testid="stToolbar"] { display:none; }
      [data-testid="stMainBlockContainer"] { max-width:1180px; padding:1.35rem 1.5rem 3rem; }
      .nav { display:flex; align-items:center; justify-content:space-between; min-height:3rem; margin-bottom:clamp(4.5rem,10vw,8.5rem); }
      .wordmark { margin:0; font-size:1.17rem; letter-spacing:-.065em; font-weight:760; } .fish { color:var(--ink); } .stop { color:var(--teal); }
      .nav-note { margin:0; color:#8a98ab; font-size:.75rem; font-weight:650; letter-spacing:.1em; text-transform:uppercase; }
      .hero { position:relative; isolation:isolate; max-width:930px; padding-bottom:5.7rem; }
      .hero::before { content:""; position:absolute; z-index:-1; width:600px; height:430px; left:-230px; top:-220px; border-radius:50%; background:radial-gradient(circle,rgba(37,99,235,.14),rgba(15,159,143,.07) 41%,transparent 72%); filter:blur(8px); pointer-events:none; }
      .eyebrow { display:inline-flex; align-items:center; gap:.5rem; margin:0 0 1.25rem; color:var(--teal); font-size:.73rem; font-weight:760; letter-spacing:.13em; text-transform:uppercase; }
      .eyebrow::before { content:""; width:.48rem; height:.48rem; border-radius:999px; background:var(--teal); box-shadow:0 0 0 5px rgba(15,159,143,.11); }
      .hero h1 { max-width:880px; margin:0; color:var(--ink); font-size:clamp(3.55rem,8.1vw,7.3rem)!important; font-weight:720; line-height:.93; letter-spacing:-.078em; }
      .hero h1 em { color:var(--teal); font-style:normal; } .hero-link { color:inherit; text-decoration:none; border-bottom:2px solid rgba(15,159,143,.25); transition:border-color .2s ease,color .2s ease; } .hero-link:hover,.hero-link:focus-visible { color:#08796d; border-color:currentColor; outline:none; } .hero-copy { max-width:630px; margin:1.7rem 0 0; color:var(--muted); font-size:1.15rem; line-height:1.65; letter-spacing:-.012em; }
      .download-section { border-top:1px solid var(--line); padding-top:2.15rem; scroll-margin-top:2rem; } .download-kicker { margin:0 0 .45rem; color:var(--blue); font-size:.73rem; font-weight:760; letter-spacing:.13em; text-transform:uppercase; }
      .download-section h3 { margin:0; color:var(--ink); font-size:clamp(2rem,4vw,3rem); line-height:1; letter-spacing:-.062em; } .release-status { margin:.85rem 0 1.8rem; color:var(--muted); font-size:.9rem; } .release-status strong { color:var(--ink); font-weight:700; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-height:278px; padding:1.45rem; border:1px solid var(--line); border-radius:22px; background:rgba(255,255,255,.84); box-shadow:0 2px 5px rgba(15,23,42,.018),0 14px 32px rgba(15,23,42,.035); transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:hover { transform:translateY(-4px); border-color:#cdd8e7; box-shadow:0 20px 42px rgba(15,23,42,.08); }
      .download-card-head { display:flex; align-items:center; gap:.72rem; } .platform-icon { display:grid; place-items:center; width:2.1rem; height:2.1rem; border-radius:10px; background:linear-gradient(145deg,#eff6ff,var(--aqua)); color:var(--teal); font-size:1.12rem; }
      .platform-label { margin:0; color:#75839a; font-size:.72rem; font-weight:740; letter-spacing:.08em; text-transform:uppercase; } h2 { margin:1.25rem 0 .52rem!important; color:var(--ink); font-size:1.45rem!important; letter-spacing:-.045em; }
      .platform-copy { min-height:3rem; margin:0 0 1.3rem; color:var(--muted); font-size:.91rem; line-height:1.55; }
      [data-testid="stLinkButton"] a { min-height:2.85rem; border:0; border-radius:999px; background:var(--ink); color:white; font-size:.88rem; font-weight:700; transition:transform .18s ease,background .18s ease; }
      [data-testid="stLinkButton"] a:hover { background:var(--teal); color:white; transform:scale(1.012); } [data-testid="stCaptionContainer"] { color:#8a98ab; font-size:.7rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
      .trust-row { display:grid; grid-template-columns:repeat(3,1fr); gap:1.3rem; margin-top:5rem; padding-top:1.65rem; border-top:1px solid var(--line); }.trust-item { color:var(--muted); font-size:.87rem; line-height:1.55; }.trust-item strong { display:block; margin-bottom:.34rem; color:var(--ink); font-size:.92rem; letter-spacing:-.018em; }
      .footer-line { display:flex; justify-content:space-between; gap:1rem; margin-top:4.5rem; color:#8a98ab; font-size:.75rem; }.footer-line a { color:var(--teal); text-decoration:none; }
      @media(max-width:760px) { [data-testid="stMainBlockContainer"] { padding:1.1rem 1.1rem 2rem; }.nav { margin-bottom:4rem; }.nav-note { display:none; }.hero { padding-bottom:4rem; }.hero h1 { font-size:clamp(3.1rem,16vw,4.5rem)!important; }.hero-copy { font-size:1.02rem; }.trust-row { grid-template-columns:1fr; gap:1.15rem; margin-top:3.7rem; }.footer-line { flex-direction:column; margin-top:3.4rem; } [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-height:auto; margin-bottom:.85rem; } }
      @media(prefers-reduced-motion:reduce) { * { transition:none!important; } }
    </style>
    <nav class="nav"><p class="wordmark"><span class="fish">Fish</span><span class="stop">Stop</span></p><p class="nav-note">Email security · desktop</p></nav>
    <section class="hero"><p class="eyebrow">Local-first email security</p><h1>Capire un’email<br>prima che conti <em><a class="hero-link" href="#download">troppo.</a></em></h1><p class="hero-copy">FishStop analizza file email, link e allegati sul tuo dispositivo. Una lettura chiara dei segnali che meritano attenzione, senza inviare il contenuto a un server.</p></section>
    <section id="download" class="download-section"><p class="download-kicker">Desktop edition</p><h3>Scarica FishStop.</h3></section>
    """,
    unsafe_allow_html=True,
)

release = latest_release()
version = (release.get("name") or release.get("tag_name")) if release else None
if release:
    st.markdown(f"<p class='release-status'>Versione più recente: <strong>{escape(str(version))}</strong></p>", unsafe_allow_html=True)
else:
    st.markdown("<p class='release-status'>La prima release pubblica sarà disponibile qui a breve.</p>", unsafe_allow_html=True)

mac_arm, mac_intel, windows = st.columns(3, gap="medium")
with mac_arm:
    download_card("⌘", "macOS · Apple Silicon", "Per Mac con chip Apple", "Compatibile con Mac M1, M2, M3, M4 e successivi.", installer_for(release, "macos", "apple-silicon"), version)
with mac_intel:
    download_card("⌘", "macOS · Intel", "Per Mac Intel", "Per i Mac con processore Intel.", installer_for(release, "macos", "intel"), version)
with windows:
    download_card("⊞", "Windows · 64 bit", "Per PC Windows", "Installer per computer Windows a 64 bit.", installer_for(release, "windows"), version)

st.markdown(
    f"""<section class="trust-row"><div class="trust-item"><strong>Privato per scelta</strong>L’analisi è eseguita sul dispositivo, non su una dashboard remota.</div><div class="trust-item"><strong>Pronto all’uso</strong>L’installer include il motore di analisi necessario a FishStop.</div><div class="trust-item"><strong>Per il triage quotidiano</strong>Leggi struttura, autenticazione, link e allegati in un unico report.</div></section><div class="footer-line"><span>FishStop · Progetto di tesi</span><a href="{RELEASES_URL}" target="_blank" rel="noopener">Tutte le release ↗</a></div>""",
    unsafe_allow_html=True,
)
