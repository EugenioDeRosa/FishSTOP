"""Public Streamlit landing page for FishStop desktop installers."""

from __future__ import annotations

from html import escape
from typing import Any

import requests
import streamlit as st


DESKTOP_REPOSITORY = "EugenioDeRosa/fishstop-desktop-email-security"
RELEASES_URL = f"https://github.com/{DESKTOP_REPOSITORY}/releases"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{DESKTOP_REPOSITORY}/releases/latest"
SITE_URL = "https://fishstop-eml.streamlit.app/"
PRIVACY_URL = f"{SITE_URL}?page=privacy"
TERMS_URL = f"{SITE_URL}?page=terms"
SUPPORT_EMAIL = "info.fishstop@gmail.com"


def render_legal_page(page: str) -> None:
    is_privacy = page == "privacy"
    title = "Privacy Policy" if is_privacy else "Terms of Service"
    eyebrow = "DATA & PRIVACY" if is_privacy else "TERMS OF USE"
    intro = (
        "How FishStop handles account information, email content and locally stored analysis data."
        if is_privacy
        else "The conditions that apply when downloading and using the FishStop desktop application."
    )
    if is_privacy:
        body = f"""
        <section class="legal-section"><h2>1. Who is responsible</h2><p>FishStop is a desktop email-security project maintained by Eugenio De Rosa. Privacy questions can be sent to <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>.</p></section>
        <section class="legal-section"><h2>2. Information FishStop accesses</h2><p>Signing in with Google or Microsoft provides the account identifier, name, email address and, when available, profile picture needed to create the local workspace. Connecting Gmail or Outlook is a separate, optional authorization.</p><p>For Gmail, FishStop requests the read-only <code>gmail.readonly</code> permission. It lists metadata and previews for up to ten recent Inbox messages. The complete MIME email is retrieved only after the user explicitly selects <strong>Analyse</strong>. FishStop cannot send, delete, move or modify mailbox messages.</p></section>
        <aside class="local-boundary"><span>LOCAL PROCESSING BOUNDARY</span><strong>Email bodies, attachments and AI analysis remain on the user's device.</strong><p>FishStop does not operate a remote analysis server and does not use Google user data to train shared AI models or for advertising.</p></aside>
        <section class="legal-section"><h2>3. How information is used</h2><p>Email data is used only to perform the security analysis requested by the user: parsing message structure and authentication headers, examining links and attachments, detecting identity inconsistencies, and producing a local risk assessment.</p><p>FishStop's use and transfer to any other app of information received from Google APIs adheres to the <a href="https://developers.google.com/terms/api-services-user-data-policy" target="_blank" rel="noopener">Google API Services User Data Policy</a>, including the Limited Use requirements.</p></section>
        <section class="legal-section"><h2>4. Storage and retention</h2><p>The mailbox refresh token is stored in the operating system's secure credential store. Analysis history and preferences are stored on the user's device and separated by account. A selected email may be written to a temporary local file while it is analysed and is removed when that processing step finishes.</p><p>FishStop does not maintain a central copy of mailbox messages or analysis history.</p></section>
        <section class="legal-section"><h2>5. Optional external intelligence</h2><p>If the user configures third-party reputation services, FishStop may submit technical indicators such as URLs, domains, IP addresses or file hashes to services such as VirusTotal or AbuseIPDB. The complete email, its body and attachments are not submitted by FishStop to those services. Their own privacy policies apply.</p><p>AlienVault OTX intelligence is downloaded into a local cache and matched locally. Public reference services used for brand identification may receive a brand name or domain, not the complete message.</p></section>
        <section class="legal-section"><h2>6. Sharing and sale</h2><p>FishStop does not sell Google user data, mailbox content or analysis history. It does not share that information for advertising, credit decisions or user profiling. Information is disclosed only when initiated by an optional user-configured reputation lookup, required by law, or necessary to protect users and the integrity of the software.</p></section>
        <section class="legal-section"><h2>7. User controls and deletion</h2><p>The user can select <strong>Disconnect</strong> in FishStop to remove the locally stored mailbox authorization, clear local analysis history from the application, revoke FishStop from the Google Account permissions page, or uninstall the application to remove its local components. Because FishStop has no central mailbox database, support cannot remotely erase files that exist only on the user's device.</p></section>
        <section class="legal-section"><h2>8. Security and changes</h2><p>FishStop uses OAuth Authorization Code with PKCE, loopback callbacks and the operating system credential store. No system can guarantee absolute security. This policy may be updated when features or legal requirements change; the revision date below will be updated accordingly.</p></section>
        """
    else:
        body = f"""
        <section class="legal-section"><h2>1. Acceptance</h2><p>By downloading or using FishStop, the user agrees to these Terms of Service and the <a href="{PRIVACY_URL}">Privacy Policy</a>. If the user does not agree, they should not connect an account or use the application.</p></section>
        <section class="legal-section"><h2>2. Purpose of the service</h2><p>FishStop is a defensive email-analysis application designed to help users inspect suspicious messages, links, attachments and authentication evidence. It provides decision support and does not guarantee that every malicious email will be detected or that every warning represents an actual attack.</p></section>
        <aside class="local-boundary"><span>USER-CONTROLLED ANALYSIS</span><strong>The user chooses which email is analysed and remains responsible for the final decision.</strong><p>Do not open, execute, reply to or pay based solely on an automated result when material risk is involved.</p></aside>
        <section class="legal-section"><h2>3. Account and mailbox access</h2><p>The user may connect only accounts they own or are authorized to access. Gmail and Outlook connections are read-only. The user is responsible for protecting their device and account and for revoking access if the device is lost or shared.</p></section>
        <section class="legal-section"><h2>4. Acceptable use</h2><p>FishStop may not be used to access another person's mailbox without authorization, evade provider safeguards, distribute malware or spam, violate privacy rights, or perform unlawful surveillance. Users must comply with applicable laws and third-party service terms.</p></section>
        <section class="legal-section"><h2>5. Third-party services</h2><p>Google, Microsoft, GitHub, VirusTotal, AbuseIPDB, AlienVault OTX and other optional sources operate under their own terms. FishStop does not control their availability, results or policies. An unavailable provider may limit a feature without affecting local analysis.</p></section>
        <section class="legal-section"><h2>6. Availability and updates</h2><p>FishStop is provided as a thesis and software-development project. Features may change, be suspended or be discontinued. Updates may be necessary for compatibility, security or provider-policy changes.</p></section>
        <section class="legal-section"><h2>7. Disclaimer and liability</h2><p>To the maximum extent permitted by applicable law, FishStop is provided without warranties of uninterrupted availability, complete detection or fitness for a specific purpose. The maintainer is not liable for decisions, payments, data loss or security incidents caused by relying exclusively on an automated assessment.</p></section>
        <section class="legal-section"><h2>8. Contact and changes</h2><p>Questions may be sent to <a href="mailto:{SUPPORT_EMAIL}">{SUPPORT_EMAIL}</a>. Updated terms become effective when published on this page with a new revision date.</p></section>
        """
    st.markdown(
        f"""
        <nav class="nav legal-nav"><a class="wordmark" href="{SITE_URL}"><span class="fish">Fish</span><span class="stop">Stop</span></a><div><a href="{PRIVACY_URL}" class="{'active' if is_privacy else ''}">Privacy</a><a href="{TERMS_URL}" class="{'active' if not is_privacy else ''}">Terms</a></div></nav>
        <main class="legal-shell"><header class="legal-hero"><p class="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{intro}</p><small>Last updated: 8 September 2026</small></header>{body}</main>
        <footer class="legal-footer"><span>FishStop · Desktop email security</span><a href="{SITE_URL}">Return to download page →</a></footer>
        """,
        unsafe_allow_html=True,
    )


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
      html,body,[class*="css"] { font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Inter",sans-serif; }
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
      .hero h1 em { color:var(--teal); font-style:normal; } .hero-copy { max-width:630px; margin:1.7rem 0 0; color:var(--muted); font-size:1.15rem; line-height:1.65; letter-spacing:-.012em; }
      .download-section { border-top:1px solid var(--line); padding-top:2.15rem; } .download-kicker { margin:0 0 .45rem; color:var(--blue); font-size:.73rem; font-weight:760; letter-spacing:.13em; text-transform:uppercase; }
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
      .footer-links { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:1rem; }
      .legal-nav { margin-bottom:clamp(3.5rem,7vw,6rem); }.legal-nav a { text-decoration:none; }.legal-nav>div { display:flex; gap:.35rem; }.legal-nav>div a { padding:.48rem .72rem; border-radius:999px; color:var(--muted); font-size:.78rem; font-weight:690; }.legal-nav>div a:hover,.legal-nav>div a.active { color:var(--ink); background:white; box-shadow:inset 0 0 0 1px var(--line); }
      .legal-shell { width:min(760px,100%); margin:0 auto; }.legal-hero { margin-bottom:3.6rem; padding-bottom:2.3rem; border-bottom:1px solid var(--line); }.legal-hero h1 { margin:0; color:var(--ink); font-size:clamp(3rem,7vw,5.4rem)!important; line-height:.93; letter-spacing:-.075em; }.legal-hero>p:not(.eyebrow) { max-width:620px; margin:1.35rem 0 .8rem; color:var(--muted); font-size:1.08rem; line-height:1.65; }.legal-hero small { color:#8a98ab; font-size:.76rem; }
      .legal-section { margin:0 0 2.6rem; }.legal-section h2 { margin:0 0 .75rem!important; color:var(--ink); font-size:1.18rem!important; letter-spacing:-.028em; }.legal-section p { margin:.55rem 0; color:var(--muted); font-size:.94rem; line-height:1.75; }.legal-section a,.legal-footer a { color:var(--teal); text-underline-offset:3px; }.legal-section code { padding:.14rem .34rem; border-radius:5px; color:#0b6f66; background:#e7f5f1; font-size:.82rem; }
      .local-boundary { position:relative; margin:3.3rem 0; padding:1.55rem 1.65rem 1.45rem; overflow:hidden; border:1px solid #cde8e1; border-radius:20px; background:linear-gradient(145deg,#eefaf6,#fff); }.local-boundary::after { content:""; position:absolute; width:130px; height:130px; right:-55px; top:-62px; border:24px solid rgba(15,159,143,.07); border-radius:50%; }.local-boundary span,.local-boundary strong,.local-boundary p { position:relative; z-index:1; display:block; }.local-boundary span { color:var(--teal); font-size:.67rem; font-weight:800; letter-spacing:.13em; }.local-boundary strong { max-width:570px; margin:.65rem 0; color:var(--ink); font-size:1.12rem; line-height:1.4; }.local-boundary p { max-width:600px; margin:0; color:var(--muted); font-size:.86rem; line-height:1.6; }
      .legal-footer { width:min(760px,100%); display:flex; justify-content:space-between; gap:1rem; margin:4rem auto 0; padding-top:1.5rem; border-top:1px solid var(--line); color:#8a98ab; font-size:.76rem; }.legal-footer a { text-decoration:none; font-weight:700; }
      @media(max-width:760px) { [data-testid="stMainBlockContainer"] { padding:1.1rem 1.1rem 2rem; }.nav { margin-bottom:4rem; }.nav-note { display:none; }.hero { padding-bottom:4rem; }.hero h1 { font-size:clamp(3.1rem,16vw,4.5rem)!important; }.hero-copy { font-size:1.02rem; }.trust-row { grid-template-columns:1fr; gap:1.15rem; margin-top:3.7rem; }.footer-line,.legal-footer { flex-direction:column; margin-top:3.4rem; }.footer-links { justify-content:flex-start; }.legal-nav { align-items:flex-start; gap:1rem; margin-bottom:3.6rem; }.legal-hero { margin-bottom:2.6rem; }.local-boundary { margin:2.7rem 0; padding:1.3rem; } [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-height:auto; margin-bottom:.85rem; } }
      @media(prefers-reduced-motion:reduce) { * { transition:none!important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

requested_page = str(st.query_params.get("page", "home")).strip().lower()
if requested_page in {"privacy", "terms"}:
    render_legal_page(requested_page)
    st.stop()

st.markdown(
    """
    <nav class="nav"><p class="wordmark"><span class="fish">Fish</span><span class="stop">Stop</span></p><p class="nav-note">Email security · desktop</p></nav>
    <section class="hero"><p class="eyebrow">Local-first email security</p><h1>Capire un’email<br>prima che conti <em>troppo.</em></h1><p class="hero-copy">FishStop analizza file email, link e allegati sul tuo dispositivo. Una lettura chiara dei segnali che meritano attenzione, senza inviare il contenuto a un server.</p></section>
    <section class="download-section"><p class="download-kicker">Desktop edition</p><h3>Scarica FishStop.</h3></section>
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
    f"""<section class="trust-row"><div class="trust-item"><strong>Privato per scelta</strong>L’analisi è eseguita sul dispositivo, non su una dashboard remota.</div><div class="trust-item"><strong>Pronto all’uso</strong>L’installer include il motore di analisi necessario a FishStop.</div><div class="trust-item"><strong>Per il triage quotidiano</strong>Leggi struttura, autenticazione, link e allegati in un unico report.</div></section><div class="footer-line"><span>FishStop · Progetto di tesi</span><span class="footer-links"><a href="{PRIVACY_URL}">Privacy</a><a href="{TERMS_URL}">Terms</a><a href="mailto:{SUPPORT_EMAIL}">Support</a><a href="{RELEASES_URL}" target="_blank" rel="noopener">Tutte le release ↗</a></span></div>""",
    unsafe_allow_html=True,
)
