import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import streamlit.components.v1 as components
import torch

from src.analyzer.llm_context_analyzer import ENABLE_LOCAL_OLLAMA, stream_phi4_email_analysis
from src.components.email_globe import render_email_globe
from src.views.backend import get_backend


def _strip_encoded_content(raw: str) -> str:
    """
    Rimuove blocchi base64/quoted-printable corposi dal debugger raw EML.
    """
    lines = raw.splitlines(keepends=True)
    result = []
    i = 0
    b64_line = re.compile(r"^[A-Za-z0-9+/\r\n]+=*[\r\n]*$")
    qp_line = re.compile(r"=[0-9A-Fa-f]{2}|=$")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if len(stripped) >= 20 and b64_line.match(stripped + "\n"):
            start = i
            while i < len(lines) and b64_line.match(lines[i].strip() + "\n") and len(lines[i].strip()) >= 4:
                i += 1
            n = i - start
            if n >= 4:
                kb = sum(len(lines[j]) for j in range(start, i)) * 3 // 4 // 1024
                result.append(f"[... contenuto base64 rimosso ({n} righe, ~{kb} KB) ...]\n")
                continue
            result.extend(lines[start:i])
            continue

        if qp_line.search(stripped):
            start = i
            while i < len(lines) and qp_line.search(lines[i].strip()):
                i += 1
            n = i - start
            if n >= 4:
                result.append(f"[... contenuto quoted-printable rimosso ({n} righe) ...]\n")
                continue
            result.extend(lines[start:i])
            continue

        result.append(line)
        i += 1

    return "".join(result)


def _flag_counts(flags: list[dict]) -> dict:
    return {
        "HIGH": sum(1 for f in flags if f.get("level") == "HIGH"),
        "MEDIUM": sum(1 for f in flags if f.get("level") == "MEDIUM"),
        "LOW": sum(1 for f in flags if f.get("level") == "LOW"),
        "INFO": sum(1 for f in flags if f.get("level") == "INFO"),
    }


def _severity(counts: dict) -> tuple[str, str]:
    if counts["HIGH"]:
        return "CRITICAL", "Presenza di indicatori ad alta priorità"
    if counts["MEDIUM"]:
        return "SUSPICIOUS", "Indicatori da validare manualmente"
    if counts["LOW"]:
        return "WATCH", "Nessun blocco critico, ma ci sono note"
    return "LOW", "Nessun indicatore SOC rilevante"


def _render_flag(flag: dict):
    level = flag.get("level", "INFO")
    label = f"**[{level}] {flag.get('field', 'Signal')}** - {flag.get('message', '')}"
    if level == "HIGH":
        st.error(label)
    elif level == "MEDIUM":
        st.warning(label)
    elif level == "LOW":
        st.info(label)
    else:
        st.caption(label)


def _copyable_value(label: str, value: str | None, key: str):
    value = str(value or "-")
    js_value = json.dumps(value)
    element_id = f"copy_{re.sub(r'[^a-zA-Z0-9_]', '_', key)}"
    text_col, button_col = st.columns([0.86, 0.14], vertical_alignment="center")
    with text_col:
        st.write(f"**{label}:** `{value}`")
    with button_col:
        components.html(
            f"""
            <button id="{element_id}"
              style="width: 100%; padding: 6px 8px; border: 1px solid #d0d7de;
                     border-radius: 6px; background: white; cursor: pointer; font-size: 13px;">
              Copy
            </button>
            <script>
              const button_{element_id} = document.getElementById("{element_id}");
              button_{element_id}.onclick = async () => {{
                try {{
                  await navigator.clipboard.writeText({js_value});
                  button_{element_id}.innerText = "Copied";
                  setTimeout(() => button_{element_id}.innerText = "Copy", 1200);
                }} catch (err) {{
                  button_{element_id}.innerText = "Copy failed";
                  setTimeout(() => button_{element_id}.innerText = "Copy", 1200);
                }}
              }};
            </script>
            """,
            height=38,
        )


def _render_phi4_analysis(soc: dict, analysis_key: str, auto_run: bool = False):
    st.markdown("#### Phi-4 mini scam/phishing explanation")
    st.caption(
        "Analisi testuale locale con Ollama: valuta contenuto, urgenza, soldi, IBAN, "
        "pagamenti, credenziali e moduli esterni; poi usa SPF/DKIM/DMARC, link e allegati solo come contesto."
    )

    result_key = f"{analysis_key}_result"
    error_key = f"{analysis_key}_error"

    if st.session_state.get(result_key):
        st.success(st.session_state[result_key])
        return None

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
        return None

    if not ENABLE_LOCAL_OLLAMA:
        st.info("Analisi Phi-4 locale disattivata per la web app. La collegheremo a Hugging Face nel prossimo passaggio.")
        return None

    if auto_run:
        placeholder = st.empty()
        placeholder.markdown(_phi4_loading_html(), unsafe_allow_html=True)
        return placeholder

    st.info("L'analisi Phi-4 mini parte automaticamente nel riquadro Executive Triage.")
    return None


def _phi4_loading_html() -> str:
    return """
    <div style="
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 14px;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        background: #f6f8fa;
        color: #57606a;
        font-size: 0.95rem;
    ">
        <span>Phi-4 mini sta analizzando contenuto e indicatori tecnici</span>
        <span class="phi4-typing-dots" aria-label="caricamento">
            <span></span><span></span><span></span>
        </span>
    </div>
    <style>
        .phi4-typing-dots {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .phi4-typing-dots span {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #57606a;
            opacity: 0.35;
            animation: phi4TypingPulse 1.2s infinite ease-in-out;
        }
        .phi4-typing-dots span:nth-child(2) {
            animation-delay: 0.15s;
        }
        .phi4-typing-dots span:nth-child(3) {
            animation-delay: 0.3s;
        }
        @keyframes phi4TypingPulse {
            0%, 80%, 100% {
                transform: translateY(0);
                opacity: 0.35;
            }
            40% {
                transform: translateY(-3px);
                opacity: 1;
            }
        }
    </style>
    """


def _stream_phi4_analysis(soc: dict, analysis_key: str, placeholder):
    if placeholder is None:
        return

    result_key = f"{analysis_key}_result"
    error_key = f"{analysis_key}_error"
    if st.session_state.get(result_key) or st.session_state.get(error_key):
        return

    last_text = ""
    try:
        placeholder.markdown(_phi4_loading_html(), unsafe_allow_html=True)
        for event in stream_phi4_email_analysis(soc):
            if event.get("status") == "stream":
                last_text = event.get("text") or last_text
                placeholder.info(last_text)
            elif event.get("status") == "ok":
                last_text = event.get("text") or last_text
                st.session_state[result_key] = last_text or "Analisi completata senza testo."
                placeholder.success(st.session_state[result_key])
                return
            elif event.get("status") == "error":
                last_text = event.get("text") or last_text
                if last_text:
                    placeholder.warning(last_text)
                st.session_state[error_key] = event.get("message") or "Errore durante l'analisi Phi-4 mini."
                st.error(st.session_state[error_key])
                return
    except Exception as exc:
        st.session_state[error_key] = f"Errore durante l'analisi Phi-4 mini: {exc}"
        st.error(st.session_state[error_key])


def _render_abuseipdb(rep: dict):
    status = rep.get("status")
    if status == "ok":
        score = int(rep.get("abuseConfidenceScore") or 0)
        if rep.get("isWhitelisted"):
            st.success("Whitelisted - provider noto")
        elif score == 0:
            st.success("Score 0/100 - nessuna segnalazione")
        elif score < 25:
            st.info(f"Score {score}/100 - basso rischio")
        elif score < 75:
            st.warning(f"Score {score}/100 - rischio moderato")
        else:
            st.error(f"Score {score}/100 - alto rischio")

        c1, c2, c3 = st.columns(3)
        c1.metric("Segnalazioni", rep.get("totalReports", 0))
        c2.metric("Utenti", rep.get("numDistinctUsers", 0))
        c3.metric("Paese", rep.get("countryCode") or "-")
        if rep.get("isp"):
            st.caption(f"ISP: `{rep['isp']}`")
        if rep.get("url"):
            st.markdown(f"[Apri su AbuseIPDB]({rep['url']})")
    elif status == "skipped":
        st.info(rep.get("message", "Lookup saltato"))
    else:
        st.warning(rep.get("message", "Lookup non disponibile"))


def _render_geo(geo: dict):
    if geo.get("status") == "ok":
        parts = [geo.get("city"), geo.get("region"), geo.get("country")]
        location = ", ".join(p for p in parts if p) or "-"
        st.markdown(f"**Geo:** {location}")
        meta = []
        if geo.get("isp"):
            meta.append(f"ISP `{geo['isp']}`")
        if geo.get("asn"):
            meta.append(f"AS `{geo['asn']}`")
        if geo.get("is_proxy"):
            meta.append("proxy/VPN")
        if geo.get("is_hosting"):
            meta.append("hosting/datacenter")
        if meta:
            st.caption(" · ".join(meta))
    else:
        st.caption(f"Geo: {geo.get('message', 'non disponibile')}")


def _render_virustotal(vt: dict):
    status = vt.get("status")
    if status == "malicious":
        st.error(f"MALEVOLO - {vt.get('detection_ratio', '-')}")
    elif status == "suspicious":
        st.warning(f"SOSPETTO - {vt.get('detection_ratio', '-')}")
    elif status == "clean":
        st.success(f"PULITO - 0 / {vt.get('total_engines', 0)} engine")
    elif status == "not_found":
        st.info("Non trovato su VirusTotal")
    elif status == "skipped":
        st.info(vt.get("message", "Lookup saltato"))
        return
    else:
        st.warning(vt.get("message", "VirusTotal non disponibile"))
        return

    if vt.get("permalink"):
        st.markdown(f"[Apri report VirusTotal]({vt['permalink']})")


def _render_vt_url(rep: dict):
    status = rep.get("status", "error")
    message = rep.get("message", "")
    permalink = rep.get("permalink")

    if status == "malicious":
        st.error(f"VirusTotal: MALEVOLO - {rep.get('detection_ratio', '-')}")
    elif status == "suspicious":
        st.warning(f"VirusTotal: SOSPETTO - {rep.get('detection_ratio', '-')}")
    elif status == "clean":
        st.success(f"VirusTotal: pulito - {rep.get('detection_ratio', '-')}")
    elif status == "not_found":
        st.info("VirusTotal: URL non trovata")
    elif status == "skipped":
        st.info(f"VirusTotal: {message}")
        if permalink:
            st.markdown(f"[Apri VirusTotal]({permalink})")
        return
    else:
        st.warning(f"VirusTotal: {message}")
        if permalink:
            st.markdown(f"[Apri VirusTotal]({permalink})")
        return

    st.caption(
        f"Malicious `{rep.get('malicious', 0)}` · Suspicious `{rep.get('suspicious', 0)}` · "
        f"Harmless `{rep.get('harmless', 0)}` · Undetected `{rep.get('undetected', 0)}`"
    )
    if rep.get("last_analysis"):
        st.caption(f"Ultima analisi: `{rep['last_analysis']}`")
    if rep.get("final_url") and rep.get("final_url") != rep.get("url"):
        st.caption(f"Final URL: `{rep['final_url']}`")
    if permalink:
        st.markdown(f"[Apri scheda VirusTotal]({permalink})")


def _auth_status_box(title: str, status: str):
    status = (status or "unknown").lower()
    if status == "pass":
        st.success(f"{title}: PASS")
    elif status in ("fail", "softfail", "permerror"):
        st.error(f"{title}: {status.upper()}")
    elif status in ("none", "neutral", "temperror"):
        st.warning(f"{title}: {status.upper()}")
    else:
        st.info(f"{title}: {status.upper()}")


def _hop_from_label(hop: dict) -> str:
    if hop.get("from_host"):
        return hop["from_host"]
    raw = (hop.get("raw") or "").lstrip().lower()
    if raw.startswith("by "):
        return "internal/by-only"
    return "unknown"


def _hop_by_label(hop: dict) -> str:
    return hop.get("by_host") or "unknown"


def _status_from_received_spf(raw: str) -> str:
    if not raw:
        return "none"
    match = re.match(r"\s*([a-zA-Z0-9_-]+)", raw)
    return match.group(1).lower() if match else "unknown"


def _auth_from_eml_header(soc: dict, protocol: str) -> dict:
    protocol = protocol.upper()
    auth_results = soc.get("auth_results") or {}
    arc_auth_results = soc.get("arc_auth_results") or {}
    auth_raw = soc.get("authentication_results_raw") or ""
    arc_auth_raw = soc.get("arc_authentication_results") or ""

    if auth_results.get(protocol):
        result = auth_results[protocol]
        return {
            "status": result.get("status") or "unknown",
            "identity": result.get("identity") or "",
            "raw": result.get("raw") or auth_raw,
            "source": "Authentication-Results",
            "source_raw": auth_raw,
        }

    if arc_auth_results.get(protocol):
        result = arc_auth_results[protocol]
        return {
            "status": result.get("status") or "unknown",
            "identity": result.get("identity") or "",
            "raw": result.get("raw") or arc_auth_raw,
            "source": "ARC-Authentication-Results",
            "source_raw": arc_auth_raw,
        }

    if protocol == "SPF" and soc.get("received_spf_raw"):
        raw = soc.get("received_spf_raw") or ""
        return {
            "status": _status_from_received_spf(raw),
            "identity": "",
            "raw": raw,
            "source": "Received-SPF",
            "source_raw": raw,
        }

    if protocol == "DKIM" and soc.get("dkim_signature_raw"):
        raw = soc.get("dkim_signature_raw") or ""
        return {
            "status": "present",
            "identity": "",
            "raw": raw,
            "source": "DKIM-Signature",
            "source_raw": raw,
        }

    return {
        "status": "none",
        "identity": "",
        "raw": "",
        "source": "Header EML",
        "source_raw": "",
    }


def _email_auth_from_eml(soc: dict) -> dict:
    return {
        "spf": _auth_from_eml_header(soc, "SPF"),
        "dkim": _auth_from_eml_header(soc, "DKIM"),
        "dmarc": _auth_from_eml_header(soc, "DMARC"),
    }


def _render_auth_evidence(result: dict) -> None:
    st.caption(f"Fonte: `{result.get('source') or '-'}`")
    if result.get("identity"):
        st.caption(f"Identita: `{result['identity']}`")
    raw = result.get("raw") or ""
    st.caption("Stringa esaminata")
    if raw:
        st.code(raw, language="text")
    else:
        st.caption("Nessuna stringa trovata nell'EML per questo controllo.")


def _safe_vt_url_lookup(validator, url: str) -> dict:
    lookup = getattr(validator, "check_url_reputation", None)
    if callable(lookup):
        return lookup(url)
    return {"status": "skipped", "url": url, "message": "Validator VirusTotal URL non disponibile"}


def _summarize_link_reputation(results: dict) -> str:
    if not results:
        return "No links found."

    counts = {"malicious": 0, "suspicious": 0, "clean": 0, "not_found": 0, "skipped": 0, "error": 0}
    for rep in results.values():
        status = rep.get("status", "error")
        counts[status] = counts.get(status, 0) + 1

    parts = [f"{value} {key}" for key, value in counts.items() if value]
    worst = "clean"
    if counts.get("malicious"):
        worst = "malicious"
    elif counts.get("suspicious"):
        worst = "suspicious"
    elif counts.get("error") or counts.get("skipped"):
        worst = "unknown"

    return f"VirusTotal link reputation: worst={worst}; " + ", ".join(parts)


def render():
    parser, validator, analyzer, tokenizer, model, model_source = get_backend()

    st.title("FishStop SOC Console")
    st.caption("Email triage, authentication checks, threat intelligence e classificazione AI")

    col_upload, col_results = st.columns([0.9, 2.1], gap="large")

    with col_upload:
        st.subheader("Case Intake")
        uploaded_file = st.file_uploader("Carica un file `.eml`", type=["eml"])
        st.caption("Il file viene analizzato localmente e convertito in un report SOC.")

        if uploaded_file is not None:
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            if st.session_state.get("current_eml_name") != uploaded_file.name:
                st.session_state["raw_eml_debug_data"] = _strip_encoded_content(raw_text)
                st.session_state["current_eml_name"] = uploaded_file.name
                st.rerun()

            temp_path = os.path.join("data", "raw", "temp_triage.eml")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success("File pronto per il triage")
            st.metric("Nome file", uploaded_file.name)
            st.metric("Dimensione", f"{len(uploaded_file.getbuffer()) / 1024:.1f} KB")

    with col_results:
        if uploaded_file is None:
            st.info("Carica un `.eml` per aprire il caso di analisi.")
            return

        try:
            with st.spinner("Parsing EML e costruzione report SOC..."):
                soc = analyzer.analyze(temp_path)

            flags = soc.get("flags", [])
            counts = _flag_counts(flags)
            severity, severity_caption = _severity(counts)
            links = soc.get("links", [])
            attachments = soc.get("attachments", [])
            lookalike_alerts = soc.get("lookalike_alerts", [])
            eml_auth = _email_auth_from_eml(soc)
            unique_links = {lnk["url"]: lnk for lnk in links if lnk.get("url")}
            vt_url_results = {}
            if unique_links:
                with st.spinner("Lookup VirusTotal URL in corso..."):
                    max_workers = min(4, max(1, len(unique_links)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(_safe_vt_url_lookup, validator, url): url
                            for url in unique_links
                        }
                        for future, url in futures.items():
                            try:
                                vt_url_results[url] = future.result()
                            except Exception as exc:
                                vt_url_results[url] = {
                                    "status": "error",
                                    "url": url,
                                    "message": f"Errore lookup VirusTotal URL: {exc}",
                                }
            soc["link_reputation"] = vt_url_results
            soc["link_reputation_summary"] = _summarize_link_reputation(vt_url_results)

            st.subheader("Executive Triage")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Severity", severity)
            c2.metric("High", counts["HIGH"])
            c3.metric("Medium", counts["MEDIUM"])
            c4.metric("Link", len(links))
            c5.metric("Allegati", len(attachments))
            st.caption(severity_caption)

            phi4_key = f"phi4_analysis_{uploaded_file.name}_{len(uploaded_file.getbuffer())}"
            with st.container(border=True):
                phi4_placeholder = _render_phi4_analysis(soc, phi4_key, auto_run=True)

            if flags:
                with st.container(border=True):
                    st.markdown("#### Alert principali")
                    for flag in flags[:5]:
                        _render_flag(flag)
                    if len(flags) > 5:
                        st.caption(f"Altri {len(flags) - 5} indicatori disponibili nella tab Dettagli SOC.")

            overview, identity, auth, links_tab, attach_tab, content_tab, raw_tab = st.tabs(
                [
                    "Overview",
                    "Identità",
                    "Auth & Routing",
                    "Link Intel",
                    "Allegati",
                    "AI & Body",
                    "Raw",
                ]
            )

            with overview:
                left, right = st.columns([1, 1])
                with left:
                    st.markdown("#### Email Snapshot")
                    _copyable_value("From", soc.get("from_"), "overview_from")
                    _copyable_value("To", soc.get("to"), "overview_to")
                    _copyable_value("Subject", soc.get("subject"), "overview_subject")
                    st.write(f"**Date:** `{soc.get('date') or '-'}`")
                    _copyable_value("Message-ID", soc.get("message_id"), "overview_message_id")
                with right:
                    st.markdown("#### Signal Matrix")
                    _auth_status_box("SPF", eml_auth["spf"].get("status", "unknown"))
                    _auth_status_box("DKIM", eml_auth["dkim"].get("status", "unknown"))
                    _auth_status_box("DMARC", eml_auth["dmarc"].get("status", "unknown"))
                    if lookalike_alerts:
                        st.error(f"Lookalike domains: {len(lookalike_alerts)}")
                    else:
                        st.success("Lookalike domains: nessun match")

                st.markdown("#### Dettagli SOC")
                if flags:
                    for flag in flags:
                        _render_flag(flag)
                else:
                    st.success("Nessun flag SOC generato.")

            with identity:
                st.markdown("#### Envelope & Identity")
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Delivered-To:** `{soc.get('delivered_to') or '-'}`")
                    _copyable_value("Return-Path", soc.get("return_path"), "identity_return_path")
                    _copyable_value("Reply-To", soc.get("reply_to"), "identity_reply_to")
                    st.write(f"**Errors-To:** `{soc.get('errors_to') or '-'}`")
                with c2:
                    st.write(f"**Content-Type:** `{soc.get('content_type') or '-'}`")
                    st.write(f"**MIME-Version:** `{soc.get('mime_version') or '-'}`")
                    st.write(f"**Importance:** `{soc.get('importance') or '-'}`")

                if soc.get("reply_to_mismatch"):
                    st.error("Reply-To differisce dal From.")
                elif soc.get("reply_to"):
                    st.success("Reply-To coerente con From.")
                else:
                    st.info("Reply-To assente.")

                if soc.get("return_path_domain_mismatch"):
                    st.error(
                        f"Return-Path mismatch: `{soc.get('return_path_domain')}` differisce dal dominio From."
                    )
                elif soc.get("return_path"):
                    st.success("Return-Path coerente con il dominio From.")

                if soc.get("display_name_spoofing"):
                    st.error(f"Display Name Spoofing: `{soc['display_name_spoofing']}`")

                st.markdown("#### Reputazione domini mittente")
                domains = {}

                def pull_domain(raw: str | None) -> str:
                    if not raw:
                        return ""
                    m = re.search(r"@([\w.\-]+)", raw)
                    return m.group(1).lower() if m else ""

                from_domain = pull_domain(soc.get("from_"))
                rp_domain = pull_domain(soc.get("return_path"))
                rt_domain = pull_domain(soc.get("reply_to"))
                if from_domain:
                    domains[f"From ({from_domain})"] = from_domain
                if rp_domain and rp_domain != from_domain:
                    domains[f"Return-Path ({rp_domain})"] = rp_domain
                if rt_domain and rt_domain not in (from_domain, rp_domain):
                    domains[f"Reply-To ({rt_domain})"] = rt_domain

                if not domains:
                    st.info("Nessun dominio mittente estraibile.")
                else:
                    for label, domain in domains.items():
                        with st.expander(label):
                            with st.spinner(f"Reputazione dominio {domain}..."):
                                _render_abuseipdb(validator.check_domain_reputation(domain))

            with auth:
                st.markdown("#### Autenticazione")
                col_spf, col_dkim, col_dmarc = st.columns(3)
                with col_spf:
                    _auth_status_box("SPF", eml_auth["spf"].get("status", "unknown"))
                    _render_auth_evidence(eml_auth["spf"])
                with col_dkim:
                    _auth_status_box("DKIM", eml_auth["dkim"].get("status", "unknown"))
                    _render_auth_evidence(eml_auth["dkim"])
                with col_dmarc:
                    _auth_status_box("DMARC", eml_auth["dmarc"].get("status", "unknown"))
                    _render_auth_evidence(eml_auth["dmarc"])

                st.markdown("#### Routing")
                hops = soc.get("received_hops", [])
                c1, c2, c3 = st.columns(3)
                c1.metric("Hop Received", len(hops))
                c2.metric("Injection IP", soc.get("injection_sender_ip") or "-")
                c3.metric("Closest sender", (soc.get("closest_to_sender") or {}).get("from_host") or "-")

                with st.expander("Percorso geografico email", expanded=False):
                    render_email_globe(soc, validator)

                for idx, hop in enumerate(hops, start=1):
                    title = f"Hop {idx}: {_hop_from_label(hop)} -> {_hop_by_label(hop)}"
                    with st.expander(title):
                        st.write(f"**Sender IP:** `{hop.get('sender_ip') or '-'}`")
                        tls_version = hop.get("tls_version")
                        tls_cipher = hop.get("tls_cipher")
                        if tls_version or tls_cipher:
                            tls_label = " ".join(part for part in (tls_version, tls_cipher) if part)
                            st.write(f"**TLS:** `{tls_label}`")
                        all_ips = hop.get("all_ips") or ([hop["sender_ip"]] if hop.get("sender_ip") else [])
                        for ip in all_ips:
                            with st.container(border=True):
                                st.write(f"IP `{ip}`")
                                with st.spinner(f"Geolocalizzazione {ip}..."):
                                    _render_geo(validator.geolocate_ip(ip))
                                with st.expander("AbuseIPDB"):
                                    with st.spinner(f"Reputazione {ip}..."):
                                        _render_abuseipdb(validator.check_ip_reputation(ip))
                        with st.expander("Raw header"):
                            st.code(hop.get("raw", ""), language="text")

            with links_tab:
                st.markdown("#### Link Intelligence")
                if not links:
                    st.info("Nessun URL trovato nel corpo dell'email.")
                else:
                    st.caption("Ogni URL viene controllata su VirusTotal. Il risultato viene passato anche a Phi-4 mini.")
                    st.info(soc.get("link_reputation_summary") or "VirusTotal link reputation non disponibile.")

                    if lookalike_alerts:
                        st.markdown("##### Lookalike / Typosquatting")
                        for alert in lookalike_alerts:
                            st.error(
                                f"`{alert['host']}` assomiglia a `{alert['matched_brand']}` - {alert['detail']}"
                            )

                    st.markdown("##### URL estratte")
                    for lnk in links:
                        rep = vt_url_results.get(lnk["url"], {})
                        risky = rep.get("status") in ("malicious", "suspicious")
                        with st.container(border=True):
                            top_left, top_right = st.columns([3, 1])
                            with top_left:
                                st.markdown(f"**`{lnk.get('host') or '-'}`**")
                                st.caption(f"`{lnk.get('url')}`")
                            with top_right:
                                if lnk.get("is_ip"):
                                    st.error("IP diretto")
                                elif risky:
                                    st.warning(rep.get("status", "suspicious"))
                                else:
                                    st.success("checked")
                            _render_vt_url(rep)
                            st.markdown(
                                f"[VirusTotal](https://www.virustotal.com/gui/domain/{lnk['host']})"
                                f" · [WHOIS](https://www.whois.com/whois/{lnk['host']})"
                            )

            with attach_tab:
                st.markdown("#### Allegati")
                if not attachments:
                    st.info("Nessun allegato rilevato.")
                for att in attachments:
                    with st.container(border=True):
                        st.markdown(f"##### `{att.get('filename') or '(senza nome)'}`")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Content-Type", att.get("content_type") or "-")
                        c2.metric("Encoding", att.get("encoding") or "-")
                        c3.metric("Estensione", att.get("extension_from_filename") or "-")
                        c4.metric("Magic", att.get("magic_detected_format") or "-")

                        if att.get("anomaly"):
                            st.error(att["anomaly"])
                        elif att.get("extension_match") is True:
                            st.success("Estensione, Content-Type e magic bytes coerenti.")
                        else:
                            st.warning("Coerenza file non determinabile.")

                        if att.get("hash_sha256"):
                            with st.expander("Hash e VirusTotal"):
                                st.code(att["hash_sha256"], language="text")
                                with st.spinner("Lookup VirusTotal allegato..."):
                                    _render_virustotal(validator.check_file_hash(att["hash_sha256"]))

            with content_tab:
                st.markdown("#### Analisi AI del contenuto")
                clean_body = soc.get("body_ai") or soc.get("body_clean") or soc.get("body") or ""
                email_text = f"Subject: {soc.get('subject') or ''}\n\n{clean_body}".strip()

                _render_phi4_analysis(soc, phi4_key, auto_run=False)

                if model_source == "company":
                    st.success("Modello aziendale attivo.")
                else:
                    st.info("Modello base attivo.")

                if not email_text or email_text.lower() == "subject:":
                    st.warning("Email senza testo significativo per la classificazione.")
                else:
                    with st.spinner("BERT sta analizzando il contenuto..."):
                        inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probabilities = torch.softmax(logits, dim=1).flatten().tolist()
                    prob_safe = probabilities[0] * 100
                    prob_phishing = probabilities[1] * 100
                    c1, c2 = st.columns(2)
                    c1.metric("Legittima", f"{prob_safe:.2f}%")
                    c2.metric("Phishing", f"{prob_phishing:.2f}%")
                    if prob_phishing > prob_safe:
                        st.error("Risultato IA: possibile phishing")
                    else:
                        st.success("Risultato IA: email probabilmente legittima")
                    with st.expander("Logit grezzi"):
                        st.json({"logits": logits.flatten().tolist()})

                st.markdown("#### Corpo estratto")
                source = soc.get("body_source", "unknown")
                st.caption(f"Sorgente: `{source}`")
                ai_context = soc.get("body_context", "normal")
                body_display = soc.get("body_extracted") or soc.get("body_ai") or soc.get("body_clean") or soc.get("body") or ""
                full_body = soc.get("body_clean_full") or soc.get("body_clean") or soc.get("body") or ""
                if ai_context == "forwarded":
                    st.info("Email inoltrata: viene mostrato e analizzato il contenuto inoltrato.")
                elif ai_context == "reply":
                    st.info("Risposta email: viene mostrata e analizzata solo la risposta corrente.")
                if soc.get("html_strip_applied"):
                    clean_tab, full_tab, html_tab = st.tabs(["Body estratto", "Conversazione completa", "HTML grezzo"])
                    with clean_tab:
                        st.text_area("Body", body_display, height=280)
                    with full_tab:
                        st.text_area("Body completo", full_body, height=280)
                    with html_tab:
                        st.code(soc.get("body_html") or "", language="html")
                else:
                    body_tab, full_tab = st.tabs(["Body estratto", "Conversazione completa"])
                    with body_tab:
                        st.text_area("Body", body_display, height=280)
                    with full_tab:
                        st.text_area("Body completo", full_body, height=280)

            with raw_tab:
                st.markdown("#### Report strutturato")
                report_copy = {k: v for k, v in soc.items() if k != "raw_eml_bytes"}
                st.json(report_copy, expanded=False)
                st.markdown("#### EML raw pulito")
                st.text_area(
                    "Raw EML",
                    st.session_state.get("raw_eml_debug_data", ""),
                    height=480,
                    disabled=True,
                )

            _stream_phi4_analysis(soc, phi4_key, phi4_placeholder)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as exc:
            st.error(f"Si è verificato un errore durante l'analisi: {exc}")
            with st.expander("Dettaglio errore"):
                st.exception(exc)
