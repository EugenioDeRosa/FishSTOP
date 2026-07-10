import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape as html_escape

import streamlit as st
import streamlit.components.v1 as components
 
from src.analyzer.html_utils import sanitize_html_for_js_preview, sanitize_html_for_preview
from src.analyzer.llm_context_analyzer import stream_phi4_email_analysis
from src.components.email_globe import render_email_globe
from src.views.backend import get_content_model, get_core_backend


def _strip_encoded_content(raw: str) -> str:
    """
    Rimuove blocchi base64/quoted-printable corposi dal raw EML.
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
                result.append(f"[... base64 content removed ({n} rows, ~{kb} KB) ...]\n")
                continue
            result.extend(lines[start:i])
            continue

        if qp_line.search(stripped):
            start = i
            while i < len(lines) and qp_line.search(lines[i].strip()):
                i += 1
            n = i - start
            if n >= 4:
                result.append(f"[... quoted-printable content removed ({n} rows) ...]\n")
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
        return "CRITICAL", "High-priority indicators found"
    if counts["MEDIUM"]:
        return "SUSPICIOUS", "Indicators require manual validation"
    if counts["LOW"]:
        return "WATCH", "No critical blockers, but notes are present"
    return "LOW", "No relevant SOC indicator"


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


def _confirm_copyable_link(url: str, key: str) -> None:
    url = str(url or "")
    if not url:
        return

    js_value = json.dumps(url)
    element_id = f"copy_link_{re.sub(r'[^a-zA-Z0-9_]', '_', key)}"
    components.html(
        f"""
        <div style="display: flex; gap: 8px; align-items: stretch; width: 100%;">
          <code style="flex: 1; display: block; overflow-wrap: anywhere; padding: 8px 10px;
                       border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa;
                       color: #24292f; font-size: 12px; line-height: 1.35;">{html_escape(url)}</code>
          <button id="{element_id}"
            style="min-width: 86px; padding: 6px 10px; border: 1px solid #d0d7de;
                   border-radius: 6px; background: white; cursor: pointer; font-size: 13px;">
            Copy
          </button>
        </div>
        <script>
          const button_{element_id} = document.getElementById("{element_id}");
          button_{element_id}.onclick = async () => {{
            const confirmed = window.confirm(
              "This link comes from a potentially dangerous email. Do you really want to copy it?"
            );
            if (!confirmed) return;
            try {{
              await navigator.clipboard.writeText({js_value});
              button_{element_id}.innerText = "Copyto";
              setTimeout(() => button_{element_id}.innerText = "Copy", 1200);
            }} catch (err) {{
              button_{element_id}.innerText = "Error";
              setTimeout(() => button_{element_id}.innerText = "Copy", 1200);
            }}
          }};
        </script>
        """,
        height=46,
    )


def _render_extracted_links_box(links: list[dict], key_prefix: str) -> None:
    unique_links = []
    seen = set()
    for link in links:
        url = link.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique_links.append(link)

    with st.container(border=True):
        st.markdown("#### Links found in the email")
        st.caption("La preview HTML non contiene link cliccabili. Copy un URL solo se serve per analisi in ambiente sicuro.")
        if not unique_links:
            st.info("No links found in the email.")
            return

        for idx, link in enumerate(unique_links, start=1):
            host = link.get("host") or "-"
            source = link.get("source") or "-"
            st.caption(f"{idx}. Host: `{host}` · Source: `{source}`")
            _confirm_copyable_link(link.get("url", ""), f"{key_prefix}_{idx}")


def _render_phi4_analysis(soc: dict, analysis_key: str, auto_run: bool = False):
    st.markdown("#### Phi-4 mini scam/phishing explanation")
    st.caption(
        "Hosted Phi-4 mini analysis: evaluates plain and HTML content, urgency, money, IBANs, "
        "payments, credentials, and external forms; then uses semantic analysis, SPF/DKIM/DMARC, links, and attachments only as context."
    )

    result_key = f"{analysis_key}_result"
    error_key = f"{analysis_key}_error"

    if st.session_state.get(result_key):
        st.success(st.session_state[result_key])
        return None

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
        return None

    if auto_run:
        placeholder = st.empty()
        placeholder.markdown(_phi4_loading_html(), unsafe_allow_html=True)
        return placeholder

    st.info("Phi-4 mini analysis starts automatically in the Executive Triage panel.")
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
        <span>Phi-4 mini is analyzing content and technical indicators</span>
        <span class="phi4-typing-dots" aria-label="loading">
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
                st.session_state[result_key] = last_text or "Analysis completed without text."
                placeholder.success(st.session_state[result_key])
                return
            elif event.get("status") == "error":
                last_text = event.get("text") or last_text
                if last_text:
                    placeholder.warning(last_text)
                st.session_state[error_key] = event.get("message") or "Error durante l'analisi Phi-4 mini."
                st.error(st.session_state[error_key])
                return
    except Exception as exc:
        st.session_state[error_key] = f"Error durante l'analisi Phi-4 mini: {exc}"
        st.error(st.session_state[error_key])


def _render_abuseipdb(rep: dict):
    status = rep.get("status")
    if status == "ok":
        score = int(rep.get("abuseConfidenceScore") or 0)
        if rep.get("isWhitelisted"):
            st.success("Whitelisted - known provider")
        elif score == 0:
            st.success("Score 0/100 - no reports")
        elif score < 25:
            st.info(f"Score {score}/100 - low risk")
        elif score < 75:
            st.warning(f"Score {score}/100 - moderate risk")
        else:
            st.error(f"Score {score}/100 - high risk")

        c1, c2, c3 = st.columns(3)
        c1.metric("Reports", rep.get("totalReports", 0))
        c2.metric("Users", rep.get("numDistinctUsers", 0))
        c3.metric("Country", rep.get("countryCode") or "-")
        if rep.get("isp"):
            st.caption(f"ISP: `{rep['isp']}`")
        if rep.get("url"):
            st.markdown(f"[Open on AbuseIPDB]({rep['url']})")
    elif status == "skipped":
        st.info(rep.get("message", "Lookup skipped"))
    else:
        st.warning(rep.get("message", "Lookup unavailable"))


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
        st.caption(f"Geo: {geo.get('message', 'unavailable')}")


def _render_virustotal(vt: dict):
    status = vt.get("status")
    if status == "malicious":
        st.error(f"MALICIOUS - {vt.get('detection_ratio', '-')}")
    elif status == "suspicious":
        st.warning(f"SUSPICIOUS - {vt.get('detection_ratio', '-')}")
    elif status == "clean":
        st.success(f"CLEAN - 0 / {vt.get('total_engines', 0)} engine")
    elif status == "not_found":
        st.info("Not found on VirusTotal")
    elif status == "skipped":
        st.info(vt.get("message", "Lookup skipped"))
        return
    else:
        st.warning(vt.get("message", "VirusTotal unavailable"))
        return

    if vt.get("permalink"):
        st.markdown(f"[Apri report VirusTotal]({vt['permalink']})")


def _render_vt_url(rep: dict):
    status = rep.get("status", "error")
    message = rep.get("message", "")
    permalink = rep.get("permalink")

    if status == "malicious":
        st.error(f"VirusTotal: MALICIOUS - {rep.get('detection_ratio', '-')}")
    elif status == "suspicious":
        st.warning(f"VirusTotal: SUSPICIOUS - {rep.get('detection_ratio', '-')}")
    elif status == "clean":
        st.success(f"VirusTotal: clean - {rep.get('detection_ratio', '-')}")
    elif status == "not_found":
        st.info("VirusTotal: URL not found")
    elif status == "skipped":
        st.info(f"VirusTotal: {message}")
        if permalink:
            st.markdown(f"[Open VirusTotal]({permalink})")
        return
    else:
        st.warning(f"VirusTotal: {message}")
        if permalink:
            st.markdown(f"[Open VirusTotal]({permalink})")
        return

    st.caption(
        f"Malicious `{rep.get('malicious', 0)}` · Suspicious `{rep.get('suspicious', 0)}` · "
        f"Harmless `{rep.get('harmless', 0)}` · Undetected `{rep.get('undetected', 0)}`"
    )
    if rep.get("last_analysis"):
        st.caption(f"Last analysis: `{rep['last_analysis']}`")
    context_items = rep.get("crowdsourced_context") or []
    if context_items:
        with st.expander("VirusTotal crowdsourced context", expanded=False):
            st.caption(rep.get("crowdsourced_context_summary") or "Additional context reported by VirusTotal users/sources.")
            for item in context_items:
                label = f"[{item.get('severity', 'INFO')}] {item.get('title') or 'Context'}"
                if item.get("source"):
                    label += f" - source: {item['source']}"
                if item.get("date"):
                    label += f" - {item['date']}"
                st.write(label)
                for detail in item.get("details") or []:
                    st.caption(detail)
    if permalink:
        st.markdown(f"[Open VirusTotal page]({permalink})")


def _auth_status_box(title: str, status: str, show_help: bool = True):
    status = (status or "unknown").lower()
    help_texts = {
        "SPF": "SPF indicates which IPs are authorized to send email for the sender domain.",
        "DKIM": "DKIM verifies message integrity through a cryptographic domain signature.",
        "DMARC": "DMARC indicates what the receiving mailbox should do when SPF or DKIM fails.",
    }
    if status == "pass":
        bg, border, color = "#ecfdf3", "#abefc6", "#067647"
    elif status in ("fail", "softfail", "permerror"):
        bg, border, color = "#fef3f2", "#fecdca", "#b42318"
    elif status in ("none", "neutral", "temperror"):
        bg, border, color = "#fffaeb", "#fedf89", "#b54708"
    else:
        bg, border, color = "#eff8ff", "#b2ddff", "#175cd3"

    label = f"{title}: {status.upper()}"
    help_text = html_escape(help_texts.get(title.upper(), "Email authentication check."), quote=True)
    help_icon = (
        f'''
        <span class="auth-help" tabindex="0" aria-label="{help_text}">
            <span class="auth-help__icon">?</span>
            <span class="auth-help__card">{help_text}</span>
        </span>
        '''
        if show_help
        else ""
    )
    help_style = """
        <style>
        .auth-help {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            outline: none;
        }
        .auth-help__icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border: 1px solid currentColor;
            border-radius: 50%;
            font-size: 12px;
            font-weight: 700;
            cursor: default;
            opacity: .85;
            line-height: 1;
        }
        .auth-help__card {
            position: absolute;
            top: calc(100% + 6px);
            right: 0;
            width: 240px;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid rgba(0, 0, 0, 0.12);
            background: rgba(255, 255, 255, 0.98);
            color: #1f2937;
            font-size: 12px;
            line-height: 1.35;
            font-weight: 500;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
            opacity: 0;
            visibility: hidden;
            transform: translateY(-4px);
            transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
            z-index: 10;
            pointer-events: none;
        }
        .auth-help:hover .auth-help__card,
        .auth-help:focus-within .auth-help__card {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        </style>
    """
    st.markdown(
        help_style
        + f"""
        <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; padding:10px 12px; margin:0 0 8px 0; border:1px solid {border}; border-radius:8px; background:{bg}; color:{color}; font-weight:600;">
            <span>{html_escape(label)}</span>
            {help_icon}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    st.caption(f"Source: `{result.get('source') or '-'}`")
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
    return {"status": "skipped", "url": url, "message": "Validator VirusTotal URL unavailable"}


def _summarize_link_reputation(results: dict) -> str:
    if not results:
        return "No links found."

    counts = {"malicious": 0, "suspicious": 0, "clean": 0, "not_found": 0, "skipped": 0, "error": 0}
    for rep in results.values():
        status = rep.get("status", "error")
        counts[status] = counts.get(status, 0) + 1

    context_count = sum(len(rep.get("crowdsourced_context") or []) for rep in results.values())
    parts = [f"{value} {key}" for key, value in counts.items() if value]
    if context_count:
        parts.append(f"{context_count} crowdsourced context item(s)")
    worst = "clean"
    if counts.get("malicious"):
        worst = "malicious"
    elif counts.get("suspicious"):
        worst = "suspicious"
    elif counts.get("error") or counts.get("skipped"):
        worst = "unknown"

    return f"VirusTotal link reputation: worst={worst}; " + ", ".join(parts)


def _render_html_preview(raw_html: str, key: str, height: int = 360, enable_javascript: bool = False) -> None:
    preview_html = sanitize_html_for_js_preview(raw_html) if enable_javascript else sanitize_html_for_preview(raw_html)
    components.html(
        preview_html,
        height=height,
        scrolling=True,
    )


def render():
    parser, validator, analyzer = get_core_backend()

    st.title("FishStop SOC Console")
    st.caption("Email triage, authentication checks, threat intelligence e classificazione AI")

    col_upload, col_results = st.columns([0.9, 2.1], gap="large")

    with col_upload:
        st.subheader("Case Intake")
        uploaded_file = st.file_uploader("Upload an `.eml` file", type=["eml"])
        st.caption("Il file viene analizzato localmente e convertito in un report SOC.")

        if uploaded_file is not None:
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
            if st.session_state.get("current_eml_name") != uploaded_file.name:
                st.session_state["raw_eml_text"] = raw_text
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
            st.info("Upload an `.eml` to open the analysis case.")
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
                with st.spinner("Running VirusTotal URL lookup..."):
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
                                    "message": f"Error lookup VirusTotal URL: {exc}",
                                }
            soc["link_reputation"] = vt_url_results
            soc["link_reputation_summary"] = _summarize_link_reputation(vt_url_results)

            st.subheader("Executive Triage")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Severity", severity)
            c2.metric("High", counts["HIGH"])
            c3.metric("Medium", counts["MEDIUM"])
            c4.metric("Link", len(links))
            c5.metric("Attachments", len(attachments))
            st.caption(severity_caption)

            phi4_key = f"phi4_analysis_v9_{uploaded_file.name}_{len(uploaded_file.getbuffer())}"
            with st.container(border=True):
                phi4_placeholder = _render_phi4_analysis(soc, phi4_key, auto_run=True)

            if flags:
                with st.container(border=True):
                    st.markdown("#### Main alerts")
                    for flag in flags[:5]:
                        _render_flag(flag)
                    if len(flags) > 5:
                        st.caption(f"Altri {len(flags) - 5} indicators are available in the SOC Details tab.")

            overview, identity, auth, links_tab, attach_tab, content_tab, raw_tab = st.tabs(
                [
                    "Overview",
                    "Identity",
                    "Auth & Routing",
                    "Link Intel",
                    "Attachments",
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
                    _auth_status_box("SPF", eml_auth["spf"].get("status", "unknown"), show_help=False)
                    _auth_status_box("DKIM", eml_auth["dkim"].get("status", "unknown"), show_help=False)
                    _auth_status_box("DMARC", eml_auth["dmarc"].get("status", "unknown"), show_help=False)
                    if lookalike_alerts:
                        st.error(f"Lookalike domains: {len(lookalike_alerts)}")
                    else:
                        st.success("Lookalike domains: no match")

                st.markdown("#### SOC Details")
                if flags:
                    for flag in flags:
                        _render_flag(flag)
                else:
                    st.success("No SOC flags generated.")

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
                    st.error("Reply-To differs from From.")
                elif soc.get("reply_to"):
                    st.success("Reply-To is consistent with From.")
                else:
                    st.info("Reply-To absent.")

                if soc.get("return_path_domain_mismatch"):
                    st.error(
                        f"Return-Path mismatch: `{soc.get('return_path_domain')}` differs from the From domain."
                    )
                elif soc.get("return_path"):
                    st.success("Return-Path is consistent with the From domain.")

                if soc.get("display_name_spoofing"):
                    st.error(f"Display Name Spoofing: `{soc['display_name_spoofing']}`")

                st.markdown("#### Sender domain reputation")
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
                    st.info("No sender domain could be extracted.")
                else:
                    for label, domain in domains.items():
                        with st.expander(label):
                            with st.spinner(f"Domain reputation {domain}..."):
                                _render_abuseipdb(validator.check_domain_reputation(domain))

            with auth:
                st.markdown("#### Authentication")
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
                routing_hops = list(reversed(hops))

                c1, c2, c3 = st.columns(3)
                c1.metric("Received hops", len(hops))
                c2.metric("Injection IP", soc.get("injection_sender_ip") or "-")
                c3.metric("Closest sender", (soc.get("closest_to_sender") or {}).get("from_host") or "-")

                with st.expander("Email geographic route", expanded=False):
                    render_email_globe(soc, validator)

                for idx, hop in enumerate(routing_hops, start=1):
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
                                with st.spinner(f"Geolocation {ip}..."):
                                    _render_geo(validator.geolocate_ip(ip))
                                with st.expander("AbuseIPDB"):
                                    with st.spinner(f"Reputation {ip}..."):
                                        _render_abuseipdb(validator.check_ip_reputation(ip))
                        with st.expander("Raw header"):
                            st.code(hop.get("raw", ""), language="text")

            with links_tab:
                st.markdown("#### Link Intelligence")
                if not links:
                    st.info("No URL found in the email body.")
                else:
                    st.caption("Each URL is checked on VirusTotal. The result is also passed to Phi-4 mini.")
                    st.info(soc.get("link_reputation_summary") or "VirusTotal link reputation unavailable.")

                    if lookalike_alerts:
                        st.markdown("##### Lookalike / Typosquatting")
                        for alert in lookalike_alerts:
                            matched_brand = alert.get("matched_brand") or "-"
                            if matched_brand == "-":
                                st.error(f"`{alert['host']}` - {alert['detail']}")
                            else:
                                st.error(
                                    f"`{alert['host']}` looks like `{matched_brand}` - {alert['detail']}"
                                )

                    st.markdown("##### Extracted URLs")
                    for lnk in links:
                        rep = vt_url_results.get(lnk["url"], {})
                        risky = rep.get("status") in ("malicious", "suspicious")
                        display_mismatch = lnk.get("display_mismatch")
                        possible_shortener = lnk.get("is_possible_shortener")
                        with st.container(border=True):
                            top_left, top_right = st.columns([3, 1])
                            with top_left:
                                st.markdown(f"**`{lnk.get('host') or '-'}`**")
                                st.caption(f"`{lnk.get('url')}`")
                                if lnk.get("display_host"):
                                    st.caption(f"Visible text: `{lnk.get('display_host')}`")
                            with top_right:
                                if lnk.get("is_ip"):
                                    st.error("Direct IP")
                                elif display_mismatch:
                                    st.error("different text")
                                elif possible_shortener:
                                    st.warning("short link")
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
                st.markdown("#### Attachments")
                if not attachments:
                    st.info("No attachments detected.")
                for att in attachments:
                    with st.container(border=True):
                        st.markdown(f"##### `{att.get('filename') or '(unnamed)'}`")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Content-Type", att.get("content_type") or "-")
                        c2.metric("Encoding", att.get("encoding") or "-")
                        c3.metric("Extension", att.get("extension_from_filename") or "-")
                        c4.metric("Magic", att.get("magic_detected_format") or "-")

                        if att.get("anomaly"):
                            st.error(att["anomaly"])
                        elif att.get("extension_match") is True:
                            st.success("Extension, Content-Type e magic bytes coerenti.")
                        else:
                            st.warning("File consistency cannot be determined.")

                        if att.get("hash_sha256"):
                            with st.expander("Hash and VirusTotal"):
                                st.code(att["hash_sha256"], language="text")
                                with st.spinner("Running VirusTotal attachment lookup..."):
                                    _render_virustotal(validator.check_file_hash(att["hash_sha256"]))

            with content_tab:
                import torch
                st.markdown("#### AI Content Analysis")
                clean_body = soc.get("body_ai") or soc.get("body_clean") or soc.get("body") or ""
                email_text = f"Subject: {soc.get('subject') or ''}\n\n{clean_body}".strip()

                _render_phi4_analysis(soc, phi4_key, auto_run=False)

                with st.spinner("Loading BERT model..."):
                    tokenizer, model, model_source = get_content_model()

                st.info("BERT model loaded from Hugging Face.")

                if not email_text or email_text.lower() == "subject:":
                    soc["bert_ai_result"] = "not available"
                    st.warning("Email has no meaningful text for classification.")
                else:
                    with st.spinner("BERT is analyzing the content..."):
                        inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probabilities = torch.softmax(logits, dim=1).flatten().tolist()
                    prob_safe = probabilities[0] * 100
                    prob_phishing = probabilities[1] * 100
                    c1, c2 = st.columns(2)
                    c1.metric("Legitimate", f"{prob_safe:.2f}%")
                    c2.metric("Phishing", f"{prob_phishing:.2f}%")
                    if prob_phishing > prob_safe:
                        soc["bert_ai_result"] = "phishing"
                        st.error("AI result: possible phishing")
                    else:
                        soc["bert_ai_result"] = "legitimate"
                        st.success("AI result: email probably legitimate")
                    with st.expander("Raw logits"):
                        st.json({"logits": logits.flatten().tolist()})

                st.markdown("#### Extracted Body")
                source = soc.get("body_source", "unknown")
                st.caption(f"Source: `{source}`")
                ai_context = soc.get("body_context", "normal")
                body_display = soc.get("body_extracted") or soc.get("body_ai") or soc.get("body_clean") or soc.get("body") or ""
                full_body = soc.get("body_clean_full") or soc.get("body_clean") or soc.get("body") or ""
                body_html = soc.get("body_html") or ""
                if ai_context == "forwarded":
                    st.info("Forwarded email: the forwarded content is shown and analyzed.")
                elif ai_context == "reply":
                    st.info("Email reply: only the current reply is shown and analyzed.")

                tab_labels = ["Extracted body", "Full conversation"]
                if body_html:
                    tab_labels.extend(["Rendered HTML", "HTML raw"])

                body_tabs = st.tabs(tab_labels)
                with body_tabs[0]:
                    if body_display:
                        st.text_area("Text used for AI and triage", body_display, height=300, disabled=True)
                    else:
                        st.warning("No text could be extracted from the message body.")
                with body_tabs[1]:
                    if full_body:
                        st.text_area("Full normalized text", full_body, height=300, disabled=True)
                    else:
                        st.info("No full conversation available.")

                next_tab = 2
                if body_html:
                    with body_tabs[next_tab]:
                        enable_html_javascript = st.checkbox(
                            "Render HTML with JavaScript",
                            value=False,
                            key=f"{phi4_key}_body_html_javascript",
                        )
                        if enable_html_javascript:
                            st.caption("HTML preview with JavaScript enabled. Links remain disabled and are not clickable.")
                        else:
                            st.caption("Isolated HTML preview: scripts, forms, active content, and clickable links are removed before display.")
                        _render_html_preview(body_html, f"{phi4_key}_body_html", enable_javascript=enable_html_javascript)
                    next_tab += 1
                    with body_tabs[next_tab]:
                        st.text_area(
                            "HTML raw",
                            body_html,
                            height=360,
                            disabled=True,
                        )
                    next_tab += 1

            with raw_tab:
                st.markdown("#### Structured report")
                report_copy = {k: v for k, v in soc.items() if k != "raw_eml_bytes"}
                st.json(report_copy, expanded=False)
                st.markdown("#### Cleaned raw EML")
                hide_encoded_content = st.checkbox(
                    "Remove base64/quoted-printable content",
                    value=False,
                    key=f"{phi4_key}_strip_encoded_raw",
                )
                raw_eml_text = st.session_state.get("raw_eml_text", "")
                raw_eml_display = _strip_encoded_content(raw_eml_text) if hide_encoded_content else raw_eml_text
                st.text_area(
                    "Cleaned raw EML",
                    raw_eml_display,
                    height=480,
                    disabled=True,
                )

            _stream_phi4_analysis(soc, phi4_key, phi4_placeholder)

            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as exc:
            st.error(f"An error occurred during analysis: {exc}")
            with st.expander("Error details"):
                st.exception(exc)
