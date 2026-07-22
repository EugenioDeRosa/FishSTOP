import json
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape as html_escape
from urllib.parse import urlsplit, urlunsplit

import streamlit as st
import streamlit.components.v1 as components
 
from src.analyzer.html_utils import sanitize_html_for_js_preview, sanitize_html_for_preview
from src.analyzer.llm_context_analyzer import (
    PROMPT_VERSION,
    active_llm_backend,
    format_email_risk_analysis,
    stream_phi4_email_analysis,
)
from src.analyzer.received_parser import order_received_hops
from src.bert_calibration import calibrated_probabilities, classify as classify_bert_result
from src.bert_inference import predict_email_logits
from src.bert_input import prepare_bert_input
from src.components.email_globe import render_email_globe
from src.ui import page_intro, risk_banner
from src.views.backend import get_calibration, get_content_model, get_core_backend


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


def _attachment_anomaly_without_pdf_risk(anomaly: str | None) -> str:
    parts = [
        part.strip()
        for part in str(anomaly or "").split(";")
        if part.strip() and not part.strip().startswith("PDF risk ")
    ]
    return "; ".join(parts)


def _pdf_indicator_lines(pdf_security: dict) -> list[str]:
    return [
        f"{item.get('label') or item.get('key') or 'indicator'} x{item.get('count') or 1}"
        for item in (pdf_security.get("indicators") or [])
    ]


def _pdf_status_text(pdf_security: dict) -> str:
    risk_level = str(pdf_security.get("risk_level") or "unknown").upper()
    return f"PDF risk: {risk_level}"


def _main_alert_flags(flags: list[dict], limit: int = 5) -> list[dict]:
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    pdf_priority_fields = {"PDF Content", "PDF Attachment"}
    return sorted(
        [flag for flag in flags if flag.get("level", "INFO") != "INFO"],
        key=lambda flag: (
            severity_rank.get(flag.get("level", "INFO"), 9),
            0 if flag.get("field") in pdf_priority_fields else 1,
            flag.get("field", ""),
        ),
    )[:limit]


def _is_meaningful_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        compact = re.sub(r"\s+", "", normalized)
        return bool(normalized) and compact not in {"-", "--", "<>", "null", "none", "n/a", "unknown"}
    return True


def _field_value(label: str, value: str | None) -> bool:
    if not _is_meaningful_value(value):
        return False
    value = str(value).strip()
    st.write(f"**{label}:** `{value}`")
    return True


def _report_table(fields: list[tuple[str, object]]) -> None:
    rows = []
    for label, value in fields:
        if not _is_meaningful_value(value):
            continue
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        rows.append({"Field": label, "Value": value})
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("No data available in this section.")


def _render_structured_report(
    soc: dict,
    eml_auth: dict,
    flags: list[dict],
    links: list[dict],
    attachments: list[dict],
) -> None:
    """Render the machine report as readable analyst sections, with JSON as a fallback."""
    report_copy = {key: value for key, value in soc.items() if key != "raw_eml_bytes"}
    report_json = json.dumps(report_copy, indent=2, ensure_ascii=False, default=str)
    actionable_flags = [flag for flag in flags if flag.get("level", "INFO") != "INFO"]

    message_tab, security_tab, evidence_tab, routing_tab, full_tab = st.tabs(
        ["Message", "Security", "Evidence", "Routing", "Full report"]
    )

    with message_tab:
        _report_table(
            [
                ("From", soc.get("from_")),
                ("To", soc.get("to")),
                ("Subject", soc.get("subject")),
                ("Date", soc.get("date")),
                ("Reply-To", soc.get("reply_to")),
                ("Return-Path", soc.get("return_path")),
                ("Message-ID", soc.get("message_id")),
                ("Body source", soc.get("body_source")),
            ]
        )

    with security_tab:
        counts = _flag_counts(actionable_flags)
        c1, c2, c3 = st.columns(3)
        c1.metric("High", counts["HIGH"])
        c2.metric("Medium", counts["MEDIUM"])
        c3.metric("Low", counts["LOW"])
        _report_table(
            [
                ("SPF", eml_auth.get("spf", {}).get("status")),
                ("DKIM", eml_auth.get("dkim", {}).get("status")),
                ("DMARC", eml_auth.get("dmarc", {}).get("status")),
                ("Injection sender IP", soc.get("injection_sender_ip")),
                ("Return-Path mismatch", soc.get("return_path_domain_mismatch")),
                ("Reply-To mismatch", soc.get("reply_to_mismatch")),
                ("Display-name spoofing", soc.get("display_name_spoofing")),
            ]
        )
        if actionable_flags:
            st.markdown("##### Findings")
            for flag in actionable_flags:
                _render_flag(flag)

    with evidence_tab:
        c1, c2 = st.columns(2)
        c1.metric("Links", len(links))
        c2.metric("Attachments", len(attachments))
        if links:
            st.markdown("##### Links")
            st.dataframe(
                [
                    {
                        "Host": link.get("host") or "-",
                        "URL": _defang_url(link.get("url") or ""),
                        "Source": link.get("source") or "-",
                    }
                    for link in links
                ],
                hide_index=True,
                width="stretch",
            )
        if attachments:
            st.markdown("##### Attachments")
            st.dataframe(
                [
                    {
                        "File": item.get("filename") or "(unnamed)",
                        "Type": item.get("content_type") or "-",
                        "Format": item.get("magic_detected_format") or item.get("extension_from_filename") or "-",
                        "SHA-256": item.get("hash_sha256") or "-",
                    }
                    for item in attachments
                ],
                hide_index=True,
                width="stretch",
            )
        if not links and not attachments:
            st.caption("No links or attachments were extracted.")

    with routing_tab:
        hops = order_received_hops(soc.get("received_hops", []))
        if hops:
            st.dataframe(
                [
                    {
                        "Hop": index,
                        "From": _hop_from_label(hop),
                        "By": _hop_by_label(hop),
                        "Sender IP": hop.get("sender_ip") or "-",
                        "Timestamp": hop.get("received_at") or "-",
                        "TLS": " ".join(
                            part for part in (hop.get("tls_version"), hop.get("tls_cipher")) if part
                        ) or "-",
                    }
                    for index, hop in enumerate(hops, start=1)
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No routing hops were extracted.")

    with full_tab:
        st.download_button(
            "Download JSON report",
            data=report_json,
            file_name="fishstop_report.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("View complete JSON", expanded=False):
            st.code(report_json, language="json")


def _defang_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except Exception:
        return value.replace("://", "[://]").replace(".", "[.]")

    if not parsed.scheme or not parsed.netloc:
        return value.replace("://", "[://]").replace(".", "[.]")

    host = parsed.netloc.replace(".", "[.]")
    path = urlunsplit(("", "", parsed.path, parsed.query, parsed.fragment))
    return f"{parsed.scheme}[://]{host}{path}"


def _render_ioc_values(label: str, values: list[str], key_prefix: str) -> None:
    st.markdown(f"##### {label}")
    if not values:
        st.info("No indicators in this category.")
        return

    for idx, value in enumerate(values, start=1):
        value = str(value or "").strip()
        if not value:
            continue
        element_id = f"ioc_copy_{re.sub(r'[^a-zA-Z0-9_]', '_', key_prefix)}_{idx}"
        js_value = json.dumps(value)
        escaped_value = html_escape(value)
        components.html(
            f'''
            <div style="display:flex; align-items:center; gap:8px; width:100%; margin:0 0 8px 0;">
              <code style="flex:1; display:block; min-width:0; overflow-x:auto; white-space:nowrap; padding:9px 10px;
                           border:1px solid #d0d7de; border-radius:6px; background:#f6f8fa;
                           color:#24292f; font-size:12px; line-height:1.35; font-weight:600; user-select:text;">
                {escaped_value}
              </code>
              <button id="{element_id}" title="Copy indicator" aria-label="Copy indicator"
                style="width:40px; min-width:40px; border:1px solid #d0d7de; border-radius:6px;
                       background:#ffffff; color:#080341; cursor:pointer; display:flex;
                       align-items:center; justify-content:center; padding:0;">
                <svg width="19" height="19" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <path fill-rule="evenodd" clip-rule="evenodd" d="M19.5 16.5L19.5 4.5L18.75 3.75H9L8.25 4.5L8.25 7.5L5.25 7.5L4.5 8.25V20.25L5.25 21H15L15.75 20.25V17.25H18.75L19.5 16.5ZM15.75 15.75L15.75 8.25L15 7.5L9.75 7.5V5.25L18 5.25V15.75H15.75ZM6 9L14.25 9L14.25 19.5L6 19.5L6 9Z" fill="currentColor"/>
                </svg>
              </button>
            </div>
            <script>
              const button_{element_id} = document.getElementById("{element_id}");
              button_{element_id}.onclick = async () => {{
                try {{
                  await navigator.clipboard.writeText({js_value});
                  button_{element_id}.style.background = "#ecfdf3";
                  button_{element_id}.style.color = "#067647";
                  setTimeout(() => {{
                    button_{element_id}.style.background = "#ffffff";
                    button_{element_id}.style.color = "#080341";
                  }}, 900);
                }} catch (err) {{
                  button_{element_id}.style.background = "#fef3f2";
                  button_{element_id}.style.color = "#b42318";
                  setTimeout(() => {{
                    button_{element_id}.style.background = "#ffffff";
                    button_{element_id}.style.color = "#080341";
                  }}, 900);
                }}
              }};
            </script>
            ''',
            height=48,
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
        st.caption("La preview HTML non contiene link cliccabili. Gli indicatori copiabili sono disponibili nella tab IoC.")
        if not unique_links:
            st.info("No links found in the email.")
            return

        for idx, link in enumerate(unique_links, start=1):
            host = link.get("host") or "-"
            source = link.get("source") or "-"
            st.caption(f"{idx}. Host: `{host}` · Source: `{source}`")
            st.code(link.get("url", ""), language="text")


def _render_phi4_analysis(soc: dict, analysis_key: str, auto_run: bool = False):
    st.markdown("#### Phi-4 mini scam/phishing explanation")
    st.caption(
        f"Phi-4 mini analysis via {active_llm_backend()}: evaluates plain and HTML content, urgency, money, IBANs, "
        "payments, credentials, and external forms; then uses semantic analysis, SPF/DKIM/DMARC, links, and attachments only as context."
    )

    result_key = f"{analysis_key}_result"
    error_key = f"{analysis_key}_error"

    if st.session_state.get(result_key):
        _show_phi4_result(st, st.session_state[result_key])
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


def _show_phi4_result(target, result):
    if not isinstance(result, dict):
        target.info(str(result))
        return

    verdict = str(result.get("final_verdict") or "review").lower()
    message = format_email_risk_analysis(result)
    if verdict == "phishing":
        target.error(message)
    elif verdict == "review":
        target.warning(message)
    else:
        target.success(message)


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
                # The streamed model output is JSON and is intentionally hidden
                # until it has been parsed and validated by the application.
                last_text = event.get("text") or last_text
            elif event.get("status") == "ok":
                analysis = event.get("analysis")
                if not isinstance(analysis, dict):
                    raise ValueError("Structured Phi-4 analysis is missing")
                st.session_state[result_key] = analysis
                _show_phi4_result(placeholder, analysis)
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
    effective_auth_results = soc.get("effective_auth_results") or {}
    auth_results = soc.get("auth_results") or {}
    arc_auth_results = soc.get("arc_auth_results") or {}
    auth_raw = soc.get("authentication_results_raw") or ""
    arc_auth_raw = soc.get("arc_authentication_results") or ""

    if effective_auth_results.get(protocol):
        result = effective_auth_results[protocol]
        source = result.get("source") or "Authentication headers"
        if source == "ARC-Authentication-Results":
            source_raw = arc_auth_raw
        elif source == "Received-SPF":
            source_raw = soc.get("received_spf_raw") or ""
        else:
            source_raw = auth_raw
        return {
            "status": result.get("status") or "unknown",
            "identity": result.get("identity") or "",
            "raw": result.get("raw") or source_raw,
            "source": source,
            "source_raw": source_raw,
            "all_results": result.get("all_results") or [],
        }

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
        return ""

    counts = {"malicious": 0, "suspicious": 0, "clean": 0, "not_found": 0, "skipped": 0, "error": 0, "unknown": 0}
    for rep in results.values():
        status = rep.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1

    context_count = sum(len(rep.get("crowdsourced_context") or []) for rep in results.values())
    positive_evidence = counts.get("malicious", 0) + counts.get("suspicious", 0)
    if positive_evidence == 0 and context_count == 0:
        return ""

    parts = [
        f"{counts['malicious']} malicious" if counts.get("malicious") else "",
        f"{counts['suspicious']} suspicious" if counts.get("suspicious") else "",
        f"{context_count} crowdsourced context item(s)" if context_count else "",
    ]
    breakdown = ", ".join(part for part in parts if part)
    risk_label = "High risk" if counts.get("malicious") else "Manual review"
    return f"VirusTotal link intelligence: {risk_label} - {breakdown}."


def _render_html_preview(raw_html: str, key: str, height: int = 360, enable_javascript: bool = False) -> None:
    preview_html = sanitize_html_for_js_preview(raw_html) if enable_javascript else sanitize_html_for_preview(raw_html)
    components.html(
        preview_html,
        height=height,
        scrolling=True,
    )


def render():
    parser, validator, analyzer = get_core_backend()

    page_intro(
        "Email analysis",
        "Investigate a suspicious email",
        "Upload an .eml file to review its risk, sender authenticity, links, and attachments in one place.",
    )

    col_upload = st.container(border=True)
    col_results = st.container()

    with col_upload:
        st.markdown('<div class="fs-section-label">Add email file</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an .eml file", type=["eml"], label_visibility="collapsed")
        st.caption("Your file is parsed locally. Only indicators are sent to reputation services when configured.")

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            current_eml_hash = hashlib.sha256(raw_bytes).hexdigest()
            if st.session_state.get("current_eml_hash") != current_eml_hash:
                st.session_state["raw_eml_text"] = raw_text
                st.session_state["current_eml_name"] = uploaded_file.name
                st.session_state["current_eml_hash"] = current_eml_hash
                st.rerun()

            temp_path = os.path.join("data", "raw", "temp_triage.eml")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            file_col, size_col = st.columns([2, 1])
            file_col.metric("File", uploaded_file.name)
            size_col.metric("Size", f"{len(uploaded_file.getbuffer()) / 1024:.1f} KB")

    with col_results:
        if uploaded_file is None:
            st.info("Upload an email to begin. The analysis will appear here automatically.")
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

            st.markdown("### Analysis summary")
            risk_banner(severity, severity_caption)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Critical signals", counts["HIGH"])
            c2.metric("Review signals", counts["MEDIUM"])
            c3.metric("Links", len(links))
            c4.metric("Attachments", len(attachments))

            eml_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            phi4_key = f"phi4_analysis_{PROMPT_VERSION}_{eml_digest}"
            with st.container(border=True):
                phi4_placeholder = _render_phi4_analysis(soc, phi4_key, auto_run=True)

            actionable_flags = [flag for flag in flags if flag.get("level", "INFO") != "INFO"]
            main_flags = _main_alert_flags(flags)
            if main_flags:
                with st.container(border=True):
                    st.markdown("#### Priority findings")
                    for flag in main_flags:
                        _render_flag(flag)
                    if len(actionable_flags) > len(main_flags):
                        st.caption(f"{len(actionable_flags) - len(main_flags)} more findings are available in the Summary tab.")

            overview, identity, auth, links_tab, attach_tab, content_tab, ioc_tab, raw_tab = st.tabs(
                [
                    "Summary",
                    "Sender",
                    "Authentication",
                    "Links",
                    "Files",
                    "Content",
                    "Indicators",
                    "Technical",
                ]
            )

            with overview:
                left, right = st.columns([1, 1])
                with left:
                    st.markdown("#### Message")
                    _field_value("From", soc.get("from_"))
                    _field_value("To", soc.get("to"))
                    _field_value("Subject", soc.get("subject"))
                    st.write(f"**Date:** `{soc.get('date') or '-'}`")
                with right:
                    st.markdown("#### Trust checks")
                    _auth_status_box("SPF", eml_auth["spf"].get("status", "unknown"), show_help=False)
                    _auth_status_box("DKIM", eml_auth["dkim"].get("status", "unknown"), show_help=False)
                    _auth_status_box("DMARC", eml_auth["dmarc"].get("status", "unknown"), show_help=False)
                    if lookalike_alerts:
                        st.error(f"Lookalike domains: {len(lookalike_alerts)}")
                    else:
                        st.success("Lookalike domains: no match")

                st.markdown("#### All findings")
                if actionable_flags:
                    for flag in actionable_flags:
                        _render_flag(flag)
                else:
                    st.success("No security findings require attention.")

            with identity:
                st.markdown("#### Sender identity")
                sender_identity_fields = [
                    ("Delivered-To", soc.get("delivered_to")),
                    ("Return-Path", soc.get("return_path")),
                    ("Reply-To", soc.get("reply_to")),
                    ("Errors-To", soc.get("errors_to")),
                    ("Importance", soc.get("importance")),
                ]
                if any(_is_meaningful_value(value) for _, value in sender_identity_fields):
                    c1, c2 = st.columns(2)
                    with c1:
                        for label, value in sender_identity_fields[:4]:
                            _field_value(label, value)
                    with c2:
                        _field_value(*sender_identity_fields[4])
                else:
                    st.info("No additional sender identity fields were found in this email.")

                if soc.get("reply_to_mismatch"):
                    st.error("Reply-To differs from From.")

                if soc.get("return_path_domain_mismatch"):
                    st.error(
                        f"Return-Path mismatch: `{soc.get('return_path_domain')}` differs from the From domain."
                    )

                if soc.get("display_name_spoofing"):
                    st.error(f"Display Name Spoofing: `{soc['display_name_spoofing']}`")

                st.markdown("#### Domain reputation")
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
                routing_hops = order_received_hops(hops)

                c1, c2, c3 = st.columns(3)
                c1.metric("Received hops", len(hops))
                c2.metric("Injection IP", soc.get("injection_sender_ip") or "-")
                c3.metric("Closest sender", (soc.get("closest_to_sender") or {}).get("from_host") or "-")

                with st.expander("Email geographic route", expanded=False):
                    render_email_globe(soc, validator)

                for idx, hop in enumerate(routing_hops, start=1):
                    title = f"Hop {idx}: {_hop_from_label(hop)} -> {_hop_by_label(hop)}"
                    with st.expander(title):
                        if hop.get("received_at"):
                            st.caption(f"Received at: `{hop['received_at']}`")
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
                st.markdown("#### Link intelligence")
                if not links:
                    st.info("No URL found in the email body.")
                else:
                    st.caption("Each URL is checked on VirusTotal. Only malicious or suspicious results are passed to Phi-4 mini.")
                    if soc.get("link_reputation_summary"):
                        st.warning(soc["link_reputation_summary"])

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
                                elif risky:
                                    st.warning(rep.get("status", "suspicious"))
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

                        non_pdf_anomaly = _attachment_anomaly_without_pdf_risk(att.get("anomaly"))
                        if non_pdf_anomaly:
                            st.error(non_pdf_anomaly)
                        elif att.get("extension_match") is True:
                            st.success("Extension, Content-Type e magic bytes coerenti.")
                        else:
                            st.warning("File consistency cannot be determined.")

                        pdf_security = att.get("pdf_security") or {}
                        if pdf_security:
                            if pdf_security.get("suspicious"):
                                st.error(_pdf_status_text(pdf_security))
                            else:
                                st.info(_pdf_status_text(pdf_security))
                            behavior_lines = [
                                f"{item.get('label') or item.get('key') or 'behavior'} x{item.get('count') or 1}"
                                for item in (pdf_security.get("behaviors") or [])
                            ]
                            if behavior_lines:
                                st.caption("PDF malicious behaviors: " + "; ".join(behavior_lines[:6]))
                            indicator_lines = _pdf_indicator_lines(pdf_security)
                            if indicator_lines:
                                st.caption("PDF indicators: " + "; ".join(indicator_lines[:6]))
                                if len(indicator_lines) > 6:
                                    with st.expander(f"Show {len(indicator_lines) - 6} more PDF indicators"):
                                        for line in indicator_lines[6:]:
                                            st.write(f"- {line}")

                        if att.get("hash_sha256"):
                            with st.expander("Hash and VirusTotal"):
                                st.code(att["hash_sha256"], language="text")
                                with st.spinner("Running VirusTotal attachment lookup..."):
                                    _render_virustotal(validator.check_file_hash(att["hash_sha256"]))


            with ioc_tab:
                st.markdown("#### Indicators of compromise")
                st.caption("Copy or export the values you need for blocking, monitoring, or threat hunting.")

                def unique_values(values):
                    result = []
                    seen = set()
                    for value in values:
                        value = str(value or "").strip()
                        if not value or value == "-" or value in seen:
                            continue
                        seen.add(value)
                        result.append(value)
                    return result

                url_iocs = unique_values(_defang_url(link.get("url")) for link in links)
                domain_iocs = unique_values(
                    [link.get("host") for link in links]
                    + [alert.get("host") for alert in lookalike_alerts]
                    + [soc.get("return_path_domain")]
                )
                ip_iocs = unique_values(
                    [soc.get("injection_sender_ip")]
                    + [ip for hop in soc.get("received_hops", []) for ip in (hop.get("all_ips") or [])]
                )
                hash_iocs = unique_values(att.get("hash_sha256") for att in attachments)
                sender_iocs = unique_values([soc.get("from_"), soc.get("reply_to"), soc.get("return_path")])

                ioc_groups = {
                    "URLs": url_iocs,
                    "Domains": domain_iocs,
                    "IPs": ip_iocs,
                    "SHA-256 hashes": hash_iocs,
                    "Senders": sender_iocs,
                }

                all_iocs_text = "\n".join(
                    line
                    for label, values in ioc_groups.items()
                    for line in ([f"# {label}"] + values + [""] if values else [])
                ).strip()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("URLs", len(url_iocs))
                c2.metric("Domains", len(domain_iocs))
                c3.metric("IPs", len(ip_iocs))
                c4.metric("Hashes", len(hash_iocs))
                c5.metric("Senders", len(sender_iocs))

                if all_iocs_text:
                    st.download_button(
                        "Download IoC list",
                        data=all_iocs_text,
                        file_name="fishstop_iocs.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    st.info("No IoC extracted from this email.")

                col_left, col_right = st.columns(2)
                for idx, (label, values) in enumerate(ioc_groups.items()):
                    target = col_left if idx % 2 == 0 else col_right
                    with target:
                        key_label = label.lower().replace(" ", "_").replace("-", "_")
                        _render_ioc_values(label, values, f"{phi4_key}_ioc_{key_label}")

            with content_tab:
                st.markdown("#### Content assessment")
                clean_body = soc.get("body_for_ai") or soc.get("body_ai") or soc.get("body_clean") or ""
                email_text = prepare_bert_input(soc.get("subject") or "", clean_body)

                _render_phi4_analysis(soc, phi4_key, auto_run=False)

                with st.spinner("Loading DistilBERT model..."):
                    tokenizer, model, model_source = get_content_model()
                calibration = get_calibration()
                model_dataset_hash = str(getattr(model.config, "fishstop_dataset_sha256", "") or "")
                calibration_dataset_hash = str(calibration.get("dataset_sha256") or "")
                if model_dataset_hash and calibration_dataset_hash and model_dataset_hash != calibration_dataset_hash:
                    raise ValueError(
                        "DistilBERT model and calibration.json were produced from different datasets"
                    )

                st.info("DistilBERT model loaded from Hugging Face.")
                if calibration["source"] != "huggingface":
                    st.caption(
                        "⚠️ Nessun `calibration.json` trovato per questo modello: uso i "
                        "default legacy (soglia 50%, banda 35-65%, nessun temperature "
                        "scaling). Rilancia il notebook di training per generarlo."
                    )

                if not email_text:
                    soc["bert_ai_result"] = "not available"
                    st.warning("Email has no meaningful text for classification.")
                else:
                    with st.spinner("DistilBERT is analyzing the complete content..."):
                        positive_label_id = int(calibration.get("positive_label_id", 1))
                        logits, chunk_count = predict_email_logits(
                            model,
                            tokenizer,
                            email_text,
                            positive_label_id=positive_label_id,
                        )
                        probabilities = calibrated_probabilities(
                            logits, temperature=calibration["temperature"]
                        ).flatten().tolist()
                    negative_label_id = 1 - positive_label_id
                    prob_safe = probabilities[negative_label_id] * 100
                    prob_malicious = probabilities[positive_label_id] * 100
                    c1, c2 = st.columns(2)
                    c1.metric("Legitimate", f"{prob_safe:.2f}%")
                    c2.metric("Malicious (phishing/spam)", f"{prob_malicious:.2f}%")
                    if calibration["source"] == "huggingface":
                        st.caption(
                            f"Calibrated probability from {chunk_count} text block(s). "
                            "It measures the content signal only."
                        )
                    else:
                        st.caption(
                            f"Uncalibrated classifier confidence from {chunk_count} text block(s); "
                            "it is not a real-world malicious-email probability."
                        )
                    soc["bert_phishing_probability"] = prob_malicious
                    soc["bert_malicious_probability"] = prob_malicious
                    soc["bert_legitimate_probability"] = prob_safe
                    soc["bert_chunk_count"] = chunk_count
                    soc["bert_probability_calibrated"] = calibration["source"] == "huggingface"

                    result = classify_bert_result(
                        probabilities[positive_label_id],
                        threshold=calibration["threshold"],
                        band=calibration["band"],
                    )
                    soc["bert_ai_result"] = result
                    if result == "phishing":
                        st.error("AI result: possible malicious email (phishing or spam)")
                    elif result == "legitimate":
                        st.success("AI result: email probably legitimate")
                    else:
                        st.warning("AI result: inconclusive content signal")

                    with st.expander("Raw logits"):
                        st.json({
                            "logits": logits.flatten().tolist(),
                            "calibration_temperature": calibration["temperature"],
                            "decision_threshold": calibration["threshold"],
                            "uncertain_band": calibration["band"],
                            "positive_label_id": positive_label_id,
                            "analyzed_chunks": chunk_count,
                            "calibration_source": calibration["source"],
                        })

                st.markdown("#### Extracted Body")
                source = soc.get("body_source", "unknown")
                st.caption(f"Source: `{source}`")
                ai_context = soc.get("body_context", "normal")
                body_display = soc.get("body_extracted") or soc.get("body_ai") or soc.get("body_clean") or ""
                full_body = soc.get("body_for_ai") or soc.get("body_clean_full") or soc.get("body_clean") or ""
                body_html = soc.get("body_html") or ""
                if ai_context == "forwarded":
                    st.info("Forwarded email: the forwarded content is shown and analyzed.")
                elif ai_context == "reply":
                    st.info("Email reply: only the current reply is shown and analyzed.")
                removed_tail = int(soc.get("body_ai_removed_tail_lines") or 0)
                if removed_tail:
                    st.caption(f"AI body cleanup removed {removed_tail} signature/disclaimer/unsubscribe line(s).")

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
                            "Render HTML with safe JavaScript guard",
                            value=False,
                            key=f"{phi4_key}_body_html_javascript",
                        )
                        if enable_html_javascript:
                            st.caption("Safe HTML preview: JavaScript from the email is removed; only the internal safety guard can run.")
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
                st.caption("A readable breakdown of the parsed message, security evidence, and routing data.")
                _render_structured_report(soc, eml_auth, flags, links, attachments)
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
