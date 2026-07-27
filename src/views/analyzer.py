import json
import hashlib
import copy
import re
import tempfile
from html import escape as html_escape
from threading import Lock
from urllib.parse import urlsplit, urlunsplit

import streamlit as st
import streamlit.components.v1 as components
 
from src.analysis_limits import EmailAnalysisLimitError, MAX_EML_BYTES
from src.analyzer.html_utils import sanitize_html_for_js_preview, sanitize_html_for_preview
from src.analyzer.llm_context_analyzer import (
    PROMPT_VERSION,
    active_llm_backend,
    apply_email_risk_policy,
    format_email_risk_analysis,
    stream_phi4_email_analysis,
)
from src.background_jobs import BackgroundJobManager
from src.analyzer.received_parser import order_received_hops
from src.bert_calibration import calibrated_probabilities, classify as classify_bert_result
from src.bert_inference import predict_email_logits
from src.bert_input import prepare_bert_input
from src.components.email_globe import render_email_globe
from src.config import get_secret, get_server_secret, is_production_mode
from src.error_handling import render_unexpected_error
from src.session_context import get_analysis_session_id
from src.ui import page_intro, risk_banner
from src.views.backend import (
    HF_MODEL_REVISION,
    get_core_backend,
    init_calibration,
    init_content_model,
)

_EMAIL_UPLOADER_KEY = "fishstop_eml_uploader"
_BROWSER_RELOAD_QUERY_KEY = "_fishstop_reload"
_BROWSER_RELOAD_STATE_KEY = "_fishstop_seen_reload"
_EMAIL_STATE_KEYS = {
    _EMAIL_UPLOADER_KEY,
    "raw_eml_text",
    "current_eml_name",
    "current_eml_hash",
}
_EMAIL_STATE_PREFIXES = (
    "soc_analysis_",
    "phi4_analysis_",
    "background_lookup_signature_",
)
_BERT_INFERENCE_LOCK = Lock()


def _is_email_analysis_state_key(key: str) -> bool:
    return key in _EMAIL_STATE_KEYS or key.startswith(_EMAIL_STATE_PREFIXES)


def _clear_email_analysis_state() -> None:
    """Remove only the current client's uploaded email and derived UI state."""
    for key in list(st.session_state):
        if _is_email_analysis_state_key(str(key)):
            del st.session_state[key]


def _reset_email_after_browser_reload() -> None:
    """
    Reset the uploaded email after a real browser refresh, not after Streamlit
    reruns used to update background analysis results.
    """
    reload_token = str(st.query_params.get(_BROWSER_RELOAD_QUERY_KEY, "") or "")
    if (
        reload_token
        and reload_token != st.session_state.get(_BROWSER_RELOAD_STATE_KEY)
    ):
        _clear_email_analysis_state()
        st.session_state[_BROWSER_RELOAD_STATE_KEY] = reload_token

    components.html(
        f"""
        <script>
        (() => {{
          try {{
            const parentWindow = window.parent;
            const navigation = parentWindow.performance
              .getEntriesByType("navigation")[0];
            if (!navigation || navigation.type !== "reload") return;

            const url = new URL(parentWindow.location.href);
            const token = (
              parentWindow.crypto && parentWindow.crypto.randomUUID
            )
              ? parentWindow.crypto.randomUUID()
              : `${{Date.now()}}-${{Math.random().toString(16).slice(2)}}`;
            url.searchParams.set("{_BROWSER_RELOAD_QUERY_KEY}", token);
            parentWindow.location.replace(url.toString());
          }} catch (_) {{
            // The explicit Clear email button remains available as a fallback.
          }}
        }})();
        </script>
        """,
        height=0,
    )


@st.cache_resource
def _get_background_job_manager() -> BackgroundJobManager:
    return BackgroundJobManager()


def _analyze_eml_bytes(raw_bytes: bytes) -> dict:
    _, analyzer = get_core_backend()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as temp_file:
            temp_file.write(raw_bytes)
            temp_path = temp_file.name
        return analyzer.analyze(temp_path)
    finally:
        if temp_path:
            try:
                import os

                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _validate_eml_size(raw_bytes: bytes) -> None:
    if len(raw_bytes) > MAX_EML_BYTES:
        raise EmailAnalysisLimitError(
            "Email file is larger than the maximum supported size of "
            f"{MAX_EML_BYTES / (1024 * 1024):.0f} MB."
        )


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


def _severity(
    counts: dict,
    phi4_analysis: dict | None = None,
) -> tuple[str, str]:
    if isinstance(phi4_analysis, dict):
        verdict = str(phi4_analysis.get("final_verdict") or "").lower()
        if verdict == "phishing":
            return "CRITICAL", "Final combined verdict: phishing"
        if verdict == "review":
            return "SUSPICIOUS", "Final combined verdict: manual review required"
        if verdict == "legitimate":
            return "LOW", "Final combined verdict: likely legitimate"
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


def _run_phi4_background(soc: dict, github_token: str = "") -> dict:
    last_error = ""
    for event in stream_phi4_email_analysis(
        soc,
        github_token=github_token or None,
    ):
        if event.get("status") == "ok":
            return event
        if event.get("status") == "error":
            last_error = event.get("message") or "Phi-4 mini analysis failed"
    raise RuntimeError(last_error or "Phi-4 mini did not return a final result")


def _schedule_phi4_background(
    manager: BackgroundJobManager,
    soc: dict,
    analysis_key: str,
    session_id: str = "",
) -> tuple[str, tuple] | None:
    session_id = session_id or get_analysis_session_id()
    backend = active_llm_backend()
    hosted = backend.startswith("github models")
    if backend == "not configured":
        return None
    if soc.get("ai_analysis_supported") is False:
        return None

    github_token = get_secret("GITHUB_MODELS_TOKEN") if hosted else ""
    token_revision = (
        hashlib.sha256(github_token.encode("utf-8")).hexdigest()
        if github_token
        else "local"
    )
    key = ("phi4", session_id, analysis_key, backend, token_revision)
    soc_snapshot = copy.deepcopy({
        name: value
        for name, value in soc.items()
        if name != "raw_eml_bytes"
    })
    accepted = manager.get_or_submit(
        "llm",
        key,
        _run_phi4_background,
        soc_snapshot,
        github_token,
    )
    return ("llm", key) if accepted is not False else ("__overloaded__", key)


def _run_bert_background(email_text: str, hf_token: str = "") -> dict:
    tokenizer, model, model_source = init_content_model(hf_token)
    calibration = dict(init_calibration(hf_token))
    model_dataset_hash = str(
        getattr(model.config, "fishstop_dataset_sha256", "") or ""
    )
    calibration_dataset_hash = str(
        calibration.get("dataset_sha256") or ""
    )
    if (
        model_dataset_hash
        and calibration_dataset_hash
        and model_dataset_hash != calibration_dataset_hash
    ):
        raise ValueError(
            "DistilBERT model and calibration.json were produced from "
            "different datasets"
        )

    positive_label_id = int(calibration.get("positive_label_id", 1))
    # The cached model/tokenizer are process-global Streamlit resources.
    # Serialize inference so this remains safe if the BERT worker count changes.
    with _BERT_INFERENCE_LOCK:
        logits, chunk_count = predict_email_logits(
            model,
            tokenizer,
            email_text,
            positive_label_id=positive_label_id,
        )
    probabilities = calibrated_probabilities(
        logits,
        temperature=calibration["temperature"],
    ).flatten().tolist()
    negative_label_id = 1 - positive_label_id
    classification = classify_bert_result(
        probabilities[positive_label_id],
        threshold=calibration["threshold"],
        band=calibration["band"],
    )
    return {
        "classification": classification,
        "probability_legitimate": probabilities[negative_label_id] * 100,
        "probability_malicious": probabilities[positive_label_id] * 100,
        "logits": logits.detach().cpu().flatten().tolist(),
        "chunk_count": chunk_count,
        "positive_label_id": positive_label_id,
        "model_source": model_source,
        "model_dataset_hash": model_dataset_hash,
        "calibration_dataset_hash": calibration_dataset_hash,
        "calibration": calibration,
    }


def _schedule_bert_background(
    manager: BackgroundJobManager,
    email_text: str,
    email_digest: str,
    session_id: str = "",
) -> tuple[str, tuple] | None:
    if not email_text:
        return None
    session_id = session_id or get_analysis_session_id()
    hf_token = (
        get_server_secret("HF_TOKEN")
        if is_production_mode()
        else get_secret("HF_TOKEN")
    )
    token_revision = (
        hashlib.sha256(hf_token.encode("utf-8")).hexdigest()
        if hf_token
        else "public"
    )
    key = (
        "bert",
        session_id,
        email_digest,
        HF_MODEL_REVISION,
        token_revision,
    )
    accepted = manager.get_or_submit(
        "bert",
        key,
        _run_bert_background,
        email_text,
        hf_token,
    )
    return ("bert", key) if accepted is not False else ("__overloaded__", key)


def _apply_bert_result_to_soc(soc: dict, result: dict) -> None:
    soc["bert_ai_result"] = result["classification"]
    soc["bert_phishing_probability"] = result["probability_malicious"]
    soc["bert_malicious_probability"] = result["probability_malicious"]
    soc["bert_legitimate_probability"] = result["probability_legitimate"]
    soc["bert_chunk_count"] = result["chunk_count"]
    soc["bert_probability_calibrated"] = (
        (result.get("calibration") or {}).get("source") == "huggingface"
    )


def _render_phi4_analysis(
    soc: dict,
    analysis_key: str,
    job_reference: tuple[str, tuple] | None,
    auto_run: bool = False,
):
    st.markdown("#### Phi-4 mini scam/phishing explanation")
    st.caption(
        f"Phi-4 mini analysis via {active_llm_backend()}: evaluates plain and HTML content, urgency, money, IBANs, "
        "payments, credentials, and external forms; then uses semantic analysis, SPF/DKIM/DMARC, links, and attachments only as context."
    )
    if soc.get("ai_analysis_supported") is False:
        st.warning(soc.get("ai_analysis_limit_message"))
        return None

    result_key = f"{analysis_key}_result"
    error_key = f"{analysis_key}_error"

    if st.session_state.get(result_key):
        _show_phi4_result(st, st.session_state[result_key])
        return None

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
        return None

    backend = active_llm_backend()

    if job_reference is not None:
        if job_reference[0] == "__overloaded__":
            st.warning(
                "Phi-4 analysis is temporarily unavailable because the "
                "analysis queue is full. Try again shortly."
            )
            return None
        snapshot = _get_background_job_manager().snapshot(*job_reference)
        if snapshot.state == "done":
            event = snapshot.result if isinstance(snapshot.result, dict) else {}
            analysis = event.get("analysis")
            if not isinstance(analysis, dict):
                st.session_state[error_key] = (
                    "Structured Phi-4 analysis is missing"
                )
                st.error(st.session_state[error_key])
                return None
            st.session_state[result_key] = analysis
            st.rerun()
        if snapshot.state == "error":
            st.session_state[error_key] = (
                f"Error during Phi-4 mini analysis: {snapshot.error}"
            )
            st.error(st.session_state[error_key])
            return None
        st.markdown(_phi4_loading_html(), unsafe_allow_html=True)
        return None

    if auto_run and backend == "not configured":
        st.error(
            "LLM analysis unavailable: start Ollama locally with Phi-4 mini "
            "or configure GITHUB_MODELS_TOKEN."
        )
        return None

    st.info("Phi-4 mini analysis starts automatically in the Executive Triage panel.")
    return None


def _show_phi4_result(target, result):
    if not isinstance(result, dict):
        target.info(str(result))
        return

    verdict = str(result.get("final_verdict") or "review").lower()
    message = format_email_risk_analysis(result)
    evidence = str(result.get("intent_evidence") or "").strip()
    if evidence:
        message += f'\n\nIntent evidence: “{evidence}”'
    signals = [
        str(value).replace("_", " ")
        for value in (result.get("intent_signals") or [])
    ]
    if signals:
        message += f"\n\nContext signals: {', '.join(signals)}."
    signal_evidence = str(result.get("signal_evidence") or "").strip()
    if signal_evidence:
        message += f'\n\nContext evidence: "{signal_evidence}"'
    claimed_brand = str(result.get("claimed_brand") or "").strip()
    if claimed_brand:
        message += f"\n\nClaimed identity: {claimed_brand}."
    if verdict == "phishing":
        target.error(message)
    elif verdict == "review":
        target.warning(message)
    else:
        target.success(message)


def _phi4_loading_html(
    detail: str = (
        "Phi-4 mini is still analyzing the email. "
        "The final verdict is not available yet."
    ),
) -> str:
    safe_detail = html_escape(str(detail or "Phi-4 mini is analyzing the email."))
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
        <span class="phi4-spinner" aria-hidden="true"></span>
        <span>
            <strong style="display:block; color:#24292f;">Analysis in progress</strong>
            __PHI4_PROGRESS_DETAIL__
        </span>
        <span class="phi4-typing-dots" aria-label="loading">
            <span></span><span></span><span></span>
        </span>
    </div>
    <style>
        .phi4-spinner {
            width: 18px;
            height: 18px;
            flex: 0 0 18px;
            border: 2px solid #afb8c1;
            border-top-color: #0969da;
            border-radius: 50%;
            animation: phi4Spin 0.8s linear infinite;
        }
        .phi4-typing-dots {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            margin-left: auto;
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
        @keyframes phi4Spin {
            to {
                transform: rotate(360deg);
            }
        }
    </style>
    """.replace("__PHI4_PROGRESS_DETAIL__", safe_detail)


def _render_abuseipdb(rep: dict):
    status = rep.get("status")
    if status == "pending":
        st.info(rep.get("message", "Reputation analysis in progress"))
    elif status == "ok":
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
    if geo.get("status") == "pending":
        st.info(geo.get("message", "Geolocation in progress"))
    elif geo.get("status") == "ok":
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
    if status == "pending":
        st.info(vt.get("message", "VirusTotal analysis in progress"))
        return
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

    if status == "pending":
        st.info(f"VirusTotal: {message or 'analysis in progress'}")
        return
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


def _safe_vt_url_lookup(
    validator,
    url: str,
    api_key: str | None = None,
) -> dict:
    lookup = getattr(validator, "check_url_reputation", None)
    if callable(lookup):
        try:
            return lookup(url, api_key=api_key)
        except TypeError:
            return lookup(url)
    return {"status": "skipped", "url": url, "message": "Validator VirusTotal URL unavailable"}


def _sender_domains(soc: dict) -> dict[str, str]:
    def pull_domain(raw: str | None) -> str:
        if not raw:
            return ""
        match = re.search(r"@([\w.\-]+)", raw)
        return match.group(1).lower() if match else ""

    domains: dict[str, str] = {}
    from_domain = pull_domain(soc.get("from_"))
    return_path_domain = pull_domain(soc.get("return_path"))
    reply_to_domain = pull_domain(soc.get("reply_to"))
    if from_domain:
        domains[f"From ({from_domain})"] = from_domain
    if return_path_domain and return_path_domain != from_domain:
        domains[f"Return-Path ({return_path_domain})"] = return_path_domain
    if reply_to_domain and reply_to_domain not in {
        from_domain,
        return_path_domain,
    }:
        domains[f"Reply-To ({reply_to_domain})"] = reply_to_domain
    return domains


def _routing_ips(soc: dict) -> list[str]:
    values = [
        soc.get("injection_sender_ip"),
        *[
            ip
            for hop in (soc.get("received_hops") or [])
            for ip in (
                hop.get("all_ips")
                or ([hop.get("sender_ip")] if hop.get("sender_ip") else [])
            )
        ],
    ]
    return list(dict.fromkeys(
        str(value).strip()
        for value in values
        if str(value or "").strip()
    ))


def _schedule_background_lookups(
    manager: BackgroundJobManager,
    validator,
    soc: dict,
    session_id: str = "",
) -> dict[str, dict[str, tuple[str, tuple]]]:
    session_id = session_id or get_analysis_session_id()
    vt_api_key = get_secret("VIRUSTOTAL_API_KEY")
    abuse_api_key = get_secret("ABUSEIPDB_API_KEY")
    vt_revision = hashlib.sha256(vt_api_key.encode("utf-8")).hexdigest() if vt_api_key else "missing"
    abuse_revision = hashlib.sha256(abuse_api_key.encode("utf-8")).hexdigest() if abuse_api_key else "missing"
    plan: dict[str, dict[str, tuple[str, tuple]]] = {
        "urls": {},
        "domains": {},
        "ip_reputation": {},
        "geolocation": {},
        "files": {},
    }

    for link in (soc.get("links") or []):
        url = str(link.get("url") or "").strip()
        if not url or url in plan["urls"]:
            continue
        key = ("url", session_id, vt_revision, url)
        accepted = manager.get_or_submit(
            "virustotal",
            key,
            _safe_vt_url_lookup,
            validator,
            url,
            vt_api_key,
        )
        plan["urls"][url] = (
            ("virustotal", key)
            if accepted is not False
            else ("__overloaded__", key)
        )

    for domain in dict.fromkeys(_sender_domains(soc).values()):
        key = ("domain", session_id, abuse_revision, domain)
        accepted = manager.get_or_submit(
            "abuseipdb",
            key,
            validator.check_domain_reputation,
            domain,
            api_key=abuse_api_key,
        )
        plan["domains"][domain] = (
            ("abuseipdb", key)
            if accepted is not False
            else ("__overloaded__", key)
        )

    for ip in _routing_ips(soc):
        reputation_key = ("ip", session_id, abuse_revision, ip)
        accepted = manager.get_or_submit(
            "abuseipdb",
            reputation_key,
            validator.check_ip_reputation,
            ip,
            api_key=abuse_api_key,
        )
        plan["ip_reputation"][ip] = (
            ("abuseipdb", reputation_key)
            if accepted is not False
            else ("__overloaded__", reputation_key)
        )

        geolocation_key = ("ip", session_id, ip)
        accepted = manager.get_or_submit(
            "geolocation",
            geolocation_key,
            validator.geolocate_ip,
            ip,
        )
        plan["geolocation"][ip] = (
            ("geolocation", geolocation_key)
            if accepted is not False
            else ("__overloaded__", geolocation_key)
        )

    for attachment in (soc.get("attachments") or []):
        sha256 = str(attachment.get("hash_sha256") or "").strip()
        if not sha256 or sha256 in plan["files"]:
            continue
        key = ("file", session_id, vt_revision, sha256)
        accepted = manager.get_or_submit(
            "virustotal",
            key,
            validator.check_file_hash,
            sha256,
            api_key=vt_api_key,
        )
        plan["files"][sha256] = (
            ("virustotal", key)
            if accepted is not False
            else ("__overloaded__", key)
        )

    return plan


def _background_lookup_result(
    manager: BackgroundJobManager,
    reference: tuple[str, tuple],
    *,
    service: str,
) -> dict:
    pool, key = reference
    if pool == "__overloaded__":
        return {
            "status": "unavailable",
            "message": (
                f"{service} was not started because the analysis queue is full"
            ),
        }
    snapshot = manager.snapshot(pool, key)
    if snapshot.state == "done":
        if isinstance(snapshot.result, dict):
            return snapshot.result
        return {
            "status": "error",
            "message": f"{service} returned an invalid result",
        }
    if snapshot.state == "error":
        return {
            "status": "error",
            "message": f"{service}: {snapshot.error}",
        }
    return {
        "status": "pending",
        "message": f"{service} analysis in progress",
    }


def _background_plan_states(
    manager: BackgroundJobManager,
    plan: dict[str, dict[str, tuple[str, tuple]]],
) -> tuple[str, ...]:
    return tuple(
        (
            "overloaded"
            if pool == "__overloaded__"
            else manager.snapshot(pool, key).state
        )
        for group in plan.values()
        for pool, key in group.values()
    )


def _background_refresh_required(
    rendered_states: tuple[str, ...],
    current_states: tuple[str, ...],
    previous_states: tuple[str, ...] | None,
) -> bool:
    return (
        current_states != rendered_states
        or (
            previous_states is not None
            and previous_states != current_states
        )
    )


@st.fragment(run_every=0.75)
def _render_background_progress(
    email_hash: str,
    plan: dict[str, dict[str, tuple[str, tuple]]],
    rendered_states: tuple[str, ...],
) -> None:
    manager = _get_background_job_manager()
    states = _background_plan_states(manager, plan)
    if not states:
        return
    terminal_states = {"done", "error", "cancelled", "overloaded"}

    def group_progress(group_names: set[str]) -> tuple[int, int]:
        group_states = [
            (
                "overloaded"
                if pool == "__overloaded__"
                else manager.snapshot(pool, key).state
            )
            for name, group in plan.items()
            if name in group_names
            for pool, key in group.values()
        ]
        return (
            sum(state in terminal_states for state in group_states),
            len(group_states),
        )

    external_done, external_total = group_progress({
        "urls", "domains", "ip_reputation", "geolocation", "files",
    })
    bert_done, bert_total = group_progress({"bert"})
    phi_done, phi_total = group_progress({"phi4"})
    progress_parts = []
    if external_total:
        progress_parts.append(
            f"External intelligence {external_done}/{external_total}"
        )
    if bert_total:
        progress_parts.append(f"DistilBERT {bert_done}/{bert_total}")
    if phi_total:
        progress_parts.append(f"Phi-4 mini {phi_done}/{phi_total}")

    completed = sum(state in terminal_states for state in states)
    if completed < len(states):
        st.info(
            "Results appear independently as soon as they are ready: "
            + " · ".join(progress_parts)
        )
    else:
        st.success("All analysis jobs completed: " + " · ".join(progress_parts))

    signature_key = f"background_lookup_signature_{email_hash}"
    previous = st.session_state.get(signature_key)
    if _background_refresh_required(
        rendered_states,
        states,
        previous,
    ):
        st.session_state[signature_key] = states
        st.rerun(scope="app")
    st.session_state[signature_key] = states


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
    _reset_email_after_browser_reload()
    validator, _ = get_core_backend()
    background_jobs = _get_background_job_manager()
    analysis_session_id = get_analysis_session_id()

    page_intro(
        "Email analysis",
        "Investigate a suspicious email",
        "Upload an .eml file to review its risk, sender authenticity, links, and attachments in one place.",
    )

    col_upload = st.container(border=True)
    col_results = st.container()

    with col_upload:
        st.markdown('<div class="fs-section-label">Add email file</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an .eml file",
            type=["eml"],
            label_visibility="collapsed",
            key=_EMAIL_UPLOADER_KEY,
        )
        st.caption(
            "The file is parsed locally. Reputation services receive only indicators. "
            "On the hosted website, anonymized email content is analyzed with GitHub Models."
        )

        if uploaded_file is not None:
            st.button(
                "Clear email",
                key="clear_email_analysis",
                on_click=_clear_email_analysis_state,
            )
            raw_bytes = uploaded_file.getvalue()
            try:
                _validate_eml_size(raw_bytes)
            except EmailAnalysisLimitError as exc:
                st.error(str(exc))
                return
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            current_eml_hash = hashlib.sha256(raw_bytes).hexdigest()
            if st.session_state.get("current_eml_hash") != current_eml_hash:
                st.session_state["raw_eml_text"] = raw_text
                st.session_state["current_eml_name"] = uploaded_file.name
                st.session_state["current_eml_hash"] = current_eml_hash
                st.rerun()

            file_col, size_col = st.columns([2, 1])
            file_col.metric("File", uploaded_file.name)
            size_col.metric("Size", f"{len(uploaded_file.getbuffer()) / 1024:.1f} KB")

    with col_results:
        if uploaded_file is None:
            st.info("Upload an email to begin. The analysis will appear here automatically.")
            return

        try:
            with st.spinner("Parsing EML e costruzione report SOC..."):
                soc_cache_key = f"soc_analysis_{current_eml_hash}"
                soc = st.session_state.get(soc_cache_key)
                if soc is None:
                    soc = _analyze_eml_bytes(raw_bytes)
                    st.session_state[soc_cache_key] = soc

            flags = soc.get("flags", [])
            links = soc.get("links", [])
            attachments = soc.get("attachments", [])
            lookalike_alerts = soc.get("lookalike_alerts", [])
            eml_auth = _email_auth_from_eml(soc)
            background_plan = _schedule_background_lookups(
                background_jobs,
                validator,
                soc,
                analysis_session_id,
            )
            rendered_background_states = list(
                _background_plan_states(
                    background_jobs,
                    background_plan,
                )
            )
            vt_url_results = {
                url: _background_lookup_result(
                    background_jobs,
                    reference,
                    service="VirusTotal URL",
                )
                for url, reference in background_plan["urls"].items()
            }
            domain_results = {
                domain: _background_lookup_result(
                    background_jobs,
                    reference,
                    service="Domain reputation",
                )
                for domain, reference in background_plan["domains"].items()
            }
            hop_reputation_results = {
                ip: _background_lookup_result(
                    background_jobs,
                    reference,
                    service="IP reputation",
                )
                for ip, reference in background_plan["ip_reputation"].items()
            }
            geolocation_results = {
                ip: _background_lookup_result(
                    background_jobs,
                    reference,
                    service="IP geolocation",
                )
                for ip, reference in background_plan["geolocation"].items()
            }
            file_results = {
                sha256: _background_lookup_result(
                    background_jobs,
                    reference,
                    service="VirusTotal file",
                )
                for sha256, reference in background_plan["files"].items()
            }
            soc["link_reputation"] = vt_url_results
            soc["link_reputation_summary"] = _summarize_link_reputation(vt_url_results)
            soc["domain_reputation"] = domain_results
            soc["hop_reputation"] = hop_reputation_results
            soc["geolocation_results"] = geolocation_results
            for attachment in attachments:
                sha256 = str(attachment.get("hash_sha256") or "").strip()
                if sha256 and sha256 in file_results:
                    attachment["file_reputation"] = file_results[sha256]

            eml_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            phi4_key = f"phi4_analysis_{PROMPT_VERSION}_{eml_digest}"
            clean_body = (
                soc.get("body_for_ai")
                or soc.get("body_ai")
                or soc.get("body_clean")
                or ""
            )
            email_text = prepare_bert_input(
                soc.get("subject") or "",
                clean_body,
                has_extracted_links=bool(soc.get("links")),
            )
            ai_analysis_supported = soc.get("ai_analysis_supported", True)
            bert_job_reference = (
                _schedule_bert_background(
                    background_jobs,
                    email_text,
                    eml_digest,
                    analysis_session_id,
                )
                if ai_analysis_supported
                else None
            )
            bert_result = None
            bert_error = (
                ""
                if ai_analysis_supported
                else str(soc.get("ai_analysis_limit_message") or "")
            )
            if bert_job_reference is not None:
                background_plan["bert"] = {
                    eml_digest: bert_job_reference,
                }
                bert_snapshot = (
                    background_jobs.snapshot(*bert_job_reference)
                    if bert_job_reference[0] != "__overloaded__"
                    else None
                )
                if bert_snapshot is None:
                    bert_error = (
                        "DistilBERT analysis is temporarily unavailable because "
                        "the analysis queue is full. Try again shortly."
                    )
                    rendered_background_states.append("overloaded")
                else:
                    rendered_background_states.append(bert_snapshot.state)
                if (
                    bert_snapshot is not None
                    and bert_snapshot.state == "done"
                    and isinstance(bert_snapshot.result, dict)
                ):
                    bert_result = bert_snapshot.result
                    _apply_bert_result_to_soc(soc, bert_result)
                elif bert_snapshot is not None and bert_snapshot.state == "error":
                    bert_error = bert_snapshot.error
            elif not email_text:
                soc["bert_ai_result"] = "not available"

            cached_phi4 = st.session_state.get(f"{phi4_key}_result")
            phi4_error = (
                st.session_state.get(f"{phi4_key}_error")
                or (
                    str(soc.get("ai_analysis_limit_message") or "")
                    if not ai_analysis_supported
                    else ""
                )
            )
            phi4_job_reference = None
            if not isinstance(cached_phi4, dict) and not phi4_error:
                phi4_job_reference = _schedule_phi4_background(
                    background_jobs,
                    soc,
                    phi4_key,
                    analysis_session_id,
                )
                if phi4_job_reference is not None:
                    background_plan["phi4"] = {
                        phi4_key: phi4_job_reference,
                    }
                    rendered_background_states.append(
                        (
                            "overloaded"
                            if phi4_job_reference[0] == "__overloaded__"
                            else background_jobs.snapshot(
                                *phi4_job_reference
                            ).state
                        )
                    )
            if (
                isinstance(cached_phi4, dict)
                and isinstance(cached_phi4.get("semantic_extraction"), dict)
            ):
                cached_phi4 = apply_email_risk_policy(
                    soc,
                    cached_phi4["semantic_extraction"],
                )
                st.session_state[f"{phi4_key}_result"] = cached_phi4
            phi4_pending = not isinstance(cached_phi4, dict) and not phi4_error
            if isinstance(cached_phi4, dict):
                phi4_verdict = str(
                    cached_phi4.get("final_verdict") or ""
                ).lower()
                if phi4_verdict in {"phishing", "review"}:
                    flags = [
                        *flags,
                        {
                            "level": "HIGH" if phi4_verdict == "phishing" else "MEDIUM",
                            "field": "Phi-4 policy",
                            "message": (
                                cached_phi4.get("content_summary")
                                or cached_phi4.get("explanation")
                                or f"Phi-4 verdict: {phi4_verdict}"
                            ),
                        },
                    ]
            counts = _flag_counts(flags)
            severity, severity_caption = _severity(counts, cached_phi4)

            if isinstance(cached_phi4, dict):
                st.markdown("### Final analysis")
            elif phi4_pending:
                st.markdown("### Analysis in progress")
                st.info(
                    "Phi-4 mini is still analyzing the email. The indicators below "
                    "are preliminary; the final verdict will appear automatically."
                )
            else:
                st.markdown("### Preliminary analysis")
            risk_banner(
                severity,
                severity_caption
                if not phi4_pending
                else f"Preliminary technical triage: {severity_caption}",
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Critical signals", counts["HIGH"])
            c2.metric("Review signals", counts["MEDIUM"])
            c3.metric("Links", len(links))
            c4.metric("Attachments", len(attachments))
            _render_background_progress(
                current_eml_hash,
                background_plan,
                tuple(rendered_background_states),
            )

            with st.container(border=True):
                _render_phi4_analysis(
                    soc,
                    phi4_key,
                    phi4_job_reference,
                    auto_run=True,
                )

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
                domains = _sender_domains(soc)

                if not domains:
                    st.info("No sender domain could be extracted.")
                else:
                    for label, domain in domains.items():
                        with st.expander(label):
                            _render_abuseipdb(
                                domain_results.get(domain, {
                                    "status": "pending",
                                    "message": "Domain reputation analysis in progress",
                                })
                            )

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
                    render_email_globe(
                        soc,
                        geolocation_results=geolocation_results,
                        reputation_results=hop_reputation_results,
                    )

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
                                _render_geo(
                                    geolocation_results.get(ip, {
                                        "status": "pending",
                                        "message": "IP geolocation in progress",
                                    })
                                )
                                with st.expander("AbuseIPDB"):
                                    _render_abuseipdb(
                                        hop_reputation_results.get(ip, {
                                            "status": "pending",
                                            "message": "IP reputation analysis in progress",
                                        })
                                    )
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
                                _render_virustotal(
                                    file_results.get(att["hash_sha256"], {
                                        "status": "pending",
                                        "message": "VirusTotal file analysis in progress",
                                    })
                                )


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
                _render_phi4_analysis(
                    soc,
                    phi4_key,
                    phi4_job_reference,
                    auto_run=False,
                )

                # DistilBERT runs in the dedicated background worker.

                if not email_text:
                    st.warning("Email has no meaningful text for classification.")
                elif bert_error:
                    st.error(f"DistilBERT analysis failed: {bert_error}")
                elif not isinstance(bert_result, dict):
                    st.info(
                        "DistilBERT is analyzing the complete content in the "
                        "background. This section will update automatically."
                    )
                else:
                    calibration = bert_result["calibration"]
                    chunk_count = int(bert_result["chunk_count"])
                    prob_safe = float(bert_result["probability_legitimate"])
                    prob_malicious = float(bert_result["probability_malicious"])
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
                    result = bert_result["classification"]
                    if result == "phishing":
                        st.error("AI result: possible malicious email (phishing or spam)")
                    elif result == "legitimate":
                        st.success("AI result: email probably legitimate")
                    else:
                        st.warning("AI result: inconclusive content signal")

                    with st.expander("Raw logits"):
                        st.json({
                            "logits": bert_result["logits"],
                            "calibration_temperature": calibration["temperature"],
                            "decision_threshold": calibration["threshold"],
                            "uncertain_band": calibration["band"],
                            "positive_label_id": bert_result["positive_label_id"],
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

        except EmailAnalysisLimitError as exc:
            st.error(f"Email exceeds the supported analysis limits: {exc}")
        except Exception as exc:
            render_unexpected_error(
                "An unexpected error occurred during email analysis.",
                exc,
                context="email analysis page",
            )
