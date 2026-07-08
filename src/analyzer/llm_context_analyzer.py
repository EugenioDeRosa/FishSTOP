import json
import os
import re

import requests

from src.config import GITHUB_MODELS_TOKEN

GITHUB_MODELS_ENDPOINT = os.getenv(
    "GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com/chat/completions"
)
# Verifica il nome esatto nel codice di esempio di GitHub Models (Marketplace ->
# Phi-4-mini-instruct -> "Get API access"): il catalogo a volte usa un id diverso.
GITHUB_MODELS_MODEL = os.getenv("GITHUB_MODELS_MODEL", "Phi-4-mini-instruct")


def _llm_enabled() -> bool:
    return bool(GITHUB_MODELS_TOKEN)


SYSTEM_MESSAGE = (
    "You are a SOC assistant that explains email intent. "
    "You do not perform independent forensic analysis. "
    "First understand the intent of the anonymized subject/body, including HTML-derived body text when present. "
    "Then use only the structured technical facts provided by FishSTOP as supporting context. "
    "Answer with one concise English paragraph and no JSON."
)


def _clip(value: str, limit: int) -> str:
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[...troncato...]"


def _anonymize_for_llm(value: str) -> str:
    value = str(value or "")
    replacements = [
        (r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", "[IBAN]"),
        (r"(?<!\w)\+?\d[\d .()/-]{7,}\d\b", "[PHONE]"),
        (r"\b(?:\d[ -]?){13,19}\b", "[POSSIBLE_CARD_OR_ACCOUNT]"),
        (r"\b(?:[A-Za-z0-9._%+-]+)@(?:[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", "[EMAIL]"),
        (r"\bhttps?://[^\s<>\"]+", "[URL]"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
        (r"\b(Ciao|Gentile|Buongiorno|Buonasera|Salve)\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{2,}", r"\1 [PERSON]"),
        (r"\b(Sig\.?|Sig\.ra|Dott\.?|Dott\.ssa|Mr\.?|Mrs\.?|Ms\.?)\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ' -]{2,}", r"\1 [PERSON]"),
    ]
    anonymized = value
    for pattern, placeholder in replacements:
        anonymized = re.sub(pattern, placeholder, anonymized, flags=re.IGNORECASE)
    return anonymized


def _normalize_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _body_context_for_llm(soc: dict) -> tuple[str, str]:
    plain_body = soc.get("body_ai") or soc.get("body_extracted") or soc.get("body_clean") or soc.get("body") or ""
    html_body = soc.get("body_html_clean") or ""
    if not html_body and soc.get("body_html"):
        try:
            from .html_utils import strip_html
        except ImportError:
            from src.analyzer.html_utils import strip_html
        html_body = strip_html(soc.get("body_html") or "")

    plain_norm = _normalize_for_compare(plain_body)
    html_norm = _normalize_for_compare(html_body)
    has_distinct_html = bool(html_norm and html_norm != plain_norm and html_norm not in plain_norm)

    if has_distinct_html:
        return (
            "\n\n".join(
                part for part in [
                    "Plain/current body text:\n" + plain_body if plain_body else "",
                    "HTML-derived visible text:\n" + html_body,
                ]
                if part
            ),
            "plain text plus distinct HTML-derived visible text",
        )
    return plain_body or html_body, "plain text" if plain_body else ("HTML-derived visible text" if html_body else "empty")


def _detect_text_risk_signals(subject: str, body: str, links: list[dict]) -> list[str]:
    text = f"{subject or ''}\n{body or ''}".lower()
    urls = " ".join(str(link.get("url") or "") for link in links).lower()
    signals: list[str] = []
    neutral_notes: list[str] = []

    money_words = (
        "iban", "coordinate bancarie", "dati bancari", "conto bancario", "bank account",
        "sort code", "routing number", "swift", "bic", "account number",
        "bank details", "billing details", "pay now", "rimborso", "refund",
        "wire transfer", "bonifico", "pagamento", "payment", "fattura", "invoice",
        "saldo", "cambio conto", "nuovo conto", "new bank details", "$", "€",
        "won you", "you have won", "has won", "winner", "lottery", "prize",
        "donation", "charity donor", "inheritance", "fund", "million", "usd",
        "dollar", "claim your", "beneficiary",
    )
    credential_words = (
        "password", "credenzial", "credentials", "login", "accesso", "sign in",
        "account verification", "verifica account", "mfa", "otp",
    )
    urgency_words = (
        "urgente", "urgent", "entro oggi", "immediately", "as soon as possible",
        "scadenza", "last warning", "final notice", "sospensione",
        "locked", "limited", "act now", "within 24 hours",
        "scade", "deadline", "overdue", "sospeso", "blocked", "bloccato",
        "azione richiesta", "action required",
    )
    form_words = (
        "forms.gle", "docs.google.com/forms", "forms.office.com", "google form",
        "questionario", "survey", "modulo",
    )
    validation_words = (
        "email address is valid", "verify your email", "confirm your email",
        "get back to me", "reply back", "kindly reply",
    )
    vague_lure_words = (
        "profitable business opportunity", "business opportunity", "lucrative opportunity",
        "international collaboration", "share for your consideration", "of interest to you",
        "would like to share", "reply me via", "responda-me via",
        "oportunidade de negócio", "oportunidade de negocio", "negócio lucrativa",
        "negocio lucrativa", "colaboração internacional", "colaboracao internacional",
        "gostaria de compartilhar", "mais detalhes",
    )
    impersonation_words = (
        "jeff bezos", "elon musk", "bill gates", "amazon.com", "ceo",
        "direttore", "amministratore delegato", "hr department", "it department",
        "support team", "assistenza clienti",
        "founder", "president",
    )
    marketing_words = (
        "sconto", "sconti", "promo", "promozione", "offerta", "offerte",
        "discount", "sale", "newsletter", "coupon", "voucher",
    )

    has_money = any(word in text for word in money_words)
    has_credentials = any(word in text for word in credential_words)
    has_urgency = any(word in text for word in urgency_words)
    has_form = any(word in text or word in urls for word in form_words)
    has_validation = any(word in text for word in validation_words)
    has_vague_lure = any(word in text for word in vague_lure_words)
    has_impersonation = any(word in text for word in impersonation_words)
    has_marketing = any(word in text for word in marketing_words)

    if has_money:
        signals.append("money/payment/bank-detail wording is present")
    if has_credentials:
        signals.append("credential or account-verification wording is present")
    if has_urgency:
        signals.append("urgency or pressure wording is present")
    if has_form and (has_money or has_credentials):
        signals.append("external form or survey is paired with sensitive money/account wording")
    elif has_form:
        neutral_notes.append("external form or survey link is present without a sensitive-data request")
    if has_money and has_validation:
        signals.append("prize/donation/money claim asks the recipient to reply or validate the email address")
    if has_money and has_impersonation:
        signals.append("money/prize claim impersonates a well-known executive or brand")
    if has_vague_lure:
        signals.append("vague profitable business opportunity or international-collaboration lure asks for a reply")

    neutral_admin_words = (
        "firma ore", "firmare le ore", "firmarmi le ore", "timesheet",
        "attendance sheet", "foglio ore", "cartellino", "presenze", "ore lavorate",
    )
    if any(word in text for word in neutral_admin_words):
        neutral_notes.append("normal administrative work request wording is present")
    if has_marketing and not (has_credentials or has_form or has_validation):
        neutral_notes.append("promotional or discount wording is present without credential or sensitive-form requests")

    if signals:
        return signals + neutral_notes
    return [
        "no explicit money, credential, bank-detail, sensitive-form, or unusual-urgency wording found",
        *neutral_notes,
    ]


def _has_actionable_text_risk(signals: list[str]) -> bool:
    neutral_markers = (
        "no explicit",
        "without a sensitive-data request",
        "normal administrative work request",
    )
    return any(
        not any(marker in signal for marker in neutral_markers)
        for signal in signals
    )


def _content_precheck_label(has_actionable_text_risk: bool) -> str:
    if has_actionable_text_risk:
        return "local keyword/rule precheck found possible risky intent"
    return (
        "no local keyword/rule match; this is not a safety decision. "
        "The model must detect the language and inspect the original subject/body intent."
    )


def _link_hint(link: dict) -> str:
    host = (link.get("host") or "").lower()
    url = (link.get("url") or "").lower()
    if "docs.google.com/forms" in url or "forms.gle" in host or "forms.office.com" in host:
        return "external form; only suspicious if the email asks for sensitive data or credentials"
    if any(part in host for part in ("sharepoint.com", "teams.microsoft.com", "office.com", "microsoft.com")):
        return "common business/collaboration link; not suspicious by itself"
    if any(part in host for part in ("linkedin.com", "youtube.com", "zoom.us", "meet.google.com", "calendar.google.com")):
        return "common informational/meeting link; not suspicious by itself"
    if link.get("is_ip"):
        return "direct IP link; mention only if paired with a risky request in the body"
    return "neutral unless paired with a risky request in the body"


def _anonymized_link_hint(link: dict) -> str:
    host = (link.get("host") or "").lower()
    if "docs.google.com/forms" in host or "forms.gle" in host or "forms.office.com" in host:
        return "external form link"
    if any(part in host for part in ("sharepoint.com", "teams.microsoft.com", "office.com", "microsoft.com")):
        return "common business/collaboration link"
    if any(part in host for part in ("linkedin.com", "youtube.com", "zoom.us", "meet.google.com", "calendar.google.com")):
        return "common informational/meeting link"
    if link.get("is_ip"):
        return "direct IP link"
    return "generic extracted link"


def _auth_status(soc: dict, protocol: str) -> str:
    protocol = protocol.upper()
    auth_results = soc.get("auth_results") or {}
    arc_auth_results = soc.get("arc_auth_results") or {}
    result = auth_results.get(protocol) or arc_auth_results.get(protocol) or {}
    return (result.get("status") or "none").lower()


def _technical_context_lines(soc: dict) -> list[str]:
    spf_status = _auth_status(soc, "SPF")
    dkim_status = _auth_status(soc, "DKIM")
    dmarc_status = _auth_status(soc, "DMARC")
    auth_overall = "neutral"
    if dmarc_status in {"pass", "bestguesspass"} and spf_status == "pass":
        auth_overall = "acceptable: SPF and DMARC pass"
    elif dmarc_status in {"fail", "permerror"} or spf_status in {"fail", "softfail", "permerror"}:
        auth_overall = "suspicious: SPF or DMARC failed"

    lines = [
        f"Authentication overall: {auth_overall}",
        f"SPF: {spf_status}",
        f"DKIM: {dkim_status} (signature_present={bool(soc.get('dkim_signature_present'))})",
        f"DMARC: {dmarc_status}",
        f"Reply-To mismatch: {bool(soc.get('reply_to_mismatch'))}",
        f"Return-Path domain mismatch: {bool(soc.get('return_path_domain_mismatch'))}",
        f"Display name spoofing: {soc.get('display_name_spoofing') or 'none'}",
    ]

    attachments = soc.get("attachments") or []
    if not attachments:
        lines.append("Attachments: none")
    else:
        for att in attachments[:5]:
            lines.append(
                "Attachment: "
                "name=[ATTACHMENT_NAME] "
                f"ext={att.get('extension_from_filename') or '-'} "
                f"mime={att.get('content_type') or '-'} "
                f"magic={att.get('magic_detected_format') or '-'} "
                f"anomaly={att.get('anomaly') or 'none'}"
            )

    flags = soc.get("flags") or []
    high_medium = [
        flag for flag in flags
        if flag.get("level") in {"HIGH", "MEDIUM"}
    ]
    if auth_overall.startswith("acceptable"):
        high_medium = [
            flag for flag in high_medium
            if not (
                flag.get("field") == "DKIM"
                and any(token in (flag.get("message") or "").lower() for token in ["dkim none", "firma dkim assente"])
            )
        ]
    if high_medium:
        lines.append("SOC technical flags:")
        for flag in high_medium[:6]:
            lines.append(
                f"- {flag.get('level')} {flag.get('field')}: {_clip(flag.get('message', ''), 160)}"
            )
    else:
        lines.append("SOC technical flags: none high/medium")

    return lines


def build_fast_email_prompt(soc: dict) -> str:
    body, body_source_for_llm = _body_context_for_llm(soc)
    links = soc.get("links") or []
    link_reputation = soc.get("link_reputation") or {}
    link_reputation_summary = soc.get("link_reputation_summary") or "VirusTotal link reputation not available."
    subject = soc.get("subject") or "Nessun Oggetto"
    recipients = " ".join(
        str(soc.get(field) or "")
        for field in ("to", "cc", "delivered_to")
    )
    anonymized_subject = _anonymize_for_llm(subject)
    anonymized_body = _anonymize_for_llm(body)
    anonymized_sender = "[SENDER]" if soc.get("from_") else "Sconosciuto"
    anonymized_recipients = "[RECIPIENTS]" if recipients.strip() else "-"
    anonymized_technical_context = "\n".join(
        _anonymize_for_llm(line) for line in _technical_context_lines(soc)
    )
    risk_signals = _detect_text_risk_signals(subject, body, links)
    has_actionable_text_risk = _has_actionable_text_risk(risk_signals)

    link_lines = []
    if has_actionable_text_risk:
        for link in links[:5]:
            rep = link_reputation.get(link.get("url") or "", {})
            vt_status = rep.get("status", "unknown")
            ratio = rep.get("detection_ratio", "0 / 0")
            link_lines.append(
                f"- link_type={_anonymized_link_hint(link)} vt_status={vt_status} detections={ratio} "
                f"hint={_link_hint(link)}"
            )
    elif links:
        link_lines.append(
            f"- {len(links)} link(s) found, but link details are intentionally omitted because the body has no risky action request."
        )

    return "\n".join(
        [
            "Privacy note: subject, body, sender, recipients, URLs, IPs, phone numbers, email addresses, "
            "IBANs and account-like numbers are anonymized before being sent to the model.",
            f"Da: {_clip(anonymized_sender, 500) or 'Sconosciuto'}",
            f"Destinatari visibili: {_clip(anonymized_recipients, 500) or '-'}",
            f"Oggetto anonimizzato: {anonymized_subject}",
            f"Body source inspected by Phi-4: {body_source_for_llm}",
            f"Content precheck: {_content_precheck_label(has_actionable_text_risk)}",
            "",
            "Text risk signals detected:",
            "\n".join(f"- {signal}" for signal in risk_signals),
            "",
            "Corpo anonimizzato, includendo il testo visibile derivato dall'HTML quando presente:",
            _clip(anonymized_body, 2000),
            "",
            "Technical corroboration inputs - do not use these for the initial thesis:",
            "VirusTotal link reputation:",
            link_reputation_summary,
            "",
            "Technical context:",
            anonymized_technical_context,
            "",
            "Link estratti:",
            "\n".join(link_lines) if link_lines else "- nessuno",
        ]
    )


def stream_phi4_email_analysis(soc: dict, model: str = GITHUB_MODELS_MODEL, timeout: int = 90):
    if not _llm_enabled():
        yield {
            "status": "error",
            "message": (
                "Analisi LLM non disponibile: configura GITHUB_MODELS_TOKEN nei secrets "
                "per usare Phi-4 mini hosted."
            ),
            "text": "",
        }
        return

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                "Write an explanation in two strict steps, then answer with one concise English paragraph.\n"
                "Step 1 - Intent from anonymized subject/body: detect the language and explain the likely intent of the "
                "anonymized subject/body. Focus on what the message asks the recipient to do, such as paying, logging in, "
                "sharing credentials, opening a form, replying, accepting a promotion, handling an invoice, or completing "
                "a normal administrative task. Do not infer identities from placeholders such as [EMAIL], [URL], [IP], "
                "[PHONE], [IBAN], or [POSSIBLE_CARD_OR_ACCOUNT].\n"
                "Step 2 - FishSTOP technical context: use only the structured facts provided below, such as SPF, DKIM, "
                "DMARC, Return-Path mismatch, Reply-To mismatch, display-name spoofing, VirusTotal status, attachment "
                "anomalies, and SOC flags. Do not perform new technical analysis and do not invent risks that are not "
                "explicitly listed.\n"
                "Start exactly with one of these phrases:\n"
                "- The email provided is suspicious because\n"
                "- The email provided is not suspicious because\n"
                "Your first sentence must be driven by the intent of the anonymized subject/body and must mention that "
                "intent first. If the body contains a clear scam/phishing pattern, classify as suspicious even if some "
                "technical checks pass. If the body looks normal, classify as not suspicious unless the provided FishSTOP "
                "technical context is strong and corroborated by multiple independent indicators. "
                "A VirusTotal status of suspicious alone is not enough to override normal body content; treat it as a manual-check note. "
                "Only a clearly malicious VirusTotal result, or VirusTotal suspicious plus identity mismatch/authentication failure/attachment anomaly, "
                "may override a normal body.\n"
                "Do not mention sender IP, injection IP, relay IP, geolocation, routing path, full URLs, email addresses, "
                "phone numbers, IBANs, account numbers, or personal data as evidence.\n"
                "A link is not suspicious by itself. Use VirusTotal link reputation as the main link evidence, and "
                "mention a link as risky only if VirusTotal marks it malicious or the body asks for a risky action through it.\n"
                "Do not classify as suspicious only because a link is embedded, repeated, redirects through Google, "
                "points to a company website, or because there is one visible recipient.\n"
                "Promotions, newsletters, discounts, events, and normal commercial messages are not phishing just because they contain links.\n"
                "Treat lottery/prize/donation/inheritance claims, large unexpected money amounts, celebrity or CEO "
                "impersonation, vague profitable business or investment opportunities, international-collaboration lures, and requests "
                "to reply to a personal/free email address for more details as explicit scam indicators even if no money amount, "
                "credential request, or bank detail is shown yet.\n"
                "Normal administrative work requests, such as asking a manager to sign hours, timesheets, attendance "
                "sheets, or work records, are not suspicious unless they also ask for money, credentials, bank details, "
                "external forms, or urgent unusual action.\n"
                "In the answer, mention the body/content reason first. Then add one short clause explaining whether "
                "technical checks support, weaken, or do not materially change the content-based assessment. "
                "Do not lead with technical failures unless the body is empty or unreadable. End with: Please verify with your IT team.\n\n"
                f"{build_fast_email_prompt(soc)}"
            ),
        },
    ]
    yield from _stream_github_models(messages, model, timeout)


def _stream_github_models(messages: list[dict], model: str, timeout: int):
    """
    Chiama GitHub Models (Azure AI Inference, API OpenAI-compatible) in streaming
    SSE. Richiede un GitHub PAT con permesso 'Models: read' in GITHUB_MODELS_TOKEN.
    """
    headers = {
        "Authorization": f"Bearer {GITHUB_MODELS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 260,
    }

    chunks: list[str] = []
    try:
        with requests.post(
            GITHUB_MODELS_ENDPOINT, headers=headers, json=payload, stream=True, timeout=timeout
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content", "")
                if content:
                    chunks.append(content)
                    yield {"status": "stream", "text": "".join(chunks)}
    except requests.exceptions.Timeout:
        yield {"status": "error", "message": f"GitHub Models ha superato il timeout di {timeout} secondi.", "text": "".join(chunks)}
        return
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "?"
        yield {"status": "error", "message": f"GitHub Models HTTP {code}: verifica GITHUB_MODELS_TOKEN/GITHUB_MODELS_MODEL. ({exc})", "text": "".join(chunks)}
        return
    except requests.exceptions.RequestException as exc:
        yield {"status": "error", "message": f"GitHub Models non raggiungibile su {GITHUB_MODELS_ENDPOINT}: {exc}", "text": "".join(chunks)}
        return
    except Exception as exc:
        yield {"status": "error", "message": f"Errore durante la generazione con GitHub Models: {exc}", "text": "".join(chunks)}
        return

    yield {"status": "ok", "model": model, "text": "".join(chunks).strip()}
