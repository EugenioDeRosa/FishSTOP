import json
import socket
import urllib.error
import urllib.request


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi4-mini:latest"

SYSTEM_MESSAGE = (
    "You are a SOC phishing and scam classifier. "
    "You must always decide whether the email looks suspicious by judging the subject/body first. "
    "Only after that content thesis is formed may you use technical indicators as supporting or weakening evidence. "
    "Never let SPF, DKIM, DMARC, links, attachments, sender IPs, or reputation data create the initial thesis. "
    "Answer with one concise English paragraph and no JSON."
)


def _clip(value: str, limit: int) -> str:
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[...troncato...]"


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
                f"name={att.get('filename') or '(unnamed)'} "
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
    body = soc.get("body_ai") or soc.get("body_extracted") or soc.get("body_clean") or soc.get("body") or ""
    links = soc.get("links") or []
    link_reputation = soc.get("link_reputation") or {}
    link_reputation_summary = soc.get("link_reputation_summary") or "VirusTotal link reputation not available."
    subject = soc.get("subject") or "Nessun Oggetto"
    recipients = " ".join(
        str(soc.get(field) or "")
        for field in ("to", "cc", "delivered_to")
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
                f"- host={link.get('host') or '-'} vt_status={vt_status} detections={ratio} "
                f"url={_clip(link.get('url', ''), 180)} hint={_link_hint(link)}"
            )
    elif links:
        link_lines.append(
            f"- {len(links)} link(s) found, but link details are intentionally omitted because the body has no risky action request."
        )

    return "\n".join(
        [
            f"Da: {soc.get('from_') or 'Sconosciuto'}",
            f"Destinatari visibili: {_clip(recipients, 500) or '-'}",
            f"Oggetto: {subject}",
            f"Content intent baseline: {'risky content pattern detected' if has_actionable_text_risk else 'no risky content pattern detected'}",
            "",
            "Text risk signals detected:",
            "\n".join(f"- {signal}" for signal in risk_signals),
            "",
            "Corpo:",
            _clip(body, 2000),
            "",
            "Technical corroboration inputs - do not use these for the initial thesis:",
            "VirusTotal link reputation:",
            link_reputation_summary,
            "",
            "Technical context:",
            "\n".join(_technical_context_lines(soc)),
            "",
            "Link estratti:",
            "\n".join(link_lines) if link_lines else "- nessuno",
        ]
    )


def stream_phi4_email_analysis(soc: dict, model: str = OLLAMA_MODEL, timeout: int = 90):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                "Analyze the email in two strict steps, then answer with one concise English paragraph.\n"
                "Step 1 - Mandatory body/subject thesis: FIRST inspect only the subject and body. "
                "Before considering any technical field, explicitly check the subject/body for urgency or pressure, "
                "money requests, payment instructions, bank coordinates such as IBAN/SWIFT/account numbers, invoices or refunds, "
                "promotions or prizes, credential or login requests, sensitive forms, requests to click links, impersonation "
                "of brands/executives/internal departments, and any unusual requested action. "
                "This first thesis must ignore SPF, DKIM, DMARC, Return-Path, Reply-To, VirusTotal, attachments, flags, "
                "routing, sender IPs, geolocation, and every other technical signal.\n"
                "Step 2 - Technical corroboration only: ONLY AFTER the body/content thesis, use SPF/DKIM/DMARC, Return-Path mismatch, "
                "Reply-To mismatch, display-name spoofing, VirusTotal link reputation, attachment anomalies, and SOC flags to say "
                "whether the technical evidence supports, weakens, or does not materially change the thesis from Step 1.\n"
                "Start exactly with one of these phrases:\n"
                "- The email provided is suspicious because\n"
                "- The email provided is not suspicious because\n"
                "Your first sentence must be driven by the body/content thesis and must mention the content reason first. "
                "If the body contains a clear scam/phishing "
                "pattern, classify as suspicious even if some technical checks pass. If the body looks normal, classify as "
                "not suspicious unless the technical evidence is strong and corroborated by multiple independent indicators. "
                "A VirusTotal status of suspicious alone is not enough to override normal body content; treat it as a manual-check note. "
                "Only a clearly malicious VirusTotal result, or VirusTotal suspicious plus identity mismatch/authentication failure/attachment anomaly, "
                "may override a normal body.\n"
                "Do not invent risks. Only mention body risks explicitly visible in the subject/body/links/recipients, "
                "and only mention technical risks explicitly listed in Technical context.\n"
                "Do not mention sender IP, injection IP, relay IP, geolocation, or routing path as evidence. "
                "In modern email these values are often missing, internal, or misleading.\n"
                "A link is not suspicious by itself. Use VirusTotal link reputation as the main link evidence, and "
                "mention a link as risky only if VirusTotal marks it malicious or the body asks for a risky action through it.\n"
                "Do not classify as suspicious only because a link is embedded, repeated, redirects through Google, "
                "points to a company website, or because there is one visible recipient.\n"
                "Promotions, newsletters, discounts, events, and normal commercial messages are not phishing just because they contain links.\n"
                "Treat lottery/prize/donation/inheritance claims, large unexpected money amounts, celebrity or CEO "
                "impersonation, and requests to reply to validate an email address as explicit money-scam indicators.\n"
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
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096,
            "num_predict": 260,
        },
    }
    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line:
                    continue
                event = json.loads(raw_line.decode("utf-8", errors="replace"))
                content = (event.get("message") or {}).get("content", "")
                if content:
                    chunks.append(content)
                    yield {"status": "stream", "text": "".join(chunks)}
                if event.get("done"):
                    break
    except TimeoutError:
        yield {
            "status": "error",
            "message": f"Ollama ha superato il timeout di {timeout} secondi.",
            "text": "".join(chunks),
        }
        return
    except socket.timeout:
        yield {
            "status": "error",
            "message": f"Ollama ha superato il timeout di {timeout} secondi.",
            "text": "".join(chunks),
        }
        return
    except urllib.error.URLError as exc:
        yield {
            "status": "error",
            "message": f"Ollama non raggiungibile su {OLLAMA_CHAT_URL}: {exc}",
            "text": "".join(chunks),
        }
        return
    except Exception as exc:
        yield {
            "status": "error",
            "message": f"Errore durante la generazione con Ollama: {exc}",
            "text": "".join(chunks),
        }
        return

    yield {
        "status": "ok",
        "model": model,
        "text": "".join(chunks).strip(),
    }
