import json
import socket
import urllib.error
import urllib.request


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi4-mini:latest"

SYSTEM_MESSAGE = (
    "You are a SOC phishing and scam text classifier. "
    "Your job is to decide whether the email text looks suspicious, not to produce a generic summary. "
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
        "wire transfer", "bonifico", "pagamento", "payment", "fattura", "invoice",
        "saldo", "cambio conto", "nuovo conto", "new bank details",
    )
    credential_words = (
        "password", "credenzial", "credentials", "login", "accesso", "sign in",
        "account verification", "verifica account", "mfa", "otp",
    )
    urgency_words = (
        "urgente", "urgent", "entro oggi", "immediately", "as soon as possible",
        "scade", "deadline", "overdue", "sospeso", "blocked", "bloccato",
        "azione richiesta", "action required",
    )
    form_words = (
        "forms.gle", "docs.google.com/forms", "forms.office.com", "google form",
        "questionario", "survey", "modulo",
    )

    has_money = any(word in text for word in money_words)
    has_credentials = any(word in text for word in credential_words)
    has_urgency = any(word in text for word in urgency_words)
    has_form = any(word in text or word in urls for word in form_words)

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

    neutral_admin_words = (
        "firma ore", "firmare le ore", "firmarmi le ore", "timesheet",
        "attendance sheet", "foglio ore", "cartellino", "presenze", "ore lavorate",
    )
    if any(word in text for word in neutral_admin_words):
        neutral_notes.append("normal administrative work request wording is present")

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
            f"Classification baseline: {'SUSPICIOUS only if the risky request is explicit' if has_actionable_text_risk else 'NOT SUSPICIOUS'}",
            "",
            "Text risk signals detected:",
            "\n".join(f"- {signal}" for signal in risk_signals),
            "",
            "VirusTotal link reputation:",
            link_reputation_summary,
            "",
            "Link estratti:",
            "\n".join(link_lines) if link_lines else "- nessuno",
            "",
            "Corpo:",
            _clip(body, 2000),
        ]
    )


def stream_phi4_email_analysis(soc: dict, model: str = OLLAMA_MODEL, timeout: int = 90):
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                "Check whether the email is suspicious based on the text/body and visible recipients only.\n"
                "Do not use SPF, DKIM, DMARC, routing headers, Reply-To, Return-Path, attachment reputation, "
                "or other technical authentication data for this answer.\n"
                "Start exactly with one of these phrases:\n"
                "- The email provided is suspicious because\n"
                "- The email provided is not suspicious because\n"
                "Only mention risks that are explicitly visible in the body, subject, links, or recipients. "
                "Do not invent IBAN changes, payment redirection, credentials, Google Forms, or urgency.\n"
                "Follow the Classification baseline in the prompt. If it says NOT SUSPICIOUS, you must start with "
                "\"The email provided is not suspicious because\" unless the body explicitly asks for money, "
                "credentials, bank details, sensitive form submission, or unusual urgent action.\n"
                "A link is not suspicious by itself. Do not classify an email as suspicious only because it has "
                "external links, many links, tracking links, document links, meeting links, newsletter links, "
                "or common business/collaboration links.\n"
                "For link risk, use the VirusTotal link reputation summary as the main evidence. If VirusTotal "
                "says links are clean, not_found, skipped, or unknown, do not call the link dangerous unless the "
                "email body explicitly asks for a risky action. If VirusTotal says malicious or suspicious, mention "
                "that reputation result as evidence.\n"
                "Do not say an email is suspicious because a link is embedded, repeated, identical in subject/body, "
                "redirects through Google services, or points to a company website. Those are not phishing evidence "
                "unless paired with a risky request in the body.\n"
                "Do not say an email is suspicious because there is only one visible recipient, because the recipient "
                "is the user's own address, or because recipients are not clearly justified.\n"
                "Mention a link as evidence only when the body asks the recipient to do a risky action through "
                "that link, such as entering credentials, filling sensitive personal/bank data, changing payment "
                "details, approving a payment, or acting under unusual urgency.\n"
                "If the detected text risk signals say that no explicit money, credential, bank-detail, "
                "sensitive-form, or unusual-urgency wording was found, default to not suspicious unless the body "
                "clearly contradicts that.\n"
                "Treat money requests, urgency, bank coordinate/IBAN changes, payment redirection, invoices, "
                "credential requests, account verification forms, and Google Forms asking for sensitive data "
                "as suspicious indicators.\n"
                "Normal administrative work requests, such as asking a manager to sign hours, timesheets, "
                "attendance sheets, or work records, are not suspicious unless they also ask for money, "
                "credentials, bank details, external forms, or urgent unusual action.\n"
                "End with a practical recommendation such as: Please verify with your IT team.\n\n"
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
            "num_ctx": 2048,
            "num_predict": 220,
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
