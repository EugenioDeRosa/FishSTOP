import json
import socket
import urllib.error
import urllib.request


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi4-mini:latest"

SYSTEM_MESSAGE = (
    "Sei un analista di sicurezza esperto. Rispondi SEMPRE e SOLO in italiano. "
    "Sii ultra-conciso: genera un resoconto di MASSIMO 4 o 5 righe. "
    "Valuta phishing, coerenza mittente/contenuto, urgenza, richieste rischiose, link e allegati. "
    "Evita introduzioni, convenevoli e ignora totalmente tag di ragionamento come <think>."
)


def _clip(value: str, limit: int) -> str:
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[...troncato...]"


def build_fast_email_prompt(soc: dict) -> str:
    body = soc.get("body_clean") or soc.get("body") or ""
    links = soc.get("links") or []
    attachments = soc.get("attachments") or []
    flags = soc.get("flags") or []

    link_lines = []
    for link in links[:5]:
        link_lines.append(
            f"- host={link.get('host') or '-'} url={_clip(link.get('url', ''), 180)}"
        )

    attachment_lines = []
    for att in attachments[:5]:
        attachment_lines.append(
            f"- file={att.get('filename') or '(senza nome)'} ext={att.get('extension_from_filename') or '-'} "
            f"mime={att.get('content_type') or '-'} anomalia={att.get('anomaly') or '-'}"
        )

    flag_lines = []
    for flag in flags[:6]:
        flag_lines.append(
            f"- {flag.get('level', 'INFO')} {flag.get('field', 'Signal')}: {_clip(flag.get('message', ''), 130)}"
        )

    return "\n".join(
        [
            f"Da: {soc.get('from_') or 'Sconosciuto'}",
            f"Reply-To: {soc.get('reply_to') or '-'}",
            f"Return-Path: {soc.get('return_path') or '-'}",
            f"Oggetto: {soc.get('subject') or 'Nessun Oggetto'}",
            f"SPF/DKIM/DMARC: {((soc.get('auth_results') or {}) or {})}",
            "",
            "Flag tecnici gia rilevati:",
            "\n".join(flag_lines) if flag_lines else "- nessuno",
            "",
            "Link estratti:",
            "\n".join(link_lines) if link_lines else "- nessuno",
            "",
            "Allegati:",
            "\n".join(attachment_lines) if attachment_lines else "- nessuno",
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
                "Analizza la seguente email per verificare se si tratta di phishing. "
                "Considera anche eventuali link, allegati e flag tecnici gia estratti.\n\n"
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
