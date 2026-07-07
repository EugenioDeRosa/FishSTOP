"""
analyzer/html_utils.py — Pulizia e normalizzazione HTML per l'analisi email.

Espone:
  - strip_html(html)  : converte HTML grezzo in testo pulito

Gli attaccanti inseriscono tag o commenti HTML invisibili in mezzo alle parole
(es. Pa<!-- x -->ypal) per aggirare i filtri basati su stringhe. Senza
stripping, BERT riceve token sporchi e le regex sui link non trovano le URL reali.
"""

import html as html_lib
import re

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


def strip_html(html: str) -> str:
    """
    Converte HTML grezzo in testo pulito adatto all'analisi AI e ai controlli
    testuali.

    Strategia (in ordine):
      1. BeautifulSoup (lxml > html.parser come backend) per un parsing robusto
         che gestisce HTML malformato, encoding errors e tag annidati.
      2. Rimozione di <script> e <style> prima dell'estrazione del testo, per
         evitare che codice JS o CSS venga passato al modello.
      3. Separatore '\\n' tra i tag per preservare la struttura dei paragrafi.
      4. Fallback regex se BeautifulSoup non è installato: rimuove tutti i tag
         con un pattern greedy-safe e decodifica le entity HTML principali.
    """
    if not html or not html.strip():
        return ""

    if _BS4_AVAILABLE:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
    else:
        # Fallback regex
        html = re.sub(r"(?is)<(script|style|head|iframe|object|embed|form|button|input|meta)\b[^>]*>.*?</\1>", " ", html)
        html = re.sub(r"(?is)<(script|style|head|iframe|object|embed|form|button|input|meta)\b[^>]*?/?>", " ", html)
        text = re.sub(r"<[^>]+>", " ", html)
        text = (text
                .replace("&amp;",  "&")
                .replace("&lt;",   "<")
                .replace("&gt;",   ">")
                .replace("&nbsp;", " ")
                .replace("&quot;", '"')
                .replace("&#39;",  "'"))

    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def sanitize_html_for_preview(html: str) -> str:
    """
    Restituisce HTML renderizzabile nella dashboard senza contenuti attivi.

    La preview serve all'analista per capire layout e testo del messaggio, non
    per eseguire codice dell'email. Rimuove quindi script, iframe, form, embed,
    event handler inline e URL javascript/data potenzialmente pericolosi.
    """
    if not html or not html.strip():
        return "<p><em>Nessun HTML disponibile.</em></p>"

    if not _BS4_AVAILABLE:
        safe_html = re.sub(r"(?is)<(script|style|head|iframe|object|embed|form|button|input|meta)\b[^>]*>.*?</\1>", " ", html)
        safe_html = re.sub(r"(?is)<(script|style|head|iframe|object|embed|form|button|input|meta)\b[^>]*?/?>", " ", safe_html)
        safe_html = re.sub(r"(?i)\b(?:javascript|vbscript|data)\s*:", "#", safe_html)
        escaped = html_lib.escape(safe_html)
        return f"<pre style='white-space: pre-wrap'>{escaped}</pre>"

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "iframe", "object", "embed", "form", "input", "button", "meta"]):
        tag.decompose()

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            attr_l = attr.lower()
            value = tag.attrs.get(attr)
            if attr_l.startswith("on"):
                del tag.attrs[attr]
                continue
            if attr_l in {"href", "src", "xlink:href", "action"}:
                raw_value = " ".join(value) if isinstance(value, list) else str(value or "")
                lowered = raw_value.strip().lower()
                if lowered.startswith(("javascript:", "data:", "vbscript:", "file:")):
                    del tag.attrs[attr]

    body = soup.body.decode_contents() if soup.body else str(soup)
    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.45; color: #24292f;">
      {body}
    </div>
    """
