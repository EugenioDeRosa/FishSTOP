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
    per eseguire codice dell'email o consentire click verso risorse esterne.
    Rimuove quindi script, iframe, form, embed, event handler inline e tutte le
    destinazioni href/src/action.
    """
    if not html or not html.strip():
        return "<p><em>Nessun HTML disponibile.</em></p>"

    if not _BS4_AVAILABLE:
        safe_html = re.sub(r"(?is)<(script|style|head|iframe|object|embed|form|button|input|meta)\b[^>]*>.*?</\1>", " ", html)
        safe_html = re.sub(r"(?is)<(script|style|head|iframe|object|embed|form|button|input|meta)\b[^>]*?/?>", " ", safe_html)
        safe_html = re.sub(r"(?is)<img\b[^>]*>", "[Immagine remota bloccata]", safe_html)
        safe_html = re.sub(r"""(?is)\son[a-z0-9_-]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", "", safe_html)
        safe_html = re.sub(r"""(?is)\s(?:href|src|xlink:href|action)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", "", safe_html)
        safe_html = re.sub(r"(?i)\b(?:javascript|vbscript|data)\s*:", "#", safe_html)
        escaped = html_lib.escape(safe_html)
        return f"<pre style='white-space: pre-wrap'>{escaped}</pre>"

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "iframe", "object", "embed", "form", "input", "button", "meta"]):
        tag.decompose()

    for img in soup.find_all("img"):
        src = str(img.get("src") or "").strip()
        alt = str(img.get("alt") or "").strip()
        label = "Immagine remota bloccata"
        if alt:
            label = f"{label}: {alt[:120]}"
        elif src:
            label = f"{label}: sorgente esterna rimossa"
        placeholder = soup.new_tag("div")
        placeholder.string = label
        placeholder["style"] = (
            "display:block; box-sizing:border-box; min-height:96px; padding:18px; "
            "margin:8px 0; border:1px dashed #d0d7de; border-radius:6px; "
            "background:#f6f8fa; color:#57606a; text-align:center; "
            "font-family:Arial,sans-serif; font-size:14px;"
        )
        placeholder["title"] = "Immagine remota non caricata per evitare tracking o contenuti esterni."
        img.replace_with(placeholder)

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            attr_l = attr.lower()
            value = tag.attrs.get(attr)
            if attr_l.startswith("on"):
                del tag.attrs[attr]
                continue
            if attr_l in {"href", "src", "xlink:href", "action"}:
                del tag.attrs[attr]
                if tag.name == "a":
                    tag.attrs["title"] = "Link rimosso per sicurezza: usa la box Link presenti nella mail."
                    tag.attrs["style"] = "color: inherit; text-decoration: none; cursor: default;"

    body = soup.body.decode_contents() if soup.body else str(soup)
    return f"""
    <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.45; color: #24292f;">
      {body}
    </div>
    """


def sanitize_html_for_js_preview(html: str) -> str:
    """
    Restituisce HTML renderizzabile con JavaScript attivo, ma con link non cliccabili.

    Questa modalita serve solo per osservare il rendering dinamico del messaggio:
    gli href vengono rimossi e i click sui link sono bloccati dentro l'iframe.
    """
    if not html or not html.strip():
        return "<p><em>Nessun HTML disponibile.</em></p>"

    link_guard = """
    <style>
      a, a:visited {
        color: inherit !important;
        text-decoration: none !important;
        cursor: default !important;
        pointer-events: none !important;
      }
    </style>
    <script>
    (function () {
      function disableLinks(root) {
        var scope = root || document;
        scope.querySelectorAll('a').forEach(function (link) {
          link.removeAttribute('href');
          link.removeAttribute('target');
          link.setAttribute('aria-disabled', 'true');
          link.setAttribute('title', 'Link disabilitato nella preview.');
        });
      }

      document.addEventListener('click', function (event) {
        if (event.target && event.target.closest && event.target.closest('a')) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      }, true);

      document.addEventListener('DOMContentLoaded', function () {
        disableLinks(document);
        var observer = new MutationObserver(function () { disableLinks(document); });
        observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true });
      });
    }());
    </script>
    """
    return f"""
    {link_guard}
    <div style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.45; color: #24292f;">
      {html}
    </div>
    """
