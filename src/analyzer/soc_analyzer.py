"""
analyzer/soc_analyzer.py - Motore di analisi statica ed euristica per il SOC.

Main class:
  EmlSOCAnalyzer.analyze(eml_path) -> dict

Coordina tutti i sotto-moduli dell'analyzer:
  - received_parser  : parsing catena Received e Authentication-Results
  - link_extractor   : URL extraction from the body
  - lookalike        : rilevamento domini lookalike
  - attachment       : analisi allegati via magic bytes e hash
  - html_utils       : stripping HTML per body_clean
"""

import email
import ipaddress
import re
from email import policy
from typing import Optional

from .attachment      import analyze_attachment
from .body_context    import select_body_for_ai
from .html_utils      import strip_html
from .link_extractor  import extract_links
from .lookalike       import check_lookalike_domains, is_ip_url
from .received_parser import parse_received_hop, parse_auth_results


def _extract_domain(email_or_addr: str) -> str:
    """Returns the domain portion of an email address, lowercased."""
    m = re.search(r"@([\w.\-]+)", email_or_addr or "")
    return m.group(1).lower() if m else ""


def _registered_domain(domain: str) -> str:
    parts = (domain or "").lower().rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain or ""


def _same_registered_domain(left: str, right: str) -> bool:
    return bool(left and right and _registered_domain(left) == _registered_domain(right))


def _decode_text_part(part) -> str:
    payload = part.get_payload(decode=True)
    charset = part.get_content_charset() or "utf-8"

    if payload:
        decoded_candidates = []
        for candidate in (charset, "utf-8", "latin-1", "cp1252"):
            try:
                decoded = payload.decode(candidate, errors="replace")
            except LookupError:
                continue
            decoded_candidates.append(decoded)
        if decoded_candidates:
            return min(decoded_candidates, key=lambda value: value.count("\ufffd"))

    raw_payload = part.get_payload(decode=False)
    if isinstance(raw_payload, str):
        return raw_payload
    return ""


def _looks_like_html(value: str) -> bool:
    if not value:
        return False
    return bool(re.search(
        r"(?is)<\s*(?:!doctype\s+html|html|body|table|div|span|p|br|a|img|style|head)\b",
        value,
    ))


def _is_public_ip(value: str | None) -> bool:
    if not value:
        return False
    try:
        return ipaddress.ip_address(value.strip("[]")).is_global
    except ValueError:
        return False


class EmlSOCAnalyzer:
    """
    Parsa un file .eml grezzo e restituisce un report strutturato per il triage SOC.
    All logic is extracted dynamically from the email - no hardcoding
    legato a messaggi specifici.
    """

    def analyze(self, eml_path: str) -> dict:
        with open(eml_path, "rb") as f:
            raw_bytes = f.read()

        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        report: dict = {}
        report["raw_eml_bytes"] = raw_bytes

        # ── 1. Campi envelope ──────────────────────────────────────────────
        report["delivered_to"] = self._header(msg, "Delivered-To")
        report["to"]           = self._header(msg, "To")
        report["from_"]        = self._header(msg, "From")
        report["subject"]      = self._header(msg, "Subject")
        report["date"]         = self._header(msg, "Date")
        report["message_id"]   = self._header(msg, "Message-Id")
        report["importance"]   = self._header(msg, "Importance") or self._header(msg, "X-Priority")
        report["mime_version"] = self._header(msg, "MIME-Version")
        report["content_type"] = self._header(msg, "Content-Type")

        # ── 2. Return-Path / Errors-To / Reply-To ─────────────────────────
        report["return_path"] = self._header(msg, "Return-Path")
        report["errors_to"]   = self._header(msg, "Errors-To")
        reply_to              = self._header(msg, "Reply-To")
        report["reply_to"]    = reply_to

        from_addr  = self._extract_address(report["from_"])
        reply_addr = self._extract_address(reply_to)
        report["reply_to_mismatch"] = bool(
            reply_addr and from_addr and reply_addr.lower() != from_addr.lower()
        )

        return_path_addr   = self._extract_address(report["return_path"])
        return_path_domain = _extract_domain(return_path_addr or "") if return_path_addr else ""
        from_domain        = _extract_domain(from_addr or "") if from_addr else ""
        report["return_path_domain_mismatch"] = bool(
            return_path_domain and from_domain
            and not _same_registered_domain(return_path_domain, from_domain)
        )
        report["return_path_domain"] = return_path_domain

        # Display Name Spoofing: the display name contains an address different from the real sender
        display_name_email_match = None
        if report["from_"]:
            dn_match = re.match(r'^"?([^"<]+)"?\s*<', report["from_"])
            if dn_match:
                dn = dn_match.group(1).strip()
                embedded = re.search(r"[\w.+\-]+@[\w.\-]+", dn)
                if embedded:
                    embedded_addr = embedded.group(0).lower()
                    if from_addr and embedded_addr != from_addr.lower():
                        display_name_email_match = embedded_addr
        report["display_name_spoofing"] = display_name_email_match

        # ── 3. Metadata Google / routing ──────────────────────────────────
        report["x_google_smtp_source"] = self._header(msg, "X-Google-Smtp-Source")
        report["x_received"]           = self._header(msg, "X-Received")

        # ── 4. Header ARC ─────────────────────────────────────────────────
        report["arc_seal"]                   = self._header(msg, "ARC-Seal")
        report["arc_message_signature"]      = self._header(msg, "ARC-Message-Signature")
        report["arc_authentication_results"] = self._header(msg, "ARC-Authentication-Results")

        # ── 5. Catena Received ────────────────────────────────────────────
        raw_received = msg.get_all("Received") or []
        hops = [parse_received_hop(r) for r in raw_received]
        report["received_hops"]         = hops
        report["closest_to_recipient"]  = hops[0]  if hops else {}
        report["injection_server"]      = hops[1]  if len(hops) > 1 else {}
        report["closest_to_sender"]     = hops[-1] if hops else {}
        report["injection_sender_ip"]   = self._extract_spf_sender_ip(msg, hops)

        # ── 6. Received-SPF raw ───────────────────────────────────────────
        report["received_spf_raw"] = self._header(msg, "Received-SPF")

        # ── 7. Authentication-Results ─────────────────────────────────────
        auth_raw     = "\n".join(self._headers(msg, "Authentication-Results"))
        arc_auth_raw = "\n".join(self._headers(msg, "ARC-Authentication-Results"))
        report["authentication_results_raw"] = auth_raw
        report["auth_results"]     = parse_auth_results(auth_raw)
        report["arc_auth_results"] = parse_auth_results(arc_auth_raw)

        # ── 8. Firma DKIM ─────────────────────────────────────────────────
        dkim_headers = self._headers(msg, "DKIM-Signature")
        report["dkim_signature_present"] = bool(dkim_headers)
        report["dkim_signature_raw"]     = "\n".join(dkim_headers)

        # ── 9. Body e allegati ────────────────────────────────────────────
        body_parts       = []
        html_parts       = []
        attachments_info = []

        for part in msg.walk():
            ct       = part.get_content_type()
            disp     = str(part.get("Content-Disposition") or "")
            encoding = str(part.get("Content-Transfer-Encoding") or "").lower().strip()
            filename = part.get_filename() or ""
            is_attach = "attachment" in disp.lower()

            if is_attach or filename:
                raw_payload = part.get_payload(decode=True)
                attachments_info.append(analyze_attachment(
                    filename=filename,
                    content_type=ct,
                    encoding=encoding,
                    raw_payload=raw_payload,
                ))
            elif ct == "text/plain":
                text = _decode_text_part(part)
                if text and text.strip():
                    if _looks_like_html(text):
                        html_parts.append(text)
                    else:
                        body_parts.append(text)
            elif ct == "text/html":
                text = _decode_text_part(part)
                if text and text.strip():
                    html_parts.append(text)

        combined_html = "\n".join(html_parts)
        html_clean = strip_html(combined_html) if combined_html else ""
        raw_body = "\n".join(body_parts) if body_parts else combined_html
        report["body"]      = raw_body.strip()
        report["body_html"] = combined_html.strip() if html_parts else None
        report["body_html_clean"] = html_clean

        if body_parts:
            report["body_clean"] = re.sub(r"\n{3,}", "\n\n", report["body"]).strip()
        else:
            report["body_clean"] = html_clean

        report["body_source"]        = "text/plain" if body_parts else ("text/html" if html_parts else "empty")
        report["html_strip_applied"] = (not bool(body_parts)) and bool(html_parts)
        report["attachments"]        = attachments_info
        report.update(select_body_for_ai(report["body_clean"]))
        report["body_clean_full"] = report["body_clean"]
        report["body_extracted"] = report.get("body_ai") or report["body_clean"]

        # ── 10. Link e lookalike ──────────────────────────────────────────
        report["links"] = extract_links(
            body_plain=report["body"],
            body_html=report.get("body_html") or "",
        )
        report["lookalike_alerts"] = check_lookalike_domains(report["links"])

        # ── 11. Flag SOC ──────────────────────────────────────────────────
        report["flags"] = self._build_flags(report)

        return report

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _header(msg, name: str) -> Optional[str]:
        val = msg.get(name)
        if val is None:
            return None
        return re.sub(r"\s+", " ", str(val)).strip()

    @staticmethod
    def _headers(msg, name: str) -> list[str]:
        return [
            re.sub(r"\s+", " ", str(val)).strip()
            for val in (msg.get_all(name) or [])
            if val is not None
        ]

    @staticmethod
    def _extract_address(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        m = re.search(r"<([^>]+)>", raw)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"[\w.+\-]+@[\w.\-]+", raw)
        return m2.group(0).strip() if m2 else None

    @staticmethod
    def _extract_spf_sender_ip(msg, hops: list) -> str | None:
        """
        Estrae l'IP corretto per la verifica SPF live.

        Priority:
          1. client-ip= in the LAST Received-SPF (closest to the sender)
          2. smtp.remote-ip= in Authentication-Results
          3. Primo IP pubblico nell'ultimo hop Received
          4. Fallback: sender_ip dall'hop [1]
        """
        all_rcvd_spf = msg.get_all("Received-SPF") or []
        for rcvd_spf in reversed(all_rcvd_spf):
            m = re.search(r"client-ip=([\d.a-fA-F:]+)", str(rcvd_spf), re.IGNORECASE)
            if m and _is_public_ip(m.group(1)):
                return m.group(1)

        auth = str(msg.get("Authentication-Results") or "")
        m = re.search(r"smtp\.remote-ip=([\d.]+)", auth, re.IGNORECASE)
        if m and _is_public_ip(m.group(1)):
            return m.group(1)

        if hops:
            last_hop = hops[-1]
            for ip in (last_hop.get("all_ips") or []):
                if _is_public_ip(ip):
                    return ip

        return (hops[1].get("sender_ip") if len(hops) > 1 else None)

    @staticmethod
    def _build_flags(report: dict) -> list[dict]:
        flags = []

        def flag(level: str, field: str, message: str):
            flags.append({"level": level, "field": field, "message": message})

        # SPF
        spf = report["auth_results"].get("SPF") or report["arc_auth_results"].get("SPF")
        if spf:
            if spf["status"] != "pass":
                flag("HIGH", "SPF", f"SPF {spf['status'].upper()} - domain does not authorize the sender server")
        else:
            flag("MEDIUM", "SPF", "No SPF result found in headers")

        # DKIM
        if not report["dkim_signature_present"]:
            flag("MEDIUM", "DKIM", "DKIM signature missing from headers")
        dkim = report["auth_results"].get("DKIM") or report["arc_auth_results"].get("DKIM")
        if dkim and dkim["status"] != "pass":
            flag("HIGH", "DKIM", f"DKIM {dkim['status'].upper()}")

        # DMARC
        dmarc = report["auth_results"].get("DMARC") or report["arc_auth_results"].get("DMARC")
        if dmarc and dmarc["status"] not in ("pass", "bestguesspass"):
            flag("HIGH", "DMARC", f"DMARC {dmarc['status'].upper()}")
        elif not dmarc:
            flag("LOW", "DMARC", "No DMARC policy detected in headers")

        # Reply-To mismatch
        if report["reply_to_mismatch"]:
            flag("HIGH", "Reply-To",
                 f"Reply-To ({report['reply_to']}) differs da From ({report['from_']}) - possible harvesting")

        # Return-Path domain mismatch
        if report.get("return_path_domain_mismatch"):
            _from_domain = _extract_domain(
                EmlSOCAnalyzer._extract_address(report.get("from_") or "") or ""
            )
            flag(
                "MEDIUM", "Return-Path",
                f"The Return-Path domain (`{report['return_path_domain']}`) differs from "
                f"the From domain (`{_from_domain}`). This can be legitimate for bulk senders, "
                "but it should be reviewed with authentication and link evidence."
            )
        elif report.get("return_path") and not report.get("return_path_domain"):
            flag("LOW", "Return-Path", "Return-Path present but domain cannot be extracted")

        # HTML stripping applicato
        if report.get("html_strip_applied"):
            flag("INFO", "Body",
                 "Email body is pure HTML: tags removed before AI analysis. "
                 "Possible hidden text obfuscation in tags.")

        # Display Name Spoofing
        dns_val = report.get("display_name_spoofing")
        if dns_val:
            flag(
                "HIGH", "Display Name",
                f"The Display Name in the From field contains an email address (`{dns_val}`). "
                "Classic Display Name Spoofing technique: email clients show "
                "the embedded address instead of the real sender."
            )

        # Injection server
        inj = report.get("injection_server", {})
        if inj.get("sender_ip"):
            flag("INFO", "Received",
                 f"Injection server: {inj.get('sender_domain') or inj.get('from_host', '?')} "
                 f"[{inj['sender_ip']}] - verify IP/domain reputation")

        # Anomalie allegati
        for att in report.get("attachments", []):
            if att.get("anomaly"):
                flag("HIGH", "Attachment",
                     f"'{att['filename']}': {att['anomaly']}")
            if att.get("magic_bytes_hex"):
                flag("INFO", "Attachment",
                     f"'{att['filename']}': magic bytes {att['magic_bytes_hex'][:8]}... "
                     f"-> detected format: {att['magic_detected_format'] or 'unknown'}")

        # Link anomalie: IP-direct e lookalike
        for lnk in report.get("links", []):
            if lnk.get("is_ip"):
                flag(
                    "HIGH", "Link",
                    "URL with bare IP detected: `" + lnk["url"] + "` - avoids DNS lookup, "
                    "typical of phishing or C2",
                )

            if lnk.get("display_mismatch"):
                flag(
                    "HIGH", "Link",
                    "The link shows `" + (lnk.get("display_host") or "?")
                    + "` but points to `" + (lnk.get("host") or "?")
                    + "`: possible masked link in HTML.",
                )
            if lnk.get("is_possible_shortener"):
                flag(
                    "MEDIUM", "Link",
                    "Possible short link/redirector without whitelist: `" + lnk["url"]
                    + "` - " + (lnk.get("shortener_reason") or "suspicious pattern"),
                )
        for alert in report.get("lookalike_alerts", []):
            technique_label = {
                "edit_distance": "Edit-distance",
                "homoglyph":     "Unicode homoglyphs",
                "unicode_homoglyph": "Unicode homoglyphs in domain",
                "unicode_domain": "Unicode characters in domain",
                "punycode_idna": "Punycode/IDNA domain",
                "punycode_homograph": "Punycode homograph attack",
                "typosquatting": "Typosquatting",
            }.get(alert["technique"], alert["technique"])
            matched_brand = alert.get("matched_brand") or "-"
            if matched_brand == "-":
                message = technique_label + ": `" + alert["host"] + "` - " + alert["detail"]
            else:
                message = (
                    technique_label + ": `" + alert["host"] + "` looks like `"
                    + matched_brand + "` - " + alert["detail"]
                )
            flag("HIGH", "Lookalike Domain", message)

        return flags
