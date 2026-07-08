"""
analyzer/lookalike.py — Rilevamento domini lookalike anti-phishing.

Espone:
  - levenshtein(a, b)             : distanza di edit tra due stringhe
  - normalize_homoglyphs(domain)  : sostituisce omoglifi Unicode con ASCII
  - strip_public_suffix(domain)   : isola il dominio di secondo livello
  - is_ip_url(host)               : True se host è un IP nudo
  - check_lookalike_domains(...)  : analisi euristica completa (edit distance,
                                    omoglifi, typosquatting)
"""

import ipaddress
import unicodedata

from .constants import KNOWN_BRANDS, HOMOGLYPH_MAP


KNOWN_URL_SHORTENERS: set[str] = {
    "t.co",
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "ow.ly",
    "buff.ly",
    "cutt.ly",
    "is.gd",
    "s.id",
    "rebrand.ly",
    "lnkd.in",
    "youtu.be",
    "aka.ms",
}

MIN_EDIT_DISTANCE_SLD_LEN = 4


def levenshtein(a: str, b: str) -> int:
    """Distanza di edit (Levenshtein) tra due stringhe, O(n·m) spazio O(n)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def normalize_homoglyphs(domain: str) -> str:
    """
    Normalizza un dominio sostituendo i caratteri omoglifi Unicode con
    il loro equivalente ASCII. Gestisce anche la forma NFC/NFKC.
    """
    domain = unicodedata.normalize("NFKC", domain.lower())
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in domain)


def decode_punycode_domain(host: str) -> str:
    """
    Decodifica le label IDNA/punycode (xn--) in Unicode.

    Se una label e' malformata, viene lasciata invariata per non interrompere
    l'analisi dell'email.
    """
    labels = []
    for label in (host or "").lower().rstrip(".").split("."):
        if label.startswith("xn--"):
            try:
                labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                labels.append(label)
        else:
            labels.append(label)
    return ".".join(labels)


def _has_punycode_label(host: str) -> bool:
    return any(label.lower().startswith("xn--") for label in (host or "").split("."))


def _has_non_ascii(value: str) -> bool:
    return any(ord(ch) > 127 for ch in value or "")


def strip_public_suffix(domain: str) -> str:
    """
    Ritorna il 'registered domain' (etichette - TLD) in forma semplificata.
    Non usa una libreria PSL completa: rimuove solo l'ultimo label (TLD)
    per confronti di distanza più significativi.

    Esempio: mail.paypa1.com → paypa1
    """
    parts = domain.rstrip(".").split(".")
    if len(parts) >= 2:
        return parts[-2]
    return domain


def _registered_domain(domain: str) -> str:
    parts = (domain or "").lower().rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain or ""


def _is_known_shortener(host: str) -> bool:
    registered = _registered_domain(host)
    return registered in KNOWN_URL_SHORTENERS


def is_ip_url(host: str) -> bool:
    """True se l'host è un indirizzo IPv4 o IPv6."""
    try:
        ipaddress.ip_address((host or "").strip("[]"))
        return True
    except ValueError:
        return False


def check_lookalike_domains(
    links: list[dict],
    known_brands: list[str] | None = None,
    edit_distance_threshold: int = 2,
) -> list[dict]:
    """
    Per ogni link controlla se il dominio assomiglia a un brand noto
    usando tre tecniche combinate:

      1. Levenshtein distance sull'SLD (Second-Level Domain) ≤ threshold
      2. Omografia Unicode — caratteri visivamente identici ad ASCII
      3. Typosquatting patterns — inserimento/duplicazione consonanti,
         sostituzione 0↔o / 1↔l / rn↔m, aggiunta prefissi ingannevoli

    Restituisce solo gli alert (lista vuota = nessun sospetto trovato).

    Ogni alert:
      {
        "url"           : str
        "host"          : str
        "matched_brand" : str
        "technique"     : "edit_distance" | "homoglyph" | "typosquatting"
        "detail"        : str
        "edit_distance" : int | None
      }
    """
    brands = known_brands if known_brands is not None else KNOWN_BRANDS
    alerts: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    def _alert(url: str, host: str, brand: str, technique: str,
                detail: str, dist: int | None = None) -> None:
        key = (host, brand, technique)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        alerts.append({
            "url":           url,
            "host":          host,
            "matched_brand": brand,
            "technique":     technique,
            "detail":        detail,
            "edit_distance": dist,
        })

    for link in links:
        host = link["host"]
        url  = link["url"]

        if not host or is_ip_url(host) or _is_known_shortener(host):
            continue

        host_norm = normalize_homoglyphs(host)
        host_sld  = strip_public_suffix(host_norm)
        decoded_host = decode_punycode_domain(host)
        decoded_norm = normalize_homoglyphs(decoded_host)

        if _has_punycode_label(host) and decoded_host != host.lower():
            detail = f"Dominio punycode `{host}` decodificato come `{decoded_host}`."
            if decoded_norm != decoded_host:
                detail += f" Normalizzazione omoglifi: `{decoded_norm}`."
            _alert(url, host, "-", "punycode_idna", detail)
            continue

        if _has_non_ascii(host):
            if host_norm != host.lower():
                detail = (
                    f"Dominio `{host}` contiene caratteri Unicode omoglifi; "
                    f"normalizzato diventa `{host_norm}`."
                )
                _alert(url, host, "-", "unicode_homoglyph", detail)
            else:
                _alert(
                    url,
                    host,
                    "-",
                    "unicode_domain",
                    f"Dominio `{host}` contiene caratteri Unicode non ASCII.",
                )
            continue

        for brand in brands:
            brand_norm = normalize_homoglyphs(brand)
            brand_sld  = strip_public_suffix(brand_norm)

            # Salta i match esatti
            if host_norm == brand_norm or host_norm.endswith("." + brand_norm):
                break

            # Tecnica 1: Levenshtein sull'SLD
            if (
                len(host_sld) >= MIN_EDIT_DISTANCE_SLD_LEN
                and len(brand_sld) >= MIN_EDIT_DISTANCE_SLD_LEN
                and abs(len(host_sld) - len(brand_sld)) <= edit_distance_threshold
            ):
                dist = levenshtein(host_sld, brand_sld)
                if 0 < dist <= edit_distance_threshold:
                    _alert(url, host, brand, "edit_distance",
                           f"SLD `{host_sld}` dista {dist} edit da `{brand_sld}` "
                           f"(brand: {brand})", dist)
                    continue

            # Tecnica 2: Omografia Unicode
            if host_norm != host.lower() and levenshtein(host_norm, brand_norm) <= edit_distance_threshold:
                _alert(url, host, brand, "homoglyph",
                       f"Il dominio `{host}` contiene caratteri Unicode omoglifi "
                       f"che lo rendono visivamente simile a `{brand}`")
                continue

            # Tecnica 3: Punycode / IDNA homograph
            if _has_punycode_label(host) and decoded_host != host.lower():
                decoded_sld = strip_public_suffix(decoded_norm)
                decoded_matches_brand = (
                    decoded_norm == brand_norm
                    or decoded_norm.endswith("." + brand_norm)
                    or decoded_sld == brand_sld
                    or levenshtein(decoded_sld, brand_sld) <= edit_distance_threshold
                )
                if decoded_matches_brand:
                    _alert(
                        url,
                        host,
                        brand,
                        "punycode_homograph",
                        f"Dominio punycode `{host}` decodificato come `{decoded_host}`; "
                        f"dopo normalizzazione omoglifi risulta simile a `{brand}`",
                    )
                    continue

            # Tecnica 4: Typosquatting — prefissi ingannatori
            for prefix in ("secure-", "login-", "verify-", "account-",
                           "update-", "signin-", "support-", "my-", "auth-"):
                candidate = host_norm.removeprefix("www.")
                if candidate.startswith(prefix):
                    inner = candidate[len(prefix):]
                    if levenshtein(strip_public_suffix(inner), brand_sld) <= 1:
                        _alert(url, host, brand, "typosquatting",
                               f"Prefisso ingannatorio `{prefix}` davanti a un dominio "
                               f"simile a `{brand}`")
                        break

            # Typosquatting — sostituzione caratteri (0↔o, 1↔i/l, rn↔m)
            subst = (host_sld
                     .replace("0", "o").replace("1", "i").replace("1", "l")
                     .replace("rn", "m").replace("vv", "w"))
            if subst != host_sld and levenshtein(subst, brand_sld) == 0:
                _alert(url, host, brand, "typosquatting",
                       f"Sostituzione caratteri (`{host_sld}` → `{subst}`) "
                       f"replica `{brand_sld}` (brand: {brand})")

    return alerts
