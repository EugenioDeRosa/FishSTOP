"""
validators/spf.py — Verifica SPF per email in ingresso.

Valutazione SPF nativa basata su RFC 7208, implementata con sole
dipendenze dnspython + ipaddress. Non usa pyspf/pydns: quella libreria
è abbandonata dal 2015 e ha bug noti su record TXT multi-stringa e su
catene di "include" annidate (es. Google Workspace, _spf.google.com),
che possono produrre "fail" spuri anche quando il record è in realtà
corretto e l'MTA di destinazione valuta "pass".

Funzione pubblica:
  check_spf(resolver, sender_ip, mail_from, helo_domain) → dict
"""

import ipaddress
import re
import dns.resolver
from typing import List, Optional, Tuple, Union

IPAddr = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

# Limite di lookup DNS "consumanti" (include, a, mx, ptr, exists, redirect)
# imposto da RFC 7208 §4.6.4. Oltre questo limite: permerror.
_MAX_DNS_LOOKUPS = 10
_MAX_RECURSION_DEPTH = 10

_DNS_TIMEOUT = 5.0
_DNS_LIFETIME = 5.0

_QUALIFIERS = {"+": "pass", "-": "fail", "~": "softfail", "?": "neutral"}


# --------------------------------------------------------------------------
# Utility di base
# --------------------------------------------------------------------------

def _apply_dns_timeout(resolver: dns.resolver.Resolver) -> None:
    if resolver.timeout is None or resolver.timeout > _DNS_TIMEOUT:
        resolver.timeout = _DNS_TIMEOUT
    if resolver.lifetime is None or resolver.lifetime > _DNS_LIFETIME:
        resolver.lifetime = _DNS_LIFETIME


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except (ValueError, TypeError):
        return False


def _extract_address(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = re.search(r"<([^>]+)>", raw)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m2 = re.search(r"[\w.+\-]+@[\w.\-]+", raw)
    return m2.group(0).strip() if m2 else None


def _extract_domain(email: str) -> str:
    m = re.search(r"@([\w.\-]+)", email)
    return m.group(1).lower() if m else ""


class _PermError(Exception):
    """Errore di sintassi/struttura del record SPF (RFC 7208 §2.6.7)."""


class _TempError(Exception):
    """Errore DNS temporaneo durante la valutazione (RFC 7208 §2.6.6)."""


# --------------------------------------------------------------------------
# Risoluzione DNS di basso livello
# --------------------------------------------------------------------------

def _query_txt_records(resolver: dns.resolver.Resolver, domain: str) -> List[str]:
    """Ritorna tutti i record TXT del dominio, riassemblando correttamente
    eventuali record divisi in più character-string (multi-stringa)."""
    _apply_dns_timeout(resolver)
    try:
        answers = resolver.resolve(domain, "TXT")
    except dns.resolver.NXDOMAIN:
        return []
    except dns.resolver.NoAnswer:
        return []
    except Exception as exc:
        raise _TempError(f"Errore DNS su TXT {domain}: {type(exc).__name__}: {exc}")

    records = []
    for rdata in answers:
        # rdata.strings contiene i singoli segmenti byte; vanno concatenati
        # SENZA separatori per ricostruire il record originale (RFC 7208 §3.3).
        if hasattr(rdata, "strings"):
            full = b"".join(rdata.strings).decode("utf-8", errors="replace")
        else:
            full = rdata.to_text().strip('"')
        records.append(full)
    return records


def _get_spf_record(resolver: dns.resolver.Resolver, domain: str) -> str:
    """Trova l'unico record v=spf1 per il dominio. RFC 7208 §4.5: zero
    record → 'none' (gestito dal chiamante); più di un record → permerror."""
    txts = _query_txt_records(resolver, domain)
    spf_records = [t for t in txts if t.lower().startswith("v=spf1")]
    if len(spf_records) > 1:
        raise _PermError(f"Più record SPF trovati per {domain} (RFC 7208 §4.5)")
    return spf_records[0] if spf_records else ""


def _resolve_a(resolver: dns.resolver.Resolver, domain: str) -> List[ipaddress.IPv4Address]:
    _apply_dns_timeout(resolver)
    try:
        answers = resolver.resolve(domain, "A")
        return [ipaddress.ip_address(r.to_text()) for r in answers]
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return []
    except Exception as exc:
        raise _TempError(f"Errore DNS su A {domain}: {type(exc).__name__}: {exc}")


def _resolve_aaaa(resolver: dns.resolver.Resolver, domain: str) -> List[ipaddress.IPv6Address]:
    _apply_dns_timeout(resolver)
    try:
        answers = resolver.resolve(domain, "AAAA")
        return [ipaddress.ip_address(r.to_text()) for r in answers]
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return []
    except Exception as exc:
        raise _TempError(f"Errore DNS su AAAA {domain}: {type(exc).__name__}: {exc}")


def _resolve_mx(resolver: dns.resolver.Resolver, domain: str) -> List[str]:
    _apply_dns_timeout(resolver)
    try:
        answers = resolver.resolve(domain, "MX")
        return [str(r.exchange).rstrip(".") for r in answers]
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
        return []
    except Exception as exc:
        raise _TempError(f"Errore DNS su MX {domain}: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Macro expansion minimale (RFC 7208 §7) — copre i casi più comuni
# --------------------------------------------------------------------------

def _expand_macros(template: str, ip: IPAddr, sender: str, helo: str, domain: str) -> str:
    sender = sender or "postmaster@unknown"
    local_part, _, sender_domain = sender.partition("@")
    repl = {
        "i": str(ip),
        "s": sender,
        "l": local_part,
        "o": sender_domain or domain,
        "d": domain,
        "h": helo or domain,
        "v": "in-addr" if ip.version == 4 else "ip6",
    }

    def _sub(m: "re.Match") -> str:
        letter = m.group(1).lower()
        return repl.get(letter, m.group(0))

    out = re.sub(r"%\{([a-zA-Z])\}", _sub, template)
    out = out.replace("%%", "%").replace("%_", " ").replace("%-", "%20")
    return out


# --------------------------------------------------------------------------
# CIDR matching
# --------------------------------------------------------------------------

def _ip_matches(ip: IPAddr, candidate: IPAddr, prefix4: Optional[int], prefix6: Optional[int]) -> bool:
    if ip.version != candidate.version:
        return False
    prefix = prefix4 if ip.version == 4 else prefix6
    if prefix is None:
        prefix = 32 if ip.version == 4 else 128
    net = ipaddress.ip_network(f"{candidate}/{prefix}", strict=False)
    return ip in net


def _parse_dual_cidr(spec: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parsa una "dual-cidr-length" RFC 7208 §4.5.1: '/24' (solo ip4),
    '/24/64' (ip4 e ip6), '//64' (solo ip6, ip4 omesso).
    """
    if not spec:
        return None, None
    if not spec.startswith("/"):
        raise _PermError(f"CIDR non valido: {spec!r}")
    rest = spec[1:]  # rimuove il primo "/" separatore obbligatorio
    parts = rest.split("/", 1)
    try:
        p4 = int(parts[0]) if parts[0] else None
        p6 = int(parts[1]) if len(parts) > 1 and parts[1] else None
    except ValueError:
        raise _PermError(f"CIDR non valido: {spec!r}")
    return p4, p6


# --------------------------------------------------------------------------
# Parsing e valutazione dei termini del record
# --------------------------------------------------------------------------

_MECH_RE = re.compile(
    r"^(?P<qual>[+\-~?]?)"
    r"(?P<name>all|include|a|mx|ptr|ip4|ip6|exists)"
    r"(?::(?P<value>[^/\s]+))?"
    r"(?P<cidr>/[^\s]*)?$",
    re.IGNORECASE,
)

_MOD_RE = re.compile(r"^(?P<name>redirect|exp|[a-zA-Z][a-zA-Z0-9_\-.]*)=(?P<value>.+)$")


class _Evaluator:
    def __init__(self, resolver: dns.resolver.Resolver, ip: IPAddr, sender: str, helo: str):
        self.resolver = resolver
        self.ip = ip
        self.sender = sender
        self.helo = helo
        self.lookups = 0

    def _count_lookup(self) -> None:
        self.lookups += 1
        if self.lookups > _MAX_DNS_LOOKUPS:
            raise _PermError(
                f"Superato il limite di {_MAX_DNS_LOOKUPS} lookup DNS previsto da RFC 7208 §4.6.4"
            )

    def evaluate(self, domain: str, depth: int = 0) -> Tuple[str, str]:
        if depth > _MAX_RECURSION_DEPTH:
            raise _PermError("Catena di include/redirect troppo profonda")

        record = _get_spf_record(self.resolver, domain)
        if not record:
            return "none", f"Nessun record SPF per {domain}"

        terms = record.split()[1:]  # salta "v=spf1"
        redirect_target = None

        for term in terms:
            mod = _MOD_RE.match(term)
            if mod and mod.group("name").lower() not in ("a", "mx", "ptr", "ip4", "ip6", "all", "include", "exists"):
                name = mod.group("name").lower()
                value = mod.group("value")
                if name == "redirect":
                    redirect_target = _expand_macros(value, self.ip, self.sender, self.helo, domain)
                # "exp=" e altri modificatori sconosciuti: ignorati (RFC 7208 §6)
                continue

            mech = _MECH_RE.match(term)
            if not mech:
                raise _PermError(f"Termine SPF non valido: {term!r}")

            qualifier = _QUALIFIERS.get(mech.group("qual") or "+", "pass")
            name = mech.group("name").lower()
            value = mech.group("value")
            cidr = mech.group("cidr") or ""

            matched = self._eval_mechanism(name, value, cidr, domain, depth)
            if matched:
                return qualifier, f"Matched '{term}' on {domain}"

        if redirect_target:
            return self.evaluate(redirect_target, depth + 1)

        return "neutral", f"Nessun meccanismo soddisfatto in {domain} (default neutral)"

    def _eval_mechanism(self, name: str, value: Optional[str], cidr: str, domain: str, depth: int) -> bool:
        if name == "all":
            return True

        if name == "ip4":
            if not value:
                raise _PermError("Meccanismo ip4 senza valore")
            p4, _ = _parse_dual_cidr(cidr)
            try:
                candidate = ipaddress.ip_address(value)
            except ValueError:
                raise _PermError(f"IP non valido in ip4: {value!r}")
            return self.ip.version == 4 and _ip_matches(self.ip, candidate, p4, None)

        if name == "ip6":
            if not value:
                raise _PermError("Meccanismo ip6 senza valore")
            _, p6 = _parse_dual_cidr(cidr)
            try:
                candidate = ipaddress.ip_address(value)
            except ValueError:
                raise _PermError(f"IP non valido in ip6: {value!r}")
            return self.ip.version == 6 and _ip_matches(self.ip, candidate, None, p6)

        if name == "a":
            target = _expand_macros(value, self.ip, self.sender, self.helo, domain) if value else domain
            self._count_lookup()
            p4, p6 = _parse_dual_cidr(cidr)
            for cand in _resolve_a(self.resolver, target):
                if _ip_matches(self.ip, cand, p4, None):
                    return True
            for cand in _resolve_aaaa(self.resolver, target):
                if _ip_matches(self.ip, cand, None, p6):
                    return True
            return False

        if name == "mx":
            target = _expand_macros(value, self.ip, self.sender, self.helo, domain) if value else domain
            self._count_lookup()
            p4, p6 = _parse_dual_cidr(cidr)
            for exchange in _resolve_mx(self.resolver, target):
                for cand in _resolve_a(self.resolver, exchange):
                    if _ip_matches(self.ip, cand, p4, None):
                        return True
                for cand in _resolve_aaaa(self.resolver, exchange):
                    if _ip_matches(self.ip, cand, None, p6):
                        return True
            return False

        if name == "exists":
            if not value:
                raise _PermError("Meccanismo exists senza valore")
            target = _expand_macros(value, self.ip, self.sender, self.helo, domain)
            self._count_lookup()
            return len(_resolve_a(self.resolver, target)) > 0

        if name == "ptr":
            # Deprecato da RFC 7208 §5.5: viene contato come lookup ma non
            # implementiamo la verifica PTR completa (sconsigliata dalla RFC
            # stessa per affidabilità/performance). Non genera mai match.
            self._count_lookup()
            return False

        if name == "include":
            if not value:
                raise _PermError("Meccanismo include senza valore")
            target = _expand_macros(value, self.ip, self.sender, self.helo, domain)
            self._count_lookup()
            sub_result, _ = self.evaluate(target, depth + 1)
            # RFC 7208 §5.2: solo un risultato "pass" nell'include produce
            # match; "fail/softfail/neutral" → il meccanismo include non
            # scatta (si prosegue con il termine successivo); "none" o
            # "permerror" nell'include → permerror per l'intera valutazione.
            if sub_result == "pass":
                return True
            if sub_result in ("none", "permerror"):
                raise _PermError(f"include:{target} ha restituito {sub_result}")
            return False

        raise _PermError(f"Meccanismo SPF sconosciuto: {name!r}")


# --------------------------------------------------------------------------
# API pubblica
# --------------------------------------------------------------------------

def check_spf(
    resolver: dns.resolver.Resolver,
    sender_ip: str,
    mail_from: str,
    helo_domain: str = "",
) -> dict:
    """
    Valutazione SPF nativa (RFC 7208) senza dipendenza da pyspf.

    Parameters
    ----------
    resolver    : istanza dns.resolver.Resolver condivisa
    sender_ip   : IP del server di iniezione (dalla catena Received, l'hop
                  esterno reale, non un hop interno dell'MTA ricevente)
    mail_from   : indirizzo envelope sender (header Return-Path)
    helo_domain : dominio HELO/EHLO — usato come dominio di valutazione
                  quando mail_from è vuoto/null (bounce, RFC 7208 §2.4)

    Returns
    -------
    {
      "status"    : "pass" | "fail" | "softfail" | "neutral" |
                    "none" | "permerror" | "temperror" | "error",
      "record"    : str,
      "domain"    : str,
      "sender_ip" : str,
      "mail_from" : str,
      "message"   : str,
      "library"   : "native-rfc7208"
    }
    """
    extracted = _extract_address(mail_from)
    is_null_sender = not extracted
    addr = extracted or ""
    domain = _extract_domain(addr) if addr else ""

    if is_null_sender:
        domain = helo_domain.lower().strip() if helo_domain else ""

    base = {
        "sender_ip": sender_ip,
        "mail_from": addr or "<>",
        "domain":    domain,
        "record":    "",
        "library":   "native-rfc7208",
    }

    if not sender_ip or not _is_valid_ip(sender_ip):
        return {**base, "status": "error",
                "message": f"sender_ip mancante o non valido: {sender_ip!r}"}

    if not domain:
        return {**base, "status": "error",
                "message": "mail_from e helo_domain mancanti — impossibile determinare il dominio da valutare"}

    ip = ipaddress.ip_address(sender_ip)
    evaluator = _Evaluator(resolver, ip, addr or f"postmaster@{domain}", helo_domain or domain)

    try:
        record = _get_spf_record(resolver, domain)
    except _TempError as exc:
        return {**base, "status": "temperror", "message": str(exc)}
    except _PermError as exc:
        return {**base, "status": "permerror", "message": str(exc)}

    base["record"] = record

    try:
        status, message = evaluator.evaluate(domain)
    except _PermError as exc:
        return {**base, "status": "permerror", "message": str(exc)}
    except _TempError as exc:
        return {**base, "status": "temperror", "message": str(exc)}
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore inatteso: {type(exc).__name__}: {exc}"}

    return {**base, "status": status, "message": message}