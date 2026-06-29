"""
tests/test_spf_normalization.py — Unit test per le funzioni di normalizzazione SPF.

Questi test NON fanno chiamate di rete: testano solo la logica pura in
validators/spf.py (parsing IP, normalizzazione risultati, mappatura verdict).
Girano sempre, anche offline e senza pyspf installato.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.validators.spf import (
    _normalize_ip,
    _normalize_spf_result,
    _meta,
    _extract_address,
    _extract_domain,
)


# ── _normalize_ip ──────────────────────────────────────────────────────────

class TestNormalizeIp:
    """Verifica che l'IP venga normalizzato correttamente prima di passarlo a pyspf."""

    def test_plain_ipv4(self):
        assert _normalize_ip("1.2.3.4") == "1.2.3.4"

    def test_ipv4_with_brackets(self):
        """MTA Outlook scrivono [1.2.3.4] nel campo Received."""
        assert _normalize_ip("[1.2.3.4]") == "1.2.3.4"

    def test_ipv4_with_whitespace(self):
        assert _normalize_ip("  209.85.220.69  ") == "209.85.220.69"

    def test_ipv4_mapped_ipv6_with_prefix(self):
        """Formato: 'IPv6:::ffff:1.2.3.4' → estrae IPv4."""
        assert _normalize_ip("IPv6:::ffff:1.2.3.4") == "1.2.3.4"

    def test_ipv4_mapped_ipv6_without_prefix(self):
        """Formato: '::ffff:192.168.1.1' → estrae IPv4."""
        assert _normalize_ip("::ffff:192.168.1.1") == "192.168.1.1"

    def test_ipv4_mapped_real_sender(self):
        """IP reale da spam_email: ::ffff:209.85.220.69 → 209.85.220.69."""
        assert _normalize_ip("::ffff:209.85.220.69") == "209.85.220.69"

    def test_pure_ipv6_with_prefix(self):
        """Formato: 'IPv6:2001:db8::1' → rimuove solo il prefisso 'IPv6:'."""
        assert _normalize_ip("IPv6:2001:db8::1") == "2001:db8::1"

    def test_pure_ipv6_without_prefix(self):
        assert _normalize_ip("2001:db8::1") == "2001:db8::1"

    def test_real_google_ip_from_test7(self):
        """IP reale estratto da test7.eml (catena Received)."""
        assert _normalize_ip("209.85.128.71") == "209.85.128.71"

    def test_invalid_ip_returned_as_is(self):
        """IP non valido: lasciato invariato per far fallire pyspf con messaggio chiaro."""
        result = _normalize_ip("not-an-ip")
        assert result == "not-an-ip"


# ── _normalize_spf_result ─────────────────────────────────────────────────

class TestNormalizeSpfResult:
    """Verifica la normalizzazione delle varianti non-standard di pyspf."""

    def test_standard_pass(self):
        assert _normalize_spf_result("pass") == "pass"

    def test_standard_fail(self):
        assert _normalize_spf_result("fail") == "fail"

    def test_standard_softfail(self):
        assert _normalize_spf_result("softfail") == "softfail"

    def test_standard_neutral(self):
        assert _normalize_spf_result("neutral") == "neutral"

    def test_standard_none(self):
        assert _normalize_spf_result("none") == "none"

    def test_standard_permerror(self):
        assert _normalize_spf_result("permerror") == "permerror"

    def test_tilde_pass_alias(self):
        """Alcune versioni di pyspf restituiscono '~pass' invece di 'softfail'."""
        assert _normalize_spf_result("~pass") == "softfail"

    def test_hardfail_alias(self):
        assert _normalize_spf_result("hardfail") == "fail"

    def test_unknown_alias(self):
        assert _normalize_spf_result("unknown") == "none"

    def test_unknown_value_passthrough(self):
        """Valori non mappati passano invariati."""
        assert _normalize_spf_result("something_new") == "something_new"


# ── _meta (verdict unificato) ─────────────────────────────────────────────

class TestMeta:
    """Verifica il campo 'verdict' e 'authenticated' per ogni status SPF."""

    def test_pass_is_authenticated(self):
        verdict, label, auth = _meta("pass")
        assert verdict == "pass"
        assert auth is True
        assert "✅" in label

    def test_fail_is_not_authenticated(self):
        verdict, label, auth = _meta("fail")
        assert verdict == "fail"
        assert auth is False
        assert "❌" in label

    def test_softfail_is_warn_not_authenticated(self):
        """Softfail NON è un'autenticazione — non mostrare verde in UI."""
        verdict, label, auth = _meta("softfail")
        assert verdict == "warn"
        assert auth is False

    def test_neutral_is_warn(self):
        verdict, label, auth = _meta("neutral")
        assert verdict == "warn"
        assert auth is False

    def test_none_is_unknown(self):
        verdict, _, auth = _meta("none")
        assert verdict == "unknown"
        assert auth is False

    def test_permerror_is_fail(self):
        """permerror (record malformato) è trattato come fallimento."""
        verdict, _, auth = _meta("permerror")
        assert verdict == "fail"
        assert auth is False

    def test_temperror_is_warn(self):
        """temperror (DNS temporaneo) è un warning, non un fail definitivo."""
        verdict, _, auth = _meta("temperror")
        assert verdict == "warn"
        assert auth is False

    def test_all_statuses_have_label(self):
        """Tutti gli status noti devono avere una severity_label non vuota."""
        for status in ["pass", "fail", "softfail", "neutral", "none", "permerror", "temperror"]:
            _, label, _ = _meta(status)
            assert label, f"severity_label vuota per status={status!r}"


# ── _extract_address e _extract_domain ───────────────────────────────────

class TestExtractHelpers:
    """Verifica i parser di indirizzo email."""

    def test_extract_address_with_display_name(self):
        raw = '"Formazione Cefla" <drive-shares-noreply@google.com>'
        assert _extract_address(raw) == "drive-shares-noreply@google.com"

    def test_extract_address_bare(self):
        assert _extract_address("user@example.com") == "user@example.com"

    def test_extract_address_none(self):
        assert _extract_address(None) is None

    def test_extract_address_empty(self):
        assert _extract_address("") is None

    def test_extract_domain_from_display_name(self):
        raw = '"Eugenio De Rosa" <eugeniomaria.derosa@studio.unibo.it>'
        assert _extract_domain(raw) == "studio.unibo.it"

    def test_extract_domain_bare(self):
        assert _extract_domain("user@cefla.it") == "cefla.it"

    def test_extract_domain_empty(self):
        assert _extract_domain("") == ""