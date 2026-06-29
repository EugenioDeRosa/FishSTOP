"""
tests/test_analyzer.py — Test di integrazione per EmlSOCAnalyzer.

Usa le email reali nelle fixtures per verificare che il parsing degli
header, degli allegati e dei link funzioni correttamente.
NON fa chiamate di rete (tutto è analisi statica del file .eml).
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyzer import EmlSOCAnalyzer

# Istanza condivisa — EmlSOCAnalyzer è stateless
analyzer = EmlSOCAnalyzer()


# ── Helper ────────────────────────────────────────────────────────────────

def _flags_by_category(report: dict, category: str) -> list[dict]:
    """Restituisce i flag SOC di una categoria specifica."""
    return [f for f in report.get("flags", []) if f.get("category") == category]

def _flag_levels(report: dict) -> set[str]:
    return {f["level"] for f in report.get("flags", [])}


# ══════════════════════════════════════════════════════════════════════════
# 1. spam_email.eml — Google Form con Reply-To diverso dal From
# ══════════════════════════════════════════════════════════════════════════

class TestSpamGoogleForm:
    """
    Email legittima (Google Forms per conto di cefla.it) ma con Reply-To
    che punta a un dominio diverso dal From. FishStop deve rilevarlo.
    """

    @pytest.fixture(autouse=True)
    def _load(self, eml_spam_google_form):
        self.report = analyzer.analyze(eml_spam_google_form)

    def test_from_is_google(self):
        assert "google.com" in self.report["from_"].lower()

    def test_reply_to_is_cefla(self):
        assert "cefla.it" in (self.report.get("reply_to") or "").lower()

    def test_reply_to_mismatch_detected(self):
        """Reply-To (cefla.it) ≠ From (google.com) deve generare un flag."""
        mismatch_flags = [
            f for f in self.report.get("flags", [])
            if "reply" in f.get("detail", "").lower() or "reply" in f.get("category", "").lower()
        ]
        assert len(mismatch_flags) > 0, (
            "Nessun flag generato per il mismatch Reply-To vs From. "
            f"Flag presenti: {self.report.get('flags', [])}"
        )

    def test_return_path_is_google_bounce(self):
        rp = self.report.get("return_path") or ""
        assert "bounces.google.com" in rp.lower() or "doclist" in rp.lower()

    def test_links_extracted(self):
        """L'email contiene link a docs.google.com e workspace.google.com."""
        links = self.report.get("links", [])
        hosts = [l["host"] for l in links]
        assert any("google.com" in h for h in hosts), f"Nessun link google trovato: {hosts}"

    def test_no_attachments(self):
        assert self.report.get("attachments", []) == []

    def test_received_hops_count(self):
        hops = self.report.get("received_hops", [])
        assert len(hops) >= 2


# ══════════════════════════════════════════════════════════════════════════
# 2. spam_email_-_2.eml — Email interna con allegati immagine
# ══════════════════════════════════════════════════════════════════════════

class TestInternalForwardWithAttachments:
    """Email interna cefla.it inoltrata. SPF/DKIM pass, allegati jpg+png."""

    @pytest.fixture(autouse=True)
    def _load(self, eml_internal_forward):
        self.report = analyzer.analyze(eml_internal_forward)

    def test_from_is_cefla(self):
        assert "cefla.it" in self.report["from_"].lower()

    def test_two_attachments(self):
        atts = self.report.get("attachments", [])
        assert len(atts) == 2, f"Attesi 2 allegati, trovati {len(atts)}: {[a['filename'] for a in atts]}"

    def test_attachment_jpg_magic(self):
        """image001.jpg deve avere magic bytes JPEG (FFD8FF)."""
        atts = self.report.get("attachments", [])
        jpg = next((a for a in atts if a["filename"].endswith(".jpg")), None)
        assert jpg is not None, "Allegato .jpg non trovato"
        assert jpg["magic_bytes_hex"] is not None
        assert jpg["magic_bytes_hex"].upper().startswith("FFD8FF"), (
            f"Magic bytes JPEG attesi (FFD8FF…), trovato: {jpg['magic_bytes_hex']}"
        )

    def test_attachment_png_magic(self):
        """image002.png deve avere magic bytes PNG (89504E47)."""
        atts = self.report.get("attachments", [])
        png = next((a for a in atts if a["filename"].endswith(".png")), None)
        assert png is not None, "Allegato .png non trovato"
        assert png["magic_bytes_hex"].upper().startswith("89504E47"), (
            f"Magic bytes PNG attesi (89504E47…), trovato: {png['magic_bytes_hex']}"
        )

    def test_attachment_no_anomaly(self):
        """Gli allegati devono essere coerenti (tipo file = magic bytes)."""
        for att in self.report.get("attachments", []):
            assert att["anomaly"] is None, (
                f"Anomalia inattesa su {att['filename']}: {att['anomaly']}"
            )

    def test_sha256_hash_present(self):
        """Ogni allegato deve avere l'hash SHA-256 per threat intelligence."""
        for att in self.report.get("attachments", []):
            assert att.get("hash_sha256"), f"SHA-256 mancante per {att['filename']}"
            assert len(att["hash_sha256"]) == 64  # 32 byte = 64 hex chars


# ══════════════════════════════════════════════════════════════════════════
# 3. test.eml — SPF FAIL + DKIM pass (caso ibrido unibo/Gmail)
# ══════════════════════════════════════════════════════════════════════════

class TestSpfFailDkimPass:
    """
    Email unibo.it inviata via Gmail → SPF fail (IP Gmail non autorizzato
    da unibo.it), DKIM pass. Caso reale di routing legittimo ma anomalo.
    """

    @pytest.fixture(autouse=True)
    def _load(self, eml_spf_fail_dkim_pass):
        self.report = analyzer.analyze(eml_spf_fail_dkim_pass)

    def test_from_is_unibo(self):
        assert "unibo.it" in self.report["from_"].lower()

    def test_received_spf_raw_present(self):
        """L'header Received-SPF deve essere estratto."""
        spf_raw = self.report.get("received_spf_raw") or ""
        assert spf_raw, "received_spf_raw non estratto"
        assert "fail" in spf_raw.lower() or "Fail" in spf_raw

    def test_attachment_pdf_present(self):
        atts = self.report.get("attachments", [])
        pdfs = [a for a in atts if a["filename"].lower().endswith(".pdf")]
        assert len(pdfs) >= 1, f"Nessun PDF trovato. Allegati: {[a['filename'] for a in atts]}"

    def test_pdf_magic_bytes(self):
        """Il PDF deve iniziare con %PDF- (hex: 255044462D)."""
        atts = self.report.get("attachments", [])
        pdf = next((a for a in atts if a["filename"].lower().endswith(".pdf")), None)
        if pdf and pdf.get("magic_bytes_hex"):
            assert pdf["magic_bytes_hex"].upper().startswith("255044462D"), (
                f"Magic bytes PDF attesi (255044462D…), trovato: {pdf['magic_bytes_hex']}"
            )

    def test_many_received_hops(self):
        """test.eml ha 8 hop Received — catena lunga tipica di relay multipli."""
        hops = self.report.get("received_hops", [])
        assert len(hops) >= 6, f"Attesi ≥6 hop, trovati {len(hops)}"

    def test_spf_flag_generated(self):
        """Un SPF fail deve generare almeno un flag HIGH o MEDIUM."""
        spf_flags = [
            f for f in self.report.get("flags", [])
            if "spf" in f.get("detail", "").lower() or "spf" in f.get("category", "").lower()
        ]
        assert len(spf_flags) > 0 or any(
            f["level"] in ("HIGH", "MEDIUM") for f in self.report.get("flags", [])
        ), "Nessun flag per SPF fail"


# ══════════════════════════════════════════════════════════════════════════
# 4. test2.eml — Email minimale senza SPF/DKIM, allegato HTML
# ══════════════════════════════════════════════════════════════════════════

class TestMinimalEmail:
    """Email senza header di autenticazione. Allegato .html (bookmarks)."""

    @pytest.fixture(autouse=True)
    def _load(self, eml_minimal):
        self.report = analyzer.analyze(eml_minimal)

    def test_from_present(self):
        assert self.report.get("from_")

    def test_no_spf_header(self):
        spf_raw = self.report.get("received_spf_raw") or ""
        assert not spf_raw, f"received_spf_raw inaspettato: {spf_raw!r}"

    def test_html_attachment_detected(self):
        atts = self.report.get("attachments", [])
        html_atts = [a for a in atts if a["filename"].lower().endswith(".html")]
        assert len(html_atts) >= 1

    def test_html_attachment_content_type(self):
        atts = self.report.get("attachments", [])
        html_att = next((a for a in atts if ".html" in a["filename"].lower()), None)
        if html_att:
            assert "html" in html_att["content_type"].lower()

    def test_single_received_hop(self):
        hops = self.report.get("received_hops", [])
        assert len(hops) <= 2, f"Troppi hop per email minimale: {len(hops)}"


# ══════════════════════════════════════════════════════════════════════════
# 5. test5_G.eml — Subject con encoding ISO-8859-1
# ══════════════════════════════════════════════════════════════════════════

class TestIso8859Subject:
    """
    Subject codificato: =?iso-8859-1?Q?Re:_Attivit=E0_tirocinio?=
    Deve essere decodificato come 'Re: Attività tirocinio'.
    """

    @pytest.fixture(autouse=True)
    def _load(self, eml_iso8859_subject):
        self.report = analyzer.analyze(eml_iso8859_subject)

    def test_subject_decoded(self):
        subject = self.report.get("subject") or ""
        assert subject, "Subject vuoto"
        # Deve contenere 'tirocinio' e non la sequenza raw =E0
        assert "tirocinio" in subject.lower(), f"Subject non decodificato: {subject!r}"
        assert "=E0" not in subject and "=e0" not in subject, (
            f"Subject non decodificato da ISO-8859-1: {subject!r}"
        )

    def test_from_is_cefla(self):
        assert "cefla.it" in self.report.get("from_", "").lower()


# ══════════════════════════════════════════════════════════════════════════
# 6. test6_G.eml — Email con 2 PDF e 1 immagine (~WRD0000.jpg)
# ══════════════════════════════════════════════════════════════════════════

class TestMultiPdfAttachments:
    """3 allegati: 2 PDF (certificati) e 1 immagine JPEG nominata .jpg."""

    @pytest.fixture(autouse=True)
    def _load(self, eml_multi_pdf):
        self.report = analyzer.analyze(eml_multi_pdf)

    def test_three_attachments(self):
        atts = self.report.get("attachments", [])
        assert len(atts) == 3, f"Attesi 3 allegati, trovati {len(atts)}: {[a['filename'] for a in atts]}"

    def test_two_pdf_attachments(self):
        atts = self.report.get("attachments", [])
        pdfs = [a for a in atts if a["filename"].lower().endswith(".pdf")]
        assert len(pdfs) == 2

    def test_pdf_magic_bytes_correct(self):
        """Entrambi i PDF devono partire con %PDF- (25 50 44 46 2D)."""
        atts = self.report.get("attachments", [])
        for att in atts:
            if att["filename"].lower().endswith(".pdf") and att.get("magic_bytes_hex"):
                assert att["magic_bytes_hex"].upper().startswith("255044462D"), (
                    f"{att['filename']}: magic bytes PDF errati: {att['magic_bytes_hex']}"
                )

    def test_jpg_magic_bytes_correct(self):
        """~WRD0000.jpg deve avere magic bytes JPEG."""
        atts = self.report.get("attachments", [])
        jpg = next((a for a in atts if a["filename"].lower().endswith(".jpg")), None)
        assert jpg is not None
        if jpg.get("magic_bytes_hex"):
            assert jpg["magic_bytes_hex"].upper().startswith("FFD8FF")

    def test_all_hashes_present(self):
        """Ogni allegato deve avere SHA-256 per il lookup VirusTotal."""
        for att in self.report.get("attachments", []):
            if att.get("hash_sha256"):
                assert len(att["hash_sha256"]) == 64

    def test_no_extension_anomaly_on_pdfs(self):
        """I PDF non devono avere anomalie di tipo (extension/magic mismatch)."""
        atts = self.report.get("attachments", [])
        for att in atts:
            if att["filename"].lower().endswith(".pdf"):
                assert att["anomaly"] is None, (
                    f"Anomalia inattesa su {att['filename']}: {att['anomaly']}"
                )


# ══════════════════════════════════════════════════════════════════════════
# 7. test7.eml — SPF pass + DKIM none, Return-Path diverso dal From
# ══════════════════════════════════════════════════════════════════════════

class TestSpfPassDkimNone:
    """
    Email cefla.it con SPF pass ma DKIM assente.
    Return-Path: teamsecurity+...@cefla.it ≠ From: davide.seller@cefla.it
    Catena lunga: 10 hop Received.
    """

    @pytest.fixture(autouse=True)
    def _load(self, eml_spf_pass_dkim_none):
        self.report = analyzer.analyze(eml_spf_pass_dkim_none)

    def test_from_is_cefla(self):
        assert "cefla.it" in self.report["from_"].lower()

    def test_return_path_different_from_from(self):
        """Return-Path usa teamsecurity+... → deve differire dal From."""
        rp = self.report.get("return_path") or ""
        from_ = self.report.get("from_") or ""
        if rp and from_:
            # Estrai solo l'indirizzo email dal campo From
            import re
            from_addr = re.search(r"<([^>]+)>", from_)
            from_email = from_addr.group(1).lower() if from_addr else from_.lower()
            assert rp.lower().strip("<>") != from_email, (
                "Return-Path e From coincidono, ma dovrebbero differire"
            )

    def test_received_spf_pass(self):
        """Received-SPF deve contenere 'pass' (o 'Pass')."""
        spf_raw = self.report.get("received_spf_raw") or ""
        assert "pass" in spf_raw.lower(), f"SPF pass non trovato in: {spf_raw!r}"

    def test_long_received_chain(self):
        """test7.eml ha 10 hop Received."""
        hops = self.report.get("received_hops", [])
        assert len(hops) >= 8, f"Attesi ≥8 hop, trovati {len(hops)}"

    def test_injection_server_ip_present(self):
        """Il server di iniezione (closest-to-sender) deve avere un IP estratto."""
        inj = self.report.get("injection_server") or {}
        # Accettiamo anche sender_ip dentro l'injection_server
        ip = inj.get("sender_ip") or inj.get("ip")
        assert ip, f"IP del server di iniezione non trovato. injection_server: {inj}"


# ══════════════════════════════════════════════════════════════════════════
# 8. test_8.eml — Email con allegato PDF, SPF pass via Google
# ══════════════════════════════════════════════════════════════════════════

class TestPdfVpnEmail:
    """Email con allegato PDF 'VPN Cefla.pdf', SPF pass via Google Workspace."""

    @pytest.fixture(autouse=True)
    def _load(self, eml_pdf_vpn):
        self.report = analyzer.analyze(eml_pdf_vpn)

    def test_from_is_cefla(self):
        assert "cefla.it" in self.report["from_"].lower()

    def test_pdf_attachment_present(self):
        atts = self.report.get("attachments", [])
        pdfs = [a for a in atts if a["filename"].lower().endswith(".pdf")]
        assert len(pdfs) >= 1, f"PDF non trovato. Allegati: {[a['filename'] for a in atts]}"

    def test_pdf_filename_contains_vpn(self):
        atts = self.report.get("attachments", [])
        vpn_pdf = next((a for a in atts if "vpn" in a["filename"].lower()), None)
        assert vpn_pdf is not None, "Allegato VPN Cefla.pdf non trovato"

    def test_pdf_magic_bytes(self):
        atts = self.report.get("attachments", [])
        pdf = next((a for a in atts if a["filename"].lower().endswith(".pdf")), None)
        if pdf and pdf.get("magic_bytes_hex"):
            assert pdf["magic_bytes_hex"].upper().startswith("255044462D")

    def test_spf_pass_in_received_header(self):
        spf_raw = self.report.get("received_spf_raw") or ""
        assert "pass" in spf_raw.lower(), f"SPF pass non trovato in: {spf_raw!r}"


# ══════════════════════════════════════════════════════════════════════════
# 9. Test trasversali — comportamento comune a tutte le email
# ══════════════════════════════════════════════════════════════════════════

ALL_FIXTURES = [
    "eml_spam_google_form",
    "eml_internal_forward",
    "eml_spf_fail_dkim_pass",
    "eml_minimal",
    "eml_good_internal",
    "eml_good_with_png",
    "eml_iso8859_subject",
    "eml_multi_pdf",
    "eml_spf_pass_dkim_none",
    "eml_pdf_vpn",
]


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_analyze_never_crashes(fixture_name, request):
    """analyze() non deve mai lanciare eccezioni, su nessuna email."""
    eml_path = request.getfixturevalue(fixture_name)
    report = analyzer.analyze(eml_path)
    assert isinstance(report, dict)


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_required_keys_always_present(fixture_name, request):
    """Il report deve sempre contenere le chiavi obbligatorie."""
    eml_path = request.getfixturevalue(fixture_name)
    report = analyzer.analyze(eml_path)
    required_keys = ["from_", "subject", "flags", "attachments", "links", "received_hops"]
    for key in required_keys:
        assert key in report, f"Chiave '{key}' mancante nel report di {fixture_name}"


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_flags_have_required_fields(fixture_name, request):
    """Ogni flag SOC deve avere level, category e detail."""
    eml_path = request.getfixturevalue(fixture_name)
    report = analyzer.analyze(eml_path)
    valid_levels = {"HIGH", "MEDIUM", "LOW", "INFO"}
    for flag in report.get("flags", []):
        assert "level" in flag, f"Flag senza 'level': {flag}"
        assert "category" in flag, f"Flag senza 'category': {flag}"
        assert "detail" in flag, f"Flag senza 'detail': {flag}"
        assert flag["level"] in valid_levels, (
            f"Flag con level non valido '{flag['level']}'. Validi: {valid_levels}"
        )


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_attachments_have_required_fields(fixture_name, request):
    """Ogni allegato nel report deve avere filename, content_type e magic_bytes_hex."""
    eml_path = request.getfixturevalue(fixture_name)
    report = analyzer.analyze(eml_path)
    for att in report.get("attachments", []):
        assert "filename" in att, f"Allegato senza 'filename': {att}"
        assert "content_type" in att, f"Allegato senza 'content_type': {att}"
        # magic_bytes_hex può essere None solo se il payload non è base64
        assert "magic_bytes_hex" in att, f"Campo 'magic_bytes_hex' mancante in {att['filename']}"


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_links_have_required_fields(fixture_name, request):
    """Ogni link estratto deve avere url, host e is_ip."""
    eml_path = request.getfixturevalue(fixture_name)
    report = analyzer.analyze(eml_path)
    for link in report.get("links", []):
        assert "url" in link, f"Link senza 'url': {link}"
        assert "host" in link, f"Link senza 'host': {link}"
        assert "is_ip" in link, f"Link senza 'is_ip': {link}"
        assert isinstance(link["is_ip"], bool)