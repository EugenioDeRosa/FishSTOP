"""
tests/conftest.py — Fixture condivise per la suite di test FishStop.

Le email di test sono nella cartella tests/fixtures/ (copia delle email
reali usate durante lo sviluppo). Ogni fixture carica il file .eml come
percorso oppure come bytes grezzi, pronti per i moduli analyzer e validators.
"""

import os
import pytest

# Cartella con le email di test
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _eml_path(filename: str) -> str:
    """Percorso assoluto a un file .eml nella cartella fixtures."""
    path = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"File di test non trovato: {path}")
    return path


def _eml_bytes(filename: str) -> bytes:
    with open(_eml_path(filename), "rb") as f:
        return f.read()


# ── Fixture per ogni email ─────────────────────────────────────────────────

@pytest.fixture
def eml_spam_google_form():
    """
    spam_email.eml — Google Form legittimo inviato per conto di cefla.it.
    SPF: pass  |  DKIM: pass  |  Reply-To diverso dal From (google.com vs cefla.it)
    """
    return _eml_path("spam_email.eml")


@pytest.fixture
def eml_spam_google_form_bytes():
    return _eml_bytes("spam_email.eml")


@pytest.fixture
def eml_internal_forward():
    """
    spam_email_-_2.eml — Email interna cefla.it inoltrata, con allegati immagine.
    SPF: pass  |  DKIM: pass  |  Reply-To: assente  |  Allegati: jpg, png
    """
    return _eml_path("spam_email_-_2.eml")


@pytest.fixture
def eml_spf_fail_dkim_pass():
    """
    test.eml — SPF FAIL + DKIM pass: mittente unibo.it inviato via Gmail.
    Caso interessante: autenticazione ibrida, DMARC pass per DKIM.
    Allegati: PDF calendario.
    """
    return _eml_path("test.eml")


@pytest.fixture
def eml_minimal():
    """
    test2.eml — Email minimale senza SPF/DKIM, allegato HTML.
    Utile per testare il comportamento con header assenti.
    """
    return _eml_path("test2.eml")


@pytest.fixture
def eml_good_internal():
    """
    test3_good.eml — Email interna cefla senza header SPF/DKIM (rete interna).
    3 hop Received, allegati immagini inline.
    """
    return _eml_path("test3_good.eml")


@pytest.fixture
def eml_good_with_png():
    """
    test4_G.eml — Email interna con allegato PNG (Outlook attachment).
    0 hop Received (generata localmente).
    """
    return _eml_path("test4_G.eml")


@pytest.fixture
def eml_iso8859_subject():
    """
    test5_G.eml — Subject codificato ISO-8859-1 (=?iso-8859-1?Q?...).
    Test per la decodifica di charset non UTF-8.
    """
    return _eml_path("test5_G.eml")


@pytest.fixture
def eml_multi_pdf():
    """
    test6_G.eml — Email con 2 allegati PDF e 1 immagine (~WRD0000.jpg).
    Test per l'analisi multipla degli allegati e magic bytes.
    """
    return _eml_path("test6_G.eml")


@pytest.fixture
def eml_spf_pass_dkim_none():
    """
    test7.eml — SPF pass + DKIM none (messaggio non firmato).
    10 hop Received, Return-Path diverso dal From (teamsecurity+...@cefla.it).
    """
    return _eml_path("test7.eml")


@pytest.fixture
def eml_pdf_vpn():
    """
    test_8.eml — Email con allegato PDF, SPF pass via Google.
    2 hop Received.
    """
    return _eml_path("test_8.eml")