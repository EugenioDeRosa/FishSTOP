"""
config.py - Gestione centralizzata delle variabili d'ambiente e dei segreti.

Priorita di risoluzione:
  1. Variabili d'ambiente / file .env locale
  2. st.secrets di Streamlit
  3. Stringa vuota come fallback
"""

import os


_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _load_env_fallback(path: str) -> bool:
    """Carica KEY=VALUE da .env senza dipendenze esterne."""
    if not os.path.exists(path):
        return False

    with open(path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and not os.environ.get(key):
                os.environ[key] = value

    return True


try:
    from dotenv import load_dotenv

    _DOTENV_LOADED = load_dotenv(dotenv_path=_env_path, override=False)
    if not _DOTENV_LOADED:
        _DOTENV_LOADED = _load_env_fallback(_env_path)
except ImportError:
    _DOTENV_LOADED = _load_env_fallback(_env_path)


def _is_streamlit_cloud() -> bool:
    try:
        import streamlit as st

        _ = st.secrets
        return True
    except Exception:
        return False


def _read_streamlit_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        return st.secrets.get(key, default)
    except Exception:
        return default


def get_secret(key: str, default: str = "") -> str:
    value = os.environ.get(key)
    if value:
        return value

    if _is_streamlit_cloud():
        value = _read_streamlit_secret(key, default)
        if value:
            return value

    return default


def print_config_status():
    print("=" * 55)
    print("  CONFIG STATUS")
    print("=" * 55)
    print(f"  .env caricato          : {_DOTENV_LOADED}")
    print(f"  Streamlit secrets OK   : {_is_streamlit_cloud()}")
    print(f"  ABUSEIPDB_API_KEY      : {'presente' if ABUSEIPDB_API_KEY else 'mancante'}")
    print(f"  VIRUSTOTAL_API_KEY     : {'presente' if VIRUSTOTAL_API_KEY else 'mancante'}")
    print("=" * 55)


ABUSEIPDB_API_KEY = get_secret("ABUSEIPDB_API_KEY")
VIRUSTOTAL_API_KEY = get_secret("VIRUSTOTAL_API_KEY")


if __name__ == "__main__":
    print_config_status()
