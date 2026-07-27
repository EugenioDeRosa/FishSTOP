"""
config.py - Gestione centralizzata delle variabili d'ambiente e dei segreti.

Priorita di risoluzione:
  1. Variabili d'ambiente / file .env locale
  2. st.secrets di Streamlit
  3. Stringa vuota come fallback
"""

import os


USER_API_KEYS_STATE_KEY = "fishstop_user_api_keys"
PRODUCTION_APP_MODES = {"prod", "production", "public"}
MANAGED_API_KEYS = [
    "VIRUSTOTAL_API_KEY",
    "ABUSEIPDB_API_KEY",
    "GITHUB_MODELS_TOKEN",
    "HF_TOKEN",
]


def is_production_mode() -> bool:
    """Return whether public-production restrictions must be enforced."""
    return os.getenv("APP_MODE", "development").strip().lower() in PRODUCTION_APP_MODES


_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def _load_env_fallback(path: str) -> bool:
    """Loads KEY=VALUE from .env without external dependencies."""
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


def _read_user_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        return str(st.session_state.get(USER_API_KEYS_STATE_KEY, {}).get(key) or default).strip()
    except Exception:
        return default


def set_user_secret(key: str, value: str) -> None:
    if key not in MANAGED_API_KEYS:
        raise ValueError(f"Unsupported API key: {key}")
    try:
        import streamlit as st
    except Exception as exc:
        raise RuntimeError("User API keys can only be updated inside Streamlit") from exc

    keys = dict(st.session_state.get(USER_API_KEYS_STATE_KEY, {}))
    value = str(value or "").strip()
    if value:
        keys[key] = value
    else:
        keys.pop(key, None)
    st.session_state[USER_API_KEYS_STATE_KEY] = keys


def get_secret(key: str, default: str = "") -> str:
    value = _read_user_secret(key)
    if value:
        return value

    value = os.environ.get(key)
    if value:
        return value

    if _is_streamlit_cloud():
        value = _read_streamlit_secret(key, default)
        if value:
            return value

    return default


def get_server_secret(key: str, default: str = "") -> str:
    """Resolve only process/server credentials, never per-session overrides."""
    value = os.environ.get(key)
    if value:
        return value
    if _is_streamlit_cloud():
        value = _read_streamlit_secret(key, default)
        if value:
            return value
    return default


def get_secret_source(key: str) -> str:
    if _read_user_secret(key):
        return "settings"
    if os.environ.get(key):
        return ".env/environment"
    if _is_streamlit_cloud() and _read_streamlit_secret(key):
        return "streamlit secrets"
    return "missing"


def print_config_status():
    print("=" * 55)
    print("  CONFIG STATUS")
    print("=" * 55)
    print(f"  .env caricato          : {_DOTENV_LOADED}")
    print(f"  Streamlit secrets OK   : {_is_streamlit_cloud()}")
    print(f"  ABUSEIPDB_API_KEY      : {'present' if ABUSEIPDB_API_KEY else 'missing'}")
    print(f"  VIRUSTOTAL_API_KEY     : {'present' if VIRUSTOTAL_API_KEY else 'missing'}")
    print(f"  HF_TOKEN               : {'present' if HF_TOKEN else 'missing'}")
    print("=" * 55)


ABUSEIPDB_API_KEY = get_secret("ABUSEIPDB_API_KEY")
VIRUSTOTAL_API_KEY = get_secret("VIRUSTOTAL_API_KEY")
GITHUB_MODELS_TOKEN = get_secret("GITHUB_MODELS_TOKEN")
HF_TOKEN = get_secret("HF_TOKEN")


if __name__ == "__main__":
    print_config_status()
