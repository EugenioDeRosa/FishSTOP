"""Safe user-facing handling for unexpected application errors."""

from __future__ import annotations

import logging
import secrets

import streamlit as st

from src.config import is_production_mode


LOGGER = logging.getLogger("fishstop")


def report_unexpected_error(exc: BaseException, *, context: str) -> str:
    """Log a traceback server-side and return a non-sensitive reference."""
    reference = f"FS-{secrets.token_hex(4).upper()}"
    LOGGER.error(
        "Unexpected FishStop error [%s] in %s",
        reference,
        context,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return reference


def render_unexpected_error(
    message: str,
    exc: BaseException,
    *,
    context: str,
) -> str:
    """Render a generic production error and development-only diagnostics."""
    reference = report_unexpected_error(exc, context=context)
    st.error(message)
    st.caption(f"Error reference: `{reference}`")
    if not is_production_mode():
        with st.expander("Development error details", expanded=False):
            st.exception(exc)
    return reference
