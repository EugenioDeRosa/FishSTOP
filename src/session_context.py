"""Per-client identifiers used to isolate work on shared process resources."""

import secrets

import streamlit as st


ANALYSIS_SESSION_STATE_KEY = "fishstop_analysis_session_id"


def get_analysis_session_id() -> str:
    """Return an opaque identifier that exists only in this Streamlit session."""
    session_id = str(
        st.session_state.get(ANALYSIS_SESSION_STATE_KEY, "") or ""
    ).strip()
    if not session_id:
        session_id = secrets.token_urlsafe(18)
        st.session_state[ANALYSIS_SESSION_STATE_KEY] = session_id
    return session_id
