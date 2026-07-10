import traceback

import streamlit as st


try:
    from src.app import main

    main()
except Exception as exc:
    st.error("FishStop could not start.")
    st.caption("This error is shown by the guarded Streamlit entrypoint.")
    with st.expander("Startup error details", expanded=True):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), language="text")
