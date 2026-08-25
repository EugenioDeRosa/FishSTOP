import streamlit as st


try:
    from src.app import main

    main()
except Exception as exc:
    try:
        from src.error_handling import render_unexpected_error

        render_unexpected_error(
            "FishStop could not start.",
            exc,
            context="guarded Streamlit entrypoint",
        )
    except Exception:
        # Keep the final fallback independent from the application package:
        # startup diagnostics remain in server logs, never in the public page.
        st.error("FishStop could not start.")
