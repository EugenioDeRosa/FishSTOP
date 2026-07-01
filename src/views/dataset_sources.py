import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.public_dataset_builder import (
    DEFAULT_OUTPUT_CSV,
    add_enron_sample,
    add_kaggle,
    add_nazario,
    add_spamassassin,
    balance_dataset,
    dataset_stats,
)


def _render_stats(csv_path: Path):
    stats = dataset_stats(csv_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Totale righe", stats["rows"])
    c2.metric("Legittime", stats["legitimate"])
    c3.metric("Phishing", stats["phishing"])

    if stats["sources"]:
        with st.expander("Distribuzione per fonte", expanded=False):
            source_df = pd.DataFrame(
                [{"source": k, "rows": v} for k, v in stats["sources"].items()]
            )
            st.dataframe(source_df, hide_index=True, width="stretch")


def _run_step(label: str, fn, *args, **kwargs):
    log_box = st.empty()
    messages: list[str] = []

    def _progress(message: str):
        messages.append(message)
        log_box.code("\n".join(messages[-12:]), language="text")

    with st.spinner(label):
        result = fn(*args, progress=_progress, **kwargs)

    if result.errors:
        st.warning(
            f"{result.source}: aggiunte {result.added}, duplicate/scartate {result.skipped}, errori {result.errors}. "
            f"{result.message}"
        )
    else:
        st.success(f"{result.source}: aggiunte {result.added}, duplicate/scartate {result.skipped}.")
    return result


def render():
    st.header("🌐 Public Dataset Builder")
    st.markdown(
        "Scarica fonti pubbliche, estrae il testo email, normalizza, deduplica e genera "
        "un CSV pronto per il training in Colab."
    )

    output_csv = Path(st.text_input("CSV finale", value=str(DEFAULT_OUTPUT_CSV)))
    st.caption("Formato: `text,label,source,source_file,text_hash` con `0=legittima`, `1=phishing`.")

    st.divider()
    st.subheader("Stato dataset")
    _render_stats(output_csv)

    st.divider()
    st.subheader("Fonti")

    with st.container(border=True):
        st.markdown("#### Kaggle Phishing Email Dataset")
        st.caption("Richiede `kagglehub` e, se necessario, credenziali Kaggle già configurate.")
        if st.button("Scarica/Importa Kaggle", use_container_width=True):
            _run_step("Import Kaggle...", add_kaggle, output_csv=output_csv)
            st.rerun()

    with st.container(border=True):
        st.markdown("#### Nazario Phishing Corpus")
        st.caption("Email phishing reali in formato mbox/raw. Label applicata: phishing.")
        if st.button("Scarica/Importa Nazario", use_container_width=True):
            _run_step("Import Nazario...", add_nazario, output_csv=output_csv)
            st.rerun()

    with st.container(border=True):
        st.markdown("#### Apache SpamAssassin Ham")
        include_hard_ham = st.checkbox("Includi hard_ham", value=True)
        st.caption("easy_ham/hard_ham vengono etichettate come legittime.")
        if st.button("Scarica/Importa SpamAssassin Ham", use_container_width=True):
            _run_step(
                "Import SpamAssassin...",
                add_spamassassin,
                output_csv=output_csv,
                include_hard_ham=include_hard_ham,
            )
            st.rerun()

    with st.container(border=True):
        st.markdown("#### Enron Email Corpus")
        st.caption("Fonte grande: usa un campione per non appesantire troppo la prima build.")
        max_enron = st.number_input("Massimo email Enron", min_value=1000, max_value=50000, value=10000, step=1000)
        if st.button("Scarica/Importa Enron sample", use_container_width=True):
            _run_step(
                "Import Enron...",
                add_enron_sample,
                output_csv=output_csv,
                max_messages=int(max_enron),
            )
            st.rerun()

    st.divider()
    st.subheader("Build automatica consigliata")
    st.markdown(
        "Esegue Kaggle, Nazario e SpamAssassin Ham. Enron resta separato perché è molto grande."
    )
    if st.button("🚀 Costruisci dataset pubblico base", type="primary", use_container_width=True):
        _run_step("Import Kaggle...", add_kaggle, output_csv=output_csv)
        _run_step("Import Nazario...", add_nazario, output_csv=output_csv)
        _run_step("Import SpamAssassin...", add_spamassassin, output_csv=output_csv, include_hard_ham=True)
        st.rerun()

    st.divider()
    st.subheader("Bilanciamento per Colab")
    stats = dataset_stats(output_csv)
    default_per_class = min(stats["legitimate"], stats["phishing"]) if stats["rows"] else 0
    per_class = st.number_input(
        "Campioni per classe",
        min_value=0,
        max_value=max(default_per_class, 1),
        value=default_per_class,
        step=100,
        disabled=default_per_class == 0,
    )
    if st.button("Genera CSV bilanciato", disabled=default_per_class == 0, use_container_width=True):
        result = balance_dataset(output_csv, per_class=int(per_class) if per_class else None)
        st.success(f"Creato `{result['output']}` con {result['rows']} righe.")

    if output_csv.exists():
        with st.expander("Anteprima CSV", expanded=False):
            df = pd.read_csv(output_csv, nrows=200)
            st.dataframe(df, hide_index=True, width="stretch")

        st.download_button(
            "Scarica CSV corrente",
            data=output_csv.read_bytes(),
            file_name=output_csv.name,
            mime="text/csv",
            use_container_width=True,
        )
