from pathlib import Path

import pandas as pd
import streamlit as st

from src.public_dataset_builder import (
    DEFAULT_OUTPUT_CSV,
    KAGGLE_COMBINED_OVERLAP_SOURCES,
    PROCESSED_DIR,
    build_balanced_public_dataset,
    dataset_stats,
)


BALANCED_OUTPUT_CSV = PROCESSED_DIR / "fishstop_train_balanced.csv"

SOURCE_OPTIONS = {
    "kaggle": {
        "label": "Kaggle Phishing Email Dataset",
        "caption": (
            "Dataset finale combinato: Enron, Ling, CEAS, Nazario, Nigerian Fraud e SpamAssassin. "
            "82.486 righe originali, circa 81.820 dopo pulizia e deduplica."
        ),
        "url": "https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset",
        "citation_url": "https://arxiv.org/abs/2405.11619",
    },
    "kaggle_phishing_legitimate": {
        "label": "Kaggle Phishing and Legitimate Emails",
        "caption": (
            "10.000 email con text, label, phishing_type, severity e confidence "
            "(6.000 phishing, 4.000 legittime nella copia locale). Usalo come fonte aggiuntiva opzionale."
        ),
        "url": "https://www.kaggle.com/datasets/kuladeep19/phishing-and-legitimate-emails-dataset",
    },
    "kaggle_subhajournal_phishingemails": {
        "label": "Kaggle Phishing Email Detection",
        "caption": (
            "18.650 email con Email Text e Email Type: 11.322 Safe Email, 7.328 Phishing Email. "
            "Fonte opzionale: contiene duplicati/quasi-duplicati, quindi viene filtrata dalla deduplica template."
        ),
        "url": "https://www.kaggle.com/datasets/subhajournal/phishingemails",
    },
    "nazario": {
        "label": "Nazario Phishing Corpus",
        "caption": "Email phishing reali in formato mbox/raw. Non selezionare insieme al Kaggle combinato.",
        "url": "https://monkey.org/~jose/phishing/",
    },
    "spamassassin": {
        "label": "Apache SpamAssassin Ham",
        "caption": "Email legittime easy_ham e hard_ham. Non selezionare insieme al Kaggle combinato.",
        "url": "https://spamassassin.apache.org/old/publiccorpus/",
    },
    "enron": {
        "label": "Enron Email Corpus",
        "caption": "Email legittime, campionate per contenere tempi e dimensioni. Non selezionare insieme al Kaggle combinato.",
        "url": "https://www.cs.cmu.edu/~enron/",
    },
}


def _render_stats(csv_path: Path) -> None:
    stats = dataset_stats(csv_path)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Totale righe", stats["rows"])
    c2.metric("Legittime", stats["legitimate"])
    c3.metric("Phishing", stats["phishing"])
    c4.metric("Doppioni", stats.get("duplicates", 0))
    c5.metric("Quasi doppioni", stats.get("template_duplicates", 0))
    c6.metric("Conflitti label", stats.get("label_conflicts", 0))

    if stats.get("missing_label"):
        st.warning("Il CSV esistente non contiene la colonna `label`: rigeneralo con il pulsante qui sotto.")

    if stats["sources"]:
        with st.expander("Distribuzione per fonte", expanded=False):
            source_df = pd.DataFrame(
                [{"source": k, "rows": v} for k, v in stats["sources"].items()]
            )
            st.dataframe(source_df, hide_index=True, width="stretch")


def _run_builder(label: str, *args, **kwargs) -> dict:
    log_box = st.empty()
    messages: list[str] = []

    def _progress(message: str) -> None:
        messages.append(message)
        log_box.code("\n".join(messages[-12:]), language="text")

    with st.spinner(label):
        result = build_balanced_public_dataset(*args, progress=_progress, **kwargs)

    for step in result.get("results", []):
        if "saltata" in step.message.lower():
            st.info(f"{step.source}: {step.message}")
        elif step.errors:
            st.warning(
                f"{step.source}: aggiunte {step.added}, duplicate/scartate {step.skipped}, "
                f"errori {step.errors}. {step.message}"
            )
        else:
            st.success(
                f"{step.source}: aggiunte {step.added}, duplicate/scartate {step.skipped}."
            )

    if result["status"] == "ok":
        st.success(f"{result['message']} File: `{result['output']}`")
    else:
        st.error(result["message"])
    return result


def render() -> None:
    st.header("Public Dataset Builder")
    st.markdown(
        "Seleziona le fonti pubbliche da includere e genera in un clic un CSV bilanciato "
        "50/50 tra email legittime e phishing, con deduplica esatta e rimozione dei quasi-duplicati/template."
    )

    output_csv = Path(st.text_input("CSV bilanciato finale", value=str(BALANCED_OUTPUT_CSV)))
    st.caption("Formato: `text,label,source,source_file,text_hash` con `0=legittima`, `1=phishing`.")

    st.divider()
    st.subheader("Stato CSV bilanciato")
    _render_stats(output_csv)

    st.divider()
    st.subheader("Fonti da includere")
    st.caption("Le fonti Kaggle richiedono `kagglehub` e le credenziali Kaggle configurate.")

    selected_sources: list[str] = []
    kaggle_selected = False
    for key, option in SOURCE_OPTIONS.items():
        default = key == "kaggle"
        disabled_by_combined = kaggle_selected and key in KAGGLE_COMBINED_OVERLAP_SOURCES
        checked = st.checkbox(
            option["label"],
            value=default and not disabled_by_combined,
            key=f"public_source_{key}",
            disabled=disabled_by_combined,
        )
        st.caption(option["caption"])
        st.markdown(f"[Apri fonte]({option['url']})")
        if option.get("citation_url"):
            st.markdown(f"[Articolo da citare]({option['citation_url']})")
        if disabled_by_combined:
            st.info("Gia incluso nel Kaggle Phishing Email Dataset combinato.")
            checked = False
        if checked:
            selected_sources.append(key)
        if key == "kaggle":
            kaggle_selected = checked

    include_hard_ham = st.checkbox(
        "Includi hard_ham SpamAssassin",
        value=True,
        disabled="spamassassin" not in selected_sources,
    )
    max_enron = st.number_input(
        "Massimo email Enron",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        disabled="enron" not in selected_sources,
    )

    st.divider()
    if st.button(
        "Genera dataset pubblico bilanciato 50/50",
        type="primary",
        disabled=not selected_sources,
        use_container_width=True,
    ):
        _run_builder(
            "Generazione dataset bilanciato...",
            selected_sources=selected_sources,
            output_csv=output_csv,
            staging_csv=DEFAULT_OUTPUT_CSV,
            include_hard_ham=include_hard_ham,
            max_enron=int(max_enron),
        )
        st.rerun()

    if output_csv.exists():
        with st.expander("Anteprima CSV", expanded=False):
            df = pd.read_csv(output_csv, nrows=200)
            st.dataframe(df, hide_index=True, width="stretch")

        st.download_button(
            "Scarica CSV bilanciato",
            data=output_csv.read_bytes(),
            file_name=output_csv.name,
            mime="text/csv",
            use_container_width=True,
        )
