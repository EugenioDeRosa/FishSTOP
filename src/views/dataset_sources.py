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
            "Dataset combinato da Enron, Ling, CEAS, Nazario, Nigerian Fraud e SpamAssassin. "
            "La classe positiva mescola spam e phishing: non e consigliata come fonte predefinita."
        ),
        "url": "https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset",
        "citation_url": "https://arxiv.org/abs/2405.11619",
    },
    "kaggle_phishing_legitimate": {
        "label": "Kaggle Phishing and Legitimate Emails",
        "caption": (
            "10.000 email generate sinteticamente con un LLM (6.000 phishing, 4.000 legitimate). "
            "Usare solo come augmentation del train; escluse da validation e test se ci sono fonti reali."
        ),
        "url": "https://www.kaggle.com/datasets/kuladeep19/phishing-and-legitimate-emails-dataset",
    },
    "kaggle_subhajournal_phishingemails": {
        "label": "Kaggle Phishing Email Detection",
        "caption": (
            "18.650 email: 11.322 Safe e 7.328 Phishing. La copia originale contiene oltre 1.100 duplicati, "
            "533 placeholder empty con label contraddittorie e una riga corrotta enorme: ora vengono eliminati."
        ),
        "url": "https://www.kaggle.com/datasets/subhajournal/phishingemails",
    },
    "github_phishing_pot": {
        "label": "GitHub Phishing Pot",
        "caption": (
            "Raccolta di campioni reali di phishing in formato .eml da honeypot. "
            "Contiene solo email malevole: viene importata sempre con label 1."
        ),
        "url": "https://github.com/rf-peixoto/phishing_pot",
    },
    "nazario": {
        "label": "Nazario Phishing Corpus",
        "caption": "Email phishing reali in formato mbox/raw. Non selezionare insieme al Kaggle combinato.",
        "url": "https://monkey.org/~jose/phishing/",
    },
    "spamassassin": {
        "label": "Apache SpamAssassin Ham",
        "caption": "Email legitimate easy_ham e hard_ham. Non selezionare insieme al Kaggle combinato.",
        "url": "https://spamassassin.apache.org/old/publiccorpus/",
    },
    "enron": {
        "label": "Enron Email Corpus",
        "caption": "Email legitimate, campionate per contenere tempi e dimensioni. Non selezionare insieme al Kaggle combinato.",
        "url": "https://www.cs.cmu.edu/~enron/",
    },
}

RECOMMENDED_DEFAULT_SOURCES = {"github_phishing_pot", "nazario", "spamassassin"}

def _render_stats(csv_path: Path) -> None:
    stats = dataset_stats(csv_path)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total rows", stats["rows"])
    c2.metric("Legitimate", stats["legitimate"])
    c3.metric("Phishing", stats["phishing"])
    c4.metric("Duplicates", stats.get("duplicates", 0))
    c5.metric("Near duplicates", stats.get("template_duplicates", 0))
    c6.metric("Label conflicts", stats.get("label_conflicts", 0))

    if stats.get("missing_label"):
        st.warning("The existing CSV does not contain the `label` column: regenerate it with the button below.")

    quality = {
        "Invalid text": stats.get("invalid_text", 0),
        "Invalid label": stats.get("invalid_label", 0),
        "Too short": stats.get("too_short", 0),
        "Too long/corrupt": stats.get("too_long", 0),
        "Exact label conflicts": stats.get("exact_label_conflicts", 0),
    }
    if any(quality.values()):
        with st.expander("Rows rejected by quality checks", expanded=False):
            st.dataframe(
                pd.DataFrame([{"check": key, "rows": value} for key, value in quality.items()]),
                hide_index=True,
                width="stretch",
            )

    if stats.get("splits"):
        st.caption("Split: " + ", ".join(f"{key}={value}" for key, value in stats["splits"].items()))
    if stats["sources"]:
        with st.expander("Distribution by source", expanded=False):
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
                f"{step.source}: added {step.added}, duplicates/discarded {step.skipped}, "
                f"errors {step.errors}. {step.message}"
            )
        else:
            st.success(
                f"{step.source}: added {step.added}, duplicates/discarded {step.skipped}."
            )

    if result["status"] == "ok":
        st.success(f"{result['message']} File: `{result['output']}`")
    else:
        st.error(result["message"])
    return result


def render() -> None:
    st.header("Public Dataset Builder")
    st.markdown(
        "Select the public sources to include and generate a balanced CSV in one click "
        "50/50 tra email legitimate e phishing, con deduplica esatta e rimozione dei quasi-duplicati/template."
    )

    output_csv = Path(st.text_input("Final balanced CSV", value=str(BALANCED_OUTPUT_CSV)))
    st.caption("Format: `text,label,source,source_file,text_hash,split` con `0=legitimate`, `1=phishing` e split gia anti-leakage.")

    st.divider()
    st.subheader("Balanced CSV status")
    _render_stats(output_csv)

    st.divider()
    st.subheader("Sources to include")
    st.caption("Kaggle sources require `kagglehub` and configured Kaggle credentials.")

    selected_sources: list[str] = []
    kaggle_selected = False
    for key, option in SOURCE_OPTIONS.items():
        default = key in RECOMMENDED_DEFAULT_SOURCES
        disabled_by_combined = kaggle_selected and key in KAGGLE_COMBINED_OVERLAP_SOURCES
        checked = st.checkbox(
            option["label"],
            value=default and not disabled_by_combined,
            key=f"public_source_{key}",
            disabled=disabled_by_combined,
        )
        st.caption(option["caption"])
        st.markdown(f"[Open source]({option['url']})")
        if option.get("citation_url"):
            st.markdown(f"[Citation paper]({option['citation_url']})")
        if disabled_by_combined:
            st.info("Already included in the combined Kaggle Phishing Email Dataset.")
            checked = False
        if checked:
            selected_sources.append(key)
        if key == "kaggle":
            kaggle_selected = checked

    include_hard_ham = st.checkbox(
        "Include SpamAssassin hard_ham",
        value=True,
        disabled="spamassassin" not in selected_sources,
    )
    max_enron = st.number_input(
        "Maximum Enron emails",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        disabled="enron" not in selected_sources,
    )

    st.divider()
    if st.button(
        "Generate balanced 50/50 public dataset",
        type="primary",
        disabled=not selected_sources,
        use_container_width=True,
    ):
        _run_builder(
            "Generating balanced dataset...",
            selected_sources=selected_sources,
            output_csv=output_csv,
            staging_csv=DEFAULT_OUTPUT_CSV,
            include_hard_ham=include_hard_ham,
            max_enron=int(max_enron),
        )
        st.rerun()

    if output_csv.exists():
        with st.expander("CSV preview", expanded=False):
            df = pd.read_csv(output_csv, nrows=200)
            st.dataframe(df, hide_index=True, width="stretch")

        st.download_button(
            "Download balanced CSV",
            data=output_csv.read_bytes(),
            file_name=output_csv.name,
            mime="text/csv",
            use_container_width=True,
        )
