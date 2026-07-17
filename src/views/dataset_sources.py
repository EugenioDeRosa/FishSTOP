from pathlib import Path

import pandas as pd
import streamlit as st

from src.public_dataset_builder import (
    DEFAULT_BALANCED_OUTPUT_CSV,
    DEFAULT_COMPLETE_OUTPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SYNTHETIC_CSV,
    build_balanced_public_dataset,
    build_complete_training_dataset,
    dataset_stats,
)

SOURCE_OPTIONS = {
    "github_phishing_pot": {
        "label": "GitHub Phishing Pot",
        "caption": (
            "Campioni reali recenti da honeypot: phishing, scam e spam vengono tutti importati "
            "come email malevole (label 1). La versione GitHub e fissata a un commit riproducibile."
        ),
        "url": "https://github.com/rf-peixoto/phishing_pot",
    },
    "nazario": {
        "label": "Nazario Phishing Corpus",
        "caption": "Email phishing reali 2022-2025 in formato mbox/raw; gli anni precedenti sono esclusi.",
        "url": "https://monkey.org/~jose/phishing/",
    },
    "ubuntu_modern_ham": {
        "label": "Ubuntu public mailing lists 2022-2025",
        "caption": (
            "Email legitimate recenti dagli archivi pubblici ubuntu-users e ubuntu-security-announce. "
            "Sostituiscono i vecchi corpus SpamAssassin ed Enron."
        ),
        "url": "https://lists.ubuntu.com/archives/ubuntu-users/",
    },
}

RECOMMENDED_DEFAULT_SOURCES = {"github_phishing_pot", "nazario", "ubuntu_modern_ham"}

def _render_stats(csv_path: Path) -> None:
    stats = dataset_stats(csv_path)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total rows", stats["rows"])
    c2.metric("Legitimate", stats["legitimate"])
    c3.metric("Malicious (phishing/spam)", stats["phishing"])
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


def _run_builder(label: str, builder=build_balanced_public_dataset, *args, **kwargs) -> dict:
    log_box = st.empty()
    messages: list[str] = []

    def _progress(message: str) -> None:
        messages.append(message)
        log_box.code("\n".join(messages[-12:]), language="text")

    with st.spinner(label):
        result = builder(*args, progress=_progress, **kwargs)

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
        "Seleziona le fonti pubbliche e genera un CSV pronto per DistilBERT: bilanciato 50/50, "
        "deduplicato e con phishing e spam entrambi nella classe malevola. Il dataset sintetico "
        "moderno viene aggiunto esclusivamente al train."
    )

    output_csv = Path(st.text_input("Complete training CSV", value=str(DEFAULT_COMPLETE_OUTPUT_CSV)))
    public_output_csv = Path(
        st.text_input("Intermediate public-only CSV", value=str(DEFAULT_BALANCED_OUTPUT_CSV))
    )
    synthetic_csv = Path(st.text_input("Synthetic augmentation CSV", value=str(DEFAULT_SYNTHETIC_CSV)))
    st.caption(
        "Format: `text,label,source,source_file,text_hash,campaign_id,split` con "
        "`0=legitimate`, `1=malicious (phishing/spam)` e split per campagna anti-leakage."
    )

    st.divider()
    st.subheader("Complete dataset status")
    _render_stats(output_csv)
    if synthetic_csv.exists():
        synthetic_stats = dataset_stats(synthetic_csv)
        st.caption(
            f"Synthetic augmentation available: {synthetic_stats['rows']} rows "
            f"({synthetic_stats['legitimate']} legitimate, {synthetic_stats['phishing']} malicious)."
        )
    else:
        st.error(f"Synthetic dataset not found: `{synthetic_csv}`")

    st.divider()
    st.subheader("Sources to include")
    st.caption("La policy moderna esclude Enron, SpamAssassin e i Kaggle derivati da corpus storici.")

    selected_sources: list[str] = []
    for key, option in SOURCE_OPTIONS.items():
        default = key in RECOMMENDED_DEFAULT_SOURCES
        checked = st.checkbox(
            option["label"],
            value=default,
            key=f"public_source_{key}",
        )
        st.caption(option["caption"])
        st.markdown(f"[Open source]({option['url']})")
        if option.get("citation_url"):
            st.markdown(f"[Citation paper]({option['citation_url']})")
        if checked:
            selected_sources.append(key)

    st.divider()
    if st.button(
        "Generate complete DistilBERT training dataset",
        type="primary",
        disabled=not selected_sources or not synthetic_csv.exists(),
        use_container_width=True,
    ):
        _run_builder(
            "Generating and validating complete training dataset...",
            builder=build_complete_training_dataset,
            selected_sources=selected_sources,
            output_csv=output_csv,
            public_output_csv=public_output_csv,
            synthetic_csv=synthetic_csv,
            staging_csv=DEFAULT_OUTPUT_CSV,
        )
        st.rerun()

    st.caption(
        "Quality rule: synthetic emails remain train-only and may represent at most 10% of the training split. "
        "Validation and test contain only public/real emails."
    )

    with st.expander("Advanced: generate only the balanced public dataset", expanded=False):
        if st.button(
            "Generate public-only 50/50 dataset",
            disabled=not selected_sources,
            use_container_width=True,
        ):
            _run_builder(
                "Generating public-only dataset...",
                selected_sources=selected_sources,
                output_csv=public_output_csv,
                staging_csv=DEFAULT_OUTPUT_CSV,
            )
            st.rerun()

    if output_csv.exists():
        with st.expander("CSV preview", expanded=False):
            df = pd.read_csv(output_csv, nrows=200)
            st.dataframe(df, hide_index=True, width="stretch")

        st.download_button(
            "Download complete training CSV",
            data=output_csv.read_bytes(),
            file_name=output_csv.name,
            mime="text/csv",
            use_container_width=True,
        )
