import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import is_production_mode
from src.ui import metric_strip, page_intro

from src.public_dataset_builder import (
    DEFAULT_BALANCED_OUTPUT_CSV,
    DEFAULT_COMPLETE_OUTPUT_CSV,
    DEFAULT_LEGITIMATE_HARD_NEGATIVE_CSV,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SYNTHETIC_CSV,
    build_balanced_public_dataset,
    build_complete_training_dataset,
)

SOURCE_OPTIONS = {
    "github_phishing_pot": {
        "label": "GitHub Phishing Pot",
        "caption": (
            "Campioni reali recenti da honeypot: phishing, scam e spam vengono tutti importati "
            "come email malevole (label 1). L'header Date non viene usato come filtro "
            "e la versione GitHub è fissata a un commit riproducibile."
        ),
        "url": "https://github.com/rf-peixoto/phishing_pot",
    },
    "nazario": {
        "label": "Nazario Phishing Corpus",
        "caption": "Email phishing reali 2022-2025 in formato mbox/raw; gli anni precedenti sono esclusi.",
        "url": "https://monkey.org/~jose/phishing/",
    },
    "zenodo_validation": {
        "label": "Zenodo phishing emails",
        "caption": (
            "Dataset inglese CC BY 4.0 del 2024. Le 2.000 righe vengono deduplicate: "
            "restano circa 100 testi distinti, distribuiti negli split per campagna."
        ),
        "url": "https://zenodo.org/records/13474746",
    },
    "spaphish": {
        "label": "SpaPhish v5",
        "caption": (
            "Email spagnole CC BY 4.0 con label umane. Tutte le righe valide della versione 5 "
            "sono incluse e mescolate con le altre fonti; l'header Date non viene usato come filtro."
        ),
        "url": "https://data.mendeley.com/datasets/hz2d6gz7pc/5",
    },
    "spamassassin": {
        "label": "SpamAssassin public corpus",
        "caption": (
            "Email legittime storiche (easy ham e hard ham). Sono mantenute perché aggiungono "
            "una seconda fonte reale di messaggi non malevoli; duplicati e template ripetuti "
            "vengono rimossi."
        ),
        "url": "https://spamassassin.apache.org/old/publiccorpus/",
    },
    "enron": {
        "label": "Enron Email Dataset",
        "caption": (
            "Campione di email aziendali legittime storiche. L'età non è usata come filtro: "
            "la fonte aumenta la varietà del train ed è sottoposta agli stessi controlli anti-leakage."
        ),
        "url": "https://www.cs.cmu.edu/~enron/",
    },
}

RECOMMENDED_DEFAULT_SOURCES = {
    "github_phishing_pot",
    "nazario",
    "zenodo_validation",
    "spaphish",
    "spamassassin",
    "enron",
}

@st.cache_data(show_spinner=False)
def _cached_ui_stats(path_value: str, csv_mtime: float, manifest_mtime: float) -> dict:
    """Read precomputed stats when available, avoiding a full dedupe scan on page load."""
    csv_path = Path(path_value)
    manifest_path = csv_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            stats = payload.get("stats")
            if isinstance(stats, dict):
                return stats
        except (OSError, json.JSONDecodeError):
            pass

    if not csv_path.exists():
        return {
            "exists": False,
            "rows": 0,
            "legitimate": 0,
            "phishing": 0,
            "sources": {},
            "splits": {},
        }

    # Only load the three small columns needed by the UI. Deep quality checks
    # still run as part of dataset generation and are saved in the manifest.
    frame = pd.read_csv(
        csv_path,
        usecols=lambda column: column in {"label", "source", "split"},
    )
    labels = (
        pd.to_numeric(frame["label"], errors="coerce")
        if "label" in frame
        else pd.Series(dtype="float64")
    )
    return {
        "exists": True,
        "rows": len(frame),
        "legitimate": int((labels == 0).sum()),
        "phishing": int((labels == 1).sum()),
        "missing_label": "label" not in frame,
        "sources": frame["source"].value_counts().to_dict() if "source" in frame else {},
        "splits": frame["split"].value_counts().to_dict() if "split" in frame else {},
    }


def _ui_stats(csv_path: Path) -> dict:
    manifest_path = csv_path.with_suffix(".manifest.json")
    csv_mtime = csv_path.stat().st_mtime if csv_path.exists() else 0.0
    manifest_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0.0
    return _cached_ui_stats(str(csv_path), csv_mtime, manifest_mtime)


def _render_stats(csv_path: Path) -> None:
    stats = _ui_stats(csv_path)
    if not stats.get("exists"):
        st.info("No generated dataset found at this path yet.")
        return

    metric_strip([
        ("Total rows", stats["rows"]),
        ("Legitimate", stats["legitimate"]),
        ("Malicious", stats["phishing"]),
        ("Duplicates", stats.get("duplicates", 0)),
        ("Near duplicates", stats.get("template_duplicates", 0)),
        ("Label conflicts", stats.get("label_conflicts", 0)),
    ])

    if stats.get("missing_label"):
        st.warning("The existing CSV does not contain the `label` column: regenerate it with the button below.")

    quality = {
        "Invalid text": stats.get("invalid_text", 0),
        "Invalid label": stats.get("invalid_label", 0),
        "Too short": stats.get("too_short", 0),
        "Too few words": stats.get("too_few_words", 0),
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
    if is_production_mode():
        st.error("Training dataset tools are disabled on the public website.")
        return

    page_intro(
        "Project tools",
        "Training datasets",
        "Build and inspect the balanced dataset used to train the FishStop content classifier.",
    )
    st.markdown(
        "Seleziona le fonti pubbliche e genera un CSV pronto per DistilBERT: tutte le righe valide "
        "e deduplicate vengono mantenute, con phishing e spam nella classe malevola. Il dataset "
        "sintetico moderno viene aggiunto esclusivamente al train."
    )

    output_csv = Path(st.text_input("Complete training CSV", value=str(DEFAULT_COMPLETE_OUTPUT_CSV)))
    public_output_csv = Path(
        st.text_input("Intermediate public-only CSV", value=str(DEFAULT_BALANCED_OUTPUT_CSV))
    )
    synthetic_csv = Path(st.text_input("Synthetic augmentation CSV", value=str(DEFAULT_SYNTHETIC_CSV)))
    legitimate_hard_negative_csv = Path(
        st.text_input(
            "Legitimate hard-negative CSV",
            value=str(DEFAULT_LEGITIMATE_HARD_NEGATIVE_CSV),
        )
    )
    st.caption(
        "Format: `text,label,source,source_file,text_hash,campaign_id,split` con "
        "`0=legitimate`, `1=malicious (phishing/spam)` e split per campagna anti-leakage."
    )

    st.divider()
    st.subheader("Complete dataset status")
    _render_stats(output_csv)
    if synthetic_csv.exists():
        synthetic_stats = _ui_stats(synthetic_csv)
        st.caption(
            f"Synthetic augmentation available: {synthetic_stats['rows']} rows "
            f"({synthetic_stats['legitimate']} legitimate, {synthetic_stats['phishing']} malicious)."
        )
    else:
        st.error(f"Synthetic dataset not found: `{synthetic_csv}`")
    if legitimate_hard_negative_csv.exists():
        hard_negative_stats = _ui_stats(legitimate_hard_negative_csv)
        st.caption(
            f"Legitimate hard negatives available: {hard_negative_stats['rows']} rows "
            "(train-only augmentation)."
        )
    else:
        st.error(
            "Legitimate hard-negative dataset not found: "
            f"`{legitimate_hard_negative_csv}`"
        )

    st.divider()
    st.subheader("Sources to include")
    st.caption(
        "Le fonti vengono mescolate con split casuale riproducibile 70/10/20, stratificato per "
        "fonte e classe. Le campagne quasi duplicate restano nello stesso split. Sono ammessi "
        "anche Enron e SpamAssassin; la data dichiarata nelle singole email non viene filtrata."
    )

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
        disabled=(
            not selected_sources
            or not synthetic_csv.exists()
            or not legitimate_hard_negative_csv.exists()
        ),
        use_container_width=True,
    ):
        _run_builder(
            "Generating and validating complete training dataset...",
            builder=build_complete_training_dataset,
            selected_sources=selected_sources,
            output_csv=output_csv,
            public_output_csv=public_output_csv,
            synthetic_csv=synthetic_csv,
            legitimate_hard_negative_csv=legitimate_hard_negative_csv,
            staging_csv=DEFAULT_OUTPUT_CSV,
        )
        st.rerun()

    st.caption(
        "Quality rule: synthetic emails remain train-only and may represent at most 10% of the training split. "
        "Validation and test contain only public/real emails."
    )

    with st.expander("Advanced: generate only the balanced public dataset", expanded=False):
        if st.button(
        "Generate public-only mixed dataset",
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
