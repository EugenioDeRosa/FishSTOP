import os
import streamlit as st

from src.eml_dataset_builder import EmlDatasetBuilder
from src.train import BERTPhishingTrainer
from src.views.backend import get_model_source


def render():
    model_source = get_model_source()
    st.header("🗃️ Dataset Builder — Aggiungi Email al Pool di Addestramento")
    st.markdown(
        "Carica file `.eml` in batch, assegna la label corretta e arricchisci il "
        "dataset custom che verrà usato al prossimo ciclo di training."
    )

    builder = EmlDatasetBuilder()
    stats   = builder.stats()

    st.subheader("📊 Stato Dataset Custom")
    m1, m2, m3 = st.columns(3)
    m1.metric("Totale campioni", stats["total"])
    m2.metric("✅ Legittime",    stats["legitimate"])
    m3.metric("🚨 Phishing",     stats["phishing"])
    if stats["last_added"]:
        st.caption(f"Ultimo aggiornamento: {stats['last_added']}")
    st.divider()

    st.subheader("📥 Carica nuovi file .eml")
    up_col, reset_col = st.columns([5, 1])
    with up_col:
        uploaded_emls = st.file_uploader(
            "Trascina qui uno o più file .eml",
            type=["eml"],
            accept_multiple_files=True,
            key=f"builder_uploader_{st.session_state.get('builder_uploader_gen', 0)}",
        )
    with reset_col:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🗑️ Reset", use_container_width=True, type="secondary", help="Svuota l'uploader e azzera tutte le label assegnate"):
            raw_cache_now = st.session_state.get("builder_raw_cache", {})
            for fname in list(raw_cache_now.keys()):
                st.session_state.pop(f"label_{fname}", None)
            st.session_state.pop("builder_raw_cache", None)
            st.session_state["builder_uploader_gen"] = (
                st.session_state.get("builder_uploader_gen", 0) + 1
            )
            st.rerun()

    if uploaded_emls:
        n_uploaded = len(uploaded_emls)
        st.markdown(f"**{n_uploaded} file caricati.** Assegna la label prima di procedere.")

        cache_key = "builder_raw_cache"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = {}
        raw_cache: dict[str, bytes] = st.session_state[cache_key]

        for upl in uploaded_emls:
            if upl.name not in raw_cache:
                raw_cache[upl.name] = upl.read()

        with st.container(border=True):
            bc1, bc2, bc3 = st.columns([2, 1, 1])
            with bc1:
                st.markdown("**🏷️ Assegna la stessa label a tutti i file**")
                st.caption("Sovrascrive le selezioni individuali sotto.")
            with bc2:
                if st.button("✅ Tutti Legittimi", use_container_width=True):
                    for u in uploaded_emls:
                        st.session_state[f"label_{u.name}"] = 0
                    st.rerun()
            with bc3:
                if st.button("🚨 Tutti Phishing", use_container_width=True):
                    for u in uploaded_emls:
                        st.session_state[f"label_{u.name}"] = 1
                    st.rerun()

        st.divider()

        import email as _email_mod

        PREVIEW_THRESHOLD = 50

        assignments: dict[str, int] = {}

        def _quick_preview(raw: bytes) -> tuple[str, str, str]:
            try:
                msg = _email_mod.message_from_bytes(raw)
                subject = str(msg.get("Subject") or "—").strip()
                sender  = str(msg.get("From")    or "—").strip()
                body_pv = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        pl = part.get_payload(decode=True)
                        if pl:
                            lines    = [l.strip() for l in pl.decode("utf-8", errors="ignore").splitlines() if l.strip()]
                            body_pv  = " ".join(lines[:2])[:160]
                            break
                return sender, subject, body_pv
            except Exception:
                return "—", "—", ""

        def _render_file_row(upl_name: str, raw: bytes) -> int:
            sender, subject, body_pv = _quick_preview(raw)

            session_key = f"label_{upl_name}"
            existing = st.session_state.get(session_key)
            if existing is None:
                default_idx = 0
            elif isinstance(existing, tuple):
                default_idx = existing[1]
            else:
                default_idx = int(existing)

            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**📧 {upl_name}**")
                    st.caption(f"From: {sender}")
                    st.caption(f"Subject: {subject}")
                    if body_pv:
                        st.caption(f"Body: {body_pv}…")
                with c2:
                    label_choice = st.radio(
                        "Label",
                        options=[("✅ Legittima", 0), ("🚨 Phishing", 1)],
                        format_func=lambda x: x[0],
                        key=session_key,
                        index=default_idx,
                    )
            return label_choice[1]

        if n_uploaded <= PREVIEW_THRESHOLD:
            for upl in uploaded_emls:
                assignments[upl.name] = _render_file_row(upl.name, raw_cache[upl.name])
        else:
            st.info(
                f"ℹ️ {n_uploaded} file caricati. "
                "L'anteprima dettagliata è collassata per migliorare le prestazioni. "
                "Usa i bottoni **Tutti Legittimi / Tutti Phishing** per assegnare la label in blocco."
            )
            with st.expander(f"📋 Mostra anteprima di tutti i {n_uploaded} file", expanded=False):
                for upl in uploaded_emls:
                    assignments[upl.name] = _render_file_row(upl.name, raw_cache[upl.name])
            for upl in uploaded_emls:
                if upl.name not in assignments:
                    val = st.session_state.get(f"label_{upl.name}", 0)
                    assignments[upl.name] = val[1] if isinstance(val, tuple) else int(val)

        st.divider()

        if st.button("💾 Aggiungi al Dataset", type="primary", use_container_width=True):
            batch_items = [
                (raw_cache[upl.name], upl.name, assignments.get(upl.name, 0))
                for upl in uploaded_emls
                if upl.name in raw_cache
            ]

            progress_bar = st.progress(0, text="Avvio processing…")
            status_placeholder = st.empty()

            def _ui_progress(done: int, total: int) -> None:
                pct  = int(done / total * 100)
                progress_bar.progress(pct, text=f"Processing… {done}/{total}")

            results = builder.add_batch(batch_items, progress_callback=_ui_progress)

            progress_bar.progress(100, text="Completato ✅")

            added = skipped = errors = 0
            error_lines: list[str] = []

            for res in results:
                lbl       = assignments.get(res.get("message", "").split("'")[1] if "'" in res.get("message","") else "", 1)
                label_str = "Phishing 🚨" if lbl == 1 else "Legittima ✅"
                if res["status"] == "added":
                    added += 1
                elif res["status"] == "duplicate":
                    skipped += 1
                else:
                    errors += 1
                    error_lines.append(res["message"])

            st.success(f"✅ **{added} aggiunte** | ⚠️ {skipped} duplicate | ❌ {errors} errori")
            if error_lines:
                with st.expander(f"❌ Dettaglio {errors} errori"):
                    for line in error_lines:
                        st.caption(line)

            new_stats = builder.stats()
            st.info(
                f"📊 Dataset: **{new_stats['total']} campioni totali** "
                f"({new_stats['legitimate']} legittime, {new_stats['phishing']} phishing)"
            )

            st.session_state.pop(cache_key, None)
            for upl in uploaded_emls:
                st.session_state.pop(f"label_{upl.name}", None)

            st.rerun()

    st.divider()
    st.subheader("📋 Campioni nel Dataset Custom")
    df_view = builder.load_df()

    if df_view.empty:
        st.info("Il dataset è vuoto. Carica dei file .eml per iniziare.")
    else:
        filter_label = st.selectbox(
            "Filtra per label",
            options=["Tutti", "✅ Legittime (0)", "🚨 Phishing (1)"],
        )
        if filter_label == "✅ Legittime (0)":
            df_view = df_view[df_view["label"] == 0]
        elif filter_label == "🚨 Phishing (1)":
            df_view = df_view[df_view["label"] == 1]

        display_df = df_view[["source_file", "label", "added_at", "text_hash", "xt_combined"]].copy()
        display_df["xt_combined"] = display_df["xt_combined"].str[:80] + "…"
        display_df["text_hash"]   = display_df["text_hash"].str[:12] + "…"
        display_df["label"]       = display_df["label"].map({0: "✅ Legittima", 1: "🚨 Phishing"})
        st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("**🗑️ Rimuovi un campione**")
        hash_to_remove = st.text_input("Incolla il text_hash (12+ caratteri)", placeholder="es. 3a7f2c1b9e04…")
        if st.button("Rimuovi", type="secondary"):
            if not hash_to_remove or len(hash_to_remove) < 8:
                st.warning("Hash troppo corto.")
            else:
                full_hashes = df_view["text_hash"].tolist()
                matches = [h for h in full_hashes if h.startswith(hash_to_remove)]
                if not matches:
                    st.error("Nessun campione trovato.")
                elif len(matches) > 1:
                    st.error(f"Prefisso ambiguo — {len(matches)} match. Inserisci più caratteri.")
                else:
                    if builder.remove_by_hash(matches[0]):
                        st.success(f"✅ Rimosso (`{matches[0][:12]}…`)")
                        st.rerun()
                    else:
                        st.error("Rimozione fallita.")

        st.markdown("---")
        st.markdown("**🔴 Reset completo dataset**")

        if not st.session_state.get("confirm_reset_dataset"):
            if st.button("🔴 Cancella tutto il dataset", type="secondary", use_container_width=True):
                st.session_state["confirm_reset_dataset"] = True
                st.rerun()
        else:
            st.warning(
                f"⚠️ Stai per cancellare **{len(df_view)} campioni** e tutti i file .eml nelle cartelle "
                f"`custom_legitimate` e `custom_phishing`. L'operazione è **irreversibile**."
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Sì, cancella tutto", type="primary", use_container_width=True):
                    import shutil
                    import csv as _csv
                    with open(builder.csv_path, "w", newline="", encoding="utf-8") as f:
                        _csv.DictWriter(f, fieldnames=["xt_combined", "label", "source_file", "text_hash", "added_at"]).writeheader()
                    for folder in [builder.legit_folder, builder.phishing_folder]:
                        if os.path.isdir(folder):
                            shutil.rmtree(folder)
                        os.makedirs(folder, exist_ok=True)
                    st.session_state.pop("confirm_reset_dataset", None)
                    st.success("✅ Dataset resettato.")
                    st.rerun()
            with col_no:
                if st.button("❌ Annulla", use_container_width=True):
                    st.session_state.pop("confirm_reset_dataset", None)
                    st.rerun()

    st.divider()

    # ── Addestra il modello aziendale ──────────────────────────────────
    st.subheader("🧠 Addestra il Tuo Modello")

    company_path = os.path.join("models", "company_model")
    meta_path    = os.path.join(company_path, "training_meta.json")
    last_meta    = None
    if os.path.exists(meta_path):
        import json as _json
        try:
            with open(meta_path) as _f:
                last_meta = _json.load(_f)
        except Exception:
            pass

    if model_source == "company":
        st.success("✅ **Modello aziendale attivo** — l'app sta usando il tuo modello personalizzato.")
    else:
        st.info("ℹ️ **Modello base attivo** (Kaggle-BERT). Addestra il tuo modello per personalizzarlo.")

    if last_meta:
        st.caption(f"Ultimo training: {last_meta.get('trained_at','—')[:19].replace('T',' ')} UTC — "
                   f"{last_meta.get('n_train',0)+last_meta.get('n_val',0)+last_meta.get('n_test',0)} campioni totali")
        m = last_meta.get("metrics") or {}
        if m:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{m.get('accuracy',0):.2%}")
            mc2.metric("F1",        f"{m.get('f1',0):.2%}")
            mc3.metric("Precision", f"{m.get('precision',0):.2%}")
            mc4.metric("Recall",    f"{m.get('recall',0):.2%}")

    st.markdown("---")

    cur_stats = builder.stats()
    n_legit    = cur_stats["legitimate"]
    n_phishing = cur_stats["phishing"]
    n_total    = cur_stats["total"]

    tc1, tc2 = st.columns(2)
    with tc1:
        num_epochs = st.slider("Numero di epoche", min_value=1, max_value=10, value=5)
    with tc2:
        st.markdown("**Dataset disponibile**")
        st.caption(f"✅ Legittime: **{n_legit}** &nbsp;|&nbsp; 🚨 Phishing: **{n_phishing}** &nbsp;|&nbsp; Totale: **{n_total}**")
        MIN = 20
        if n_legit < MIN or n_phishing < MIN:
            st.warning(f"⚠️ Servono almeno **{MIN} campioni per classe**. "
                       f"Mancano: {max(0, MIN-n_legit)} legittime, {max(0, MIN-n_phishing)} phishing.")
        elif max(n_legit, n_phishing) / max(min(n_legit, n_phishing), 1) > 5:
            st.warning("⚠️ Dataset sbilanciato — considera di aggiungere più email della classe minoritaria.")
        else:
            st.success("✅ Dataset pronto per il training.")

    can_train = False
    if st.button("🚀 Avvia Training", type="primary",
                 disabled=not can_train,
                 use_container_width=True):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def _ui_progress(step: str, pct: int):
            progress_bar.progress(pct)
            status_text.caption(f"⏳ {step}")

        with st.spinner("Training in corso… non chiudere questa pagina."):
            try:
                trainer_obj = BERTPhishingTrainer()
                result = trainer_obj.finetune_on_custom(
                    base_model_path="./models/saved_models",
                    output_dir="./models/company_model",
                    num_epochs=num_epochs,
                    progress_callback=_ui_progress,
                )
            except Exception as _exc:
                result = {"status": "error", "message": str(_exc), "metrics": None}

        progress_bar.progress(100)
        status_text.empty()

        if result["status"] == "ok":
            st.success(f"✅ **Training completato!** {result['message']}")
            m = result.get("metrics") or {}
            if m:
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Accuracy",  f"{m.get('accuracy',0):.2%}")
                rc2.metric("F1",        f"{m.get('f1',0):.2%}")
                rc3.metric("Precision", f"{m.get('precision',0):.2%}")
                rc4.metric("Recall",    f"{m.get('recall',0):.2%}")
            st.info("🔄 Riavvia l'app (`streamlit run`) per caricare il nuovo modello aziendale.")
        elif result["status"] == "insufficient_data":
            st.warning(f"⚠️ {result['message']}")
        else:
            st.error(f"❌ Errore durante il training: {result['message']}")
