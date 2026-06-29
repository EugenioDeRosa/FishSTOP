import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import torch

from src.components.email_globe import render_email_globe
from src.views.backend import get_backend


def _strip_encoded_content(raw: str) -> str:
    """
    Rimuove i blocchi di contenuto encoded dal testo grezzo dell'EML
    per migliorare la leggibilità nel debugger della sidebar.

    Sostituisce:
      - Blocchi base64 (righe di soli char base64, ≥ 4 righe consecutive)
      - Blocchi quoted-printable corposi (righe con =XX ≥ 4 consecutive)
    con un placeholder che indica tipo e numero di righe rimosso.
    """
    import re

    lines = raw.splitlines(keepends=True)
    result = []
    i = 0

    # Pattern per riconoscere righe base64 pure
    _b64_line = re.compile(r'^[A-Za-z0-9+/\r\n]+=*[\r\n]*$')
    # Pattern per righe quoted-printable (contengono =XX)
    _qp_line  = re.compile(r'=[0-9A-Fa-f]{2}|=$')

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Blocco base64: almeno 4 righe consecutive di soli char b64 ──
        if len(stripped) >= 20 and _b64_line.match(stripped + "\n"):
            block_start = i
            while i < len(lines) and _b64_line.match(lines[i].strip() + "\n") and len(lines[i].strip()) >= 4:
                i += 1
            n = i - block_start
            if n >= 4:
                kb = sum(len(lines[j]) for j in range(block_start, i)) * 3 // 4 // 1024
                result.append(f"[... contenuto base64 rimosso ({n} righe, ~{kb} KB) ...]\n")
                continue

            # Meno di 4 righe: non è un blocco, reinserisci normalmente
            result.extend(lines[block_start:i])
            continue

        # ── Blocco quoted-printable: almeno 4 righe con encoding =XX ──
        if _qp_line.search(stripped):
            block_start = i
            while i < len(lines) and _qp_line.search(lines[i].strip()):
                i += 1
            n = i - block_start
            if n >= 4:
                result.append(f"[... contenuto quoted-printable rimosso ({n} righe) ...]\n")
                continue
            result.extend(lines[block_start:i])
            continue

        result.append(line)
        i += 1

    return "".join(result)


def render():
    parser, validator, analyzer, tokenizer, model, model_source = get_backend()
    # ── helpers ────────────────────────────────────────────────────────────────

    def _badge(level: str) -> str:
        colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "INFO": "🔵"}
        return colors.get(level, "⚪")

    def _status_icon(ok: bool) -> str:
        return "✅" if ok else "❌"

    def _render_abuseipdb(rep: dict, label: str = ""):
        if rep["status"] == "ok":
            score = rep["abuseConfidenceScore"]
            if rep.get("isWhitelisted"):
                st.success("✅ **Whitelisted** — provider noto e affidabile")
            elif score == 0:
                st.success(f"✅ **Score: {score}/100** — nessuna segnalazione")
            elif score < 25:
                st.info(f"🟡 **Score: {score}/100** — basso rischio")
            elif score < 75:
                st.warning(f"🟠 **Score: {score}/100** — rischio moderato")
            else:
                st.error(f"🔴 **Score: {score}/100** — alto rischio!")

            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Segnalazioni",    rep["totalReports"])
            rc2.metric("Utenti distinti", rep["numDistinctUsers"])
            rc3.metric("Paese",           rep["countryCode"] or "—")

            if rep.get("isp"):
                st.caption(f"**ISP:** {rep['isp']}")
            if rep.get("domain"):
                st.caption(f"**Dominio ISP:** {rep['domain']}")
            if rep.get("usageType"):
                st.caption(f"**Tipo utilizzo:** {rep['usageType']}")
            if rep.get("lastReportedAt"):
                st.caption(f"**Ultima segnalazione:** {rep['lastReportedAt'][:10]}")

            method = rep.get("lookup_method", "")
            if method == "dns-resolved" and rep.get("resolved_ip"):
                st.caption(f"ℹ️ Dominio risolto in IP `{rep['resolved_ip']}` via DNS")

            st.markdown(f"[🔗 Apri su AbuseIPDB]({rep['url']})")

        elif rep["status"] == "skipped":
            msg = rep.get("message", "")
            if "non esiste" in msg or "NXDOMAIN" in msg or "non ha record A" in msg:
                st.warning(f"⚠️ **Dominio non risolvibile** — {msg}")
            elif "Timeout" in msg or "nameserver" in msg.lower():
                st.warning(f"⏱️ **DNS timeout** — {msg}")
            else:
                st.info(f"ℹ️ {msg}")
        else:
            st.warning(f"⚠️ {rep['message']}")

    def _render_geo(geo: dict):
        if geo["status"] == "skipped":
            st.caption(f"🌍 Geo: {geo['message']}")
            return
        if geo["status"] == "error":
            st.caption(f"🌍 Geo non disponibile: {geo['message']}")
            return

        flag = ""
        cc   = geo.get("country_code", "")
        if cc:
            try:
                flag = "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in cc.upper()) + " "
            except Exception:
                flag = ""

        location_parts = [p for p in [geo.get("city"), geo.get("region"), geo.get("country")] if p]
        location_str   = ", ".join(location_parts) if location_parts else "—"

        badges = []
        if geo.get("is_proxy"):
            badges.append("⚠️ **Proxy/VPN**")
        if geo.get("is_hosting"):
            badges.append("☁️ **Datacenter/Hosting**")
        badge_str = "  " + "  ".join(badges) if badges else ""

        st.markdown(f"🌍 {flag}**{location_str}**{badge_str}")

        details = []
        if geo.get("timezone"):
            details.append(f"🕐 `{geo['timezone']}`")
        if geo.get("isp"):
            details.append(f"ISP: `{geo['isp']}`")
        if geo.get("asn"):
            details.append(f"AS: `{geo['asn']}`")
        if details:
            st.caption("  ·  ".join(details))

        if geo.get("lat") and geo.get("lon"):
            maps_url = f"https://maps.google.com/?q={geo['lat']},{geo['lon']}"
            st.caption(f"[📍 Apri su Maps]({maps_url})  ·  lat {geo['lat']:.4f}, lon {geo['lon']:.4f}")

    def _render_virustotal(vt: dict):
        status = vt["status"]

        if status == "malicious":
            st.error(
                f"🔴 **MALEVOLO** — {vt['detection_ratio']} engine lo segnalano"
                + (f" come `{vt['threat_label']}`" if vt.get("threat_label") else "")
            )
        elif status == "suspicious":
            st.warning(
                f"🟠 **SOSPETTO** — {vt['detection_ratio']} engine lo segnalano come sospetto"
                + (f" (`{vt['threat_label']}`)" if vt.get("threat_label") else "")
            )
        elif status == "clean":
            st.success(f"✅ **PULITO** — 0 / {vt['total_engines']} engine lo segnalano")
        elif status == "not_found":
            st.info("🔵 **Non trovato su VirusTotal** — file mai sottomesso o molto recente")
            st.caption("⚠️ 'Non trovato' non significa necessariamente pulito.")
        elif status == "skipped":
            st.info(f"ℹ️ {vt['message']}")
            return
        else:
            st.warning(f"⚠️ {vt['message']}")
            return

        if status in ("malicious", "suspicious", "clean"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🔴 Malevoli",  vt["malicious"])
            mc2.metric("🟠 Sospetti",  vt["suspicious"])
            mc3.metric("✅ Puliti",    vt["undetected"])

            if vt.get("file_type"):
                st.caption(f"**Tipo file (VT):** {vt['file_type']}")
            if vt.get("file_name"):
                st.caption(f"**Nome originale (VT):** {vt['file_name']}")
            if vt.get("first_submission"):
                st.caption(f"**Prima sottomissione:** {vt['first_submission']}")
            if vt.get("last_analysis"):
                st.caption(f"**Ultima analisi:** {vt['last_analysis']}")

        st.markdown(f"[🔗 Apri report completo su VirusTotal]({vt['permalink']})")

    # ── layout ─────────────────────────────────────────────────────────────────
    col_upload, col_results = st.columns([1, 2])

    with col_upload:
        st.subheader("📥 Input Email")
        uploaded_file = st.file_uploader(
            "Trascina qui il file .eml da analizzare", type=["eml"]
        )

        if uploaded_file is not None:
            # Estrae il testo dell'email appena caricata
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

            # Controlla se il file è cambiato rispetto a quello precedentemente in memoria
            if st.session_state.get("current_eml_name") != uploaded_file.name:
                # Applica la rimozione dei blocchi Base64/QP
                raw_text_debug = _strip_encoded_content(raw_text)

                # Salva i dati puliti e il nome del file nello stato della sessione
                st.session_state["raw_eml_debug_data"] = raw_text_debug
                st.session_state["current_eml_name"] = uploaded_file.name
                st.rerun()

            st.success("File caricato correttamente! Elaborazione in corso…")
            temp_path = os.path.join("data", "raw", "temp_triage.eml")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    # ── results panel ──────────────────────────────────────────────────────────
    with col_results:
        st.subheader("📊 Pannello di Analisi e Triage")

        if uploaded_file is None:
            st.info("In attesa di un file `.eml` per avviare il triage.")
        else:
            try:
                # ── 1. DEEP HEADER ANALYSIS ────────────────────────────────
                soc = analyzer.analyze(temp_path)

                # ── 1a. Envelope / identità ────────────────────────────────
                with st.expander("📬 Envelope & Identità", expanded=True):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**Delivered-To:** `{soc['delivered_to'] or '—'}`")
                        st.markdown(f"**To:** `{soc['to'] or '—'}`")
                        st.markdown(f"**From:** `{soc['from_'] or '—'}`")
                        st.markdown(f"**Subject:** `{soc['subject'] or '—'}`")
                    with cols[1]:
                        st.markdown(f"**Date:** `{soc['date'] or '—'}`")
                        st.markdown(f"**Message-Id:** `{soc['message_id'] or '—'}`")
                        st.markdown(f"**Importance:** `{soc['importance'] or '—'}`")
                        st.markdown(f"**MIME-Version:** `{soc['mime_version'] or '—'}`")

                    st.markdown("---")
                    st.markdown(f"**Return-Path:** `{soc['return_path'] or '—'}`")
                    st.markdown(f"**Errors-To:** `{soc['errors_to'] or '—'}`")

                    if not soc.get("reply_to"):
                        reply_icon = "⚪ Assente"
                    else:
                        reply_icon = "🔴 MISMATCH" if soc["reply_to_mismatch"] else "✅ Coerente"
                    st.markdown(
                        f"**Reply-To:** `{soc['reply_to'] or '—'}` — {reply_icon}"
                    )
                    if soc["reply_to_mismatch"]:
                        st.warning(
                            "⚠️ Reply-To differisce dal From: un eventuale reply verrebbe "
                            "recapitato a un indirizzo diverso dal mittente dichiarato. "
                            "Indicatore tipico di phishing/harvesting."
                        )

                    if soc.get("return_path_domain_mismatch"):
                        st.error(
                            f"🔴 **Return-Path Mismatch** — dominio `{soc['return_path_domain']}` "
                            f"≠ dominio From `{soc['from_']}`. "
                            "I bounce vengono recapitati a un dominio diverso dal mittente dichiarato."
                        )
                    elif soc.get("return_path"):
                        st.success("✅ Return-Path coerente con il dominio From")

                    dns_embedded = soc.get("display_name_spoofing")
                    if dns_embedded:
                        st.error(
                            f"🔴 **Display Name Spoofing rilevato** — il Display Name contiene "
                            f"`{dns_embedded}` ma il mittente reale è `{soc['from_']}`. "
                            "I client di posta mostrano l'indirizzo nel nome, non quello reale."
                        )
                    st.markdown(f"**Content-Type:** `{soc['content_type'] or '—'}`")

                    # ── Reputazione domini mittente (AbuseIPDB) ───────────────
                    st.markdown("---")
                    st.markdown("**🔎 Reputazione Domini Mittente (AbuseIPDB)**")

                    _domains_to_check: dict[str, str] = {}
                    import re as _re
                    from concurrent.futures import ThreadPoolExecutor  # <-- IMPORTA QUESTO

                    def _pull_domain(raw: str | None) -> str:
                        if not raw:
                            return ""
                        m = _re.search(r"@([\w.\-]+)", raw)
                        return m.group(1).lower() if m else ""

                    _from_domain = _pull_domain(soc.get("from_"))
                    _rp_domain   = _pull_domain(soc.get("return_path"))
                    _rt_domain   = _pull_domain(soc.get("reply_to"))

                    if _from_domain:
                        _domains_to_check[f"From (`{_from_domain}`)"] = _from_domain
                    if _rp_domain and _rp_domain != _from_domain:
                        _domains_to_check[f"Return-Path (`{_rp_domain}`)"] = _rp_domain
                    if _rt_domain and _rt_domain not in (_from_domain, _rp_domain):
                        _domains_to_check[f"Reply-To (`{_rt_domain}`)"] = _rt_domain

                    if not _domains_to_check:
                        st.info("Nessun dominio mittente estraibile dagli header.")
                    else:
                        # Creiamo una funzione helper per parallelizzare il check e il potenziale fallback
                        def _fetch_domain_data(label, domain):
                            # Chiamata principale
                            rep = validator.check_domain_reputation(domain)

                            # Logica di fallback sul parent domain (se il sottodominio non esiste)
                            if rep["status"] == "skipped" and ("non esiste" in rep.get("message", "") or "NXDOMAIN" in rep.get("message", "")):
                                parts = domain.split(".")
                                if len(parts) >= 3 and len(parts[-2]) <= 3 and parts[-1] in ["uk", "it", "au", "br", "za", "jp"]:
                                    _parent_dom = ".".join(parts[-3:])
                                elif len(parts) > 2:
                                    _parent_dom = ".".join(parts[-2:])
                                else:
                                    _parent_dom = domain

                                if _parent_dom != domain:
                                    # Esegue la seconda chiamata di fallback
                                    rep = validator.check_domain_reputation(_parent_dom)
                                    rep["used_parent_fallback"] = _parent_dom  # Flag custom per la UI

                            return label, domain, rep

                        # Eseguiamo i controlli di reputazione in parallelo usando i thread
                        with st.spinner("Analisi reputazione domini in corso in parallelo..."):
                            with ThreadPoolExecutor(max_workers=3) as executor:
                                # Lanciamo i thread per ogni dominio da controllare
                                futures = [
                                    executor.submit(_fetch_domain_data, _lbl, _dom)
                                    for _lbl, _dom in _domains_to_check.items()
                                ]
                                # Raccogliamo i risultati completati
                                parallel_results = [f.result() for f in futures]

                        # Ora che abbiamo TUTTI i dati istantaneamente, eseguiamo solo il rendering visivo
                        for _lbl, _dom, _dom_rep in parallel_results:
                            with st.expander(f"🌐 {_lbl}"):
                                # Se è stato usato il fallback, lo notifichiamo all'utente
                                if "_parent_dom" in _dom_rep.get("message", "") or _dom_rep.get("used_parent_fallback"):
                                    parent_used = _dom_rep.get("used_parent_fallback", "parent")
                                    st.warning(f"⚠️ Il dominio `{_dom}` non era risolvibile. È stato analizzato il dominio parent: `{parent_used}`")

                                _render_abuseipdb(_dom_rep, label=_lbl)
                # ── 1b. Catena Received ────────────────────────────────────
                with st.expander("📡 Catena Received (routing hop-by-hop)"):
                    hops = soc["received_hops"]
                    if not hops:
                        st.info("Nessun header Received trovato.")
                    else:
                        labels = []
                        for i, _ in enumerate(hops):
                            if i == 0:
                                labels.append("Hop 1 — Closest to Recipient (server ricevente)")
                            elif i == len(hops) - 1:
                                labels.append(f"Hop {i+1} — Closest to Sender (server di origine)")
                            elif i == 1:
                                labels.append(f"Hop {i+1} — Injection Server (server usato dal mittente)")
                            else:
                                labels.append(f"Hop {i+1} — Relay intermedio")

                        for label, hop in zip(labels, hops):
                            st.markdown(f"**{label}**")
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(f"From host: `{hop.get('from_host') or '—'}`")
                            c2.markdown(f"Sender IP: `{hop.get('sender_ip') or '—'}`")
                            c3.markdown(f"By host: `{hop.get('by_host') or '—'}`")
                            if hop.get("sender_domain"):
                                st.markdown(f"Sender domain (parenthetical): `{hop['sender_domain']}`")
                            if hop.get("tls_version"):
                                st.markdown(
                                    f"TLS: `{hop['tls_version']}` — Cipher: `{hop['tls_cipher']}`"
                                )
                            if hop.get("for_address"):
                                st.markdown(f"For: `{hop['for_address']}`")

                            all_ips = hop.get("all_ips") or (
                                [hop["sender_ip"]] if hop.get("sender_ip") else []
                            )
                            for _ip in all_ips:
                                if _ip == hop.get("sender_ip"):
                                    _ip_role = "Sender"
                                else:
                                    _ip_role = "By (ricevente)"

                                with st.spinner(f"Geolocalizzazione {_ip}…"):
                                    _geo = validator.geolocate_ip(_ip)
                                _render_geo(_geo)

                                with st.expander(f"🔍 Reputazione AbuseIPDB `{_ip}` ({_ip_role})"):
                                    with st.spinner(f"Interrogazione AbuseIPDB per {_ip}…"):
                                        ip_rep = validator.check_ip_reputation(_ip)
                                    _render_abuseipdb(ip_rep)

                            with st.expander("Raw Received header"):
                                st.code(hop["raw"], language="text")
                            st.markdown("---")
                # ── 1b-bis. Mappa percorso geografico ─────────────────────
                with st.expander("🌍 Percorso geografico email", expanded=True):
                    render_email_globe(soc, validator)

                # ── 1c. Autenticazione ─────────────────────────────────────
                with st.expander("🔑 Autenticazione (SPF / DKIM / DMARC)", expanded=True):

                    spf_live = validator.check_spf(
                        sender_ip  = soc.get("injection_sender_ip") or "",
                        mail_from  = soc.get("return_path") or soc.get("from_") or "",
                        helo_domain= (soc.get("injection_server") or {}).get("from_host") or "",
                    )

                    dkim_live = validator.check_dkim(soc.get("raw_eml_bytes") or b"")

                    dmarc_live = validator.check_dmarc(
                        from_address = soc.get("from_") or "",
                        spf_result   = spf_live["status"],
                        spf_domain   = spf_live.get("domain") or "",
                        dkim_results = dkim_live.get("signatures") or [],
                    )

                    auth_header = soc["auth_results"] or soc["arc_auth_results"]

                    col_spf, col_dkim, col_dmarc = st.columns(3)

                    with col_spf:
                        st.markdown("#### SPF")
                        status = spf_live["status"]
                        if status == "pass":
                            st.success(f"PASS ✅")
                        elif status in ("fail", "softfail"):
                            st.error(f"{status.upper()} ❌")
                        elif status in ("none", "neutral"):
                            st.warning(f"{status.upper()} ⚠️")
                        elif status == "record-found":
                            st.warning("Record trovato (pyspf non installato)")
                        else:
                            st.warning(f"{status.upper()}")

                        st.caption(f"Sender IP: `{spf_live.get('sender_ip') or '—'}`")
                        st.caption(f"MAIL FROM domain: `{spf_live.get('domain') or '—'}`")
                        st.caption(f"Libreria: `{spf_live.get('library')}`")

                        if spf_live.get("record"):
                            with st.expander("Record SPF"):
                                st.code(spf_live["record"], language="text")
                        if soc.get("received_spf_raw"):
                            with st.expander("Received-SPF (header MTA)"):
                                st.code(soc["received_spf_raw"], language="text")

                        spf_header = auth_header.get("SPF")
                        if spf_header:
                            match = spf_header["status"] == status
                            icon = "✅" if match else "⚠️ diverge"
                            st.caption(f"Authentication-Results header: `{spf_header['status']}` {icon}")

                    with col_dkim:
                        st.markdown("#### DKIM")
                        dkim_status = dkim_live["status"]
                        if dkim_status == "pass":
                            st.success("PASS ✅")
                        elif dkim_status == "fail":
                            st.error("FAIL ❌")
                        elif dkim_status in ("none", "present"):
                            st.warning(f"{'ASSENTE 🚫' if dkim_status == 'none' else 'PRESENTE (non verificato) ⚠️'}")
                        else:
                            st.warning(f"{dkim_status.upper()}")

                        st.caption(f"Libreria: `{dkim_live.get('library')}`")
                        st.caption(dkim_live.get("message", ""))

                        for sig in dkim_live.get("signatures") or []:
                            sig_ok = sig["result"] == "pass"
                            label  = f"Firma #{sig['index']+1} — `{sig.get('d_domain','?')}` s=`{sig.get('selector','?')}`"
                            if sig_ok:
                                st.success(label + " ✅")
                            else:
                                st.error(label + " ❌")
                            st.caption(f"DNS key record: `{sig.get('dns_key_record','')}`")
                            st.caption(sig.get("message",""))

                        dkim_header = auth_header.get("DKIM")
                        if dkim_header:
                            match = dkim_header["status"] == dkim_status
                            icon  = "✅" if match else "⚠️ diverge"
                            st.caption(f"Authentication-Results header: `{dkim_header['status']}` {icon}")

                    with col_dmarc:
                        st.markdown("#### DMARC")
                        dmarc_status = dmarc_live["status"]
                        if dmarc_status == "pass":
                            st.success("PASS ✅")
                        elif dmarc_status == "fail":
                            st.error("FAIL ❌")
                        elif dmarc_status == "none":
                            st.warning("NESSUN RECORD ⚠️")
                        else:
                            st.warning(f"{dmarc_status.upper()}")

                        st.caption(f"Policy: `{dmarc_live.get('policy','—')}` ({dmarc_live.get('pct',100)}%)")
                        st.caption(f"adkim: `{dmarc_live.get('adkim','r')}` · aspf: `{dmarc_live.get('aspf','r')}`")
                        spf_align_icon  = "✅" if dmarc_live.get("spf_aligned")  else "❌"
                        dkim_align_icon = "✅" if dmarc_live.get("dkim_aligned") else "❌"
                        st.caption(f"SPF allineato: {spf_align_icon} · DKIM allineato: {dkim_align_icon}")

                        if dmarc_live.get("record"):
                            with st.expander("Record DMARC"):
                                st.code(dmarc_live["record"], language="text")
                        if dmarc_live.get("rua"):
                            st.caption(f"RUA: `{dmarc_live['rua']}`")

                        dmarc_header = auth_header.get("DMARC")
                        if dmarc_header:
                            match = dmarc_header["status"] in ("pass","bestguesspass") and dmarc_status == "pass"
                            icon  = "✅" if match else "⚠️ diverge"
                            st.caption(f"Authentication-Results header: `{dmarc_header['status']}` {icon}")

                    st.markdown("---")
                    st.markdown("**Riepilogo allineamento DMARC**")
                    c1, c2 = st.columns(2)
                    c1.markdown(
                        f"SPF domain (`{spf_live.get('domain','—')}`) vs "
                        f"From domain (`{dmarc_live.get('domain','—')}`) — "
                        f"modalità `{dmarc_live.get('aspf','r')}`: "
                        + ("✅ allineato" if dmarc_live.get("spf_aligned") else "❌ non allineato")
                    )
                    dkim_sigs = dkim_live.get("signatures") or []
                    if dkim_sigs:
                        passing_sigs = [s for s in dkim_sigs if s["result"] == "pass"]
                        for s in passing_sigs:
                            c2.markdown(
                                f"DKIM d=`{s.get('d_domain','?')}` vs "
                                f"From domain (`{dmarc_live.get('domain','—')}`) — "
                                f"modalità `{dmarc_live.get('adkim','r')}`: "
                                + ("✅ allineato" if dmarc_live.get("dkim_aligned") else "❌ non allineato")
                            )
                    else:
                        c2.markdown("Nessuna firma DKIM da verificare per l'allineamento")

                    if soc["arc_seal"]:
                        st.markdown("---")
                        st.markdown("**ARC Headers (intermediary signing)**")
                        with st.expander("ARC-Seal"):
                            st.code(soc["arc_seal"], language="text")
                        if soc["arc_message_signature"]:
                            with st.expander("ARC-Message-Signature"):
                                st.code(soc["arc_message_signature"], language="text")
                        if soc["arc_authentication_results"]:
                            with st.expander("ARC-Authentication-Results"):
                                st.code(soc["arc_authentication_results"], language="text")

                # ── 1d. Allegati ───────────────────────────────────────────
                attachments = soc.get("attachments", [])
                with st.expander(f"📎 Allegati ({len(attachments)} trovati)"):
                    if not attachments:
                        st.info("Nessun allegato rilevato.")
                    for att in attachments:
                        st.markdown(f"### `{att['filename']}`")
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"**Content-Type:** `{att['content_type']}`")
                        c2.markdown(f"**Encoding:** `{att['encoding']}`")
                        c3.markdown(f"**Ext. da filename:** `{att['extension_from_filename'] or '—'}`")

                        c4, c5 = st.columns(2)
                        c4.markdown(
                            f"**Magic Bytes (hex, primi 8B):** "
                            f"`{att['magic_bytes_hex'][:16] + '…' if att['magic_bytes_hex'] else '—'}`"
                        )
                        c5.markdown(
                            f"**Formato rilevato (magic):** `{att['magic_detected_format'] or '—'}`"
                        )
                        if att.get("size_bytes") is not None:
                            sz = att["size_bytes"]
                            sz_str = f"{sz:,} B" if sz < 1024 else f"{sz/1024:.1f} KB" if sz < 1_048_576 else f"{sz/1_048_576:.2f} MB"
                            st.caption(f"📦 Dimensione: **{sz_str}**")

                        match_ok = att.get("extension_match")
                        if match_ok is True:
                            st.success("✅ Estensione, Content-Type e magic bytes coerenti")
                        elif att.get("anomaly"):
                            st.error(f"🔴 Anomalia: {att['anomaly']}")
                        else:
                            st.warning("⚠️ Impossibile verificare la coerenza (dati insufficienti)")

                        sha256 = att.get("hash_sha256")
                        if sha256:
                            st.markdown("**🔐 Hash crittografici**")
                            hc1, hc2, hc3 = st.columns(3)
                            hc1.caption("MD5")
                            hc1.code(att["hash_md5"],  language="text")
                            hc2.caption("SHA-1")
                            hc2.code(att["hash_sha1"], language="text")
                            hc3.caption("SHA-256")
                            hc3.code(sha256,           language="text")

                            st.markdown("**🛡️ VirusTotal — Threat Intelligence**")
                            with st.spinner(f"Interrogazione VirusTotal per `{sha256[:16]}…`"):
                                vt_result = validator.check_file_hash(sha256)
                            _render_virustotal(vt_result)

                            with st.expander("🔍 Altri servizi threat intelligence"):
                                lc1, lc2 = st.columns(2)
                                lc1.markdown(
                                    f"[🧪 Any.run](https://app.any.run/tasks/#{sha256})"
                                )
                                lc2.markdown(
                                    f"[🦅 Hybrid Analysis](https://www.hybrid-analysis.com/search?query={sha256})"
                                )
                                st.caption(
                                    "⚠️ Prima di caricare un allegato su servizi online, "
                                    "verifica che non contenga dati riservati o PII."
                                )

                        st.markdown("---")

                # ── 1e-bis. Link & Lookalike Domains ──────────────────────
                links            = soc.get("links", [])
                lookalike_alerts = soc.get("lookalike_alerts", [])
                n_links     = len(links)
                n_lookalike = len(lookalike_alerts)
                _exp_label = f"🔗 Link & Lookalike Domains ({n_links} link"
                if n_lookalike:
                    _exp_label += f", ⚠️ {n_lookalike} sospetti)"
                else:
                    _exp_label += ", nessun sospetto)"
                with st.expander(_exp_label, expanded=bool(n_lookalike)):
                    if not links:
                        st.info("Nessun URL trovato nel corpo dell'email.")
                    else:
                        if lookalike_alerts:
                            st.markdown("#### ⚠️ Lookalike / Typosquatting Alerts")
                            for la in lookalike_alerts:
                                technique_icon = {
                                    "edit_distance": "✏️",
                                    "homoglyph":     "🔤",
                                    "typosquatting": "🔀",
                                }.get(la["technique"], "⚠️")
                                technique_name = {
                                    "edit_distance": "Edit-distance",
                                    "homoglyph":     "Omoglifi Unicode",
                                    "typosquatting": "Typosquatting",
                                }.get(la["technique"], la["technique"])
                                st.error(
                                    f"{technique_icon} **{technique_name}** — "
                                    f"`{la['host']}` assomiglia a **{la['matched_brand']}**"
                                    + (f" (distanza edit: {la['edit_distance']})" if la.get("edit_distance") else "")
                                )
                                st.caption(f"🔍 Dettaglio: {la['detail']}")
                                st.caption(f"🌐 URL completo: `{la['url']}`")
                                st.markdown("---")
                        else:
                            st.success("✅ Nessun dominio lookalike / typosquatting rilevato.")

                        st.markdown("#### 🔗 Tutti i link estratti")

                        _src_icon = {
                            "html_href":   "🖇️ `<a href>`",
                            "html_text":   "📄 testo HTML",
                            "plain_text":  "📝 testo plain",
                        }

                        for lnk in links:
                            _ip_badge = " 🔴 **IP diretto**" if lnk["is_ip"] else ""
                            _la_match = any(a["host"] == lnk["host"] for a in lookalike_alerts)
                            _la_badge = " ⚠️ **lookalike**" if _la_match else ""
                            _scheme_badge = (
                                " 🔓" if lnk["scheme"] == "http" else
                                " 🔒" if lnk["scheme"] == "https" else ""
                            )
                            _header = f"{_src_icon.get(lnk['source'], lnk['source'])}{_scheme_badge} `{lnk['host']}`{_ip_badge}{_la_badge}"
                            with st.container(border=False):
                                lc1, lc2 = st.columns([3, 1])
                                with lc1:
                                    st.markdown(f"**{_header}**")
                                    st.caption(f"`{lnk['url'][:120]}`" + ("…" if len(lnk["url"]) > 120 else ""))
                                with lc2:
                                    st.markdown(
                                        f"[🔍 VT](https://www.virustotal.com/gui/domain/{lnk['host']})"
                                        f" · [🌐 WHOIS](https://www.whois.com/whois/{lnk['host']})"
                                    )
                            st.divider()

                # ── 1e. Corpo testo ────────────────────────────────────────
                with st.expander("📄 Corpo Email (testo estratto)"):
                    body_source = soc.get("body_source", "unknown")
                    strip_applied = soc.get("html_strip_applied", False)

                    if body_source == "text/plain":
                        st.caption("📝 Sorgente: `text/plain` — nessuno stripping necessario")
                    elif body_source == "text/html":
                        st.caption("🌐 Sorgente: `text/html` — stripping HTML applicato prima dell'analisi AI")
                    else:
                        st.caption("⚠️ Corpo email non rilevato")

                    if strip_applied:
                        tab_clean, tab_raw = st.tabs(["✅ Testo pulito (input BERT)", "🔍 HTML grezzo originale"])
                        with tab_clean:
                            st.text_area(
                                "Testo dopo HTML stripping:",
                                soc.get("body_clean") or "(vuoto dopo stripping)",
                                height=220,
                            )
                        with tab_raw:
                            st.code(soc.get("body_html") or "(nessun HTML)", language="html")
                    else:
                        st.text_area("Body:", soc.get("body_clean") or soc["body"] or "(vuoto)", height=220)

                # ── 1f. Flags SOC summary ──────────────────────────────────
                st.subheader("🚨 Riepilogo Flags SOC")
                flags = soc.get("flags", [])
                if not flags:
                    st.success("Nessun flag critico rilevato.")
                else:
                    for f in flags:
                        icon = _badge(f["level"])
                        lvl  = f["level"]
                        if lvl == "HIGH":
                            st.error(f"{icon} **[{lvl}] {f['field']}** — {f['message']}")
                        elif lvl == "MEDIUM":
                            st.warning(f"{icon} **[{lvl}] {f['field']}** — {f['message']}")
                        elif lvl == "LOW":
                            st.warning(f"{icon} **[{lvl}] {f['field']}** — {f['message']}")
                        else:
                            st.info(f"{icon} **[{lvl}] {f['field']}** — {f['message']}")

                st.divider()

                # ── 2. AI CONTENT ANALYSIS (BERT) ─────────────────────────
                st.subheader("🤖 Analisi Contenuto con Intelligenza Artificiale (BERT)")
                if model_source == "company":
                    st.success("🏢 Modello **aziendale** attivo — addestrato sulle email della tua organizzazione.")
                else:
                    st.info("🌐 Modello **base** attivo (Kaggle-BERT). Popola il dataset e addestra il tuo modello personalizzato nel Dataset Builder.")

                clean_body = soc.get("body_clean") or soc.get("body") or ""
                email_text = f"Subject: {soc['subject'] or ''}\n\n{clean_body}".strip()

                if soc.get("html_strip_applied"):
                    st.caption("ℹ️ Input BERT: testo estratto dopo HTML stripping — tag e commenti offuscanti rimossi.")

                if not email_text or email_text.lower() == "subject:":
                    st.warning("⚠️ Impossibile eseguire la classificazione: l'email non contiene testo significativo nel corpo o nell'oggetto.")
                else:
                    with st.spinner("Messa a punto dei token... BERT sta analizzando il testo..."):
                        inputs = tokenizer(
                            email_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        )

                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probabilities = torch.softmax(logits, dim=1).flatten().tolist()

                        prob_safe     = probabilities[0] * 100
                        prob_phishing = probabilities[1] * 100

                        if prob_phishing > prob_safe:
                            st.error(f"🚨 **Risultato IA: RILEVATO POSSIBILE PHISHING**")
                            st.progress(int(prob_phishing))
                            st.write(f"**Confidenza del Modello:** {prob_phishing:.2f}% Probability Phishing")
                        else:
                            st.success(f"🟢 **Risultato IA: EMAIL LEGITTIMA**")
                            st.progress(int(prob_phishing))
                            st.write(f"**Confidenza del Modello:** {prob_safe:.2f}% Probability Legitimate")

                        with st.expander("Vedi metriche grezze dei logit"):
                            st.json({
                                "Logits (Safe, Phishing)": logits.flatten().tolist(),
                                "Probabilità Safe": f"{prob_safe:.4f}%",
                                "Probabilità Phishing": f"{prob_phishing:.4f}%"
                            })

                # ── cleanup ────────────────────────────────────────────────
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                st.error(f"Si è verificato un errore durante l'analisi: {str(e)}")
                import traceback
