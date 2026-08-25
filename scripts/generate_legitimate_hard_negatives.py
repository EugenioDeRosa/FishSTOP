"""Generate modern legitimate hard negatives for FishSTOP training.

The corpus is intentionally train-only. It targets benign messages that contain
words and structures commonly associated with phishing without copying the
custom evaluation emails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fishstop_synthetic_legitimate_hard_negatives_v1.csv"
)
SOURCE = "synthetic_legitimate_hard_negative_v1"


@dataclass(frozen=True)
class Event:
    subject: str
    detail: str
    action: str


@dataclass(frozen=True)
class Category:
    key: str
    rationale: str
    it_entities: tuple[str, ...]
    en_entities: tuple[str, ...]
    it_events: tuple[Event, ...]
    en_events: tuple[Event, ...]


IT_TEMPLATES = (
    (
        "Oggetto: {subject}\n\nBuongiorno,\n{detail} Il servizio coinvolto è {entity}. "
        "{action} Non comunicare password o codici via email; per dubbi usa i contatti "
        "pubblicati nella intranet.\n\nGrazie,\n{signature}"
    ),
    (
        "Oggetto: {subject}\n\nCiao,\nquesta è una comunicazione informativa relativa a "
        "{entity}. {detail} {action} Se hai già completato l'operazione puoi ignorare il "
        "promemoria.\n\nCordiali saluti,\n{signature}"
    ),
    (
        "Oggetto: {subject}\n\nGentile collega,\nPer {entity}: {detail}\n\nIndicazione operativa: "
        "{action} Accedi sempre dal preferito aziendale o dall'app ufficiale, senza usare "
        "link ricevuti da mittenti sconosciuti.\n\n{signature}"
    ),
    (
        "Oggetto: {subject}\n\nSalve,\nper {entity} registriamo il seguente aggiornamento: "
        "{detail} {action} Per assistenza apri una richiesta dal catalogo servizi interno "
        "indicando l'oggetto di questa comunicazione.\n\nA disposizione,\n{signature}"
    ),
)

EN_TEMPLATES = (
    (
        "Subject: {subject}\n\nHello,\n{detail} This notice concerns {entity}. {action} "
        "Never send passwords or verification codes by email; use the support details in "
        "the company directory if you need help.\n\nRegards,\n{signature}"
    ),
    (
        "Subject: {subject}\n\nHi,\nthis is an informational message about {entity}. "
        "{detail} {action} If you have already completed the task, no further action is "
        "required.\n\nBest regards,\n{signature}"
    ),
    (
        "Subject: {subject}\n\nDear colleague,\nFor {entity}: {detail}\n\nRecommended step: {action} "
        "Open the service from your saved corporate portal or its official application, "
        "not from an unexpected message.\n\n{signature}"
    ),
    (
        "Subject: {subject}\n\nGood morning,\n{entity} recorded this routine update: "
        "{detail} {action} For assistance, open a request from the internal service catalog "
        "and quote this notification's subject.\n\n{signature}"
    ),
)

IT_SIGNATURES = (
    "Service Desk — Aurora Manufacturing",
    "Sistemi Informativi — Università Delta",
    "Amministrazione — Cooperativa Orione",
    "Customer Care — Banca Esempio",
)
EN_SIGNATURES = (
    "IT Service Desk — Northwind Research",
    "Operations Team — Contoso Europe",
    "Customer Support — Example Financial",
    "People Services — Blue Yonder Group",
)


CATEGORIES = (
    Category(
        "password",
        "Benign password expiry, rotation, and confirmation notices.",
        ("Portale HR Nova", "Area Ricerca Ateneo", "Gestionale RDAP", "Workspace Atlas"),
        ("Employee Hub", "Research Portal", "Document Registry", "Atlas Workspace"),
        (
            Event("Promemoria scadenza password", "La password scadrà tra sette giorni secondo la normale policy annuale.", "Apri il portale dal segnalibro e scegli Impostazioni > Sicurezza."),
            Event("Cambio password completato", "La modifica richiesta questa mattina è stata registrata correttamente.", "Se non riconosci l'attività telefona al service desk usando il numero interno."),
            Event("Rotazione credenziali programmata", "Nel prossimo accesso verrà proposto il rinnovo periodico delle credenziali.", "Segui la procedura mostrata dopo il normale login aziendale."),
        ),
        (
            Event("Password expiry reminder", "Your password is due to expire in seven days under the annual security policy.", "Open the bookmarked portal and select Settings, then Security."),
            Event("Password change confirmed", "The password change requested earlier today was completed successfully.", "If this was not you, call the internal service desk using the staff directory."),
        ),
    ),
    Category(
        "vpn_mfa",
        "Legitimate VPN, MFA, token activation, and certificate maintenance.",
        ("VPN Sede Centrale", "FortiToken Aziendale", "Accesso Remoto Tecnici", "Portale MFA"),
        ("Corporate VPN", "Mobile Authenticator", "Remote Engineering Access", "MFA Portal"),
        (
            Event("Attivazione token approvata", "L'assegnazione del token mobile richiesta dal responsabile è disponibile per la configurazione.", "Avvia l'app aziendale e associa il dispositivo dalla sezione Sicurezza."),
            Event("Rinnovo certificato VPN", "Il certificato del profilo VPN arriverà alla normale scadenza questo mese.", "Connetti il computer alla rete aziendale e avvia l'aggiornamento dal client installato."),
            Event("Metodo MFA registrato", "Il nuovo metodo di autenticazione è stato associato al tuo account di lavoro.", "Se riconosci l'operazione non devi fare altro; altrimenti apri un ticket interno."),
        ),
        (
            Event("VPN certificate renewal", "The certificate used by your managed VPN profile reaches its scheduled expiry this month.", "Connect through the managed client and start the update from its settings page."),
            Event("Authenticator enrollment confirmed", "The mobile authentication method requested through IT has been linked to your work account.", "No action is needed if you recognize it; otherwise open an internal support ticket."),
        ),
    ),
    Category(
        "banking",
        "Authentic-looking but benign card, account, and transaction notifications.",
        ("App Banca Esempio", "Carta Aziendale Orione", "Tesoreria Digitale", "Portale Rimborsi"),
        ("Example Banking App", "Corporate Travel Card", "Treasury Online", "Expense Portal"),
        (
            Event("Carta virtuale creata", "È stata generata una carta virtuale di prova con importo massimo di un euro.", "Controlla i dettagli direttamente nell'app; se non la riconosci chiama il numero sul retro della carta fisica."),
            Event("Pagamento contabilizzato", "Un pagamento autorizzato è stato contabilizzato sul conto con la data prevista.", "Verifica il movimento nell'area riservata senza rispondere a questa email automatica."),
            Event("Estratto conto disponibile", "Il nuovo estratto conto mensile è pronto nell'archivio documenti.", "Apri l'app ufficiale e consulta Documenti > Estratti conto."),
        ),
        (
            Event("Virtual card created", "A low-limit virtual card requested through the approved workflow is now active.", "Review it in the official app; call the number printed on the physical card if it is unexpected."),
            Event("Monthly statement available", "Your monthly account statement has been published in the secure document archive.", "Open the official application and select Documents, then Statements."),
        ),
    ),
    Category(
        "security_access",
        "Routine security, sign-in, device compliance, and training notices.",
        ("Centro Sicurezza Endpoint", "Portale Formazione", "Microsoft 365 Aziendale", "Console Dispositivi"),
        ("Endpoint Security Center", "Learning Portal", "Corporate Microsoft 365", "Device Console"),
        (
            Event("Accesso recente verificato", "È stato registrato un accesso dal computer aziendale assegnato al tuo profilo.", "Consulta la cronologia dal portale abituale solo se desideri verificare data e posizione."),
            Event("Promemoria formazione sicurezza", "Il corso annuale sulla protezione dei dati è disponibile fino alla fine del mese.", "Apri il catalogo formazione dalla intranet e riprendi il modulo assegnato."),
            Event("Controllo conformità dispositivo", "L'aggiornamento del sistema operativo ha riportato il dispositivo nello stato conforme.", "Non è richiesta alcuna operazione finché la console mostra il segno verde."),
        ),
        (
            Event("Recent sign-in verified", "A sign-in from your assigned corporate laptop was recorded by the normal monitoring service.", "Review the activity from your usual security dashboard only if you need the details."),
            Event("Security training reminder", "The annual data-protection course remains available until the end of the month.", "Open the learning catalog from the intranet and continue the assigned module."),
        ),
    ),
    Category(
        "attachments",
        "Legitimate corporate documents and attachments with potentially risky wording.",
        ("Ufficio Formazione", "Amministrazione Fornitori", "Segreteria Progetti", "Qualità e Compliance"),
        ("Training Office", "Supplier Administration", "Project Office", "Quality and Compliance"),
        (
            Event("Certificati formazione allegati", "In allegato trovi i certificati PDF richiesti per completare il fascicolo del personale.", "Archiviali nella cartella del progetto; non contengono macro né moduli di accesso."),
            Event("Copia documento contabile", "È allegata la copia della fattura già registrata e approvata nel gestionale.", "Confronta il numero del documento con l'ordine presente nel portale acquisti."),
            Event("Verbale riunione disponibile", "Il verbale in formato DOCX riassume decisioni, responsabili e prossime scadenze.", "Salvalo nello spazio condiviso del gruppo dopo la normale revisione."),
        ),
        (
            Event("Requested training certificates attached", "The attached PDF files are the training certificates requested for the employee record.", "Store them in the approved project folder; they contain no macros or login forms."),
            Event("Approved invoice copy attached", "A copy of the invoice already approved in the procurement system is attached for reference.", "Match its document number with the purchase order in the saved procurement portal."),
        ),
    ),
    Category(
        "newsletter",
        "Opt-in newsletters, product updates, music analytics, and event invitations.",
        ("Studio Artisti", "Community Sviluppatori", "Programma Clienti", "Osservatorio Innovazione"),
        ("Artist Studio", "Developer Community", "Customer Program", "Innovation Network"),
        (
            Event("Il tuo contenuto è online", "La nuova pubblicazione è disponibile e le prime statistiche di ascolto sono state elaborate.", "Apri la dashboard ufficiale per consultare pubblico, andamento e suggerimenti editoriali."),
            Event("Novità del profilo", "Sono disponibili nuovi strumenti per aggiornare immagine, presentazione e contenuti in evidenza.", "Gestisci il profilo dalla normale area creator oppure ignora questa newsletter."),
            Event("Invito al webinar mensile", "Il prossimo incontro presenta funzioni già incluse nel piano e casi d'uso dei clienti.", "Iscriviti dal calendario eventi del sito ufficiale se l'argomento ti interessa."),
        ),
        (
            Event("Your new release is live", "The new publication is available and its first audience statistics have been processed.", "Open the official creator dashboard to review performance and editorial suggestions."),
            Event("Monthly product newsletter", "This opt-in update summarizes new profile tools, release notes, and community events.", "Read it in the normal customer portal or ignore the newsletter if it is not relevant."),
        ),
    ),
    Category(
        "forwarded_threads",
        "Normal replies, forwards, quoted history, and internal coordination.",
        ("Progetto Vega", "Tirocinio Sicurezza", "Ordine Cliente Europa", "Calendario Dipartimento"),
        ("Project Vega", "Security Internship", "European Customer Order", "Department Calendar"),
        (
            Event("R: aggiornamento attività", "Confermo quanto scritto nella conversazione precedente e allego il riepilogo delle attività completate.", "Rispondi al gruppo interno se manca qualche informazione."),
            Event("I: richiesta del fornitore", "Inoltro la richiesta ricevuta dal referente abituale per coordinare data e luogo della consegna.", "Verifica disponibilità e comunica la fascia oraria nella stessa conversazione."),
            Event("R: conferma appuntamento", "La riunione è confermata nell'orario concordato e l'invito di calendario è già aggiornato.", "Usa il collegamento presente nell'app Calendario al momento dell'incontro."),
        ),
        (
            Event("RE: project status update", "I confirm the points in the previous conversation and attached the agreed activity summary.", "Reply to the internal group if any project detail is missing."),
            Event("FW: supplier scheduling request", "I am forwarding the routine request from our known supplier to coordinate the delivery window.", "Confirm availability in the existing conversation without changing payment details."),
        ),
    ),
    Category(
        "remote_support",
        "Authorized remote support involving TeamViewer, VNC, and support sessions.",
        ("Service Desk Stabilimento", "Supporto Applicativo", "Team Infrastrutture", "Assistenza Macchine"),
        ("Plant Service Desk", "Application Support", "Infrastructure Team", "Equipment Support"),
        (
            Event("Sessione assistenza pianificata", "Il ticket aperto ieri prevede una sessione remota con il tecnico assegnato.", "Avvia il client già installato all'orario concordato e verifica il nome del tecnico nel ticket."),
            Event("Uso temporaneo TeamViewer autorizzato", "Per il collaudo approvato è consentito usare TeamViewer sulla postazione di test.", "Condividi l'identificativo solo nella chiamata aziendale già programmata, mai via email."),
            Event("Accesso VNC di manutenzione", "La finestra di manutenzione include un collegamento VNC dalla rete tecnica interna.", "Non aprire porte Internet; la connessione deve partire dal gateway gestito."),
        ),
        (
            Event("Scheduled remote support session", "Yesterday's support ticket includes a remote session with the assigned engineer.", "Start the managed client at the agreed time and verify the engineer name in the ticket."),
            Event("Temporary TeamViewer use approved", "TeamViewer is approved for the documented test on the lab workstation.", "Share the session identifier only during the scheduled company call, never by email."),
        ),
    ),
    Category(
        "cloud_sharing",
        "Legitimate Drive, OneDrive, forms, and document-sharing notifications.",
        ("OneDrive Progetti", "Google Workspace Aziendale", "SharePoint Qualità", "Portale Moduli"),
        ("Project OneDrive", "Corporate Google Workspace", "Quality SharePoint", "Forms Portal"),
        (
            Event("Documento condiviso con il team", "Il responsabile ha aggiunto il gruppo di progetto al documento già presente nello spazio aziendale.", "Aprilo dalla sezione Condivisi con me del portale abituale."),
            Event("Cartella di reparto disponibile", "È stato concesso l'accesso in sola lettura alla cartella con procedure e modelli aggiornati.", "Raggiungi la raccolta dalla home della intranet senza inoltrare l'invito."),
            Event("Modulo feedback pubblicato", "Il questionario interno sul corso concluso ieri è disponibile per una settimana.", "Compilalo dall'elenco Moduli assegnati; non vengono richieste password nel questionario."),
        ),
        (
            Event("Team document shared", "The project owner added your work group to an existing document in the corporate tenant.", "Open it from Shared with me in your usual workspace."),
            Event("Internal feedback form available", "The feedback form for yesterday's training session is open for one week.", "Use the Assigned forms list; the questionnaire does not request passwords."),
        ),
    ),
    Category(
        "mixed_operations",
        "Mixed operational notices with urgency, payments, deliveries, or account setup.",
        ("Portale Trasferte", "Magazzino Centrale", "People Services", "Gestione Contratti"),
        ("Travel Portal", "Central Warehouse", "People Services", "Contract Management"),
        (
            Event("Rimborso pronto per la verifica", "La nota spese è stata elaborata e attende il normale controllo del centro di costo.", "Apri la pratica dal portale trasferte e confronta ricevute e importi già inseriti."),
            Event("Ritiro ordine confermato", "Il corriere abituale ha confermato il ritiro presso il magazzino nella fascia concordata.", "Controlla il numero d'ordine nel gestionale; non sono richiesti nuovi pagamenti."),
            Event("Invito account collaboratore", "L'account temporaneo per il consulente approvato è pronto per l'attivazione amministrativa.", "Il referente deve completare la procedura dalla console interna entro la data indicata."),
        ),
        (
            Event("Expense report ready for review", "The expense report was processed and awaits the standard cost-center review.", "Open the case in the travel portal and compare the receipts already submitted."),
            Event("Order pickup confirmed", "The regular courier confirmed pickup from the warehouse during the agreed time window.", "Check the purchase order in the internal system; no additional payment is requested."),
        ),
    ),
)


def normalize_text(value: str) -> str:
    value = value.replace("\u0000", " ")
    return re.sub(r"\s+", " ", value).strip().lower()


def build_rows() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    audit: list[dict] = []

    for batch, category in enumerate(CATEGORIES, start=1):
        counter = 0
        for language, entities, events, templates, signatures in (
            ("it", category.it_entities, category.it_events, IT_TEMPLATES, IT_SIGNATURES),
            ("en", category.en_entities, category.en_events, EN_TEMPLATES, EN_SIGNATURES),
        ):
            for entity in entities:
                for event in events:
                    counter += 1
                    template = templates[(counter - 1) % len(templates)]
                    signature = signatures[(counter + batch) % len(signatures)]
                    raw_text = template.format(
                        subject=event.subject,
                        detail=event.detail,
                        action=event.action,
                        entity=entity,
                        signature=signature,
                    )
                    text = normalize_text(raw_text)
                    item_id = f"b{batch:02d}-{category.key}-{counter:02d}"
                    row = {
                        "text": text,
                        "label": 0,
                        "source": SOURCE,
                        "source_file": f"batch_{batch:02d}/{item_id}.txt",
                        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                    rows.append(row)
                    audit.append(
                        {
                            "id": item_id,
                            "batch": batch,
                            "category": category.key,
                            "language": language,
                            "subject": event.subject,
                            "entity": entity,
                            "rationale": category.rationale,
                            **row,
                        }
                    )

    return rows, audit


def validate_rows(rows: list[dict], audit: list[dict]) -> dict:
    expected_per_batch = {index: 20 for index in range(1, 11)}
    batch_counts = Counter(item["batch"] for item in audit)
    category_counts = Counter(item["category"] for item in audit)
    language_counts = Counter(item["language"] for item in audit)
    forbidden = (
        "reply with your password",
        "send your password",
        "invia la password",
        "comunica il codice otp",
    )
    problems = []

    if len(rows) != 200:
        problems.append(f"expected 200 rows, found {len(rows)}")
    if batch_counts != expected_per_batch:
        problems.append(f"unexpected batch counts: {dict(batch_counts)}")
    if len({row["text_hash"] for row in rows}) != len(rows):
        problems.append("duplicate normalized texts detected")
    if len({row["source_file"] for row in rows}) != len(rows):
        problems.append("duplicate source_file values detected")
    if any(row["label"] != 0 for row in rows):
        problems.append("a non-legitimate label was generated")
    if any(not row["source"].startswith("synthetic_") for row in rows):
        problems.append("source prefix is incompatible with train-only policy")
    if any(len(row["text"]) < 160 or len(row["text"].split()) < 25 for row in rows):
        problems.append("one or more messages are too short")
    if any(phrase in row["text"] for row in rows for phrase in forbidden):
        problems.append("unsafe credential-request phrasing detected")
    if problems:
        raise ValueError("; ".join(problems))

    return {
        "rows": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "batch_counts": dict(sorted(batch_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "unique_hashes": len({row["text_hash"] for row in rows}),
        "minimum_characters": min(len(row["text"]) for row in rows),
        "maximum_characters": max(len(row["text"]) for row in rows),
        "mean_characters": sum(len(row["text"]) for row in rows) / len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, audit = build_rows()
    summary = validate_rows(rows, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("text", "label", "source", "source_file", "text_hash"),
        )
        writer.writeheader()
        writer.writerows(rows)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps({"summary": summary, "rows": audit}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
