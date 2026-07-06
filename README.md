# FishSTOP

FishSTOP e una piattaforma locale per il triage SOC di email sospette in formato `.eml`.
Il progetto combina parsing forense dell'email, controlli di autenticazione, reputazione di link/allegati, analisi del contenuto con BERT e spiegazione locale con Phi-4 mini tramite Ollama.

Il flusso principale e pensato per una tesi: carichi un file `.eml`, il sistema estrae header, body, link e allegati, produce indicatori tecnici e restituisce una valutazione leggibile per l'analista.

## Funzionalita principali

- Analisi di file `.eml` con estrazione di mittente, destinatari, oggetto, body, header e raw EML pulito.
- Controlli SPF, DKIM e DMARC da header e record DNS.
- Rilevamento di mismatch tra `From`, `Reply-To` e `Return-Path`.
- Parsing della catena `Received` con visualizzazione del percorso email.
- Geolocalizzazione IP e reputazione tramite AbuseIPDB.
- Estrazione URL dal body, rilevamento link diretti a IP e controllo reputazione URL con VirusTotal.
- Analisi degli allegati con estensione, MIME type, magic bytes, hash SHA-256 e lookup VirusTotal.
- Rilevamento di domini lookalike/typosquatting rispetto a brand noti.
- Classificazione AI del contenuto con BERT fine-tuned.
- Spiegazione locale con Phi-4 mini: prima valuta oggetto e body, poi usa SPF/DKIM/DMARC, link, allegati e flag SOC come conferma o indebolimento della tesi.
- Sezioni Streamlit per analisi EML, settings, training e gestione fonti dataset pubbliche.

## Architettura

```text
FishSTOP/
|-- src/
|   |-- app.py                         # Entry point Streamlit
|   |-- config.py                      # Lettura .env / Streamlit secrets
|   |-- train.py                       # Training BERT base e modello aziendale
|   |-- parser.py                      # Parsing batch di file .eml
|   |-- public_dataset_builder.py      # Import dataset pubblici
|   |-- eml_dataset_builder.py         # Dataset custom da EML locali
|   |-- analyzer/
|   |   |-- soc_analyzer.py            # Analisi SOC completa del file EML
|   |   |-- attachment.py              # Metadati, magic bytes e hash allegati
|   |   |-- body_context.py            # Estrazione body utile per AI
|   |   |-- html_utils.py              # Pulizia HTML
|   |   |-- link_extractor.py          # Estrazione URL
|   |   |-- llm_context_analyzer.py    # Prompt e streaming Phi-4 mini/Ollama
|   |   |-- lookalike.py               # Domini simili a brand noti
|   |   `-- received_parser.py         # Parsing header Received
|   |-- validators/
|   |   |-- spf.py                     # Controllo SPF
|   |   |-- dkim.py                    # Controllo DKIM
|   |   |-- dmarc.py                   # Controllo DMARC
|   |   |-- file_reputation.py         # VirusTotal file e URL
|   |   |-- geolocation.py             # ip-api.com
|   |   `-- ip_reputation.py           # AbuseIPDB e reputazione domini/IP
|   |-- views/
|   |   |-- analyzer.py                # Dashboard principale di analisi
|   |   |-- backend.py                 # Caricamento parser, validator e modelli
|   |   |-- dataset_sources.py         # Fonti dataset pubbliche
|   |   |-- settings.py                # Stato configurazione e API key
|   |   `-- train.py                   # UI training modello aziendale
|   `-- components/
|       `-- email_globe.py             # Visualizzazione percorso email
|-- data/
|   |-- raw/                           # File EML temporanei/locali
|   `-- custom_dataset.csv             # Dataset aziendale opzionale
|-- models/
|   |-- saved_models/                  # Modello BERT base fine-tuned
|   `-- company_model/                 # Modello aziendale fine-tuned
|-- notebooks/
|-- requirements.txt
|-- run_clean.ps1
|-- Architecture.md
|-- ROADMAP.md
`-- README.md
```

## Requisiti

- Python 3.10 o superiore consigliato.
- Ambiente virtuale Python.
- Ollama installato se si vuole usare la spiegazione Phi-4 mini.
- Connessione internet per lookup DNS, AbuseIPDB, VirusTotal, ip-api.com, Kaggle e download modelli Hugging Face.

Dipendenze principali:

- `streamlit`
- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `beautifulsoup4`
- `lxml`
- `dnspython`
- `dkimpy`
- `pyspf`
- `kagglehub`
- `folium`

## Installazione

Da PowerShell, nella root del progetto:

```powershell
cd C:\Users\DEROSAEU\Documents\Progetti\FishSTOP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Su macOS/Linux:

```bash
cd /path/to/FishSTOP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configurazione

Le chiavi API sono opzionali, ma abilitano le funzioni di reputazione esterna.
Crea un file `.env` nella root del progetto:

```env
ABUSEIPDB_API_KEY=la_tua_chiave_abuseipdb
VIRUSTOTAL_API_KEY=la_tua_chiave_virustotal
```

Priorita di lettura:

1. Variabili d'ambiente e file `.env` locale.
2. `st.secrets` se l'app viene eseguita su Streamlit Cloud.
3. Fallback vuoto: la funzione viene saltata senza bloccare l'app.

## Phi-4 mini con Ollama

FishSTOP usa Phi-4 mini per generare una spiegazione locale e sintetica del caso.
Il modello viene interrogato su:

```text
http://localhost:11434/api/chat
```

Per abilitarlo:

```powershell
ollama pull phi4-mini:latest
ollama serve
```

La logica del prompt e in `src/analyzer/llm_context_analyzer.py`.
Il comportamento richiesto e:

1. prima analizzare solo oggetto e body per urgenza, denaro, coordinate bancarie, promozioni, credenziali, impersonificazioni e richieste anomale;
2. poi usare SPF, DKIM, DMARC, link, allegati, VirusTotal e flag SOC solo per rafforzare o indebolire la tesi basata sul contenuto.

Se Ollama non e attivo, il resto della dashboard continua a funzionare.

## Avvio dell'app

Metodo rapido su Windows:

```powershell
.\run_clean.ps1
```

Lo script ferma eventuali processi Streamlit precedenti del progetto, rimuove le cache Python fuori da `.venv` e avvia l'app.

Avvio manuale:

```powershell
.\.venv\Scripts\streamlit.exe run src\app.py
```

Su macOS/Linux:

```bash
streamlit run src/app.py
```

Poi apri:

```text
http://localhost:8501
```

## Uso della dashboard

1. Apri `Analyze EML`.
2. Carica un file `.eml`.
3. Leggi la sezione `Executive Triage` per severity, flag e spiegazione Phi-4 mini.
4. Usa le tab:
   - `Overview`: snapshot email e matrice segnali.
   - `Identita`: From, Reply-To, Return-Path, domini e reputazione.
   - `Auth & Routing`: SPF, DKIM, DMARC, header raw, hop Received e mappa.
   - `Link Intel`: URL estratte e reputazione VirusTotal.
   - `Allegati`: metadati, anomalie, hash e VirusTotal.
   - `AI & Body`: classificazione BERT e corpo estratto.
   - `Raw`: report strutturato e raw EML pulito.

## Training del modello

Il training principale usa BERT (`bert-base-uncased`) per classificare email legittime e phishing.

Avvio da terminale:

```powershell
python src\train.py
```

Il modello base viene salvato in:

```text
models/saved_models/
```

Il modello aziendale, addestrato sul dataset custom, viene salvato in:

```text
models/company_model/
```

L'app carica il modello aziendale se presente; altrimenti usa il modello base o il fallback Hugging Face.

## Dataset

FishSTOP supporta piu sorgenti dati:

- dataset Kaggle tramite `kagglehub`;
- email `.eml` locali;
- dataset custom in `data/custom_dataset.csv`;
- fonti pubbliche gestite dalla pagina `Public Datasets`.

La logica di costruzione dataset si trova in:

- `src/public_dataset_builder.py`
- `src/eml_dataset_builder.py`
- `src/parser.py`

## Test e controlli rapidi

Controllo sintattico dei file principali:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m py_compile src\app.py src\views\analyzer.py src\analyzer\llm_context_analyzer.py
```

Se vengono aggiunti test in `tests/`, si possono eseguire con:

```powershell
pytest
```

## Note di sicurezza

- I file `.eml` vengono analizzati localmente.
- Le chiamate esterne avvengono solo per servizi configurati o pubblici: DNS, AbuseIPDB, VirusTotal, ip-api.com, Kaggle/Hugging Face.
- Le chiavi API non devono essere committate: usa `.env` locale o Streamlit secrets.
- Phi-4 mini gira localmente tramite Ollama, quindi la spiegazione LLM non richiede invio del contenuto email a provider cloud.

## Troubleshooting

Se Streamlit non parte:

```powershell
.\run_clean.ps1
```

Se Phi-4 mini non risponde:

```powershell
ollama list
ollama pull phi4-mini:latest
ollama serve
```

Se VirusTotal o AbuseIPDB risultano mancanti, controlla `.env` o la pagina `Settings`.

Se il training fallisce per dipendenze Hugging Face:

```powershell
pip install --upgrade "accelerate>=1.1.0" "transformers[torch]"
```

Se DKIM/SPF non sono disponibili, installa o aggiorna:

```powershell
pip install --upgrade dkimpy pyspf dnspython
```

## Stato progetto

Build attuale:

```text
dev-2026-07-06-phi4-vt-lookalike
```

Il progetto e in evoluzione e contiene componenti sperimentali legati alla tesi, in particolare:

- calibrazione dei falsi positivi sui domini lookalike;
- raffinamento della severita dei risultati SPF softfail/permerror;
- miglioramento progressivo del dataset custom;
- confronto tra classificazione BERT e spiegazione Phi-4 mini.
