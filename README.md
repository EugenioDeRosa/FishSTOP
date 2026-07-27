# FishSTOP

FishSTOP is a Streamlit application for SOC-oriented triage of suspicious
`.eml` files. It combines email parsing, sender and routing evidence,
reputation services, a calibrated DistilBERT classifier, and Phi-4 mini intent
analysis in a single interface.

The application is intended to support an analyst. A LOW or “likely
legitimate” result is not proof that an email is safe, and no automated verdict
should replace manual validation when the message requests credentials,
payments, confidential information, or execution of a file.

Current development version: `dev-v.1.1.0`.

## Main capabilities

- Parse `.eml` files and extract sender, recipients, subject, routing headers,
  plain text, HTML, links, and attachments.
- Display SPF, DKIM, and DMARC results reported by the message headers.
- Reconstruct and visualize the `Received` route.
- Detect reply-to, return-path, display-name, and lookalike-domain anomalies.
- Query VirusTotal for URLs and attachment hashes.
- Query AbuseIPDB for public IP and domain reputation.
- Inspect attachment names, MIME types, magic bytes, hashes, and basic PDF
  behavior.
- Classify message content with a calibrated multilingual DistilBERT model.
- Analyze the requested action and phishing intent with Phi-4 mini through
  local Ollama or GitHub Models.
- Show independent results as soon as background tasks complete.
- Build reproducible training datasets and train the FishSTOP classifier.

## Analysis flow

1. The uploaded file is checked against application safety limits.
2. FishSTOP parses the MIME structure and extracts one canonical visible body,
   links, attachments, routing evidence, and authentication-result headers.
3. Static SOC findings are displayed immediately.
4. Reputation, DistilBERT, and Phi-4 tasks run in bounded background pools.
5. The UI refreshes individual sections as their results become available.
6. When Phi-4 finishes, its structured semantic result is combined with the
   deterministic risk policy to produce the final verdict.

Background resources are process-global, but jobs are namespaced by an opaque
Streamlit session identifier. Cached model inference is serialized to keep the
shared tokenizer and model safe across concurrent users.

## Project structure

```text
FishSTOP/
|-- src/
|   |-- analyzer/          # MIME, body, link, attachment and policy analysis
|   |-- validators/        # VirusTotal, AbuseIPDB and geolocation clients
|   |-- views/             # Streamlit pages
|   |-- background_jobs.py # bounded asynchronous job manager
|   |-- bert_inference.py  # email-level chunked inference
|   |-- public_dataset_builder.py
|   `-- train.py
|-- scripts/               # repeatable evaluation and maintenance utilities
|-- tests/                 # unit and regression tests
|-- data/
|   |-- raw/               # selected tracked `.eml` samples
|   `-- processed/         # reproducible datasets, stored with Git LFS
|-- reports/               # evaluation outputs
|-- notebooks/             # historical exploratory notebook
|-- streamlit_app.py       # guarded application entry point
`-- Dockerfile
```

## Requirements

- Python 3.11 is recommended.
- Git LFS is required to download the tracked processed datasets.
- Internet access is required to download the DistilBERT model and use hosted
  reputation or LLM services.
- Ollama is optional and is used only for local Phi-4 analysis.

## Clone and local installation

Install Git LFS before cloning:

```powershell
git lfs install
git clone https://github.com/EugenioDeRosa/FishSTOP.git
cd FishSTOP
```

Create the virtual environment and install the complete development
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the application:

```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Configuration

Create `.env` in the project root for local development:

```env
APP_MODE=development

VIRUSTOTAL_API_KEY=
ABUSEIPDB_API_KEY=
GITHUB_MODELS_TOKEN=
HF_TOKEN=

FISHSTOP_LLM_PROVIDER=auto
FISHSTOP_ENABLE_URL_DESTINATION_CHECK=0
```

Do not commit `.env` or `.streamlit/secrets.toml`.

### Credential resolution

For normal service requests, FishSTOP resolves credentials in this order:

1. values entered in the current client's **Connections** page;
2. environment variables or the local `.env`;
3. Streamlit secrets;
4. an empty value, which causes the related service to be skipped.

In production, the shared DistilBERT resource uses only the server-side
`HF_TOKEN`. A token entered by a client is not used to create a global cached
model.

### Runtime variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_MODE` | `development`, `prod`, `production`, or `public` | `development` |
| `VIRUSTOTAL_API_KEY` | URL and file-hash reputation | empty |
| `ABUSEIPDB_API_KEY` | Domain and IP reputation | empty |
| `GITHUB_MODELS_TOKEN` | Hosted Phi-4 mini analysis | empty |
| `HF_TOKEN` | Optional Hugging Face model access | empty |
| `FISHSTOP_LLM_PROVIDER` | `auto`, `ollama`, or `github` | `auto` |
| `FISHSTOP_HF_MODEL_REVISION` | Override the pinned model revision | pinned commit |
| `FISHSTOP_ENABLE_URL_DESTINATION_CHECK` | Opt-in destination redirect request | `0` |
| `GITHUB_MODELS_ENDPOINT` | GitHub Models-compatible endpoint | Azure inference endpoint |
| `GITHUB_MODELS_MODEL` | Hosted model identifier | `Phi-4-mini-instruct` |
| `OLLAMA_CHAT_ENDPOINT` | Local Ollama chat endpoint | `http://localhost:11434/api/chat` |
| `OLLAMA_MODEL` | Local model tag | `phi4-mini:3.8b-q4_K_M` |
| `OLLAMA_NUM_CTX` | Ollama context window | `4096` |

## Local Phi-4 through Ollama

Install Ollama and download the exact configured model:

```powershell
ollama pull phi4-mini:3.8b-q4_K_M
```

Then use:

```env
FISHSTOP_LLM_PROVIDER=ollama
```

With `auto`, FishSTOP prefers a reachable local Ollama instance and otherwise
uses GitHub Models when a token is configured. The production Docker image sets
the provider to `github`.

## DistilBERT

The application loads:

```text
eugenioderodev/fishstop-bert
```

The default revision is pinned in `src/views/backend.py` to:

```text
b29e3334457d942bb5c05fe8f6639edeccf59692
```

Runtime inference:

- normalizes and pseudonymizes selected content;
- tokenizes the full selected body once;
- selects at most eight evenly distributed, overlapping 512-token windows;
- evaluates only those bounded windows;
- chooses the window with the strongest malicious-content logit margin;
- applies temperature scaling and the saved uncertainty band.

The tokenizer implementation supports both older Transformers interfaces and
the newer `BertTokenizer` backend that does not expose
`prepare_for_model()`.

## Safety limits

Untrusted emails are bounded before expensive analysis:

| Resource | Limit |
| --- | ---: |
| Uploaded `.eml` | 10 MB |
| MIME parts | 200 |
| MIME nesting | 10 levels |
| Decoded text | 240,000 characters |
| AI body | 120,000 characters |
| Attachments | 25 |
| Unique links | 100 |
| `Received` hops | 50 |
| Phi-4 sections | 12 |
| DistilBERT windows | 8 |

When the AI body is too large, FishSTOP keeps static analysis available and
states explicitly that AI analysis was not performed. Input is never silently
reported as completely analyzed after exceeding a supported limit.

The background manager also limits running and queued work for each provider.
When capacity is exhausted, the UI reports temporary unavailability instead of
building an unbounded queue.

## Production deployment

Build and run the container:

```powershell
docker build -t fishstop .
docker run --rm -p 8501:8501 `
  -e VIRUSTOTAL_API_KEY `
  -e ABUSEIPDB_API_KEY `
  -e GITHUB_MODELS_TOKEN `
  fishstop
```

The Docker image sets:

```env
APP_MODE=production
FISHSTOP_LLM_PROVIDER=github
```

Production mode currently hides the training-dataset page. Unexpected
exceptions are logged server-side with a reference such as `FS-12AB34CD`; the
public UI receives only that reference. Development mode retains expandable
diagnostics.

Before exposing the app publicly, review the items under
[Security and privacy limitations](#security-and-privacy-limitations).

## Dataset generation

The canonical dataset builder is `src/public_dataset_builder.py`. The
development-only **Training datasets** page downloads and prepares configured
public sources under `data/training_sources/`.

Generated processed datasets include immutable `train`, `validation`, and
`test` splits. The builder:

- deduplicates exact and near-duplicate campaigns;
- keeps related campaigns in the same split;
- removes contradictory labels;
- uses reproducible source-stratified 70/10/20 splits;
- keeps synthetic data train-only and limits it to 10% of the training split;
- records versions and checksums in the manifest.

The processed CSV files are tracked with Git LFS. Downloaded source archives
under `data/training_sources/` are intentionally ignored.

## Model training

Generate or update `data/processed/fishstop_train_complete.csv`, then run:

```powershell
python -m src.train `
  --dataset data/processed/fishstop_train_complete.csv `
  --output-dir models/fishstop-distilbert-multilingual
```

The training pipeline:

1. fine-tunes `distilbert/distilbert-base-multilingual-cased`;
2. uses `LEGITIMATE=0` and `MALICIOUS=1`;
3. weights chunks so each email contributes equally;
4. selects the best checkpoint by email-level validation F1;
5. calibrates probabilities on the validation split;
6. derives the decision threshold and uncertainty band;
7. evaluates once on the held-out test split;
8. writes the model, tokenizer, calibration, metadata, model card, and EDA
   artifacts.

`notebooks/Phishing_detection.ipynb` is retained for historical exploration.
`src/train.py` is the canonical training implementation.

## Tests and validation

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Compile the main runtime modules:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  streamlit_app.py `
  src/app.py `
  src/views/analyzer.py `
  src/analyzer/soc_analyzer.py
```

Run the repeatable Phi-4 intent benchmark with Ollama available:

```powershell
.\.venv\Scripts\python.exe scripts\eval_phi4_intent.py
.\.venv\Scripts\python.exe scripts\eval_phi4_intent.py `
  --case credential_form `
  --case late_credential_request
```

## Data sent to external services

Behavior depends on deployment and configured providers:

| Service | Data sent |
| --- | --- |
| VirusTotal | extracted URLs and SHA-256 attachment hashes |
| AbuseIPDB | public IP addresses and sender-related domains |
| ipwho.is | public routing IP addresses for HTTPS geolocation |
| GitHub Models | subject, selected visible content, and limited structural metadata after partial pseudonymization |
| Hugging Face | model files are downloaded; email content is not sent for inference |
| Ollama | selected email content remains on the local Ollama host |

Uploaded emails are parsed by the machine running Streamlit. In a local
deployment, that is the local computer. In a hosted deployment, the `.eml` is
uploaded to and parsed by the application server.

The hosted-text transformations replace common email addresses, URLs, IP
addresses, telephone numbers, IBANs, and some account-like values. This is
**partial pseudonymization**, not guaranteed anonymization. Names, addresses,
organizations, order numbers, invoice references, BICs, confidential prose,
and conversation history may remain.

## Security and privacy limitations

The following items must be considered before public or organizational use:

- SPF, DKIM, and DMARC statuses are extracted from potentially untrusted
  message headers. FishSTOP does not currently perform independent
  cryptographic DKIM verification or live SPF/DMARC evaluation.
- Some inline `message/rfc822` content is excluded from the canonical AI body.
- Normal reply parsing can remove quoted history; forwarded-message context is
  generally preserved.
- IP geolocation uses the external `ipwho.is` HTTPS API. The free endpoint does
  not provide proxy, VPN, Tor, or hosting detection; those fields remain
  unavailable unless a provider response explicitly includes security data.
- The **Connections** page currently exposes partial masks and sources for
  configured fallback credentials. It should be hidden in production before a
  public launch.
- Hosted Phi-4 analysis starts automatically when the production provider is
  configured. A public deployment needs an accurate privacy notice, a lawful
  basis, a retention policy, and review of provider terms.
- Session separation prevents one client from reading another client's job
  result, but cached models and bounded worker pools remain global by design.

FishSTOP does not provide a legal compliance guarantee. Consult the appropriate
privacy or security owner before processing personal, confidential, or
regulated messages.

## Moving to another computer

Everything tracked and committed can be restored with `git clone`, including
source code, tests, selected `.eml` samples, reports, the thesis draft, and the
processed Git LFS datasets.

The following paths are intentionally local and must be copied separately if
they are still needed:

| Path | Notes |
| --- | --- |
| `.env` | credentials and local configuration; transfer securely or recreate |
| `data/custom_dataset.csv` | custom dataset generated by the legacy builder |
| `data/training_sources/` | downloaded public-source archives |
| `models/` | local checkpoints, training metadata, and EDA figures |
| external email-test folders | not part of the repository |
| hosting-platform secrets | configured outside GitHub |

Do not copy `.venv`, Python caches, pytest caches, or Hugging Face download
caches. Recreate the environment from `requirements.txt`.

After cloning on the new computer:

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify access separately for GitHub, Hugging Face, GitHub Models, VirusTotal,
AbuseIPDB, and the hosting platform. Never commit API keys to recover them.

## Troubleshooting

### DistilBERT does not load

- Verify network access to `eugenioderodev/fishstop-bert`.
- Check `HF_TOKEN` only if the repository requires authenticated access.
- Confirm that model and calibration revisions match.
- Remove only the affected Hugging Face cache entry if a download was
  interrupted.

### Phi-4 is unavailable

- For Ollama, run `ollama list` and verify
  `phi4-mini:3.8b-q4_K_M`.
- For GitHub Models, verify `GITHUB_MODELS_TOKEN`,
  `GITHUB_MODELS_MODEL`, and provider quotas.
- Check the server log using the error reference displayed by the UI.

### Reputation results are missing

- Verify the related API key and service quota.
- A result can remain unavailable when a bounded provider queue is full.
- URL destination requests are disabled unless explicitly enabled.

### Processed CSV files are missing after clone

Install Git LFS and run:

```powershell
git lfs install
git lfs pull
```

### Public datasets cannot be rebuilt

Verify network access, Kaggle credentials where required, and the pinned source
versions recorded by the builder.
