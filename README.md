# FishStop

FishStop is a local platform for SOC triage of suspicious emails in `.eml` format.

The project combines forensic email parsing, authentication checks, link and attachment reputation, calibrated DistilBERT content analysis, and local Phi-4 mini explanations through Ollama or GitHub Models.

The main workflow is designed for a thesis project: upload an `.eml` file, let the system extract headers, body, links, and attachments, then review technical indicators and a readable verdict for the analyst.

## Features

- `.eml` analysis with extraction of sender, recipients, subject, body, headers, and cleaned raw EML.
- SPF, DKIM, and DMARC evidence extraction and display.
- `Received` chain parsing with email-route visualization.
- Sender domain and IP reputation checks through AbuseIPDB.
- URL extraction, lookalike-domain detection, redirect checks, and VirusTotal URL reputation.
- Attachment analysis with extension, MIME type, magic bytes, SHA-256 hash, and VirusTotal lookup.
- Calibrated DistilBERT content classification for legitimate vs phishing emails, including long-email chunking.
- Local/hosted Phi-4 mini intent extraction with Unicode deobfuscation, exact
  action evidence, secondary lure/threat signals, and claimed-brand/domain checks.
- Streamlit pages for EML analysis, settings, and public dataset management.
- Custom EML dataset builder with deduplication.
- Public dataset builder with a balanced training CSV export.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Configuration

Create a `.env` file in the project root:

```env
ABUSEIPDB_API_KEY=your_abuseipdb_key
VIRUSTOTAL_API_KEY=your_virustotal_key
GITHUB_MODELS_TOKEN=your_github_models_token
```

Secret loading priority:

1. Environment variables and the local `.env` file.
2. Streamlit Cloud secrets.
3. Empty value, with the related lookup skipped.

## How to Use

1. Start the app.
2. Open `Analyze EML`.
3. Upload an `.eml` file.
4. Review the tabs: `Summary`, `Sender`, `Authentication`, `Links`, `Files`, `Content`, `Indicators`, and `Technical`.

## DistilBERT training

Generate `data/processed/fishstop_train_complete.csv` from the Public Datasets page, then run:

```powershell
python -m src.train --dataset data/processed/fishstop_train_complete.csv --output-dir models/fishstop-distilbert-multilingual
```

The CSV contains immutable `train`, `validation`, and `test` splits. The training command:

1. fine-tunes `distilbert/distilbert-base-multilingual-cased` with explicit `LEGITIMATE=0` and `MALICIOUS=1` labels; the malicious class includes phishing, scam and spam;
2. weights chunks so every email contributes equally and selects the best checkpoint by email-level validation F1;
3. calibrates its probabilities on validation with temperature scaling;
4. derives the decision threshold and uncertainty band from validation;
5. evaluates once on the held-out test set;
6. writes the model, tokenizer, `calibration.json`, `training_meta.json`, and model card to the output directory.

Long emails use up to eight evenly spaced overlapping 512-token windows in training, calibration, test and runtime. FishSTOP chooses the window with the strongest malicious-content logit margin. Public dataset generation retains all valid deduplicated rows, including the historical Enron and SpamAssassin corpora, without trusting the potentially absent or forged `Date` header of individual messages. Sources are mixed with a reproducible 70/10/20 random split stratified by source and label; near-duplicate campaigns are kept in one split and contradictory campaigns are removed. Class-weighted training compensates for the remaining class imbalance without discarding real email. Synthetic email remains train-only and is automatically sampled to at most 10% of the training split. Every downloaded release is pinned by version and checksum in the reproducibility manifest. Training emits a dataset audit in `training_meta.json` and uses early stopping. The deployed app loads `eugenioderodev/fishstop-bert`, pinned by default to commit `b29e3334457d942bb5c05fe8f6639edeccf59692`; `FISHSTOP_HF_MODEL_REVISION` can override it explicitly. Upload the model, tokenizer, `calibration.json`, and metadata together. Copying the model only to Google Drive does not update the deployed app.

The old `notebooks/Phishing_detection.ipynb` is retained only as historical exploratory work; `src/train.py` is the canonical training pipeline.

## Validation

```powershell
python -m py_compile src/views/analyzer.py
python -m py_compile src/views/settings.py
python -m py_compile src/analyzer/soc_analyzer.py
python -m py_compile src/analyzer/html_utils.py
python -m py_compile src/analyzer/lookalike.py
pytest
```

With Ollama running, the repeatable Phi-4 intent benchmark can be run in full
or on selected cases:

```powershell
.\.venv\Scripts\python.exe scripts\eval_phi4_intent.py
.\.venv\Scripts\python.exe scripts\eval_phi4_intent.py --case credential_form --case late_credential_request
```

## Privacy Notes

- `.eml` files are analyzed locally.
- Reputation lookups send only IPs, domains, URLs, or hashes to the configured providers.
- FishStop does not open URLs extracted from emails by default. Direct redirect inspection is opt-in through `FISHSTOP_ENABLE_URL_DESTINATION_CHECK=1` and rejects non-public destinations.
- Phi-4 mini can run locally through Ollama with the explicit `phi4-mini:3.8b-q4_K_M` quantized tag and a bounded 4096-token context.
- The hosted website uses GitHub Models automatically. Email content is anonymized before it is sent; local deployments can continue to use Ollama.

## Troubleshooting

If VirusTotal or AbuseIPDB results are missing, check the `.env` file or the `Settings` page.

If the DistilBERT model does not load, verify the Hugging Face model name, network access, and installed `transformers` dependencies.

If public datasets cannot be downloaded, verify Kaggle credentials and the availability of `kagglehub`.
