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
- Local/hosted Phi-4 mini explanation focused on scam and phishing patterns.
- Streamlit pages for EML analysis, settings, Colab training, and public dataset management.
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
4. Review the tabs: `Overview`, `Identity`, `Auth & Routing`, `Link Intel`, `Attachments`, `AI & Body`, and `Raw`.

## DistilBERT training

Generate `data/processed/fishstop_train_complete.csv` from the Public Datasets page, then run:

```powershell
python -m src.train --dataset data/processed/fishstop_train_complete.csv --output-dir models/fishstop-distilbert
```

The CSV contains immutable `train`, `validation`, and `test` splits. The training command:

1. fine-tunes `distilbert-base-uncased` with explicit `LEGITIMATE=0` and `MALICIOUS=1` labels; the malicious class includes phishing, scam and spam;
2. selects the best checkpoint by validation F1;
3. calibrates its probabilities on validation with temperature scaling;
4. derives the decision threshold and uncertainty band from validation;
5. evaluates once on the held-out test set;
6. writes the model, tokenizer, `calibration.json`, `training_meta.json`, and model card to the output directory.

Long emails use up to eight evenly spaced overlapping 512-token windows in training, calibration, test and runtime. FishSTOP chooses the window with the strongest malicious-content logit margin. Public dataset generation uses only modern 2022-2025 sources, keeps semantic campaign variants in one split, writes a reproducibility manifest and reserves synthetic email for training. The deployed app always loads `eugenioderodev/fishstop-bert`, so review the generated metrics before uploading the complete output directory to that repository.

The old `notebooks/Phishing_detection.ipynb` is retained only as historical exploratory work; `src/train.py` is the canonical training pipeline.

## Validation

```powershell
python -m py_compile src/views/train.py
python -m py_compile src/views/analyzer.py
python -m py_compile src/views/settings.py
python -m py_compile src/analyzer/soc_analyzer.py
python -m py_compile src/analyzer/html_utils.py
python -m py_compile src/analyzer/lookalike.py
pytest
```

## Privacy Notes

- `.eml` files are analyzed locally.
- Reputation lookups send only IPs, domains, URLs, or hashes to the configured providers.
- Phi-4 mini can run locally through Ollama, so LLM explanations do not require sending email content to a cloud provider.

## Troubleshooting

If VirusTotal or AbuseIPDB results are missing, check the `.env` file or the `Settings` page.

If the DistilBERT model does not load, verify the Hugging Face model name, network access, and installed `transformers` dependencies.

If public datasets cannot be downloaded, verify Kaggle credentials and the availability of `kagglehub`.
