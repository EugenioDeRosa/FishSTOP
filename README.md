# FishStop

FishStop is a local platform for SOC triage of suspicious emails in `.eml` format.

The project combines forensic email parsing, authentication checks, link and attachment reputation, BERT-based content analysis, and local Phi-4 mini explanations through Ollama or GitHub Models.

The main workflow is designed for a thesis project: upload an `.eml` file, let the system extract headers, body, links, and attachments, then review technical indicators and a readable verdict for the analyst.

## Features

- `.eml` analysis with extraction of sender, recipients, subject, body, headers, and cleaned raw EML.
- SPF, DKIM, and DMARC evidence extraction and display.
- `Received` chain parsing with email-route visualization.
- Sender domain and IP reputation checks through AbuseIPDB.
- URL extraction, lookalike-domain detection, redirect checks, and VirusTotal URL reputation.
- Attachment analysis with extension, MIME type, magic bytes, SHA-256 hash, and VirusTotal lookup.
- BERT content classification for legitimate vs phishing emails.
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

## Training

The main training flow uses BERT (`bert-base-uncased`) to classify legitimate and phishing emails.

The app can export a balanced CSV for Google Colab. After training, publish the resulting model to Hugging Face and configure the app to use that model.

Training data can come from Kaggle datasets through `kagglehub`, local `.eml` files, `data/custom_dataset.csv`, and public sources managed from the `Public Datasets` page.

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

If the BERT model does not load, verify the Hugging Face model name, network access, and installed `transformers` dependencies.

If public datasets cannot be downloaded, verify Kaggle credentials and the availability of `kagglehub`.
