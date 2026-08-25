# FishStop Architecture

This repository contains the core components for forensic analysis and automated phishing detection in `.eml` files. The system combines traditional security checks (DNS and reputation lookups) with an AI-based predictive model (BERT).

## Module Structure and Responsibilities

### `src/analyzer/soc_analyzer.py`

**File responsibility:** Dynamic extraction and SOC-style forensic analysis of the structured content of an `.eml` file.

- `analyze(eml_path)`: Coordinates extraction of the envelope, headers, text, links, and attachments.
- `_extract_spf_sender_ip(msg, hops)`: Identifies the original public sender IP by analyzing the server chain.
- `_build_flags(report)`: Generates quick warning indicators based on the extracted data.
- **Interactions:** Uses the native `email` and `re` libraries and works with global text-cleaning routines and lookalike-domain checks (`KNOWN_BRANDS`).

### `src/validators/*`

**File responsibility:** Queries to external reputation and geolocation services. SPF, DKIM, and DMARC evidence is parsed from the authentication headers by `received_parser.py`.

- `check_ip_reputation(ip)`: Queries AbuseIPDB for IP reputation.
- `check_domain_reputation(domain)`: Resolves the domain and checks the resulting IP through `check_ip_reputation`.
- `check_file_hash(sha256)`: Queries VirusTotal for attachment hashes.
- `check_url(url)`: Queries VirusTotal and follows only validated public HTTP(S) destinations.
- `geolocate_ip(ip)`: Geolocates public IP addresses.

### `src/parser.py` (standalone utility)

**File responsibility:** Low-level parsing and sanitization for offline/batch utilities. The interactive application uses `EmlSOCAnalyzer` directly.

- `parse_single_eml(eml_path)`: Parses a single `.eml` file and extracts sender, recipient, subject, date, and body.
- `load_batch_emls(folder_path)`: Scans a local folder and combines all parsed emails into a DataFrame.
- **Interactions:** Uses `_sanitize_eml_bytes` to repair non-standard `.eml` files before parsing and returns data structures compatible with `pandas`.

### `src/eml_dataset_builder.py` (standalone utility)

**File responsibility:** Building, deduplicating, and managing a custom CSV dataset from local `.eml` files outside the current Streamlit workflow.

- `_load_hashes()`: Loads hashes of already indexed texts to avoid duplicate inserts.
- `add_eml(eml_bytes, ...)`: Preprocesses a single email and appends it to the dataset if it is not duplicated.
- `add_batch(items, ...)`: Handles multithreaded processing of email batches.
- `remove_by_hash(text_hash)`: Removes a specific record from the CSV file using its identity hash.
- **Interactions:** Writes to `data/custom_dataset.csv` and uses global text-normalization routines.

### `src/train.py`

**File responsibility:** Orchestration of the training cycle, data balancing, and evaluation of the deep-learning model (BERT).

- `load_training_dataframe(...)`: Loads and audits the immutable dataset splits.
- `DistilBERTPhishingTrainer.prepare_datasets(...)`: Tokenizes the dataset into weighted email chunks.
- `run_training(...)`: Fine-tunes, calibrates, evaluates, and stores the classifier artifacts.

### Supporting Modules

- `src/config.py`: Centralizes secure API-key loading through `get_secret()`, prioritizing the local `.env` file or Streamlit Cloud `st.secrets`.
- `src/app.py`: Streamlit graphical interface. It instantiates `EmlSOCAnalyzer`, validators, and the BERT model to provide an analyst dashboard for uploading emails and reviewing security verdicts.
- `src/views/*`: Streamlit pages for EML analysis, settings, and public dataset management.
- `src/analyzer/link_extractor.py`: Extracts URLs and detects visible-text versus destination mismatches.
- `src/analyzer/lookalike.py`: Detects typosquatting, homoglyphs, Punycode, and other lookalike-domain patterns.
- `src/components/email_globe.py`: Visualizes the email route on a globe/map component.
