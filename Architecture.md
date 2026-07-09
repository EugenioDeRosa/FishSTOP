# FishStop Architecture

This repository contains the core components for forensic analysis and automated phishing detection in `.eml` files. The system combines traditional security checks (DNS and reputation lookups) with an AI-based predictive model (BERT).

## Module Structure and Responsibilities

### `src/analyzer/soc_analyzer.py`

**File responsibility:** Dynamic extraction and SOC-style forensic analysis of the structured content of an `.eml` file.

- `analyze(eml_path)`: Coordinates extraction of the envelope, headers, text, links, and attachments.
- `_extract_spf_sender_ip(msg, hops)`: Identifies the original public sender IP by analyzing the server chain.
- `_check_dkim_signature_present(msg)`: Checks whether the DKIM signature header is physically present.
- `_build_flags(report)`: Generates quick warning indicators based on the extracted data.
- **Interactions:** Uses the native `email` and `re` libraries and works with global text-cleaning routines and lookalike-domain checks (`KNOWN_BRANDS`).

### `src/validators/*`

**File responsibility:** Technical validation of authentication protocols and queries to external reputation services.

- `check_spf(sender_ip, mail_from, ...)`: Checks whether the sender IP is authorized by the domain's SPF DNS record.
- `check_dmarc(domain)`: Retrieves and parses the DMARC policy.
- `check_ip_reputation(ip)`: Queries AbuseIPDB and VirusTotal for IP reputation.
- `check_domain_reputation(domain)`: Resolves the domain and checks the resulting IP through `check_ip_reputation`.
- `check_file_hash(sha256)`: Queries VirusTotal for attachment hashes.

### `src/parser.py`

**File responsibility:** Low-level parsing and sanitization of raw `.eml` files on disk.

- `parse_single_eml(eml_path)`: Parses a single `.eml` file and extracts sender, recipient, subject, date, and body.
- `load_batch_emls(folder_path)`: Scans a local folder and combines all parsed emails into a DataFrame.
- **Interactions:** Uses `_sanitize_eml_bytes` to repair non-standard `.eml` files before parsing and returns data structures compatible with `pandas`.

### `src/eml_dataset_builder.py`

**File responsibility:** Building, deduplicating, and managing a custom CSV dataset from local `.eml` files.

- `_load_hashes()`: Loads hashes of already indexed texts to avoid duplicate inserts.
- `add_eml(eml_bytes, ...)`: Preprocesses a single email and appends it to the dataset if it is not duplicated.
- `add_batch(items, ...)`: Handles multithreaded processing of email batches.
- `remove_by_hash(text_hash)`: Removes a specific record from the CSV file using its identity hash.
- **Interactions:** Writes to `data/custom_dataset.csv` and uses global text-normalization routines.

### `src/train.py`

**File responsibility:** Orchestration of the training cycle, data balancing, and evaluation of the deep-learning model (BERT).

- `download_and_combine_data(...)`: Merges and balances Kaggle data, personal emails, and the custom dataset.
- `prepare_datasets(df)`: Concatenates text fields, tokenizes for BERT, and splits into Train/Validation/Test sets.
- `train_model(...)`: Fine-tunes the classifier and stores artifacts under `models/`.

### Supporting Modules

- `src/config.py`: Centralizes secure API-key loading through `get_secret()`, prioritizing the local `.env` file or Streamlit Cloud `st.secrets`.
- `src/app.py`: Streamlit graphical interface. It instantiates `EmlSOCAnalyzer`, validators, and the BERT model to provide an analyst dashboard for uploading emails and reviewing security verdicts.
- `src/views/*`: Streamlit pages for EML analysis, settings, Colab training, and public dataset management.
- `src/analyzer/link_extractor.py`: Extracts URLs and detects visible-text versus destination mismatches.
- `src/analyzer/lookalike.py`: Detects typosquatting, homoglyphs, Punycode, and other lookalike-domain patterns.
- `src/components/email_globe.py`: Visualizes the email route on a globe/map component.
