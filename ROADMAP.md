# FishStop Roadmap

This file tracks planned functionality and improvements for the existing modules.

## Already Implemented

- EML parsing with extraction of headers, body, links, and attachments.
- SPF, DKIM, and DMARC evidence extraction from headers.
- Attachment analysis: magic bytes, content type, extension, and consistency checks.
- AI classification with fine-tuned BERT (legitimate email vs phishing).
- URL extraction from plain text and HTML.
- VirusTotal checks for URL and attachment reputation.
- AbuseIPDB checks for IP and domain reputation.
- Lookalike-domain detection for known brands.
- Streamlit dashboard with analysis, settings, training, and public dataset pages.
- Custom EML dataset builder with deduplication.
- Public dataset builder with balanced 50/50 export.
- Local Phi-4 mini explanation through Ollama/GitHub Models integration.

## Planned Improvements

### 1. URL and HTML Link Analysis

Extract all `<a href="...">` tags from the HTML body and compare the visible clickable text with the real destination URL. A mismatch, for example visible text `paypal.com` pointing to `evil.ru`, is a strong phishing signal.

Compare domains extracted from links against the sender domain (`From:`). Implement exact-domain mismatch checks, suspicious subdomain checks, Punycode and Unicode homoglyph handling, and URL shortener or redirector heuristics.

Modern phishing emails may avoid textual links and use QR codes in images to hide malicious URLs. Add QR-code extraction from images and attachments where feasible.

### 2. Attachment Reputation

For every attachment extracted from the `.eml`, calculate the SHA-256 hash of the decoded raw bytes and expose it in the SOC report.

Query VirusTotal API v3 (`/files/{hash}`) for each SHA-256 hash. In the attachment panel, show the number of engines that flag it and a permalink to the VirusTotal report.

### 3. Identity and Header Consistency

Improve correlation between identity headers: compare `From:` with `Return-Path:`, compare `From:` with the closest-to-sender `Received:` hop, and handle missing `Reply-To` clearly.

When `Reply-To` is absent, show `Absent` instead of `Consistent`; green implies a positive check that was not actually performed.

### 4. Received Chain and IP Reputation

`Received` headers should be read from bottom to top: the last one in the list is closest to the real sender. Verify that the UI presents labels correctly and that the injection server is the one used for SPF and threat-intelligence lookups.

Add country and ASN for each IP in the chain using a geolocation API such as `ip-api.com` free tier or local MaxMind GeoLite2. This helps detect anomalous routing.

### 5. Dataset and Training

- Add an easier workflow to integrate additional emails into the training dataset.
- Improve the custom dataset progressively with analyst-labelled examples.
- Add better reporting for class balance, duplicates, and template-like near duplicates.
- Keep the training export compatible with Colab and Hugging Face publication.

### 6. Miscellaneous Fixes

- Review DMARC policy rendering when the result is pass or fail.
- Add more SOC Analyst Lab style checks for phishing-email analysis.
- Run nslookup on extracted links and check whether the resolved IP is malicious on VirusTotal.
- Improve error messages when API keys are missing.
