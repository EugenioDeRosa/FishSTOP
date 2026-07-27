# FishSTOP TODO

Items are ordered by recommended implementation priority.

## Before a public deployment

- [ ] Hide the **Connections** page in production, or protect it with
  authentication. Do not expose partial masks or sources of server credentials
  to public visitors.
- [ ] Add authentication and per-user or per-IP request limits before allowing
  public use of shared VirusTotal, AbuseIPDB, GitHub Models, or Hugging Face
  credentials.
- [ ] Minimize the Docker build context and image. Replace `COPY . .` with
  explicit runtime copies and exclude processed datasets, reports, notebooks,
  the thesis draft, tests, and development scripts.
- [ ] Review tracked `.eml` samples, reports, and the thesis draft for personal
  or confidential data. Replace real messages with anonymized or synthetic
  fixtures before making the repository public.
- [ ] Add a clear privacy notice covering uploaded email processing, external
  services, retention, and partial pseudonymization.

## Reliability and reproducibility

- [ ] Add CI to run the full test suite and Python compilation on every push and
  pull request.
- [ ] Add a development dependency file or optional dependency group containing
  `pytest` and code-quality tools.
- [ ] Pin production dependencies through exact versions or a constraints/lock
  file, and define a controlled dependency-update process.
- [ ] Add caching and rate-limit handling for IP geolocation so repeated routing
  IPs do not consume the free provider quota unnecessarily.
- [ ] Decide whether production needs proxy/VPN/hosting detection. The free
  `ipwho.is` endpoint provides geolocation over HTTPS but not security signals.
  If required, configure a suitable paid provider without silently treating
  unavailable signals as negative results.

## Repository maintenance

- [ ] Audit Git history for large dataset/model blobs that predate Git LFS, then
  plan a coordinated history cleanup if repository size remains a problem.
- [ ] Run appropriate Git/LFS maintenance after the history decision; document
  the clone and data-download workflow.
- [ ] Correct the generated DistilBERT model card: runtime preprocessing uses the
  normalized email body and intentionally excludes the subject.
- [ ] Split the largest modules into focused components, starting with
  `llm_context_analyzer.py`, `views/analyzer.py`, and
  `public_dataset_builder.py`.
- [ ] Remove or clearly label legacy entry points and exploratory files that are
  not part of the supported runtime.

## Recently completed

- [x] Expose Phi-4 section progress from the background worker and display the
  current section in the automatically refreshed UI.
- [x] Make Ollama and GitHub Models streaming incremental, avoiding repeated
  reconstruction of the complete response for every generated token.
- [x] Cache Ollama availability checks, resolve the Phi-4 backend once per UI
  render, and support the canonical `FISHSTOP_LLM_PROVIDER` configuration.
- [x] Cancel queued analysis jobs when an email is cleared and cooperatively
  stop an active Phi-4 analysis, rotating the analysis session before reuse.

## Longer-term security accuracy

- [ ] Evaluate independent SPF, DKIM, and DMARC verification instead of relying
  only on potentially untrusted message headers.
- [ ] Define retention and deletion rules for uploaded messages and derived
  analysis data in hosted deployments.
- [ ] Review external-provider terms, quotas, and data-processing requirements
  before organizational or commercial use.
