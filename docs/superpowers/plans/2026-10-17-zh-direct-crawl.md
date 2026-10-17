# v2 120ask/iiyi Direct Crawl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Crawl only missing-live-doc v2 120ask/iiyi URLs under the approved D20 policy and produce resumable zh segment batches for S15.

**Architecture:** A strict v2 target builder subtracts active segment IDs from the corpus allowlist. A single-client sequential crawler writes atomic raw shards and a durable request ledger; every 10,000 fetched HTML page groups are cut, extracted, chunked, embedded, indexed, and recorded with a provenance sidecar.

**Tech Stack:** Python 3.12, httpx, SQLite, Polars/Parquet, existing `vicrawl` raw shards and `zh_full_ingest` pipeline, S15 `zh_v2` segment contract.

**Spec:** `docs/superpowers/specs/2026-10-17-zh-direct-crawl.md`

## Global Constraints

- Only URL/IDs from `data/corpus_v2_ids.parquet`; only missing active zh-v2 docs.
- Exact UA from the spec; one IP, no proxy, no redirects; global spacing ≥1 second plus positive jitter.
- Exponential backoff on 403/429/CAPTCHA/bot-blocking; cap five minutes unless server `Retry-After` is longer.
- Evaluate the >20% error stop after 500 requests, then on the rolling last-500 window.
- Never solve CAPTCHA, mutate the existing crawler policy, or write to OLD.
- Emit completed segments in 10,000-document batches and preserve source/timestamp provenance without changing the Parquet schema.

## Review Focus

- A URL or ID outside v2 reaches the HTTP client — tested by target-builder scope checks.
- Active v2 doc IDs are fetched or ingested twice — tested with active manifest fixtures.
- Redirects/proxies or an altered UA are used — tested on client configuration and a mocked redirect response.
- Backoff and the first-500 threshold diverge from the approved policy — tested at boundaries.
- A crash loses successful pages or provenance — tested with atomic shard/status ordering and resume fixtures.

### Task 1: Target inventory and request policy

**Files:** Create `src/r2ai/zh_v2/direct.py`; create `tests/test_zh_v2_direct.py`.

**Interfaces:** `build_targets(corpus_path, manifest_path, chunks_root)`, `client_options()`, `classify_response(...)`, `RollingErrors`, and `backoff_seconds(...)`.

- [x] Write failing tests for v2/live-ID subtraction, host allowlist, exact UA/no proxy/no redirects, 1-second pacing, challenge detection, Retry-After, and the first-500 / rolling-500 boundary.
- [x] Run the focused test file and confirm the expected missing implementation failures.
- [x] Implement the target builder and pure request-policy functions.
- [x] Run the focused tests and verify the target inventory against the measured 76,152 IDs (76,144 normalized URL groups).

### Task 2: Resumable direct fetch and raw provenance

**Files:** Modify `src/r2ai/zh_v2/direct.py`; create the package bootstrap/roots adapter if needed; modify `tests/test_zh_v2_direct.py`.

**Interfaces:** `prepare()`, `run()`, `status()`, and one-process lock; one exact URL request at a time; per-URL queue plus durable request-event log.

- [x] Add failing tests for redirect non-following, checkpoint/resume, atomic shard pointer ordering, error-stop, and row-level `source=direct` plus timestamp.
- [x] Run tests red, then implement and run tests green (22 focused tests pass).
- [x] After local outbound was enabled, probe one page from the v2 queue. The first iiyi sample (`https://article.iiyi.com/detail/10058.html`, ID 2,131,085) timed out on Windows; stop the sample set and omit iiyi.

### Task 3: QA extraction, 10k ingest and handoff contract

**Files:** Add the S15 `zh_v2` extraction/index/ingest modules to this checkout, adjusted only to read the direct queue; add iiyi QA selectors after the live probe; add focused extractor/provenance tests.

**Interfaces:** Existing `zh_v2.extract` output and `zh_v2.ingest` segment layout; `provenance.parquet` maps every v2 ID to segment `doc_id`, `url_norm`, `source`, and `fetched_at`.

- [x] Probe iiyi only after 120ask completes. The first of five requested samples timed out; record the result and skip the remaining four samples plus 729 article URLs. Do not infer selectors without HTML.
- [x] Add a provenance-sidecar unit test for v2 membership, row mapping, and good-text counting; actual emitted-segment audit remains pending.
- [ ] Verify existing chunking/embedding outputs and S15 manifest adoption without changing their algorithms or schemas.

### Task 4: Run and report

**Files:** Update `docs/PROGRESS.md`; write ignored run reports under `out/runs/zh-v2-direct/`.

- [ ] Run the approved crawl from the prepared, hash-verified target inventory; halt on the configured error threshold.
- [ ] Ingest each complete 10k batch and send S15 the paths/schema/counts as soon as each segment is ready.
- [ ] Report actual URL/document counts, success/error rates, runtime, AWS cost, good-text count, v2-ID coverage, and empty docs.
- [ ] Verify git status, stage only source/tests/docs allowlisted for this phase, then commit and push after the direct phase completes.

### Task 5: 120ask origin recovery check

**Files:** Create `src/r2ai/zh_v2/healthcheck.py`; create `tests/test_zh_v2_healthcheck.py`; update the S21 report and progress log.

- [x] Check one `HEAD` request every six hours, same local IP, truthful UA, no proxy or redirects, ending 2026-10-26.
- [x] On HTTP 200 only, launch the existing 120ask queue; classify 521 as origin down; keep a durable attempt log and resume-safe state.
- [x] Register the six-hour heartbeat and run the initial health check (first result: ConnectTimeout; queue unchanged).

**Current phase outcome (2026-10-17): 120ask unreachable; queue paused, monitoring enabled.** S11's latest request returned HTTP 521, classified per Vu as origin down. The first iiyi sample timed out on Windows; iiyi is skipped. The health-check script resumes only the 120ask queue after HTTP 200. See `out/runs/zh-v2-direct/REPORT.md` and `docs/PROGRESS.md`.
