# Direct crawl v2: 120ask and iiyi

## Goal

Fetch missing-live-document 120ask and iiyi pages from the v2 Chinese corpus, extract question/answer text, and add zh segments that S15 can probe in 10,000-document batches.

## Scope

- Use the v2 corpus at `data/corpus_v2_ids.parquet` as the only URL allowlist.
- Direct-fetch only v2 IDs without a live document in `data/index/zh_v2_segments/manifest.json` and its active segment docs.
- Current measured scope: 75,418 `120ask.com`, 729 `article.iiyi.com`, 5 `bingli.iiyi.com` IDs; 76,152 IDs total before URL normalization/grouping.
- Skip Common Crawl. The user authorized direct access under D20.

## Fetch policy

- One local process and one HTTP client; no proxy and no IP rotation.
- Exact User-Agent: `r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)`.
- At least one second between requests globally, plus positive jitter. This is stricter than the per-host ceiling and matches the approximately 21-hour estimate.
- Do not follow redirects or request any URL outside the v2 allowlist.
- On HTTP 403, 429, CAPTCHA, or bot-blocking pages, use exponential backoff, capped at five minutes except when `Retry-After` requires longer; never solve or bypass a challenge.
- After the first 500 HTTP requests, stop if more than 20% of the latest 500 are errors; then keep applying the rolling window.
- Persist per-URL state, request events, raw shards, and direct source/timestamp provenance. Resume without refetching successful pages.

## Extraction and handoff

- Use the existing 120ask question/answer extractor and the S15 `zh_v2` ingest/index contract.
- Crawl all 120ask targets first. Only after 120ask is complete, fetch five iiyi sample pages, inspect their markup, and infer selectors; if iiyi is still DNS/timeout-blocked, omit iiyi and record the reason. If bot-blocked or structure is ambiguous, do not guess.
- Build each completed batch from 10,000 fetched HTML page groups, ingest it into the zh-v2 bundle, and hand off completed segments as they appear; ingest a smaller final batch if the crawl stops or completes below 10,000.
- Each handoff reports segment and chunk paths/schema, doc count, 100% v2 `doc_id` check, empty-doc count, and good-text count.
- Update the progress report, then commit/push the completed direct phase. Do not commit corpus, state, raw data, output, credentials, or tokens.

## Stop conditions

- Never solve CAPTCHA or evade a bot challenge.
- Stop on an error ratio above 20% in the latest 500 requests.
- Stop before any fetch if the target queue contains an ID/URL outside v2 or any output path resolves into OLD.
- Per Vu's follow-up decision, if local S11 logs show a prior HTTP 200 for `120ask.com/robots.txt`, stop Phase B and mark the phase blocked; do not fetch page content or proceed to iiyi.
