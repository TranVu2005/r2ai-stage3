# S27 Pharmacity: Common Crawl and incremental handoff

## Goal

Recover usable text for the 5,182 pending mapped URLs from the latest 13 Common Crawl indexes, hand off accepted documents in immutable S15 lots, and resume live crawling only after the approved cooldown and one-probe gate. Never embed before S26 completes, and never write the shared index.

## Constraints

- Keep all source changes on `feat/s27-pharmacity` in its dedicated worktree. Do not modify the dirty primary checkout.
- Preserve existing S27 raw data and SQLite checkpoint. Never retry doc_id 317850. Keep the seven mismatched redirects raw and excluded.
- CC index first; WARC over HTTPS Range from `data.commoncrawl.org`, at most five workers, retry 503. Record exact and normalized URL match and full WARC provenance.
- A URL is live-probe eligible only if no CC capture yields accepted text and not every capture is 404/410. Report zero-capture, unusable-capture, and dead counts before any probe.
- Site requests are forbidden until 22:55:05 Asia/Bangkok on 2026-10-19. Then one probe; only ordinary HTTP 200 permits further requests, each at least five seconds apart with jitter. Any challenge/CAPTCHA stops permanently.
- Existing accepted docs must be exported immediately as `cc-s27-pharmacity`, S15 schema, t256/min 50 Vietnamese tokens. Next lots contain up to 1,000 new docs.
- No GPU/embedding until S26's 887,151-doc job is verifiably complete. The gate now passes; embed each delivered S15 lot with the same encoder/settings and keep vector files under S27's isolated handoff directory.

## Implementation and verification

1. Add tests for deterministic, duplicate-free S15 lot selection and CC URL matching/classification before implementing those APIs.
2. Add lot-scoped docs, provenance, chunks, hashes, and manifest without changing S15 schemas or writing a shared index. Export the current 146 accepted docs and verify counts and file integrity.
3. Add resumable CC index discovery/fetch/extraction, using the repo's CC parquet/WARC helpers. Match WARC target URI exactly and through `vicrawl.urlnorm.normalize_url`; select an accepted capture; classify (a), (b), and dead. Test with fixtures/mocks; run the real index scan and Range fetch only for mapped corpus URLs.
4. Add a persisted cooldown/probe gate and ≥5-second jittered live-resume loop. Test time gating, no retry of doc 317850, one-probe behavior, and permanent challenge halt. Implementation is ready; execute only after 22:55:05 UTC+7, and stop on any challenge.
   If a later challenge/block halts crawling, flush any accepted remainder below 1,000 docs as the final S15 lot before returning.
5. Update `docs/PROGRESS.md` after each completed step. Run focused S27 tests and output verification. Stage an allowlist only, commit on `feat/s27-pharmacity`, and push.

## Acceptance evidence

- CC report: 13 crawl IDs, exact/normalized coverage, accepted text, (a)/(b)/dead counts, low-title count and samples, Range request/error totals.
- Handoff manifest: batch name, immutable lot IDs, S15 artifact paths and SHA-256, doc/chunk counts, `shared_index_written=false`. The separate S26 vector handoff manifest lists per-lot doc-level vectors and checksums.
- Probe report: cooldown start/end, eligible pool categories, probe request/status, and either permanent halt or rate/jitter evidence. Current measured pre-probe groups: (a) 5,182, (b) 0, dead 0; earliest probe 22:55:05 UTC+7.
- S26 completion marker and exact vector shape verified before any S27 embedding; otherwise embedding remains pending.
