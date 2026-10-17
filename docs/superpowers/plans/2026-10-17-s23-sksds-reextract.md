# S23 SKĐS Re-extract Plan

**Goal:** Re-extract the 1,336 local SKĐS WARC records with the S22 selector and, when there are new eligible documents, publish one isolated active CC segment for S15.

**Architecture:** Read only existing WARC and S23 provenance; use the `vicrawl` selector whose source blob matches `feat/s22-reextract`; filter to documents meeting the established 200-character and 50-token criteria and absent from segment 20004; run the existing t256 chunker and BGE-M3 segment embedder with a new CC manifest row. Update the S23 report and `docs/PROGRESS.md`.

**Tech Stack:** Python, gzip/WARC parsing, pyarrow, `vicrawl.extractors`, BAAI/bge-m3, SQLite segment manifest.

**Spec:** User request in this task, 2026-10-17; S15 VI CC segment layout was confirmed by Vu earlier in this session.

## Global Constraints

- Use the 1,336 downloaded WARC files only; do not fetch network data.
- Keep segment 20004 unchanged and exclude its document IDs from the new segment.
- Write no common index data; use only the CC segment vector path and `state/cc_longchau/manifest.db`.
- Do not continue Lao Động; record dead-in-CC based on the observed Cloudrity challenge. VOV direct crawl is assigned to S24.
- Use `r2ai.paths` roots and guard write targets before I/O; never write under OLD.
- Preserve crawl ID, WARC date, and capture timestamp provenance for each segment document.

## Review Focus

- WARC target, record ID, and WARC date must match the stored provenance row.
- Token threshold and old-segment ID exclusion must match the S15 segment contract.
- Manifest rows, docs, chunks, and vector rows must agree for the new segment.
- Segment 20004 hashes and manifest fields must remain unchanged.
- No writes may reach `data/index/t256` or another common index.

### Task 1: Offline re-extraction

Read and parse all 1,336 local WARC records with the S22-matched SKĐS extractor; count valid WARC, ≥200-character docs, ≥50-token docs, and newly eligible IDs after excluding segment 20004. Preserve source provenance. Do not create a segment if the new eligible count is zero.

### Task 2: New segment, if eligible docs exist

Write the new batch source docs and provenance under `cc-s23-sksds-20261017/vi-news`; allocate the next CC segment, run the existing t256 chunker and BGE-M3 embedder, and activate it in the CC manifest.

### Task 3: Report and delivery

Update the S23 report and `docs/PROGRESS.md`; verify segment 20004 unchanged, new segment row/files/provenance/embeddings, and no common-index writes. Commit and push the S23 branch, then close this session.
