# r2ai.compose — rebuild C1 from pinned inputs

C1 (LB 0.2655) = base S6-f-K150 + zh v2 batch-2 docs/chunks, ZIP_LZMA dict 256 MiB. The builder was written in S8 under
`out/diag-lang/v2/compose` (ignored by git); it now lives here with tests (`tests/test_compose.py`).

    python -m r2ai.compose build   --manifest configs/best-c1.yaml --prepare-only   # pins, selection, plan; no JSON/ZIP
    python -m r2ai.compose build   --manifest configs/best-c1.yaml --json-only      # re-render the JSON, SHA256 gate + row diff, no ZIP
    python -m r2ai.compose replay  --manifest configs/best-c1.yaml                  # JSON gate -> ZIP -> native validator -> ZIP SHA report
    python -m r2ai.compose.replay_s6 --manifest configs/best-c1.yaml                # base S6-f JSON from cc-A-cut115 + pool-B cache
    python -m r2ai.compose.release_check --manifest configs/best-c1.yaml --zip out/runs/c1/replay/C1.zip   # before every upload
    python -m r2ai.compose.derive --manifest configs/best-c1.yaml --src C1.zip --dst C1-no-vinmec.zip --drop-host vinmec.com

## Rules

* `configs/best-c1.yaml` pins every input with a path and SHA256: base S6-f stats/JSON/ZIP, zh candidate (batch 2), zh docs snapshot
  manifest and its 57 `docs.parquet` files, `corpus_v2_ids.parquet`, `query.parquet`, the vi/CC docs directory (tree SHA256, native
  validator only), and for the S6-f replay the cc-A-cut115 JSON and the pool-B cache. A missing, malformed or mismatching pin raises
  `PinError` before any output is written. The selection policy (absolute tau_doc -0.87255859375, cap 30, ZL chunks, dict 256,
  ceiling 102,600,000 B) is fixed in `manifest.POLICY`; the manifest can only restate it.
* The gate is the JSON SHA256 (`expected.json_sha256`). On a difference the build stops, writes `<name>.replay-json.json` with the
  first rows that differ per query and field (docs only in reference / only in new, chunk ids and text hashes) and does not edit
  anything to match. ZIP bytes are reported (`expected.zip_sha256`) but are not the gate.
* Selection is unchanged from S8: doc rank order, primary-present skip, v2 aliases only; chunks = logit >= 0.96, top 3 docs,
  whole `doc_text`, one chunk per v2 id, only docs already selected; every base doc and chunk is kept.
* Output goes to `{OUT_DIR}/runs/c1/replay` (never overwrites; a partial publish stops). Nothing is uploaded.

## Layout

| module | from (S8 worktree `diag-lang`) |
|---|---|
| `base_io` | `generate_probe` (sha, baseline stream, lzma dictionary writer, candidate schema) |
| `zh_select` | `zg_pair.select_zg`, `prepare_sources`, `build_E.select_chunks` |
| `archive` | `build_E` (row assembly, `write_archive`, readback, pins, native-result checks) |
| `build` | `compose_builder` + `locked_compose` (locked batch-2 policy) |
| `native` | `validate_compose` (read-only doc-text adapter over the repo validator) |
| `manifest`, `replay_s6`, `release_check`, `derive` | new in S14 |
