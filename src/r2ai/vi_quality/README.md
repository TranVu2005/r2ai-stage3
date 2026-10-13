# r2ai.vi_quality (S12)

Rule-based quality tools for vi docs. No LLM. Nothing here is applied to the index or to any submission; every tool reads
`data/chunks/docs.parquet` / a submission JSON and writes under `reports/`.

| module | use |
|---|---|
| `rules.py` | `strip_tail` (cut a boilerplate tail: laodong comment widget, giadinhonline "Nên đọc"/"Tin liên quan", baonghean ONECMS footer) and `drop_reason` (`lottery_result`, `non_medical_template`, `error_title`, `title_is_site`, `menu_only`) |
| `scan` | per-host counts, drop / strip / duplicate lists, chunk impact on `chunks_t256` |
| `k80_impact` | what the rules would change in a submission JSON (docs, chunks, bytes) |
| `chunker.py` | structure-aware chunker (heading + body, Q&A pair, list/table kept whole, no overlap, verbatim spans) |
| `compare_chunkers` | old t256 vs structure chunker on a doc sample (chunk count, tokens, bytes, ZIP) |

## Chunker usage (decided 13/10, S12 close)

- The structure-aware chunker (`chunker.structured_chunks`) is the **default for the en index (S10b)**.
- **zh keeps t256** (`r2ai.index.chunk`): the zh reranker threshold τ = 0,96 (ZG) was calibrated on t256 chunks, so changing the
  zh chunker would shift τ.
- **Submissions keep `chunk_mode full`** (whole answer / body per doc). The probe that replaced the body by sub-chunks was
  rejected by C0 (window ±1, Chunk F2 0,1312), so the chunker only affects retrieval over the index, not `chunk_text`.
- The vi index is not rebuilt: `data/chunks/chunks_t256.parquet` is unchanged.

## Numbers (reports/s12_vi_quality.md)

Index scan: 9.354 docs would be dropped (5.793 lottery results, 3.555 non-medical templates, 4 menu-only, 2 `/404` titles) and
2.208 docs stripped. On the K80 submission the rules touch 86 chunks / 86.132 B (0,014% of chunk_text), so no ZIP gain.

    python -m r2ai.vi_quality.scan --out-dir reports
    python -m r2ai.vi_quality.k80_impact --json SUB.json --report reports/s12_vi_k80_impact.json
    python -m r2ai.vi_quality.compare_chunkers --per-host 300
