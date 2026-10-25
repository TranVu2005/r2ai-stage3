"""English PubMed segment for canonical-20261025 (S37): doc_text, doc-level BGE-M3 vectors, manifest with SHA256 of every file.

    python -m r2ai.s37.en build                          # data/s37_en/seg37900-en-pubmed/{docs.parquet, manifest.json} from data/pubmed_en
    python -m r2ai.s37.en embed --device cuda --max-util 70   # resumable (parts of 10,000 docs), then vectors.npz
    python -m r2ai.s37.en report                         # CPU: how many en docs land in pool13 / top-300 if ranked together with vi+zh

doc_text = title + "\\n\\n" + abstract sections joined by "\\n\\n" (a labelled section is "label\\ntext"); verified on 3,000 sampled docs: every
chunk offset (chunks.parquet start/end) of the existing PubMed chunks lands on this text. Title-only docs (no abstract) are quality `thin`.
Embedding recipe = S26 `full_embed.py` through r2ai.s37.ingest.embed_texts: first 4096 chars, max_len 512, dense CLS, L2, fp16; length-sorted
batches (64 / 32 / 16 by text length). Nothing is scored here (no 4B / Jina / C-max for en).
"""
from __future__ import annotations

from r2ai.paths import DATA_DIR, assert_writable

import argparse
import json
import time

import numpy as np
import pandas as pd

from r2ai.s37.ingest import EMBED_CHARS, EMBED_LEN, atomic_json, gpu_free, sha256_file

SRC = DATA_DIR / 'pubmed_en'
SEG = DATA_DIR / 's37_en' / 'seg37900-en-pubmed'
BLOCK = 10_000


def doc_text(title: str, sections, present: bool) -> str:
    if not present:
        return title
    body = '\n\n'.join((f"{x['label']}\n{x['text']}" if x['label'] else x['text']) for x in sections)
    return title + '\n\n' + body


def build() -> dict:
    t0 = time.time()
    r = pd.read_parquet(SRC / 'records.parquet', columns=['doc_id', 'pmid', 'title', 'abstract_sections', 'abstract_present', 'journal', 'year', 'source_url'])
    text = [doc_text(t, s, p) for t, s, p in zip(r.title, r.abstract_sections, r.abstract_present)]
    docs = pd.DataFrame({'doc_id': r.doc_id.astype('int64'), 'pmid': r.pmid.astype('int64'), 'lang': 'en', 'title': r.title, 'doc_text': text,
                         'quality': np.where(r.abstract_present, 'good', 'thin'), 'journal': r.journal, 'year': r.year.astype('int32'),
                         'source_url': r.source_url})
    docs = docs.sort_values('doc_id').reset_index(drop=True)
    assert docs.doc_id.is_unique and len(docs) == 239_771
    assert_writable(SEG)
    SEG.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(SEG / 'docs.parquet', index=False)
    n_chars = docs.doc_text.str.len()
    man = {'schema': 'r2ai.s37.segment-en.v1', 'segment': SEG.name, 'canonical_base': 'canonical-20261018 (unchanged); this segment joins canonical-20261025',
           'source': {'records.parquet': sha256_file(SRC / 'records.parquet'), 'chunks.parquet': sha256_file(SRC / 'chunks.parquet')},
           'doc_text_rule': 'title + "\\n\\n" + abstract sections joined by "\\n\\n" (labelled section = "label\\ntext"); title only when no abstract',
           'docs': len(docs), 'docs_good': int((docs.quality == 'good').sum()), 'docs_thin': int((docs.quality == 'thin').sum()),
           'chars_p50_p90': [int(n_chars.quantile(.5)), int(n_chars.quantile(.9))], 'doc_id_range': [int(docs.doc_id.min()), int(docs.doc_id.max())],
           'files_sha256': {'docs.parquet': sha256_file(SEG / 'docs.parquet')}, 'state': 'waiting_gpu', 'seconds': round(time.time() - t0, 1)}
    atomic_json(SEG / 'manifest.json', man)
    return man


def embed(device: str, max_util: int) -> dict:
    from r2ai.index.bge_m3 import M3Encoder
    docs = pd.read_parquet(SEG / 'docs.parquet', columns=['doc_id', 'doc_text'])
    parts = SEG / 'vec_parts'
    parts.mkdir(exist_ok=True)
    n_blocks = (len(docs) + BLOCK - 1) // BLOCK
    todo = [b for b in range(n_blocks) if not (parts / f'part-{b:04d}.npy').exists()]
    if todo:
        ok, msg = gpu_free(3072, max_util)
        if not ok:
            raise SystemExit(f'GPU busy: {msg}')
        enc = M3Encoder(device=device, fp16=(device != 'cpu'), max_len=EMBED_LEN)
        t0 = time.time()
        for k, b in enumerate(todo):
            ok, msg = gpu_free(1024, 100)
            texts = [t[:EMBED_CHARS] for t in docs.doc_text.iloc[b * BLOCK:(b + 1) * BLOCK]]
            order = np.argsort([len(x) for x in texts], kind='stable')
            out = np.zeros((len(texts), 1024), np.float16)
            i = 0
            while i < len(order):
                bs = 64 if len(texts[order[i]]) < 600 else 32 if len(texts[order[i]]) < 1500 else 16
                sel = order[i:i + bs]
                out[sel] = enc.encode_batch([texts[j] for j in sel])[0]
                i += bs
            tmp = parts / f'part-{b:04d}.tmp.npy'
            np.save(tmp, out)
            tmp.replace(parts / f'part-{b:04d}.npy')
            el = time.time() - t0
            print(json.dumps({'block': b, 'of': n_blocks, 'blocks_done_this_run': k + 1, 'elapsed_s': round(el), 'eta_s': round(el / (k + 1) * (len(todo) - k - 1))}), flush=True)
    vec = np.concatenate([np.load(parts / f'part-{b:04d}.npy') for b in range(n_blocks)])
    assert vec.shape == (len(docs), 1024) and np.isfinite(vec.astype(np.float32)).all()
    norms = np.linalg.norm(vec.astype(np.float32), axis=1)
    assert abs(norms - 1).max() < 5e-3, 'vectors are not L2-normalised'
    np.savez(SEG / 'vectors.npz', doc_ids=docs.doc_id.to_numpy(np.int64), vectors=vec)
    man = json.loads((SEG / 'manifest.json').read_text('utf-8'))
    man['files_sha256']['vectors.npz'] = sha256_file(SEG / 'vectors.npz')
    man.update({'state': 'embedded', 'embed': {'model': 'BAAI/bge-m3 dense CLS L2 fp16', 'max_len': EMBED_LEN, 'chars': EMBED_CHARS, 'rows': len(vec),
                                              'norm_min_max': [float(norms.min()), float(norms.max())]}})
    atomic_json(SEG / 'manifest.json', man)
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('build', 'embed', 'report'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-util', type=int, default=10)
    a = ap.parse_args(argv)
    if a.command == 'build':
        res = build()
    elif a.command == 'embed':
        res = embed(a.device, a.max_util)
    else:
        from r2ai.s37.enreport import report
        res = report()
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
