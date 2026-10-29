"""S45 probe EN-translated (Vu 29/10, low priority): does the public gold hold en docs that a TRANSLATED query would retrieve?

  en_translated.py translate <out>   1,200 queries vi -> en with a local Ollama model (temperature 0, seed 0), checkpoint translations.jsonl
  en_translated.py encode <out>      BGE-M3 dense CLS of the translations (S26 recipe, r2ai.s36.enlane.embed_texts), top-20 en docs per query by
                                     cosine over the en lane (data/s37_en/seg37900-en-pubmed, 239,771 PubMed docs) -> pairs_en_tr.parquet
  en_translated.py zip <out>         CH-rrf ZIP (b12b8591...) + the top-20 en docs appended at the END of relevant_docs (after the CH-rrf docs,
                                     cosine order, docs already present skipped). No chunk for the en docs: 24,000 en chunks (~410 B each in
                                     EN-hedge3) would push the ZIP past the BTC ceiling, and the probe reads Doc R only. S26 validator, SHA.

Reading (Vu): LB Doc R above CH-rrf => the public gold has en docs of the translated kind (reopen en for private); Doc R unchanged => close en.
Run from the worktree with R2AI_OUT_DIR = out/runs/s45/restore/out (validator inputs) for ``zip``.
"""
import json, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd  # noqa: E402

MODEL = 'qwen2.5:7b'                     # Ollama id 845dbda0ea48, Qwen2.5-7B-Instruct GGUF Q4_K_M
OLLAMA = 'http://127.0.0.1:11434'
SYSTEM = 'You translate Vietnamese medical questions into English. Output only the English translation, nothing else.'
QUERIES = Path('D:/GitHub/r2ai-stage3/data/raw/query.parquet')
CH_RRF_ZIP = Path('D:/_r2ai_archive/zips/out/runs/s41/ch/compose/CH-rrf-full+w768-t1536.zip')
CH_RRF_SHA = 'b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8'
TOP = 20


def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode()), timeout=600))


def translate(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ck = out / 'translations.jsonl'
    done = {json.loads(x)['id'] for x in ck.read_text(encoding='utf-8').splitlines()} if ck.exists() else set()
    show = post('/api/show', {'model': MODEL})
    (out / 'model.json').write_text(json.dumps({'model': MODEL, 'ollama_id': '845dbda0ea48', 'modified_at': show.get('modified_at'),
                                                'details': show.get('details'), 'system': SYSTEM,
                                                'options': {'temperature': 0, 'seed': 0, 'num_predict': 256}}, indent=1), encoding='utf-8')
    q = pd.read_parquet(QUERIES)
    t0 = time.monotonic()
    with ck.open('a', encoding='utf-8') as f:
        for i, (qid, text) in enumerate(zip(q.id.astype(int), q['query'])):
            if qid in done:
                continue
            r = post('/api/chat', {'model': MODEL, 'stream': False, 'options': {'temperature': 0, 'seed': 0, 'num_predict': 256},
                                   'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]})
            f.write(json.dumps({'id': qid, 'vi': text, 'en': r['message']['content'].strip()}, ensure_ascii=False) + '\n'); f.flush()
            if i % 50 == 0:
                print(f'{i}/{len(q)} {time.monotonic() - t0:.0f}s', flush=True)
    print('translate done', flush=True)


STRICT = ('Translate the Vietnamese text between the triple quotes into English. Do not answer it, do not add advice or comments; '
          'keep it a question if it is a question. Output only the translation.')


def flagged(t: pd.DataFrame) -> pd.Series:
    """Translations that look like an answer, not a translation: more than 2x the Vietnamese length, or a question mark lost."""
    return (t.en.str.len() > 2 * t.vi.str.len()) | (t.vi.str.contains(r'\?') & ~t.en.str.contains(r'\?'))


def fix(out: Path):
    """Re-translate the flagged rows once with the stricter prompt (appended to translations.jsonl; the last row of an id wins)."""
    ck = out / 'translations.jsonl'
    t = pd.read_json(ck, lines=True).drop_duplicates('id', keep='last')
    bad = t[flagged(t)]
    print('flagged', len(bad), flush=True)
    with ck.open('a', encoding='utf-8') as f:
        for qid, text in zip(bad.id.astype(int), bad.vi):
            r = post('/api/chat', {'model': MODEL, 'stream': False, 'options': {'temperature': 0, 'seed': 0, 'num_predict': 256},
                                   'messages': [{'role': 'system', 'content': STRICT}, {'role': 'user', 'content': f'"""{text}"""'}]})
            f.write(json.dumps({'id': qid, 'vi': text, 'en': r['message']['content'].strip().strip('"').strip(), 'retry': 1},
                               ensure_ascii=False) + '\n'); f.flush()
    t = pd.read_json(ck, lines=True).drop_duplicates('id', keep='last')
    print('still flagged after retry', int(flagged(t).sum()), flush=True)


def encode(out: Path):
    from r2ai.s36 import enlane
    tr = pd.read_json(out / 'translations.jsonl', lines=True).drop_duplicates('id', keep='last').sort_values('id')
    if len(tr) != 1200 or (tr.en.str.len() == 0).any():
        raise ValueError('translations incomplete')
    q_ids = tr.id.astype(np.int64).to_numpy()
    q_dense = enlane.embed_texts(tr.en.tolist(), 'cuda')
    np.savez(out / 'query_vectors_en.npz', q_ids=q_ids, q_dense=q_dense)
    ids, vectors = enlane.load_vectors()
    pairs = enlane.top_pairs(q_ids, q_dense, ids, vectors, TOP)
    pairs.to_parquet(out / 'pairs_en_tr.parquet', index=False)
    print(json.dumps({'pairs': len(pairs), 'en_docs': int(len(ids)), 'cos_p10_p50_p90': pairs.cosine_score.quantile([.1, .5, .9]).round(4).tolist()}))


def make_zip(out: Path):
    from r2ai.s29.qwen_judge import _atomic_json, _sha256
    from r2ai.s38 import compose
    from r2ai.s38.build import write_bundle
    from r2ai.s38.sources import read_bundle
    from r2ai.s43 import enhedge

    if _sha256(CH_RRF_ZIP) != CH_RRF_SHA:
        raise ValueError('base ZIP is not CH-rrf')
    pairs = pd.read_parquet(out / 'pairs_en_tr.parquet').sort_values(['query_id', 'en_rank'])
    rows = read_bundle(CH_RRF_ZIP)
    base = {int(r['id']): list(r['relevant_docs']) for r in rows}
    by_q = {int(q): g.doc_id.astype(int).tolist() for q, g in pairs.groupby('query_id')}
    added = {}
    for r in rows:
        have = set(r['relevant_docs'])
        new = [d for d in by_q.get(int(r['id']), []) if d not in have]
        r['relevant_docs'] = r['relevant_docs'] + new
        added[int(r['id'])] = len(new)
    label = 'EN-tr20-docs'
    target = out / f'{label}.zip'
    rep = {'label': label, 'base_zip_sha256': CH_RRF_SHA, 'model': MODEL, 'added': int(sum(added.values())),
           'queries_with_20': int(sum(v == TOP for v in added.values())),
           'base_prefix_kept': all(r['relevant_docs'][:len(base[int(r['id'])])] == base[int(r['id'])] for r in rows)}
    rep.update(write_bundle(rows, target, f'{label}.json'))
    rep.update({'zip': str(target), 'zip_bytes': target.stat().st_size, 'zip_sha256': _sha256(target),
                'under_ceiling': target.stat().st_size < compose.BTC_CEILING - compose.RESERVE,
                'docs': sum(len(r['relevant_docs']) for r in rows), 'chunks': sum(len(r['relevant_chunks']) for r in rows)})
    ids = {int(d) for r in rows for d in r['relevant_docs']}
    del rows
    extra = enhedge.write_extra(out / f'{label}.validator-extra-en', enhedge.en_texts(pairs.doc_id.unique()))
    compose.NO_P1 = True             # P1-vi+10.zip (the validator's P1 verbatim source) was deleted; CH-rrf alone is validated the same way
    rep['validator'] = compose.run_validator(out, label, 'full+w768', target, ids, [extra])
    rep['validator_base'] = compose.run_validator(out, 'CH-rrf-base', 'full+w768', CH_RRF_ZIP, set().union(*map(set, base.values())), [])
    _atomic_json(out / f'{label}.json.report', rep)
    print(json.dumps({k: v for k, v in rep.items() if k != 'validator'}, default=str), 'validator ok', rep['validator'].get('ok'))


if __name__ == '__main__':
    {'translate': translate, 'fix': fix, 'encode': encode, 'zip': make_zip}[sys.argv[1]](Path(sys.argv[2]))
