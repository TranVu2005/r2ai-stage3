"""S45 doc-only probe (Vu 29/10): swap only the doc list of a composed ZIP, keep EVERY chunk byte for byte.

relevant_docs of query q = the docsel docs (docsel rank order) + the carrier docs of the base ZIP that the docsel lacks (base order),
so every chunk keeps its doc_id in relevant_docs (Vu: chunk subset of relevant_docs is mandatory). relevant_chunks = the base row's,
unchanged. Written with the S38 bundle writer (LZMA dict 256 MiB, native layout), then the S26 validator (``--max-chunks-per-doc 2
--no-p1`` + the base's extra sources), whose ``chunk_doc_not_in_docs`` is the chunk-in-docs check.

``trim`` (Vu 29/10, non-carrier tail cut): relevant_docs = the base ZIP's docs (base order) without the non-carrier docs (no chunk in
the row) whose cosine rank (docsel column ``cosine_rank``) is above ``MAX``; ``nc`` drops every non-carrier (relevant_docs = the
chunk owners). Chunks unchanged.

  doc_only.py <base.zip> <docsel.parquet> <out.zip> [extra_dir ...]
  doc_only.py trim <MAX|nc> <base.zip> <docsel.parquet> <out.zip> [extra_dir ...]
"""
import json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd  # noqa: E402
from r2ai.s38.build import write_bundle  # noqa: E402


def rows_of(zp):
    with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            line = line.strip().rstrip(b',')
            if line.startswith(b'{'):
                yield json.loads(line)


def main():
    args = sys.argv[1:]
    trim = args[1] if args[0] == 'trim' else None
    if trim is not None:
        args = args[2:]
    base, docsel, out, extra = Path(args[0]), Path(args[1]), Path(args[2]), args[3:]
    sel = pd.read_parquet(docsel)
    order = {int(q): g.sort_values('rank').doc_id.astype(int).tolist() for q, g in sel.groupby('query_id')}
    cos = dict(zip(zip(sel.query_id.astype(int), sel.doc_id.astype(int)), sel.cosine_rank.astype(int)))
    rows, kept, dropped, added = [], [], [], []
    for r in rows_of(base):
        q = int(r['id']); new = order.get(q, []); have = set(new)
        carriers = list(dict.fromkeys(int(c['doc_id']) for c in r['relevant_chunks']))
        if trim is not None:          # base docs minus the non-carriers past the cut (nc: every non-carrier)
            own = set(carriers)
            docs = [d for d in r['relevant_docs'] if d in own or (trim != 'nc' and cos[(q, d)] <= int(trim))]
            extra_car = []
        else:
            extra_car = [d for d in r['relevant_docs'] if d in set(carriers) and d not in have]
            docs = new + extra_car
        old = set(r['relevant_docs'])
        kept.append(len(extra_car)); dropped.append(len(old - set(docs))); added.append(len(set(docs) - old))
        rows.append({**r, 'relevant_docs': docs})
    out.parent.mkdir(parents=True, exist_ok=True)
    res = write_bundle(rows, out, out.with_suffix('.json').name)
    n = len(rows)
    rep = {'base': str(base), 'docsel': str(docsel), 'trim': trim, 'zip': str(out), 'zip_bytes': out.stat().st_size, **res,
           'docs': sum(len(r['relevant_docs']) for r in rows), 'chunks': sum(len(r['relevant_chunks']) for r in rows),
           'base_carriers_outside_docsel_per_query': sum(kept) / n, 'queries_with_such_carrier': sum(k > 0 for k in kept),
           'docs_dropped_per_query': sum(dropped) / n, 'docs_added_per_query': sum(added) / n}
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/s26/validate_pa3.py'), '--max-chunks-per-doc', '2', '--no-p1', str(out), *extra]
    v = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    out.with_suffix('.validate.json').write_text(v.stdout, encoding='utf-8')
    vj = json.loads(v.stdout)
    rep['validator'] = {'ok': vj['ok'], 'zip_sha256': vj['zip_sha256'], 'errors': {k: x for k, x in vj['errors'].items() if x},
                        'chunk_check': vj['chunk_check']}
    out.with_suffix('.doconly.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
