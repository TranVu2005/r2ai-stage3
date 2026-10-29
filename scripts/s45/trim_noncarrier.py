"""S45 optional post-processing (Vu 29/10): drop the deep non-carrier docs of a submission ZIP, keep every chunk byte for byte.

A non-carrier is a relevant_docs entry of a row with no chunk of that row. It is dropped when its cosine rank in the run's pool table
is above ``--max-rank`` (default 250: P-D250-doconly +0.0005 on the public LB, DO-200 +0.0003, docs/s45.md 5f-5g). Carriers stay
whatever their rank; the doc order of the row is kept; relevant_chunks are not touched, so chunk subset of relevant_docs still holds
(checked here, then by the S26 validator, ``--max-chunks-per-doc 2 --no-p1``). Not part of newq: a stand-alone step for a side ZIP.

  trim_noncarrier.py <in.zip> <cosine.parquet> <out.zip> [--max-rank 250] [--queries <queries.parquet>] [extra_dir ...]

``cosine.parquet``: the run's pool order with ``query_id, doc_id, cosine_rank`` (newq run ``top.parquet``; public R0: S45
``exact/top300.parquet``). ``--queries``: a new query set (as the validator's option). ``extra_dir``: the validator's extra verbatim
sources (Pharmacity handoff, the ``.validator-extra`` folder of the compose).
"""
import argparse, hashlib, json, os, subprocess, sys, zipfile
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
    ap = argparse.ArgumentParser()
    ap.add_argument('zip', type=Path); ap.add_argument('cosine', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('extra', nargs='*'); ap.add_argument('--max-rank', type=int, default=250); ap.add_argument('--queries')
    args = ap.parse_args()
    if args.out.resolve() == args.zip.resolve():
        sys.exit('out.zip must differ from the input ZIP')
    top = pd.read_parquet(args.cosine, columns=['query_id', 'doc_id', 'cosine_rank'])
    cos = dict(zip(zip(top.query_id.astype(int), top.doc_id.astype(int)), top.cosine_rank.astype(int)))
    rows, dropped, non_carriers, outside = [], [], [], 0
    for r in rows_of(args.zip):
        q = int(r['id']); own = {int(c['doc_id']) for c in r['relevant_chunks']}
        missing = [d for d in r['relevant_docs'] if d not in own and (q, d) not in cos]
        if missing:
            sys.exit(f'query {q}: {len(missing)} non-carrier docs without a cosine rank in {args.cosine}, e.g. {missing[:3]}')
        docs = [d for d in r['relevant_docs'] if d in own or cos[(q, d)] <= args.max_rank]
        outside += len(own - set(docs))
        non_carriers.append(sum(d not in own for d in r['relevant_docs'])); dropped.append(len(r['relevant_docs']) - len(docs))
        rows.append({**r, 'relevant_docs': docs})
    if outside:
        sys.exit(f'{outside} chunks would lose their doc: not written')
    res = write_bundle(rows, args.out, args.out.with_suffix('.json').name)
    n = len(rows)
    rep = {'in_zip': str(args.zip), 'in_zip_sha256': hashlib.sha256(args.zip.read_bytes()).hexdigest(), 'cosine': str(args.cosine),
           'max_rank': args.max_rank, 'out_zip': str(args.out), 'zip_bytes': args.out.stat().st_size, **res,
           'queries': n, 'docs': sum(len(r['relevant_docs']) for r in rows), 'chunks': sum(len(r['relevant_chunks']) for r in rows),
           'non_carriers_per_query': sum(non_carriers) / n, 'docs_dropped_per_query': sum(dropped) / n,
           'queries_changed': sum(k > 0 for k in dropped), 'chunk_doc_not_in_docs': outside}
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/s26/validate_pa3.py'), '--max-chunks-per-doc', '2', '--no-p1']
    cmd += ['--queries', args.queries] if args.queries else []
    v = subprocess.run(cmd + [str(args.out), *args.extra], capture_output=True, text=True, encoding='utf-8',
                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    args.out.with_suffix('.validate.json').write_text(v.stdout, encoding='utf-8')
    vj = json.loads(v.stdout)
    rep['validator'] = {'ok': vj['ok'], 'zip_sha256': vj['zip_sha256'], 'errors': {k: x for k, x in vj['errors'].items() if x},
                        'chunk_check': vj['chunk_check']}
    args.out.with_suffix('.trim.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))
    sys.exit(0 if vj['ok'] else 1)


if __name__ == '__main__':
    main()
