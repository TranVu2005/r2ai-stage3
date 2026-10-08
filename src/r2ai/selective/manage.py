"""Apply only after stopped/WAL0; rank/status reversible until any selected URL is fetched."""
from __future__ import annotations
from .runio import RUN, DB, guard, preflight, ro, writer, atomic_json
from r2ai.zh_full.common import exclusive
from r2ai.zh_full.selection_queue import next_batch

import argparse
from collections import Counter
import hashlib
import json
import shutil
import sqlite3
import time
from types import SimpleNamespace

SCHEMA = '''CREATE TABLE IF NOT EXISTS select_meta(
 url_norm TEXT PRIMARY KEY, domain TEXT NOT NULL, old_status TEXT NOT NULL,
 old_rank INTEGER NOT NULL, score REAL NOT NULL, method TEXT NOT NULL,
 cut_version TEXT NOT NULL, new_status TEXT NOT NULL, new_rank INTEGER NOT NULL,
 baseline_sha TEXT NOT NULL);
 CREATE INDEX IF NOT EXISTS select_meta_domain_score ON select_meta(domain,score DESC,url_norm);
 CREATE INDEX IF NOT EXISTS urls_select_rank ON urls(domain,rank,url_norm);'''


def digest_row(row):
    return hashlib.sha256(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def urls_digest(conn):
    h=hashlib.sha256()
    for row in conn.execute('SELECT * FROM urls ORDER BY url_norm'):
        h.update(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode()+b'\n')
    return h.hexdigest()


def baseline_digest(row, columns):
    return digest_row([v for c,v in zip(columns,row) if c not in ('rank','status')])


def apply_rows(conn, rows, version):
    rows=list(rows)
    if not rows or len({r['url_norm'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate selection plan')
    conn.execute('BEGIN IMMEDIATE')
    try:
        # executescript commits implicitly, so execute each statement separately.
        for sql in SCHEMA.split(';'):
            if sql.strip(): conn.execute(sql)
        if conn.execute('SELECT 1 FROM select_meta LIMIT 1').fetchone():
            raise ValueError('Selection already applied; rollback before another cut version')
        by_domain={}
        for r in rows: by_domain.setdefault(r['domain'],[]).append(r)
        for domain, group in by_domain.items():
            actual={r[0] for r in conn.execute("SELECT url_norm FROM urls WHERE domain=? AND status IN ('pending','network_error')",(domain,))}
            if actual!={r['url_norm'] for r in group}:
                raise ValueError(f'Stale/incomplete eligible domain: {domain}')
            if sorted(r['old_rank'] for r in group)!=sorted(r['new_rank'] for r in group):
                raise ValueError(f'Rank permutation differs: {domain}')
        cur=conn.execute('SELECT * FROM urls LIMIT 0')
        columns=[v[0] for v in cur.description]
        for r in rows:
            cur=conn.execute('SELECT * FROM urls WHERE url_norm=?',(r['url_norm'],))
            old=cur.fetchone(); value=dict(zip(columns,old)) if old else {}
            if (value.get('domain')!=r['domain'] or value.get('rank')!=r['old_rank']
                    or value.get('status')!=r['old_status'] or r['old_status'] not in ('pending','network_error')
                    or r['new_status'] not in (r['old_status'],'deferred_select')):
                raise ValueError(f'Stale/invalid selection: {r["url_norm"]}')
            conn.execute('INSERT INTO select_meta VALUES(?,?,?,?,?,?,?,?,?,?)',
                (r['url_norm'],r['domain'],r['old_status'],r['old_rank'],float(r['score']),r['method'],version,
                 r['new_status'],r['new_rank'],baseline_digest(old,columns)))
            conn.execute('UPDATE urls SET rank=?,status=? WHERE url_norm=?',(r['new_rank'],r['new_status'],r['url_norm']))
        conn.execute('COMMIT')
        return len(rows)
    except BaseException:
        conn.execute('ROLLBACK')
        raise


def reopen(conn, domain, top):
    if top<0: raise ValueError('top must be nonnegative')
    conn.execute('BEGIN IMMEDIATE')
    try:
        rows=conn.execute('''SELECT u.url_norm,m.old_status FROM urls u JOIN select_meta m USING(url_norm)
            WHERE u.domain=? AND u.status='deferred_select'
            ORDER BY m.score DESC,json_array_length(u.doc_ids) DESC,u.url_norm LIMIT ?''',(domain,top)).fetchall()
        conn.executemany('UPDATE urls SET status=? WHERE url_norm=?',[(s,u) for u,s in rows])
        conn.execute('COMMIT')
        return len(rows)
    except BaseException:
        conn.execute('ROLLBACK'); raise


def rollback(conn, version):
    conn.execute('BEGIN IMMEDIATE')
    try:
        meta=conn.execute('SELECT url_norm,old_status,old_rank,baseline_sha,new_status,new_rank FROM select_meta WHERE cut_version=?',(version,)).fetchall()
        if not meta: raise ValueError('Unknown cut_version')
        columns=[r[1] for r in conn.execute('PRAGMA table_info(urls)')]
        for u,status,rank,sha,new_status,new_rank in meta:
            row=conn.execute('SELECT * FROM urls WHERE url_norm=?',(u,)).fetchone()
            current=dict(zip(columns,row)) if row else {}
            if (not row or baseline_digest(row,columns)!=sha or current['rank']!=new_rank
                    or current['status'] not in (status,new_status)):
                raise ValueError(f'Selected URL changed after apply; refuse rollback: {u}')
        conn.executemany('UPDATE urls SET status=?,rank=? WHERE url_norm=?',[(s,r,u) for u,s,r,*_ in meta])
        conn.execute('DELETE FROM select_meta WHERE cut_version=?',(version,))
        conn.execute('COMMIT')
        return len(meta)
    except BaseException:
        conn.execute('ROLLBACK'); raise


def stopped_wal0(path=DB):
    from r2ai.zh_full.watchdog import _live_procs
    if _live_procs(): raise RuntimeError('Crawler still running')
    wal=path.with_name(path.name+'-wal')
    if wal.exists() and wal.stat().st_size: raise RuntimeError('WAL must be 0 before selection/backup')
    with ro(path) as conn:
        if conn.execute("SELECT 1 FROM urls WHERE status='in_progress' LIMIT 1").fetchone():
            raise RuntimeError('in_progress URLs present; finish graceful stop first')


def backup(path):
    path=guard(path)
    if path.exists(): raise ValueError('Backup destination exists; never overwrite')
    stopped_wal0()
    shutil.copyfile(DB,path)
    with DB.open('rb') as f: source_sha=hashlib.file_digest(f,'sha256').hexdigest()
    with path.open('rb') as f: backup_sha=hashlib.file_digest(f,'sha256').hexdigest()
    if source_sha!=backup_sha: raise ValueError('Backup byte SHA256 mismatch')
    return backup_sha


def load_plan():
    import pandas as pd
    from r2ai.zh_full.crawl import DOMAINS
    manifest=json.loads((RUN/'plan.json').read_text('utf-8'))
    if not manifest.get('complete'): raise ValueError('Selection manifest is not complete')
    if not set(manifest['domains'])<=DOMAINS: raise ValueError('Unknown domain in plan')
    rows=[]
    for domain in manifest['domains']:
        path=RUN/f'order-{domain}.parquet'
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        meta=manifest['order_files'][domain]
        if sha!=meta['sha256']: raise ValueError(f'Order SHA mismatch: {domain}')
        df=pd.read_parquet(path)
        if (len(df)!=meta['rows'] or set(df.domain)!={domain} or set(df.cut_version)!={manifest['cut_version']}
                or int((df.new_status!='deferred_select').sum())!=manifest['budgets'][domain]['cut']):
            raise ValueError(f'Order version/count/budget mismatch: {domain}')
        df=df.rename(columns={'rank':'old_rank','status':'old_status'})
        rows.extend(df.to_dict('records'))
    return manifest,rows


def verify(conn, original, manifest, plan_rows):
    columns=[r[1] for r in conn.execute('PRAGMA table_info(urls)')]
    bad=Counter(); changed=0
    # Sorted merge compares every column of every URL, including terminal rows.
    for a,b in zip(original.execute('SELECT * FROM urls ORDER BY url_norm'),conn.execute('SELECT * FROM urls ORDER BY url_norm'),strict=True):
        if a[0]!=b[0]: raise ValueError('URL identity/count changed')
        eligible=a[columns.index('status')] in ('pending','network_error')
        for c,x,y in zip(columns,a,b):
            if x!=y:
                if not eligible or c not in ('rank','status'): bad[c]+=1
        changed+=a!=b
    if bad: raise ValueError(f'Forbidden column diff: {bad}')
    before=list(original.execute('SELECT domain,status,count(*) FROM urls GROUP BY domain,status'))
    after=list(conn.execute('SELECT domain,status,count(*) FROM urls GROUP BY domain,status'))
    dry={}; conservation={}
    for domain in manifest['domains']:
        conservation[domain]={}
        for status in ('pending','network_error'):
            n0=original.execute('SELECT count(*) FROM urls WHERE domain=? AND status=?',(domain,status)).fetchone()[0]
            n1=conn.execute('SELECT count(*) FROM urls WHERE domain=? AND status=?',(domain,status)).fetchone()[0]
            deferred=conn.execute("SELECT count(*) FROM urls u JOIN select_meta m USING(url_norm) WHERE u.domain=? AND u.status='deferred_select' AND m.old_status=?",(domain,status)).fetchone()[0]
            if n0!=n1+deferred: raise ValueError('Status conservation failed')
            conservation[domain][status]={'before':n0,'after':n1,'deferred':deferred}
        # now=+infinity tests the complete eligible order without resetting retry clocks.
        rows=next_batch(SimpleNamespace(conn=conn),domain,1000,1e30,dry_run=True)
        # Expected is the frozen scored parquet order, not another queue SQL query.
        expected=[r['url_norm'] for r in plan_rows if r['domain']==domain and r['new_status']!='deferred_select'][:1000]
        if [r['url_norm'] for r in rows]!=expected: raise ValueError('dry-run plan order differs')
        dry[domain]={'urls':len(rows),'correct':True,'no_fetch':True}
    return {'before':before,'after':after,'forbidden_column_diffs':dict(bad),'changed_rows':changed,'dry_run':dry,'conservation':conservation}


def apply_and_gate():
    preflight()
    with exclusive('crawl'):
        manifest,rows=load_plan()
        version=manifest['cut_version']
        backup_path=guard(DB.parent/f'crawl-pre-{version}.db')
        sha=backup(backup_path)
        conn=writer(DB)
        original=ro(backup_path,immutable=True)
        try:
            original_digest=urls_digest(original)
            count=apply_rows(conn,rows,version)
            gate=verify(conn,original,manifest,rows)
            # Real rollback on an independent SQLite backup under the allowed run.
            copy=guard(RUN/f'rollback-{version}.db')
            with writer(copy) as test:
                conn.backup(test)
                test.execute("UPDATE urls SET status=(SELECT old_status FROM select_meta m WHERE m.url_norm=urls.url_norm) WHERE status='deferred_select'")
                dry_uncut={}
                for domain in manifest['domains']:
                    got=next_batch(SimpleNamespace(conn=test),domain,1000,1e30,dry_run=True)
                    expected=[r['url_norm'] for r in rows if r['domain']==domain][:1000]
                    if [r['url_norm'] for r in got]!=expected:
                        raise ValueError(f'Uncut scored order differs: {domain}')
                    dry_uncut[domain]={'urls':len(got),'correct':True,'no_fetch':True,'copy_only':True}
                rollback(test,version)
                restored=urls_digest(test)
            if restored!=original_digest: raise ValueError('Rollback urls SHA mismatch')
            gate.update({'backup':str(backup_path),'backup_sha256':sha,'applied_rows':count,
                         'original_urls_sha256':original_digest,'rollback_urls_sha256':restored,'rollback_pass':True,
                         'dry_run_all_scored_on_copy':dry_uncut,
                         'cut_version':version,'verified_at':time.time()})
            atomic_json(RUN/'gates.json',gate)
            print(json.dumps({k:v for k,v in gate.items() if k not in ('before','after')},ensure_ascii=False))
            return gate
        except BaseException:
            # No fetch can happen under crawl lock. Restore rank/status if a gate fails.
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='select_meta'").fetchone() and conn.execute('SELECT 1 FROM select_meta LIMIT 1').fetchone():
                rollback(conn,version)
            raise
        finally:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            conn.close(); original.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='step',required=True)
    sub.add_parser('apply')
    r=sub.add_parser('reopen'); r.add_argument('--domain',required=True); r.add_argument('--top',type=int,required=True)
    r=sub.add_parser('rollback'); r.add_argument('--cut-version',required=True)
    args=parser.parse_args(argv)
    if args.step=='apply': apply_and_gate(); return 0
    preflight()
    with exclusive('crawl'):
        stopped_wal0()
        with writer(DB) as conn:
            result=reopen(conn,args.domain,args.top) if args.step=='reopen' else rollback(conn,args.cut_version)
    print(json.dumps({'step':args.step,'changed_rows':result}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
