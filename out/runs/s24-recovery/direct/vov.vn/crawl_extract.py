from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'src'))
from r2ai.paths import ROOT, assert_writable, CONFIG_DIR
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer
from vicrawl.extract_pipeline import SCHEMA, MIN_OK_CHARS, extract_records
from vicrawl.settings import Config
from vicrawl.shards import list_shards, read_shard
import zstandard
from vicrawl.urlnorm import normalize_url

DIRECT = ROOT / 'out/runs/s24-recovery/direct/vov.vn'
S15 = ROOT / 'out/runs/s24-recovery/s15'
RAW_ROOT = S15 / 'raw'
RAW_DOMAIN = RAW_ROOT / 'vov.vn'
DOCS_DIR = S15 / 'docs/s24-direct-vov'
BATCH_DIR = S15 / 'run/batches/s24-direct-vov'
PROV_DIR = S15 / 'run/provenance/s24-direct-vov'
TARGETS = DIRECT / 'targets.parquet'
REQUEST_LOG = DIRECT / 'requests.jsonl'
STOP_FILE = DIRECT / 'stop.json'
SAMPLE_REPORT = DIRECT / 'sample-report.json'
UA = 'r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)'
DOMAIN = 'vov.vn'
MIN_START_GAP = 1.2


def checked(path: Path) -> Path:
    return assert_writable(path)


def atomic_json(path: Path, value: dict) -> None:
    p = checked(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = checked(p.with_name(p.name + '.tmp'))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def append_jsonl(path: Path, value: dict) -> None:
    p = checked(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
        f.flush(); os.fsync(f.fileno())


def read_targets() -> list[dict]:
    df = pq.read_table(TARGETS).to_pandas().sort_values('id')
    targets = [{'id': int(r.id), 'url': str(r.url)} for r in df.itertuples()]
    for t in targets:
        h = (urlsplit(t['url']).hostname or '').lower()
        if h != 'vov.vn':
            raise ValueError(f'Out-of-scope host in target file: {h} {t["url"]}')
    return targets


def persisted_records() -> tuple[set[int], list[dict]]:
    done, records = set(), []
    for p in list_shards(RAW_ROOT, DOMAIN):
        for r in read_shard(p):
            records.append((p, r))
            done.update(map(int, r.get('doc_ids') or []))
    return done, records


def last_request_epoch() -> float | None:
    if not REQUEST_LOG.exists(): return None
    last = None
    with REQUEST_LOG.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): last = json.loads(line)
    return float(last['request_start_epoch']) if last and last.get('request_start_epoch') else None


def captcha_page(body: bytes) -> bool:
    head = body[:200_000].decode('utf-8', errors='ignore').lower()
    title_m = re.search(r'<title[^>]*>(.*?)</title>', head, re.I | re.S)
    title = re.sub(r'\s+', ' ', title_m.group(1)).strip() if title_m else ''
    hard = ('/cdn-cgi/challenge-platform/', 'cf-chl-', 'g-recaptcha', 'h-captcha',
            'verify you are human', 'captcha-container', 'bot verification')
    return any(s in head for s in hard) or any(s in title for s in ('just a moment', 'attention required', 'captcha'))


def local_tokenizer():
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=True)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def extract_rows(records: list[dict]) -> list[dict]:
    cfg = Config.load(CONFIG_DIR / 'domains.yaml')
    extractor = cfg.for_domain(DOMAIN).extractor
    return extract_records(records, {DOMAIN: extractor}, tokenizer=local_tokenizer())


def write_docs(rows: list[dict], path: Path) -> None:
    p = checked(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = checked(p.with_name(p.name + '.tmp'))
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression='zstd')
    os.replace(tmp, p)


def write_raw_record(record: dict) -> Path:
    domain_dir=RAW_DOMAIN
    ident=int(record['doc_ids'][0]); name=f'vov-{ident:08d}.jsonl.zst'
    dst=checked(domain_dir/name); tmp=checked(domain_dir/(name+'.tmp'))
    if dst.exists(): raise FileExistsError(dst)
    payload=(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
    frame=zstandard.ZstdCompressor(level=6).compress(payload)
    with tmp.open('wb') as f: f.write(frame); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,dst)
    return dst


def log_event(**kw):
    append_jsonl(REQUEST_LOG, kw)


def do_request(session: requests.Session, target: dict, last_start: float | None):
    if (urlsplit(target['url']).hostname or '').lower() != 'vov.vn':
        raise ValueError('Only vov.vn is in scope')
    if last_start is not None:
        delay = MIN_START_GAP - (time.time() - last_start)
        if delay > 0: time.sleep(delay)
    started = time.time(); iso = datetime.fromtimestamp(started, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    rec = {'doc_ids':[target['id']], 'url':target['url'], 'final_url':target['url'], 'domain':DOMAIN,
           'url_norm':normalize_url(target['url']), 'fetched_at':started, 'requested_at_utc':iso}
    try:
        with session.get(target['url'], timeout=(15, 45), allow_redirects=False, stream=False) as r:
            body = r.content; status = int(r.status_code); ct = r.headers.get('Content-Type','')
            base = {'id':target['id'], 'url':target['url'], 'request_start_epoch':started,
                    'requested_at_utc':iso, 'http_status':status, 'response_bytes':len(body),
                    'response_sha256':hashlib.sha256(body).hexdigest(), 'content_type':ct,
                    'elapsed_s':round(time.time()-started,3), 'user_agent':UA,
                    'proxy_disabled':True, 'redirects_followed':False}
            if status in (403,429):
                base['stop_reason']=f'HTTP {status}'; log_event(**base)
                return None, started, f'HTTP {status}'
            if status == 200 and captcha_page(body):
                base['stop_reason']='CAPTCHA/challenge HTML'; log_event(**base)
                return None, started, 'CAPTCHA/challenge HTML'
            if status >= 500:
                base['stop_reason']=f'HTTP {status} server error'; log_event(**base)
                return None, started, base['stop_reason']
            if status == 200:
                rec.update({'status':'ok','html':r.text,'content_type':ct,
                            'response_bytes':len(body),'response_sha256':base['response_sha256'],
                            'requested_at_utc':iso,'http_status':status})
                shard = write_raw_record(rec)
                base['raw_shard']=str(shard.relative_to(ROOT))
                base['raw_shard_sha256']=hashlib.sha256(shard.read_bytes()).hexdigest()
            else:
                base['stop_reason']=None
            log_event(**base)
            return (rec if status==200 else None), started, None
    except requests.RequestException as e:
        base={'id':target['id'],'url':target['url'],'request_start_epoch':started,
              'requested_at_utc':iso,'error':f'{type(e).__name__}: {e}',
              'elapsed_s':round(time.time()-started,3),'user_agent':UA,'proxy_disabled':True,
              'redirects_followed':False,'stop_reason':'network exception; no retry'}
        log_event(**base)
        return None, started, base['stop_reason']


def sample() -> None:
    if SAMPLE_REPORT.exists(): raise SystemExit('sample report already exists; refusing duplicate sample requests')
    targets=read_targets()[:5]
    if len(targets)!=5: raise ValueError('fewer than five corpus targets')
    sess=requests.Session(); sess.trust_env=False; sess.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5','Accept-Language':'vi-VN,vi;q=0.9,en;q=0.5'})
    last=last_request_epoch(); fetched=[]; stop=None
    for i,t in enumerate(targets,1):
        r,last,stop=do_request(sess,t,last)
        if r: fetched.append(r)
        print(f'sample {len(fetched)+sum(1 for x in targets if False)}/{len(targets)} id={t["id"]} status={r.get("http_status") if r else "no-text-record"} stop={stop}',flush=True)
        if stop:
            atomic_json(STOP_FILE,{'reason':stop,'at_utc':datetime.now(timezone.utc).isoformat(),'id':t['id']})
            break
    sess.close()
    rows=extract_rows(fetched) if fetched else []
    docs_path=DIRECT/'sample-docs.parquet'
    write_docs(rows,docs_path)
    result_by_id={int(r['doc_ids'][0]):r for r in rows}
    statuses=[{'id':t['id'],'url':t['url'],'extracted_status':result_by_id.get(t['id'],{}).get('status','no-text'),'n_chars':len(result_by_id.get(t['id'],{}).get('body',''))} for t in targets]
    good=sum(1 for x in statuses if x['extracted_status']=='ok' and x['n_chars']>=MIN_OK_CHARS)
    summary={'phase':'sample','sample_target_urls':5,'sample_attempts':len([1 for x in targets if x['id'] in {int(r['id']) for r in _read_log()[-5:]}]),
             'sample_good_text_ge_200_chars':good,'all_five_good':good==5 and len(rows)==5,
             'results':statuses,'extractor':Config.load(CONFIG_DIR/'domains.yaml').for_domain(DOMAIN).extractor,
             'docs_path':str(docs_path.relative_to(ROOT)),'raw_root':str((RAW_ROOT/DOMAIN).relative_to(ROOT)),
             'no_chunking':True,'no_embedding':True,'shared_index_written':False,'stop_reason':stop}
    atomic_json(SAMPLE_REPORT,summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def _read_log() -> list[dict]:
    if not REQUEST_LOG.exists(): return []
    out=[]
    with REQUEST_LOG.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def extract_all() -> dict:
    done, packed = persisted_records()
    cfg=Config.load(CONFIG_DIR/'domains.yaml'); extractor=cfg.for_domain(DOMAIN).extractor
    tok=local_tokenizer()
    docs_path=DOCS_DIR/'vov.vn.parquet'; prov_path=PROV_DIR/'vov.vn-direct.parquet'
    docs_path=checked(docs_path); prov_path=checked(prov_path)
    docs_path.parent.mkdir(parents=True,exist_ok=True); prov_path.parent.mkdir(parents=True,exist_ok=True)
    docs_tmp=checked(docs_path.with_name(docs_path.name+'.tmp')); prov_tmp=checked(prov_path.with_name(prov_path.name+'.tmp'))
    for tmp in (docs_tmp,prov_tmp):
        if tmp.exists(): checked(tmp).unlink()
    prov_schema=pa.schema([('doc_id',pa.int64()),('url',pa.string()),('source',pa.string()),('host',pa.string()),
      ('requested_at_utc',pa.string()),('http_status',pa.int16()),('response_bytes',pa.int64()),('raw_shard',pa.string()),
      ('response_sha256',pa.string()),('raw_shard_sha256',pa.string()),('extractor',pa.string()),('status',pa.string()),
      ('n_chars',pa.int32()),('text_sha1',pa.string())])
    doc_writer=pq.ParquetWriter(docs_tmp,SCHEMA,compression='zstd')
    prov_rows=[]; docs_buf=[]; docs_n=good_n=thin_n=0
    try:
        for idx,(p,r) in enumerate(packed,1):
            rows=extract_records([r],{DOMAIN:extractor},tokenizer=tok)
            if not rows: continue
            docs_buf.extend(rows)
            d=rows[0]; docs_n+=1
            good_n+=int(d['status']=='ok' and len(d['body'])>=MIN_OK_CHARS); thin_n+=int(d['status']!='ok')
            prov_rows.append({'doc_id':int(d['doc_ids'][0]),'url':d['url'],'source':'direct','host':DOMAIN,
              'requested_at_utc':r.get('requested_at_utc',''),'http_status':int(r.get('http_status',200)),
              'response_bytes':int(r.get('response_bytes',0)),'raw_shard':str(p.relative_to(ROOT)),
              'response_sha256':r.get('response_sha256',''),'raw_shard_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
              'extractor':d['extractor'],'status':d['status'],'n_chars':len(d['body']),'text_sha1':d['text_sha1']})
            if len(docs_buf)>=500:
                doc_writer.write_table(pa.Table.from_pylist(docs_buf,schema=SCHEMA)); docs_buf.clear()
            if idx%500==0: print(f'extract progress raw={idx}/{len(packed)} docs={docs_n}',flush=True)
        if docs_buf: doc_writer.write_table(pa.Table.from_pylist(docs_buf,schema=SCHEMA))
        if docs_n==0:
            doc_writer.write_table(pa.Table.from_pylist([],schema=SCHEMA))
    finally:
        doc_writer.close()
    pq.write_table(pa.Table.from_pylist(prov_rows,schema=prov_schema),prov_tmp,compression='zstd')
    os.replace(docs_tmp,docs_path); os.replace(prov_tmp,prov_path)
    return {'extracted_docs':docs_n,'good_text_ge_200_chars':good_n,'thin_or_short_docs':thin_n,
            'docs_path':str(docs_path.relative_to(ROOT)),'provenance_path':str(prov_path.relative_to(ROOT)),
            'raw_records':len(packed),'extractor':extractor,'tokenizer':'BAAI/bge-m3 local_files_only'}


def finalize(crawl_status:dict, extraction:dict) -> None:
    logs=_read_log(); targets=read_targets(); raw_done,packed=persisted_records()
    statuses={}; gaps=[]; starts=[]; total_bytes=0
    for r in logs:
        status=str(r.get('http_status','error')); statuses[status]=statuses.get(status,0)+1
        if r.get('response_bytes'): total_bytes+=int(r['response_bytes'])
        if r.get('request_start_epoch'): starts.append(float(r['request_start_epoch']))
    if len(starts)>1: gaps=[b-a for a,b in zip(starts,starts[1:])]
    sample=json.loads(SAMPLE_REPORT.read_text(encoding='utf-8')) if SAMPLE_REPORT.exists() else {}
    manifest={'batch':'s24-direct-vov','domain':DOMAIN,'source':'direct','robots_txt_bypassed':True,'ua':UA,
      'proxy_disabled':True,'redirects_followed':False,'target_urls':len(targets),'unique_http200_urls':len(raw_done),
      'sample_urls':5,'sample_good_text':sample.get('sample_good_text_ge_200_chars',0),'attempts':len(logs),
      'attempt_status_counts':statuses,'response_bytes_all_attempts':total_bytes,'extracted_docs':extraction.get('extracted_docs',0),
      'good_text_urls_ge_200_chars':extraction.get('good_text_ge_200_chars',0),'raw_shards':len(list_shards(RAW_ROOT,DOMAIN)),
      'raw_root':str((RAW_ROOT/DOMAIN).relative_to(ROOT)),'docs_path':extraction.get('docs_path'),
      'provenance_path':extraction.get('provenance_path'),'extractor':extraction.get('extractor'),
      'schema':'vicrawl.extract_pipeline.SCHEMA','chunking_performed':False,'embedding_performed':False,
      'shared_index_written':False,'stop_reason':crawl_status.get('stop_reason'),
      'attempted_unique_urls':len({int(r['id']) for r in logs if 'id' in r}),
      'completed_all_targets':len({int(r['id']) for r in logs if 'id' in r})==len(targets) and not crawl_status.get('stop_reason'),
      'request_gap_min_s':round(min(gaps),3) if gaps else None,
      'crawl_span_s':round(max(starts)-min(starts),3) if starts else 0,
      'target_vov2_excluded':True,'wayback_used':False,'thaythuocvietnam_crawled':False}
    atomic_json(BATCH_DIR/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def full(extract_only=False) -> None:
    if not extract_only:
        if STOP_FILE.exists(): raise SystemExit('stop marker exists; refusing any further vov.vn request')
        if not SAMPLE_REPORT.exists() or not json.loads(SAMPLE_REPORT.read_text(encoding='utf-8')).get('all_five_good'):
            raise SystemExit('sample gate not passed; refusing full crawl')
        targets=read_targets(); done,_=persisted_records(); sess=requests.Session(); sess.trust_env=False
        sess.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5','Accept-Language':'vi-VN,vi;q=0.9,en;q=0.5'})
        last=last_request_epoch(); start=time.time(); stop=None; attempted=0
        try:
            for t in targets:
                if t['id'] in done: continue
                rec,last,stop=do_request(sess,t,last); attempted+=1
                if rec: done.update(rec['doc_ids'])
                if stop:
                    atomic_json(STOP_FILE,{'reason':stop,'at_utc':datetime.now(timezone.utc).isoformat(),'id':t['id']})
                    break
                if attempted%25==0: print(f'progress attempted={attempted} persisted={len(done)}/{len(targets)}',flush=True)
        finally: sess.close()
        crawl_status={'stop_reason':stop,'attempted_this_run':attempted,'elapsed_s':round(time.time()-start,3)}
    else:
        crawl_status={'stop_reason':json.loads(STOP_FILE.read_text(encoding='utf-8')).get('reason') if STOP_FILE.exists() else None,'attempted_this_run':0,'extract_only':True}
    extraction=extract_all(); finalize(crawl_status,extraction)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['sample','full','extract-only'],required=True); a=ap.parse_args()
    dirs=[DIRECT,RAW_ROOT,RAW_DOMAIN,DOCS_DIR,BATCH_DIR,PROV_DIR]
    for p in dirs: checked(p)
    for p in dirs: p.mkdir(parents=True,exist_ok=True)
    if a.phase=='sample': sample()
    else: full(extract_only=a.phase=='extract-only')

if __name__=='__main__': main()
