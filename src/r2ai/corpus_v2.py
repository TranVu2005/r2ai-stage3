"""Corpus v2 (links_corpus_vi_en_zh.parquet, replaces links_corpus.parquet on 12/10/2026).

The BTC file has only (id, url). `lang` is derived from the URL host with the explicit tables below; an unknown
host raises instead of being guessed. A doc_id outside v2 can never be a positive, so submissions filter on
`is_member` / `member_mask` (decision D7).

    python -m r2ai.corpus_v2 build [--raw PATH] [--out PATH]
    python -m r2ai.corpus_v2 check        # run before every submission: exit 0 ok, 1 mismatch, 2 cannot verify
"""
from __future__ import annotations

from r2ai.paths import RAW_DATA_DIR, RAW_DOWNLOAD_DIR, WORK_DATA_DIR, assert_writable

import argparse
import hashlib
import os
import re
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pyarrow.parquet as pq

HF_REPO = 'AIGuruTinix/ViBioMIR'
HF_REVISION = '71be0e1239a19dfd7a717e2bc13e3bec21bd0334'
RAW_NAME = 'links_corpus_vi_en_zh.parquet'
RAW_SHA256 = '87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7'
RAW_SIZE = 37689434
RAW_PATH = RAW_DOWNLOAD_DIR / RAW_NAME
QUERY_SHA256 = 'ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08'   # query.parquet is unchanged since v1
QUERY_PATH = RAW_DATA_DIR / 'query.parquet'
HF_API = f'https://huggingface.co/api/datasets/{HF_REPO}?blobs=true'
IDS_PATH = WORK_DATA_DIR / 'corpus_v2_ids.parquet'

EN_DOMAINS = ('ncbi.nlm.nih.gov',)
ZH_DOMAINS = (
    '39.net', '120ask.com', 'cnkang.com', 'familydoctor.com.cn', 'familydoctor.cn', 'zysjonline.com',
    'zhongyibaodian.net', 'a-hospital.com', 'youlai.cn', 'pmphai.com', 'msdmanuals.cn', 'iiyi.com', 'wujue.com',
    'jb39.com', 'zydcd.com', 'people.com.cn', 'baidu.com', 'qihuangzhishu.com',
)
# vi hosts outside the .vn TLD
VI_DOMAINS = ('vinmec.com', 'hellobacsi.com', 'vnexpress.net', 'benhvienvietduc.org', 'pharmacity.io')

_HOST_RE = re.compile(r'^[a-z][a-z0-9+.-]*://([^/:?#]+)', re.I)


def host_of(url: str) -> str:
    m = _HOST_RE.match(url)
    if not m:
        raise ValueError(f'lang: no host in url {url!r}')
    host = m.group(1).lower()
    return host[4:] if host.startswith('www.') else host


def _under(host: str, domains) -> bool:
    return any(host == d or host.endswith('.' + d) for d in domains)


def lang_of_host(host: str) -> str:
    if _under(host, EN_DOMAINS):
        return 'en'
    if _under(host, ZH_DOMAINS):
        return 'zh'
    if host.endswith('.vn') or _under(host, VI_DOMAINS):
        return 'vi'
    raise ValueError(f'lang: unknown host {host!r}; add it to EN/ZH/VI tables in corpus_v2.py')


def build(raw_path=None, out_path=None) -> pl.DataFrame:
    """Read the BTC parquet, check id/url, add lang/host, write (id, url, lang, host) sorted by id."""
    raw = pl.read_parquet(raw_path or RAW_PATH)
    if raw.columns != ['id', 'url'] or raw.schema['id'] != pl.Int64 or raw.schema['url'] != pl.String:
        raise ValueError(f'schema: expected id:int64,url:string, got {dict(raw.schema)}')
    if raw.null_count().row(0) != (0, 0):
        raise ValueError('schema: null id or url')
    if raw['id'].n_unique() != raw.height:
        raise ValueError('id not unique')
    if raw['url'].n_unique() != raw.height:
        raise ValueError('url not unique')
    hosts = [host_of(u) for u in raw['url']]
    cache: dict[str, str] = {}
    for h in set(hosts):
        cache[h] = lang_of_host(h)
    df = raw.with_columns(pl.Series('lang', [cache[h] for h in hosts]), pl.Series('host', hosts)).sort('id')
    out = assert_writable(out_path or IDS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = assert_writable(out.with_name(out.name + '.tmp'))
    df.write_parquet(tmp, compression='zstd')
    os.replace(tmp, out)
    return df


_cache: dict[str, tuple[tuple[int, int], np.ndarray]] = {}


def load_ids(path=None) -> np.ndarray:
    """Sorted int64 array of v2 ids; reloaded when the file's mtime/size changes."""
    p = Path(path or IDS_PATH)
    st = p.stat()
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(str(p))
    if hit is None or hit[0] != key:
        ids = np.sort(pl.read_parquet(p, columns=['id'])['id'].to_numpy())
        hit = _cache[str(p)] = (key, ids)
    return hit[1]


def member_mask(ids, path=None) -> np.ndarray:
    arr = np.asarray(list(ids), dtype=np.int64)
    known = load_ids(path)
    pos = np.searchsorted(known, arr)
    pos[pos == known.size] = 0
    return (known[pos] == arr) if known.size else np.zeros(arr.shape, dtype=bool)


def host_ids(hosts, path=None) -> np.ndarray:
    """Sorted ids of every v2 URL whose normalised host (lower case, no www.) is in `hosts`."""
    wanted = {h.lower().removeprefix('www.') for h in hosts}
    df = pl.read_parquet(Path(path or IDS_PATH), columns=['id', 'host']).filter(pl.col('host').is_in(list(wanted)))
    return np.sort(df['id'].to_numpy())


def is_member(doc_id, path=None) -> bool:
    if not isinstance(doc_id, int) or isinstance(doc_id, bool):
        return False
    known = load_ids(path)
    i = int(np.searchsorted(known, doc_id))
    return i < known.size and int(known[i]) == doc_id


def fetch_hf_info() -> dict:
    r = httpx.get(HF_API, timeout=30, follow_redirects=True)   # plain HTTP: independent of HF_HUB_OFFLINE
    r.raise_for_status()
    return r.json()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def check_revision(api: dict, raw_path=None, ids_path=None, query_path=None) -> tuple[bool, list[str]]:
    """Compare HF state with the pinned revision and the local copies. Never raises for a mismatch."""
    lines: list[str] = []

    def rec(ok: bool, msg: str) -> bool:
        lines.append(('PASS ' if ok else 'FAIL ') + msg)
        return ok

    files = {s.get('rfilename'): s for s in api.get('siblings', [])}
    allok = rec(api.get('sha') == HF_REVISION, f'HF revision {api.get("sha")} (pinned {HF_REVISION})')
    corpus = files.get(RAW_NAME, {})
    lfs = corpus.get('lfs') or {}
    allok &= rec(lfs.get('sha256') == RAW_SHA256 and corpus.get('size') == RAW_SIZE,
                 f'HF file sha256 {lfs.get("sha256")} size {corpus.get("size")} (pinned {RAW_SHA256} {RAW_SIZE})')
    q = (files.get('query.parquet') or {}).get('lfs') or {}
    allok &= rec(q.get('sha256') == QUERY_SHA256, f'HF query.parquet sha256 {q.get("sha256")} (pinned {QUERY_SHA256})')
    raw = Path(raw_path or RAW_PATH)
    if not raw.is_file():
        return rec(False, f'local raw missing: {raw}') and allok, lines
    allok &= rec(_sha256(raw) == RAW_SHA256, f'local raw sha256 of {raw}')
    ids = Path(ids_path or IDS_PATH)
    if not ids.is_file():
        return rec(False, f'ids parquet missing: {ids}') and allok, lines
    try:
        same = (pq.ParquetFile(ids).metadata.num_rows == pq.ParquetFile(raw).metadata.num_rows
                and np.array_equal(load_ids(ids), np.sort(pl.read_parquet(raw, columns=['id'])['id'].to_numpy())))
    except Exception:   # unreadable local file counts as a mismatch, not a crash
        same = False
    allok &= rec(same, f'ids parquet {ids} has exactly the ids of the local raw')
    if query_path is not None:
        qp = Path(query_path)
        if qp.is_file():
            allok &= rec(_sha256(qp) == QUERY_SHA256, f'local query.parquet sha256 of {qp}')
        else:
            lines.append(f'SKIP local query.parquet not found: {qp}')
    return bool(allok), lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--raw', default=None)
    b.add_argument('--out', default=None)
    sub.add_parser('check')
    args = ap.parse_args(argv)
    if args.cmd == 'check':
        try:
            api = fetch_hf_info()
        except Exception as e:   # fail closed: an unverifiable corpus must not be submitted
            print(f'FAIL cannot reach HF to verify revision: {e!r}')
            return 2
        ok, lines = check_revision(api, RAW_PATH, IDS_PATH, QUERY_PATH)
        print(chr(10).join(lines))
        print('corpus v2 OK, safe to submit' if ok else 'corpus v2 MISMATCH, do not submit')
        return 0 if ok else 1
    df = build(args.raw, args.out)
    print(f'{df.height} ids -> {args.out or IDS_PATH}')
    print(df['lang'].value_counts().sort('lang'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
