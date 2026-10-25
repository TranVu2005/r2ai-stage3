"""S25 phase 2 step 3: choose exactly one text version per corpus-v2 doc_id (vi, zh; en counted only, D17).

Rules, in order:
  (a) the version P2 serves: live rows of the P2 index view (vi: vi_s18_segments manifest = base-t256 + seg00400..406
      minus superseded, plus the cc facade seg20000..20004 of the P1 snapshot; zh: zh_v2_segments manifest = 73
      segments incl. S22 seg30001..30009 and webMedQA seg20006, minus 11,064 superseded edges of which 9,496 are S22);
  (b) otherwise the newest extraction that passes the threshold (vi: body >= 200 code points; zh: >= 200 code points
      or >= 50 Han characters). "Newest" = birth of the extraction run: the earliest mtime of the doc-level files in
      the source directory that first holds the exact text (one run = one directory = one extractor generation);
  (c) ties on (b) go to the longer body; a doc_id with no passing version keeps its longest body as quality=thin.
Every doc_id with more than one distinct text gets a decision row (rule, reason, alternatives).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import polars as pl

DATA = Path('D:/GitHub/r2ai-stage3/data')
OUT = Path('D:/GitHub/r2ai-stage3/out')
P1_FACADE = OUT / 'runs/reranker-trial/vi-snapshots/564eb3511ebc1c06-s23-cumulative/facade-map.json'
VI_MANIFEST = DATA / 'index/vi_s18_segments/manifest.json'
ZH_MANIFEST = DATA / 'index/zh_v2_segments/manifest.json'
S22_SUPERSEDED = OUT / 'runs/reranker-trial/zh/delta-s22-webmedqa-20261018/superseded-s22.json'
INDEX_SEARCH_ROOTS = (DATA / 'index', OUT / 'runs/zh-full/segments')
VI_MIN_CP, ZH_MIN_CP, ZH_MIN_HAN = 200, 200, 50


@dataclass
class LiveSegment:
    name: str
    lang: str
    docs: Path
    chunks: Path
    index: Path | None
    global_start: int
    superseded: set = field(default_factory=set)


def norm(p) -> str:
    return str(Path(p)).replace('/', '\\').lower()


TEST_PART = re.compile(r'^(pytest.*|test_.*\d|tests|fixtures|test-temp|synthetic.*)$', re.I)


def is_test_path(p: str) -> bool:
    """Test fixtures/pytest scratch hold synthetic text; they are never a version of a real document."""
    return any(TEST_PART.match(part) for part in PureWindowsPath(p).parts)


def live_segments() -> list[LiveSegment]:
    vi = json.loads(VI_MANIFEST.read_text(encoding='utf-8'))
    zh = json.loads(ZH_MANIFEST.read_text(encoding='utf-8'))
    facade = json.loads(P1_FACADE.read_text(encoding='utf-8'))
    sup: dict[str, set] = {}
    for e in vi['superseded'] + zh['superseded']:
        sup.setdefault(e['segment'], set()).add(int(e['doc_id']))
    out = [LiveSegment('base-t256', 'vi', DATA / 'chunks/docs.parquet', DATA / 'chunks/chunks_t256.parquet',
                       DATA / 'index/t256', 0, sup.get('base-t256', set()))]
    for s in vi['segments'][1:]:
        if s['source'] == 'vi_s18_reingest':
            seg = DATA / 'chunks_vi_s18' / s['name']
        else:
            seg = (DATA / s['source']).parent
        out.append(LiveSegment(s['name'], 'vi', seg / 'docs.parquet', seg / 'chunks_t256.parquet',
                               DATA / 'index/vi_s18_segments' / s['name'], int(s['global_start']), sup.get(s['name'], set())))
    for s in facade['segments']:
        if s['category'] != 'cc':
            continue
        seg = DATA / 'chunks_cc_longchau' / s['name']
        out.append(LiveSegment(s['name'], 'vi', seg / 'docs.parquet', seg / 'chunks_t256.parquet',
                               DATA / 'index/zh_full_segments/cc_longchau' / s['name'], int(s['global_start']),
                               sup.get(s['name'], set())))
    for s in zh['segments']:
        seg = DATA / 'chunks_zh_v2' / s['name']
        if int(s['docs_v2']) == 0:
            continue                                       # seg00011 qihuangzhishu: 0 v2 rows, no docs file
        out.append(LiveSegment(s['name'], 'zh', seg / 'docs.parquet', seg / 'chunks_t256.parquet',
                               DATA / 'index/zh_v2_segments' / s['name'], int(s['global_start']), sup.get(s['name'], set())))
    return out


def index_dirs_by_name() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for root in INDEX_SEARCH_ROOTS:
        for meta in root.rglob('segment.json'):
            found.setdefault(meta.parent.name, []).append(meta.parent)
    return found


def threshold_expr() -> pl.Expr:
    return (pl.when(pl.col('lang') == 'vi').then(pl.col('body_cp') >= VI_MIN_CP)
            .otherwise((pl.col('body_cp') >= ZH_MIN_CP) | (pl.col('han') >= ZH_MIN_HAN)))


def choose(versions: pl.DataFrame, sources: pl.DataFrame, corpus: pl.DataFrame, live: list[LiveSegment]):
    """-> (chosen per doc_id, decisions for doc_ids with >1 text, per-version table)."""
    test_src = sources.filter(pl.col('path').map_elements(is_test_path, return_dtype=pl.Boolean))['src']
    sources = sources.filter(~pl.col('src').is_in(test_src.implode()) & ~pl.col('alias_of').is_in(test_src.implode()))
    versions = versions.filter(~pl.col('src').is_in(test_src.implode()))
    src = sources.with_columns(pl.col('path').map_elements(norm, return_dtype=pl.Utf8).alias('npath'),
                               pl.when(pl.col('alias_of') >= 0).then(pl.col('alias_of')).otherwise(pl.col('src')).alias('real'))
    # Extraction-run birth: earliest mtime of doc-level files in the same directory.
    src = src.with_columns(pl.col('npath').map_elements(lambda p: p.rsplit('\\', 1)[0], return_dtype=pl.Utf8).alias('dir'))
    src = src.with_columns(pl.col('mtime').min().over('dir').alias('dir_birth'))
    live_rows = []
    for seg in live:
        hit = src.filter(pl.col('npath') == norm(seg.docs))
        if hit.height != 1:
            raise ValueError(f'live docs file not catalogued exactly once: {seg.docs} ({hit.height})')
        live_rows.append({'real': int(hit['real'][0]), 'segment': seg.name,
                          'superseded': sorted(seg.superseded)})
    live_df = pl.DataFrame(live_rows).explode('superseded')
    live_src = pl.DataFrame({'real': [r['real'] for r in live_rows], 'segment': [r['segment'] for r in live_rows]})
    sup = live_df.drop_nulls('superseded').select(pl.col('real'), pl.col('superseded').alias('phys_id'), pl.lit(True).alias('sup'))

    v = (versions.join(src.select('src', 'real', 'kind', 'owner', 'npath', 'mtime', 'dir_birth'), on='src', how='left')
         .join(corpus.select(pl.col('id').alias('doc_id'), 'lang', 'host'), on='doc_id', how='left')
         .filter(pl.col('lang') != 'en'))
    v = (v.join(live_src, on='real', how='left')
         .join(sup, on=['real', 'phys_id'], how='left')
         .with_columns((pl.col('segment').is_not_null() & pl.col('sup').is_null()).alias('live'))
         .drop('sup'))
    # One row per (doc_id, sha256): attributes of the text version across all files that hold it.
    per_version = (v.group_by('doc_id', 'sha256').agg(
        pl.first('lang'), pl.first('host'), pl.max('body_cp'), pl.max('han'), pl.max('doc_cp'),
        pl.col('live').any().alias('live'),
        pl.col('segment').filter(pl.col('live')).first().alias('live_segment'),
        pl.col('dir_birth').filter(pl.col('kind') != 'overlay').min().alias('birth'),
        pl.col('src').n_unique().alias('n_files'),
        pl.col('extractor').drop_nulls().first().cast(pl.Utf8).alias('extractor'),
        pl.col('status').drop_nulls().first().cast(pl.Utf8).alias('status'),
    ).with_columns(threshold_expr().alias('passes')))
    per_version = per_version.with_columns(pl.col('birth').fill_null(0.0))
    ranked = per_version.sort(['doc_id', 'live', 'passes', 'birth', 'body_cp', 'sha256'],
                              descending=[False, True, True, True, True, False])
    first = ranked.group_by('doc_id', maintain_order=True).first()
    stats = per_version.group_by('doc_id').agg(pl.len().alias('n_versions'), pl.col('live').sum().alias('n_live'),
                                               pl.col('passes').sum().alias('n_passing'))
    second = (ranked.with_columns(pl.int_range(pl.len()).over('doc_id').alias('rk')).filter(pl.col('rk') == 1)
              .select('doc_id', pl.col('birth').alias('birth2'), pl.col('passes').alias('passes2')))
    chosen = (first.join(stats, on='doc_id').join(second, on='doc_id', how='left')
              .with_columns(pl.when(pl.col('live')).then(pl.lit('a'))
                            .when(~pl.col('passes')).then(pl.lit('thin'))
                            .when(pl.col('passes2') & (pl.col('birth2') == pl.col('birth'))).then(pl.lit('c'))
                            .otherwise(pl.lit('b')).alias('rule'),
                            pl.when(pl.col('passes')).then(pl.lit('good')).otherwise(pl.lit('thin')).alias('quality')))
    reason = (pl.when(pl.col('rule') == 'a').then(pl.format('P2 serves this text (live in {})', pl.col('live_segment')))
              .when(pl.col('rule') == 'b').then(pl.lit('not in P2; newest extraction run passing threshold'))
              .when(pl.col('rule') == 'c').then(pl.lit('not in P2; same extraction run as another passing text; longer body'))
              .otherwise(pl.lit('no version passes threshold; longest body kept as thin')))
    chosen = chosen.with_columns(reason.alias('reason'))
    alts = (ranked.join(chosen.select('doc_id', pl.col('sha256').alias('chosen_sha')), on='doc_id')
            .filter(pl.col('sha256') != pl.col('chosen_sha'))
            .group_by('doc_id').agg(pl.struct(pl.col('sha256').bin.encode('hex').alias('sha256'), 'body_cp', 'han',
                                              'birth', 'live', 'passes', 'n_files').alias('alternatives')))
    decisions = (chosen.filter(pl.col('n_versions') > 1).join(alts, on='doc_id', how='left')
                 .select('doc_id', 'lang', 'host', 'n_versions', 'n_live', 'n_passing', 'rule', 'reason', 'quality',
                         pl.col('sha256').bin.encode('hex').alias('chosen_sha256'), 'body_cp', 'han', 'birth',
                         'live_segment', 'alternatives'))
    return chosen, decisions, per_version, v
