"""Read-only QA of the extractor output of the new zh v2 hosts (docs already ingested into the zh v2 index).

Samples N docs per host (seed 42) from data/chunks_zh_v2/<segment>/docs.parquet and flags, per doc: Q&A pair (ask.39.net: question
header + non-empty answer), site chrome (menu / footer lines), related-article blocks, ads, comment blocks, length, error or
interstitial pages. Line-level rules: a section marker only counts as a line that is short (a heading), so a sentence that merely
contains "相关文章" is not flagged. Nothing is written outside out/runs/zh-v2/qa_extract and nothing is fetched.

    python -m r2ai.zh_v2.qa_extract [--n 100] [--seed 42] [--top-39 3]
"""
from __future__ import annotations

from r2ai.paths import RUNS_DIR, assert_writable

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from . import index as zindex

OUT = RUNS_DIR / 'zh-v2' / 'qa_extract'
FIXED = ('ask.39.net', 'familydoctor.com.cn', 'baby.39.net')
ERROR_RE = re.compile(r'404|页面不存在|页面未找到|您访问的页面|找不到(?:该|此)?页面|出错了|访问频繁|验证码|安全验证|请开启\s*JavaScript|'
                      r'禁止访问|系统繁忙|正在跳转|页面跳转|Access Denied|Not Found|Bad Gateway|Service Unavailable', re.I)
MENU = ('首页', '网站地图', '返回顶部', '上一篇', '下一篇', '联系我们', '关于我们', '免责声明', '版权所有', '登录', '注册', '加入收藏',
        '设为首页', '友情链接', '法律声明', '隐私政策', '广告服务', '招聘信息', 'ICP备', 'Copyright', '客服电话', '意见反馈')
RELATED = ('相关文章', '相关阅读', '相关推荐', '推荐阅读', '热门文章', '猜你喜欢', '更多相关', '延伸阅读', '相关疾病', '相关问题',
           '相关专题', '热门推荐', '精彩推荐', '最新文章', '热点推荐', '大家还关注', '您可能感兴趣')
ADS = ('咨询电话', '扫码', '关注公众号', '微信公众号', '点击咨询', '在线咨询', '立即预约', '加微信', '免费咨询', '专家在线',
       '立即挂号', '下载APP', '下载app', '扫一扫', '二维码', '内容合作请联系', '未经授权不得转载')
COMMENTS = ('网友评论', '我要评论', '发表评论', '评论列表', '跟帖', '我要留言', '查看全部评论', '共有评论', '条评论', '登录后评论')
PHONE = re.compile(r'(?:0\d{2,3}[- ]?\d{7,8}|400[- ]?\d{3}[- ]?\d{4}|1[3-9]\d{9})')
HEADING_MAX = 40        # a marker counts as a section heading only on a line this short
AD_LINE_MAX = 90
MIN_CHARS = 60
MAX_CHARS = 30_000


def guard(path) -> Path:
    p = assert_writable(path)
    if not p.is_relative_to(assert_writable(OUT)):
        raise ValueError(f'Not a qa_extract output: {p}')
    return p


def top_39_subs(manifest, n) -> list[str]:
    docs = Counter()
    for r in manifest['segments']:
        d = r['domain']
        if d.endswith('.39.net') and d not in FIXED and r.get('docs_v2') and r.get('source') == 'zh_v2_ingest':
            docs[d] += r['docs_v2']
    return [d for d, _ in docs.most_common(n)]


def host_segments(manifest, host) -> list[str]:
    return [r['name'] for r in manifest['segments'] if r['domain'] == host and r.get('docs_v2')]


def lines_of(text: str) -> list[str]:
    return [x.strip() for x in text.split('\n') if x.strip()]


def qa_state(host: str, text: str, lines: list[str]) -> str | None:
    """ask.39.net body = title, question lines, 发病时间, tags line, then the answer(s). None = not a Q&A host."""
    if host != 'ask.39.net':
        return None
    idx = next((i for i, x in enumerate(lines) if x.startswith('发病时间')), None)
    if idx is None:
        return 'no_question_header'
    rest = [x for x in lines[idx + 1:] if not re.fullmatch(r'\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?', x)]
    if len(rest) < 2:                                   # tags line only, or nothing: no answer text
        return 'no_answer'
    answer = ''.join(rest[1:])
    return 'ok' if len(answer) >= 15 else 'no_answer'


def check(host: str, row: dict) -> dict:
    text = row['doc_text'] or ''
    lines = lines_of(text)
    flags, evidence = [], {}

    def hit(name, line):
        flags.append(name)
        evidence.setdefault(name, line[:160])

    if len(text) < MIN_CHARS:
        hit('too_short', text[:160])
    if len(text) > MAX_CHARS:
        hit('too_long', f'{len(text)} chars')
    for x in lines[:6] + lines[-3:]:
        if len(x) <= 60 and ERROR_RE.search(x):
            hit('error_page', x)
            break
    if len(text) < 400 and ERROR_RE.search(text):
        hit('error_page', text[:160])
    for x in lines:
        if len(x) <= HEADING_MAX and any(k in x for k in MENU):
            hit('menu_footer', x)
            break
    for x in lines:
        if len(x) <= HEADING_MAX and any(k in x for k in RELATED):
            hit('related_block', x)
            break
    for x in lines:
        if len(x) <= AD_LINE_MAX and (any(k in x for k in ADS) or PHONE.search(x) and len(x) <= 50 or x in ('广告', '广告位')):
            hit('ads', x)
            break
    for x in lines:
        if len(x) <= HEADING_MAX and any(k in x for k in COMMENTS):
            hit('comments', x)
            break
    qa = qa_state(host, text, lines)
    if qa not in (None, 'ok'):
        hit(f'qa_{qa}', lines[0] if lines else '')
    return {'doc_id': row['doc_id'], 'url': row['url'], 'chars': len(text), 'qa': qa, 'flags': flags, 'evidence': evidence}


SEVERITY = {'error_page': 5, 'qa_no_answer': 5, 'qa_no_question_header': 4, 'too_short': 4, 'comments': 3, 'related_block': 2, 'ads': 2,
            'menu_footer': 2, 'too_long': 1}


def score(c) -> int:
    return sum(SEVERITY.get(f, 1) for f in set(c['flags']))


def load_host(manifest, host) -> pl.DataFrame:
    parts = [zindex.segment_docs(manifest, n, columns=['doc_id', 'url', 'title', 'doc_text'])
             for n in host_segments(manifest, host)]
    return pl.concat(parts) if parts else pl.DataFrame()


def run(n=100, seed=42, top39=3) -> dict:
    manifest = json.loads(zindex.MANIFEST.read_text('utf-8'))
    hosts = list(FIXED) + top_39_subs(manifest, top39)
    table, examples, per_flag_full = [], {}, {}
    for host in hosts:
        df = load_host(manifest, host)
        total = df.height
        rows = df.to_dicts()
        rng = random.Random(seed)
        sample = rng.sample(rows, min(n, total))
        checked = [check(host, r) for r in sample]
        bad = [c for c in checked if c['flags']]
        flag_counts = Counter(f for c in checked for f in set(c['flags']))
        chars = sorted(c['chars'] for c in checked)
        table.append({'host': host, 'docs_in_index': total, 'sampled': len(checked), 'bad_docs': len(bad),
                      'bad_pct': round(100 * len(bad) / max(1, len(checked)), 1),
                      'qa_ok': sum(1 for c in checked if c['qa'] == 'ok'), 'qa_applicable': sum(1 for c in checked if c['qa'] is not None),
                      'chars_p10_p50_p90': [chars[len(chars) // 10], chars[len(chars) // 2], chars[(9 * len(chars)) // 10]],
                      'flags': dict(sorted(flag_counts.items()))})
        worst = sorted(bad, key=lambda c: (-score(c), c['doc_id']))[:5]
        by_id = {r['doc_id']: r for r in sample}
        examples[host] = [{**w, 'score': score(w), 'head': by_id[w['doc_id']]['doc_text'][:240], 'from': 'sample'} for w in worst]
        # same detector over every doc of the host (read-only, CPU): the re-ingest estimate for hosts above 5%
        full = Counter()
        full_bad = 0
        full_worst = []
        for r in rows:
            c = check(host, r)
            if c['flags']:
                full_bad += 1
                full_worst.append((score(c), r['doc_id'], c, r['doc_text'][:240]))
            full.update(set(c['flags']))
        full_worst.sort(key=lambda t: (-t[0], t[1]))
        seen = {e['doc_id'] for e in examples[host]}
        for sc, did, c, head in full_worst:
            if len(examples[host]) >= 5:
                break
            if did not in seen:
                examples[host].append({**c, 'score': sc, 'head': head, 'from': 'full_scan'})
        per_flag_full[host] = {'docs': total, 'bad_docs': full_bad, 'bad_pct': round(100 * full_bad / max(1, total), 2),
                               'flags': dict(sorted(full.items()))}
    out = guard(OUT)
    out.mkdir(parents=True, exist_ok=True)
    result = {'n': n, 'seed': seed, 'hosts': hosts, 'table': table, 'full_scan': per_flag_full, 'examples': examples}
    (guard(out / 'qa_extract.json')).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def redo_list(manifest=None) -> dict:
    """Docs to re-extract after the D13 extractor fixes: ask.39.net answers lost to nested <p>, familydoctor.com.cn chrome lines."""
    manifest = manifest or json.loads(zindex.MANIFEST.read_text('utf-8'))
    out = {}
    for host, want in (('ask.39.net', lambda f: 'qa_no_answer' in f), ('familydoctor.com.cn', lambda f: bool(f))):
        out[host] = sorted(r['doc_id'] for r in load_host(manifest, host).to_dicts() if want(check(host, r)['flags']))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--top-39', type=int, default=3)
    ap.add_argument('--redo-list', help='write the doc ids to re-extract (JSON {host: [ids]}) and stop')
    a = ap.parse_args(argv)
    if a.redo_list:
        lst = redo_list()
        guard(a.redo_list).write_text(json.dumps(lst), encoding='utf-8')
        print(json.dumps({h: len(v) for h, v in lst.items()}))
        return 0
    r = run(a.n, a.seed, a.top_39)
    for t in r['table']:
        print(json.dumps(t, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
