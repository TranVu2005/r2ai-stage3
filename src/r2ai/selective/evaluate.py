"""Offline A/B/C comparison on hash URL holdout; binary dense top100 proxy."""
from __future__ import annotations
from r2ai.paths import WORK_DATA_DIR, RUNS_DIR, require_inputs
from .runio import RUN, DB, guard, preflight, ro, atomic_json
from . import scoring as s

import argparse
import datetime as dt
import hashlib
import itertools
import json
import pickle
import time

import numpy as np
import pandas as pd

DEADLINE = '2026-10-20T23:59:00+07:00'
FRACTIONS = tuple(np.round(np.arange(.05, 1.01, .05), 2))


def proxy_ids(cache):
    return {int(i) for row in cache.values() for i, _ in row['dense'][:100]}


def group_sample(df, hits):
    out = df[['url_norm', 'url', 'domain']].copy()
    out['documents'] = 1
    out['relevant'] = [int(bool(set(ids) & hits)) for ids in df.doc_ids]
    out['aliases'] = [len(ids) for ids in df.doc_ids]
    return out.groupby('url_norm', sort=True, as_index=False).agg(
        url=('url', 'first'), domain=('domain', 'first'), documents=('documents', 'sum'),
        relevant=('relevant', 'sum'), aliases=('aliases', 'sum'))


def features(df):
    return list(zip(*map(s.url_features, df.url)))


def structural_scores(train, target):
    if not len(train):
        return {m:np.zeros(len(target)) for m in ('A1','A2','C20','C100')}
    tf, pf = features(train), features(target)
    scores = {}
    for method, key in [('A1', 0), ('A2', 1)]:
        prior, mean = s.fit_beta(tf[key], train.relevant, train.documents)
        scores[method] = s.score_prior(pf[key], prior, mean)
    for buckets in (20, 100):
        bounds = s.fit_id_bounds(tf[2], buckets)
        prior, mean = s.fit_beta(s.id_keys(tf[2], bounds), train.relevant, train.documents)
        scores[f'C{buckets}'] = s.score_prior(s.id_keys(pf[2], bounds), prior, mean)
    return scores


def combinations(scores):
    result = dict(scores)
    for n in (2, 3):
        for parts in itertools.combinations(scores, n):
            if sum(p.startswith('A') for p in parts) > 1 or sum(p.startswith('C') for p in parts) > 1:
                continue
            result['+'.join(parts)] = np.mean([scores[p] for p in parts], axis=0)
    return result


def evaluate_domain(df, dense_scores=None):
    fit = s.holdout_half(df.url_norm)
    train, test = df[fit], df[~fit]
    enough = int(df.relevant.sum()) >= 200 and len(train) > 0 and int(test.relevant.sum()) > 0
    scores = structural_scores(train, test) if len(train) and len(test) else {}
    if dense_scores is not None:
        scores['B'] = np.asarray(dense_scores)[~fit]
    scores = combinations(scores)
    orders = {'alphabetical': np.argsort(test.url_norm.to_numpy(), kind='stable'),
              'random42': np.argsort([hashlib.sha256(f'42\0{u}'.encode()).hexdigest() for u in test.url_norm])}
    orders.update({m: s.ranked_indices(v, test.aliases, test.url_norm) for m, v in scores.items()})
    curves = {m: {str(float(f)): v for f, v in s.recall_at(o, test.relevant, FRACTIONS).items()}
              for m, o in orders.items()}
    candidates = list(scores) if enough else (['B'] if dense_scores is not None else [])
    if candidates:
        # Predeclared utility: mean recall at requested 10/20/30/50%.
        chosen = max(candidates, key=lambda m: (sum(curves[m][str(f)] for f in (.1,.2,.3,.5)), -len(m), m))
        saturation = s.saturation_fraction({float(f): v for f, v in curves[chosen].items()})
    else:
        chosen, saturation = 'insufficient_signal', None
    return {'sample_docs': int(df.documents.sum()), 'sample_urls': len(df), 'relevant_docs': int(df.relevant.sum()),
            'fit_urls': int(fit.sum()), 'holdout_urls': int((~fit).sum()),
            'fit_relevant': int(train.relevant.sum()), 'holdout_relevant': int(test.relevant.sum()),
            'chosen_method': chosen, 'saturation_fraction': saturation, 'curves': curves,
            'proxy': 'binary doc in dense reranker top100 of >=1/1200 queries; not gold',
            'selection_utility': 'mean holdout recall@10/20/30/50%; same holdout used to choose method'}


class SlugScorer:
    """Same local BGE-M3 encoder; query cache reused, no translation/download."""
    def __init__(self):
        self.encoder = None
        self.queries = pickle.load((RUNS_DIR / 'zh-sample/retrieval/queries.pkl').open('rb'))[0]

    def __call__(self, df, tag):
        slug = [s.url_features(u)[3] for u in df.url]
        unique = sorted(set(slug) - {''})
        if not unique:
            return None
        path = guard(RUN / f'slug-{tag}.parquet')
        fingerprint = hashlib.sha256(json.dumps(unique, ensure_ascii=False).encode()).hexdigest()
        marker = RUN / f'slug-{tag}.json'
        if marker.exists() and json.loads(marker.read_text('utf-8'))['texts_sha256'] == fingerprint:
            cache = pd.read_parquet(path)
        else:
            if self.encoder is None:
                import torch
                from r2ai.index.bge_m3 import M3Encoder
                self.encoder = M3Encoder(device='cuda' if torch.cuda.is_available() else 'cpu', fp16=False, max_len=64)
            vectors = []
            for i in range(0, len(unique), 32):
                vectors.append(self.encoder.encode_batch(unique[i:i+32])[0])
                if i % 1024 == 0:
                    print(f'slug {tag}: {i}/{len(unique)}', flush=True)
            scores = s.max_cosine(self.queries, np.concatenate(vectors))
            cache = pd.DataFrame({'slug': unique, 'score': scores})
            cache.to_parquet(path, index=False)
            atomic_json(marker, {'texts_sha256': fingerprint, 'texts':len(unique), 'model':'BAAI/bge-m3',
                                 'max_len':64, 'fp16':False, 'query_cache':'zh-sample/retrieval/queries.pkl'})
        by_slug = dict(zip(cache.slug, cache.score))
        return np.array([by_slug.get(t, 0.) for t in slug])


def build_plan(deadline=DEADLINE):
    from r2ai.zh_full.crawl import LANES
    from r2ai.zh_sample.append import _load_cache
    preflight()
    require_inputs(DB, RUN / 'measurements.json', WORK_DATA_DIR / 'docs_zh_sample/bundle')
    now = time.time()
    version = f'zh-select-{int(now)}'
    finish = dt.datetime.fromisoformat(deadline)
    if finish.tzinfo is None:
        raise ValueError('Deadline requires timezone, e.g. +07:00')
    hours = max(0., (finish.timestamp() - now) / 3600)
    measurements = json.loads((RUN / 'measurements.json').read_text('utf-8'))
    cache, cache_config, cache_hashes = _load_cache(list(range(1,1201)))
    hits = proxy_ids({qi:{'dense':ranking} for qi,ranking in cache.items()})
    del cache
    # An interrupted rebuild cannot leave a success manifest pointing at mixed files.
    atomic_json(RUN / 'plan.json', {'complete':False,'cut_version':version,'reason':'building'})
    slug_scorer = SlugScorer()
    evaluations, budgets, orders, order_files = {}, {}, {}, {}
    md = {r['domain']: r for r in measurements['domains']}
    with ro() as conn:
        for lane_num, lane in enumerate(LANES):
            for domain in lane:
                pending = pd.read_sql_query("SELECT url_norm,url,doc_ids,rank,status,next_try_at FROM urls WHERE domain=? AND status IN ('pending','network_error') ORDER BY url_norm",conn,params=(domain,))
                if not len(pending):
                    continue
                pending['aliases'] = [len(json.loads(v)) for v in pending.doc_ids]
                sample_path = WORK_DATA_DIR / f'docs_zh_sample/bundle/{domain}.parquet'
                sample = (pd.read_parquet(sample_path, columns=['url_norm','url','domain','doc_ids']) if sample_path.exists()
                          else pd.DataFrame(columns=['url_norm','url','domain','doc_ids']))
                sample = group_sample(sample, hits)
                dense = slug_scorer(sample, f'{domain}-sample')
                evaluation = evaluate_domain(sample, dense)
                pending_dense = None
                if evaluation['chosen_method'] == 'insufficient_signal' and not len(sample):
                    pending_dense = slug_scorer(pending, f'{domain}-pending')
                    if pending_dense is not None:
                        evaluation = evaluate_domain(sample, np.array([]))
                        evaluation['no_holdout_estimate'] = True
                method = evaluation['chosen_method']
                sc = structural_scores(sample, pending)
                if method != 'insufficient_signal' and 'B' in method:
                    sc['B'] = pending_dense if pending_dense is not None else slug_scorer(pending, f'{domain}-pending')
                score = combinations(sc).get(method, np.zeros(len(pending)))
                index = s.ranked_indices(score, pending.aliases, pending.url_norm)
                order = pending.iloc[index].copy().reset_index(drop=True)
                order['score'] = score[index]
                order['method'] = method
                order['domain'] = domain
                order['new_rank'] = np.sort(pending['rank'].to_numpy())  # permute only existing eligible ranks
                order['position'] = np.arange(len(order))
                order['cut_version'] = version
                orders[domain] = order
                session = md[domain]['last_session']
                rate = session['ok_per_h'] if session else None
                evaluation.update({'lane':lane_num, 'eligible':len(order), 'ok_per_h_measured':rate,
                                   'slug_sample_urls':int(sum(bool(s.url_features(u)[3]) for u in sample.url))})
                evaluations[domain] = evaluation
                print(domain, method, evaluation['relevant_docs'], evaluation['curves'].get(method), flush=True)
            inputs = [(d,len(orders[d]), evaluations[d]['ok_per_h_measured'], evaluations[d]['saturation_fraction'] or 0.)
                      for d in lane if d in orders]
            for budget in s.allocate_lane(inputs, hours, measurements['uptime']['uptime_48h']):
                domain = budget['domain']
                evaluation = evaluations[domain]
                if evaluation['chosen_method'] == 'insufficient_signal':
                    budget['hold_reason'] = 'insufficient_signal: propose random42 scout <=2000 before selection'
                elif evaluation['ok_per_h_measured'] is None:
                    budget['hold_reason'] = 'no measured full-crawl throughput: propose scout before budget'
                elif evaluation['ok_per_h_measured'] == 0:
                    budget['hold_reason'] = 'waiting_robots: observed 0 ok/hour; never bypass robots'
                fraction = budget['cut'] / evaluation['eligible']
                curve = evaluation['curves'].get(evaluation['chosen_method'])
                budget['recall_proxy_at_cut_estimated'] = float(np.interp(fraction, [0]+[float(f) for f in curve], [0]+list(curve.values()))) if curve and evaluation['holdout_relevant'] else None
                order = orders[domain]
                order['new_status'] = np.where(order.position < budget['cut'], order.status, 'deferred_select')
                budget['doc_ids_selected'] = int(order.loc[order.position < budget['cut'],'aliases'].sum())
                path = guard(RUN / f'order-{domain}.parquet')
                order.to_parquet(path,index=False)
                with path.open('rb') as f:
                    sha = hashlib.file_digest(f,'sha256').hexdigest()
                order_files[domain] = {'sha256':sha,'rows':len(order)}
                budgets[domain] = budget
                del orders[domain]
    manifest = {'complete':True, 'cut_version':version, 'built_at':now, 'deadline':deadline, 'hours_remaining':hours,
                'uptime':measurements['uptime'], 'domains':evaluations, 'budgets':budgets,
                'threshold': '95% binary proxy recall; evaluated in 5% URL increments',
                'lane_rule':'subtract each selected domain active hours from that lane only',
                'insufficient_action':'hold; no blind relevance cutoff; scout requires user approval'}
    manifest.update(order_files=order_files, cache_hashes=cache_hashes, cache_config=cache_config)
    atomic_json(RUN / 'plan.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', default=DEADLINE)
    args = parser.parse_args(argv)
    build_plan(args.deadline)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
