"""URL-only scores. Recall is a cached-retrieval proxy, never a gold metric."""
from __future__ import annotations

from r2ai import paths  # dotenv before numpy/HF imports
import hashlib
import math
import re
from collections import defaultdict
from urllib.parse import unquote, urlsplit

import numpy as np


def holdout_half(url_norms, seed=42):
    """True = fit half A. All documents/aliases of a URL stay together."""
    return np.array([hashlib.sha256(f'{seed}\0{u}'.encode()).digest()[0] < 128
                     for u in url_norms], dtype=bool)


def url_features(url):
    segments = [unquote(s) for s in urlsplit(url).path.split('/') if s]
    parents = segments[:-1]
    last = segments[-1] if segments else ''
    stem = re.sub(r'\.(?:html?|shtml|aspx?|php)$', '', last, flags=re.I)
    nums = re.findall(r'\d+', stem)
    number = int(nums[-1]) if nums else None
    # Number + random suffix is a code, not a language slug. Directory names
    # from an ID-only URL are not presented to the model as a medical title.
    semantic = (bool(re.search('[\u3400-\u9fff]', stem)) or
                bool(re.fullmatch(r'[A-Za-z]+(?:[-_ ][A-Za-z]+)*', stem)))
    slug = re.sub('[-_]+', ' ', stem) if semantic else ''
    return '/'.join(parents[:1]) or '/', '/'.join(parents[:2]) or '/', number, slug


def fit_beta(keys, relevant, documents, strength=5.0):
    hits, total = np.asarray(relevant, float), np.asarray(documents, float)
    if (len(keys) != len(hits) or len(hits) != len(total) or strength < 0
            or not np.isfinite(hits).all() or not np.isfinite(total).all()
            or np.any(hits < 0) or np.any(total <= 0) or np.any(hits > total)):
        raise ValueError('Invalid binary document counts / beta prior')
    mean = float(hits.sum() / total.sum()) if len(total) else 0.0
    counts = defaultdict(lambda: [0., 0.])
    for k, h, n in zip(keys, hits, total):
        counts[k][0] += h
        counts[k][1] += n
    return {k: (h + strength * mean) / (n + strength) for k, (h, n) in counts.items()}, mean


def score_prior(keys, prior, mean):
    return np.array([prior.get(k, mean) for k in keys], dtype=float)


def fit_id_bounds(values, buckets=20):
    valid = np.array([n for n in values if n is not None], dtype=float)
    if buckets < 1:
        raise ValueError('buckets must be positive')
    return np.unique(np.quantile(valid, np.arange(1, buckets) / buckets)) if len(valid) else np.array([])


def id_keys(values, boundaries):
    return [-1 if n is None else int(np.searchsorted(boundaries, n, side='right')) for n in values]


def max_cosine(queries, embeddings, batch=1024):
    q, e = np.asarray(queries, np.float32), np.asarray(embeddings, np.float32)
    if q.ndim != 2 or e.ndim != 2 or q.shape[1] != e.shape[1] or len(q) == 0 or batch <= 0:
        raise ValueError('Invalid embeddings')
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    result = np.zeros(len(e), dtype=np.float32)
    for begin in range(0, len(e), batch):
        block = e[begin:begin + batch]
        block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        result[begin:begin + len(block)] = (q @ block.T).max(axis=0)
    return result


def ranked_indices(scores, doc_ids_count, url_norms):
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite score')
    return np.lexsort((np.asarray(url_norms), -np.asarray(doc_ids_count), -scores))


def recall_at(order, relevant, fractions=(.1, .2, .3, .5)):
    hits = np.asarray(relevant, float)[order]
    if not len(hits):
        return {f: 0. for f in fractions}
    c = np.cumsum(hits) / max(float(hits.sum()), 1.)
    return {f: float(c[min(len(hits), max(1, math.ceil(f * len(hits)))) - 1]) for f in fractions}


def saturation_fraction(curve, target=.95):
    """Predeclared saturation criterion: 95% of binary relevant-doc proxy."""
    return next((f for f in sorted(curve) if curve[f] >= target), 1.)


def allocate_lane(domains, hours, uptime):
    """Domains run sequentially within one lane; each lane owns wall hours.

    Each input is (domain, eligible URL count, measured ok/hour, saturation fraction).
    Success throughput is conservative for URL capacity (thin/errors are not ok).
    Missing or zero measured throughput keeps the domain on hold.
    """
    if hours < 0 or not 0 <= uptime <= 1:
        raise ValueError('Invalid wall hours / uptime')
    left, result = hours * uptime, []
    for domain, count, rate, saturation in domains:
        if count < 0 or not 0 <= saturation <= 1 or (rate is not None and rate < 0):
            raise ValueError('Invalid domain budget')
        capacity = math.floor(left * rate) if rate else 0
        saturated = math.ceil(count * saturation)
        cut = min(count, capacity, saturated)
        active = cut / rate if rate else 0.
        result.append({'domain': domain, 'budget_urls': capacity, 'saturation_urls': saturated,
                       'cut': cut, 'hours_active': active, 'lane_active_hours_before': left})
        left = max(0., left - active)
    return result
