"""Duplicate / near-duplicate clustering of docs (H1). Pure functions; the corpus scan lives in r2ai.dupes.scan.

exact   key = blake2b-16 of the body after the official metric normalisation (NFKC, html.unescape, lowercase,
        whitespace collapsed, strip); the same function as r2ai.eval.scorer.normalize_metric.
near    docs with >= MIN_TOKENS BGE-M3 tokens: words = \\w+ runs of the normalised body, shingles = 5 consecutive words
        (64-bit hashes, unique), MinHash 128 permutations (seed 42), LSH 32 bands x 4 rows, then the TRUE Jaccard of the
        two shingle sets must be >= 0.8. Exact groups are collapsed first (one representative per group).
cluster connected components of exact groups + verified near links. A cluster is flagged
        short     median n_tokens < SHORT_TOKENS (boilerplate with a short body),
        template  >= 3 docs, one domain, >= 80 % distinct titles (same template, different page),
        boilerplate = short or template.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from r2ai.eval.scorer import normalize_metric

SEED = 42
NUM_PERM = 128
BANDS, ROWS = 32, 4
SHINGLE = 5
MIN_TOKENS = 50
JACCARD = 0.8
EST_FLOOR = 0.6               # MinHash estimate below this is not verified (std of the estimate at J = 0.8 is 0.035)
SHORT_TOKENS = 100
TEMPLATE_MIN_DOCS = 3
TEMPLATE_TITLE_RATIO = 0.8
RUN_CAP = 50                  # LSH bucket larger than this: each doc is paired with its next RUN_CAP neighbours only

_WORD = re.compile(r'\w+')
_U64 = np.uint64
_MASK = (1 << 64) - 1


def exact_key(text: str | None) -> bytes | None:
    """16-byte digest of the metric-normalised text; None for empty text."""
    norm = normalize_metric(text)
    return hashlib.blake2b(norm.encode('utf-8'), digest_size=16).digest() if norm else None


def title_key(title: str | None) -> bytes:
    return hashlib.blake2b(normalize_metric(title).encode('utf-8'), digest_size=8).digest()


def chunk_key(body: str | None, answer: str | None, title: str | None, description: str | None) -> bytes | None:
    """exact_key of the whole-doc chunk text make_submission.load_full_texts builds (answer if inside the body, else body,
    else title + description)."""
    body = body or ''
    ans = (answer or '').strip()
    if ans and ans in body:
        return exact_key(ans)
    if body.strip():
        return exact_key(body)
    return exact_key('\n\n'.join(x.strip() for x in (title, description) if x and x.strip()))


def words(text: str | None) -> list[str]:
    return _WORD.findall(normalize_metric(text))


class WordCoder:
    """word -> stable 64-bit code (blake2b-8), cached; the cache is reset when it grows past `limit` entries."""

    def __init__(self, limit: int = 4_000_000):
        self.cache: dict[str, int] = {}
        self.limit = limit

    def codes(self, ws: list[str]) -> np.ndarray:
        if len(self.cache) > self.limit:
            self.cache.clear()
        c = self.cache
        out = []
        for w in ws:
            v = c.get(w)
            if v is None:
                v = c[w] = int.from_bytes(hashlib.blake2b(w.encode('utf-8'), digest_size=8).digest(), 'little')
            out.append(v)
        return np.array(out, dtype=_U64)


_CODER = WordCoder()
_POS = np.random.default_rng(SEED).integers(1, 2**63, size=SHINGLE, dtype=np.uint64) | _U64(1)


def shingle_hashes(ws: list[str], coder: WordCoder | None = None, k: int = SHINGLE) -> np.ndarray:
    """Sorted unique uint64 hashes of the k-word shingles (one shingle of all words when len(ws) < k)."""
    n = len(ws)
    if n == 0:
        return np.empty(0, dtype=_U64)
    c = (coder or _CODER).codes(ws)
    with np.errstate(over='ignore'):
        if n < k:
            idx = np.arange(n)
            s = (c * _POS[idx]).sum(dtype=_U64, keepdims=True)
        else:
            m = n - k + 1
            s = c[:m] * _POS[0]
            for j in range(1, k):
                s = s + c[j:j + m] * _POS[j]
        s = s ^ (s >> _U64(29))
        s = s * _U64(0xBF58476D1CE4E5B9)
        s = s ^ (s >> _U64(32))
    return np.unique(s)


class MinHasher:
    def __init__(self, num_perm: int = NUM_PERM, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | _U64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)

    def signature(self, sh: np.ndarray, block: int = 20000) -> np.ndarray:
        """uint32[num_perm]: per permutation the minimum of the high 32 bits of a*s + b (mod 2^64); empty set -> all max."""
        sig = np.full(self.num_perm, 0xFFFFFFFF, dtype=np.uint32)
        with np.errstate(over='ignore'):
            for i in range(0, len(sh), block):
                x = sh[i:i + block]
                h = ((self.a[:, None] * x[None, :] + self.b[:, None]) >> _U64(32)).astype(np.uint32)
                sig = np.minimum(sig, h.min(axis=1))
        return sig


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Exact Jaccard of two sorted unique arrays."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    return inter / (len(a) + len(b) - inter)


def lsh_pairs(sigs: np.ndarray, bands: int = BANDS, rows: int = ROWS, run_cap: int = RUN_CAP,
              stats: dict | None = None) -> set[tuple[int, int]]:
    """Candidate pairs (i < j) whose signatures agree on all rows of at least one band."""
    n = sigs.shape[0]
    assert sigs.shape[1] >= bands * rows
    out: set[tuple[int, int]] = set()
    capped = 0
    prime = _U64(0x9E3779B97F4A7C15)
    for b in range(bands):
        blk = sigs[:, b * rows:(b + 1) * rows]
        with np.errstate(over='ignore'):
            key = blk[:, 0].astype(_U64)
            for j in range(1, rows):
                key = key * prime + blk[:, j].astype(_U64)
        order = np.argsort(key, kind='stable')
        ks = key[order]
        cuts = np.flatnonzero(np.diff(ks)) + 1
        starts, ends = np.concatenate(([0], cuts)), np.concatenate((cuts, [n]))
        for s0, e0 in zip(starts[ends - starts >= 2].tolist(), ends[ends - starts >= 2].tolist()):
            run = order[s0:e0]
            m = e0 - s0
            if m <= run_cap + 1:
                for x in range(m):
                    for y in range(x + 1, m):
                        i, j = int(run[x]), int(run[y])
                        out.add((i, j) if i < j else (j, i))
            else:
                capped += 1
                for x in range(m):
                    for y in range(x + 1, min(m, x + 1 + run_cap)):
                        i, j = int(run[x]), int(run[y])
                        out.add((i, j) if i < j else (j, i))
    if stats is not None:
        stats['lsh_buckets_capped'] = stats.get('lsh_buckets_capped', 0) + capped
    return out


def est_jaccard(sigs: np.ndarray, i: int, j: int) -> float:
    return float((sigs[i] == sigs[j]).mean())


def components(n: int, links) -> np.ndarray:
    """Connected-component label (smallest member index) per node; links = iterable of (i, j)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in links:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def flags(ntokens, titles: list[bytes], domains: list) -> tuple[bool, bool]:
    """(short, template) for one cluster."""
    n = len(titles)
    short = float(np.median(np.asarray(ntokens, dtype=float))) < SHORT_TOKENS
    template = n >= TEMPLATE_MIN_DOCS and len(set(domains)) == 1 and len(set(titles)) / n >= TEMPLATE_TITLE_RATIO
    return short, template


@dataclass(frozen=True)
class Doc:
    doc_id: int
    domain: str
    title: str
    body: str
    n_tokens: int
    answer: str = ''
    description: str = ''


@dataclass
class Cluster:
    cluster_id: int
    doc_ids: list[int]
    kind: str                       # 'exact' (one normalised body) or 'near' (>= 1 verified near link)
    exact_groups: int
    domains: int
    short: bool
    template: bool
    min_jaccard: float | None       # smallest verified near-link Jaccard in the cluster (None for exact clusters)

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    @property
    def boilerplate(self) -> bool:
        return self.short or self.template


@dataclass
class Result:
    clusters: list[Cluster]
    cluster_of: dict[int, int]      # doc_id -> cluster_id (docs in clusters of size >= 2 only)
    stats: dict = field(default_factory=dict)


def build_clusters(doc_ids, ntokens, titles, domains, group_of, near_links: dict[tuple[int, int], float]) -> list[Cluster]:
    """Clusters (size >= 2) from per-doc exact-group labels `group_of` (node index of the group's first doc) and verified
    near links between group representatives. Cluster ids are 0.. in order of the smallest member doc position."""
    n = len(doc_ids)
    links = [(int(group_of[i]), i) for i in range(n) if group_of[i] != i] + list(near_links)
    comp = components(n, links)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(int(comp[i]), []).append(i)
    jmin: dict[int, float] = {}
    for (i, j), jac in near_links.items():
        r = int(comp[i])
        jmin[r] = min(jmin.get(r, 1.0), jac)
    out = []
    for r in sorted(members):
        m = members[r]
        if len(m) < 2:
            continue
        sh, tp = flags([ntokens[i] for i in m], [titles[i] for i in m], [domains[i] for i in m])
        out.append(Cluster(cluster_id=len(out), doc_ids=[int(doc_ids[i]) for i in m], kind='near' if r in jmin else 'exact',
                           exact_groups=len({int(group_of[i]) for i in m}), domains=len({domains[i] for i in m}),
                           short=sh, template=tp, min_jaccard=jmin.get(r)))
    return out


def cluster_corpus(docs: list[Doc], *, min_tokens: int = MIN_TOKENS, jaccard_min: float = JACCARD) -> Result:
    """In-memory clustering (used by tests and small inputs); the corpus scan calls the same primitives in batches."""
    n = len(docs)
    first: dict[bytes, int] = {}
    group_of = np.arange(n)
    for i, d in enumerate(docs):
        k = exact_key(d.body)
        if k is not None:
            group_of[i] = first.setdefault(k, i)
    reps = [i for i in range(n) if group_of[i] == i and docs[i].n_tokens >= min_tokens and exact_key(docs[i].body) is not None]
    sh = {i: shingle_hashes(words(docs[i].body)) for i in reps}
    mh = MinHasher()
    sigs = np.stack([mh.signature(sh[i]) for i in reps]) if reps else np.empty((0, NUM_PERM), np.uint32)
    stats: dict = {}
    near: dict[tuple[int, int], float] = {}
    for a, b in sorted(lsh_pairs(sigs, stats=stats)) if len(reps) > 1 else []:
        i, j = reps[a], reps[b]
        jac = jaccard(sh[i], sh[j])
        if jac >= jaccard_min:
            near[(i, j)] = jac
    cl = build_clusters([d.doc_id for d in docs], [d.n_tokens for d in docs], [title_key(d.title) for d in docs],
                        [d.domain for d in docs], group_of, near)
    of = {d: c.cluster_id for c in cl for d in c.doc_ids}
    stats |= {'docs': n, 'clusters': len(cl), 'docs_in_clusters': len(of)}
    return Result(cl, of, stats)
