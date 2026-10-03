"""Embed chunks with BGE-M3 (dense + sparse, fp16, max_len 512) and build the index.

    python -m index.build bench --target 256 [--n 2000] [--batch-sizes 4,8,16,32,64]
    python -m index.build build --target 256 [--batch-size 16] [--shard-size 10000]

bench: 2,000 chunks sampled uniformly (seed 42) from data/chunks/chunks_t{target}.parquet, encoded per batch size with
       the same length-sorted batching as build. Reports chunk/s and peak VRAM (torch max_memory_allocated /
       max_memory_reserved) and an extrapolation to every chunk of the config (labelled as extrapolated).
build: shards of --shard-size chunks in chunk_id order -> data/index/t{target}/shards/{i:05d}.dense.npy / .sparse.npz,
       each written to *.tmp then os.replace'd (sparse last = shard done). Ctrl+C loses at most the current shard;
       re-running skips finished shards. Then:
         dense.npy    float16 [N, 1024] (open with np.load(mmap_mode='r'))
         sparse.npz   scipy CSR float16->float32 [N, vocab] (lexical weights)
         faiss.index  IndexFlatIP over float32 if N*1024*4 bytes <= --flat-max-gb, else IVF (nlist = 4*sqrt(N)), IP
         meta.json    model, max_len, N, chunk file, build times
Row i of every file = chunk_id i of chunks_t{target}.parquet.
"""
from __future__ import annotations

from r2ai.paths import INDEX_DIR, assert_writable, chunks_file, data_label, index_dir, require_inputs

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault('HF_HUB_OFFLINE', '1')


def load_texts(target: int) -> list[str]:
    t = pq.read_table(chunks_file(target), columns=['chunk_id', 'text'])
    ids = t['chunk_id'].to_numpy()
    assert (ids == np.arange(len(ids))).all(), 'chunk_id must be 0..N-1 in file order'
    return t['text'].to_pylist()


def encode(enc, texts: list[str], batch_size: int, progress: bool = False):
    """Length-sorted batches; returns dense [n,1024] fp16 and sparse rows in the input order."""
    import torch
    lens = [len(x) for x in enc.tok(texts, add_special_tokens=True, truncation=True, max_length=enc.max_len)['input_ids']]
    order = np.argsort(lens)[::-1]
    dense = np.zeros((len(texts), enc.model.config.hidden_size), dtype=np.float16)
    sparse = [None] * len(texts)
    for b in range(0, len(texts), batch_size):
        idx = order[b:b + batch_size]
        d, s = enc.encode_batch([texts[i] for i in idx])
        dense[idx] = d
        for k, i in enumerate(idx):
            sparse[i] = s[k]
        if progress and (b // batch_size) % 200 == 0:
            print(f'  {b}/{len(texts)}', flush=True)
    torch.cuda.synchronize()
    return dense, sparse


def to_csr(sparse, vocab: int):
    from scipy.sparse import csr_matrix
    indptr = np.zeros(len(sparse) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(t) for t, _ in sparse])
    indices = np.concatenate([t for t, _ in sparse]) if sparse else np.zeros(0, np.int32)
    data = np.concatenate([v for _, v in sparse]).astype(np.float32) if sparse else np.zeros(0, np.float32)
    return csr_matrix((data, indices, indptr), shape=(len(sparse), vocab))


def cmd_bench(a):
    import random

    import torch
    from r2ai.index.bge_m3 import M3Encoder
    texts = load_texts(a.target)
    rnd = random.Random(42)
    sample = [texts[i] for i in rnd.sample(range(len(texts)), a.n)]
    enc = M3Encoder(max_len=512)
    lens = [len(x) for x in enc.tok(sample, truncation=True, max_length=512)['input_ids']]
    res = {'target': a.target, 'n_sample': a.n, 'n_total_chunks': len(texts), 'gpu': torch.cuda.get_device_name(0),
           'sample_tokens_median': float(np.median(lens)), 'sample_tokens_p95': float(np.percentile(lens, 95)),
           'sample_pct_truncated_512': round(100 * float(np.mean(np.array(lens) >= 512)), 2), 'runs': []}
    encode(enc, sample[:64], 16)                                    # warm-up (kernels, allocator)
    for bs in [int(x) for x in a.batch_sizes.split(',')]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            encode(enc, sample, bs)
        except torch.cuda.OutOfMemoryError:
            res['runs'].append({'batch_size': bs, 'oom': True})
            torch.cuda.empty_cache()
            continue
        dt = time.perf_counter() - t0
        res['runs'].append({'batch_size': bs, 'seconds': round(dt, 1), 'chunks_per_s': round(a.n / dt, 1),
                            'peak_vram_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20),
                            'peak_vram_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20)})
        print(res['runs'][-1], flush=True)
    best = max((r for r in res['runs'] if not r.get('oom')), key=lambda r: r['chunks_per_s'])
    res['best_batch_size'] = best['batch_size']
    res['extrapolated_hours_all_chunks'] = round(len(texts) / best['chunks_per_s'] / 3600, 2)
    res['note'] = 'extrapolated_hours_all_chunks = n_total_chunks / measured chunks_per_s of the best batch size (not measured end-to-end)'
    out = assert_writable(INDEX_DIR / f'bench_t{a.target}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    return 0


def cmd_build(a):
    import faiss
    import torch
    from scipy.sparse import load_npz, save_npz, vstack
    from r2ai.index.bge_m3 import M3Encoder
    out = assert_writable(index_dir(a.target))
    sh = out / 'shards'
    sh.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(sh / '*.tmp*')):
        os.remove(f)
    texts = load_texts(a.target)
    n = len(texts)
    n_shards = (n + a.shard_size - 1) // a.shard_size
    enc = M3Encoder(max_len=512)
    t0 = time.time()
    done_before = sum((sh / f'{i:05d}.sparse.npz').exists() for i in range(n_shards))
    print(f'{n} chunks, {n_shards} shards, {done_before} already done', flush=True)
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_shards):
        sp = sh / f'{i:05d}.sparse.npz'
        if sp.exists():
            continue
        ts = time.time()
        lo, hi = i * a.shard_size, min(n, (i + 1) * a.shard_size)
        dense, sparse = encode(enc, texts[lo:hi], a.batch_size)
        with open(sh / f'{i:05d}.dense.tmp.npy', 'wb') as f:
            np.save(f, dense)
            f.flush()
            os.fsync(f.fileno())
        os.replace(sh / f'{i:05d}.dense.tmp.npy', sh / f'{i:05d}.dense.npy')
        save_npz(sh / f'{i:05d}.sparse.tmp.npz', to_csr(sparse, enc.vocab))
        os.replace(sh / f'{i:05d}.sparse.tmp.npz', sp)
        el = time.time() - t0
        print(f'shard {i + 1}/{n_shards}: {hi - lo} chunks in {time.time() - ts:.0f}s '
              f'({(hi - lo) / (time.time() - ts):.1f}/s), elapsed {el / 60:.1f} min', flush=True)
    t_embed = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated() / 2**20)
    del enc
    torch.cuda.empty_cache()

    # assemble
    dense = np.lib.format.open_memmap(out / 'dense.tmp.npy', mode='w+', dtype=np.float16, shape=(n, 1024))
    mats = []
    for i in range(n_shards):
        lo = i * a.shard_size
        d = np.load(sh / f'{i:05d}.dense.npy')
        dense[lo:lo + len(d)] = d
        mats.append(load_npz(sh / f'{i:05d}.sparse.npz'))
    dense.flush()
    del dense
    os.replace(out / 'dense.tmp.npy', out / 'dense.npy')
    csr = vstack(mats, format='csr')
    assert csr.shape[0] == n
    save_npz(out / 'sparse.tmp.npz', csr)
    os.replace(out / 'sparse.tmp.npz', out / 'sparse.npz')

    dense = np.load(out / 'dense.npy', mmap_mode='r')
    flat_gb = n * 1024 * 4 / 2**30
    if flat_gb <= a.flat_max_gb:
        index = faiss.IndexFlatIP(1024)
        kind = 'IndexFlatIP'
    else:
        nlist = int(4 * np.sqrt(n))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(1024), 1024, nlist, faiss.METRIC_INNER_PRODUCT)
        rs = np.random.RandomState(42).choice(n, min(n, 50 * nlist), replace=False)
        index.train(np.asarray(dense[np.sort(rs)], dtype=np.float32))
        index.nprobe = 64
        kind = f'IndexIVFFlat(nlist={nlist}, nprobe=64)'
    for b in range(0, n, 50000):
        index.add(np.asarray(dense[b:b + 50000], dtype=np.float32))
    faiss.write_index(index, str(out / 'faiss.tmp.index'))
    os.replace(out / 'faiss.tmp.index', out / 'faiss.index')
    cf = chunks_file(a.target)
    meta = {'model': 'BAAI/bge-m3', 'dense': 'CLS, L2-normalised, fp16', 'sparse': 'relu(sparse_linear) max per token id',
            'max_len': 512, 'fp16': True, 'batch_size': a.batch_size, 'n_chunks': n, 'faiss': kind,
            'faiss_float32_gb': round(flat_gb, 2), 'sparse_nnz': int(csr.nnz), 'chunks_file': data_label(cf),
            'chunks_file_mtime': os.path.getmtime(cf), 'embed_seconds_this_run': round(t_embed),
            'shards_done_before_this_run': done_before, 'peak_vram_allocated_mib_this_run': peak,
            'built_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(meta, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('bench')
    p.add_argument('--target', type=int, default=256)
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--batch-sizes', default='4,8,16,32,64')
    p = sub.add_parser('build')
    p.add_argument('--target', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--shard-size', type=int, default=10000)
    p.add_argument('--flat-max-gb', type=float, default=4.0)
    a = ap.parse_args(argv)
    assert_writable(INDEX_DIR)
    require_inputs(chunks_file(a.target))
    if pq.ParquetFile(chunks_file(a.target)).metadata.num_rows == 0:
        raise ValueError(f'Empty chunk input: {chunks_file(a.target)}')
    return cmd_bench(a) if a.cmd == 'bench' else cmd_build(a)


if __name__ == '__main__':
    sys.exit(main())
