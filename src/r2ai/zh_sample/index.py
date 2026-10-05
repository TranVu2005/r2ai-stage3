"""Existing verbatim chunker and BGE-M3 algorithms, on explicit zh paths only."""
from __future__ import annotations

from r2ai.paths import require_inputs
from .common import RUN, DOCS, CHUNKS, INDEX, preflight, atomic_json, exclusive
import argparse
import hashlib
import json
import os
import re
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compute_processes(snapshot: str, own_pid: int):
    rows = re.findall(r'\|\s*\d+\s+\S+\s+\S+\s+(\d+)\s+(C\+G|C|G)\s+(.+?)\s+(?:N/A|\d+MiB)\s*\|', snapshot)
    return [{'pid': int(pid), 'type': kind, 'name': name.strip()} for pid, kind, name in rows
            if int(pid) != own_pid and (kind == 'C' or (kind == 'C+G' and 'python' in name.lower()))]


def gpu_idle():
    snapshot_text = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout
    if 'Processes:' not in snapshot_text:
        raise RuntimeError('Cannot parse nvidia-smi process table; refusing GPU')
    active = compute_processes(snapshot_text, os.getpid())
    snapshot = {'checked_at': time.time(), 'pid': os.getpid(), 'other_compute_processes': active,
                'nvidia_smi': snapshot_text, 'guard': 'CUDA compute; WDDM desktop C+G processes excluded'}
    atomic_json(RUN / 'gpu_preflight.json', snapshot)
    if active:
        raise RuntimeError('GPU_BUSY: ' + json.dumps(active))
    return snapshot


def chunk():
    from r2ai.index.chunk import main
    preflight()
    require_inputs(DOCS / 'bundle', RUN / 'extract_manifest.json')
    if (INDEX / 'input_manifest.json').exists():
        raise ValueError('Index checkpoint exists; do not change its chunk IDs. Use current chunk files.')
    with exclusive('chunk'):
        return main(['--docs-dir', str(DOCS / 'bundle'), '--out-dir', str(CHUNKS), '--targets', '256'])


def embed():
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.sparse import save_npz
    from r2ai.index.build import encode, to_csr, assemble_index
    from r2ai.index.bge_m3 import M3Encoder
    import torch
    preflight()
    cf = CHUNKS / 'chunks_t256.parquet'
    require_inputs(cf)
    with exclusive('gpu'):
        gpu_idle()
        manifest = {'chunks_sha256': digest(cf), 'seed': 42, 'batch_size': 16, 'shard_size': 1000,
                    'extract_manifest_sha256': digest(RUN / 'extract_manifest.json')}
        old = INDEX / 'input_manifest.json'
        if old.exists() and json.loads(old.read_text('utf-8')) != manifest:
            raise ValueError('Index resume input differs; refusing to mix bundles')
        atomic_json(old, manifest)
        table = pq.read_table(cf, columns=['chunk_id', 'text'])
        ids = table['chunk_id'].to_numpy()
        if not np.array_equal(ids, np.arange(len(ids))) or not len(ids):
            raise ValueError('Chunk IDs must be 0..N-1, nonempty')
        texts = table['text'].to_pylist()
        shards = INDEX / 'shards'
        shards.mkdir(parents=True, exist_ok=True)
        n, size, t0 = len(texts), manifest['shard_size'], time.perf_counter()
        np.random.seed(42)
        torch.manual_seed(42)
        enc = M3Encoder(max_len=512)
        embedded, seconds = 0, 0.0
        try:
            for i, lo in enumerate(range(0, n, size)):
                sparse_path = shards / f'{i:05d}.sparse.npz'
                if sparse_path.exists() and (shards / f'{i:05d}.dense.npy').exists():
                    continue
                gpu_idle()
                ts = time.perf_counter()
                dense, sparse = encode(enc, texts[lo:lo+size], 16)
                with (shards / f'{i:05d}.dense.tmp.npy').open('wb') as f:
                    np.save(f, dense)
                os.replace(shards / f'{i:05d}.dense.tmp.npy', shards / f'{i:05d}.dense.npy')
                save_npz(shards / f'{i:05d}.sparse.tmp.npz', to_csr(sparse, enc.vocab))
                os.replace(shards / f'{i:05d}.sparse.tmp.npz', sparse_path)
                dt = time.perf_counter() - ts
                seconds += dt
                embedded += len(dense)
                with (RUN / 'embed_sessions.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'shard': i, 'chunks': len(dense), 'seconds': dt}) + '\n')
                print(f'zh embed {min(lo+size,n)}/{n}: {dt:.1f}s', flush=True)
        finally:
            del enc
            torch.cuda.empty_cache()
        info = assemble_index(INDEX, n, size, 4, ann=False)
        atomic_json(INDEX / 'meta.json', {'model': 'BAAI/bge-m3', 'max_len': 512, 'fp16': True,
                  'n_chunks': n, **info, 'chunks_file': str(cf), 'input_manifest': manifest,
                  'embed_seconds_this_run': seconds, 'chunks_embedded_this_run': embedded,
                  'wall_seconds_this_run': time.perf_counter() - t0})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=['chunk', 'embed'])
    a = ap.parse_args()
    raise SystemExit((chunk() if a.step == 'chunk' else embed()) or 0)
