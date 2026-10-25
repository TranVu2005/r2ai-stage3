"""S26 P-A: translate the 1,200 Vietnamese queries to Simplified Chinese with a local Ollama model (temperature 0).

Writes <out>/query_zh.parquet (id, query, query_zh); resumes from <out>/query_zh.part.jsonl.
"""
import json, os, sys, time
from pathlib import Path
import pandas as pd, requests
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout

MODEL = sys.argv[2] if len(sys.argv) > 2 else 'qwen2.5:7b'
SHARD, NSHARD = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '0/1').split('/'))
out = Path(sys.argv[1]); part = out / 'query_zh.part.jsonl'
mypart = out / f'query_zh.part{SHARD}of{NSHARD}.jsonl'
SYSTEM = 'You are a professional medical translator. Translate the user\'s Vietnamese medical question into Simplified Chinese. Keep medical terms, drug names, numbers and units accurate. Output only the Chinese translation, nothing else.'
q = pd.read_parquet('data/raw/query.parquet').sort_values('id')
done = {}
for pf in [part, *out.glob('query_zh.part*of*.jsonl')]:
    if pf.exists():
        for line in pf.read_text(encoding='utf-8').splitlines():
            r = json.loads(line); done[r['id']] = r['query_zh']
t0 = time.time()
with mypart.open('a', encoding='utf-8') as f:
    for i, (qid, text) in enumerate(zip(q.id, q['query'])):
        if int(qid) in done or i % NSHARD != SHARD:
            continue
        r = requests.post('http://127.0.0.1:11434/api/chat', json={'model': MODEL, 'stream': False,
                          'options': {'temperature': 0, 'seed': 42, 'num_ctx': 4096},
                          'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]}, timeout=600)
        r.raise_for_status()
        zh = r.json()['message']['content'].strip()
        done[int(qid)] = zh
        f.write(json.dumps({'id': int(qid), 'query_zh': zh}, ensure_ascii=False) + '\n'); f.flush()
        if i % 50 == 0:
            print(i, f'{time.time() - t0:.0f}s', flush=True)
q['query_zh'] = q.id.map(done); q['model'] = MODEL
if q.query_zh.isna().any():
    print('shard done; missing', int(q.query_zh.isna().sum())); sys.exit(0)
q.to_parquet(out / 'query_zh.parquet', index=False)
print('done', time.time() - t0)
