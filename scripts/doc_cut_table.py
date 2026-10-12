import json, sys, subprocess, numpy as np
from pathlib import Path
rows = []
for v in sys.argv[1:]:
    d = Path('out/runs/doc-cut') / v
    b = json.loads((d / 'build.json').read_text())
    n = []
    with open(d / f'sub_{v}.json', encoding='utf-8') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if line.startswith('{'):
                n.append(len(json.loads(line)['relevant_docs']))
    n = np.array(n); c = b['doc_cut']
    r = subprocess.run([sys.executable, '-m', 'r2ai.submit.validate_submission', str(d / f'sub_{v}.zip'), '--skip-near', '--docs-dir', 'D:/GitHub/r2ai-stage3/data/docs_vi'], capture_output=True, text=True)
    print(v, 'rc', r.returncode, (r.stdout + r.stderr).strip().splitlines()[-3:])
    print(json.dumps(dict(v=v, tau=b['tau'], final_mean=round(n.mean(), 2), p10=np.percentile(n, 10), p50=np.percentile(n, 50), p90=np.percentile(n, 90), prim=c['primary_after_mean'], prot=c['protected_docs_mean'],
        prim_p=[c['p10'], c['p50'], c['p90']], floor=c['queries_logit_rule_at_floor'], ceil=c['queries_logit_rule_at_ceil'], at50=c['queries_at_protect_min'],
        zip=b['zip_mib'], sha=b['zip_sha256'][:12], jsha=b['json_sha256'][:12])))
