"""Validate VS_append.zip against the NEW docs_vi bundle and record the result in stats.json."""
from __future__ import annotations

import contextlib
import io
import json

from r2ai.paths import CORPUS_FILE, QUERY_FILE, ROOT
from r2ai.selective.common import OUT
from r2ai.submit.validate_submission import main as validate


def main():
    out = OUT / 'VS-append'
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        rc = validate([str(out / 'VS_append.zip'), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE),
                       '--docs-dir', str(ROOT / 'data/docs_vi'), '--max-zip-mib', '100'])
    (out / 'validator.log').write_text(cap.getvalue(), encoding='utf-8')
    st = json.loads((out / 'stats.json').read_text(encoding='utf-8'))
    st['validator_exit'] = rc
    st['validator'] = cap.getvalue().strip()[:800]
    (out / 'stats.json').write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding='utf-8')
    print(rc, st['validator'])


if __name__ == '__main__':
    main()
