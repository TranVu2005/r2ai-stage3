"""Write a submission file: one valid JSON array, one query per line."""
from __future__ import annotations

from r2ai.paths import ROOT

import json
from pathlib import Path


def format_submission(rows) -> str:
    lines = [json.dumps(r, ensure_ascii=False, separators=(', ', ': ')) for r in sorted(rows, key=lambda r: r['id'])]
    return '[\n' + ',\n'.join(lines) + '\n]\n' if lines else '[\n]\n'


def write_submission(rows, path) -> None:
    with open(Path(path), 'w', encoding='utf-8', newline='\n') as f:
        f.write(format_submission(rows))
