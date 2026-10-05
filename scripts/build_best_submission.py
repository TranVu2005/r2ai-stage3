"""Build the best submission (configs/submission-best.yaml) and check its JSON SHA256 against the uploaded file.

    python scripts/build_best_submission.py [--config configs/submission-best.yaml] [--from-retrieval] [--out X.zip]
"""
import sys

from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.submit.best import main

if __name__ == '__main__':
    raise SystemExit(main(['build', *sys.argv[1:]]))
