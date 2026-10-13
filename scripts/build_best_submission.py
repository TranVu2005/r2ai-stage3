"""Build the configured best variant (A1-115 by default) and check its JSON / configured ZIP SHA256.

    python scripts/build_best_submission.py [--config configs/submission-best.yaml] [--variant A1-115|VS-append]
                                               [--from-retrieval] [--out X.zip]
"""
import sys

from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.submit.best import main

if __name__ == '__main__':
    raise SystemExit(main(['build', *sys.argv[1:]]))
