"""Retrieval of the best configuration (configs/submission-best.yaml): candidates, then deep rerank. Rerun to resume.

    python scripts/run_best_retrieval.py [--config configs/submission-best.yaml] [--step candidates|rerank|all]
"""
import sys

from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.submit.best import main

if __name__ == '__main__':
    raise SystemExit(main(['retrieve', *sys.argv[1:]]))
