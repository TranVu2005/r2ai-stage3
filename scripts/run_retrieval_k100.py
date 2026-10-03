"""Compatibility CLI for r2ai.retrieve.run_retrieval_k100."""
from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.retrieve.run_retrieval_k100 import main

if __name__ == '__main__':
    raise SystemExit(main())
