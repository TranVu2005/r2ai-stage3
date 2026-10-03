"""Compatibility CLI for r2ai.index.build."""
from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.index.build import main

if __name__ == '__main__':
    raise SystemExit(main())
