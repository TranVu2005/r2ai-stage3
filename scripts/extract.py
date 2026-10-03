"""Compatibility CLI for r2ai.extract.cli."""
from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.extract.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
