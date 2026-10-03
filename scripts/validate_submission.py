"""Compatibility CLI for r2ai.submit.validate_submission."""
from r2ai import paths  # .env must load before ML/HF libraries
from r2ai.submit.validate_submission import main

if __name__ == '__main__':
    raise SystemExit(main())
