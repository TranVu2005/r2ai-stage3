#!/usr/bin/env bash
# One-time setup on the VPS (Ubuntu 24.04): tmux + python venv + requests. Run as a normal user with sudo.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip tmux ca-certificates
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet 'requests>=2.31'
chmod +x run.sh status.sh pack.sh
.venv/bin/python -c "import requests, sys; print('python', sys.version.split()[0], 'requests', requests.__version__)"
test -s urls.tsv && echo "urls.tsv: $(wc -l < urls.tsv) lines" || echo "WARNING: urls.tsv missing (scp it next to crawler.py)"
