#!/usr/bin/env bash
set -euo pipefail
if [[ "${OS:-}" == "Windows_NT" ]]; then
  powershell.exe -NoProfile -File "$(dirname "$0")/setup.ps1"
else
  echo "The frozen runtime is native Windows/Python 3.12.13/cu128; Linux/WSL setup requires its own validated freeze." >&2
  exit 2
fi
