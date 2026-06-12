#!/usr/bin/env bash
# Launch the Stock Analysis Agent — wraps cd + venv activation + python agent.py.
#
# Usage:
#   ./run.sh            # menu mode (default — pick 1–4 from a list)
#   ./run.sh --ask      # LangChain natural-language mode
#
# Works whether invoked as ./run.sh, bash run.sh, or with a full path
# from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Error: .venv/ not found in $SCRIPT_DIR" >&2
  echo "Run setup once:" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python -m textblob.download_corpora" >&2
  exit 1
fi

source .venv/bin/activate
exec python agent.py "$@"
