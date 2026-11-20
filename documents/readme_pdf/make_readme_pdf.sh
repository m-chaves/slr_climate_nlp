#!/usr/bin/env bash
# Build README.pdf from README.md, keeping internal links, images and the Mermaid flowchart.
# Usage (from anywhere): bash documents/readme_pdf/make_readme_pdf.sh [output.pdf]
# Saves the pdf at the directory where you ran the command.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-README.pdf}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac

# Node.js / mermaid-cli installed in the home directory (no sudo needed).
export PATH="$HOME/.local/node/bin:$PATH"

cd "$REPO_ROOT"
pandoc README.md \
  -f markdown+gfm_auto_identifiers \
  --lua-filter=documents/readme_pdf/readme-pdf.lua \
  --pdf-engine=lualatex \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  -o "$OUT"
echo "Wrote $OUT"
