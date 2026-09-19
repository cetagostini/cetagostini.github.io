#!/bin/bash

# Rebuild the multilingual site and serve it locally.
#
# Usage:
#   bash quarto-rebuild.sh          # full multilingual build + local server
#
# The multilingual build (EN → ES → PT) cannot use --clean because re-executing
# articles is expensive and the freeze cache is shared.  To force a full
# re-execution, delete _freeze/ and .quarto/ manually, then re-run.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--clean" ]]; then
    echo "ERROR: --clean is not supported with the multilingual build." >&2
    echo "  The multilingual pipeline re-executes articles; --clean would" >&2
    echo "  wipe the shared freeze cache and force expensive re-runs." >&2
    echo "  To force re-execution:  rm -rf .quarto/ _freeze/" >&2
    exit 1
fi

bash scripts/render-all.sh

echo "Starting local server on http://localhost:8000 ..."
conda run -n cetagostini_site python3 -m http.server --directory docs
