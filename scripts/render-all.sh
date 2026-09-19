#!/bin/bash
# Full bilingual build: dump → EN → ES.
#
# Usage:
#   bash scripts/render-all.sh          # three-pass bilingual build
#   bash scripts/render-all.sh --dry-run # preflight checks only (no renders)
#
# Prerequisites:
#   - .env present with a real MIMO_API_KEY
#   - conda envs provisioned (bash scripts/setup_envs.sh)
#   - alchemize kernel registered (scripts/check_kernels.py)

set -euo pipefail
cd "$(dirname "$0")/.."

# ── dry-run mode ──────────────────────────────────────────────────────
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── env ───────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# ── MIMO_API_KEY guard ────────────────────────────────────────────────
if [[ -z "${MIMO_API_KEY:-}" || "${MIMO_API_KEY}" == "your-api-key-here" ]]; then
    echo "ERROR: MIMO_API_KEY is unset, empty, or the placeholder value." >&2
    echo "  Set a real key in .env (see .env.example)." >&2
    exit 1
fi

# ── leaked QUARTO_PROFILE guard ───────────────────────────────────────
if [[ -n "${QUARTO_PROFILE:-}" ]]; then
    echo "ERROR: QUARTO_PROFILE is already set to '$QUARTO_PROFILE'." >&2
    echo "  render-all.sh manages profiles internally; unset it first." >&2
    echo "  Run:  unset QUARTO_PROFILE" >&2
    exit 1
fi

# ── kernel preflight ──────────────────────────────────────────────────
python3 scripts/check_kernels.py

if [[ $DRY_RUN == 1 ]]; then
    echo "── dry-run: all preflight checks passed ──"
    exit 0
fi

# ── incomplete-build trap ─────────────────────────────────────────────
# If the script dies between EN pass (which wipes docs/es/) and the
# completed ES pass, the worktree is broken.  Detect this and print an
# actionable message; also remove the stale marker.
_EN_STARTED=0
_ES_DONE=0

_cleanup() {
    local rc=$?
    if [[ $_EN_STARTED == 1 && $_ES_DONE == 0 ]]; then
        echo "" >&2
        echo "INCOMPLETE BILINGUAL BUILD — docs/es/ was removed by the EN" >&2
        echo "pass and not rebuilt; re-run bash scripts/render-all.sh" >&2
        echo "before committing." >&2
        rm -f .i18n-es-built 2>/dev/null
    fi
    exit "$rc"
}
trap _cleanup EXIT INT TERM

# ── helper: wall-time logging ─────────────────────────────────────────
pass_start=""
start_clock() { pass_start=$(date +%s); }
elapsed()     { echo "  ($(($(date +%s) - pass_start))s)"; }

# ── Pass 0: dump (conditional) ───────────────────────────────────────
EXTRACTED="i18n/es/_extracted"
NEED_DUMP=0

if [[ ! -d "$EXTRACTED" ]] || [[ -z "$(ls -A "$EXTRACTED" 2>/dev/null)" ]]; then
    NEED_DUMP=1
else
    # Any .qmd source newer than the newest extracted record?
    NEWEST_EXTRACTED=$(find "$EXTRACTED" -name '*.json' -type f -print0 |
                       xargs -0 stat -f '%m' 2>/dev/null | sort -rn | head -1)
    if [[ -z "$NEWEST_EXTRACTED" ]]; then
        NEED_DUMP=1
    else
        while IFS= read -r -d '' src; do
            src_mtime=$(stat -f '%m' "$src")
            if [[ "$src_mtime" -gt "$NEWEST_EXTRACTED" ]]; then
                NEED_DUMP=1
                break
            fi
        done < <(find . -name '*.qmd' -not -path './.quarto/*' -not -path './_freeze/*' \
                         -not -path './docs/*' -not -path './_hidden*' -print0)
    fi
fi

if [[ "$NEED_DUMP" == "1" ]]; then
    echo "── Pass 0: dump ──"
    start_clock
    env -u QUARTO_PROFILE conda run -n cetagostini_site \
        quarto render --profile es-dump
    echo "  dump complete $(elapsed)"

    # After a dump refresh, the dictionaries are stale: changed prose
    # produces new `match` keys with no `es` value.  Run the extractor
    # in UPDATE mode to merge new/changed blocks (es: null) and move
    # vanished keys to `obsolete`, preserving existing translations.
    echo "── updating dictionary skeletons ──"
    conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang es

    # Report how many entries now need translating.
    # --check exits nonzero when coverage is incomplete; that is NOT a
    # hard failure here — the coverage gate guards the release path.
    CHECK_OUT=$(conda run -n cetagostini_site \
        python3 scripts/i18n_extract.py --lang es --check 2>&1) || true
    echo "$CHECK_OUT"
else
    echo "── Pass 0: dump skipped (extracted records up to date) ──"
fi

# ── Pass 1: English ──────────────────────────────────────────────────
echo "── Pass 1: English ──"
_EN_STARTED=1
start_clock
I18N_RENDER_ALL=1 env -u QUARTO_PROFILE conda run -n cetagostini_site \
    quarto render
echo "  EN complete $(elapsed)"

if [[ ! -f docs/index.html ]]; then
    echo "FATAL: docs/index.html missing after EN pass" >&2
    exit 1
fi

# ── Pass 2: Spanish ──────────────────────────────────────────────────
echo "── Pass 2: Spanish ──"
start_clock
conda run -n cetagostini_site quarto render --profile es
echo "  ES complete $(elapsed)"

if [[ ! -f docs/es/index.html ]]; then
    echo "FATAL: docs/es/index.html missing after ES pass" >&2
    exit 1
fi

_ES_DONE=1
echo "── bilingual build complete ──"