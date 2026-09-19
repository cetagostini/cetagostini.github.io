#!/bin/bash
# Full multilingual build: dump → EN → every language, in order.
#
# Usage:
#   bash scripts/render-all.sh           # full build
#   bash scripts/render-all.sh --dry-run # preflight checks only (no renders)
#
# Prerequisites:
#   - .env present with a real MIMO_API_KEY
#   - conda envs provisioned (bash scripts/setup_envs.sh)
#   - alchemize kernel registered (scripts/check_kernels.py)

set -euo pipefail
cd "$(dirname "$0")/.."

# Languages with a profile, in render order. English must stay first: its pass
# deletes every docs/<lang>/ subtree before the language passes rebuild them.
LANGS=(es pt)

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
# If the script dies between the EN pass (which wipes every docs/<lang>/) and
# the completed language passes, the worktree is broken.  Detect this and print
# an actionable message; also remove the stale markers.
_EN_STARTED=0
_LANGS_DONE=0

_cleanup() {
    local rc=$?
    if [[ $_EN_STARTED == 1 && $_LANGS_DONE -lt ${#LANGS[@]} ]]; then
        echo "" >&2
        echo "INCOMPLETE MULTILINGUAL BUILD — the EN pass removed the language" >&2
        echo "trees and not every one was rebuilt; re-run" >&2
        echo "bash scripts/render-all.sh before committing." >&2
        rm -f .i18n-*-built 2>/dev/null
    fi
    exit "$rc"
}
trap _cleanup EXIT INT TERM

# ── helper: wall-time logging ─────────────────────────────────────────
pass_start=""
start_clock() { pass_start=$(date +%s); }
elapsed()     { echo "  ($(($(date +%s) - pass_start))s)"; }

# ── helper: dump inputs ───────────────────────────────────────────────
# Everything the runtime AST is derived from. A change to any of these can move
# a block key or an envelope render-id, so the extracted records go stale — not
# just .qmd edits.
source_files() {
    find . \( -name '*.qmd' -o -name '_quarto*.yml' -o -name '*.lua' -o -path './_includes/*' \) \
         -not -path './.quarto/*' -not -path './_freeze/*' -not -path './docs/*' \
         -not -path './_hidden*' -print0
}

# Is `lang`'s extraction dump missing or older than any dump input?
needs_dump() {
    local extracted="i18n/$1/_extracted"
    [[ -d "$extracted" ]] || return 0
    [[ -n "$(ls -A "$extracted" 2>/dev/null)" ]] || return 0

    local newest src
    newest=$(find "$extracted" -name '*.json' -type f -print0 |
             xargs -0 stat -f '%m' 2>/dev/null | sort -rn | head -1)
    [[ -n "$newest" ]] || return 0

    while IFS= read -r -d '' src; do
        if [[ "$(stat -f '%m' "$src")" -gt "$newest" ]]; then
            return 0
        fi
    done < <(source_files)
    return 1
}

# ── Pass 0: dump (per language, conditional) ──────────────────────────
# The records are language-independent (they describe the English AST), but each
# language keeps its own copy under i18n/<lang>/_extracted/ — the stats files
# written during its translate pass live there too.
pass_no=0
for lang in "${LANGS[@]}"; do
    if ! needs_dump "$lang"; then
        echo "── Pass 0 ($lang): dump skipped (extracted records up to date) ──"
        continue
    fi

    echo "── Pass 0 ($lang): dump ──"
    start_clock
    env -u QUARTO_PROFILE conda run -n cetagostini_site \
        quarto render --profile "${lang}-dump"
    echo "  dump complete $(elapsed)"

    # After a dump refresh, the dictionaries are stale: changed prose produces
    # new `match` keys with no translation.  Run the extractor in UPDATE mode to
    # merge new/changed blocks (target: null) and move vanished keys to
    # `obsolete`, preserving existing translations.
    echo "── updating dictionary skeletons ($lang) ──"
    conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang "$lang"

    # Report how many entries now need translating.
    # --check exits nonzero when coverage is incomplete; that is NOT a
    # hard failure here — the coverage gate guards the release path.
    CHECK_OUT=$(conda run -n cetagostini_site \
        python3 scripts/i18n_extract.py --lang "$lang" --check 2>&1) || true
    echo "$CHECK_OUT"
done

# ── Pass 1: English ──────────────────────────────────────────────────
pass_no=$((pass_no + 1))
echo "── Pass $pass_no: English ──"
_EN_STARTED=1
start_clock
I18N_RENDER_ALL=1 env -u QUARTO_PROFILE conda run -n cetagostini_site \
    quarto render
echo "  EN complete $(elapsed)"

if [[ ! -f docs/index.html ]]; then
    echo "FATAL: docs/index.html missing after EN pass" >&2
    exit 1
fi

# ── Passes 2..n: one per language ────────────────────────────────────
for lang in "${LANGS[@]}"; do
    pass_no=$((pass_no + 1))
    echo "── Pass $pass_no: $lang ──"
    start_clock
    conda run -n cetagostini_site quarto render --profile "$lang"
    echo "  $lang complete $(elapsed)"

    if [[ ! -f "docs/$lang/index.html" ]]; then
        echo "FATAL: docs/$lang/index.html missing after the $lang pass" >&2
        exit 1
    fi
    _LANGS_DONE=$((_LANGS_DONE + 1))
done

echo "── multilingual build complete (en + ${LANGS[*]}) ──"
