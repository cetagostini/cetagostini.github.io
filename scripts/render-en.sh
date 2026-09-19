#!/bin/bash
# English-only build that leaves the committed language trees in place.
#
#   bash scripts/render-en.sh
#
# `scripts/render-all.sh` builds en -> es -> pt and *wants* the English pass to
# wipe docs/<lang>/ — the language passes rebuild those trees from the reviewed
# dictionaries. This script is the opposite trade: publish English changes now,
# keep docs/es and docs/pt exactly as committed, and let the language passes
# pick up the new routes later.
#
# Two things make that safe:
#   * `--no-clean`: without it Quarto removes every docs/<lang>/ before the
#     render. That is what a bare `quarto render` does.
#   * `generate_sitemap.py` runs on this pass too and rewrites the combined
#     sitemap, replacing the single-tree file Quarto writes before the
#     post-render hooks. A route with no counterpart yet is published for the
#     trees that have it, without alternates for the ones that do not, and
#     reported on stderr.
#
# Prerequisites: .env with a real MIMO_API_KEY, conda envs provisioned
# (bash scripts/setup_envs.sh), alchemize kernel registered
# (scripts/check_kernels.py).

set -euo pipefail
cd "$(dirname "$0")/.."

LANGS=(es pt)

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

# ─ leaked QUARTO_PROFILE guard ───────────────────────────────────────
if [[ -n "${QUARTO_PROFILE:-}" ]]; then
    echo "ERROR: QUARTO_PROFILE is already set to '$QUARTO_PROFILE'." >&2
    echo "  This script manages the profile itself; unset it first." >&2
    echo "  Run:  unset QUARTO_PROFILE" >&2
    exit 1
fi

# ── kernel preflight ──────────────────────────────────────────────────
python3 scripts/check_kernels.py

count_files() { find "$1" -type f 2>/dev/null | wc -l | tr -d ' '; }

# Each language tree as it stands, so the run can prove it left them alone.
es_before=$(count_files docs/es)
pt_before=$(count_files docs/pt)

# ─ pass: English ────────────────────────────────────────────────────
# I18N_RENDER_ALL tells the pre-render guard that the trees are not at risk:
# with --no-clean nothing inside docs/<lang>/ is removed.
echo "── English pass (--no-clean; language trees preserved) ──"
start=$(date +%s)
I18N_RENDER_ALL=1 env -u QUARTO_PROFILE conda run -n cetagostini_site \
    quarto render --no-clean
echo "  done ($(($(date +%s) - start))s)"

# ── shared assets the language passes would refresh ───────────────────
# A language pass copies the project stylesheet into its own tree and the
# sitemap hook writes every tree; --no-clean leaves the committed copies in
# place, so bring the stylesheet up to date with this render here.
for lang in "${LANGS[@]}"; do
    if [[ -d "docs/$lang" ]]; then
        cp docs/styles.css "docs/$lang/styles.css"
    fi
done

# ── the trees must still hold exactly what they held ──────────────────
status=0

check_tree() {
    local lang="$1" was="$2" now
    # A tree that was not there before is not this script's business.
    if [[ "$was" == "0" ]]; then
        return 0
    fi
    now=$(count_files "docs/$lang")
    if [[ "$now" != "$was" ]]; then
        echo "ERROR: docs/$lang went from $was to $now files — this render" >&2
        echo "  cleaned a language tree. Restore it before committing:" >&2
        echo "    git checkout -- docs/$lang" >&2
        return 1
    fi
    echo "  docs/$lang intact ($now files)"
}

check_tree es "$es_before" || status=1
check_tree pt "$pt_before" || status=1

if [[ $status != 0 ]]; then
    exit $status
fi

echo "── English-only build complete; docs/es and docs/pt left as committed ─"