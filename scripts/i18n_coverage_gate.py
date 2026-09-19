#!/usr/bin/env python3
"""Coverage and runtime gates for i18n.

Modes
-----
(default)
    Reads compiled JSON under ``i18n/<lang>/compiled/``.  Exits nonzero when
    per-page coverage falls below 60 % or overall coverage below 75 %.

``--runtime``
    Reads per-route stats files written by the Lua translate filter during a
    language render (``i18n/<lang>/_extracted/<record>.stats.json``).  Exits
    nonzero when any page with ``total > 0`` has ``matched == 0`` (entire page
    rendered English), or when the overall matched/total ratio falls below
    90 %.  Includes unmatched sample strings in the failure output.

``--lang`` selects the tree (default ``es``).  Floors are the same for every
language; a language whose dictionary is genuinely thinner needs its own.

Override with env ``I18N_ALLOW_PARTIAL=1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MIN_PER_PAGE = 0.60
MIN_OVERALL = 0.75
RUNTIME_FLOOR = 0.90

# The compiled-JSON fields carrying a translation. Named `es` for every language
# (it means "the target text") — see i18n/ADDING-A-LANGUAGE.md §4.
TARGET = "es"
TARGET_HTML = "es_html"


# ---------------------------------------------------------------------------
# Default mode: compiled-JSON coverage
# ---------------------------------------------------------------------------

def _run_compiled_gate(root: Path, lang: str, allow_partial: bool) -> None:
    compiled_dir = root / "i18n" / lang / "compiled"

    if not compiled_dir.exists():
        print(
            "ERROR: No compiled i18n files found.\n"
            f"  1. Dump:    quarto render --profile {lang}-dump\n"
            f"  2. Extract: python3 scripts/i18n_extract.py --lang {lang}\n"
            f"  3. Compile: python3 scripts/i18n_extract.py --lang {lang} --compile",
            file=sys.stderr,
        )
        sys.exit(1)

    compiled_files = sorted(compiled_dir.rglob("*.json"))
    if not compiled_files:
        print(
            f"ERROR: No compiled i18n files found in i18n/{lang}/compiled/.\n"
            f"  1. Dump:    quarto render --profile {lang}-dump\n"
            f"  2. Extract: python3 scripts/i18n_extract.py --lang {lang}\n"
            f"  3. Compile: python3 scripts/i18n_extract.py --lang {lang} --compile",
            file=sys.stderr,
        )
        sys.exit(1)

    total_translated = 0
    total_active = 0
    failures: list[str] = []

    for path in compiled_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = str(path.relative_to(compiled_dir))

        n_blocks = len(data.get("blocks", []))
        n_raw = len(data.get("raw_blocks", []))
        n_env = len(data.get("envelope", {}))
        active = n_blocks + n_raw + n_env

        translated = (
            sum(1 for b in data.get("blocks", []) if b.get(TARGET) is not None)
            + sum(
                1
                for r in data.get("raw_blocks", [])
                if r.get(TARGET_HTML) is not None
            )
            + sum(
                1
                for v in data.get("envelope", {}).values()
                if v is not None
            )
        )

        page_cov = translated / active if active > 0 else 1.0
        total_translated += translated
        total_active += active

        if page_cov < MIN_PER_PAGE:
            failures.append(
                f"  {record}: {page_cov:.2%} < {MIN_PER_PAGE:.0%} floor"
            )

    overall = total_translated / total_active if total_active > 0 else 1.0

    if overall < MIN_OVERALL:
        failures.append(
            f"  OVERALL: {overall:.2%} < {MIN_OVERALL:.0%} floor"
        )

    if failures and not allow_partial:
        print("Coverage gate FAILED:", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        sys.exit(1)

    print(f"Coverage gate passed: overall={overall:.2%}")


# ---------------------------------------------------------------------------
# Runtime mode: post-render stats from translate.lua
# ---------------------------------------------------------------------------

def _run_runtime_gate(root: Path, lang: str, allow_partial: bool) -> None:
    stats_dir = root / "i18n" / lang / "_extracted"

    if not stats_dir.exists():
        print(
            f"ERROR: No i18n/{lang}/_extracted/ directory found.\n"
            f"  Run a {lang} render first:  quarto render --profile {lang}",
            file=sys.stderr,
        )
        sys.exit(1)

    stats_files = sorted(stats_dir.rglob("*.stats.json"))
    if not stats_files:
        print(
            f"ERROR: No .stats.json files found in i18n/{lang}/_extracted/.\n"
            f"  Run a {lang} render first:  quarto render --profile {lang}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_matched = 0
    total_missed = 0
    failures: list[str] = []

    for path in stats_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        source = data.get("source", str(path.name))
        matched = data.get("matched", 0)
        # Lua stats: "total" = lookup misses; "matched" = lookup hits
        missed = data.get("total", 0)
        # "unmatched" may be a list or an empty dict from Lua's JSON encoder
        raw_unmatched = data.get("unmatched", [])
        if isinstance(raw_unmatched, dict):
            unmatched: list[str] = list(raw_unmatched.values()) if raw_unmatched else []
        elif isinstance(raw_unmatched, list):
            unmatched = raw_unmatched
        else:
            unmatched = []

        total_matched += matched
        total_missed += missed

        page_lookups = matched + missed
        # Fail: page has translation lookups but zero matched
        if page_lookups > 0 and matched == 0:
            detail = ""
            if unmatched:
                samples = unmatched[:5]
                detail = "\n    unmatched: " + "; ".join(
                    str(s)[:80] for s in samples
                )
            failures.append(
                f"  {source}: 0/{page_lookups} matched"
                f" (entire page rendered English)"
                f"{detail}"
            )

    total_lookups = total_matched + total_missed
    overall_ratio = total_matched / total_lookups if total_lookups > 0 else 1.0

    if overall_ratio < RUNTIME_FLOOR:
        failures.append(
            f"  OVERALL: {total_matched}/{total_lookups}"
            f" = {overall_ratio:.2%} < {RUNTIME_FLOOR:.0%} floor"
        )

    if failures and not allow_partial:
        print("Runtime gate FAILED:", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        sys.exit(1)

    print(
        f"Runtime gate passed: {total_matched}/{total_lookups}"
        f" = {overall_ratio:.2%}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="i18n coverage and runtime gates")
    parser.add_argument(
        "--lang", default="es",
        help="Language directory under i18n/ (default: es)",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Post-render runtime stats gate (reads .stats.json from a language render)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    allow_partial = os.environ.get("I18N_ALLOW_PARTIAL") == "1"

    if args.runtime:
        _run_runtime_gate(root, args.lang, allow_partial)
    else:
        _run_compiled_gate(root, args.lang, allow_partial)


if __name__ == "__main__":
    main()