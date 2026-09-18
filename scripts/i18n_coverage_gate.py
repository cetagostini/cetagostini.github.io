#!/usr/bin/env python3
"""Coverage gate for i18n compiled JSON.

Reads compiled JSON under ``i18n/es/compiled/`` and exits nonzero when
per-page coverage falls below 60 % or overall coverage below 75 %.

Override with env ``I18N_ALLOW_PARTIAL=1``.

If NO compiled files exist, exits nonzero with an actionable message
naming the dump + extract commands.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MIN_PER_PAGE = 0.60
MIN_OVERALL = 0.75


def main() -> None:
    root = Path.cwd()
    compiled_dir = root / "i18n" / "es" / "compiled"

    if not compiled_dir.exists():
        print(
            "ERROR: No compiled i18n files found.\n"
            "  1. Dump:    quarto render --profile es-dump\n"
            "  2. Extract: python3 scripts/i18n_extract.py\n"
            "  3. Compile: python3 scripts/i18n_extract.py --compile",
            file=sys.stderr,
        )
        sys.exit(1)

    compiled_files = sorted(compiled_dir.rglob("*.json"))
    if not compiled_files:
        print(
            "ERROR: No compiled i18n files found in i18n/es/compiled/.\n"
            "  1. Dump:    quarto render --profile es-dump\n"
            "  2. Extract: python3 scripts/i18n_extract.py\n"
            "  3. Compile: python3 scripts/i18n_extract.py --compile",
            file=sys.stderr,
        )
        sys.exit(1)

    allow_partial = os.environ.get("I18N_ALLOW_PARTIAL") == "1"

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
            sum(1 for b in data.get("blocks", []) if b.get("es") is not None)
            + sum(
                1
                for r in data.get("raw_blocks", [])
                if r.get("es_html") is not None
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


if __name__ == "__main__":
    main()