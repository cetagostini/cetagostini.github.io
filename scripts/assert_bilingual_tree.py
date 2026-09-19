#!/usr/bin/env python3
"""EN pre-render guard for the multilingual build.

Wired into _quarto.yml as a pre-render hook.  When Quarto runs the base
(no-profile) English pass, a bare `quarto render` would wipe every docs/<lang>/
tree BEFORE any post-render hook could detect it.  This guard runs *before*
the wipe and blocks the render unless the caller has signalled that a full
multilingual build is in progress.

Exit codes:
  0  — safe to proceed (language profile, fresh clone, or full-build flag set)
  1  — blocked: a built language tree exists but the caller did not set
       I18N_RENDER_ALL or I18N_BOOTSTRAP
"""

import os
import sys

# Languages with a profile, in the order scripts/render-all.sh renders them.
LANGS = ("es", "pt")


def main() -> None:
    profile = os.environ.get("QUARTO_PROFILE", "")
    tokens = {token.strip() for token in profile.split(",") if token.strip()}

    # Language and dump profiles are always allowed — they write INTO docs/<lang>/.
    if any(lang in tokens or f"{lang}-dump" in tokens for lang in LANGS):
        sys.exit(0)

    cwd = os.getcwd()

    # A marker records that this worktree built the tree; the tree itself may
    # already be gone (interrupted build, manual deletion), in which case the
    # marker is stale and there is nothing left to guard.
    at_risk = []
    for lang in LANGS:
        marker = os.path.join(cwd, f".i18n-{lang}-built")
        if not os.path.exists(marker):
            continue  # fresh clone: nothing to guard
        if not os.path.exists(os.path.join(cwd, "docs", lang, "index.html")):
            print(
                f"NOTE: {marker} present but docs/{lang}/ is already missing"
                " — the tree needs rebuilding via scripts/render-all.sh",
                file=sys.stderr,
            )
            continue
        at_risk.append(lang)

    if not at_risk:
        sys.exit(0)

    # Both marker AND tree exist → a bare render would wipe the tree. Only
    # proceed if a full-build flag is set.
    if os.environ.get("I18N_RENDER_ALL") == "1" or os.environ.get("I18N_BOOTSTRAP") == "1":
        sys.exit(0)

    trees = ", ".join(f"docs/{lang}" for lang in at_risk)
    print(
        f"ERROR: {trees} exists — a bare `quarto render` would delete it.\n"
        "  Run  scripts/render-all.sh  for a full multilingual build,\n"
        "  run  scripts/render-en.sh   for an English-only build that keeps it, or\n"
        "  set  I18N_RENDER_ALL=1  to bypass this guard intentionally.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
