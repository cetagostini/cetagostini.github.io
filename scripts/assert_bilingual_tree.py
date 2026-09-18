#!/usr/bin/env python3
"""EN pre-render guard for the bilingual build.

Wired into _quarto.yml as a pre-render hook.  When Quarto runs the base
(no-profile) English pass, a bare `quarto render` would wipe docs/es/
BEFORE any post-render hook could detect it.  This guard runs *before*
the wipe and blocks the render unless the caller has signalled that
a full bilingual build is in progress.

Exit codes:
  0  — safe to proceed (ES profile, fresh clone, or full-build flag set)
  1  — blocked: docs/es exists but caller did not set I18N_RENDER_ALL or I18N_BOOTSTRAP
"""

import os
import sys


def main() -> None:
    profile = os.environ.get("QUARTO_PROFILE", "")

    # ES / es-dump profiles are always allowed — they write INTO docs/es/.
    if "es" in profile:
        sys.exit(0)

    marker = os.path.join(os.getcwd(), ".i18n-es-built")

    # Fresh clone: marker absent → nothing to guard.
    if not os.path.exists(marker):
        sys.exit(0)

    # Marker exists → docs/es has been built previously.
    # Only proceed if a full bilingual build flag is set.
    if os.environ.get("I18N_RENDER_ALL") == "1" or os.environ.get("I18N_BOOTSTRAP") == "1":
        sys.exit(0)

    print(
        "ERROR: docs/es exists — a bare `quarto render` would delete it.\n"
        "  Run  scripts/render-all.sh  for a full bilingual build, or\n"
        "  set  I18N_RENDER_ALL=1  to bypass this guard intentionally.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()