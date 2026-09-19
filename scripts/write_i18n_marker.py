#!/usr/bin/env python3
"""Write the .i18n-<lang>-built marker after a successful language render.

Content: ISO-8601 timestamp + SHA-256 hash of docs/<lang>/index.html.
No-op when QUARTO_PROFILE names no language (the EN pass, a dump pass).

Designed to run as a post-render hook in _quarto.yml (base list, so every pass
sees it and it dispatches on the profile itself).
"""

import hashlib
import os
import sys
from datetime import datetime, timezone

# Languages with a profile, in the order scripts/render-all.sh renders them.
LANGS = ("es", "pt")


def main() -> None:
    tokens = {
        token.strip()
        for token in os.environ.get("QUARTO_PROFILE", "").split(",")
        if token.strip()
    }
    lang = next((lang for lang in LANGS if lang in tokens), None)
    if lang is None:
        return

    index_path = os.path.join(os.getcwd(), "docs", lang, "index.html")
    if not os.path.exists(index_path):
        print(
            f"WARNING: {index_path} not found — marker not written",
            file=sys.stderr,
        )
        return

    sha = hashlib.sha256(open(index_path, "rb").read()).hexdigest()
    ts = datetime.now(timezone.utc).isoformat()

    marker_path = os.path.join(os.getcwd(), f".i18n-{lang}-built")
    with open(marker_path, "w") as f:
        f.write(f"{ts}  sha256:{sha}\n")

    print(f"✓ wrote {marker_path}")


if __name__ == "__main__":
    main()
