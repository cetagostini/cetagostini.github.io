#!/usr/bin/env python3
"""Write the .i18n-es-built marker after a successful ES render.

Content: ISO-8601 timestamp + SHA-256 hash of docs/es/index.html.
No-op when QUARTO_PROFILE does not contain "es" (i.e. during the EN pass
or the dump pass).

Designed to run as a post-render hook in _quarto-es.yml.
"""

import hashlib
import os
import sys
from datetime import datetime, timezone


def main() -> None:
    profile = os.environ.get("QUARTO_PROFILE", "")

    # Only act on the es translate pass — not es-dump, not base.
    if profile != "es":
        return

    index_path = os.path.join(os.getcwd(), "docs", "es", "index.html")
    if not os.path.exists(index_path):
        print(
            f"WARNING: {index_path} not found — marker not written",
            file=sys.stderr,
        )
        return

    sha = hashlib.sha256(open(index_path, "rb").read()).hexdigest()
    ts = datetime.now(timezone.utc).isoformat()

    marker_path = os.path.join(os.getcwd(), ".i18n-es-built")
    with open(marker_path, "w") as f:
        f.write(f"{ts}  sha256:{sha}\n")

    print(f"✓ wrote {marker_path}")


if __name__ == "__main__":
    main()