#!/usr/bin/env python3
"""Post-render: build LLM-friendly artifacts (Track A: llms.txt + .md mirrors).

1. Copy the curated root `llms.txt` -> `docs/llms.txt` (served at /llms.txt).
2. For each key page, extract the <main> content and write a clean
   GitHub-Flavored Markdown mirror at `<page>.html.md` (per the llms.txt spec:
   same URL with `.md` appended). Conversion uses the pandoc bundled with Quarto.

Run automatically via `project: post-render` in _quarto.yml, or manually:
    python3 js/build-llms-md.py
"""
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE = "https://cetagostini.github.io"

TARGETS = [
    "index.html",
    "about.html",
    "articles.html",
    "talks.html",
    "diary.html",
]
# Also mirror every article html
for p in sorted((DOCS / "articles").glob("*/*.html")):
    TARGETS.append(str(p.relative_to(DOCS)))
# Also mirror every diary entry html
if (DOCS / "diary").is_dir():
    for p in sorted((DOCS / "diary").glob("*.html")):
        TARGETS.append(str(p.relative_to(DOCS)))

MAIN_RE = re.compile(r"<main\b[^>]*>(.*)</main>", re.S)


def find_pandoc() -> list:
    if shutil.which("pandoc"):
        return ["pandoc"]
    bundled = Path("/Applications/quarto/bin/tools/pandoc")
    if bundled.exists():
        return [str(bundled)]
    return ["quarto", "pandoc"]


def html_main_to_gfm(pandoc: list, html: str) -> str:
    m = MAIN_RE.search(html)
    body = m.group(1) if m else html
    # strip script/style/svg noise that pandoc would otherwise carry as raw
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<svg\b.*?</svg>", "", body, flags=re.S)
    try:
        out = subprocess.run(
            pandoc + ["-f", "html", "-t", "gfm", "--wrap=none"],
            input=body, text=True, capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"  ! pandoc failed: {e.stderr[:200]}")
        return ""
    return out.strip() + "\n"


def main() -> None:
    pandoc = find_pandoc()
    print("Building LLM-friendly artifacts...")

    src = ROOT / "llms.txt"
    if src.exists():
        shutil.copy(src, DOCS / "llms.txt")
        print(f"  llms.txt -> docs/llms.txt")

    for rel in TARGETS:
        html_path = DOCS / rel
        if not html_path.exists():
            continue
        md = html_main_to_gfm(pandoc, html_path.read_text(encoding="utf-8", errors="replace"))
        out_path = DOCS / (rel + ".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"  {rel}.md  ({len(md)} chars)")
    print("Done.")


if __name__ == "__main__":
    main()
