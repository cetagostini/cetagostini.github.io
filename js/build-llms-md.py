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
OG_TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]*)"')
TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.S)
CANON_RE = re.compile(r'<link rel="canonical" href="([^"]*)"')
DESC_RE = re.compile(r'<meta name="description" content="([^"]*)"')
AUTHOR_RE = re.compile(r'<meta name="author" content="([^"]*)"')
DATE_RE = re.compile(r'<meta name="dcterms\.date" content="([^"]*)"')


def strip_element(html: str, marker: str) -> str:
    """Drop the element carrying `marker` together with its matching close tag.

    Quarto renders the title block as a <header>, and it nests elements of its
    own, so the element name is read off the tag that carries the marker and the
    depth is balanced by counting that tag.
    """
    i = html.find(marker)
    if i == -1:
        return html
    start = html.rfind("<", 0, i)
    if start == -1:
        return html
    m = re.match(r"<([A-Za-z][A-Za-z0-9-]*)", html[start:])
    if not m:
        return html
    tag = m.group(1)
    depth = 0
    for t in re.finditer(rf"<{tag}\b|</{tag}\s*>", html[start:], re.I):
        depth += -1 if t.group(0).startswith("</") else 1
        if depth == 0:
            return html[:start] + html[start + t.end():]
    return html


def mirror_header(html: str) -> str:
    """Front matter for the mirror: clean H1 + description + canonical URL.

    The visible title block is stripped from the body, so the page title has to
    be reintroduced here (from og:title, minus the site/author suffix).
    """
    m = OG_TITLE_RE.search(html) or TITLE_RE.search(html)
    title = ""
    if m:
        title = re.sub(r"\s*[–—-]\s*(Marketing Science Blog|Carlos Trujillo)\s*$", "", m.group(1)).strip()
    d = DESC_RE.search(html)
    c = CANON_RE.search(html)
    authors = [a.strip() for a in AUTHOR_RE.findall(html) if a.strip()]
    date = DATE_RE.search(html)
    lines = []
    if title:
        lines.append(f"# {title}")
    if d and d.group(1).strip():
        lines += ["", f"> {d.group(1).strip()}"]
    byline = "By " + ", ".join(authors) if authors else ""
    if date and date.group(1).strip():
        byline = f"{byline} · {date.group(1).strip()}" if byline else date.group(1).strip()
    if byline:
        lines += ["", byline]
    if c:
        lines += ["", f"Source: {c.group(1)}"]
    return "\n".join(lines) + "\n\n" if lines else ""


def find_pandoc() -> list:
    if quarto := shutil.which("quarto"):
        return [quarto, "pandoc"]
    bundled = Path("/Applications/quarto/bin/tools/pandoc")
    if bundled.exists():
        return [str(bundled)]
    if pandoc := shutil.which("pandoc"):
        return [pandoc]
    raise FileNotFoundError("No Pandoc executable found")


def html_main_to_gfm(pandoc: list, html: str) -> str:
    m = MAIN_RE.search(html)
    body = m.group(1) if m else html

    # Drop generated chrome that is not prose: the title block (title, author,
    # category chips plus the "Show All Code" / "View Source" code-tools menu)
    # and the skip link. The title comes back via mirror_header().
    # NB: do not strip <button>. On this site buttons carry real content — the
    # talks cards and the About career-rail roles — so removing them emptied
    # those mirrors.
    body = strip_element(body, 'id="title-block-header"')
    body = re.sub(r'<a\b[^>]*class="[^"]*skip-link[^"]*"[^>]*>.*?</a>', "", body, flags=re.S)
    body = re.sub(r'<a\b[^>]*href="javascript:[^"]*"[^>]*>.*?</a>', "", body, flags=re.S)

    # strip script/style/svg noise that pandoc would otherwise carry as raw
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<svg\b.*?</svg>", "", body, flags=re.S)

    # Drop lazy-loading attributes so pandoc renders <img> as Markdown instead of
    # falling back to raw HTML for an attribute Markdown cannot express.
    body = re.sub(r'\s+loading="lazy"', "", body)

    # Unwrap layout containers and interactive wrappers. Quarto wraps each
    # section in <section> and each cell in <div>, and pandoc re-emits those (and
    # <button>, which Markdown cannot express) verbatim as raw HTML in the
    # mirror, burying the prose. Only the tags go, their contents stay — so the
    # talk cards and career roles survive as text.
    body = re.sub(r"</?(?:div|section|nav|aside|header|footer|button|span)\b[^>]*>", "", body)

    try:
        out = subprocess.run(
            pandoc + ["-f", "html", "-t", "gfm", "--wrap=none"],
            input=body, text=True, capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        print(f"  ! pandoc failed: {e.stderr[:200]}")
        return ""
    text = out.strip()
    if not text:
        return ""
    return mirror_header(html) + text + "\n"


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
        if not md.strip():
            # Redirect stubs and other pages without <main> prose yield an empty
            # mirror; write nothing rather than leaving a blank file to fetch.
            if out_path.exists():
                out_path.unlink()
            print(f"  {rel}.md  (skipped — no content)")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"  {rel}.md  ({len(md)} chars)")
    print("Done.")


if __name__ == "__main__":
    main()
