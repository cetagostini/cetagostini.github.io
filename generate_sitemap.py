#!/usr/bin/env python3
"""Post-render: write the bilingual sitemap for docs/ and docs/es/.

The sitemap has to name both language trees, so it can only be written once
both exist. The English pass therefore defers and the Spanish pass writes the
combined file (identical bytes) to `docs/sitemap.xml` and `docs/es/sitemap.xml`,
each route carrying `xhtml:link` alternates for en, es and x-default.

Runs from `project: post-render` in _quarto.yml (both passes; it dispatches on
QUARTO_PROFILE itself), or manually after a full bilingual build:

    QUARTO_PROFILE=es python3 generate_sitemap.py
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

ROOT = Path(__file__).resolve().parent
EN_DIRNAME = "docs"
ES_DIRNAME = "es"
DEFAULT_SITE_URL = "https://cetagostini.github.io/"

SITE_URL_RE = re.compile(r"^\s*site-url:\s*(\S+)", re.M)
SITEMAP_LINE_RE = re.compile(r"^Sitemap:.*$", re.M | re.I)
# Quarto renders `aliases:` as a JS redirect page, older Quarto as a meta
# refresh. Neither is content: a sitemap that lists them asks crawlers to index
# a redirect.
STUB_RE = re.compile(r"<title>\s*Redirect\s*</title>|http-equiv=[\"']?refresh", re.I)


def profile_tokens(env: os._Environ | dict) -> set[str]:
    """QUARTO_PROFILE is a comma-separated list; match tokens, never substrings.

    `"es" in profile` would also fire on `es-dump` (and on any profile whose
    name happens to contain those letters).
    """
    return {token.strip() for token in env.get("QUARTO_PROFILE", "").split(",") if token.strip()}


def resolve_lang(env: os._Environ | dict) -> str | None:
    """Language of this render pass, or None when there is nothing to do.

    The `es-dump` pass writes a disposable extraction tree, not a site.
    """
    tokens = profile_tokens(env)
    if "es-dump" in tokens:
        return None
    return "es" if "es" in tokens else "en"


def output_dir(lang: str, env: os._Environ | dict) -> Path:
    """The tree this pass wrote.

    Quarto exports QUARTO_PROJECT_OUTPUT_DIR (absolute, resolved) to post-render
    hooks. Outside a render, fall back to the profile's configured directory.
    """
    configured = env.get("QUARTO_PROJECT_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    base = ROOT / EN_DIRNAME
    return base / ES_DIRNAME if lang == "es" else base


def site_url() -> str:
    """Canonical English site URL, from _quarto.yml, always slash-terminated."""
    match = SITE_URL_RE.search((ROOT / "_quarto.yml").read_text(encoding="utf-8"))
    url = match.group(1).strip().strip("\"'") if match else DEFAULT_SITE_URL
    return url if url.endswith("/") else url + "/"


def is_stub(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    return STUB_RE.search(head) is not None


def tree_routes(tree: Path, skip: Path | None = None) -> set[str]:
    """Every real page in `tree`, as a posix path relative to it.

    `skip` excludes the Spanish tree nested inside the English one — by resolved
    path, so a route never grows a second `es/` segment.
    """
    found = set()
    for path in tree.rglob("*.html"):
        if skip is not None and path.is_relative_to(skip):
            continue
        if is_stub(path):
            continue
        found.add(path.relative_to(tree).as_posix())
    return found


def url_for(base: str, route: str) -> str:
    return base if route == "index.html" else base + route


def priority_for(route: str) -> str:
    if route == "index.html":
        return "1.0"
    if route == "articles.html":
        return "0.9"
    if route == "about.html":
        return "0.8"
    return "0.7"


def changefreq_for(route: str) -> str:
    return "weekly" if route.startswith("articles/") else "monthly"


def lastmod(path: Path) -> str:
    return dt.date.fromtimestamp(path.stat().st_mtime).isoformat()


def render_sitemap(routes: list[str], en_dir: Path, es_dir: Path, en_base: str, es_base: str) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ]
    for route in routes:
        en_url = url_for(en_base, route)
        es_url = url_for(es_base, route)
        alternates = [
            f'    <xhtml:link rel="alternate" hreflang="en" href={quoteattr(en_url)}/>',
            f'    <xhtml:link rel="alternate" hreflang="es" href={quoteattr(es_url)}/>',
            f'    <xhtml:link rel="alternate" hreflang="x-default" href={quoteattr(en_url)}/>',
        ]
        for url, page in ((en_url, en_dir / route), (es_url, es_dir / route)):
            lines.append("  <url>")
            lines.append(f"    <loc>{escape(url)}</loc>")
            lines.append(f"    <lastmod>{lastmod(page)}</lastmod>")
            lines.append(f"    <changefreq>{changefreq_for(route)}</changefreq>")
            lines.append(f"    <priority>{priority_for(route)}</priority>")
            lines.extend(alternates)
            lines.append("  </url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` in one step: a crawler never reads a half-written sitemap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def patch_robots(robots: Path, sitemap_url: str) -> bool:
    """Point the Spanish robots.txt at the one sitemap that lists both trees."""
    if not robots.is_file():
        print(f"  ! {robots} not found — Sitemap line not updated")
        return False
    text = robots.read_text(encoding="utf-8")
    line = f"Sitemap: {sitemap_url}"
    if SITEMAP_LINE_RE.search(text):
        patched = SITEMAP_LINE_RE.sub(line, text)
    else:
        patched = text.rstrip("\n") + f"\n\n{line}\n"
    if patched != text:
        write_atomic(robots, patched)
    return True


def main() -> int:
    lang = resolve_lang(os.environ)
    if lang is None:
        print("Sitemap: skipped (es-dump pass writes no site).")
        return 0

    if lang == "en":
        print("Sitemap: DEFER — the bilingual sitemap is written by the ES pass "
              "(quarto render --profile es).")
        return 0

    es_dir = output_dir("es", os.environ)
    # Resolved on both sides: QUARTO_PROJECT_OUTPUT_DIR is a realpath, and only
    # matching realpaths let tree_routes() recognise (and skip) the nested tree.
    en_dir = (ROOT / EN_DIRNAME).resolve()

    if not en_dir.is_dir():
        print(f"ERROR: {en_dir} is missing — refusing to write a sitemap without the "
              "English tree.", file=sys.stderr)
        return 1

    en_routes = tree_routes(en_dir, skip=es_dir)
    es_routes = tree_routes(es_dir)
    routes = sorted(en_routes | es_routes)
    if not routes:
        print("ERROR: no rendered pages found — refusing to write an empty sitemap.",
              file=sys.stderr)
        return 1

    orphans = [(route, "es" if route in en_routes else "en")
               for route in routes if route not in en_routes or route not in es_routes]
    if orphans:
        print(f"ERROR: {len(orphans)} route(s) exist in one tree only — keeping the "
              "current sitemap rather than publishing alternates that 404:", file=sys.stderr)
        for route, side in orphans[:10]:
            print(f"    {route}  (missing in {side})", file=sys.stderr)
        if len(orphans) > 10:
            print(f"    ... and {len(orphans) - 10} more", file=sys.stderr)
        return 1

    en_base = site_url()
    es_base = f"{en_base}{ES_DIRNAME}/"
    xml = render_sitemap(routes, en_dir, es_dir, en_base, es_base)
    write_atomic(en_dir / "sitemap.xml", xml)
    write_atomic(es_dir / "sitemap.xml", xml)
    patch_robots(es_dir / "robots.txt", f"{en_base}sitemap.xml")

    print(f"Sitemap: {len(routes)} routes x 2 languages -> {en_dir / 'sitemap.xml'} "
          f"and {es_dir / 'sitemap.xml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
