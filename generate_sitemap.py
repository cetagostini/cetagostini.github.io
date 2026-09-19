#!/usr/bin/env python3
"""Post-render: write the sitemap covering docs/ and every docs/<lang>/ tree.

Written on every *site* pass, English included. Quarto builds its own
single-tree sitemap (`updateSitemap` in quarto.js) before the post-render hooks
run, so a pass that skipped this file would leave that narrower one in place —
that is what a bare `quarto render` used to publish. Each pass covers every
tree that exists at that moment, writes identical bytes to all of them, and
gives each route `xhtml:link` alternates for the languages that have that route.

A route that exists in some trees only is published for the trees that have it,
without alternates for the ones that do not, and reported on stderr. An article
that is rendered in English but not yet translated therefore keeps a valid
English entry instead of blocking the sitemap, and no alternate ever points at
a URL that does not exist. The language passes clear the warning.

Runs from `project: post-render` in _quarto.yml (every pass; it dispatches on
QUARTO_PROFILE itself), or manually after a build:

    QUARTO_PROFILE=pt python3 generate_sitemap.py
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
# Languages with a profile, in the order scripts/render-all.sh renders them.
LANGS = ("es", "pt")
# BCP47 tag per tree — keep in sync with HREFLANG in filters/llm-seo.lua.
HREFLANG = {"en": "en", "es": "es", "pt": "pt-PT"}
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

    A `<lang>-dump` pass writes a disposable extraction tree, not a site.
    """
    tokens = profile_tokens(env)
    if any(f"{lang}-dump" in tokens for lang in LANGS):
        return None
    for lang in LANGS:
        if lang in tokens:
            return lang
    return "en"


def output_dir(lang: str, env: os._Environ | dict) -> Path:
    """The tree this pass wrote.

    Quarto exports QUARTO_PROJECT_OUTPUT_DIR (absolute, resolved) to post-render
    hooks. Outside a render, fall back to the profile's configured directory.
    """
    configured = env.get("QUARTO_PROJECT_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    base = ROOT / EN_DIRNAME
    return base / lang if lang in LANGS else base


def site_url() -> str:
    """Canonical English site URL, from _quarto.yml, always slash-terminated."""
    match = SITE_URL_RE.search((ROOT / "_quarto.yml").read_text(encoding="utf-8"))
    url = match.group(1).strip().strip("\"'") if match else DEFAULT_SITE_URL
    return url if url.endswith("/") else url + "/"


def is_stub(path: Path) -> bool:
    head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    return STUB_RE.search(head) is not None


def tree_routes(tree: Path, skip: list[Path] | None = None) -> set[str]:
    """Every real page in `tree`, as a posix path relative to it.

    `skip` excludes the language trees nested inside the English one — by
    resolved path, so a route never grows a second `es/` (or `pt/`) segment.
    """
    skip = skip or []
    found = set()
    for path in tree.rglob("*.html"):
        if any(path.is_relative_to(other) for other in skip):
            continue
        if is_stub(path):
            continue
        found.add(path.relative_to(tree).as_posix())
    return found


def url_for(base: str, route: str) -> str:
    return base if route == "index.html" else base + route


def tree_base(base: str, lang: str) -> str:
    """URL prefix of one tree: the site root for English, `<lang>/` otherwise."""
    return base if lang == "en" else f"{base}{lang}/"


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


def render_sitemap(
    routes: list[str],
    trees: list[tuple[str, Path]],
    base: str,
    partial: dict[str, list[str]] | None = None,
) -> str:
    """One `<url>` per route per tree that has it, plus that route's alternates.

    `partial` maps a route to the languages that lack it. Those languages get
    neither a `<url>` entry nor an alternate link: the file never advertises a
    URL that does not exist.
    """
    partial = partial or {}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ]
    bases = {lang: tree_base(base, lang) for lang, _ in trees}
    for route in routes:
        missing = set(partial.get(route, ()))
        present = [lang for lang, _ in trees if lang not in missing]
        en_url = url_for(bases["en"], route)
        alternates = [
            f'    <xhtml:link rel="alternate" hreflang="en" href={quoteattr(en_url)}/>',
        ]
        for lang in present:
            if lang == "en":
                continue
            alternates.append(
                f'    <xhtml:link rel="alternate" hreflang={quoteattr(HREFLANG[lang])}'
                f' href={quoteattr(url_for(bases[lang], route))}/>'
            )
        alternates.append(
            f'    <xhtml:link rel="alternate" hreflang="x-default" href={quoteattr(en_url)}/>'
        )
        for lang, tree in trees:
            if lang not in present:
                continue
            lines.append("  <url>")
            lines.append(f"    <loc>{escape(url_for(bases[lang], route))}</loc>")
            lines.append(f"    <lastmod>{lastmod(tree / route)}</lastmod>")
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
    """Point a translated robots.txt at the one sitemap that lists every tree."""
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
        print("Sitemap: skipped (a dump pass writes no site).")
        return 0

    # Resolved on both sides: QUARTO_PROJECT_OUTPUT_DIR is a realpath, and only
    # matching realpaths let tree_routes() recognise (and skip) the nested trees.
    en_dir = (ROOT / EN_DIRNAME).resolve()
    if not en_dir.is_dir():
        print(f"ERROR: {en_dir} is missing — refusing to write a sitemap without the "
              "English tree.", file=sys.stderr)
        return 1

    # Every tree that exists right now, English first. The running pass may
    # write somewhere else entirely (QUARTO_PROJECT_OUTPUT_DIR), so its own tree
    # is taken from output_dir() rather than assumed to be docs/<lang>.
    current = output_dir(lang, os.environ)
    trees: list[tuple[str, Path]] = [("en", en_dir)]
    for other in LANGS:
        tree = current if other == lang else en_dir / other
        if tree.is_dir():
            trees.append((other, tree))

    lang_dirs = [tree for name, tree in trees if name != "en"]
    per_tree = {name: tree_routes(tree, skip=lang_dirs if name == "en" else None)
                for name, tree in trees}
    routes = sorted(set().union(*per_tree.values()))
    if not routes:
        print("ERROR: no rendered pages found — refusing to write an empty sitemap.",
              file=sys.stderr)
        return 1

    # Routes that exist in some trees only: publish them for the trees that have
    # them and say so. Refusing here (the previous behaviour) left the sitemap
    # stuck on whatever Quarto wrote before the hooks — an English-only file —
    # which is exactly the state an untranslated article creates.
    partial: dict[str, list[str]] = {}
    for route in routes:
        missing = [name for name, _ in trees if route not in per_tree[name]]
        if missing:
            partial[route] = missing
    if partial:
        print(f"Sitemap: {len(partial)} route(s) exist in some trees only — "
              "published without the missing alternates:", file=sys.stderr)
        for route, missing in list(partial.items())[:10]:
            print(f"    {route}  (missing in {', '.join(missing)})", file=sys.stderr)
        if len(partial) > 10:
            print(f"    ... and {len(partial) - 10} more", file=sys.stderr)

    base = site_url()
    xml = render_sitemap(routes, trees, base, partial)
    for name, tree in trees:
        write_atomic(tree / "sitemap.xml", xml)
        if name != "en":
            patch_robots(tree / "robots.txt", f"{base}sitemap.xml")

    written = ", ".join(str(tree / "sitemap.xml") for _, tree in trees)
    print(f"Sitemap: {len(routes)} routes x {len(trees)} languages -> {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
