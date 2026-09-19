#!/usr/bin/env python3
"""Post-render: translate the diary listing card labels in docs/<lang>/diary.html.

Quarto builds listing cards from the English sources after the translation
filter has run, so a translated tree ships English titles, descriptions and
category chips. This rewrites those *labels only*.

Everything that encodes a category stays byte-identical: the card's
`data-categories`, each chip's `onclick` key and each sidebar `data-category`
are `base64(urllib.parse.quote(label, safe=""))` of the ENGLISH label, and
quarto-listing.js matches the sidebar key against the cards' decoded
`data-categories`. Re-encoding one side in the target language silently breaks
category filtering, so the keys are never recomputed — only the visible text
between the tags changes. Listing dates are also left alone: Quarto localises
them natively under the profile's `lang`.

Runs first among a language's post-render hooks — js/build-llms-md.py mirrors
this same HTML into diary.html.md, so it has to see the translated labels.

    QUARTO_PROFILE=pt python3 scripts/i18n_listing_rewrite.py
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN_DIRNAME = "docs"
# Languages with a profile, in the order scripts/render-all.sh renders them.
LANGS = ("es", "pt")
PAGE = "diary.html"

CARD_RE = re.compile(r'<div class="quarto-post\b[^>]*>')
LISTING_END_RE = re.compile(r'<div class="listing-no-matching\b')
TITLE_RE = re.compile(
    r'(<h3\b[^>]*class="[^"]*\blisting-title\b[^"]*"[^>]*>\s*<a\b[^>]*>)(.*?)(</a>)', re.S)
DESC_RE = re.compile(
    r'(<div\b[^>]*class="[^"]*\blisting-description\b[^"]*"[^>]*>\s*<a\b[^>]*>\s*<p>)(.*?)(</p>)', re.S)
CHIP_RE = re.compile(r'(<div class="listing-category" onclick="[^"]*">)([^<]*)(</div>)')
SIDEBAR_TITLE_RE = re.compile(r'(<h5 class="quarto-listing-category-title">)([^<]*)(</h5>)')
SIDEBAR_CAT_RE = re.compile(
    r'(<div class="category" data-category="[^"]*">)(.*?)(<span class="quarto-category-count">)', re.S)
HREF_RE = re.compile(r'<a\b[^>]*href="([^"]+)"')


class StructuralMismatch(RuntimeError):
    """The page is not the listing this script knows how to patch."""


def profile_tokens(env) -> set[str]:
    """QUARTO_PROFILE is a comma-separated list; match tokens, never substrings.

    `"es" in profile` would also fire on `es-dump` (and on any profile whose
    name happens to contain those letters).
    """
    return {token.strip() for token in env.get("QUARTO_PROFILE", "").split(",") if token.strip()}


def resolve_lang(env) -> str | None:
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


def output_dir(lang: str, env) -> Path:
    """The tree this pass wrote.

    Quarto exports QUARTO_PROJECT_OUTPUT_DIR (absolute, resolved) to post-render
    hooks. Outside a render, fall back to the profile's configured directory.
    """
    configured = env.get("QUARTO_PROJECT_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    base = ROOT / EN_DIRNAME
    return base / lang if lang in LANGS else base


def compiled_dir(lang: str) -> Path:
    """The compiled dictionaries of `lang` (written by the pre-render hooks)."""
    return ROOT / "i18n" / lang / "compiled"


def load_compiled(path: Path) -> dict:
    """One compiled dictionary, or {} when it is absent or unreadable.

    A missing dictionary leaves those labels English rather than failing the
    render — the coverage gate is what enforces translation completeness.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"  ! unreadable dictionary {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def load_entry_records(directory: Path) -> dict[str, dict]:
    """`meta` of every compiled diary record, keyed by entry name (its date)."""
    records = {}
    for path in sorted(directory.glob("*.json")):
        meta = load_compiled(path).get("meta")
        if isinstance(meta, dict):
            records[path.stem] = meta
    return records


def entry_record(href: str) -> str:
    """Record name behind a card link: ./diary/2026-08-09.html -> 2026-08-09."""
    return Path(href.split("?", 1)[0].split("#", 1)[0]).stem


def card_spans(page: str) -> list[tuple[int, int]]:
    """(start, end) of every listing card, in document order."""
    starts = [m.start() for m in CARD_RE.finditer(page)]
    end = LISTING_END_RE.search(page)
    return list(zip(starts, starts[1:] + [end.start() if end else len(page)]))


def keep_padding(original: str, replacement: str) -> str:
    """Swap the text but keep the whitespace Quarto laid out around it."""
    head = original[: len(original) - len(original.lstrip())]
    tail = original[len(original.rstrip()):]
    return head + html_lib.escape(replacement, quote=False) + tail


def swap_label(current: str, labels: dict[str, str], translated: set[str]) -> str | None:
    """Translated label for `current`, or None when the dictionary has no entry.

    A label that is already translated is returned unchanged, so a second run is
    a silent no-op instead of a flood of "missing entry" reports.
    """
    text = current.strip()
    if text in labels:
        return labels[text]
    if text in translated:
        return text
    return None


def replace_group(text: str, match: re.Match, replacement: str) -> str:
    """Rewrite the label `match` captured in group 2, leaving the tags alone."""
    return text[:match.start(2)] + keep_padding(match.group(2), replacement) + text[match.end(2):]


def rewrite_card(card: str, meta: dict, labels: dict[str, str], translated: set[str],
                 missing: list[str]) -> str:
    """Translate one card's title, description and chips; keys are untouched."""
    title = TITLE_RE.search(card)
    if not title:
        raise StructuralMismatch("listing card without a .listing-title link")
    es_title = str(meta.get("title") or "").strip()
    if "<" in title.group(2):
        # A title carrying inline markup cannot be replaced with plain dictionary
        # text without dropping that markup.
        missing.append(f"title (inline markup): {title.group(2).strip()[:60]}")
    elif es_title:
        card = replace_group(card, title, es_title)
    else:
        missing.append(f"title: {title.group(2).strip()}")

    description = DESC_RE.search(card)
    if description is None and "listing-description" in card:
        raise StructuralMismatch("unexpected .listing-description markup")
    if description is not None:
        es_description = " ".join(str(meta.get("description") or "").split())
        if "<" in description.group(2):
            missing.append(f"description (inline markup): {description.group(2).strip()[:60]}")
        elif es_description:
            card = replace_group(card, description, es_description)
        else:
            missing.append(f"description: {description.group(2).strip()[:60]}")

    chips = list(CHIP_RE.finditer(card))
    if not chips and 'class="listing-category"' in card:
        raise StructuralMismatch("unexpected .listing-category markup")
    return swap_all(card, chips, labels, translated, missing, "category")


def swap_all(text: str, matches: list[re.Match], labels: dict[str, str], translated: set[str],
             missing: list[str], kind: str) -> str:
    """Translate every matched label in one pass; untranslatable ones stay put."""
    out = []
    cursor = 0
    for match in matches:
        label = swap_label(match.group(2), labels, translated)
        if label is None:
            missing.append(f"{kind}: {match.group(2).strip()}")
            continue
        out.append(text[cursor:match.start(2)])
        out.append(keep_padding(match.group(2), label))
        cursor = match.end(2)
    out.append(text[cursor:])
    return "".join(out)


def rewrite_sidebar(page: str, ui: dict[str, str], labels: dict[str, str],
                    translated: set[str], missing: list[str]) -> str:
    """Translate the category filter: every filter label, then its heading."""
    if not SIDEBAR_TITLE_RE.search(page):
        raise StructuralMismatch("no .quarto-listing-category-title heading")
    categories = list(SIDEBAR_CAT_RE.finditer(page))
    if not categories:
        raise StructuralMismatch("no .category filter entries in the sidebar")

    # "All" and the heading are listing chrome, the rest are real categories.
    sidebar_labels = dict(ui)
    sidebar_labels.update(labels)
    page = swap_all(page, categories, sidebar_labels, translated | set(ui.values()),
                    missing, "sidebar category")

    heading = SIDEBAR_TITLE_RE.search(page)
    es_heading = swap_label(heading.group(2), ui, set(ui.values()))
    if es_heading is None:
        missing.append(f"listing heading: {heading.group(2).strip()}")
        return page
    return replace_group(page, heading, es_heading)


def rewrite(page: str, records: dict[str, dict], site: dict, missing: list[str]) -> str:
    """Translate every label in the listing page; keys and dates stay as they are."""
    if "quarto-listing" not in page:
        raise StructuralMismatch("no .quarto-listing container")
    spans = card_spans(page)
    if not spans:
        raise StructuralMismatch("no .quarto-post listing cards")

    labels = {str(k): str(v) for k, v in (site.get("categories") or {}).items()}
    ui = {str(k): str(v) for k, v in (site.get("ui") or {}).items()}
    translated = set(labels.values())

    for start, end in reversed(spans):  # back to front: earlier spans stay valid
        card = page[start:end]
        link = HREF_RE.search(card)
        if not link:
            raise StructuralMismatch("listing card without a link")
        meta = records.get(entry_record(link.group(1)), {})
        if not meta:
            missing.append(f"entry: {link.group(1)}")
        page = page[:start] + rewrite_card(card, meta, labels, translated, missing) + page[end:]

    return rewrite_sidebar(page, ui, labels, translated, missing)


def main() -> int:
    lang = resolve_lang(os.environ)
    if lang is None:
        print("Listing labels: skipped (a dump pass writes no site).")
        return 0
    if lang == "en":
        print("Listing labels: skipped (the English listing needs no rewrite).")
        return 0

    target = output_dir(lang, os.environ) / PAGE
    if ".quarto" in target.parts:
        print(f"ERROR: refusing to patch a build intermediate: {target}", file=sys.stderr)
        return 1
    if not target.is_file():
        print(f"ERROR: {target} not found — the {lang} diary listing is missing.",
              file=sys.stderr)
        return 1

    page = target.read_text(encoding="utf-8")
    missing: list[str] = []
    compiled = compiled_dir(lang)
    try:
        patched = rewrite(page, load_entry_records(compiled / "diary"),
                          load_compiled(compiled / "site.json"), missing)
    except StructuralMismatch as exc:
        print(f"ERROR: {target} is not the listing this script knows ({exc}) — "
              "the page was left untouched.", file=sys.stderr)
        return 1

    if patched != page:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(patched, encoding="utf-8")
        os.replace(tmp, target)

    if missing:
        print(f"Listing labels: {len(missing)} label(s) left in English (no dictionary entry):")
        for item in missing:
            print(f"    {item}")
    print(f"Listing labels: {target} "
          f"{'rewritten' if patched != page else 'already current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
