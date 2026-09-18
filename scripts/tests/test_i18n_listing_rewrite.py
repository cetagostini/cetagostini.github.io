"""Test scripts/i18n_listing_rewrite.py — label-only mutation with
data-* bytes unchanged, base64/onclick key preservation, multi-word
category round-trip, structural-mismatch failure, and import safety."""

from __future__ import annotations

import base64
import importlib.util
import os
import tempfile
import unittest
import urllib.parse
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("i18n_listing_rewrite", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module(_ROOT / "scripts" / "i18n_listing_rewrite.py")


# ── helpers ──────────────────────────────────────────────────────────

def _b64_quote(label: str) -> str:
    """Quarto encodes categories as base64(quote(label, safe=''))."""
    return base64.b64encode(urllib.parse.quote(label, safe="").encode()).decode()


def _chip(onclick_key: str, label: str) -> str:
    return (
        f'<div class="listing-category" '
        f"onclick=\"window.quartoListingCategory('{onclick_key}'); return false;\">"
        f"{label}</div>"
    )


def _card(index, href, data_cat, chip_key, chip_label, title, desc):
    return f"""
<div class="quarto-post image-right" data-index="{index}" data-categories="{data_cat}">
<div class="body">
<h3 class="no-anchor listing-title">
<a href="{href}" class="no-external">{title}</a>
</h3>
<div class="listing-categories">
{_chip(chip_key, chip_label)}
</div>
<div class="delink listing-description"><a href="{href}" class="no-external">
<p>{desc}</p>
</a></div>
</div>
<div class="metadata"><a href="{href}" class="no-external"><div class="listing-date">Jan 1, 2026</div></a></div>
</div>
"""


_KEY_DS = _b64_quote("data science")
_KEY_META = _b64_quote("meta")

# Real Quarto layout: sidebar (heading + categories) BEFORE listing container.
FULL_PAGE = f"""
<html lang="en"><head><title>Diary</title></head><body>
<h5 class="quarto-listing-category-title">Categories</h5>
<div class="quarto-listing-category category-default">
<div class="category" data-category="">All <span class="quarto-category-count">(3)</span></div>
<div class="category" data-category="YWk=">ai <span class="quarto-category-count">(2)</span></div>
<div class="category" data-category="{_KEY_DS}">data science <span class="quarto-category-count">(1)</span></div>
<div class="category" data-category="{_KEY_META}">meta <span class="quarto-category-count">(2)</span></div>
</div>
<div class="quarto-listing quarto-listing-container-default" id="listing-listing">
<div class="list quarto-listing-default">
{_card(0, "./diary/2026-01-01.html", _KEY_DS, _KEY_DS, "data science",
       "first entry", "English description.")}
{_card(1, "./diary/2026-02-02.html", _KEY_META, _KEY_META, "meta",
       "second entry", "Another English desc.")}
{_card(2, "./diary/2026-03-03.html", _KEY_META, _KEY_META, "meta",
       "third entry", "No compiled record for this card.")}
</div>
<div class="listing-no-matching d-none">No matching items</div>
</div>
</body></html>
"""

SITE_DICT = {
    "categories": {"data science": "ciencia de datos", "meta": "meta"},
    "ui": {"Categories": "Categorías", "All": "Todos"},
}

RECORDS = {
    "2026-01-01": {"title": "primera entrada", "description": "Descripción en español."},
    "2026-02-02": {"title": "segunda entrada", "description": "Segunda descripción."},
}


class TestBase64RoundTrip(unittest.TestCase):
    """The multi-word category 'data science' round-trips through base64."""

    def test_data_science_encoding(self):
        b64 = _b64_quote("data science")
        self.assertEqual(b64, "ZGF0YSUyMHNjaWVuY2U=")
        decoded = urllib.parse.unquote(base64.b64decode(b64).decode())
        self.assertEqual(decoded, "data science")


class TestCardSpans(unittest.TestCase):
    def test_finds_three_cards(self):
        spans = _MOD.card_spans(FULL_PAGE)
        self.assertEqual(len(spans), 3)
        for i in range(len(spans) - 1):
            self.assertLessEqual(spans[i][1], spans[i + 1][0])


class TestRewrite(unittest.TestCase):
    """Full rewrite: titles and descriptions from records, categories from site.json."""

    def test_titles_translated(self):
        missing = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertIn("primera entrada", patched)
        self.assertIn("segunda entrada", patched)
        self.assertNotIn(">first entry<", patched)
        self.assertNotIn(">second entry<", patched)
        # Card 3 has no dictionary record → title stays English
        self.assertIn("third entry", patched)

    def test_descriptions_translated(self):
        missing = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertIn("Descripción en español.", patched)
        self.assertIn("Segunda descripción.", patched)
        self.assertNotIn("English description.", patched)

    def test_categories_translated(self):
        missing = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertIn("ciencia de datos", patched)

    def test_sidebar_translated(self):
        missing = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertIn("Categorías", patched)
        self.assertIn("Todos", patched)

    def test_data_categories_bytes_unchanged(self):
        """data-categories, onclick keys and data-category values are byte-identical."""
        missing = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertIn(f'data-categories="{_KEY_DS}"', patched)
        self.assertIn(f'data-categories="{_KEY_META}"', patched)
        self.assertIn(f"window.quartoListingCategory('{_KEY_DS}')", patched)
        self.assertIn(f"window.quartoListingCategory('{_KEY_META}')", patched)
        self.assertIn(f'data-category="{_KEY_DS}"', patched)
        self.assertIn(f'data-category="{_KEY_META}"', patched)

    def test_listing_dates_unchanged(self):
        missing = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertIn("Jan 1, 2026", patched)

    def test_idempotent(self):
        missing1 = []
        patched = _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing1)
        missing2 = []
        patched2 = _MOD.rewrite(patched, RECORDS, SITE_DICT, missing2)
        self.assertEqual(patched, patched2)

    def test_missing_entry_reported(self):
        missing = []
        _MOD.rewrite(FULL_PAGE, RECORDS, SITE_DICT, missing)
        self.assertTrue(any("2026-03-03" in m for m in missing))


class TestStructuralMismatch(unittest.TestCase):
    """Structural mismatches (wrong template) must raise, not silently succeed."""

    def test_no_listing_container(self):
        with self.assertRaises(_MOD.StructuralMismatch):
            _MOD.rewrite("<div>no listing here</div>", {}, {}, [])

    def test_no_cards(self):
        page = (
            '<div class="quarto-listing-container-default" id="listing-listing">'
            '<div class="list quarto-listing-default"></div>'
            '<div class="listing-no-matching d-none">X</div></div>'
        )
        with self.assertRaises(_MOD.StructuralMismatch):
            _MOD.rewrite(page, {}, {}, [])

    def test_no_sidebar_heading(self):
        page = """
<div class="quarto-listing-container-default" id="listing-listing">
<div class="list quarto-listing-default">
<div class="quarto-post image-right" data-index="0" data-categories="dGVzdA==">
<div class="body">
<h3 class="no-anchor listing-title"><a href="./diary/2026-01-01.html" class="no-external">title</a></h3>
<div class="listing-categories"><div class="listing-category" onclick="window.quartoListingCategory('dGVzdA=='); return false;">test</div></div>
<div class="delink listing-description"><a href="./diary/2026-01-01.html" class="no-external"><p>desc</p></a></div>
</div>
<div class="metadata"><a href="./diary/2026-01-01.html" class="no-external"><div class="listing-date">Jan 1, 2026</div></a></div>
</div>
</div>
<div class="listing-no-matching d-none">X</div>
</div>
"""
        with self.assertRaises(_MOD.StructuralMismatch):
            _MOD.rewrite(page, {}, {}, [])

    def test_no_sidebar_categories(self):
        """Heading present but no <div class="category"> entries."""
        page = """
<h5 class="quarto-listing-category-title">Categories</h5>
<div class="quarto-listing-container-default" id="listing-listing">
<div class="list quarto-listing-default">
<div class="quarto-post image-right" data-index="0" data-categories="dGVzdA==">
<div class="body">
<h3 class="no-anchor listing-title"><a href="./diary/2026-01-01.html" class="no-external">title</a></h3>
<div class="listing-categories"><div class="listing-category" onclick="window.quartoListingCategory('dGVzdA=='); return false;">test</div></div>
<div class="delink listing-description"><a href="./diary/2026-01-01.html" class="no-external"><p>desc</p></a></div>
</div>
<div class="metadata"><a href="./diary/2026-01-01.html" class="no-external"><div class="listing-date">Jan 1, 2026</div></a></div>
</div>
</div>
<div class="listing-no-matching d-none">X</div>
</div>
"""
        with self.assertRaises(_MOD.StructuralMismatch):
            _MOD.rewrite(page, {}, {}, [])


class TestProfileDispatch(unittest.TestCase):
    def test_empty_profile(self):
        self.assertEqual(_MOD.resolve_lang({}), "en")

    def test_es(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "es"}), "es")

    def test_es_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es-dump"}))


class TestImportSafety(unittest.TestCase):
    def test_import_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).rglob("*"))
            os.chdir(tmp)
            try:
                _load_module(_ROOT / "scripts" / "i18n_listing_rewrite.py")
            finally:
                os.chdir(_ROOT)
            after = set(Path(tmp).rglob("*"))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()