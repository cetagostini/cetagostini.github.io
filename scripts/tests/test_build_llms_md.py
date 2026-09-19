"""Test js/build-llms-md.py — profile dispatch, EN-vs-ES title-suffix strip,
output-dir scoping, llms source selection, and import safety."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("build_llms_md", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module(_ROOT / "js" / "build-llms-md.py")


class TestProfileDispatch(unittest.TestCase):
    def test_empty_profile(self):
        self.assertEqual(_MOD.resolve_lang({}), "en")

    def test_profile_es(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "es"}), "es")

    def test_profile_es_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es-dump"}))

    def test_es_dump_beats_es(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es,es-dump"}))

    def test_profile_pt(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "pt"}), "pt")

    def test_profile_pt_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "pt-dump"}))

    def test_pt_dump_beats_pt(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "pt,pt-dump"}))

    def test_every_language_has_an_index_and_a_title(self):
        for lang in _MOD.LANGS:
            self.assertIn(lang, _MOD.LLMS_SOURCE)
            self.assertTrue((_MOD.ROOT / _MOD.LLMS_SOURCE[lang]).is_file())
            self.assertIn(lang, _MOD.SITE_TITLE)


class TestOutputDir(unittest.TestCase):
    def test_env_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "docs"
            self.assertEqual(
                _MOD.output_dir("en", {"QUARTO_PROJECT_OUTPUT_DIR": str(p)}),
                p.resolve(),
            )

    def test_fallback_es(self):
        self.assertEqual(_MOD.output_dir("es", {}), _MOD.ROOT / "docs" / "es")

    def test_fallback_pt(self):
        self.assertEqual(_MOD.output_dir("pt", {}), _MOD.ROOT / "docs" / "pt")


class TestTargets(unittest.TestCase):
    def test_es_tree_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            es = docs / "es"
            es.mkdir(parents=True)
            (es / "index.html").write_text("<html></html>")
            (es / "about.html").write_text("<html></html>")
            (docs / "index.html").write_text("<html></html>")
            (docs / "about.html").write_text("<html></html>")
            result = _MOD.targets(docs)
            self.assertIn("index.html", result)
            self.assertIn("about.html", result)
            self.assertNotIn("es/index.html", result)
            self.assertNotIn("es/about.html", result)


class TestTitleStrip(unittest.TestCase):
    def _html(self, og_title: str) -> str:
        return f'<html><head><meta property="og:title" content="{og_title}"></head><body></body></html>'

    def test_en_strip(self):
        header = _MOD.mirror_header(
            self._html("My Page — Marketing Science Blog"), "Marketing Science Blog"
        )
        self.assertEqual(header.split("\n")[0], "# My Page")

    def test_es_strip(self):
        header = _MOD.mirror_header(
            self._html("Mi Página — Blog de ciencia del marketing"),
            "Blog de ciencia del marketing",
        )
        self.assertEqual(header.split("\n")[0], "# Mi Página")

    def test_cross_language_no_strip(self):
        header = _MOD.mirror_header(
            self._html("Mi Página — Blog de ciencia del marketing"),
            "Marketing Science Blog",
        )
        self.assertIn("Blog de ciencia del marketing", header)


class TestImportSafety(unittest.TestCase):
    def test_import_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).rglob("*"))
            os.chdir(tmp)
            try:
                _load_module(_ROOT / "js" / "build-llms-md.py")
            finally:
                os.chdir(_ROOT)
            after = set(Path(tmp).rglob("*"))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()