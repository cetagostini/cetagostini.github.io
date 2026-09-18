"""Test generate_sitemap.py — profile dispatch, absolute-path scoping,
bilingual sitemap generation, counterpart-refusal, and import safety."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_sitemap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module(_ROOT / "generate_sitemap.py")


def _es_test_env(root: Path):
    en = root / "docs"
    es = en / "es"
    for d in (en, es):
        d.mkdir(parents=True)
        (d / "index.html").write_text("<html></html>")
        (d / "about.html").write_text("<html></html>")
    (es / "robots.txt").write_text("Sitemap: https://example.com/sitemap.xml\n")
    (root / "_quarto.yml").write_text("website:\n  site-url: https://example.com\n")
    return en, es


def _run_es(main_fn, es: Path):
    os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
    os.environ["QUARTO_PROFILE"] = "es"
    os.environ["QUARTO_PROJECT_OUTPUT_DIR"] = str(es)
    try:
        return main_fn()
    finally:
        os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
        os.environ.pop("QUARTO_PROFILE", None)


class TestProfileDispatch(unittest.TestCase):
    def test_empty_profile(self):
        self.assertEqual(_MOD.resolve_lang({}), "en")

    def test_profile_es(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "es"}), "es")

    def test_profile_es_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es-dump"}))

    def test_es_dump_beats_es(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es,es-dump"}))

    def test_unrelated_profile(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "dark"}), "en")


class TestOutputDir(unittest.TestCase):
    def test_env_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out"
            self.assertEqual(_MOD.output_dir("en", {"QUARTO_PROJECT_OUTPUT_DIR": str(p)}), p.resolve())

    def test_fallback_en(self):
        self.assertEqual(_MOD.output_dir("en", {}), _MOD.ROOT / "docs")

    def test_fallback_es(self):
        self.assertEqual(_MOD.output_dir("es", {}), _MOD.ROOT / "docs" / "es")


class TestSitemapGeneration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self._en, self._es = _es_test_env(self.root)
        self._orig_root = _MOD.ROOT
        _MOD.ROOT = self.root

    def tearDown(self):
        _MOD.ROOT = self._orig_root
        self.tmpdir.cleanup()

    def test_en_pass_defers(self):
        os.environ.pop("QUARTO_PROFILE", None)
        ret = _MOD.main()
        self.assertEqual(ret, 0)
        self.assertFalse((self._en / "sitemap.xml").exists())

    def test_es_pass_writes_both_trees(self):
        ret = _run_es(_MOD.main, self._es)
        self.assertEqual(ret, 0)
        self.assertTrue((self._en / "sitemap.xml").exists())
        self.assertTrue((self._es / "sitemap.xml").exists())
        en_xml = (self._en / "sitemap.xml").read_text()
        es_xml = (self._es / "sitemap.xml").read_text()
        self.assertEqual(en_xml, es_xml)
        self.assertIn("xhtml:link", en_xml)
        self.assertIn('hreflang="es"', en_xml)
        self.assertIn('hreflang="x-default"', en_xml)
        self.assertNotIn("example.com/es/es/", en_xml)

    def test_robots_patched(self):
        _run_es(_MOD.main, self._es)
        robots = (self._es / "robots.txt").read_text()
        self.assertIn("Sitemap: https://example.com/sitemap.xml", robots)

    def test_refuse_when_counterpart_missing(self):
        sentinel = "OLD SITEMAP"
        (self._en / "sitemap.xml").write_text(sentinel)
        (self._es / "about.html").unlink()
        ret = _run_es(_MOD.main, self._es)
        self.assertEqual(ret, 1)
        self.assertEqual((self._en / "sitemap.xml").read_text(), sentinel)

    def test_es_dump_is_noop(self):
        os.environ["QUARTO_PROFILE"] = "es-dump"
        try:
            ret = _MOD.main()
        finally:
            os.environ.pop("QUARTO_PROFILE", None)
        self.assertEqual(ret, 0)

    def test_no_es_es_in_output(self):
        _run_es(_MOD.main, self._es)
        xml = (self._en / "sitemap.xml").read_text()
        self.assertNotIn("/es/es/", xml)


class TestImportSafety(unittest.TestCase):
    def test_import_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).rglob("*"))
            os.chdir(tmp)
            try:
                _load_module(_ROOT / "generate_sitemap.py")
            finally:
                os.chdir(_ROOT)
            after = set(Path(tmp).rglob("*"))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()