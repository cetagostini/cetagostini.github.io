"""Test generate_sitemap.py — profile dispatch, absolute-path scoping,
bilingual sitemap generation, partial-route publishing, and import safety."""

from __future__ import annotations

import importlib.util
import os
import shutil
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


def _es_test_env(root: Path, langs=("es",)):
    en = root / "docs"
    trees = [en] + [en / lang for lang in langs]
    for d in trees:
        d.mkdir(parents=True)
        (d / "index.html").write_text("<html></html>")
        (d / "about.html").write_text("<html></html>")
    for d in trees[1:]:
        (d / "robots.txt").write_text("Sitemap: https://example.com/sitemap.xml\n")
    (root / "_quarto.yml").write_text("website:\n  site-url: https://example.com\n")
    return en, en / langs[0]


def _run_lang(main_fn, lang: str, tree: Path):
    os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
    os.environ["QUARTO_PROFILE"] = lang
    os.environ["QUARTO_PROJECT_OUTPUT_DIR"] = str(tree)
    try:
        return main_fn()
    finally:
        os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
        os.environ.pop("QUARTO_PROFILE", None)


def _run_es(main_fn, es: Path):
    return _run_lang(main_fn, "es", es)


def _url_block(xml: str, loc: str) -> str:
    """The `<url>` entry whose `<loc>` is `loc`, or "" when there is none."""
    for block in xml.split("  <url>")[1:]:
        body = block.split("</url>")[0]
        if f"<loc>{loc}</loc>" in body:
            return body
    return ""


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

    def test_profile_pt(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "pt"}), "pt")

    def test_profile_pt_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "pt-dump"}))

    def test_pt_dump_beats_pt(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "pt,pt-dump"}))


class TestOutputDir(unittest.TestCase):
    def test_env_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out"
            self.assertEqual(_MOD.output_dir("en", {"QUARTO_PROJECT_OUTPUT_DIR": str(p)}), p.resolve())

    def test_fallback_en(self):
        self.assertEqual(_MOD.output_dir("en", {}), _MOD.ROOT / "docs")

    def test_fallback_es(self):
        self.assertEqual(_MOD.output_dir("es", {}), _MOD.ROOT / "docs" / "es")

    def test_fallback_pt(self):
        self.assertEqual(_MOD.output_dir("pt", {}), _MOD.ROOT / "docs" / "pt")


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

    def test_en_pass_replaces_quartos_single_tree_sitemap(self):
        # Quarto writes its own single-tree sitemap before the post-render hooks
        # run. This pass has to overwrite it, or an English-only build publishes
        # a sitemap that omits every language tree.
        (self._en / "sitemap.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n<urlset>\n'
            "  <url><loc>https://example.com/index.html</loc></url>\n</urlset>\n"
        )
        os.environ.pop("QUARTO_PROFILE", None)
        self.assertEqual(_MOD.main(), 0)
        en_xml = (self._en / "sitemap.xml").read_text()
        self.assertIn('hreflang="es"', en_xml)
        self.assertIn("<loc>https://example.com/es/about.html</loc>", en_xml)
        self.assertEqual(en_xml, (self._es / "sitemap.xml").read_text())

    def test_en_pass_without_language_trees(self):
        shutil.rmtree(self._es)
        os.environ.pop("QUARTO_PROFILE", None)
        self.assertEqual(_MOD.main(), 0)
        xml = (self._en / "sitemap.xml").read_text()
        self.assertIn("<loc>https://example.com/about.html</loc>", xml)
        self.assertIn('hreflang="en"', xml)
        self.assertNotIn('hreflang="es"', xml)

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

    def test_partial_route_omits_missing_alternates(self):
        # An untranslated route: rendered in English, absent from the es tree.
        (self._es / "about.html").unlink()
        self.assertEqual(_run_es(_MOD.main, self._es), 0)
        xml = (self._en / "sitemap.xml").read_text()

        orphan = _url_block(xml, "https://example.com/about.html")
        self.assertTrue(orphan, "the untranslated route is still published")
        self.assertIn('hreflang="en"', orphan)
        self.assertIn('hreflang="x-default"', orphan)
        self.assertNotIn('hreflang="es"', orphan)

        complete = _url_block(xml, "https://example.com/")
        self.assertIn('hreflang="es"', complete)
        self.assertNotIn("example.com/es/about.html", xml)

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


class TestThreeLanguageSitemap(unittest.TestCase):
    """The pt pass rewrites the sitemap for every tree that exists."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self._en, self._es = _es_test_env(self.root, langs=("es", "pt"))
        self._pt = self._en / "pt"
        self._orig_root = _MOD.ROOT
        _MOD.ROOT = self.root

    def tearDown(self):
        _MOD.ROOT = self._orig_root
        self.tmpdir.cleanup()

    def test_pt_pass_lists_every_tree(self):
        ret = _run_lang(_MOD.main, "pt", self._pt)
        self.assertEqual(ret, 0)

        xmls = [
            (tree / "sitemap.xml").read_text()
            for tree in (self._en, self._es, self._pt)
        ]
        self.assertEqual(len(set(xmls)), 1, "every tree gets identical bytes")
        xml = xmls[0]
        for tag in ('hreflang="en"', 'hreflang="es"', 'hreflang="pt-PT"',
                    'hreflang="x-default"'):
            self.assertIn(tag, xml)
        # One <url> per route per tree, and no doubled prefixes.
        self.assertEqual(xml.count("<loc>"), 6)
        self.assertIn("<loc>https://example.com/pt/about.html</loc>", xml)
        self.assertNotIn("/pt/pt/", xml)

    def test_es_pass_ignores_a_tree_that_does_not_exist(self):
        shutil.rmtree(self._pt)
        ret = _run_lang(_MOD.main, "es", self._es)
        self.assertEqual(ret, 0)
        xml = (self._en / "sitemap.xml").read_text()
        self.assertIn('hreflang="es"', xml)
        self.assertNotIn('hreflang="pt-PT"', xml)
        self.assertFalse((self._pt / "sitemap.xml").exists())

    def test_pt_robots_points_at_the_one_sitemap(self):
        _run_lang(_MOD.main, "pt", self._pt)
        self.assertIn(
            "Sitemap: https://example.com/sitemap.xml",
            (self._pt / "robots.txt").read_text(),
        )


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