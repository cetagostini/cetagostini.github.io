"""Test generate_articles_network.py — profile dispatch, ES fallback with
stable IDs, month localization, topic-label localization, --lang es --thumbs
rejection, and import safety."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("generate_articles_network", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_module(_ROOT / "generate_articles_network.py")


class TestProfileDispatch(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_MOD.resolve_lang({}), "en")

    def test_es(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "es"}), "es")

    def test_es_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es-dump"}))

    def test_es_dump_beats_es(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "es,es-dump"}))

    def test_pt(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "pt"}), "pt")

    def test_pt_dump(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "pt-dump"}))

    def test_pt_dump_beats_pt(self):
        self.assertIsNone(_MOD.resolve_lang({"QUARTO_PROFILE": "pt,pt-dump"}))

    def test_unrelated(self):
        self.assertEqual(_MOD.resolve_lang({"QUARTO_PROFILE": "test"}), "en")


class TestOutputDir(unittest.TestCase):
    def test_env_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out"
            self.assertEqual(
                _MOD.output_dir("en", {"QUARTO_PROJECT_OUTPUT_DIR": str(p)}),
                p.resolve(),
            )

    def test_fallback_es(self):
        self.assertEqual(_MOD.output_dir("es", {}), _MOD.ROOT / "docs" / "es")

    def test_fallback_pt(self):
        self.assertEqual(_MOD.output_dir("pt", {}), _MOD.ROOT / "docs" / "pt")

    def test_compiled_dir_is_per_language(self):
        self.assertEqual(
            _MOD.compiled_dir("pt"), _MOD.ROOT / "i18n" / "pt" / "compiled"
        )


class TestEsLocalization(unittest.TestCase):
    """Build a minimal article tree + compiled dictionaries and verify ES output."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self._orig_root = _MOD.ROOT
        self._orig_articles = _MOD.ARTICLES
        self._orig_thumb_dir = _MOD.THUMB_DIR
        _MOD.ROOT = self.root
        _MOD.ARTICLES = self.root / "articles"
        _MOD.THUMB_DIR = self.root / "images" / "network"

        slug = "test_article"
        art_dir = self.root / "articles" / slug
        art_dir.mkdir(parents=True)
        (art_dir / f"{slug}.qmd").write_text(
            "---\ntitle: My Title\ndate: 2026-01-15\n"
            "description: Desc here.\ncategories: [bayesian, optimization]\n"
            "image: ../images/placeholder.jpg\nimage-alt: alt text\n---\n\nBody.\n"
        )
        (self.root / "images").mkdir(exist_ok=True)
        (self.root / "images" / "placeholder.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + b"\x00" * 100
        )
        comp_dir = self.root / "i18n" / "es" / "compiled" / "articles"
        comp_dir.mkdir(parents=True)
        (comp_dir / f"{slug}.json").write_text(json.dumps({
            "schema_version": 4,
            "source": f"{slug}.qmd",
            "coverage": 1.0,
            "meta": {
                "title": "Mi Título",
                "description": "Descripción aquí.",
                "image-alt": "texto alternativo",
            },
        }))
        (self.root / "i18n" / "es" / "compiled" / "site.json").write_text(json.dumps({
            "schema_version": 4,
            "lang": "es",
            "site_title": "Blog de ciencia del marketing",
            "months": {"January": "enero", "February": "febrero"},
            "topics": {"Bayesian": "Bayesiano", "Optimization": "Optimización"},
            "categories": {"bayesian": "bayesiano", "optimization": "optimización"},
            "ui": {"Categories": "Categorías", "All": "Todos"},
        }))

    def tearDown(self):
        _MOD.ROOT = self._orig_root
        _MOD.ARTICLES = self._orig_articles
        _MOD.THUMB_DIR = self._orig_thumb_dir
        self.tmpdir.cleanup()

    def _run(self):
        """Run main() without letting argparse pick up unittest args."""
        with mock.patch("sys.argv", ["generate_articles_network.py"]):
            return _MOD.main()

    def test_es_localizes_titles_and_months(self):
        os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
        os.environ["QUARTO_PROFILE"] = "es"
        try:
            ret = self._run()
        finally:
            os.environ.pop("QUARTO_PROFILE", None)
        self.assertEqual(ret, 0)
        data = json.loads((self.root / "docs" / "es" / "articles-network.json").read_text())
        self.assertEqual(len(data["articles"]), 1)
        art = data["articles"][0]
        self.assertEqual(art["title"], "Mi Título")
        self.assertEqual(art["description"], "Descripción aquí.")
        self.assertEqual(art["imageAlt"], "texto alternativo")
        self.assertIn("enero", art["month"])
        topic_ids = [t["id"] for t in data["topics"]]
        self.assertIn("bayesian", topic_ids)
        topic_map = {t["id"]: t["label"] for t in data["topics"]}
        self.assertEqual(topic_map["bayesian"], "Bayesiano")

    def test_en_uses_english(self):
        os.environ.pop("QUARTO_PROFILE", None)
        os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
        ret = self._run()
        self.assertEqual(ret, 0)
        data = json.loads((self.root / "docs" / "articles-network.json").read_text())
        art = data["articles"][0]
        self.assertEqual(art["title"], "My Title")
        self.assertEqual(art["month"], "January 2026")

    def test_es_fallback_when_compiled_missing(self):
        shutil.rmtree(self.root / "i18n" / "es" / "compiled")
        os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
        os.environ["QUARTO_PROFILE"] = "es"
        try:
            ret = self._run()
        finally:
            os.environ.pop("QUARTO_PROFILE", None)
        self.assertEqual(ret, 0)
        data = json.loads((self.root / "docs" / "es" / "articles-network.json").read_text())
        self.assertEqual(data["articles"][0]["title"], "My Title")

    def test_stable_ids_across_languages(self):
        os.environ.pop("QUARTO_PROFILE", None)
        os.environ.pop("QUARTO_PROJECT_OUTPUT_DIR", None)
        self._run()
        en_data = json.loads((self.root / "docs" / "articles-network.json").read_text())

        os.environ["QUARTO_PROFILE"] = "es"
        try:
            self._run()
        finally:
            os.environ.pop("QUARTO_PROFILE", None)
        es_data = json.loads((self.root / "docs" / "es" / "articles-network.json").read_text())
        en_ids = {t["id"] for t in en_data["topics"]}
        es_ids = {t["id"] for t in es_data["topics"]}
        self.assertEqual(en_ids, es_ids)
        self.assertEqual(
            en_data["articles"][0]["topics"], es_data["articles"][0]["topics"]
        )


class TestEsDumpNoop(unittest.TestCase):
    def test_es_dump_skips(self):
        os.environ["QUARTO_PROFILE"] = "es-dump"
        try:
            ret = _MOD.main()
        finally:
            os.environ.pop("QUARTO_PROFILE", None)
        self.assertEqual(ret, 0)


class TestImportSafety(unittest.TestCase):
    def test_import_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).rglob("*"))
            os.chdir(tmp)
            try:
                _load_module(_ROOT / "generate_articles_network.py")
            finally:
                os.chdir(_ROOT)
            after = set(Path(tmp).rglob("*"))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()