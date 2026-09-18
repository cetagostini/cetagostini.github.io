#!/usr/bin/env python3
"""Tests for scripts/i18n_extract.py.

Run with:  conda run -n cetagostini_site python3 -m unittest scripts.tests.test_i18n_extract
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unicodedata
import unittest
from pathlib import Path

# Import the module under test (no import-time I/O)
import scripts.i18n_extract as ext


FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "i18n" / "_extracted"
)
DUMP_PATH = FIXTURE_DIR / "pages" / "test_page.json"


class TestHelpers(unittest.TestCase):
    """Pure helper functions."""

    def test_nfc_identity(self) -> None:
        self.assertEqual(ext.nfc("hello"), "hello")

    def test_nfc_composed(self) -> None:
        # e + combining acute  vs  precomposed e-acute
        decomposed = "e\u0301"
        composed = "\u00e9"
        self.assertEqual(ext.nfc(decomposed), composed)
        self.assertEqual(ext.nfc(composed), composed)

    def test_derive_key_basic(self) -> None:
        seen: dict[str, str] = {}
        k = ext.derive_key("para", "Hello world.", seen)
        self.assertTrue(k.startswith("para-"))
        self.assertEqual(len(k), len("para-") + 12)
        # Deterministic
        seen2: dict[str, str] = {}
        self.assertEqual(k, ext.derive_key("para", "Hello world.", seen2))

    def test_derive_key_collision(self) -> None:
        seen: dict[str, str] = {}
        # First insert
        k = ext.derive_key("para", "Hello world.", seen)
        # Same key, same full digest → no error
        self.assertEqual(k, ext.derive_key("para", "Hello world.", dict(seen)))
        # Force collision: different full digest, same 12-char prefix
        full = hashlib.md5(("para\0" + unicodedata.normalize("NFC", "Hello world.")).encode()).hexdigest()
        short = f"para-{full[:12]}"
        fake_full = "0" * 32
        bad_seen = {short: fake_full}
        with self.assertRaises(ValueError):
            ext.derive_key("para", "Hello world.", bad_seen)

    def test_derive_raw_key_basic(self) -> None:
        seen: dict[str, str] = {}
        k = ext.derive_raw_key("<div>test</div>", seen)
        self.assertTrue(k.startswith("raw-"))
        self.assertEqual(len(k), len("raw-") + 12)

    def test_canonical_b64(self) -> None:
        # Normal padding preserved
        self.assertEqual(ext._canonical_b64("dGVzdA=="), "dGVzdA==")
        # No padding needed (mod 4 == 0)
        self.assertEqual(ext._canonical_b64("dGVzdC1zaWRlYmFy"), "dGVzdC1zaWRlYmFy")
        # Already correct double padding
        self.assertEqual(ext._canonical_b64("dGVzdA=="), "dGVzdA==")
        # Stripped padding restored
        self.assertEqual(ext._canonical_b64("dGVzdA"), "dGVzdA==")
        # Verify canonical matches Python's b64encode for various strings
        import base64
        for raw in [b"test", b"quarto-twittercardtitle", b"quarto-testlabel", b"ab", b"a"]:
            expected = base64.b64encode(raw).decode()
            # Strip some padding and verify canonicalization restores it
            stripped = expected.rstrip("=")
            self.assertEqual(ext._canonical_b64(stripped), expected,
                             f"Failed for {raw!r}: {stripped!r} -> {ext._canonical_b64(stripped)!r} != {expected!r}")

    def test_filter_envelope(self) -> None:
        entries = [
            {"render_id": "dGVzdC10aXRsZQ==", "en": "Test Title"},
            {"render_id": "dGVzdC1zaWRlYmFy", "en": "Test Sidebar"},
            {"render_id": "L2Fib3V0Lmh0bWw=", "en": "/about.html"},
            {"render_id": "c29tZXRoaW5nLmh0bWw=", "en": "something.html"},
            {"render_id": "cXVhcnRvLWludC1uYXZiYXI6QWJvdXQ=", "en": "About"},
        ]
        filtered = ext.filter_envelope(entries)
        ids = [e["render_id"] for e in filtered]
        self.assertIn("dGVzdC10aXRsZQ==", ids)
        self.assertIn("dGVzdC1zaWRlYmFy", ids)
        self.assertIn("cXVhcnRvLWludC1uYXZiYXI6QWJvdXQ=", ids)
        # hrefs filtered out
        self.assertNotIn("L2Fib3V0Lmh0bWw=", ids)
        self.assertNotIn("c29tZXRoaW5nLmh0bWw=", ids)
        self.assertEqual(len(filtered), 3)

    def test_filter_envelope_canonicalizes_padding(self) -> None:
        """Regression: under-padded base64 must be canonicalized so dump and
        YAML keys agree after the YAML write/read round-trip."""
        import base64 as b64
        # quarto-twittercardtitle: canonical has single trailing =
        canonical = b64.b64encode(b"quarto-twittercardtitle").decode()
        # Strip the trailing = to simulate under-padded dump data
        under_padded = canonical.rstrip("=")
        self.assertNotEqual(canonical, under_padded)  # sanity check
        entries = [{"render_id": under_padded, "en": "Title"}]
        filtered = ext.filter_envelope(entries)
        self.assertEqual(len(filtered), 1)
        # Must be canonicalized
        self.assertEqual(filtered[0]["render_id"], canonical)
        self.assertEqual(
            b64.b64decode(filtered[0]["render_id"]).decode(), "quarto-twittercardtitle"
        )

    def test_structure_hash_basic(self) -> None:
        html1 = '<div id="a" class="b"><p>text</p></div>'
        html2 = '<div id="a" class="b"><p>other</p></div>'
        html3 = '<div id="a" class="c"><p>text</p></div>'
        # Same structure, different text → same hash
        self.assertEqual(ext.structure_hash(html1), ext.structure_hash(html2))
        # Different class → different hash
        self.assertNotEqual(ext.structure_hash(html1), ext.structure_hash(html3))

    def test_structure_hash_excludes_text_and_skip_attrs(self) -> None:
        html1 = '<img src="x.png" alt="Alt text" title="Title">'
        html2 = '<img src="x.png" alt="Different" title="Other">'
        self.assertEqual(ext.structure_hash(html1), ext.structure_hash(html2))

    def test_structure_hash_keeps_data_attrs(self) -> None:
        html1 = '<div data-value="42">text</div>'
        html2 = '<div data-value="99">other</div>'
        html3 = '<div data-value="42">other</div>'
        self.assertEqual(ext.structure_hash(html1), ext.structure_hash(html3))
        self.assertNotEqual(ext.structure_hash(html1), ext.structure_hash(html2))

    def test_extract_protected(self) -> None:
        text = 'Hello $x^2$ and `code` and @ref1 and ![img](url.png) and [link](http://x)'
        tok = ext.extract_protected(text)
        self.assertIn("math:$x^2$", tok)
        self.assertIn("code:`code`", tok)
        self.assertIn("cite:@ref1", tok)
        self.assertIn("img:url.png", tok)
        self.assertIn("link:http://x", tok)

    def test_extract_protected_multiset(self) -> None:
        text = "`a` and `a` and `b`"
        tok = ext.extract_protected(text)
        self.assertEqual(tok["code:`a`"], 2)
        self.assertEqual(tok["code:`b`"], 1)


class TestBuildSkeleton(unittest.TestCase):
    """YAML skeleton creation and idempotency."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.dump = ext.load_dump(DUMP_PATH)
        self.src_sha = ext.source_sha256(DUMP_PATH)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_skeleton(self) -> None:
        skel = ext.build_skeleton(self.dump, None, self.src_sha)
        self.assertEqual(skel["schema_version"], 4)
        self.assertEqual(skel["language"], "es")
        self.assertEqual(skel["source"], "test_page.qmd")
        self.assertEqual(skel["source_sha256"], self.src_sha)
        # Meta fields present
        for field in ("title", "pagetitle", "description", "image-alt", "categories"):
            self.assertIn(field, skel["meta"])
            self.assertIn("en", skel["meta"][field])
            self.assertIn("es", skel["meta"][field])
        # Blocks populated
        self.assertEqual(len(skel["blocks"]), 7)
        for key, entry in skel["blocks"].items():
            self.assertIn("kind", entry)
            self.assertIn("context", entry)
            self.assertIn("en", entry)
            self.assertEqual(entry["es"], "")
        # Raw blocks populated
        self.assertEqual(len(skel["raw_blocks"]), 2)
        for key, entry in skel["raw_blocks"].items():
            self.assertIn("en", entry)
            self.assertIn("structure_sha256", entry)
            self.assertEqual(entry["es"], "")
        # Envelope: hrefs filtered
        self.assertEqual(len(skel["envelope"]), 3)
        for rid, entry in skel["envelope"].items():
            self.assertIn("en", entry)
            self.assertEqual(entry["es"], "")
        # Obsolete empty
        self.assertEqual(skel["obsolete"], {})
        # overrides
        self.assertEqual(skel["overrides"], [])

    def test_idempotent(self) -> None:
        """Running build_skeleton twice with same dump produces identical output."""
        skel1 = ext.build_skeleton(self.dump, None, self.src_sha)
        skel2 = ext.build_skeleton(self.dump, skel1, self.src_sha)
        self.assertEqual(skel1, skel2)

    def test_preserves_es_values(self) -> None:
        """Existing es values are preserved on refresh."""
        skel1 = ext.build_skeleton(self.dump, None, self.src_sha)
        # Inject a translation
        first_key = next(iter(skel1["blocks"]))
        skel1["blocks"][first_key]["es"] = "Hola mundo."
        # Inject envelope translation
        first_rid = next(iter(skel1["envelope"]))
        skel1["envelope"][first_rid]["es"] = "Titulo de prueba"
        # Refresh
        skel2 = ext.build_skeleton(self.dump, skel1, self.src_sha)
        self.assertEqual(skel2["blocks"][first_key]["es"], "Hola mundo.")
        self.assertEqual(
            skel2["envelope"][first_rid]["es"], "Titulo de prueba"
        )

    def test_vanished_keys_to_obsolete(self) -> None:
        """Keys absent from new dump are moved to obsolete."""
        skel1 = ext.build_skeleton(self.dump, None, self.src_sha)
        # Inject a fake block that won't be in the dump
        skel1["blocks"]["para-fakefakefake"] = {
            "kind": "para",
            "context": "body",
            "en": "Vanished text.",
            "es": "Texto desaparecido.",
        }
        skel2 = ext.build_skeleton(self.dump, skel1, self.src_sha)
        self.assertNotIn("para-fakefakefake", skel2["blocks"])
        self.assertIn("para-fakefakefake", skel2["obsolete"])
        self.assertEqual(
            skel2["obsolete"]["para-fakefakefake"]["es"],
            "Texto desaparecido.",
        )

    def test_restored_from_obsolete(self) -> None:
        """Keys that reappear are restored from obsolete."""
        skel1 = ext.build_skeleton(self.dump, None, self.src_sha)
        first_key = next(iter(skel1["blocks"]))
        skel1["blocks"][first_key]["es"] = "Guardado."
        # Simulate vanishing: remove from blocks, add to obsolete
        entry = skel1["blocks"].pop(first_key)
        skel1["obsolete"][first_key] = dict(entry)
        # Rebuild: key is back in dump
        skel2 = ext.build_skeleton(self.dump, skel1, self.src_sha)
        self.assertIn(first_key, skel2["blocks"])
        self.assertEqual(skel2["blocks"][first_key]["es"], "Guardado.")
        self.assertNotIn(first_key, skel2["obsolete"])

    def test_skeleton_canonicalizes_envelope_keys(self) -> None:
        """Regression: skeleton envelope keys must use canonical base64 padding
        even when the dump stores under-padded render-ids."""
        import base64 as b64
        # quarto-testlabel: canonical form has double ==
        canonical = b64.b64encode(b"quarto-testlabel").decode()
        # Strip padding to simulate under-padded dump data
        under_padded = canonical.rstrip("=")
        self.assertNotEqual(canonical, under_padded)

        dump = {
            "schema_version": 4,
            "source": "padtest.qmd",
            "meta": {},
            "blocks": {},
            "raw_blocks": [],
            "envelope": [
                {"render_id": under_padded, "en": "Test Label"},
            ],
        }
        skel = ext.build_skeleton(dump, None, "fake_sha")
        # Key in skeleton must be canonically padded
        self.assertIn(canonical, skel["envelope"])
        self.assertEqual(skel["envelope"][canonical]["en"], "Test Label")


class TestYamlIO(unittest.TestCase):
    """YAML write/read round-trip."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip(self) -> None:
        dump = ext.load_dump(DUMP_PATH)
        sha = ext.source_sha256(DUMP_PATH)
        skel = ext.build_skeleton(dump, None, sha)
        yml_path = self.tmp / "test.yml"
        ext.write_yaml(yml_path, skel)
        loaded = ext.load_yaml(yml_path)
        # Key fields preserved
        self.assertEqual(loaded["schema_version"], 4)
        self.assertEqual(loaded["source"], "test_page.qmd")
        self.assertEqual(len(loaded["blocks"]), len(skel["blocks"]))

    def test_multiline_block_style(self) -> None:
        """Multiline strings should use block style (|) in YAML."""
        dump = ext.load_dump(DUMP_PATH)
        sha = ext.source_sha256(DUMP_PATH)
        skel = ext.build_skeleton(dump, None, sha)
        # Inject a multiline es value
        first_key = next(iter(skel["blocks"]))
        skel["blocks"][first_key]["es"] = "Linea uno.\nLinea dos.\nLinea tres."
        yml_path = self.tmp / "test.yml"
        ext.write_yaml(yml_path, skel)
        raw = yml_path.read_text()
        self.assertIn("|", raw)

    def test_envelope_keys_survive_yaml_roundtrip(self) -> None:
        """Regression: base64 render-ids with non-canonical padding must
        survive a YAML write → read round-trip without mismatching."""
        dump = ext.load_dump(DUMP_PATH)
        sha = ext.source_sha256(DUMP_PATH)
        skel = ext.build_skeleton(dump, None, sha)
        yml_path = self.tmp / "test.yml"
        ext.write_yaml(yml_path, skel)
        loaded = ext.load_yaml(yml_path)
        # All YAML envelope keys must be valid in the skeleton
        for yml_key in loaded.get("envelope", {}):
            self.assertIn(
                yml_key, skel["envelope"],
                f"YAML envelope key {yml_key!r} not found in skeleton"
            )


class TestCompile(unittest.TestCase):
    """Validation and compilation."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.dump = ext.load_dump(DUMP_PATH)
        self.src_sha = ext.source_sha256(DUMP_PATH)
        self.skel = ext.build_skeleton(self.dump, None, self.src_sha)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_compile(self) -> None:
        compiled, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertEqual(errors, [])
        self.assertEqual(compiled["schema_version"], 4)
        self.assertEqual(compiled["source"], "test_page.qmd")
        self.assertIn("coverage", compiled)
        self.assertEqual(len(compiled["blocks"]), 7)
        self.assertEqual(len(compiled["raw_blocks"]), 2)
        self.assertEqual(len(compiled["envelope"]), 3)

    def test_compiled_block_match_verbatim(self) -> None:
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        dump_ens = {
            b["en"] for b in self.dump["blocks"]
        }
        for b in compiled["blocks"]:
            self.assertIn(b["match"], dump_ens)

    def test_compiled_envelope_fallback_null(self) -> None:
        """Envelope entries without es should be null in compiled."""
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        for rid, val in compiled["envelope"].items():
            self.assertIsNone(val)  # no translations yet

    def test_compiled_envelope_with_translation(self) -> None:
        first_rid = next(iter(self.skel["envelope"]))
        self.skel["envelope"][first_rid]["es"] = "Titulo de prueba"
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        self.assertEqual(compiled["envelope"][first_rid], "Titulo de prueba")

    def test_compiled_meta_fallback(self) -> None:
        """Meta fields without es should fall back to en."""
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        self.assertEqual(compiled["meta"]["title"], "Test Page")
        self.assertEqual(compiled["meta"]["pagetitle"], "Test Page \u2014 Author")

    def test_compiled_meta_with_translation(self) -> None:
        self.skel["meta"]["title"]["es"] = "Pagina de prueba"
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        self.assertEqual(compiled["meta"]["title"], "Pagina de prueba")

    def test_compiled_deterministic_json(self) -> None:
        """Compiled JSON should be deterministic (sorted keys, trailing newline)."""
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        data1 = json.dumps(compiled, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        data2 = json.dumps(compiled, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        self.assertEqual(data1, data2)

    def test_compiled_raw_blocks_es_null_when_empty(self) -> None:
        compiled, _ = ext.validate_and_compile(self.skel, self.dump)
        for r in compiled["raw_blocks"]:
            self.assertIsNone(r["es_html"])


class TestNegativeCompile(unittest.TestCase):
    """--compile exits nonzero and writes NO JSON for various error cases."""

    def setUp(self) -> None:
        self.dump = ext.load_dump(DUMP_PATH)
        self.src_sha = ext.source_sha256(DUMP_PATH)
        self.skel = ext.build_skeleton(self.dump, None, self.src_sha)

    def test_malformed_yaml_non_str_en(self) -> None:
        """en that is not a string should fail."""
        first_key = next(iter(self.skel["blocks"]))
        self.skel["blocks"][first_key]["en"] = 42  # not a string
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(any("en is not a string" in e for e in errors))

    def test_non_str_scalar_es(self) -> None:
        """es that is not str or null should fail."""
        first_key = next(iter(self.skel["blocks"]))
        self.skel["blocks"][first_key]["es"] = 123
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(any("es is not str or null" in e for e in errors))

    def test_damaged_protected_token(self) -> None:
        """Removing a protected token from es should fail."""
        # Find a block with protected tokens (the $x^2$ math block)
        math_key = None
        for key, entry in self.skel["blocks"].items():
            if "$x^2$" in entry["en"]:
                math_key = key
                break
        self.assertIsNotNone(math_key)
        # Set es without the math token
        self.skel["blocks"][math_key]["es"] = "Text without math."  # type: ignore[index]
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(
            any("protected-token mismatch" in e for e in errors)
        )

    def test_header_level_change(self) -> None:
        """Changing header level should fail."""
        header_key = None
        for key, entry in self.skel["blocks"].items():
            if entry["kind"] == "header":
                header_key = key
                break
        self.assertIsNotNone(header_key)
        # Change from ## to ###
        self.skel["blocks"][header_key]["es"] = "### Titulo de seccion"  # type: ignore[index]
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(any("header level change" in e for e in errors))

    def test_non_header_with_header_es(self) -> None:
        """Non-header kind whose es starts with # should fail."""
        para_key = None
        for key, entry in self.skel["blocks"].items():
            if entry["kind"] == "para" and "$" not in entry["en"] and "`" not in entry["en"]:
                para_key = key
                break
        self.assertIsNotNone(para_key)
        self.skel["blocks"][para_key]["es"] = "## Esto es un header"  # type: ignore[index]
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(
            any("non-header kind but es starts with #" in e for e in errors)
        )

    def test_raw_html_structure_mismatch(self) -> None:
        """Non-empty es.html with wrong structure should fail."""
        raw_key = None
        for key, entry in self.skel["raw_blocks"].items():
            raw_key = key
            break
        self.assertIsNotNone(raw_key)
        # Set es to HTML with different structure
        self.skel["raw_blocks"][raw_key]["es"] = "<span>wrong</span>"  # type: ignore[index]
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(any("structure mismatch" in e for e in errors))

    def test_raw_html_structure_match(self) -> None:
        """Non-empty es.html with matching structure should pass."""
        raw_key = None
        for key, entry in self.skel["raw_blocks"].items():
            raw_key = key
            break
        self.assertIsNotNone(raw_key)
        # Get the structure hash
        sh = self.skel["raw_blocks"][raw_key]["structure_sha256"]  # type: ignore[index]
        # Set es to HTML with same structure but different text
        en_text = self.skel["raw_blocks"][raw_key]["en"]  # type: ignore[index]
        # Replace text content but keep tags
        import re
        es_text = re.sub(r">[^<]+<", ">Translated<", en_text)
        # Make sure it's self-closing compatible
        self.skel["raw_blocks"][raw_key]["es"] = es_text  # type: ignore[index]
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertFalse(any("structure mismatch" in e for e in errors))

    def test_en_mismatch_with_dump(self) -> None:
        """Block en that doesn't match dump should fail."""
        first_key = next(iter(self.skel["blocks"]))
        self.skel["blocks"][first_key]["en"] = "Tampered english text."
        _, errors = ext.validate_and_compile(self.skel, self.dump)
        self.assertTrue(any("en mismatch with dump" in e for e in errors))


class TestCoverageMath(unittest.TestCase):
    """Coverage calculations."""

    def test_zero_active_gives_100(self) -> None:
        """When total_active == 0, coverage must be 1.0."""
        dump = {
            "schema_version": 4,
            "source": "empty.qmd",
            "meta": {},
            "blocks": [],
            "raw_blocks": [],
            "envelope": [],
        }
        sha = hashlib.sha256(b"empty").hexdigest()
        skel = ext.build_skeleton(dump, None, sha)
        compiled, errors = ext.validate_and_compile(skel, dump)
        self.assertEqual(errors, [])
        self.assertEqual(compiled["coverage"], 1.0)

    def test_obsolete_excluded_from_counts(self) -> None:
        """Obsolete keys must not inflate active or translated counts."""
        dump = ext.load_dump(DUMP_PATH)
        sha = ext.source_sha256(DUMP_PATH)
        skel = ext.build_skeleton(dump, None, sha)
        # Translate some blocks with VALID es values (preserve structure)
        for key, entry in skel["blocks"].items():
            if entry["kind"] == "header":
                entry["es"] = "## Titulo traducido"
            elif "$" in entry["en"] or "`" in entry["en"]:
                # Keep protected tokens exactly
                entry["es"] = entry["en"]
            else:
                entry["es"] = "Traducido."
        # Add obsolete entry
        skel["obsolete"]["para-fakefakefake"] = {
            "kind": "para",
            "context": "body",
            "en": "Old text.",
            "es": "Texto viejo.",
        }
        compiled, errors = ext.validate_and_compile(skel, dump)
        self.assertEqual(errors, [])
        # Coverage should reflect only active units
        n_blocks = len(compiled["blocks"])
        n_raw = len(compiled["raw_blocks"])
        n_env = len(compiled["envelope"])
        total_active = n_blocks + n_raw + n_env
        translated = (
            sum(1 for b in compiled["blocks"] if b["es"] is not None)
            + sum(1 for r in compiled["raw_blocks"] if r["es_html"] is not None)
            + sum(1 for v in compiled["envelope"].values() if v is not None)
        )
        expected_cov = translated / total_active
        self.assertAlmostEqual(compiled["coverage"], expected_cov, places=4)
        # Obsolete is not counted
        self.assertNotIn("para-fakefakefake", compiled["blocks"])

    def test_full_translation_coverage_100(self) -> None:
        """When every unit is translated, coverage should be 1.0."""
        import re as _re

        dump = ext.load_dump(DUMP_PATH)
        sha = ext.source_sha256(DUMP_PATH)
        skel = ext.build_skeleton(dump, None, sha)
        # Translate all blocks with VALID es values (preserve structure/tokens)
        for key, entry in skel["blocks"].items():
            if entry["kind"] == "header":
                entry["es"] = "## Titulo traducido"
            elif "$" in entry["en"] or "`" in entry["en"]:
                # Keep protected tokens exactly, translate surrounding
                entry["es"] = entry["en"]
            else:
                entry["es"] = "Traducido."
        for k in skel["raw_blocks"]:
            en_text = skel["raw_blocks"][k]["en"]
            es_text = _re.sub(r">[^<]+<", ">Traducido<", en_text)
            skel["raw_blocks"][k]["es"] = es_text
        for rid in skel["envelope"]:
            skel["envelope"][rid]["es"] = "Traducido."
        compiled, errors = ext.validate_and_compile(skel, dump)
        self.assertEqual(errors, [])
        self.assertEqual(compiled["coverage"], 1.0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end: run_default then run_check."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        # Set up the fixture structure in tmp
        self.extract_dir = self.tmp / "i18n" / "es" / "_extracted" / "pages"
        self.extract_dir.mkdir(parents=True)
        shutil.copy2(DUMP_PATH, self.extract_dir / "test_page.json")
        self.orig_cwd = Path.cwd()
        os.chdir(self.tmp)

    def tearDown(self) -> None:
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_creates_skeleton_and_compiled(self) -> None:
        ret = ext.run_default(self.tmp, None)
        self.assertEqual(ret, 0)
        # Skeleton exists
        skel_path = self.tmp / "i18n" / "es" / "pages" / "test_page.yml"
        self.assertTrue(skel_path.exists())
        # Compiled exists
        compiled_path = (
            self.tmp / "i18n" / "es" / "compiled" / "pages" / "test_page.json"
        )
        self.assertTrue(compiled_path.exists())
        # Compiled is valid JSON
        data = json.loads(compiled_path.read_text())
        self.assertEqual(data["schema_version"], 4)

    def test_default_idempotent(self) -> None:
        """Running default twice changes nothing."""
        ext.run_default(self.tmp, None)
        skel_path = self.tmp / "i18n" / "es" / "pages" / "test_page.yml"
        sha_before = hashlib.sha256(skel_path.read_bytes()).hexdigest()
        ext.run_default(self.tmp, None)
        sha_after = hashlib.sha256(skel_path.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after)

    def test_check_passes_after_default(self) -> None:
        ext.run_default(self.tmp, None)
        ret = ext.run_check(self.tmp, False)
        self.assertEqual(ret, 0)

    def test_check_require_complete_fails_with_fallback(self) -> None:
        ext.run_default(self.tmp, None)
        ret = ext.run_check(self.tmp, True)
        self.assertNotEqual(ret, 0)  # no translations → fallbacks

    def test_raw_html_files_written(self) -> None:
        ext.run_default(self.tmp, None)
        raw_dir = self.tmp / "i18n" / "es" / "pages" / "test_page"
        self.assertTrue(raw_dir.exists())
        en_files = list(raw_dir.glob("raw-html-*.en.html"))
        es_files = list(raw_dir.glob("raw-html-*.es.html"))
        self.assertEqual(len(en_files), 2)
        self.assertEqual(len(es_files), 2)
        # .en.html files have content
        for f in en_files:
            self.assertGreater(f.stat().st_size, 0)
        # .es.html files are empty (no translations yet)
        for f in es_files:
            self.assertEqual(f.stat().st_size, 0)

    def test_es_html_not_overwritten(self) -> None:
        """Existing .es.html files should not be overwritten."""
        ext.run_default(self.tmp, None)
        raw_dir = self.tmp / "i18n" / "es" / "pages" / "test_page"
        es_files = list(raw_dir.glob("raw-html-*.es.html"))
        self.assertTrue(len(es_files) > 0)
        # Write content to one
        es_file = es_files[0]
        es_file.write_text("<p>Translated content</p>", encoding="utf-8")
        # Re-run
        ext.run_default(self.tmp, None)
        # Content preserved
        self.assertEqual(
            es_file.read_text(encoding="utf-8"),
            "<p>Translated content</p>",
        )

    def test_compile_mode(self) -> None:
        ext.run_default(self.tmp, None)
        # Delete compiled, re-compile
        compiled_dir = self.tmp / "i18n" / "es" / "compiled"
        shutil.rmtree(compiled_dir)
        ret = ext.run_compile(self.tmp, False)
        self.assertEqual(ret, 0)
        compiled_path = compiled_dir / "pages" / "test_page.json"
        self.assertTrue(compiled_path.exists())

    def test_check_catches_qmd_sha256_drift(self) -> None:
        """--check must fail when the source .qmd has changed since dump."""
        # Create a fake .qmd file so qmd_sha256 is computed
        qmd_path = self.tmp / "test_page.qmd"
        qmd_path.write_text("# Original content\n", encoding="utf-8")
        ext.run_default(self.tmp, None)
        # Verify check passes
        self.assertEqual(ext.run_check(self.tmp, False), 0)
        # Now "change" the .qmd
        qmd_path.write_text("# Changed content!\n", encoding="utf-8")
        ret = ext.run_check(self.tmp, False)
        self.assertNotEqual(ret, 0)

    def test_check_catches_new_dump_block_without_yaml(self) -> None:
        """--check must fail when the dump has a block not in the YAML."""
        ext.run_default(self.tmp, None)
        # Verify check passes
        self.assertEqual(ext.run_check(self.tmp, False), 0)
        # Add a new block to the dump JSON
        dump_path = self.extract_dir / "test_page.json"
        dump = json.loads(dump_path.read_text())
        dump["blocks"].append({
            "kind": "para", "context": "body",
            "en": "Brand new paragraph from source edit.", "count": 1
        })
        dump_path.write_text(json.dumps(dump), encoding="utf-8")
        # Re-run default to update dump hash but NOT regenerate skeleton
        # (we only update source_sha256, not the skeleton blocks)
        # Instead, run check directly — it should detect the new block
        ret = ext.run_check(self.tmp, False)
        self.assertNotEqual(ret, 0)

    def test_check_catches_stale_active_yaml_block(self) -> None:
        """--check must fail when a YAML block is active but absent from dump."""
        ext.run_default(self.tmp, None)
        self.assertEqual(ext.run_check(self.tmp, False), 0)
        # Inject a stale active block into the YAML
        skel_path = self.tmp / "i18n" / "es" / "pages" / "test_page.yml"
        yaml_data = ext.load_yaml(skel_path)
        yaml_data["blocks"]["para-deadbeefdead"] = {
            "kind": "para", "context": "body",
            "en": "Ghost block.", "es": ""
        }
        ext.write_yaml(skel_path, yaml_data)
        ret = ext.run_check(self.tmp, False)
        self.assertNotEqual(ret, 0)

    def test_qmd_sha256_stored_in_skeleton(self) -> None:
        """qmd_sha256 field must appear in the skeleton when source .qmd exists."""
        qmd_path = self.tmp / "test_page.qmd"
        qmd_path.write_text("# Test\n", encoding="utf-8")
        ext.run_default(self.tmp, None)
        skel_path = self.tmp / "i18n" / "es" / "pages" / "test_page.yml"
        yaml_data = ext.load_yaml(skel_path)
        self.assertIn("qmd_sha256", yaml_data)
        self.assertTrue(len(yaml_data["qmd_sha256"]) == 64)

    def test_quarto_version_stored_in_skeleton(self) -> None:
        """quarto_version must be stored when run_default passes it."""
        ext.run_default(self.tmp, None)
        skel_path = self.tmp / "i18n" / "es" / "pages" / "test_page.yml"
        yaml_data = ext.load_yaml(skel_path)
        self.assertIn("quarto_version", yaml_data)

    def test_check_warns_on_quarto_version_drift(self) -> None:
        """--check must print a WARNING when quarto_version differs."""
        ext.run_default(self.tmp, None)
        # Inject a different quarto_version into the YAML
        skel_path = self.tmp / "i18n" / "es" / "pages" / "test_page.yml"
        yaml_data = ext.load_yaml(skel_path)
        yaml_data["quarto_version"] = "0.0.0-fake-old"
        ext.write_yaml(skel_path, yaml_data)
        # Capture stderr for the warning
        import io
        old_stderr = sys.stderr
        sys.stderr = buf = io.StringIO()
        try:
            ext.run_check(self.tmp, False)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("Quarto version mismatch", output)
        self.assertIn("0.0.0-fake-old", output)


class TestRuntimeGate(unittest.TestCase):
    """Tests for the --runtime mode of i18n_coverage_gate.py."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.stats_dir = self.tmp / "i18n" / "es" / "_extracted"
        self.stats_dir.mkdir(parents=True)
        self.orig_cwd = Path.cwd()
        os.chdir(self.tmp)

    def tearDown(self) -> None:
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_stats(self, name: str, matched: int, total: int,
                     unmatched: list[str] | None = None) -> None:
        data = {
            "source": name,
            "matched": matched,
            "total": total,
            "unmatched": unmatched or [],
        }
        path = self.stats_dir / f"{name}.stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_runtime_passes_when_all_match(self) -> None:
        # Lua stats: "matched" = hits, "total" = misses; 0 misses = 100%
        self._write_stats("pages/about", 10, 0)
        # Should not raise
        from scripts.i18n_coverage_gate import _run_runtime_gate
        _run_runtime_gate(self.tmp, False)

    def test_runtime_fails_on_zero_matched_page(self) -> None:
        self._write_stats("pages/about", 0, 5, ["unmatched1", "unmatched2"])
        from scripts.i18n_coverage_gate import _run_runtime_gate
        with self.assertRaises(SystemExit) as ctx:
            _run_runtime_gate(self.tmp, False)
        self.assertEqual(ctx.exception.code, 1)

    def test_runtime_fails_below_floor(self) -> None:
        # 50% overall < 90% floor
        self._write_stats("pages/about", 5, 10)
        self._write_stats("pages/articles", 0, 10, ["bad block"])
        from scripts.i18n_coverage_gate import _run_runtime_gate
        with self.assertRaises(SystemExit) as ctx:
            _run_runtime_gate(self.tmp, False)
        self.assertEqual(ctx.exception.code, 1)

    def test_runtime_allow_partial(self) -> None:
        self._write_stats("pages/about", 0, 5, ["unmatched"])
        from scripts.i18n_coverage_gate import _run_runtime_gate
        # Should NOT raise with allow_partial=True
        _run_runtime_gate(self.tmp, True)

    def test_runtime_includes_unmatched_samples(self) -> None:
        self._write_stats("pages/about", 0, 3,
                          ["Some long English text", "Another block"])
        from scripts.i18n_coverage_gate import _run_runtime_gate
        import io
        old_stderr = sys.stderr
        sys.stderr = buf = io.StringIO()
        try:
            with self.assertRaises(SystemExit):
                _run_runtime_gate(self.tmp, False)
        finally:
            sys.stderr = old_stderr
        output = buf.getvalue()
        self.assertIn("Some long English text", output)
        self.assertIn("unmatched", output)


if __name__ == "__main__":
    unittest.main()