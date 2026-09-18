#!/usr/bin/env python3
"""i18n extraction, YAML skeleton management, and compilation for Quarto pages.

Modes
-----
default (``--lang es [SOURCE...]``)
    Read dump JSON from ``i18n/es/_extracted/``, create/refresh YAML skeletons
    under ``i18n/es/{pages,articles,diary}/``, preserving existing *es* values,
    then compile to ``i18n/es/compiled/``.

``--check [--require-complete]``
    Read-only validation.  Reports per-page + totals; exits nonzero on schema
    drift, missing skeletons, invalid translations, stale compiled output, or
    hash collisions.

``--compile [--require-complete]``
    Validate YAML against dump records and atomically write compiled JSON.
    Fails (nonzero, no file written) on any validation error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required but not installed.\n"
        "  Install with:  pip install pyyaml\n"
        "  Or in conda:   conda install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# YAML Dumper: block-style for multiline strings
# ---------------------------------------------------------------------------

class _BlockDumper(yaml.SafeDumper):
    """SafeDumper variant that uses block style for multiline strings."""


def _str_repr(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str", data, style="|"
        )
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _str_repr)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 4
META_FIELDS = ("title", "pagetitle", "description", "image-alt")

# Attributes excluded from structure-hash token stream
_SKIP_ATTR = frozenset(
    {"alt", "title", "aria-label", "aria-description", "data-caption"}
)
# Attributes included in structure-hash token stream
_KEEP_ATTR = frozenset({"id", "class", "href", "src"})


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def nfc(s: str) -> str:
    """NFC-normalise a Unicode string."""
    return unicodedata.normalize("NFC", s)


def derive_key(kind: str, en: str, seen: dict[str, str]) -> str:
    """Return ``f"{kind}-{md5hex[:12]}"``; raise on 12-char collision."""
    nfc_en = nfc(en)
    full = hashlib.md5((kind + "\0" + nfc_en).encode()).hexdigest()
    short = f"{kind}-{full[:12]}"
    if short in seen and seen[short] != full:
        raise ValueError(f"12-char hash collision on key {short!r}")
    seen[short] = full
    return short


def derive_raw_key(text: str, seen: dict[str, str]) -> str:
    """Return ``f"raw-{md5hex[:12]}"`` for a raw block; raise on collision."""
    full = hashlib.md5(nfc(text).encode()).hexdigest()
    short = f"raw-{full[:12]}"
    if short in seen and seen[short] != full:
        raise ValueError(f"12-char hash collision on raw key {short!r}")
    seen[short] = full
    return short


def _canonical_b64(s: str) -> str:
    """Normalize base64 padding to the canonical form (mod 4)."""
    s = s.rstrip("=")
    rem = len(s) % 4
    return s + "=" * ((4 - rem) % 4)


def filter_envelope(entries: list[dict]) -> list[dict]:
    """Keep only translatable label entries; skip href-like render-ids.

    Canonicalises base64 padding so dump-side keys always match YAML keys
    (PyYAML re-encodes base64 with full padding on write).
    """
    out: list[dict] = []
    for e in entries:
        rid = _canonical_b64(e.get("render_id", ""))
        try:
            decoded = base64.b64decode(rid).decode("utf-8")
        except Exception:
            continue
        if "/" in decoded or decoded.endswith(".html"):
            continue
        out.append({"render_id": rid, "en": e["en"]})
    return out


def _as_list(val: Any) -> list:
    """Normalise dump ``blocks``/``raw_blocks``/``envelope``: empty dict -> []."""
    if isinstance(val, dict):
        return []
    return val if isinstance(val, list) else []


# ---------------------------------------------------------------------------
# Structure-hash for raw HTML
# ---------------------------------------------------------------------------

class _StructParser(HTMLParser):
    """Extract tag/attr token stream for structure comparison."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[str] = []

    def _filtered(self, attrs: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for name, val in attrs:
            if name in _SKIP_ATTR:
                continue
            if name in _KEEP_ATTR or name.startswith("data-"):
                out.append((name, val or ""))
        out.sort()
        return out

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fa = self._filtered(attrs)
        a = " ".join(f'{n}="{v}"' for n, v in fa)
        self.tokens.append(f"<{tag} {a}>" if a else f"<{tag}>")

    def handle_endtag(self, tag: str) -> None:
        self.tokens.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        fa = self._filtered(attrs)
        a = " ".join(f'{n}="{v}"' for n, v in fa)
        self.tokens.append(f"<{tag} {a}/>" if a else f"<{tag}/>")

    def handle_data(self, data: str) -> None:
        pass  # text nodes excluded


def structure_hash(html: str) -> str:
    """SHA-256 of the tag/attr token stream of *html*."""
    p = _StructParser()
    p.feed(html)
    return hashlib.sha256("\n".join(p.tokens).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Protected-token extraction
# ---------------------------------------------------------------------------

_PAT_CODE = re.compile(r"`([^`]+)`")
_PAT_MATH_D = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_PAT_MATH_I = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_PAT_CITE = re.compile(r"\[?@([\w:-]+)\]?")
_PAT_IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_PAT_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")
_PAT_RAW = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>")


def extract_protected(text: str) -> Counter[str]:
    """Return multiset of protected tokens in *text*."""
    c: Counter[str] = Counter()
    for m in _PAT_CODE.finditer(text):
        c[f"code:{m.group(0)}"] += 1
    for m in _PAT_MATH_D.finditer(text):
        c[f"math:{m.group(0)}"] += 1
    for m in _PAT_MATH_I.finditer(text):
        c[f"math:{m.group(0)}"] += 1
    for m in _PAT_CITE.finditer(text):
        c[f"cite:@{m.group(1)}"] += 1
    for m in _PAT_IMG.finditer(text):
        c[f"img:{m.group(1)}"] += 1
    for m in _PAT_LINK.finditer(text):
        c[f"link:{m.group(1)}"] += 1
    for m in _PAT_RAW.finditer(text):
        c[f"raw:<{m.group(1)}>"] += 1
    return c


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def atomic_write(path: Path, data: str) -> None:
    """Write *data* to *path* atomically (tmp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------

def load_dump(path: Path) -> dict:
    """Load a dump JSON, verifying schema_version."""
    with open(path) as f:
        d = json.load(f)
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, "
            f"got {d.get('schema_version')}"
        )
    return d


def source_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_quarto_version() -> str:
    """Return current Quarto version string (from env or ``quarto --version``)."""
    import subprocess as _sp
    env_v = os.environ.get("QUARTO_VERSION", "").strip()
    if env_v:
        return env_v
    try:
        r = _sp.run(
            ["quarto", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# YAML skeleton builder
# ---------------------------------------------------------------------------

def build_skeleton(
    dump: dict, existing: dict | None, src_sha: str,
    qmd_sha: str = "", quarto_version: str = "",
) -> dict:
    """Build or refresh a YAML skeleton, preserving existing *es* values.

    Keys that vanished from the dump are moved to ``obsolete:``.
    Keys that reappear from obsolete are restored.
    """
    seen_keys: dict[str, str] = {}

    # ---- meta ----
    meta: dict[str, dict] = {}
    dump_meta = dump.get("meta", {})
    for field in META_FIELDS:
        en_val = dump_meta.get(field, "")
        if not isinstance(en_val, str):
            en_val = str(en_val) if en_val is not None else ""
        prev = (existing or {}).get("meta", {}).get(field, {})
        es_val = prev.get("es", "")
        if not isinstance(es_val, str):
            es_val = ""
        meta[field] = {"en": en_val, "es": es_val}

    # categories (always list)
    en_cat = dump_meta.get("categories", [])
    if isinstance(en_cat, str):
        en_cat = [en_cat]
    en_cat = [str(c) for c in en_cat] if isinstance(en_cat, list) else []
    prev_cat = (existing or {}).get("meta", {}).get("categories", {})
    es_cat = prev_cat.get("es", [])
    if not isinstance(es_cat, list):
        es_cat = []
    meta["categories"] = {"en": en_cat, "es": es_cat}

    # ---- blocks ----
    blocks: dict[str, dict] = {}
    existing_blocks = (existing or {}).get("blocks", {})
    existing_obsolete = (existing or {}).get("obsolete", {})

    dump_blocks = _as_list(dump.get("blocks", []))
    dump_block_keys: set[str] = set()
    for blk in dump_blocks:
        kind = blk["kind"]
        en = blk["en"]
        ctx = blk.get("context", "")
        key = derive_key(kind, en, seen_keys)
        dump_block_keys.add(key)
        prev = existing_blocks.get(key, existing_obsolete.get(key, {}))
        blocks[key] = {
            "kind": kind,
            "context": ctx,
            "en": en,
            "es": prev.get("es", "") if isinstance(prev.get("es"), str) else "",
        }

    # ---- raw_blocks ----
    raw_blocks: dict[str, dict] = {}
    existing_raw = (existing or {}).get("raw_blocks", {})

    dump_raws = _as_list(dump.get("raw_blocks", []))
    dump_raw_keys: set[str] = set()
    for rb in dump_raws:
        en_text = rb["en_text"]
        key = derive_raw_key(en_text, seen_keys)
        dump_raw_keys.add(key)
        sh = structure_hash(en_text)
        prev = existing_raw.get(key, existing_obsolete.get(key, {}))
        raw_blocks[key] = {
            "en": en_text,
            "es": prev.get("es", "") if isinstance(prev.get("es"), str) else "",
            "structure_sha256": sh,
        }

    # ---- envelope (filtered: no hrefs) ----
    envelope: dict[str, dict] = {}
    existing_env = (existing or {}).get("envelope", {})

    filtered_env = filter_envelope(_as_list(dump.get("envelope", [])))
    dump_env_ids: set[str] = set()
    for entry in filtered_env:
        rid = entry["render_id"]
        dump_env_ids.add(rid)
        prev = existing_env.get(rid, existing_obsolete.get(rid, {}))
        envelope[rid] = {
            "en": entry["en"],
            "es": prev.get("es", "") if isinstance(prev.get("es"), str) else "",
        }

    # ---- obsolete: keys that vanished ----
    obsolete: dict[str, dict] = {}
    for _section_name, existing_section, dump_keys in [
        ("blocks", existing_blocks, dump_block_keys),
        ("raw_blocks", existing_raw, dump_raw_keys),
        ("envelope", existing_env, dump_env_ids),
    ]:
        for key, val in existing_section.items():
            if key not in dump_keys and key not in existing_obsolete:
                obsolete[key] = dict(val)

    # Preserve entries already in obsolete (unless they came back)
    for key, val in existing_obsolete.items():
        if (
            key not in dump_block_keys
            and key not in dump_raw_keys
            and key not in dump_env_ids
        ):
            obsolete[key] = dict(val)

    # qmd_sha256: prefer fresh value, fall back to existing
    stored_qmd = (existing or {}).get("qmd_sha256", "")
    final_qmd = qmd_sha if qmd_sha else stored_qmd

    # quarto_version: prefer fresh, fall back to existing
    stored_qv = (existing or {}).get("quarto_version", "")
    final_qv = quarto_version if quarto_version else stored_qv

    return {
        "schema_version": SCHEMA_VERSION,
        "language": "es",
        "source": dump.get("source", ""),
        "source_sha256": src_sha,
        "qmd_sha256": final_qmd,
        "quarto_version": final_qv,
        "meta": meta,
        "blocks": blocks,
        "raw_blocks": raw_blocks,
        "envelope": envelope,
        "overrides": [],
        "obsolete": obsolete,
    }


# ---------------------------------------------------------------------------
# Raw HTML paired files
# ---------------------------------------------------------------------------

def write_raw_html_files(skeleton: dict, record_dir: Path) -> None:
    """Write paired .en.html (always overwrite) and .es.html (never overwrite)."""
    record_dir.mkdir(parents=True, exist_ok=True)
    for key, rb in skeleton.get("raw_blocks", {}).items():
        hash_part = key.replace("raw-", "", 1)
        en_path = record_dir / f"raw-html-{hash_part}.en.html"
        es_path = record_dir / f"raw-html-{hash_part}.es.html"
        en_path.write_text(rb["en"], encoding="utf-8")
        if not es_path.exists():
            es_path.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def skeleton_path(root: Path, record_rel: str) -> Path:
    """``i18n/es/<record-without-.json>.yml``"""
    yml_rel = record_rel.replace(".json", ".yml")
    return root / "i18n" / "es" / yml_rel


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        data,
        Dumper=_BlockDumper,
        allow_unicode=True,
        sort_keys=False,
        width=100,
        default_flow_style=False,
    )
    path.write_text(text, encoding="utf-8")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Compilation: YAML + dump -> compiled JSON
# ---------------------------------------------------------------------------

def validate_and_compile(
    yaml_data: dict, dump: dict
) -> tuple[dict, list[str]]:
    """Validate YAML against dump; return (compiled, errors).

    If *errors* is non-empty the compiled dict is incomplete and MUST NOT
    be written to disk.
    """
    errors: list[str] = []
    seen_keys: dict[str, str] = {}

    # Build dump lookup maps
    dump_blocks_list = _as_list(dump.get("blocks", []))
    dump_block_map: dict[str, dict] = {}
    for blk in dump_blocks_list:
        k = derive_key(blk["kind"], blk["en"], seen_keys)
        dump_block_map[k] = blk

    dump_raws_list = _as_list(dump.get("raw_blocks", []))
    dump_raw_map: dict[str, dict] = {}
    for rb in dump_raws_list:
        k = derive_raw_key(rb["en_text"], seen_keys)
        dump_raw_map[k] = rb

    filtered_env = filter_envelope(_as_list(dump.get("envelope", [])))
    dump_env_map = {e["render_id"]: e for e in filtered_env}

    # ---- meta ----
    compiled_meta: dict[str, Any] = {}
    yaml_meta = yaml_data.get("meta", {})
    for field in META_FIELDS:
        entry = yaml_meta.get(field, {})
        en = entry.get("en", "")
        es = entry.get("es", "")
        if not isinstance(en, str):
            en = ""
        if not isinstance(es, str):
            es = ""
        compiled_meta[field] = es if es else en

    cat_entry = yaml_meta.get("categories", {})
    en_cat = cat_entry.get("en", [])
    es_cat = cat_entry.get("es", [])
    if not isinstance(en_cat, list):
        en_cat = []
    if not isinstance(es_cat, list):
        es_cat = []
    compiled_meta["categories"] = es_cat if es_cat else en_cat

    # ---- blocks ----
    compiled_blocks: list[dict] = []
    yaml_blocks = yaml_data.get("blocks", {})
    for key, entry in yaml_blocks.items():
        kind = entry.get("kind", "")
        en = entry.get("en", "")
        es = entry.get("es", "")

        # en/es type check
        if not isinstance(en, str):
            errors.append(f"block {key!r}: en is not a string")
            continue
        if es is not None and not isinstance(es, str):
            errors.append(f"block {key!r}: es is not str or null")
            continue

        # key must exist in dump
        if key not in dump_block_map:
            errors.append(f"block key {key!r} not found in dump")
            continue

        # match must be verbatim
        if en != dump_block_map[key]["en"]:
            errors.append(f"block {key!r}: en mismatch with dump")
            continue

        # Validate es if non-empty
        if es:
            # Non-header kind whose es parses to a header
            if kind != "header" and re.match(r"^#{1,6}\s", es):
                errors.append(
                    f"block {key!r}: non-header kind but es starts with #"
                )
                continue

            # Header level change
            if kind == "header":
                en_level = re.match(r"^(#{1,6})\s", en)
                es_level = re.match(r"^(#{1,6})\s", es)
                if en_level and es_level:
                    if len(en_level.group(1)) != len(es_level.group(1)):
                        errors.append(
                            f"block {key!r}: header level change "
                            f"({len(en_level.group(1))} -> {len(es_level.group(1))})"
                        )
                        continue
                elif en_level and not es_level:
                    errors.append(
                        f"block {key!r}: header lost # prefix in es"
                    )
                    continue

            # Protected-token damage
            en_tok = extract_protected(en)
            es_tok = extract_protected(es)
            if en_tok != es_tok:
                missing = en_tok - es_tok
                extra = es_tok - en_tok
                detail = ""
                if missing:
                    detail += f" missing={dict(missing)}"
                if extra:
                    detail += f" extra={dict(extra)}"
                errors.append(
                    f"block {key!r}: protected-token mismatch{detail}"
                )
                continue

        compiled_blocks.append(
            {"kind": kind, "match": en, "es": es or None}
        )

    # ---- raw_blocks ----
    compiled_raws: list[dict] = []
    yaml_raws = yaml_data.get("raw_blocks", {})
    for key, entry in yaml_raws.items():
        en_text = entry.get("en", "")
        es_text = entry.get("es", "")
        stored_sh = entry.get("structure_sha256", "")

        if not isinstance(en_text, str) or not isinstance(stored_sh, str):
            errors.append(f"raw_block {key!r}: invalid types")
            continue
        if es_text is not None and not isinstance(es_text, str):
            errors.append(f"raw_block {key!r}: es is not str or null")
            continue

        if key not in dump_raw_map:
            errors.append(f"raw_block key {key!r} not found in dump")
            continue
        if en_text != dump_raw_map[key]["en_text"]:
            errors.append(f"raw_block {key!r}: en mismatch with dump")
            continue

        # Validate es_html structure if non-empty
        if es_text:
            es_sh = structure_hash(es_text)
            if es_sh != stored_sh:
                errors.append(
                    f"raw_block {key!r}: es_html structure mismatch "
                    f"(expected {stored_sh[:16]}..., got {es_sh[:16]}...)"
                )
                continue

        compiled_raws.append(
            {"match": en_text, "es_html": es_text or None}
        )

    # ---- envelope ----
    compiled_env: dict[str, str | None] = {}
    yaml_env = yaml_data.get("envelope", {})
    for rid, entry in yaml_env.items():
        en = entry.get("en", "")
        es = entry.get("es", "")
        if not isinstance(en, str):
            en = ""
        if not isinstance(es, str):
            es = ""

        if rid not in dump_env_map:
            errors.append(f"envelope key {rid!r} not found in dump")
            continue
        if en != dump_env_map[rid]["en"]:
            errors.append(f"envelope {rid!r}: en mismatch with dump")
            continue

        compiled_env[rid] = es if es else None

    # ---- coverage ----
    total_active = (
        len(compiled_blocks) + len(compiled_raws) + len(compiled_env)
    )
    translated = (
        sum(1 for b in compiled_blocks if b["es"] is not None)
        + sum(1 for r in compiled_raws if r["es_html"] is not None)
        + sum(1 for v in compiled_env.values() if v is not None)
    )
    coverage = translated / total_active if total_active > 0 else 1.0

    compiled = {
        "schema_version": SCHEMA_VERSION,
        "source": yaml_data.get("source", ""),
        "coverage": round(coverage, 6),
        "meta": compiled_meta,
        "blocks": compiled_blocks,
        "raw_blocks": compiled_raws,
        "envelope": compiled_env,
    }

    return compiled, errors


# ---------------------------------------------------------------------------
# Discover dump files
# ---------------------------------------------------------------------------

def discover_dumps(
    root: Path, sources: list[str] | None = None
) -> list[Path]:
    """Find dump JSON files under ``i18n/es/_extracted/``."""
    extract_dir = root / "i18n" / "es" / "_extracted"
    if not extract_dir.exists():
        return []

    if sources:
        paths: list[Path] = []
        for s in sources:
            p = Path(s)
            if p.is_absolute() and p.exists():
                if p.is_dir():
                    paths.extend(sorted(p.rglob("*.json")))
                elif p.suffix == ".json":
                    paths.append(p)
            else:
                # Relative to extract_dir
                p2 = extract_dir / s
                if p2.is_dir():
                    paths.extend(sorted(p2.rglob("*.json")))
                elif p2.suffix == ".json" and p2.exists():
                    paths.append(p2)
                elif p.exists():
                    paths.append(p)
        return [p for p in paths if ".stats.json" not in p.name]

    return sorted(
        p for p in extract_dir.rglob("*.json") if ".stats.json" not in p.name
    )


# ---------------------------------------------------------------------------
# Mode: default (extract + compile)
# ---------------------------------------------------------------------------

def run_default(root: Path, sources: list[str] | None) -> int:
    dumps = discover_dumps(root, sources)
    if not dumps:
        print("No dump files found.", file=sys.stderr)
        return 1

    quarto_ver = get_quarto_version()

    stats: dict[str, Any] = {
        "processed": 0,
        "created": 0,
        "refreshed": 0,
        "errors": [],
    }

    for dump_path in dumps:
        record_rel = str(
            dump_path.relative_to(root / "i18n" / "es" / "_extracted")
        )
        try:
            dump = load_dump(dump_path)
        except (json.JSONDecodeError, ValueError) as exc:
            stats["errors"].append(f"{record_rel}: {exc}")
            continue

        src_sha = source_sha256(dump_path)
        skel_path = skeleton_path(root, record_rel)

        # Compute qmd_sha256 if the source .qmd exists
        qmd_rel = dump.get("source", "")
        qmd_path = root / qmd_rel if qmd_rel else None
        qmd_sha = source_sha256(qmd_path) if qmd_path and qmd_path.exists() else ""

        existing = None
        if skel_path.exists():
            try:
                existing = load_yaml(skel_path)
            except Exception:
                existing = None

        skeleton = build_skeleton(dump, existing, src_sha, qmd_sha, quarto_ver)
        write_yaml(skel_path, skeleton)

        # Write raw HTML paired files
        record_stem = record_rel.replace(".json", "")
        raw_dir = root / "i18n" / "es" / record_stem
        write_raw_html_files(skeleton, raw_dir)

        if existing is None:
            stats["created"] += 1
        else:
            stats["refreshed"] += 1
        stats["processed"] += 1

    # Compile all
    compile_errors: list[str] = []
    for dump_path in dumps:
        record_rel = str(
            dump_path.relative_to(root / "i18n" / "es" / "_extracted")
        )
        skel_path = skeleton_path(root, record_rel)
        if not skel_path.exists():
            compile_errors.append(f"{record_rel}: skeleton missing")
            continue

        try:
            dump = load_dump(dump_path)
            yaml_data = load_yaml(skel_path)
        except Exception as exc:
            compile_errors.append(f"{record_rel}: {exc}")
            continue

        compiled, errs = validate_and_compile(yaml_data, dump)
        if errs:
            compile_errors.extend(f"{record_rel}: {e}" for e in errs)
            continue

        compiled_path = root / "i18n" / "es" / "compiled" / record_rel
        data = (
            json.dumps(compiled, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        )
        atomic_write(compiled_path, data)

    # Report
    print(
        f"Processed {stats['processed']} dump(s): "
        f"{stats['created']} created, {stats['refreshed']} refreshed"
    )
    if stats["errors"]:
        for e in stats["errors"]:
            print(f"  ERROR: {e}", file=sys.stderr)
    if compile_errors:
        for e in compile_errors:
            print(f"  COMPILE ERROR: {e}", file=sys.stderr)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Mode: check
# ---------------------------------------------------------------------------

def run_check(root: Path, require_complete: bool) -> int:
    dumps = discover_dumps(root)
    if not dumps:
        print("No dump files found.", file=sys.stderr)
        return 1

    current_quarto = get_quarto_version()
    quarto_warned = False
    grand_active = 0
    grand_translated = 0
    grand_fallback = 0
    grand_obsolete = 0
    issues: list[str] = []

    for dump_path in dumps:
        record_rel = str(
            dump_path.relative_to(root / "i18n" / "es" / "_extracted")
        )
        skel_path = skeleton_path(root, record_rel)

        if not skel_path.exists():
            issues.append(f"{record_rel}: skeleton MISSING")
            continue

        try:
            dump = load_dump(dump_path)
        except Exception as exc:
            issues.append(f"{record_rel}: dump load error: {exc}")
            continue

        src_sha = source_sha256(dump_path)
        try:
            yaml_data = load_yaml(skel_path)
        except Exception as exc:
            issues.append(f"{record_rel}: YAML load error: {exc}")
            continue

        # source_sha256 check (YAML vs dump)
        if yaml_data.get("source_sha256", "") != src_sha:
            issues.append(f"{record_rel}: stale source_sha256 (dump changed)")

        # qmd_sha256 check (source .qmd changed since dump)
        qmd_rel = dump.get("source", "")
        qmd_path = root / qmd_rel if qmd_rel else None
        if qmd_path and qmd_path.exists():
            current_qmd_sha = source_sha256(qmd_path)
            stored_qmd_sha = yaml_data.get("qmd_sha256", "")
            if stored_qmd_sha and current_qmd_sha != stored_qmd_sha:
                issues.append(
                    f"{record_rel}: source .qmd changed since dump"
                    f" — re-run: quarto render --profile es-dump"
                    f" && python3 scripts/i18n_extract.py --lang es"
                )

        # Quarto version drift warning (not a hard failure)
        stored_quarto = yaml_data.get("quarto_version", "")
        if stored_quarto and current_quarto and stored_quarto != current_quarto and not quarto_warned:
            print(
                f"\nWARNING: Quarto version mismatch — "
                f"dumped with {stored_quarto}, current is {current_quarto}.\n"
                f"  Upgrades can change callout/FloatRefTarget scaffolding and\n"
                f"  envelope render-ids, causing Spanish pages to silently fall\n"
                f"  back to English.  Re-dump and re-verify:\n"
                f"    quarto render --profile es-dump && python3 scripts/i18n_extract.py --lang es\n",
                file=sys.stderr,
            )
            quarto_warned = True

        # Completeness: dump entries must have matching active YAML entries
        seen_keys: dict[str, str] = {}
        dump_block_map: dict[str, dict] = {}
        for blk in _as_list(dump.get("blocks", [])):
            k = derive_key(blk["kind"], blk["en"], seen_keys)
            dump_block_map[k] = blk
        yaml_blocks = yaml_data.get("blocks", {})
        for k, blk in dump_block_map.items():
            if k not in yaml_blocks:
                snippet = blk["en"][:60].replace("\n", " ")
                issues.append(
                    f"{record_rel}: new/changed block in dump with no YAML entry: "
                    f"{blk['kind']}: {snippet!r}"
                )
        for k in yaml_blocks:
            if k not in dump_block_map and k not in yaml_data.get("obsolete", {}):
                issues.append(
                    f"{record_rel}: stale active YAML block {k!r}"
                    f" (absent from dump, should be in obsolete)"
                )

        dump_raw_map: dict[str, dict] = {}
        for rb in _as_list(dump.get("raw_blocks", [])):
            k = derive_raw_key(rb["en_text"], seen_keys)
            dump_raw_map[k] = rb
        yaml_raws = yaml_data.get("raw_blocks", {})
        for k in dump_raw_map:
            if k not in yaml_raws:
                issues.append(
                    f"{record_rel}: new/changed raw_block in dump with no YAML entry: {k}"
                )
        for k in yaml_raws:
            if k not in dump_raw_map and k not in yaml_data.get("obsolete", {}):
                issues.append(
                    f"{record_rel}: stale active YAML raw_block {k!r}"
                    f" (absent from dump, should be in obsolete)"
                )

        dump_env_filtered = filter_envelope(_as_list(dump.get("envelope", [])))
        dump_env_ids = {e["render_id"] for e in dump_env_filtered}
        yaml_env = yaml_data.get("envelope", {})
        for rid in dump_env_ids:
            if rid not in yaml_env:
                issues.append(
                    f"{record_rel}: new/changed envelope in dump with no YAML entry: {rid}"
                )
        for rid in yaml_env:
            if rid not in dump_env_ids and rid not in yaml_data.get("obsolete", {}):
                issues.append(
                    f"{record_rel}: stale active YAML envelope {rid!r}"
                    f" (absent from dump, should be in obsolete)"
                )

        # Count active/translated/fallback
        active = 0
        translated = 0

        for entry in yaml_data.get("blocks", {}).values():
            active += 1
            if entry.get("es"):
                translated += 1

        for entry in yaml_data.get("raw_blocks", {}).values():
            active += 1
            if entry.get("es"):
                translated += 1

        for entry in yaml_data.get("envelope", {}).values():
            active += 1
            if entry.get("es"):
                translated += 1

        fallback = active - translated
        obsolete = len(yaml_data.get("obsolete", {}))

        grand_active += active
        grand_translated += translated
        grand_fallback += fallback
        grand_obsolete += obsolete

        # Validate translations
        _, errs = validate_and_compile(yaml_data, dump)
        if errs:
            for e in errs:
                issues.append(f"{record_rel}: {e}")

        # Stale compiled check
        compiled_path = root / "i18n" / "es" / "compiled" / record_rel
        if compiled_path.exists():
            compiled, c_errs = validate_and_compile(yaml_data, dump)
            if not c_errs:
                expected = (
                    json.dumps(
                        compiled,
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                actual = compiled_path.read_text(encoding="utf-8")
                if expected != actual:
                    issues.append(f"{record_rel}: stale compiled JSON")
        else:
            issues.append(f"{record_rel}: compiled JSON MISSING")

        # Require-complete check
        if require_complete and fallback > 0:
            issues.append(
                f"{record_rel}: {fallback} untranslated unit(s)"
            )

        print(
            f"{record_rel}: active={active} translated={translated} "
            f"fallback={fallback} obsolete={obsolete}"
        )

    grand_coverage = (
        grand_translated / grand_active if grand_active > 0 else 1.0
    )
    print(
        f"\nTOTALS: active={grand_active} translated={grand_translated} "
        f"fallback={grand_fallback} obsolete={grand_obsolete}"
    )
    print(f"Coverage: {grand_coverage:.2%}")

    if issues:
        print(f"\n{len(issues)} issue(s):", file=sys.stderr)
        for i in issues:
            print(f"  - {i}", file=sys.stderr)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Mode: compile
# ---------------------------------------------------------------------------

def run_compile(root: Path, require_complete: bool) -> int:
    dumps = discover_dumps(root)
    if not dumps:
        print("No dump files found.", file=sys.stderr)
        return 1

    errors: list[str] = []

    for dump_path in dumps:
        record_rel = str(
            dump_path.relative_to(root / "i18n" / "es" / "_extracted")
        )
        skel_path = skeleton_path(root, record_rel)

        if not skel_path.exists():
            errors.append(f"{record_rel}: skeleton MISSING")
            continue

        try:
            dump = load_dump(dump_path)
            yaml_data = load_yaml(skel_path)
        except Exception as exc:
            errors.append(f"{record_rel}: {exc}")
            continue

        compiled, errs = validate_and_compile(yaml_data, dump)
        if errs:
            errors.extend(f"{record_rel}: {e}" for e in errs)
            continue

        # require-complete: check every active unit has a translation
        if require_complete:
            for b in compiled.get("blocks", []):
                if b.get("es") is None:
                    errors.append(
                        f"{record_rel}: untranslated block: "
                        f"{b['match'][:60]!r}"
                    )
            for r in compiled.get("raw_blocks", []):
                if r.get("es_html") is None:
                    errors.append(f"{record_rel}: untranslated raw_block")
            for rid, val in compiled.get("envelope", {}).items():
                if val is None:
                    errors.append(
                        f"{record_rel}: untranslated envelope: {rid}"
                    )

        if errors:
            continue  # Don't write if any errors accumulated

        compiled_path = root / "i18n" / "es" / "compiled" / record_rel
        data = (
            json.dumps(compiled, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        )
        atomic_write(compiled_path, data)
        print(f"  compiled {record_rel}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Compiled {len(dumps)} record(s)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="i18n extraction, YAML skeleton management, and compilation"
    )
    parser.add_argument(
        "--lang", default="es", help="Target language (default: es)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Read-only validation"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Validate and write compiled JSON",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all units to have translations",
    )
    parser.add_argument(
        "sources", nargs="*", help="Specific dump files or dirs to process"
    )
    args = parser.parse_args()

    root = Path.cwd()

    if args.check:
        sys.exit(run_check(root, args.require_complete))
    elif args.compile:
        sys.exit(run_compile(root, args.require_complete))
    else:
        sys.exit(run_default(root, args.sources or None))


if __name__ == "__main__":
    main()