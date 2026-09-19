#!/usr/bin/env python3
"""Build the interactive Articles-page network data.

Reads every `articles/<slug>/<slug>.qmd` frontmatter (title, date, description,
categories, image, image-alt), canonicalises categories into topics, resolves
each article image to a site-relative path, and writes
`docs/articles-network.json` for js/articles-network.js to fetch.

The network draws circular thumbnails, so `--thumbs` also writes square
`images/network/<slug>.jpg` crops (committed, like every other site asset).
That step needs Pillow, which lives in the `cetagostini_site` conda env:

    conda run -n cetagostini_site python generate_articles_network.py --thumbs

Quarto runs the plain form after every render (see `project: post-render` in
_quarto.yml), so the JSON always matches the committed frontmatter. The pass's
language comes from QUARTO_PROFILE unless `--lang` says otherwise; a translated
network reads its titles, descriptions and labels from that language's compiled
`i18n/<lang>/compiled/` dictionaries and lands in `docs/<lang>/`:

    QUARTO_PROFILE=pt python3 generate_articles_network.py
    python3 generate_articles_network.py --lang pt --output-dir docs/pt
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # a render must not die on a machine without PyYAML
    yaml = None

ROOT = Path(__file__).resolve().parent
ARTICLES = ROOT / "articles"
THUMB_DIR = ROOT / "images" / "network"
THUMB_SIZE = 384
THUMB_QUALITY = 78
EN_DIRNAME = "docs"
# Languages with a profile, in the order scripts/render-all.sh renders them.
LANGS = ("es", "pt")
OUT_NAME = "articles-network.json"

# Categories are free-form in article frontmatter; the network needs one label
# per topic. Extend this table when a new category spelling shows up.
TOPIC_ALIASES = {
    "mmm": "MMM",
    "media mix modeling": "MMM",
    "pymc-marketing": "PyMC-Marketing",
    "pymc": "PyMC",
    "python": "Python",
    "bayesian": "Bayesian",
    "experimentation": "Experimentation",
    "quasi-experiments": "Quasi-experiments",
    "causal discovery": "Causal discovery",
    "causal learning": "Causal discovery",
    "prior elicitation": "Priors",
    "priors": "Priors",
    "optimization": "Optimization",
    "robust optimization": "Optimization",
    "decision theory": "Decision theory",
    "pytensor": "PyTensor",
    "llm": "LLM",
    "gguf": "GGUF",
    "gemma": "Gemma",
    "mlx": "MLX",
    "numba": "Numba",
    "causalpy": "CausalPy",
    "synthetic control": "Synthetic control",
    "spillovers": "Spillovers",
    "marketing": "Marketing",
    # Conference and city tags all mean "this one was a talk".
    "pydata": "Talk",
    "germany": "Talk",
    "berlin": "Talk",
    "darmstadt": "Talk",
    "tallinn": "Talk",
    "estonia": "Talk",
}

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.S)


def profile_tokens(env) -> set[str]:
    """QUARTO_PROFILE is a comma-separated list; match tokens, never substrings.

    `"es" in profile` would also fire on `es-dump` (and on any profile whose
    name happens to contain those letters).
    """
    return {token.strip() for token in env.get("QUARTO_PROFILE", "").split(",") if token.strip()}


def resolve_lang(env) -> str | None:
    """Language of this render pass, or None when there is nothing to do.

    A `<lang>-dump` pass writes a disposable extraction tree, not a site — it is
    skipped even when `--lang` is passed, since nothing reads that tree.
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

    A missing dictionary leaves the network English rather than failing the
    render — the coverage gate is what enforces translation completeness.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"  ! no compiled dictionary: {path}")
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"  ! unreadable dictionary {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def localize(records: list[dict], site: dict, compiled: Path) -> None:
    """Swap English display strings for the compiled translated ones, in place.

    Per field: an entry that is missing or empty keeps the English text, so a
    half-translated article still renders. Topic ids are untouched here — they
    stay slugified from the English label so both networks share one id space.
    """
    months = site.get("months") or {}
    for record in records:
        meta = load_compiled(compiled / "articles" / f"{record['slug']}.json").get("meta") or {}
        title = str(meta.get("title") or "").strip()
        if title:
            record["title"] = title
            record["shortTitle"] = short_title(title)
        description = " ".join(str(meta.get("description") or "").split())
        if description:
            record["description"] = description
        image_alt = str(meta.get("image-alt") or "").strip()
        if image_alt:
            record["imageAlt"] = image_alt
        month, _, year = record["month"].partition(" ")
        record["month"] = f"{months.get(month, month)} {year}".strip()


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def topic_label(raw: str) -> str:
    key = " ".join(str(raw).split()).lower()
    if key in TOPIC_ALIASES:
        return TOPIC_ALIASES[key]
    if len(key) <= 5 and key.isalpha():  # acronym-ish (roas, ctv, ...)
        return key.upper()
    return key[:1].upper() + key[1:]


def short_title(title: str) -> str:
    """Label-sized title: the part before a colon when that part stands alone."""
    head = title.split(":", 1)[0].strip()
    if head != title and len(head) >= 12 and " " in head:
        return head
    return title


def _clean(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if " #" in text:
        text = text.split(" #", 1)[0]
    return text.strip()


def parse_frontmatter(block: str, qmd: Path) -> dict:
    """Read the handful of frontmatter fields the network needs.

    PyYAML is used when present (it is, in the documented render environment).
    The fallback keeps `quarto render` working without it; it understands inline
    and block lists, and prints a note for anything it cannot read rather than
    guessing.
    """
    if yaml is not None:
        try:
            return yaml.safe_load(block) or {}
        except yaml.YAMLError as exc:
            print(f"  ! bad YAML in {qmd.relative_to(ROOT)}: {exc}")
            return {}

    meta: dict = {}
    key = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- "):
            if key is None:
                continue
            if not isinstance(meta.get(key), list):
                meta[key] = [] if meta.get(key) in ("", None) else [meta[key]]
            meta[key].append(_clean(line.lstrip()[2:]))
            continue
        match = re.match(r"^([A-Za-z][\w -]*):\s*(.*)$", line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if value in (">", "|", ">-", "|-", ">+", "|+"):
            print(f"  ! {qmd.parent.name}: block scalars need PyYAML, skipping {key}")
            meta[key] = ""
        elif value.startswith("[") and value.endswith("]"):
            meta[key] = [_clean(part) for part in value[1:-1].split(",") if part.strip()]
        else:
            meta[key] = _clean(value)
    return meta


def parse_date(value) -> dt.date | None:
    """Frontmatter dates arrive quoted ("2026-08-07") or bare (2026-08-07)."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def resolve_image(raw: str, qmd: Path) -> str | None:
    """Article `image:` is written root-relative, article-relative or bare.

    Some frontmatter uses `../images/x.jpg`, which resolves to `articles/images/`
    from `articles/<slug>/` — there is no such folder. Fall back to the same
    path read from the repo root so the network still gets a thumbnail.
    """
    value = str(raw).strip().strip("\"'")
    if not value:
        return None
    stripped = value.lstrip("/")
    candidates = [
        qmd.parent / value,                    # article-relative (correct form)
        ROOT / stripped,                       # site-relative ("/images/x.jpg")
        ROOT / stripped.lstrip("../"),         # legacy "../images/x.jpg"
    ]
    for candidate in candidates:
        try:
            rel = candidate.resolve().relative_to(ROOT)
        except ValueError:
            continue
        if candidate.is_file():
            return str(rel)
    print(f"  ! image not found for {qmd.parent.name}: {value}")
    return None


def load_articles() -> tuple[list[dict], dict[str, str]]:
    records = []
    labels_by_id: dict[str, str] = {}
    for qmd in sorted(ARTICLES.glob("*/*.qmd")):
        match = _FRONTMATTER.match(qmd.read_text(encoding="utf-8"))
        if not match:
            print(f"  ! no frontmatter: {qmd.relative_to(ROOT)}")
            continue
        meta = parse_frontmatter(match.group(1), qmd)

        slug = qmd.parent.name
        date = parse_date(meta.get("date"))
        if date is None:
            print(f"  ! no date: {qmd.relative_to(ROOT)}")
            continue

        title = str(meta.get("title") or slug).strip()
        labels: list[str] = []
        for raw in meta.get("categories") or []:
            label = topic_label(raw)
            if label not in labels:
                labels.append(label)
        if not labels:
            labels = ["Unfiled"]
        for label in labels:
            labels_by_id.setdefault(slugify(label), label)

        image = resolve_image(meta.get("image", ""), qmd)
        thing = {
            "slug": slug,
            "title": title,
            "shortTitle": short_title(title),
            "url": f"articles/{slug}/{qmd.stem}.html",
            "date": date.isoformat(),
            "year": date.year,
            "month": date.strftime("%B %Y"),
            "description": " ".join(str(meta.get("description") or "").split()),
            "topics": [slugify(label) for label in labels],
            "image": f"images/network/{slug}.jpg" if image else None,
            "imageSource": image,
            "imageAlt": str(meta.get("image-alt") or f"Thumbnail: {title}").strip(),
        }
        if image and not (THUMB_DIR / f"{slug}.jpg").is_file():
            thing["image"] = image  # no thumb yet: fall back to the original
        # Article voice-overs are generated locally (*.wav is gitignored), so the
        # player only appears for the ones that actually shipped.
        # `articles/*/audio/*.wav` is gitignored, so only formats that can ship
        # are considered here.
        audio = sorted(
            path
            for ext in ("mp3", "m4a", "ogg", "opus")
            for path in qmd.parent.glob(f"audio/*.{ext}")
        )
        thing["audio"] = str(audio[0].relative_to(ROOT)) if audio else None
        records.append(thing)

    records.sort(key=lambda a: a["date"], reverse=True)
    return records, labels_by_id


def build_thumbs(records: list[dict]) -> None:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        sys.exit("Pillow is required for --thumbs: conda run -n cetagostini_site python "
                 f"{Path(__file__).name} --thumbs")

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = record["imageSource"]
        if not source:
            continue
        dest = THUMB_DIR / f"{record['slug']}.jpg"
        with Image.open(ROOT / source) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            # Bias the crop slightly above centre: article art is top-weighted.
            img = ImageOps.fit(img, (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS, centering=(0.5, 0.42))
            img.save(dest, format="JPEG", quality=THUMB_QUALITY, optimize=True, progressive=True)
        print(f"  {source} -> {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1024:.0f}KB)")


def main() -> int:
    pass_lang = resolve_lang(os.environ)
    if pass_lang is None:
        print("Articles network: skipped (a dump pass writes no site).")
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en",) + LANGS, default=pass_lang,
                        help="language to build (default: from QUARTO_PROFILE)")
    parser.add_argument("--output-dir",
                        help="tree to write articles-network.json into "
                             "(default: the tree this render pass wrote)")
    parser.add_argument("--thumbs", action="store_true",
                        help="also (re)build images/network/<slug>.jpg with Pillow")
    args = parser.parse_args()

    if args.lang != "en" and args.thumbs:
        parser.error("--thumbs crops images/network/ from the English sources and every "
                     "language shares those files; run it without --lang")

    records, labels_by_id = load_articles()
    if not records:
        sys.exit("No articles found — refusing to write an empty network.")

    if args.thumbs:
        build_thumbs(records)
        for record in records:  # thumbs now exist for every source image
            if record["imageSource"]:
                record["image"] = f"images/network/{record['slug']}.jpg"

    translated = args.lang != "en"
    compiled = compiled_dir(args.lang) if translated else None
    site = load_compiled(compiled / "site.json") if compiled else {}
    if compiled:
        localize(records, site, compiled)
    topic_labels = site.get("topics") or {}

    topics: dict[str, int] = {}
    for record in records:
        for topic_id in record["topics"]:
            label = labels_by_id[topic_id]
            topics[label] = topics.get(label, 0) + 1

    payload = {
        "generated": dt.date.today().isoformat(),
        "articles": records,
        "topics": [
            # The id stays slugified from the English label in both languages: it
            # is what each article record points at and what the page filters on.
            {"id": slugify(label), "label": topic_labels.get(label, label), "count": count}
            for label, count in sorted(topics.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "years": sorted({record["year"] for record in records}, reverse=True),
    }

    out_dir = Path(args.output_dir).resolve() if args.output_dir else output_dir(args.lang, os.environ)
    out_path = out_dir / OUT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  {out_path}  ({len(records)} articles, {len(topics)} topics, {args.lang})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
