# Adding another language

How to add a language to this site. Spanish (`es`) and Portuguese (`pt`, pt-PT)
are the worked examples; every step below generalizes them to a new code,
written `XX`.

Read this together with `AGENTS.md` §11 ("Multilingual (EN/ES/PT) build"), which
documents the mechanics and the gotchas the hard way.

---

## 1. The model

English is the source. Every other language is a **mirror tree** rendered from
the *same* `.qmd` files by a Quarto profile. Nothing is duplicated by hand:

```
articles/<slug>/<slug>.qmd        (English, never edited for translation)
        │
        ├── quarto render                     -> docs/          (English)
        └── quarto render --profile XX        -> docs/XX/       (language XX)

text substitution happens at render time, from dictionaries
i18n/XX/**  (YAML, reviewed)  ->  i18n/XX/compiled/**  (JSON, generated)
```

The filter decides what to replace by hashing the **runtime** pandoc AST, so a
dictionary key only matches if the same normalizer produced it. Extraction and
translation share one implementation — see `filters/translate.lua`.

---

## 2. Touchpoints for a new language `XX`

Every language is registered in one place per file: a `LANGS` tuple (Python /
Lua) or the language table in the JS. Add `"XX"` to each, then copy the `es`
files.

| # | File | What to add for `XX` |
|---|---|---|
| 1 | `_quarto-XX.yml` (copy `_quarto-pt.yml`) | `lang: XX`, `output-dir: docs/XX`, `site-url: …/XX`, translated `title`; the pre/post-render hook list (`--lang XX`) |
| 2 | `_quarto-XX-dump.yml` (copy `_quarto-pt-dump.yml`) | `output-dir: _i18n_dump` (same disposable dir); no pre-render |
| 3 | `i18n/XX/site.yml` (copy `i18n/pt/site.yml`) | months, topic labels, SEO/nav labels, banner, footer strings |
| 4 | `i18n/XX/**` | created by the extractor in step 5 — do not hand-write |
| 5 | `filters/translate.lua` | `LANGS` — mode detection and the two `i18n/<lang>/` paths follow from it |
| 6 | `filters/llm-seo.lua` | `LANGS`, `HREFLANG` (BCP47 tag), `LABELS` (nav + site title) |
| 7 | `scripts/i18n_extract.py` | nothing to add — `--lang XX` is the whole switch |
| 8 | `scripts/i18n_coverage_gate.py` | nothing to add — `--lang XX`; a language with a genuinely thinner dictionary needs its own floor |
| 9 | `scripts/i18n_listing_rewrite.py` | `LANGS` |
| 10 | `generate_sitemap.py` | `LANGS` + `HREFLANG` |
| 11 | `js/build-llms-md.py` | `LANGS`, `SITE_TITLE["XX"]`, `LLMS_SOURCE["XX"]` |
| 12 | `generate_articles_network.py` | `LANGS` (`--lang` choices follow) |
| 13 | `scripts/assert_bilingual_tree.py`, `scripts/write_i18n_marker.py`, `scripts/render-all.sh` | `LANGS` / the `LANGS=(…)` array |
| 14 | `js/site-i18n.js` | `LANG` detection + the `pt:`-style column in every dictionary entry |
| 15 | `js/language-switcher.js` | `LANG_NAMES` + `LANG_WORD` |
| 16 | `_quarto.yml` | one more entry in the navbar language menu |
| 17 | `llms-XX.txt` | curated Markdown index in `XX` (copy `llms-pt.txt`) |
| 18 | `scripts/tests/**` | extend the dispatch assertions with the new token |

**One place does NOT follow `LANGS`: `js/language-switcher.js` and
`js/site-i18n.js` read the language off `<html lang>`**, because they run in the
browser on a rendered page. `pt-PT` is matched by prefix, so any regional
variant of a supported language is picked up.

---

## 3. What is already generalized (and what is not)

The three-language work replaced the old two-language assumptions. What each of
them does now:

- **`js/language-switcher.js`** — no longer computes "the other language". It
  reads every `hreflang` alternate, maps each one onto the navbar menu entry
  that points at that tree, rewrites the entry to *this* page's counterpart and
  marks the current one with `aria-current`. The toggle becomes a globe whose
  accessible name names the current language. A fourth language needs only a
  `LANG_NAMES` entry.
- **`filters/llm-seo.lua`** — emits one `hreflang` alternate per entry in
  `LANGS`, plus `x-default` → English, and derives the canonical URL, the Blog
  node URL, the breadcrumb items and every `inLanguage` from the profile.
- **`generate_sitemap.py`** — one `<url>` per route per tree that exists, each
  carrying the full alternate set. Any language pass rewrites the file for every
  tree built so far, so a partial build still gets a consistent sitemap.
- **`scripts/render-all.sh`** — one pass per language, English first, with the
  incomplete-build trap and per-language tree assertions.
- **Coverage gate** — `--lang` selects the tree; floors are per language;
  `I18N_ALLOW_PARTIAL=1` bypasses them for local iteration only.
- **Markers** — `.i18n-<lang>-built`, one per language, written by
  `scripts/write_i18n_marker.py` and checked by `assert_bilingual_tree.py`.

Still language-specific, by design:

- **`i18n/XX/site.yml` is not read by anything yet.** No script compiles it into
  `i18n/XX/compiled/site.json`, which is what `generate_articles_network.py`
  (`months`, `topics`) and `scripts/i18n_listing_rewrite.py` (`categories`,
  `ui`) expect. Until that compile step exists, a language's network months and
  topic labels, and its diary listing chips, stay English — for `es` as well.
  The `categories`/`ui` sections the consumers want are not in `site.yml` yet.

---

## 4. Procedure

```bash
# 0. envs (renders abort without the key)
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
cd <repo>; set -a && . ./.env && set +a

# 1. profiles + glossary
cp _quarto-pt.yml      _quarto-XX.yml          # then: lang, output-dir, site-url, title
cp _quarto-pt-dump.yml _quarto-XX-dump.yml
mkdir -p i18n/XX && cp i18n/pt/site.yml i18n/XX/site.yml   # then translate its values

# 2. teach the code about XX  (see the table in §2)

# 3. extract: dump the runtime AST for every page, then build skeletons
env -u QUARTO_PROFILE conda run -n cetagostini_site quarto render --profile XX-dump
conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang XX

# 4. translate i18n/XX/** — fill the translated values.
#
#    NOTE the schema field holding a translation is literally named `es`
#    (YAML `es:` entries, JSON `"es":`, `blocks[].es`, `meta.*.es`). It means
#    "the target text", not "Spanish". Two options for a new language:
#      (a) leave the name alone — zero code change, slightly confusing; or
#      (b) rename it to something language-neutral (e.g. `target`) across
#          filters/translate.lua, scripts/i18n_extract.py,
#          scripts/i18n_coverage_gate.py and scripts/tests/**. Do it before
#          translating, not after.
#    Either way the *directory* (`i18n/XX/`) carries the language. `pt` took
#    option (a); `scripts/i18n_extract.py` and `scripts/i18n_coverage_gate.py`
#    name the field in one constant (`TARGET`) so option (b) is now a rename of
#    two literals plus the dictionaries.

# 5. validate + compile
conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang XX --check
conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang XX --compile

# 6. build every tree (English first)
bash scripts/render-all.sh
```

Then check the results — see §6.

---

## 5. What the pipeline guarantees (and will fail loudly on)

| Gate | Catches |
|---|---|
| `--check` | a `.qmd` edited since the dump (`qmd_sha256`), a dump entry with no dictionary entry, an active entry no longer in the dump, a differing Quarto version (warning) |
| `--compile` | protected-token damage (code/math/`@fig-` refs/URLs), header-level changes, `##` inside a callout title, raw-HTML structure drift, non-string scalars |
| static coverage floor | dictionaries that are too incomplete to publish |
| runtime gate (post-render) | a render that matched nothing — i.e. missing or wholly stale compiled JSON |
| `assert_bilingual_tree.py` + `render-all.sh` trap | a bare `quarto render` that would delete `docs/<lang>/`, a leaked `QUARTO_PROFILE`, a build interrupted between passes |

`--check` also catches **project-config drift**: a change to `_quarto.yml`, a
filter or `_includes/` moves navbar/sidebar envelope render-ids, so the dump no
longer matches the dictionaries. Re-dump after any such change.

---

## 6. Validation checklist

```bash
# every tree exists
ls docs/index.html docs/es/index.html docs/XX/index.html

# language attribute + canonical + alternates
grep -o '<html[^>]*lang="[^"]*"' docs/XX/index.html
grep -o 'rel="canonical" href="[^"]*"' docs/XX/index.html
grep -o 'hreflang="[^"]*"' docs/XX/index.html | sort -u

# social tags must be in <head> and in the target language
python3 - <<'EOF'
h=open('docs/XX/index.html').read(); he=h.find('</head>')
for p in ('og:title','og:description','twitter:title'):
    i=h.find(p); print(p, 'HEAD' if 0 < i < he else 'BODY/MISSING')
EOF

# navbar: one menu entry per language, each labelled in its own language
#   - one chip per category, same count as English
#   - no run-on strings like [MMMpythonprior elicitation…]

# no nested-output pollution and no re-execution
test -d docs/XX/docs && echo "POLLUTION" || echo "clean"
git diff --name-only origin/main -- _freeze   # must be empty: freeze replayed, not re-run

# tests
conda run -n cetagostini_site python3 -m unittest discover -s scripts/tests -t .
```

---

## 7. Gotchas that cost real time

1. **Never single-file-render an article** (`quarto render articles/x/x.qmd`).
   Quarto *always executes* incremental renders, which rewrites `_freeze` and
   re-runs expensive MCMC. Use the whole-project wrapper.
2. **English first.** The English pass deletes every `docs/<lang>/` subtree;
   then each language pass rebuilds its own.
3. **Profile lists concatenate.** `navbar`, `page-footer`, `filters`,
   `post-render` are *appended*, not replaced, and `!override` is unsupported.
   Never declare them in a language profile.
4. **`QUARTO_DOCUMENT_FILE` / `_PATH` are stale for freeze-replayed documents.**
   Derive the page identity from the working directory plus
   `PANDOC_STATE.output_file`.
5. **Extract from the runtime AST**, not `quarto pandoc -t json`: Quarto turns
   callouts into `__quarto_custom_type=Callout` scaffolds and figures into
   `FloatRefTarget` scaffolds, and `pandoc.write` needs a `Pandoc({…})` wrapper.
6. **`og:`/`twitter:` and navbar labels ignore `doc.meta`.** They are computed
   before user filters; rewrite the spans inside Quarto's hidden
   `#quarto-meta-markdown` / `#quarto-navigation-envelope` envelopes. A navbar
   *menu* contributes one envelope entry per entry label **and** per href (the
   href-keyed ones are dropped by `filter_envelope`), so adding or renaming a
   navbar entry invalidates every page's envelope.
7. **Language names are labelled in themselves.** `English` / `Español` /
   `Português` keep their own spelling on every tree: their dictionary entries
   stay empty. Only the menu toggle's word ("Language"/"Idioma") is translated.
8. **Anchor every `project.resources` glob with `./`**, or the nested output dir
   re-copies the rendered English tree into `docs/XX/docs/`.
9. **Category keys stay English-derived** (base64 of `quote(label, safe='')`);
   translating the keys breaks the listing filter. Labels only.
10. **Listing cards** render from Quarto's listing pipeline — they arrive as
    ordinary text blocks, so translate them there; sidebar dates/chrome are
    localized natively by `lang:`.
11. **`pandoc.json.decode` maps JSON `null` to a userdata sentinel** — type-check
    every value before using it.
12. **No icon fonts.** Use inline SVG for any icon.
13. **The freeze is shared across languages**: replaying costs seconds, executing
    costs hours. Keep `.qmd` sources byte-identical.
14. **The extractor re-serializes every skeleton it touches.** The committed
    dictionaries were written by an older emitter, so the first
    `i18n_extract.py` run after a dump refresh rewrites all of them in the
    pipeline's current style (block literals instead of folded quoted scalars).
    The parsed content is unchanged — diff the loaded YAML, not the file, when
    reviewing such a change.
