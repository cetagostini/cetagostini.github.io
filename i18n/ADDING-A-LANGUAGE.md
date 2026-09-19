# Adding another language

How to add a third (or fourth) language to this site. Spanish (`es`) is the
worked example; every step below generalizes it to a new code, written `XX`.

Read this together with `AGENTS.md` §11 ("Bilingual (EN/ES) build"), which
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

Everything below currently names `es` explicitly. Use the greps to find the
exact lines (`grep -rn '"es"\|i18n/es\|docs/es\|es-dump' --include='*.lua'
--include='*.py' --include='*.js' --include='*.yml' --include='*.sh' .`).

| # | File | What to add for `XX` |
|---|---|---|
| 1 | `_quarto-XX.yml` (copy `_quarto-es.yml`) | `lang: XX`, `output-dir: docs/XX`, `site-url: …/XX`, translated `title`; keep the pre/post-render hook list |
| 2 | `_quarto-XX-dump.yml` (copy `_quarto-es-dump.yml`) | `output-dir: _i18n_dump` (same disposable dir); no pre-render |
| 3 | `i18n/XX/site.yml` (copy `i18n/es/site.yml`) | months, topic labels, SEO/nav labels, banner, footer strings |
| 4 | `i18n/XX/**` | created by the extractor in step 6 — do not hand-write |
| 5 | `filters/translate.lua` | mode detection: add the `XX-dump` and `XX` tokens; paths: `i18n/XX/compiled/…` and `i18n/XX/_extracted/…` (3 literals) |
| 6 | `filters/llm-seo.lua` | token check, `LANG`, the `XX/` URL prefix, one more `hreflang` alternate, breadcrumb labels |
| 7 | `scripts/i18n_extract.py` | `--lang XX`; the `i18n/es` path constants it uses for records/skeletons/compiled |
| 8 | `scripts/i18n_coverage_gate.py` | the `i18n/es` compiled/stats dirs and the per-language floor |
| 9 | `scripts/i18n_listing_rewrite.py` | `ES_DIRNAME = "es"` → the new code, and its language detection |
| 10 | `generate_sitemap.py` | `ES_DIRNAME`, the alternate set, and the target tree list |
| 11 | `js/build-llms-md.py` | the curated index filename (`llms-XX.txt`) and the per-language site title used to strip title suffixes |
| 12 | `generate_articles_network.py` | `--lang XX` output dir + localized month/topic labels |
| 13 | `scripts/assert_bilingual_tree.py`, `scripts/write_es_marker.py`, `scripts/render-all.sh` | the marker name, the per-language passes, the guard's expectations |
| 14 | `js/site-i18n.js` | a string table for `XX` |
| 15 | `js/language-switcher.js` | see §3 — this one assumes exactly two languages |
| 16 | `_quarto.yml` | a static navbar link to the new tree (the server-rendered fallback) |
| 17 | `llms-XX.txt` | curated Markdown index in `XX` (copy `llms-es.txt`) |
| 18 | `scripts/tests/**` | extend the dispatch assertions with the new token |

---

## 3. The two-language assumption (read before adding a third)

These are built for exactly English + one other language. A third language
needs them generalized:

- **`js/language-switcher.js`** — computes "the other language" from the
  `hreflang` alternates and renders a single globe link. With 3+, switch back to
  a dropdown listing every language. Draw the icon as **inline SVG**; this site
  does **not** load the bootstrap-icons font, so `<i class="bi bi-globe">`
  renders as an invisible blank (that was a real bug).
- **`filters/llm-seo.lua`** — emits `hreflang="en"` and `hreflang="es"` plus
  `x-default` → English. Add one alternate per language, keeping `x-default`.
- **`generate_sitemap.py`** — same: alternates per language.
- **`scripts/render-all.sh`** — currently two passes (English, then Spanish).
  Each language is one more pass, and **English must stay first** (its render
  deletes every `docs/<lang>/` subtree).
- **Coverage gate** — floors are per language; `I18N_ALLOW_PARTIAL=1` bypasses
  them for local iteration only.

---

## 4. Procedure

```bash
# 0. envs (renders abort without the key)
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
cd <repo>; set -a && . ./.env && set +a

# 1. profiles + glossary
cp _quarto-es.yml      _quarto-XX.yml          # then: lang, output-dir, site-url, title
cp _quarto-es-dump.yml _quarto-XX-dump.yml
mkdir -p i18n/XX && cp i18n/es/site.yml i18n/XX/site.yml   # then translate its values

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
#    Either way the *directory* (`i18n/XX/`) carries the language.

# 5. validate + compile
conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang XX --check
conda run -n cetagostini_site python3 scripts/i18n_extract.py --lang XX --compile

# 6. build both trees (English first)
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

---

## 6. Validation checklist

```bash
# both (all) trees exist
ls docs/index.html docs/XX/index.html

# language attribute + canonical + alternates
grep -o '<html[^>]*lang="[^"]*"' docs/XX/index.html
grep -o 'rel="canonical" href="[^"]*"' docs/XX/index.html
grep -c 'hreflang=' docs/XX/index.html

# social tags must be in <head> and in the target language
python3 - <<'EOF'
h=open('docs/XX/index.html').read(); he=h.find('</head>')
for p in ('og:title','og:description','twitter:title'):
    i=h.find(p); print(p, 'HEAD' if 0 < i < he else 'BODY/MISSING')
EOF

# navbar + tags
#   - one chip per category, same count as English
#   - one globe/link to change language
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
2. **English first.** The English pass deletes `docs/<lang>/` subtrees; then each
   language pass rebuilds its own.
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
   `#quarto-meta-markdown` / `#quarto-navigation-envelope` envelopes.
7. **Anchor every `project.resources` glob with `./`**, or the nested output dir
   re-copies the rendered English tree into `docs/XX/docs/`.
8. **Category keys stay English-derived** (base64 of `quote(label, safe='')`);
   translating the keys breaks the listing filter. Labels only.
9. **Listing cards** render from Quarto's listing pipeline — they arrive as
   ordinary text blocks, so translate them there; sidebar dates/chrome are
   localized natively by `lang:`.
10. **`pandoc.json.decode` maps JSON `null` to a userdata sentinel** — type-check
    every value before using it.
11. **No icon fonts.** Use inline SVG for any icon.
12. **The freeze is shared across languages**: replaying costs seconds, executing
    costs hours. Keep `.qmd` sources byte-identical.
