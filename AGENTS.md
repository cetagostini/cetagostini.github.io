# AGENTS.md — Maintainer guide for the cetagostini.github.io repo

This is a [Quarto](https://quarto.org) website for Carlos Trujillo (marketing/causal-inference
researcher), deployed to GitHub Pages from the committed `docs/` folder. This file is the
operating manual for any LLM (or human) working in the repo.

## 0. Golden rules

1. **Always run `quarto render` after editing source and commit the regenerated `docs/`.**
   CI does **not** render — it serves the committed `docs/` verbatim. Forgetting this means
   your source change never ships.
2. **Never hand-edit files under `docs/`.** It is generated output. Change the source
   (`.qmd`, `styles.css`, `_quarto.yml`, filters, JS) and re-render.
3. **Use the CSS design tokens** (variables in `styles.css` `:root`), never hardcode hex
   colors. The palette and contrast ratios are intentional.
4. **`section` is a reserved Quarto frontmatter field.** Use `schema-section` for custom
   per-page schema flags.
5. **Quarto listings use `contents:`, not `path:`.**
6. **On the home page, do not use a markdown `#` H1** for the name — Quarto promotes it into
   a title-block at the top. Use `<h2 class="hero-name">` (already wired up).

## 1. Quick start

```bash
quarto render          # build the site into docs/ (fast — uses _freeze)
quarto preview         # local dev server (watches + hot-reloads)
bash quarto-rebuild.sh           # render + preview
bash quarto-rebuild.sh --clean   # wipe _freeze/.quarto, re-execute everything, then preview
```

`quarto preview` serves on http://localhost:4321 by default. If that port is taken by
another app, use `--port 4323` and open `http://127.0.0.1:4323/` (IPv4 — `localhost` may
hit a conflicting IPv6 service).

## 2. Conda environment

- Env name: **`cetagostini_web`** (declared in `_quarto.yml` → `execute: conda`).
- Python + Jupyter. Article notebooks (under `articles/`) execute Python (PyMC, etc.).
- `execute: freeze: true` — Quarto caches computed outputs in `_freeze/`. Normal
  `quarto render` reuses the cache and does **not** re-run kernels, so it's fast and
  works without the env being fully set up. Use `--clean` only when you must re-execute.
- Pillow is installed (used by `scripts/optimize_images.py`).

## 3. Project structure

```
_quarto.yml            # site config (navbar, footer, theme, fonts, filters, post-render)
styles.css             # all custom CSS (design tokens + components)
index.qmd              # Home
about.qmd              # About (career DAG + accordion experience cards)
articles.qmd           # Articles listing
articles/<slug>/<slug>.qmd   # individual articles (notebooks)
talks.qmd              # Talks (single-card infinite video carousel + lightbox)
diary.qmd              # Diary listing (contents: diary)
diary/<YYYY-MM-DD>.qmd # diary entries (auto-listed, newest first)
diary/_metadata.yml    # defaults for diary entries
filters/llm-seo.lua    # JSON-LD structured-data filter (Article/Person/Video/...)
js/                    # hero-dag.js, experience-cards.js, cookie-consent.js,
                       #   video-carousel.js, build-llms-md.py (post-render)
scripts/optimize_images.py   # Pillow image optimizer
generate_sitemap.py    # sitemap generator
llms.txt               # curated LLM index (copied to docs/ by post-render)
docs/                  # GENERATED output — committed, served by GitHub Pages
```

## 4. How to create pages

### Top-level page
Create `foo.qmd` with frontmatter:
```yaml
---
title: "Page Title"
pagetitle: "Page Title — Carlos Trujillo"   # <title> tag
description: "Short SEO summary."            # → <meta name="description"> (hidden on page)
---
```
Add it to the navbar in `_quarto.yml` → `website.navbar.right`.

### Article
Create `articles/<slug>/<slug>.qmd`:
```yaml
---
title: "Article Title"
author: "Carlos Trujillo"          # or list of {name: ...} for co-authors
date: "2026-04-07"
description: "One-line summary."
categories: [python, bayesian, causal]
image: "../images/<thumb>.png"
format:
  html:
    code-fold: true
    code-tools: true
---
```
The Lua filter auto-emits `Article` + `BreadcrumbList` JSON-LD (URL reconstructed as
`articles/<slug>/<slug>.html`). Add the article to `articles.qmd` listing + the
"All Articles" list. Add a thumbnail to `images/`.

### Diary entry
Create `diary/YYYY-MM-DD.qmd`:
```yaml
---
title: "Entry title"
date: "2026-07-05"
description: "One-line summary."
categories: [meta, site]
schema-section: diary        # REQUIRED — tells the Lua filter this is a diary Article
---
Body in markdown…
```
It auto-appears on `diary.html` (newest first) and gets `Article` + `BreadcrumbList`
JSON-LD with URL `diary/<slug>.html`. No other wiring needed.

## 5. Styling rules

- **Design tokens** live in `styles.css` `:root`. Use them:
  - `--bg #FDF6ED` (soft white) · `--green #778873` · `--green-soft #A1BC98` · `--brown #DCCFC0`
  - Derived AA text shades: `--green-strong #4F6B4A`, `--ink #2B2A26`, `--ink-muted #6B665C`,
    `--brown-strong #6B5A48`, `--surface #FFFFFF`, `--surface-2 #F2EDE3`, `--line #E6DFD2`.
  - `--radius`, `--shadow`, `--ease`, `--maxw`.
- **Fonts**: Manrope SemiBold (600) for headings; Inter Regular (400) for body. Loaded via
  Google Fonts in `_quarto.yml` (`display=swap`). No serif/Newsreader.
- **No gradient text anywhere.** Green is used as left-accent bars / underline-grow; brown
  for rules/dividers; `--brown-strong` for heading ink.
- **`--green` (#778873) fails AA for normal-size text** (3.5:1) — only use it for large text
  / fills / non-text. For body links/small green text use `--green-strong` (#4F6B4A, 5.5:1).
- The `description` frontmatter renders a visible subtitle; it's hidden via
  `.quarto-title-block .description { display: none; }` but kept in `<meta name="description">`
  for SEO. Don't remove that CSS rule.
- Cards (`.experience-card`, `.article-preview`, `.skills-card`, listing cards) share a
  pattern: `--surface` bg, `--line` border, `--shadow`, hover lift + green left-accent.

## 6. Build & deploy

- **Output dir:** `docs/` (set in `_quarto.yml` → `project.output-dir`).
- **Deploy:** push to `main` → `.github/workflows/deploy-static.yml` uploads `docs/` to
  GitHub Pages. There is **no render in CI** — commit the regenerated `docs/`.
- **PR check:** `.github/workflows/quarto-publish.yml` runs a build artifact check on PRs
  to main (does not deploy).
- **Sitemap:** `generate_sitemap.py` regenerates `docs/sitemap.xml` from `.qmd` files.
  `robots.txt` points to `https://cetagostini.github.io/sitemap.xml`.
- After any source change: `quarto render` → review `docs/` → commit → push.

## 7. LLM-friendly layer

- **`/llms.txt`** (source: `llms.txt` at root) — curated Markdown index of the site's
  content per the [llms-txt](https://llmstxt.org/) spec. Hand-maintain it when adding
  major pages. The post-render script copies it to `docs/llms.txt`.
- **`.md` mirrors** — `js/build-llms-md.py` (runs as `project.post-render`) extracts the
  `<main>` content of each page and writes a clean GFM mirror at `<page>.html.md`
  (strips nav/footer/script/svg). Google ignores llms.txt but other LLMs use these.
- **JSON-LD** — `filters/llm-seo.lua` (registered in `_quarto.yml` → `filters`) reads
  frontmatter and injects schema.org JSON-LD:
  - `about.html` → `Person` + `ProfilePage`
  - `articles/**` → `Article` + `BreadcrumbList` (ISO `datePublished`, multi-author,
    `articleSection` from first category)
  - `diary/**` (flagged `schema-section: diary`) → `Article` + `BreadcrumbList`
    (URL `diary/<slug>.html`)
  - `diary.html` → `CollectionPage`
  - `talks.html` → one `VideoObject` per `data-embed` card
  - It reads `PANDOC_STATE.output_file` (basename) + metadata. When adding a new page
    type, extend the filter's branches.

## 8. Scripts & JS

- `scripts/optimize_images.py` — Pillow resizer (profile photo: 800px/q80). Extend `TARGETS`
  to optimize more images.
- `js/build-llms-md.py` — post-render llms.txt copy + `.md` mirror generation (pandoc).
- `js/hero-dag.js` — home hero cursor→node connector lines (reduced-motion + touch guards).
- `js/experience-cards.js` — About experience accordion (toggles `.is-open` + `aria-expanded`).
- `js/video-carousel.js` — Talks single-card infinite carousel + lightbox.
- `js/cookie-consent.js` — cookie consent popup.

## 9. Accessibility

- All animations (DAG hero, carousel, cursor) are disabled under
  `@media (prefers-reduced-motion: reduce)`.
- Accordion uses `<button aria-expanded>`; carousel cards are buttons; lightbox is
  `role="dialog" aria-modal` with Esc-to-close.
- Images have alt text. The career DAG has `role="img"` + `<title>`/`<desc>` + a
  visually-hidden text alternative.
- Skip-to-content link is the first focusable element.

## 10. Common gotchas

- **Port conflict on `localhost:4321`** — another app may hold IPv6. Use `--port 4323`
  and `http://127.0.0.1:4323/`.
- **`#` H1 on the home page** gets promoted to a title-block, separating the name from
  the subtitle. Use `<h2 class="hero-name">` + `title-block: false` in frontmatter.
- **`section:` frontmatter is reserved** by Quarto — use `schema-section`.
- **Listings use `contents:`, not `path:`.**
- **`string.gsub` returns two values** (string + count) — wrap in parens
  `(s:gsub(...))` before passing to `table.insert`, or it's read as a position arg.
- **`pandoc.utils.type` returns `"List"`** for both `MetaList` and `MetaInlines` in this
  pandoc — don't rely on `.t == "MetaList"`; iterate `MetaList` elements and stringify.
- The conda env (`cetagostini_web`) is only needed to re-execute notebooks; normal renders
  use `_freeze` and don't need it.
