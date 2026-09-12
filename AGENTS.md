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
index.qmd              # Home (causal-field hero + selected writing)
about.qmd              # About (editorial hero, career DAG rail, line-delimited lists)
articles.qmd           # Articles listing
articles/<slug>/<slug>.qmd   # individual articles (notebooks)
talks.qmd              # Talks (single-card infinite video carousel + lightbox)
diary.qmd              # Diary listing (contents: diary)
diary/<YYYY-MM-DD>.qmd # diary entries (auto-listed, newest first)
diary/_metadata.yml    # defaults for diary entries
filters/llm-seo.lua    # JSON-LD structured-data filter (Article/Person/Video/...)
js/                    # hero-dag.js, career-rail.js, cookie-consent.js,
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

### Editorial wide page (home, about)

`index.qmd` and `about.qmd` opt out of the article layout and share one visual system:

```yaml
body-classes: home-page        # or about-page
format:
  html:
    title-block: false         # title block stays in the DOM but is hidden by CSS
    page-layout: full
    toc: false
    anchor-sections: false
```

Both pages are written as one raw-HTML block (` ```{=html} `) wrapped in
`.home-shell` / `.about-shell`, and both rely on the shared page tokens in
`styles.css`: `--page-gutter` (side padding), `.page-section` (hairline-topped
cream band), `.section-heading` + `.section-eyebrow`, `.hero-*`, `.btn-*`.
Keep new wide pages inside that vocabulary instead of inventing container names.

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
- Cards (`.article-preview`, listing cards) use `--surface` bg, `--line` border,
  `--shadow`, hover lift + green left-accent. The wide editorial pages prefer
  **lines over boxes**: hairline rules (`--brown` / `--line`) with hover colour shifts
  (`.home-card`, `.rule-card`, `.rule-list`, `.about-strip`).

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
- `js/hero-dag.js` — home causal-field engine: builds the drifting nodes/edges, specks,
  cursor mesh, pulse and the pause/resume controls from the static SVG in `index.qmd`.
  No-ops unless `.home-shell` + `.dag-stage` exist.
- `js/career-rail.js` — About career rail. The roles are a `role="tablist"` of buttons;
  the graph (dots, wires, arrowheads, leader line) is drawn in SVG from measured DOM
  positions, so the same rail is horizontal on wide screens and vertical below 992px.
  It sets `[data-career-ready]`, which is what switches the panels from stacked-in-flow
  (no-JS fallback) to one floating panel.
- `js/video-carousel.js` — Talks single-card infinite carousel + lightbox.
- `js/cookie-consent.js` — cookie consent popup.

## 9. Accessibility

- All animations (home causal field, About ambient field, career rail, carousel) are
  disabled under `@media (prefers-reduced-motion: reduce)`.
- The career rail is a `role="tablist"` with roving tabindex: Arrow keys / Home / End move
  between roles, the panels are `role="tabpanel"`, and inactive panels use
  `visibility: hidden` so they leave the accessibility tree. Carousel cards are buttons;
  the lightbox is `role="dialog" aria-modal` with Esc-to-close.
- Images have alt text. The About field is `aria-hidden` decoration; the rail carries the
  career structure itself, so there is no duplicate visually-hidden transcript.
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
- **Render the whole project.** `quarto render <file>.qmd` cleans `docs/` first and deletes
  every other page's output. Always run plain `quarto render`.
- **`MIMO_API_KEY` must be in the environment** or the render aborts during profile setup
  (`MissingEnvVarsError`, from `.env.example`). `set -a && . ./.env && set +a` before
  rendering; a fresh worktree has no `.env` (it is gitignored).
- The conda env (`cetagostini_web`) is only needed to re-execute notebooks; normal renders
  use `_freeze` and don't need it. `articles/alchemize_pytensor_mlx_gemma_3n` sets
  `eval: false, freeze: false` so a full render never executes the MLX code.
