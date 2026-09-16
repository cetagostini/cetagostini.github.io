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

## 2. Conda environments (one per article)

Envs are named **exactly after the article slug**, defined by an `environment.yml`
inside each article folder, and registered as Jupyter kernels with the same name:

| Env / kernel | Defined by | Used by |
|---|---|---|
| `cetagostini_site` | `environment.yml` (root) | Quarto project engine (`execute.conda`), post-render scripts, every page without its own kernel (index, about, diary, talks, listings) |
| `<slug>` (×8) | `articles/<slug>/environment.yml` | the article's `.qmd`, via `jupyter: <slug>` in its frontmatter |

- Provision all envs + kernels (idempotent): `bash scripts/setup_envs.sh`
  (`--recreate` to rebuild every env from its yml).
- After changing packages in an env, re-export its yml so the committed spec stays
  the source of truth: `python3 scripts/export_envs.py [<slug> ...]`.
  Exports strip machine-specific content (editable installs, local paths) and keep
  `git+https` pip deps as URLs.
- Articles are self-contained folders: `.qmd` + `environment.yml` + data/images/audio.
- `execute: freeze: true` — Quarto caches computed outputs in `_freeze/`. Most renders
  reuse the cache and never start an article kernel. Use `--clean` only when you must
  re-execute. Render from any shell; Quarto picks the kernel per page.
- Pillow lives in `cetagostini_site` (used by `scripts/optimize_images.py` and
  `generate_articles_network.py --thumbs`).

## 3. Project structure

```
_quarto.yml            # site config (navbar, footer, theme, fonts, filters, post-render)
styles.css             # all custom CSS (design tokens + components)
index.qmd              # Home (causal-field hero + selected writing)
about.qmd              # About (editorial hero, career DAG rail, line-delimited lists)
articles.qmd           # Articles index (interactive topic network + year list)
articles/<slug>/<slug>.qmd   # individual articles (notebooks)
talks.qmd              # Talks (single-card infinite video carousel + lightbox)
diary.qmd              # Diary listing (contents: diary)
diary/<YYYY-MM-DD>.qmd # diary entries (auto-listed, newest first)
diary/_metadata.yml    # defaults for diary entries
filters/llm-seo.lua    # JSON-LD structured-data filter (Article/Person/Video/...)
js/                    # hero-dag.js, career-rail.js, articles-network.js,
                       #   cookie-consent.js, video-carousel.js,
                       #   build-llms-md.py (post-render)
images/network/        # square article thumbnails for the Articles network (committed)
scripts/optimize_images.py   # Pillow image optimizer
generate_sitemap.py    # sitemap generator
generate_articles_network.py # Articles network data + thumbnails (post-render)
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
jupyter: <slug>                    # REQUIRED — the article's own kernel/env
format:
  html:
    code-fold: true
    code-tools: true
---
```
Each article folder is self-contained: `.qmd` + `environment.yml` + data/images/audio.
Bootstrap its env from the closest existing one (usually the base stack), e.g.:
```bash
conda create -n <slug> --clone cetagostini_web        # or another article's env
conda run -n <slug> python -m ipykernel install --user --name <slug> --display-name "Python (<slug>)"
python3 scripts/export_envs.py <slug>                 # writes articles/<slug>/environment.yml
```
Or hand-write `articles/<slug>/environment.yml` and run `bash scripts/setup_envs.sh`.
The Lua filter auto-emits `Article` + `BreadcrumbList` JSON-LD (URL reconstructed as
`articles/<slug>/<slug>.html`).

Wiring an article into the Articles page:
1. `image:` should be **site-relative** (`/images/<thumb>.jpg`). A `../images/...` value
   resolves against `articles/<slug>/` and silently breaks the page's `og:image`; the
   network generator still finds it, but fix the frontmatter when you touch the file.
2. Run `conda run -n cetagostini_site python generate_articles_network.py --thumbs`
   to (re)build `images/network/<slug>.jpg` and `docs/articles-network.json`. Commit both.
3. Add the article to the year list in `articles.qmd` (the section between the network
   and the closing strip). The network itself picks the article up from frontmatter.
4. Normal render (`quarto render`) regenerates `docs/articles-network.json` only.

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

### Editorial wide page (home, about, articles)

`index.qmd`, `about.qmd` and `articles.qmd` opt out of the article layout and share one
visual system:

```yaml
body-classes: home-page        # or about-page / articles-page
format:
  html:
    title-block: false         # title block stays in the DOM but is hidden by CSS
    page-layout: full
    toc: false
    anchor-sections: false
```

They are written as raw-HTML blocks (` ```{=html} `) wrapped in `.home-shell` /
`.about-shell` / `.articles-shell`, and rely on the shared page tokens in `styles.css`:
`--page-gutter` (side padding), `.page-section` (hairline-topped cream band),
`.section-heading` + `.section-eyebrow`, `.hero-*`, `.btn-*`, `:is(.about-strip,
.articles-strip)`. Keep new wide pages inside that vocabulary instead of inventing
container names. The shared shell selectors are written as
`body:is(.home-page, .about-page, .articles-page)` — add the new body class there
rather than duplicating a rule.

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
- **`docs/articles-network.json`** — generated by `generate_articles_network.py`
  (post-render) from article frontmatter: title, date, description, topics, thumbnail.
  `js/articles-network.js` fetches it; the year list in `articles.qmd` is the no-JS
  fallback and the part the `.md` mirror carries.

## 8. Scripts & JS

- `scripts/optimize_images.py` — Pillow resizer (profile photo: 800px/q80). Extend `TARGETS`
  to optimize more images.
- `js/build-llms-md.py` — post-render llms.txt copy + `.md` mirror generation (pandoc).
- `js/hero-dag.js` — home causal-field engine: builds the drifting nodes/edges, specks,
  cursor mesh, pulse and the pause/resume controls from the static SVG in `index.qmd`.
  No-ops unless `.home-shell` + `.dag-stage` exist.
- `js/career-rail.js` — About career DAG. Each dot and label is one native button.
  SVG edges use only `.career-track` dimensions and HTML dot centers; role descriptions
  never participate in diagram geometry. The horizontal track scrolls on narrow screens.
  No description opens initially. Clicking a role moves its existing article into a native
  modal `<dialog>`; Close, Escape, or a backdrop click restores it to source order.
  Previous/next controls browse roles within the dialog. `[data-career-ready]` hides the
  in-flow articles only after initialization; without JS they remain readable. Printing
  restores all six articles, including the one currently open.
- `generate_articles_network.py` — reads `articles/*/*.qmd` frontmatter, canonicalises
  `categories` into topics (`TOPIC_ALIASES`), resolves each `image:`, and writes
  `docs/articles-network.json`. `--thumbs` additionally builds `images/network/<slug>.jpg`
  (Pillow, so run it with the `cetagostini_site` env). It refuses to write an empty network.
- `js/articles-network.js` — Articles page network. The SVG force field hosts two kinds
  of marks: circular article thumbnails and keyword ellipses (one per topic, article
  count below). "By topic" shows the keyword graph: topics linked by the articles they
  share, similarity springs (Jaccard over topics) against long-range repulsion, run in a
  square metric so a wide stage gets a wide field. Activating a keyword (click or
  Enter/Space) breaks the graph open: its articles bloom out of the node into orbit
  around it, bonded by spokes and similarity links, while the other keywords fold away;
  activating it again, Esc or an empty click folds them back. "By date" lays the articles
  on a timeline with year rules, oldest left. Clicking an article breaks the layout (the
  others float and bounce) and opens the summary sheet; closing rebuilds it. Drag pans,
  wheel/pinch zooms, pulses run along the links, and `[data-network-ready]` marks
  initialization. The year list remains available without JS. The transparent canvas
  fills the opening viewport below
  the navbar; header and footer controls overlay it. Their measured bounds keep nodes and
  year labels clear. The text-first preview shows a compact thumbnail, the full title,
  and the frontmatter description without line clamping; topic metadata follows the prose.
  Desktop uses a full-height reading column beside the network. Mobile temporarily hides
  browsing controls to give the preview more room. Read more and Back to network follow the
  text and remain accessible when longer content scrolls. Closing restores browsing controls
  and keyboard focus to the selected node.
- `js/video-carousel.js` — Talks single-card infinite carousel + lightbox.
- `js/cookie-consent.js` — cookie consent popup.

## 9. Accessibility

- Animations (home causal field, About ambient field, carousel) are disabled under
  `@media (prefers-reduced-motion: reduce)`. Career DAG geometry is stationary.
- Career buttons support Enter/Space to open details. Arrow keys / Home / End move focus
  without opening a role. The native modal makes the background inert; closing returns
  focus to the original node. Only its content scrolls, keeping Close and navigation visible.
- Carousel cards are buttons; the lightbox is `role="dialog" aria-modal` with Esc-to-close.
- Network marks (articles and keywords) are focusable `role="button"` groups with a full
  accessible name (title, month, topics; keyword labels carry the count and the open
  state via `aria-expanded`). Enter/Space opens the summary sheet on an article and
  opens or folds a keyword's articles on a keyword; arrow keys move focus to the nearest
  visible mark in that direction; Escape closes the sheet first, then folds the open
  keyword. Panning to a focused mark happens on keyboard focus only (`:focus-visible`),
  so the view never jumps under a mouse click. The status line is `aria-live="polite"`;
  the sheet is a non-modal `role="dialog"` whose heading takes focus on open. Node
  captions are drawn in SVG `<text>` — they are part of the node's accessible name, not
  separate labels.
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
- **Quarto's `page-columns` grid wins over your `display`.** Every top-level div gets
  `page-columns page-full`, and Quarto ships `body .page-columns { display: grid }` — a
  plain `.my-component { display: block }` loses to it (specificity 0-1-1), and absolutely
  positioned children then resolve against a *grid area*, not the element. Override with a
  matching-or-higher selector such as `body .network-stage.page-columns { display: block; overflow: hidden; }`.
  Quarto can also override stage overflow at tablet widths, exposing a translated-offscreen sheet.
- **Render the whole project before committing.** Use plain `quarto render` so `docs/`
  includes every page, listing, stylesheet, script, and post-render mirror.
- **`MIMO_API_KEY` must be in the environment** or the render aborts during profile setup
  (`MissingEnvVarsError`, from `.env.example`). `set -a && . ./.env && set +a` before
  rendering; a fresh worktree has no `.env` (it is gitignored).
- You don't need to activate any env to render: Quarto starts each article's kernel
  from its `jupyter: <slug>` frontmatter and runs the project engine from
  `execute.conda: cetagostini_site`. Missing kernels → `bash scripts/setup_envs.sh`.
  The article `articles/alchemize_pytensor_mlx_gemma_3n` sets `eval: false, freeze: false`:
  it starts a Jupyter kernel during a full render, but does not execute the MLX code.
