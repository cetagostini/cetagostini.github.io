-- filters/llm-seo.lua
-- Injects schema.org JSON-LD structured data and <link rel="canonical">
-- for SEO / AI-search visibility.
--   index.qmd          -> WebSite
--   about.qmd          -> Person + ProfilePage
--   articles.qmd       -> CollectionPage
--   articles/**/*.qmd  -> Article + BreadcrumbList
--   diary.qmd          -> CollectionPage
--   diary/*.qmd        -> Article + BreadcrumbList
--   talks.qmd          -> VideoObject per embedded video
-- Google ignores llms.txt but uses structured data, so this is the
-- high-leverage move for AI Overviews / rich results.

local stringify = pandoc.utils.stringify
local mtype = pandoc.utils.type
local SITE = "https://cetagostini.github.io/"

local MONTHS = {
  january=1, february=2, march=3, april=4, may=5, june=6,
  july=7, august=8, september=9, october=10, november=11, december=12
}

local function to_iso_date(s)
  if s == nil then return nil end
  local y, m, d = s:match("^(%d%d%d%d)%-(%d%d)%-(%d%d)$")
  if y then return s end
  local mon, day, year = s:match("^(%a+)%s+(%d+),%s+(%d+)$")
  if mon and MONTHS[mon:lower()] then
    return string.format("%04d-%02d-%02d", tonumber(year), MONTHS[mon:lower()], tonumber(day))
  end
  return s
end

local function meta_str(meta, key)
  local v = meta[key]
  if v == nil then return nil end
  local t = mtype(v)
  if t == "MetaInlines" or t == "MetaString" then return stringify(v) end
  if t == "List" then
    local parts = {}
    for _, x in ipairs(v) do table.insert(parts, stringify(x)) end
    return table.concat(parts, ", ")
  end
  return stringify(v)
end

local function authors_list(meta)
  local a = meta.author
  if a == nil then return { { name = "Carlos Trujillo" } } end
  local out = {}
  if mtype(a) == "List" then
    for _, x in ipairs(a) do
      local nm = stringify(x)
      if nm and nm ~= "" then table.insert(out, { name = nm }) end
    end
  else
    local nm = stringify(a)
    if nm and nm ~= "" then table.insert(out, { name = nm }) end
  end
  if #out == 0 then return { { name = "Carlos Trujillo" } } end
  return out
end

local function first_category(meta)
  local c = meta.categories
  if c == nil then return nil end
  if mtype(c) == "List" and c[1] then return stringify(c[1]) end
  return stringify(c)
end

local function build_talk_videos(doc)
  local embeds, captions, titles = {}, {}, {}
  local function walk(blocks)
    for _, b in ipairs(blocks) do
      if b.t == "RawBlock" and b.format == "html" then
        for e in b.text:gmatch('data%-embed="([^"]+)"') do table.insert(embeds, (e:gsub("&amp;", "&"))) end
        for c in b.text:gmatch('data%-caption="([^"]+)"') do table.insert(captions, (c:gsub("&amp;", "&"))) end
        for t in b.text:gmatch('video%-card%-title">([^<]*)</span>') do table.insert(titles, t) end
      elseif b.t == "Div" then
        walk(b.content or b.blocks or {})
      end
    end
  end
  walk(doc.blocks)
  local videos = {}
  for i, e in ipairs(embeds) do
    local vo = { ["@type"] = "VideoObject", embedUrl = e }
    if titles[i] then vo.name = titles[i] end
    if captions[i] then vo.description = captions[i] end
    table.insert(videos, vo)
  end
  return videos
end

local function canonical_url(base)
  if base == "index" then
    return SITE
  elseif base == "about" or base == "articles" or base == "talks" or base == "diary" then
    return SITE .. base .. ".html"
  else
    -- article or diary entry
    return nil  -- caller provides the full path
  end
end

function Pandoc(doc)
  local meta = doc.meta
  local out = PANDOC_STATE.output_file or ""
  local base = out:gsub("^docs/", ""):gsub("%.html$", "")
  local graph = {}

  local title = meta_str(meta, "title") or meta_str(meta, "pagetitle")
  local desc = meta_str(meta, "description")
  local date_iso = to_iso_date(meta_str(meta, "date"))
  local date_mod = to_iso_date(meta_str(meta, "last-modified"))
  local image = meta_str(meta, "image")

  -- ── Canonical URL ──────────────────────────────────────────────────
  local canon = canonical_url(base)
  if not canon then
    if meta_str(meta, "schema-section") == "diary" then
      canon = SITE .. "diary/" .. base .. ".html"
    else
      canon = SITE .. "articles/" .. base .. "/" .. base .. ".html"
    end
  end
  table.insert(doc.blocks, 1, pandoc.RawBlock("html",
    '<link rel="canonical" href="' .. canon .. '" />'))

  -- ── JSON-LD per page type ──────────────────────────────────────────

  if base == "index" then
    table.insert(graph, {
      ["@type"] = "WebSite",
      name = title or "Marketing Science Blog",
      url = SITE,
      description = desc,
      author = { { ["@type"] = "Person", name = "Carlos Trujillo" } },
      publisher = { ["@type"] = "Person", name = "Carlos Trujillo" }
    })

  elseif base == "about" then
    local person = {
      ["@type"] = "Person",
      name = "Carlos Trujillo",
      jobTitle = "Principal Data Scientist",
      url = SITE,
      image = SITE .. "images/profile.jpg",
      sameAs = {
        "https://github.com/cetagostini",
        "https://twitter.com/cetagostini",
        "https://linkedin.com/in/cetagostini",
        "https://instagram.com/cetagostini"
      },
      worksFor = { { name = "PyMC Labs" } },
      alumniOf = {
        { ["@type"] = "EducationalOrganization", name = "Universidad José Antonio Páez" },
        { ["@type"] = "EducationalOrganization", name = "Acámica" }
      },
      email = "carlos.trujillo.agostini@gmail.com"
    }
    if desc then person.description = desc end
    table.insert(graph, person)
    table.insert(graph, {
      ["@type"] = "ProfilePage",
      url = SITE .. "about.html",
      mainEntity = { id = "#person" }
    })

  elseif base == "articles" then
    table.insert(graph, {
      ["@type"] = "CollectionPage",
      name = "Articles",
      url = SITE .. "articles.html",
      description = desc,
      publisher = { ["@type"] = "Person", name = "Carlos Trujillo" }
    })

  elseif meta_str(meta, "schema-section") == "diary" then
    local url = SITE .. "diary/" .. base .. ".html"
    local article = { ["@type"] = "Article", headline = title, url = url }
    if date_iso then article.datePublished = date_iso end
    if date_mod then article.dateModified = date_mod end
    article.author = authors_list(meta)
    if desc then article.description = desc end
    local cat = first_category(meta)
    if cat then article.articleSection = cat end
    article.publisher = { ["@type"] = "Person", name = "Carlos Trujillo" }
    article.mainEntityOfPage = url
    table.insert(graph, article)
    table.insert(graph, {
      ["@type"] = "BreadcrumbList",
      itemListElement = {
        { ["@type"] = "ListItem", position = 1, name = "Home", item = SITE },
        { ["@type"] = "ListItem", position = 2, name = "Diary", item = SITE .. "diary.html" },
        { ["@type"] = "ListItem", position = 3, name = title, item = url }
      }
    })

  elseif base == "diary" then
    table.insert(graph, {
      ["@type"] = "CollectionPage",
      name = "Diary",
      url = SITE .. "diary.html",
      description = desc
    })

  elseif base:match("^articles/") then
    -- Path-based article detection: articles/<slug>/<slug>
    local slug = base:gsub("^articles/", "")
    local url = SITE .. "articles/" .. slug .. "/" .. slug .. ".html"
    local article = { ["@type"] = "Article", headline = title, url = url }
    if date_iso then article.datePublished = date_iso end
    if date_mod then article.dateModified = date_mod end
    article.author = authors_list(meta)
    if desc then article.description = desc end
    local image_path = image and (image:gsub("^%.%./", "")):gsub("^/", "")
    if image_path then article.image = SITE .. image_path end
    local cat = first_category(meta)
    if cat then article.articleSection = cat end
    article.publisher = { ["@type"] = "Person", name = "Carlos Trujillo" }
    article.mainEntityOfPage = url
    table.insert(graph, article)
    table.insert(graph, {
      ["@type"] = "BreadcrumbList",
      itemListElement = {
        { ["@type"] = "ListItem", position = 1, name = "Home", item = SITE },
        { ["@type"] = "ListItem", position = 2, name = "Articles", item = SITE .. "articles.html" },
        { ["@type"] = "ListItem", position = 3, name = title, item = url }
      }
    })

  elseif base == "talks" then
    for _, vo in ipairs(build_talk_videos(doc)) do
      table.insert(graph, vo)
    end
  end

  if #graph == 0 then return doc end

  local payload = { ["@context"] = "https://schema.org", ["@graph"] = graph }
  local ok, json_str = pcall(pandoc.json.encode, payload)
  if not ok then return doc end
  local script = '<script type="application/ld+json">\n' .. json_str .. '\n</script>'
  table.insert(doc.blocks, pandoc.RawBlock("html", script))
  return doc
end
