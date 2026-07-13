-- filters/llm-seo.lua
-- Injects schema.org JSON-LD structured data for SEO / AI-search visibility.
--   about.qmd          -> Person + ProfilePage
--   articles/**/*.qmd  -> Article + BreadcrumbList
--   talks.qmd          -> one VideoObject per embedded video
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
  -- Videos now live as <button class="video-card" data-embed=... data-caption=...>
  -- in raw HTML (the carousel). Extract one VideoObject per card.
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

function Pandoc(doc)
  local meta = doc.meta
  local out = PANDOC_STATE.output_file or ""
  local base = out:gsub("^docs/", ""):gsub("%.html$", "")
  local graph = {}

  local title = meta_str(meta, "title") or meta_str(meta, "pagetitle")
  local desc = meta_str(meta, "description")
  local date_iso = to_iso_date(meta_str(meta, "date"))
  local image = meta_str(meta, "image")

  if base == "about" then
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
      worksFor = { { name = "PyMC Labs" } }
    }
    if desc then person.description = desc end
    table.insert(graph, person)
    table.insert(graph, {
      ["@type"] = "ProfilePage",
      url = SITE .. "about.html",
      mainEntity = { id = "#person" }
    })

  elseif meta_str(meta, "schema-section") == "diary" then
    -- Diary entry: URL is diary/<slug>.html
    local url = SITE .. "diary/" .. base .. ".html"
    local article = { ["@type"] = "Article", headline = title, url = url }
    if date_iso then article.datePublished = date_iso end
    article.author = authors_list(meta)
    if desc then article.description = desc end
    local cat = first_category(meta)
    if cat then article.articleSection = cat end
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

  elseif meta.date and meta.image and base ~= "articles" and base ~= "index" and base ~= "talks" and base ~= "about" then
    local url = SITE .. "articles/" .. base .. "/" .. base .. ".html"
    local article = { ["@type"] = "Article", headline = title, url = url }
    if date_iso then article.datePublished = date_iso end
    article.author = authors_list(meta)
    if desc then article.description = desc end
    local image_path = image and (image:gsub("^%.%./", "")):gsub("^/", "")
    if image_path then article.image = SITE .. image_path end
    local cat = first_category(meta)
    if cat then article.articleSection = cat end
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
