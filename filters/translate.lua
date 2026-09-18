-- filters/translate.lua
-- Single filter, two modes, one block-identity function.
--
--   QUARTO_PROFILE contains token "es-dump" -> DUMP: write the runtime AST records
--   QUARTO_PROFILE contains token "es"      -> TRANSLATE: replace English nodes
--   anything else                           -> strict no-op (no I/O, output unchanged)
--
-- Registration: BASE _quarto.yml, BEFORE filters/llm-seo.lua. Profile-declared
-- filter lists APPEND, so declaring this in _quarto-es.yml would run it too late.
--
-- Verified mechanics relied on here (Quarto 1.7.31 / pandoc 3.6.3):
--   * PANDOC_STATE.input_files[1] is a temp .md; identity comes from
--     QUARTO_DOCUMENT_FILE / QUARTO_DOCUMENT_PATH / QUARTO_PROJECT_ROOT.
--   * cwd is the document's directory for nested docs -> all file I/O is
--     QUARTO_PROJECT_ROOT-absolute.
--   * pandoc.write(block, "markdown") THROWS; the Pandoc({}) wrapper is mandatory.
--   * Runtime AST: callouts are Div{__quarto_custom_type=Callout} with scaffold
--     children (title = first NON-EMPTY scaffold Plain); figures are
--     Div{__quarto_custom_type=FloatRefTarget} scaffolds; list items are Plain.
--   * og/twitter + navbar/sidebar labels come from hidden envelope Divs and are
--     only reachable by rewriting their spans (render-id keyed).

local function split_tokens(s)
  local out = {}
  for piece in (s or ""):gmatch("[^,]+") do
    local t = piece:gsub("^%s+", ""):gsub("%s+$", "")
    if t ~= "" then out[#out + 1] = t end
  end
  return out
end

local function has_token(s, want)
  for _, t in ipairs(split_tokens(s)) do
    if t == want then return true end
  end
  return false
end

local PROFILE = os.getenv("QUARTO_PROFILE") or ""
local MODE
if has_token(PROFILE, "es-dump") then
  MODE = "dump"
elseif has_token(PROFILE, "es") then
  MODE = "translate"
else
  return {} -- strict no-op on EN and on any unknown profile
end

--------------------------------------------------------------------------------
-- page identity
--------------------------------------------------------------------------------

-- Page identity.
--
-- QUARTO_DOCUMENT_FILE/QUARTO_DOCUMENT_PATH are only refreshed when Quarto
-- EXECUTES a document; for freeze-replayed documents they keep the previously
-- executed document's values (verified: all 7 frozen articles reported the
-- alchemize article). Identity therefore comes from the document's working
-- directory (which IS correct for replayed documents) plus the output basename.
local ROOT = os.getenv("QUARTO_PROJECT_ROOT")

local function pwd()
  local ph = io.popen("pwd")
  if not ph then return nil end
  local d = ph:read("*l")
  ph:close()
  return d
end

local CWD = pwd()
local OUT = PANDOC_STATE and PANDOC_STATE.output_file or ""
local stem_out = OUT:match("([^/]+)%.html$") or OUT:match("([^/]+)$") or ""

-- Missing identity is fatal for DUMP (a silently mis-keyed dump would publish an
-- all-English /es/ tree) but merely disables translation for TRANSLATE.
if not (ROOT and CWD and stem_out ~= "") then
  if MODE == "dump" then
    error("translate.lua: cannot determine document identity (ROOT/CWD/output_file)")
  end
  io.stderr:write("[translate] indeterminate document identity; page stays English\n")
  return {}
end

local reldir = (#CWD > #ROOT) and CWD:sub(#ROOT + 2) or ""
if reldir == "/" then reldir = "" end
local REL = (reldir == "") and (stem_out .. ".qmd") or (reldir .. "/" .. stem_out .. ".qmd")

-- articles/<slug>/<slug>.qmd -> articles/<slug>.json ; diary/<date>.qmd -> diary/<date>.json
-- index|about|articles|talks|diary .qmd -> pages/<name>.json
local function record_rel()
  local dir, name = REL:match("^(.*)/([^/]+)$")
  local stem = (name or REL):gsub("%.qmd$", "")
  if not dir then return "pages/" .. stem .. ".json" end
  if dir == "articles" or dir:match("^articles/") then return "articles/" .. stem .. ".json" end
  if dir == "diary" or dir:match("^diary/") then return "diary/" .. stem .. ".json" end
  return "pages/" .. stem .. ".json"
end

local RECORD = record_rel()

local function readfile(path)
  local fh = io.open(path, "r")
  if not fh then return nil end
  local s = fh:read("*a")
  fh:close()
  return s
end

local function ensure_dir(path)
  -- io.open cannot create directories; shell out once per run.
  local dir = path:match("^(.*)/[^/]+$")
  if dir then os.execute("mkdir -p '" .. dir:gsub("'", "'\\''") .. "'") end
end

local function writefile(path, data)
  ensure_dir(path)
  local fh = io.open(path, "w")
  if not fh then
    -- DUMP write failures are fatal: a silently empty dump would publish an
    -- all-English /es/ tree.
    error("translate.lua: cannot write " .. path)
  end
  fh:write(data)
  fh:close()
end

--------------------------------------------------------------------------------
-- compiled dictionary
--------------------------------------------------------------------------------

local DICT = { blocks = {}, raw = {}, envelope = {}, meta = {} }

if MODE == "translate" then
  local raw = readfile(ROOT .. "/i18n/es/compiled/" .. RECORD)
  if raw then
    local ok, decoded = pcall(pandoc.json.decode, raw)
    if ok and type(decoded) == "table" then
      -- NOTE: pandoc.json.decode maps JSON null to a userdata sentinel, so every
      -- value read out of the dictionary must be type-checked before use.
      local function s(v) return type(v) == "string" and v or nil end
      for _, e in ipairs(decoded.blocks or {}) do
        local match, es = s(e.match), s(e.es)
        if match and es and es:match("%S") then
          DICT.blocks[(type(e.kind) == "string" and e.kind or "para") .. "\0" .. match] = es
        end
      end
      for _, e in ipairs(decoded.raw_blocks or {}) do
        local match, html = s(e.match), s(e.es_html)
        if match and html and html:match("%S") then
          DICT.raw[match] = html
        end
      end
      for id, v in pairs(decoded.envelope or {}) do
        local es = s(v)
        if es and es:match("%S") then DICT.envelope[id] = es end
      end
      for k, v in pairs(decoded.meta or {}) do
        local es = s(v)
        if es then DICT.meta[k] = es end
        if type(v) == "table" then
          local list = {}
          for _, item in ipairs(v) do
            local si = s(item)
            if si then list[#list + 1] = si end
          end
          if #list > 0 then DICT.meta[k] = list end
        end
      end
    else
      io.stderr:write("[translate] cannot decode " .. RECORD .. "; page stays English\n")
    end
  else
    io.stderr:write("[translate] no compiled dictionary for " .. RECORD .. "; page stays English\n")
  end
end

--------------------------------------------------------------------------------
-- normalization (shared by DUMP and TRANSLATE -> keys cannot diverge)
--------------------------------------------------------------------------------

local function normalize(node)
  local ok, s = pcall(pandoc.write, pandoc.Pandoc({ node }), "markdown")
  if not ok then return nil end
  s = s:gsub("\r\n", "\n")
  s = s:gsub("[ \t]+", " ")
  s = s:gsub("\n+$", "")
  return s
end

local PROTECTED = {
  Math = true, Code = true, Cite = true, RawInline = true, Image = true, Note = true,
}

-- Does this inline list carry human language (i.e. is it a translation unit)?
local function has_text(inlines)
  local found = false
  local function walk(xs)
    for _, x in ipairs(xs) do
      if x.t == "Str" then
        if x.text:match("%a") then found = true end
      elseif x.t == "SoftBreak" or x.t == "Space" or x.t == "LineBreak" then
        -- neutral
      elseif PROTECTED[x.t] then
        -- contribute nothing
      elseif x.content then
        walk(x.content)
      end
    end
  end
  walk(inlines)
  return found
end

local function blocks_have_text(blocks)
  for _, b in ipairs(blocks) do
    if (b.t == "Para" or b.t == "Plain") and has_text(b.content) then return true end
    if b.content then
      local nested = type(b.content) == "table" and b.content or nil
      if nested and nested[1] and type(nested[1]) == "table" and nested[1].t then
        if blocks_have_text(nested) then return true end
      end
    end
  end
  return false
end

--------------------------------------------------------------------------------
-- DUMP bookkeeping
--------------------------------------------------------------------------------

local DUMP = { meta = {}, blocks = {}, raw = {}, envelope = {} }
local STATS = { matched = 0, total = 0, unmatched = {}, orphans = 0 }

local function dump_emit(kind, node, context)
  local match = normalize(node)
  if not match or match == "" then return end
  local key = kind .. "\0" .. match
  local entry = DUMP.blocks[key]
  if entry then
    entry.count = (entry.count or 1) + 1
    return
  end
  DUMP.blocks[key] = {
    kind = kind,
    context = context or "body",
    en = match,
    count = 1,
  }
end

-- Raw HTML is a translation unit ONLY when it is authored prose. Quarto injects
-- generated markup into the document (listing card fragments, xarray/pandas
-- reprs, Jupyter widget scripts, inline icon sprites) which must never be
-- translated. Note the editorial pages legitimately contain inline <svg> and a
-- trailing <script src>, so those tags alone do NOT disqualify a block.
local GENERATED_MARKERS = {
  "class='xr-", 'class="xr-', "xarray.Dataset", "xarray.DataArray",
  "dataframe", "jupyter.widget", "vnd.jupyter", 'symbol id="icon-',
  "quarto-listing", "quarto-listing-pipeline", "cell-output",
  "listing-category", "listing-title", "listing-description", "listing-date",
  "nutpie", "Sampler Progress",
}

local function raw_is_translatable(text)
  if #text > 40000 then return false end
  local head = text:gsub("^%s+", ""):sub(1, 40):lower()
  -- Standalone generated artifacts: datatable/plot CSS, altair/plotly charts,
  -- sampled-progress widgets. Authored editorial blocks are HTML containers that
  -- may *contain* an inline svg, so only the block's own opening tag counts.
  if head:match("^<style") or head:match("^<svg") or head:match("^<table") then return false end
  for _, m in ipairs(GENERATED_MARKERS) do
    if text:find(m, 1, true) then return false end
  end
  local plain = text:gsub("<[^>]->", " "):gsub("&[%a#0-9]-;", " ")
  local words = 0
  for _ in plain:gmatch("%a[%a'%-]+") do words = words + 1 end
  return words >= 4
end

local function dump_emit_raw(text)
  if not raw_is_translatable(text) then return end
  if DUMP.raw[text] then return end
  DUMP.raw[text] = { en_text = text }
end

local function dump_emit_envelope(render_id, text)
  if not render_id or not text or text == "" then return end
  -- href-like ids are filtered on the Python side, which can base64-decode.
  if DUMP.envelope[render_id] then return end
  DUMP.envelope[render_id] = { en = text }
end

--------------------------------------------------------------------------------
-- envelope + span walking
--------------------------------------------------------------------------------

local ENVELOPE_IDS = {
  ["quarto-meta-markdown"] = true,
  ["quarto-navigation-envelope"] = true,
}

local function walk_envelope_div(div)
  for _, para in ipairs(div.content) do
    if para.content then
      for _, span in ipairs(para.content) do
        if span.t == "Span" then
          local rid = span.attr and span.attr.attributes and span.attr.attributes["render-id"]
          if rid then
            if MODE == "dump" then
              dump_emit_envelope(rid, pandoc.utils.stringify(span))
            else
              local es = DICT.envelope[rid]
              if es then span.content = pandoc.Inlines({ pandoc.Str(es) }) end
            end
          end
        end
      end
    end
  end
end

--------------------------------------------------------------------------------
-- traversal
--------------------------------------------------------------------------------

local function lookup(kind, node)
  local match = normalize(node)
  if not match then return nil end
  local es = DICT.blocks[kind .. "\0" .. match]
  if es == nil then
    if MODE == "translate" then
      STATS.total = STATS.total + 1
      if #STATS.unmatched < 50 then STATS.unmatched[#STATS.unmatched + 1] = kind .. ":" .. match:sub(1, 80) end
    end
    return nil
  end
  if MODE == "translate" then STATS.matched = STATS.matched + 1 end
  return es
end

-- Replace a Para/Plain body with the Spanish markdown, keeping the node kind.
local function replace_flow(node, es, kind)
  local ok, parsed = pcall(pandoc.read, es, "markdown")
  if not ok or not parsed or #parsed.blocks == 0 then return nil end
  local first = parsed.blocks[1]
  if kind == "list-item" or kind == "plain" then
    -- list items and Plain are inline containers
    if first.t == "Plain" then return first.content end
    if first.t == "Para" then return first.content end
    return nil
  end
  if first.t == "Para" or first.t == "Plain" then return first.content end
  return nil
end

local function walk_blocks(blocks, context, out)
  out = out or pandoc.List()
  for _, b in ipairs(blocks) do
    if b.t == "Div" and b.attr and b.attr.identifier and ENVELOPE_IDS[b.attr.identifier] then
      walk_envelope_div(b)
      out:insert(b)

    elseif b.t == "Div" and b.attr and b.attr.attributes then
      local ctype = b.attr.attributes["__quarto_custom_type"]
      if ctype == "Callout" then
        -- title = first NON-EMPTY scaffold Plain; remaining non-empty scaffold = body
        local scaffolds = {}
        for _, ch in ipairs(b.content) do
          if ch.t == "Div" and ch.attr.attributes["__quarto_custom_scaffold"] and #ch.content > 0 then
            scaffolds[#scaffolds + 1] = ch
          end
        end
        for i, sc in ipairs(scaffolds) do
          local is_title = (i == 1 and #scaffolds > 1)
          if is_title then
            local first = sc.content[1]
            if first and (first.t == "Plain" or first.t == "Para") and has_text(first.content) then
              if MODE == "dump" then
                dump_emit("callout-title", first, "callout")
              else
                local es = lookup("callout-title", first)
                local inl = es and replace_flow(first, es, "plain")
                if inl then first.content = inl end
              end
            end
            sc.content = walk_blocks(sc.content, { callout = true }, pandoc.List())
          else
            sc.content = walk_blocks(sc.content, { callout_body = true }, pandoc.List())
          end
        end
        out:insert(b)

      elseif ctype == "FloatRefTarget" then
        local scaffolds = {}
        for _, ch in ipairs(b.content) do
          if ch.t == "Div" and ch.attr.attributes["__quarto_custom_scaffold"] then
            scaffolds[#scaffolds + 1] = ch
          end
        end
        -- scaffold[1] holds the image (Plain may be empty); scaffold[2] the caption
        for idx, sc in ipairs(scaffolds) do
          local first = sc.content[1]
          if not first then goto continue end
          if idx == 1 then
            -- image + its alt text
            local imgs = {}
            local function collect(inlines)
              for _, x in ipairs(inlines) do
                if x.t == "Image" then imgs[#imgs + 1] = x end
                if x.content then collect(x.content) end
              end
            end
            for _, blk in ipairs(sc.content) do
              if blk.content then collect(blk.content) end
            end
            for _, img in ipairs(imgs) do
              local alt = img.attr and img.attr.attributes and img.attr.attributes["fig-alt"]
              local altnode
              if alt and alt:match("%S") then
                altnode = pandoc.Plain({ pandoc.Str(alt) })
              elseif img.caption and #img.caption > 0 and has_text(img.caption) then
                altnode = pandoc.Plain(img.caption)
              end
              if altnode then
                if MODE == "dump" then
                  dump_emit("image-alt", altnode, "figure")
                else
                  local es = lookup("image-alt", altnode)
                  if es then
                    local txt = es:gsub("^%s+", ""):gsub("%s+$", "")
                    if img.attr and img.attr.attributes["fig-alt"] then
                      img.attr.attributes["fig-alt"] = txt
                    else
                      img.caption = pandoc.Inlines({ pandoc.Str(txt) })
                    end
                  end
                end
              end
            end
            sc.content = walk_blocks(sc.content, context, pandoc.List())
          else
            if (first.t == "Plain" or first.t == "Para") and has_text(first.content) then
              if MODE == "dump" then
                dump_emit("figure-caption", first, "figure")
              else
                local es = lookup("figure-caption", first)
                local inl = es and replace_flow(first, es, "plain")
                if inl then first.content = inl end
              end
            end
            sc.content = walk_blocks(sc.content, context, pandoc.List())
          end
          ::continue::
        end
        out:insert(b)

      else
        b.content = walk_blocks(b.content, context, pandoc.List())
        out:insert(b)
      end

    elseif b.t == "Header" then
      if has_text(b.content) then
        if MODE == "dump" then
          dump_emit("header", pandoc.Plain(b.content), context.kind)
        else
          local es = lookup("header", pandoc.Plain(b.content))
          local inl = es and replace_flow(pandoc.Plain(b.content), es, "plain")
          if inl then b.content = inl end
        end
      end
      out:insert(b)

    elseif b.t == "Para" or b.t == "Plain" then
      local kind = context.in_list and "list-item" or "para"
      if kind == "para" and b.t == "Plain" then kind = "plain" end
      -- Bare markdown images (`![alt](src)`) are Image inlines inside a flow
      -- block, not FloatRefTarget scaffolds: their alt text is translatable too.
      local imgs = {}
      local function collect(xs)
        for _, x in ipairs(xs) do
          if x.t == "Image" then imgs[#imgs + 1] = x end
          if x.content then collect(x.content) end
        end
      end
      collect(b.content)
      for _, img in ipairs(imgs) do
        local alt = img.attr and img.attr.attributes and img.attr.attributes["fig-alt"]
        local node
        if alt and alt:match("%S") then
          node = pandoc.Plain({ pandoc.Str(alt) })
        elseif img.caption and #img.caption > 0 and has_text(img.caption) then
          node = pandoc.Plain(img.caption)
        end
        if node then
          if MODE == "dump" then
            dump_emit("image-alt", node, context.kind)
          else
            local es = lookup("image-alt", node)
            if es then
              local txt = es:gsub("^%s+", ""):gsub("%s+$", "")
              if img.attr and img.attr.attributes["fig-alt"] then
                img.attr.attributes["fig-alt"] = txt
              else
                img.caption = pandoc.Inlines({ pandoc.Str(txt) })
              end
            end
          end
        end
      end
      if has_text(b.content) then
        if MODE == "dump" then
          dump_emit(kind, b, context.kind)
        else
          local es = lookup(kind, b)
          local inl = es and replace_flow(b, es, kind)
          if inl then b.content = inl end
        end
      end
      out:insert(b)

    elseif b.t == "BulletList" or b.t == "OrderedList" then
      -- In this pandoc binding BOTH list types expose `content` as the list of
      -- items (each item is itself a list of blocks) — verified by probe.
      for idx, item in ipairs(b.content) do
        if type(item) == "table" and item.t then
          -- defensive: a bare Block where an item block-list was expected
          b.content[idx] = walk_blocks({ item }, { in_list = true, kind = "list" }, pandoc.List())
        else
          b.content[idx] = walk_blocks(item, { in_list = true, kind = "list" }, pandoc.List())
        end
      end
      out:insert(b)

    elseif b.t == "BlockQuote" then
      b.content = walk_blocks(b.content, context, pandoc.List())
      out:insert(b)

    elseif b.t == "RawBlock" and b.format == "html" then
      if MODE == "dump" then
        dump_emit_raw(b.text)
      else
        local es = DICT.raw[b.text]
        if es then b.text = es end
      end
      out:insert(b)

    else
      out:insert(b)
    end
  end
  return out
end

--------------------------------------------------------------------------------
-- meta translation
--------------------------------------------------------------------------------

local META_FIELDS = { "title", "pagetitle", "description", "image-alt" }

local function translate_meta(m)
  if MODE ~= "translate" then return m end
  for _, key in ipairs(META_FIELDS) do
    local spec = DICT.meta[key]
    if spec and spec ~= "" then
      local ok, parsed = pcall(pandoc.read, spec, "markdown")
      if ok and parsed and #parsed.blocks > 0 and parsed.blocks[1].content then
        m[key] = pandoc.MetaInlines(parsed.blocks[1].content)
      end
    end
  end
  if type(DICT.meta.categories) == "table" then
    local cats = pandoc.List()
    for _, c in ipairs(DICT.meta.categories) do cats:insert(pandoc.MetaString(c)) end
    if #cats > 0 then m["categories"] = pandoc.MetaList(cats) end
  end
  return m
end

--------------------------------------------------------------------------------
-- entry point
--------------------------------------------------------------------------------

function Pandoc(doc)
  if MODE == "dump" then
    -- Reset per page: one Lua environment may serve several documents, and
    -- leftover state would leak the previous page's units into this record.
    DUMP = { meta = {}, blocks = {}, raw = {}, envelope = {} }
    DUMP.meta = {}
    for _, key in ipairs(META_FIELDS) do
      local v = doc.meta[key]
      if v then DUMP.meta[key] = pandoc.utils.stringify(v) end
    end
    if doc.meta.categories then
      DUMP.meta.categories = {}
      local c = doc.meta.categories
      if c.t == "MetaList" then
        for _, x in ipairs(c) do DUMP.meta.categories[#DUMP.meta.categories + 1] = pandoc.utils.stringify(x) end
      else
        DUMP.meta.categories[1] = pandoc.utils.stringify(c)
      end
    end
    doc.blocks = walk_blocks(doc.blocks, { kind = "body" }, pandoc.List())
    -- persist
    local blocks = {}
    for k, e in pairs(DUMP.blocks) do
      blocks[#blocks + 1] = {
        kind = e.kind, context = e.context, en = e.en, count = e.count or 1,
      }
    end
    table.sort(blocks, function(a, b)
      if a.kind ~= b.kind then return a.kind < b.kind end
      return a.en < b.en
    end)
    local raws = {}
    for k in pairs(DUMP.raw) do raws[#raws + 1] = k end
    table.sort(raws)
    local rawlist = {}
    for i, k in ipairs(raws) do rawlist[i] = { en_text = k } end
    local envs = {}
    for k in pairs(DUMP.envelope) do envs[#envs + 1] = k end
    table.sort(envs)
    local envlist = {}
    for i, k in ipairs(envs) do envlist[i] = { render_id = k, en = DUMP.envelope[k].en } end
    local payload = {
      schema_version = 4,
      source = REL,
      meta = DUMP.meta,
      blocks = blocks,
      raw_blocks = rawlist,
      envelope = envlist,
    }
    local ok, json = pcall(pandoc.json.encode, payload)
    if not ok then error("translate.lua: cannot encode dump for " .. REL) end
    writefile(ROOT .. "/i18n/es/_extracted/" .. RECORD, json)
    return doc
  end

  -- Reset per page: stats are written per route, so accumulating across pages
  -- in one Lua environment would misreport coverage.
  STATS = { matched = 0, total = 0, unmatched = {}, orphans = 0 }
  doc.meta = translate_meta(doc.meta)
  doc.blocks = walk_blocks(doc.blocks, { kind = "body" }, pandoc.List())

  -- per-route stats for the divergence gate
  local stats = {
    source = REL,
    matched = STATS.matched,
    total = STATS.total,
    unmatched = STATS.unmatched,
  }
  writefile(ROOT .. "/i18n/es/_extracted/" .. RECORD:gsub("%.json$", ".stats.json"),
    pandoc.json.encode(stats))

  return doc
end
