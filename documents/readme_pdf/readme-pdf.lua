-- Pandoc Lua filter for turning the GitHub README into a PDF.
--   * HTML <img> tags      -> centered figures (LaTeX drops raw HTML)
--   * tables               -> centered tabular in a float, so they are never split across pages
--   * relative file links  -> GitHub URLs (a PDF cannot open repo files)
--   * ```mermaid blocks    -> centered figure rendered by mermaid-cli (mmdc) if it is installed
--
-- Figures and tables use the [H] placement of the `float` package: they stay exactly where they
-- are in the text, and move to the next page as a whole when they do not fit. Change PLACEMENT
-- to "htbp" to let LaTeX move them around like ordinary floats.
--
-- Usage (from the repository root): bash documents/readme_pdf/make_readme_pdf.sh

local REPO_URL = "https://github.com/m-chaves/slr_climate_nlp/blob/main/"
local PLACEMENT = "H"

-- Helpers -----------------------------------------------------------------------------------

local function img_from_html(html)
  local src = html:match('<img[^>]-src="([^"]+)"')
  if not src then return nil end
  local alt = html:match('alt="([^"]*)"') or ""
  local width = html:match('width="(%d+)"')
  local attr = {}
  if width then
    -- GitHub widths are in pixels; map them to a share of the text width.
    attr.width = string.format("%d%%", math.min(100, math.floor(tonumber(width) / 7)))
  end
  return pandoc.Image(alt, src, "", pandoc.Attr("", {}, attr))
end

local function centered_figure(images)
  local blocks = { pandoc.RawBlock("latex", "\\begin{figure}[" .. PLACEMENT .. "]\n\\centering") }
  for _, img in ipairs(images) do
    table.insert(blocks, pandoc.Plain({ img }))
  end
  table.insert(blocks, pandoc.RawBlock("latex", "\\end{figure}"))
  return blocks
end

local function plain_find(s, needle, init)
  return s:find(needle, init or 1, true)
end

-- Turn pandoc's longtable (which breaks across pages) into a tabular that cannot break.
local function longtable_to_tabular(tex)
  local body_start = plain_find(tex, "\\begin{longtable}")
  if not body_start then return nil end

  -- Drop the repeated header used on continuation pages (only present with \endfirsthead).
  local first = plain_find(tex, "\\endfirsthead")
  if first then
    local head_end = select(2, plain_find(tex, "\\endhead", first))
    tex = tex:sub(1, first - 1) .. tex:sub(head_end + 1)
  else
    tex = tex:gsub("\\endhead\n?", "", 1)
  end

  -- The footer (\bottomrule) is declared before the rows in a longtable; move it to the end.
  local footer = ""
  local lastfoot = plain_find(tex, "\\endlastfoot")
  if lastfoot then
    local s = plain_find(tex, "\\bottomrule")
    if s and s < lastfoot then
      footer = tex:sub(s, lastfoot - 1)
      tex = tex:sub(1, s - 1) .. tex:sub(lastfoot + #"\\endlastfoot")
    else
      tex = tex:gsub("\\endlastfoot\n?", "", 1)
    end
  end
  tex = tex:gsub("\\endfoot\n?", "")

  tex = tex:gsub("\\begin{longtable}%[[^%]]*%]", "\\begin{tabular}", 1)
  tex = tex:gsub("\\end{longtable}", function() return footer .. "\\end{tabular}" end, 1)
  return tex
end

-- Inline and block elements ------------------------------------------------------------------

function RawInline(el)
  if el.format == "html" then
    local img = img_from_html(el.text)
    if img then return img end
  end
end

function RawBlock(el)
  if el.format == "html" then
    local img = img_from_html(el.text)
    if img then return centered_figure({ img }) end
  end
end

-- A paragraph that holds nothing but images becomes a centered figure.
local function image_only(el)
  local images = {}
  for _, inl in ipairs(el.content) do
    if inl.t == "Image" then
      table.insert(images, inl)
    elseif inl.t ~= "Space" and inl.t ~= "SoftBreak" and inl.t ~= "LineBreak" then
      return nil
    end
  end
  if #images > 0 then return centered_figure(images) end
end
Para = image_only
Plain = image_only

function Table(el)
  local tex = longtable_to_tabular(pandoc.write(pandoc.Pandoc({ el }), "latex"))
  if not tex then return nil end
  return pandoc.RawBlock("latex",
    "\\begin{table}[" .. PLACEMENT .. "]\n\\centering\n" .. tex .. "\n\\end{table}")
end

function Link(el)
  local t = el.target
  if not t:match("^%a[%w+.-]*:") and not t:match("^#") then
    el.target = REPO_URL .. t:gsub("^%./", "")
  end
  return el
end

local function have_mmdc()
  return os.execute("command -v mmdc >/dev/null 2>&1")
end

local counter = 0
function CodeBlock(el)
  if not el.classes:includes("mermaid") then return nil end
  if not have_mmdc() then
    io.stderr:write("[readme-pdf] mmdc not found: Mermaid block left as code\n")
    return nil
  end
  counter = counter + 1
  local name = string.format("mermaid_%d.png", counter)
  local png = pandoc.system.with_temporary_directory("readme-mermaid", function(dir)
    local src = dir .. "/diagram.mmd"
    local out = dir .. "/diagram.png"
    local f = io.open(src, "w")
    f:write(el.text)
    f:close()
    if not os.execute(string.format("mmdc -i %q -o %q -s 3 -b white >/dev/null 2>&1", src, out)) then
      return nil
    end
    local g = io.open(out, "rb")
    if not g then return nil end
    local data = g:read("a")
    g:close()
    return data
  end)
  if not png then
    io.stderr:write("[readme-pdf] mmdc failed: Mermaid block left as code\n")
    return nil
  end
  -- Images in the media bag are handed to LaTeX by pandoc; nothing is written to the repo.
  pandoc.mediabag.insert(name, "image/png", png)
  return centered_figure({ pandoc.Image("", name, "", pandoc.Attr("", {}, { width = "90%" })) })
end

-- Document metadata: load the packages the raw LaTeX above relies on --------------------------

function Meta(meta)
  local includes = meta["header-includes"]
  if includes == nil then
    includes = pandoc.MetaList({})
  elseif includes.t ~= "MetaList" and pandoc.utils.type(includes) ~= "List" then
    includes = pandoc.MetaList({ includes })
  end
  -- booktabs, array and calc are loaded by pandoc's template only when the document has
  -- tables, which no longer holds once they are raw LaTeX.
  table.insert(includes, pandoc.MetaBlocks({ pandoc.RawBlock("latex",
    "\\usepackage{float}\n\\usepackage{booktabs}\n\\usepackage{array}\n\\usepackage{calc}") }))
  meta["header-includes"] = includes
  return meta
end
