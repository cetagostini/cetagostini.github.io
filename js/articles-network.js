// Articles network — the Articles page drawn as a field of connected marks.
//
// Data comes from docs/articles-network.json (generate_articles_network.py).
// "By topic" shows the keywords themselves as one graph: every topic is a
// node, linked to the topics it shares articles with. Activating a keyword
// breaks that graph open — its articles bloom out of the node as photo
// circles, held in orbit around it, while the rest of the keyword graph
// folds away. Activating the keyword again (or Esc, or an empty click) folds
// them back. "By date" lays the articles along a timeline instead.
// Opening an article summary breaks whichever layout is live: the rest float
// and bounce off the walls. Drag pans, wheel or pinch zooms, pulses run
// along the links.
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  // --- tuning -------------------------------------------------------------
  var LABEL_CHARS = 24;     // caption characters per line
  var LINK_KEEP = 3;        // strongest relationships drawn per article
  var R_MIN = 30;           // radius of the oldest article
  var R_MAX = 44;           // radius of the newest article
  var ZOOM_MIN = 0.55;
  var ZOOM_MAX = 2.6;
  var REPULSION = 7.5e6;    // px³/s² between marks: keeps unlinked bodies apart
  var MAX_FORCE = 3200;     // px/s² clamp on each force
  var DAMPING = 2.3;        // 1/s exponential velocity decay
  var SPRING_PAIR = 6.0;    // 1/s² per unit of similarity between two marks
  var SPRING_HUB = 3.1;     // 1/s² pull of an article toward its open keyword
  var SPRING_DATE = 4.2;    // 1/s² pull toward a timeline slot
  var SPRING_FOCUS = 8.0;   // 1/s² pull for the article held in the sheet
  var RING_PUSH = 2.6;      // 1/s² soft orbit radius around an open keyword
  var CENTER_PULL = 0.32;   // 1/s² pull keeping the field inside the stage
  var SPRING_SLOT = 1.1;    // 1/s² pull toward a body's slot in the ring
  var PAIR_SPACING = 0.5;   // share of the stage a similarity link spans at rest
  var PAIR_MIN = 160;       // px floor for that resting length
  var PAIR_MAX = 340;       // px ceiling for that resting length
  var LINK_FLOOR = 0.05;    // similarity below which two articles are not linked
  var TOPIC_LINK_FLOOR = 1; // shared articles a topic pair needs to be linked
  var WANDER = 30;          // px/s² drift once the graph is broken
  var RESTITUTION = 0.94;   // wall bounce
  var MOUSE_PULL = 200;     // px/s² at the pointer, fading over MOUSE_RANGE
  var MOUSE_RANGE = 260;
  var PULSE_EVERY = 1.2;    // seconds between ambient pulses
  var FLASH = 0.7;          // seconds a node stays lit after a pulse lands
  var FADE = 7;             // 1/s rate at which marks bloom in and fold out

  function init() {
    var root = document.querySelector("[data-network]");
    if (!root) return;

    var stage = root.querySelector("[data-network-stage]");
    var header = root.querySelector("[data-network-header]");
    var footer = root.querySelector("[data-network-footer]");
    var navbar = document.querySelector("#quarto-header");
    var svg = root.querySelector(".network-canvas");
    var sheet = root.querySelector("[data-network-sheet]");
    var statusEl = root.querySelector("[data-network-status]");
    var hintEl = root.querySelector("[data-network-hint]");
    var modeButtons = Array.prototype.slice.call(root.querySelectorAll("[data-network-mode]"));
    var zoomInButton = root.querySelector('[data-network-zoom="in"]');
    var zoomOutButton = root.querySelector('[data-network-zoom="out"]');
    if (!stage || !svg || !sheet) return;

    var reduce = matchMedia("(prefers-reduced-motion: reduce)");
    var fine = matchMedia("(hover: hover) and (pointer: fine)");

    var data = null;
    var topicById = {};
    var nodes = [];         // article bodies
    var topics = [];        // keyword bodies
    var links = [];
    var pulses = [];
    var view = { k: 1, tx: 0, ty: 0 };
    var size = { w: 0, h: 0 };
    var field = { top: 0, bottom: 0, h: 0, cy: 0 };
    var pairRest = 200;   // resting length of a similarity link, set from the stage size
    var nodeScale = 1;    // circle scale for narrow stages
    var state = { mode: "topic", topic: null, open: null, hover: null, broken: false, trigger: null };
    var pointer = { x: 0, y: 0, inside: false, down: false, moved: false, suppress: false, panning: false, pinch: null };
    var touches = {};
    var layers = {};
    var raf = 0;
    var last = 0;
    var clock = 0;
    var pulseTimer = 0;
    var sheetLimit = { right: 0, bottom: 0 };

    // --- small helpers ----------------------------------------------------
    function clamp(value, low, high) { return value < low ? low : value > high ? high : value; }
    function element(tag, className, parent) {
      var el = document.createElementNS(NS, tag);
      if (className) el.setAttribute("class", className);
      if (parent) parent.appendChild(el);
      return el;
    }
    function layer(className, decorative) {
      var el = element("g", className, layers.view);
      if (decorative) el.setAttribute("aria-hidden", "true");
      return el;
    }
    function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
    function ease(current, target, dt, rate) {
      return current + (target - current) * (1 - Math.exp(-dt * rate));
    }
    function topicLabel(id) { return topicById[id] ? topicById[id].label : id; }
    function allBodies() { return topics.concat(nodes); }
    // Two lines at most, broken on word boundaries; a title that still does not
    // fit ends with an ellipsis instead of a chopped word.
    function wrapTitle(text, limit) {
      var words = String(text).split(/\s+/).filter(Boolean);
      var lines = [];
      var current = "";
      for (var i = 0; i < words.length; i++) {
        var candidate = current ? current + " " + words[i] : words[i];
        if (current && candidate.length > (limit || LABEL_CHARS)) {
          lines.push(current);
          if (lines.length === 2) return [lines[0], lines[1] + "…"];
          current = words[i];
        } else {
          current = candidate;
        }
      }
      if (current) lines.push(current);
      return lines.length ? lines : ["…"];
    }

    // --- data -------------------------------------------------------------
    fetch(root.getAttribute("data-network-source") || "articles-network.json", { cache: "no-cache" })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(start)
      .catch(function (error) {
        root.setAttribute("data-network-state", "error");
        root.setAttribute("data-network-error", String(error && error.message ? error.message : error));
        statusEl.textContent = "The interactive network could not load — the index below has every article.";
        if (window.console) console.warn("articles network:", error);
      });

    function start(payload) {
      data = payload;
      (payload.topics || []).forEach(function (topic) { topicById[topic.id] = topic; });
      if (!data.articles || !data.articles.length) throw new Error("no articles in payload");

      // Defs: one shared objectBoundingBox clip turns every thumbnail round.
      var defs = element("defs", null, svg);
      var clip = element("clipPath", null, defs);
      clip.setAttribute("id", "an-clip");
      clip.setAttribute("clipPathUnits", "objectBoundingBox");
      var clipCircle = element("circle", null, clip);
      clipCircle.setAttribute("cx", "0.5");
      clipCircle.setAttribute("cy", "0.5");
      clipCircle.setAttribute("r", "0.5");

      layers.view = element("g", "an-view", svg);
      layers.years = layer("an-years", true);
      layers.links = layer("an-links", true);
      layers.pulses = layer("an-pulses", true);
      layers.topics = layer("an-topics");
      layers.nodes = layer("an-nodes");

      resize();               // measure the stage before anything is placed
      buildTopicNodes(payload.topics || []);
      buildNodes();
      buildField();
      bindEvents();
      settle();   // lay the field out before the first frame is due
      updateStatus();
      startMotion();
      root.setAttribute("data-network-ready", "1");
    }

    // --- keyword nodes ----------------------------------------------------
    // Every topic is a mark in the graph: an ellipse wide enough for its
    // label, with the article count hanging below like an article's date.
    function buildTopicNodes(list) {
      list.forEach(function (topic, index) {
        var body = {
          kind: "topic",
          topic: topic,
          index: index,
          hubIds: [topic.id],
          members: [],
          x: 0, y: 0, vx: 0, vy: 0, fx: 0, fy: 0,
          phase: index * 2.3,
          scale: 1, flash: -1,
          vis: 0, active: false, anchor: null,
          near: false
        };

        var group = element("g", "an-topic", layers.topics);
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("aria-expanded", "false");
        body.el = group;
        body.shape = element("ellipse", "an-topic-shape", group);
        body.label = element("text", "an-topic-label", group);
        body.count = element("text", "an-topic-count", group);

        body.fit = function () {
          var lines = wrapTitle(topic.label, 16);
          clear(body.label);
          var longest = 0;
          lines.forEach(function (line, lineIndex) {
            var tspan = element("tspan", null, body.label);
            tspan.setAttribute("x", 0);
            tspan.setAttribute("y", lines.length > 1 ? (lineIndex ? 10 : -4) : 4.5);
            tspan.textContent = line;
            longest = Math.max(longest, line.length);
          });
          body.rx = longest * 3.4 + 15;
          body.ry = lines.length > 1 ? 25 : 18;
          body.shape.setAttribute("rx", body.rx);
          body.shape.setAttribute("ry", body.ry);
          // Heavier topics read as heavier marks.
          body.shape.setAttribute("stroke-width", (1.2 + Math.min(topic.count, 8) * 0.22).toFixed(2));
          body.count.setAttribute("y", body.ry + 15);
          body.count.textContent = topic.count + (topic.count === 1 ? " article" : " articles");
          body.r = body.ry;
          body.cr = Math.max(body.rx, body.ry);
          body.labelHalf = body.rx;
        };
        body.fit();
        body.aria = function () {
          var open = state.topic === topic.id;
          group.setAttribute("aria-expanded", open ? "true" : "false");
          group.setAttribute("aria-label", topic.label + " — " + topic.count +
            (topic.count === 1 ? " article. " : " articles. ") +
            (open ? "Open. Activate to fold its articles back." : "Activate to open its articles."));
        };
        body.aria();

        // Start on a ring so the first frame is a spread field, not a pile.
        var angle = (index / Math.max(1, list.length)) * Math.PI * 2 - Math.PI / 2;
        var depth = index % 2 ? 0.66 : 1;
        body.x = size.w / 2 + Math.cos(angle) * size.w * 0.34 * depth;
        body.y = field.cy + Math.sin(angle) * field.h * 0.3 * depth;

        bindTopic(body);
        topics.push(body);
      });
    }

    function bindTopic(body) {
      body.el.addEventListener("click", function (event) {
        event.stopPropagation();
        if (pointer.suppress) { pointer.suppress = false; return; }
        setExpanded(state.topic === body.topic.id ? null : body.topic.id, body.el);
      });
      body.el.addEventListener("pointerenter", function () {
        state.hover = body;
        refreshClasses();
        if (!state.broken && fine.matches) {
          links.forEach(function (link) {
            if (link.a === body || link.b === body) spawnPulse(link);
          });
        }
      });
      body.el.addEventListener("pointerleave", function () {
        if (state.hover === body) { state.hover = null; refreshClasses(); }
      });
      body.el.addEventListener("blur", function () {
        if (state.hover === body) { state.hover = null; refreshClasses(); }
      });
      body.el.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " " || event.key === "Spacebar") {
          event.preventDefault();
          setExpanded(state.topic === body.topic.id ? null : body.topic.id, body.el);
        }
      });
    }

    // --- article nodes ----------------------------------------------------
    function setExpanded(id, trigger) {
      var next = id || null;
      if (state.topic === next) return;
      if (state.open) closeSheet(false);
      state.topic = next;
      state.trigger = trigger || null;
      buildField();
      updateStatus();
      if (!reduce.matches) startMotion(); else settle();
    }

    function setMode(mode) {
      if (state.mode === mode) return;
      state.mode = mode;
      modeButtons.forEach(function (button) {
        button.setAttribute("aria-pressed", button.getAttribute("data-network-mode") === mode ? "true" : "false");
      });
      if (state.open) closeSheet(false);
      state.topic = null;
      buildField();
      updateStatus();
      if (!reduce.matches) startMotion(); else settle();
    }

    // --- field construction ----------------------------------------------
    function radiusFor(node, scale) {
      if (node.total < 2) return R_MAX * scale;
      return (R_MAX - (node.index / (node.total - 1)) * (R_MAX - R_MIN)) * scale;
    }

    // Narrow stages get a smaller field: a 240px-wide panel cannot hold eight
    // full-size circles without piling them up.
    function nodeScaleValue() { return clamp(size.w / 900, 0.55, 1); }

    function buildNodes() {
      var count = data.articles.length;
      nodeScale = nodeScaleValue();
      data.articles.forEach(function (article, index) {
        var node = {
          kind: "article",
          article: article,
          index: index,
          hubIds: article.topics.filter(function (id) { return topicById[id]; }),
          x: 0, y: 0, vx: 0, vy: 0, fx: 0, fy: 0,
          phase: index * 1.7,
          scale: 1, flash: -1,
          vis: 0, active: false, anchor: null,
          near: false
        };
        node.total = count;
        node.r = radiusFor(node, nodeScale);
        node.cr = node.r;

        var group = element("g", "an-node", layers.nodes);
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("aria-label", article.title + " — " + article.month +
          (article.topics.length ? " · " + article.topics.map(topicLabel).join(", ") : ""));

        node.el = group;
        node.halo = element("circle", "an-node-halo", group);
        node.outline = element("circle", "an-node-outline", group);
        if (article.image) {
          node.image = element("image", "an-node-img", group);
          node.image.setAttribute("preserveAspectRatio", "xMidYMid slice");
          node.image.setAttribute("clip-path", "url(#an-clip)");
          node.image.setAttribute("href", article.image);
        } else {
          node.fallback = element("circle", "an-node-fallback", group);
        }
        node.ring = element("circle", "an-node-ring", group);

        var lines = wrapTitle(article.shortTitle || article.title);
        node.lines = lines;
        node.lineCount = lines.length;
        node.longestLine = Math.max.apply(null, lines.map(function (line) { return line.length; }));
        node.label = element("text", "an-node-label", group);
        lines.forEach(function (line, lineIndex) {
          var tspan = element("tspan", null, node.label);
          tspan.setAttribute("x", 0);
          if (lineIndex) tspan.setAttribute("dy", "13");
          tspan.textContent = lineIndex === 0 && lines.length > 1 ? line + " " : line;
        });
        node.date = element("text", "an-node-date", group);
        node.date.textContent = article.month;
        node.hit = element("circle", "an-node-hit", group);

        // Radii follow the stage width, so every mark is placed by fit().
        node.fit = function () {
          var r = node.r;
          node.halo.setAttribute("r", r + 7);
          node.outline.setAttribute("r", r + 2.5);
          if (node.image) {
            node.image.setAttribute("x", -r);
            node.image.setAttribute("y", -r);
            node.image.setAttribute("width", r * 2);
            node.image.setAttribute("height", r * 2);
          } else if (node.fallback) {
            node.fallback.setAttribute("r", r);
          }
          node.ring.setAttribute("r", r);
          node.hit.setAttribute("r", r + 6);
          node.label.setAttribute("y", r + 20);
          node.date.setAttribute("y", r + 34 + (node.lineCount - 1) * 13);
          // Half-width of the widest caption line that will actually be drawn.
          var narrow = svg.classList.contains("is-narrow");
          var caption = narrow ? wrapTitle(article.shortTitle || article.title, 14)[0] : node.lines[0];
          if (narrow && caption !== (article.shortTitle || article.title)) caption += "…";
          node.label.firstChild.textContent = caption + (!narrow && node.lineCount > 1 ? " " : "");
          node.labelHalf = (narrow ? caption.length : node.longestLine) * (narrow ? 3.1 : 3.6);
        };
        node.fit();

        // Start somewhere near the middle so the first frame is not a pile-up.
        node.x = size.w / 2 + (index % 2 ? 1 : -1) * (60 + index * 18);
        node.y = field.cy + ((index % 3) - 1) * (70 + index * 12);

        nodes.push(node);
      });
      // Now that article bodies exist, wire each keyword to its members.
      topics.forEach(function (body) {
        body.members = nodes.filter(function (node) {
          return node.hubIds.indexOf(body.topic.id) >= 0;
        });
      });
    }

    function similarity(a, b) {
      var shared = 0;
      for (var i = 0; i < a.hubIds.length; i++) {
        if (b.hubIds.indexOf(a.hubIds[i]) >= 0) shared++;
      }
      var union = a.hubIds.length + b.hubIds.length - shared;
      return union ? shared / union : 0;
    }

    function linkElement(className) {
      var line = element("line", className, layers.links);
      line.setAttribute("x1", 0);
      line.setAttribute("y1", 0);
      line.setAttribute("x2", 0);
      line.setAttribute("y2", 0);
      return line;
    }

    // Similarity web over a set of articles: the strongest few bonds each,
    // not the complete graph that shares one broad topic with everything.
    function pairLinks(set) {
      var candidates = [];
      for (var i = 0; i < set.length; i++) {
        for (var j = i + 1; j < set.length; j++) {
          var weight = similarity(set[i], set[j]);
          if (weight >= LINK_FLOOR) candidates.push({ a: set[i], b: set[j], weight: weight });
        }
      }
      var kept = [];
      set.forEach(function (node) {
        candidates
          .filter(function (pair) { return pair.a === node || pair.b === node; })
          .sort(function (x, y) { return y.weight - x.weight; })
          .slice(0, LINK_KEEP)
          .forEach(function (pair) { if (kept.indexOf(pair) < 0) kept.push(pair); });
      });
      kept.forEach(function (pair) {
        var link = {
          a: pair.a, b: pair.b, weight: pair.weight, kind: "pair",
          articles: [pair.a, pair.b],
          el: linkElement("an-link")
        };
        link.base = 0.16 + 0.66 * pair.weight;
        links.push(link);
      });
    }

    function buildField() {
      clear(layers.links);
      clear(layers.years);
      clear(layers.pulses);
      links = [];
      pulses = [];
      allBodies().forEach(function (body) { body.pulls = []; });

      if (state.mode === "topic") {
        var open = state.topic ? topics.filter(function (body) { return body.topic.id === state.topic; })[0] : null;
        if (!open) {
          // The keyword graph: topics linked by the articles they share, held
          // on a soft ring so the field spreads across the stage.
          topics.forEach(function (body, index) {
            body.active = true;
            body.anchor = null;
            var angle = (index / topics.length) * Math.PI * 2 - Math.PI / 2;
            var depth = index % 2 ? 0.66 : 1;
            body.slot = {
              x: size.w / 2 + Math.cos(angle) * size.w * 0.33 * depth,
              y: field.cy + Math.sin(angle) * field.h * 0.3 * depth
            };
          });
          nodes.forEach(function (node) {
            node.active = false;
            node.slot = null;
          });
          for (var i = 0; i < topics.length; i++) {
            for (var j = i + 1; j < topics.length; j++) {
              var shared = 0;
              for (var m = 0; m < topics[i].members.length; m++) {
                if (topics[j].members.indexOf(topics[i].members[m]) >= 0) shared++;
              }
              if (shared < TOPIC_LINK_FLOOR) continue;
              var smaller = Math.min(topics[i].topic.count, topics[j].topic.count);
              var weight = smaller ? shared / smaller : 0;
              var link = {
                a: topics[i], b: topics[j], weight: weight, kind: "topic",
                articles: [topics[i], topics[j]],
                el: linkElement("an-link is-topic")
              };
              link.base = 0.14 + 0.5 * weight;
              links.push(link);
            }
          }
        } else {
          // The keyword breaks open: its articles bloom out of the node and
          // settle in orbit around it, bonded to it and to each other.
          open.active = true;
          open.slot = null;
          open.anchor = null;
          open.members.forEach(function (node) {
            if (!node.active && node.vis < 0.05) {
              node.x = open.x + (Math.random() - 0.5) * 24;
              node.y = open.y + (Math.random() - 0.5) * 24;
              node.vx = node.vy = 0;
            }
            node.active = true;
            node.slot = null;
            node.anchor = open;
            node.pulls = [open];
          });
          topics.forEach(function (body) {
            if (body === open) return;
            body.active = false;
            body.slot = null;
            body.anchor = null;
          });
          open.members.forEach(function (node) {
            links.push({
              a: open, b: node, weight: 1, kind: "spoke",
              articles: [open, node],
              el: linkElement("an-link is-spoke"),
              base: 0.5
            });
          });
          pairLinks(open.members);
        }
      } else {
        var pad = Math.min(160, size.w * 0.14);
        var span = Math.max(1, size.w - pad * 2);
        var previous = null;
        // The timeline reads left to right: the catalog arrives newest
        // first, but the field must place the oldest article on the left
        // and the newest (and any future year) on the right.
        var timeline = nodes.slice().sort(function (a, b) {
          return a.article.date < b.article.date ? -1 : a.article.date > b.article.date ? 1 : 0;
        });
        timeline.forEach(function (node, index) {
          node.active = true;
          node.anchor = null;
          node.slot = {
            x: pad + (timeline.length < 2 ? span / 2 : (index / (timeline.length - 1)) * span),
            y: field.cy + Math.sin(index * 1.9) * field.h * 0.15
          };
          if (previous) {
            links.push({ a: previous, b: node, weight: 1, kind: "chain", articles: [previous, node], el: linkElement("an-link is-chain"), base: 0.6 });
          }
          previous = node;
        });
        topics.forEach(function (body) {
          body.active = false;
          body.slot = null;
          body.anchor = null;
        });
        var seen = {};
        timeline.forEach(function (node) {
          var year = node.article.year;
          if (seen[year]) return;
          seen[year] = true;
          var rule = element("line", "an-year-rule", layers.years);
          rule.setAttribute("x1", node.slot.x);
          rule.setAttribute("x2", node.slot.x);
          rule.setAttribute("y1", field.top + 30);
          rule.setAttribute("y2", field.bottom - 26);
          var text = element("text", "an-year-label", layers.years);
          text.setAttribute("x", node.slot.x);
          text.setAttribute("y", field.top + 22);
          text.textContent = year;
        });
      }

      topics.forEach(function (body) { body.aria(); });
      refreshClasses();
      updateHint();
      if (reduce.matches) settle();
    }

    function refreshClasses() {
      allBodies().forEach(function (body) {
        var isOpen = body.kind === "article" && state.open === body.article.slug;
        var isOpenTopic = body.kind === "topic" && state.topic === body.topic.id;
        body.el.classList.toggle("is-active", isOpen || isOpenTopic);
        body.el.classList.toggle("is-near", state.hover === body);
      });
      var spotlight = state.open || (state.hover ? (state.hover.kind === "article" ? state.hover.article.slug : state.hover.topic.id) : null);
      links.forEach(function (link) {
        var lit = link.articles.some(function (body) {
          return body.kind === "article" ? body.article.slug === spotlight : body.topic.id === spotlight;
        });
        link.el.classList.toggle("is-lit", !!spotlight && lit);
        link.el.classList.toggle("is-dim", !!spotlight && !lit);
      });
    }

    // --- physics ----------------------------------------------------------
    // Bounds are asymmetric: a caption hangs below its circle and can be wider
    // than it, so both are kept clear of the stage edges and of the sheet.
    function boundsFor(body) {
      var inset = Math.max(body.cr + 14, Math.min(110, body.labelHalf + 14));
      var box = {
        left: inset,
        right: size.w - inset,
        top: field.top + body.r + 10,
        bottom: field.bottom - body.r - (body.kind === "article" ? 56 : 40)
      };
      if (state.open && !sheetStacked()) {
        box.right = Math.max(box.left, size.w - sheetLimit.right - inset);
      } else if (body.kind === "article" && state.open === body.article.slug) {
        box.bottom = Math.max(box.top, field.bottom - sheetLimit.bottom - body.r - 30);
      }
      return box;
    }

    // Where the article held in the sheet waits: the middle of what is still
    // visible, not the middle of the sheet.
    function focusPoint() {
      if (!state.open || !sheetStacked()) {
        var free = state.open ? size.w - sheetLimit.right : size.w;
        return { x: clamp(free / 2, 120, size.w - 120), y: field.cy };
      }
      return {
        x: size.w / 2,
        y: field.top + Math.max(0, field.h - sheetLimit.bottom) / 2
      };
    }

    function sheetStacked() { return window.innerWidth < 768; }

    function live(body) { return body.active || body.vis > 0.02; }

    function step(dt) {
      var i, j, a, b, dx, dy, d2, d, force, ux, uy;
      // The field is laid out in a square metric and stretched onto the stage, so
      // a wide screen gets a wide network instead of a circle in the middle.
      // Distances are therefore measured in "field units" while positions stay
      // in stage pixels.
      var base = Math.min(size.w, field.h) || 1;
      var ax = size.w / base;
      var ay = field.h / base;
      var bodies = allBodies().filter(live);

      for (i = 0; i < bodies.length; i++) { bodies[i].fx = 0; bodies[i].fy = 0; }

      for (i = 0; i < bodies.length; i++) {
        a = bodies[i];
        for (j = i + 1; j < bodies.length; j++) {
          b = bodies[j];
          dx = (b.x - a.x) / ax;
          dy = (b.y - a.y) / ay;
          d2 = dx * dx + dy * dy;
          if (d2 < 1) { d2 = 1; dx = (i - j) || 1; dy = 1; }
          d = Math.sqrt(d2);
          ux = dx / d;
          uy = dy / d;
          force = clamp(REPULSION / d2, 0, MAX_FORCE);
          a.fx -= ux * force / ax; a.fy -= uy * force / ay;
          b.fx += ux * force / ax; b.fy += uy * force / ay;

          var rest = a.cr + b.cr + (a.kind === "article" && b.kind === "article" ? 26 : 18);
          if (d < rest) {
            force = (rest - d) * 7;
            a.fx -= ux * force / ax; a.fy -= uy * force / ay;
            b.fx += ux * force / ax; b.fy += uy * force / ay;
          }
        }
      }

      // Springs along the live links: shared-topic bonds, keyword spokes and
      // the timeline chain all read as distances the field tries to keep.
      if (!state.broken) {
        for (i = 0; i < links.length; i++) {
          var pair = links[i];
          if (!pair.a.active || !pair.b.active) continue;
          dx = (pair.b.x - pair.a.x) / ax;
          dy = (pair.b.y - pair.a.y) / ay;
          d = Math.hypot(dx, dy) || 1;
          var restLen = pairRest;
          var strength = SPRING_PAIR * pair.weight;
          if (pair.kind === "spoke") {
            restLen = pair.a.cr + pair.b.cr + 64;
            strength = 2.4;
          } else if (pair.kind === "chain") {
            restLen = pairRest;
            strength = SPRING_DATE * 0.6;
          }
          force = (d - restLen) * strength;
          ux = dx / d;
          uy = dy / d;
          pair.a.fx += ux * force / ax;
          pair.a.fy += uy * force / ay;
          pair.b.fx -= ux * force / ax;
          pair.b.fy -= uy * force / ay;
        }
      }

      for (i = 0; i < bodies.length; i++) {
        a = bodies[i];

        if (!a.active) {
          // Folding away: suck back into the keyword that opened them.
          if (a.anchor) {
            a.fx += (a.anchor.x - a.x) * 6;
            a.fy += (a.anchor.y - a.y) * 6;
          }
        } else if (state.broken) {
          if (a.kind === "article" && state.open === a.article.slug) {
            var focus = focusPoint();
            a.fx += (focus.x - a.x) * SPRING_FOCUS;
            a.fy += (focus.y - a.y) * SPRING_FOCUS;
          } else if (a.kind === "topic") {
            // The open keyword stays put as the anchor of its bloom.
            var anchor = focusPoint();
            a.fx += (anchor.x - a.x) * 3;
            a.fy += (anchor.y - a.y) * 3;
          } else if (!reduce.matches) {
            a.fx += Math.sin(clock * 0.8 + a.phase) * WANDER;
            a.fy += Math.cos(clock * 0.66 + a.phase * 1.3) * WANDER;
          }
        } else if (state.mode === "date") {
          if (a.slot) {
            a.fx += (a.slot.x - a.x) * SPRING_DATE;
            a.fy += (a.slot.y - a.y) * SPRING_DATE;
          }
        } else if (a.kind === "topic" && state.topic === a.topic.id) {
          var hold = focusPoint();
          a.fx += (hold.x - a.x) * 3;
          a.fy += (hold.y - a.y) * 3;
        } else if (a.pulls && a.pulls.length) {
          // An open keyword pulls its articles into orbit around itself.
          var weight = 1 / a.pulls.length;
          for (var k = 0; k < a.pulls.length; k++) {
            var hub = a.pulls[k];
            a.fx += (hub.x - a.x) * SPRING_HUB * weight;
            a.fy += (hub.y - a.y) * SPRING_HUB * weight;
            dx = (a.x - hub.x) / ax;
            dy = (a.y - hub.y) / ay;
            d = Math.hypot(dx, dy) || 1;
            var orbit = hub.cr + a.cr + 46;
            if (d < orbit) {
              force = (orbit - d) * RING_PUSH;
              a.fx += (dx / d) * force / ax;
              a.fy += (dy / d) * force / ay;
            }
          }
        } else {
          // Free field: the ring of slots plus a light centring keep the field
          // spread across the stage instead of collapsing into one cluster.
          if (a.slot) {
            a.fx += (a.slot.x - a.x) * SPRING_SLOT;
            a.fy += (a.slot.y - a.y) * SPRING_SLOT;
          }
          a.fx += (size.w / 2 - a.x) * CENTER_PULL;
          a.fy += (field.cy - a.y) * CENTER_PULL;
        }

        if (pointer.inside && !state.broken && a.active) {
          dx = (pointer.x - a.x) / ax;
          dy = (pointer.y - a.y) / ay;
          d = Math.hypot(dx, dy) || 1;
          if (d < MOUSE_RANGE) {
            force = (1 - d / MOUSE_RANGE) * MOUSE_PULL;
            a.fx += (dx / d) * force / ax;
            a.fy += (dy / d) * force / ay;
          }
        }

        a.fx = clamp(a.fx, -MAX_FORCE, MAX_FORCE);
        a.fy = clamp(a.fy, -MAX_FORCE, MAX_FORCE);
      }

      var damp = Math.exp(-DAMPING * dt);
      for (i = 0; i < bodies.length; i++) {
        a = bodies[i];
        a.vx = (a.vx + a.fx * dt) * damp;
        a.vy = (a.vy + a.fy * dt) * damp;
        var speed = Math.hypot(a.vx, a.vy);
        if (speed > 1100) { a.vx *= 1100 / speed; a.vy *= 1100 / speed; }
        a.x += a.vx * dt;
        a.y += a.vy * dt;

        var box = boundsFor(a);
        var left = box.left;
        var right = box.right;
        var top = box.top;
        var bottom = box.bottom;
        if (state.broken && a.active) {
          if (a.x < left) { a.x = left; a.vx = Math.abs(a.vx) * RESTITUTION + 14; }
          else if (a.x > right) { a.x = right; a.vx = -Math.abs(a.vx) * RESTITUTION - 14; }
          if (a.y < top) { a.y = top; a.vy = Math.abs(a.vy) * RESTITUTION + 14; }
          else if (a.y > bottom) { a.y = bottom; a.vy = -Math.abs(a.vy) * RESTITUTION - 14; }
        } else {
          a.x = clamp(a.x, left, Math.max(left, right));
          a.y = clamp(a.y, top, Math.max(top, bottom));
        }

        // Hover easing and the bloom/fold fade share the frame clock.
        a.scale = ease(a.scale, a.near || (a.kind === "article" && state.open === a.article.slug) ||
          (a.kind === "topic" && state.topic === a.topic.id) ? 1.09 : 1, dt, 10);
        a.vis = ease(a.vis, a.active ? 1 : 0, dt, FADE);
      }
    }

    function settle(steps) {
      var count = steps || 220;
      for (var i = 0; i < count; i++) step(1 / 60);
      draw();
    }

    // --- pulses -----------------------------------------------------------
    function spark(x, y) {
      if (reduce.matches) return;
      var ripple = element("circle", "an-pulse-tail", layers.pulses);
      ripple.setAttribute("cx", x);
      ripple.setAttribute("cy", y);
      ripple.setAttribute("r", 6);
      pulses.push({ ripple: ripple, t: 0, life: 1.1 });
    }

    function spawnPulse(link, delay) {
      if (!link || reduce.matches) return;
      var pulse = {
        link: link,
        t: -(delay || 0),
        speed: 0.5 + Math.random() * 0.4,
        dot: element("circle", "an-pulse", layers.pulses),
        tail: element("circle", "an-pulse-tail", layers.pulses)
      };
      pulse.dot.setAttribute("r", 3);
      pulse.tail.setAttribute("r", 5);
      pulses.push(pulse);
    }

    function updatePulses(dt) {
      var arrival = [];
      for (var i = pulses.length - 1; i >= 0; i--) {
        var pulse = pulses[i];
        pulse.t += dt;
        if (pulse.ripple) {
          if (pulse.t > pulse.life) {
            pulse.ripple.remove();
            pulses.splice(i, 1);
            continue;
          }
          pulse.ripple.setAttribute("r", 6 + pulse.t * 90);
          pulse.ripple.setAttribute("opacity", Math.max(0, 1 - pulse.t / pulse.life).toFixed(3));
          continue;
        }
        if (pulse.t < 0) {
          pulse.dot.setAttribute("opacity", "0");
          pulse.tail.setAttribute("opacity", "0");
          continue;
        }
        if (pulse.t >= 1) {
          arrival.push(pulse.link.b);
          pulse.dot.remove();
          pulse.tail.remove();
          pulses.splice(i, 1);
          continue;
        }
        var from = pulse.link.a;
        var to = pulse.link.b;
        var x = from.x + (to.x - from.x) * pulse.t;
        var y = from.y + (to.y - from.y) * pulse.t;
        var back = Math.max(0, pulse.t - 0.035);
        pulse.dot.setAttribute("cx", x);
        pulse.dot.setAttribute("cy", y);
        pulse.dot.setAttribute("opacity", "1");
        pulse.tail.setAttribute("cx", from.x + (to.x - from.x) * back);
        pulse.tail.setAttribute("cy", from.y + (to.y - from.y) * back);
        pulse.tail.setAttribute("opacity", "1");
      }
      arrival.forEach(function (target) {
        if (target && target.kind === "article") target.flash = clock + FLASH;
      });
    }

    // --- render -----------------------------------------------------------
    function draw() {
      var i;
      var bodies = allBodies();
      for (i = 0; i < bodies.length; i++) {
        var body = bodies[i];
        var hidden = !body.active && body.vis <= 0.02;
        if (hidden && body.el.style.display !== "none") body.el.style.display = "none";
        else if (!hidden && body.el.style.display === "none") body.el.style.display = "";
        if (hidden) continue;
        var bloom = 0.55 + 0.45 * body.vis;
        body.el.setAttribute("transform",
          "translate(" + body.x.toFixed(2) + " " + body.y.toFixed(2) + ") scale(" + (body.scale * bloom).toFixed(3) + ")");
        body.el.style.opacity = body.vis.toFixed(3);
        body.el.classList.toggle("is-fading", body.vis < 0.5);
        if (body.kind === "article") body.outline.classList.toggle("is-lit", body.flash > clock);
      }
      for (i = 0; i < links.length; i++) {
        var link = links[i];
        link.el.setAttribute("x1", link.a.x.toFixed(2));
        link.el.setAttribute("y1", link.a.y.toFixed(2));
        link.el.setAttribute("x2", link.b.x.toFixed(2));
        link.el.setAttribute("y2", link.b.y.toFixed(2));
        link.el.setAttribute("opacity", (link.base * Math.min(link.a.vis, link.b.vis)).toFixed(3));
      }
    }

    // --- view (zoom / pan) ------------------------------------------------
    // Keyboard focus must not sit on a node the stage is clipping.
    function ensureVisible(body) {
      var pad = body.cr * view.k + 44;
      var screenX = body.x * view.k + view.tx;
      var screenY = body.y * view.k + view.ty;
      var dx = 0;
      var dy = 0;
      if (screenX < pad) dx = pad - screenX;
      else if (screenX > size.w - pad) dx = size.w - pad - screenX;
      if (screenY < field.top + pad) dy = field.top + pad - screenY;
      else if (screenY > field.bottom - pad) dy = field.bottom - pad - screenY;
      if (!dx && !dy) return;
      view.tx += dx;
      view.ty += dy;
      clampView();
      applyView();
    }

    function applyView() {
      layers.view.setAttribute("transform",
        "translate(" + view.tx.toFixed(2) + " " + view.ty.toFixed(2) + ") scale(" + view.k.toFixed(3) + ")");
      svg.classList.toggle("is-zoomed-out", view.k < 0.8);
      if (zoomInButton) zoomInButton.disabled = view.k >= ZOOM_MAX - 0.01;
      if (zoomOutButton) zoomOutButton.disabled = view.k <= ZOOM_MIN + 0.01;
    }

    function clampView() {
      view.tx = clamp(view.tx, size.w * 0.25 - size.w * view.k, size.w * 0.75);
      view.ty = clamp(view.ty, size.h * 0.25 - size.h * view.k, size.h * 0.75);
    }

    function zoomAt(factor, localX, localY) {
      var previous = view.k;
      view.k = clamp(view.k * factor, ZOOM_MIN, ZOOM_MAX);
      if (view.k === previous) return;
      var worldX = (localX - view.tx) / previous;
      var worldY = (localY - view.ty) / previous;
      view.tx = localX - worldX * view.k;
      view.ty = localY - worldY * view.k;
      clampView();
      applyView();
    }

    function resetView() {
      view.k = 1;
      view.tx = 0;
      view.ty = 0;
      applyView();
    }

    function resize() {
      root.style.setProperty("--network-nav-height", (root.getBoundingClientRect().top + window.scrollY) + "px");
      var rect = stage.getBoundingClientRect();
      if (rect.width < 40 || rect.height < 40) return;
      var top = header ? Math.ceil(header.getBoundingClientRect().bottom - rect.top) : 0;
      var bottom = footer ? Math.floor(footer.getBoundingClientRect().top - rect.top) : rect.height;
      var changed = Math.abs(size.w - rect.width) > 2 || Math.abs(size.h - rect.height) > 2 ||
        Math.abs(field.top - top) > 2 || Math.abs(field.bottom - bottom) > 2;
      size.w = rect.width;
      size.h = rect.height;
      field.top = top;
      field.bottom = Math.max(top + 1, bottom);
      field.h = field.bottom - field.top;
      field.cy = (field.top + field.bottom) / 2;
      // The canvas fills the screen; only marks avoid the overlaid page chrome.
      root.style.setProperty("--network-top", field.top + "px");
      root.style.setProperty("--network-bottom", (size.h - field.bottom) + "px");
      pairRest = clamp(Math.min(size.w, field.h) * PAIR_SPACING, PAIR_MIN, PAIR_MAX);
      scaleNodes();
      svg.setAttribute("viewBox", "0 0 " + size.w.toFixed(1) + " " + size.h.toFixed(1));
      applyView();
      measureSheet();
      if (!changed || !nodes.length) return;
      buildField();
    }

    // Circles, captions and hit areas follow the screen width.
    function scaleNodes() {
      var scale = nodeScaleValue();
      var narrowed = svg.classList.contains("is-narrow");
      svg.classList.toggle("is-narrow", size.w < 620);
      var changed = Math.abs(scale - nodeScale) > 0.01 || narrowed !== svg.classList.contains("is-narrow");
      nodeScale = scale;
      if (!changed) return;
      nodes.forEach(function (node) {
        node.r = radiusFor(node, scale);
        node.cr = node.r;
        node.fit();
      });
    }

    function measureSheet() {
      var rect = sheet.getBoundingClientRect();
      var stacked = sheetStacked();
      sheetLimit.right = stacked ? 0 : rect.width + 16;
      sheetLimit.bottom = stacked ? rect.height : 0;
    }

    // --- motion loop ------------------------------------------------------
    function running() { return !reduce.matches && !document.hidden; }

    function frame(now) {
      raf = 0;
      if (!running()) return;
      var dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
      last = now;
      clock += dt;
      step(dt);
      draw();
      updatePulses(dt);

      pulseTimer += dt;
      if (pulseTimer > PULSE_EVERY && links.length) {
        pulseTimer = 0;
        var candidates = links.filter(function (link) {
          return link.a.active && link.b.active && !link.el.classList.contains("is-dim");
        });
        if (candidates.length) spawnPulse(candidates[Math.floor(Math.random() * candidates.length)]);
      }

      raf = requestAnimationFrame(frame);
    }

    function startMotion() {
      last = 0;
      if (!raf && running()) raf = requestAnimationFrame(frame);
    }

    function stopMotion() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
    }

    // --- summary sheet ----------------------------------------------------
    function openSheet(node, trigger) {
      var article = node.article;
      if (state.open === article.slug) return;
      state.open = article.slug;
      state.broken = true;
      state.trigger = trigger || null;
      root.classList.add("is-reading");
      renderSheet(article);
      resetView();
      updateStatus();
      resize();
      allBodies().forEach(function (other) {
        var box = boundsFor(other);
        other.x = clamp(other.x, box.left, box.right);
        other.y = clamp(other.y, box.top, box.bottom);
      });
      draw();
      sheet.classList.add("is-open");
      sheet.setAttribute("aria-hidden", "false");
      node.vx += (node.x - size.w / 2) * 0.4;
      node.vy += (node.y - field.cy) * 0.4;
      allBodies().forEach(function (other) {
        if (other === node || !other.active) return;
        var dx = other.x - node.x;
        var dy = other.y - node.y;
        var d = Math.hypot(dx, dy) || 1;
        var kick = Math.max(0, 1 - d / 420) * 520 + 90;
        other.vx += (dx / d) * kick;
        other.vy += (dy / d) * kick - 60;
      });
      refreshClasses();
      var title = sheet.querySelector(".network-sheet-title");
      if (title) title.focus({ preventScroll: true });
      if (reduce.matches) settle();
    }

    function closeSheet(returnFocus) {
      if (!state.open) return;
      var trigger = state.trigger;
      state.open = null;
      state.broken = false;
      state.trigger = null;
      root.classList.remove("is-reading");
      sheet.classList.remove("is-open");
      sheet.setAttribute("aria-hidden", "true");
      refreshClasses();
      updateStatus();
      resize();
      if (returnFocus !== false && trigger) trigger.focus({ preventScroll: true });
      if (reduce.matches) settle();
    }

    function renderSheet(article) {
      sheet.innerHTML = "";
      var close = document.createElement("button");
      close.type = "button";
      close.className = "network-sheet-close";
      close.setAttribute("data-network-close", "");
      close.setAttribute("aria-label", "Close the summary");
      close.textContent = "×";
      sheet.appendChild(close);

      var body = document.createElement("div");
      body.className = "network-sheet-body";
      var previewHeader = document.createElement("div");
      previewHeader.className = "network-sheet-header";
      var source = article.image || article.imageSource;
      if (source) {
        var media = document.createElement("div");
        media.className = "network-sheet-media";
        var img = document.createElement("img");
        img.src = source;
        img.alt = article.imageAlt || "";
        media.appendChild(img);
        previewHeader.appendChild(media);
      }
      var byline = document.createElement("div");
      var kicker = document.createElement("p");
      kicker.className = "network-sheet-kicker";
      kicker.textContent = "Article preview";
      byline.appendChild(kicker);
      var meta = document.createElement("time");
      meta.className = "network-sheet-meta";
      meta.dateTime = article.date;
      meta.textContent = article.month;
      byline.appendChild(meta);
      previewHeader.appendChild(byline);
      body.appendChild(previewHeader);

      sheet.setAttribute("aria-labelledby", "network-sheet-title");
      var title = document.createElement("h3");
      title.className = "network-sheet-title";
      title.id = "network-sheet-title";
      title.tabIndex = -1;
      title.textContent = article.title;
      body.appendChild(title);

      if (article.description) {
        var summary = document.createElement("p");
        summary.className = "network-sheet-summary";
        summary.textContent = article.description;
        body.appendChild(summary);
      }
      if (article.topics.length) {
        var topicsLine = document.createElement("p");
        topicsLine.className = "network-sheet-topics";
        topicsLine.textContent = article.topics.map(topicLabel).join(" · ");
        body.appendChild(topicsLine);
      }

      var actions = document.createElement("div");
      actions.className = "network-sheet-actions";
      var more = document.createElement("a");
      more.className = "btn btn-primary";
      more.href = article.url;
      more.innerHTML = 'Read more <span aria-hidden="true">↗</span>';
      actions.appendChild(more);
      if (article.audio) {
        var play = document.createElement("button");
        play.type = "button";
        play.className = "btn btn-audio-play";
        play.setAttribute("aria-label", "Listen to this article");
        play.textContent = "▶";
        var audio = document.createElement("audio");
        audio.preload = "none";
        audio.src = article.audio;
        play.addEventListener("click", function () {
          if (audio.paused) { audio.play(); play.textContent = "⏸"; }
          else { audio.pause(); play.textContent = "▶"; }
        });
        actions.appendChild(play);
        actions.appendChild(audio);
      }
      var back = document.createElement("button");
      back.type = "button";
      back.className = "btn btn-quiet network-sheet-back";
      back.setAttribute("data-network-close", "");
      back.textContent = "Back to network";
      actions.appendChild(back);
      sheet.appendChild(body);
      sheet.appendChild(actions);
    }

    // --- status -----------------------------------------------------------
    function updateStatus() {
      var text;
      if (state.mode === "date") {
        text = nodes.length + " articles · along the timeline";
      } else if (state.topic) {
        var open = topics.filter(function (body) { return body.topic.id === state.topic; })[0];
        text = topicLabel(state.topic) + " · " + (open ? open.members.length : 0) + " articles · keyword open";
      } else {
        text = topics.length + " keywords · " + links.length + " connections";
      }
      statusEl.textContent = text + (state.open ? " · summary open" : "");
    }

    function updateHint() {
      if (!hintEl) return;
      if (state.mode === "date") {
        hintEl.textContent = "Drag to pan · Scroll or pinch to zoom · Select an article for its summary · Esc to close";
      } else if (state.topic) {
        hintEl.textContent = "Select an article for its summary · Click the keyword again, Esc or an empty spot to fold it back";
      } else {
        hintEl.textContent = "Pick a keyword to open its articles · Drag to pan · Scroll or pinch to zoom";
      }
    }

    // --- events -----------------------------------------------------------
    function bindEvents() {
      modeButtons.forEach(function (button) {
        button.addEventListener("click", function () { setMode(button.getAttribute("data-network-mode")); });
      });

      root.addEventListener("click", function (event) {
        if (event.target.closest("[data-network-close]")) closeSheet();
        var zoom = event.target.closest("[data-network-zoom]");
        if (zoom) {
          zoomAt(zoom.getAttribute("data-network-zoom") === "in" ? 1.35 : 1 / 1.35, size.w / 2, field.cy);
        }
        if (event.target.closest("[data-network-reset]")) resetView();
      });

      nodes.forEach(function (node) {
        node.el.addEventListener("click", function (event) {
          event.stopPropagation();
          if (pointer.suppress) { pointer.suppress = false; return; }
          openSheet(node, node.el);
        });
        node.el.addEventListener("pointerenter", function () {
          state.hover = node;
          refreshClasses();
          if (!state.broken && fine.matches) {
            links.forEach(function (link) {
              if (link.a === node || link.b === node) spawnPulse(link);
            });
          }
        });
        node.el.addEventListener("pointerleave", function () {
          if (state.hover === node) {
            state.hover = null;
            refreshClasses();
          }
        });
        node.el.addEventListener("blur", function () {
          if (state.hover === node) { state.hover = null; refreshClasses(); }
        });
        node.el.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " " || event.key === "Spacebar") {
            event.preventDefault();
            openSheet(node, node.el);
          }
        });
      });

      // focusin bubbles (focus does not), so keyboard focus both lights the mark
      // up and pans it back into view.
      svg.addEventListener("focusin", function (event) {
        var body = null;
        var bodies = allBodies();
        for (var i = 0; i < bodies.length; i++) {
          if (bodies[i].el === event.target) { body = bodies[i]; break; }
        }
        if (!body) return;
        state.hover = body;
        refreshClasses();
        // Only keyboard focus pans the field: a mouse click focuses the mark
        // too, and jumping the view under the pointer would feel like a bug.
        if (event.target.matches(":focus-visible")) ensureVisible(body);
      });

      svg.addEventListener("wheel", function (event) {
        event.preventDefault();
        var rect = stage.getBoundingClientRect();
        var step = event.deltaMode === 1 ? 0.04 : 0.0016;
        zoomAt(Math.exp(-event.deltaY * step), event.clientX - rect.left, event.clientY - rect.top);
      }, { passive: false });

      svg.addEventListener("pointerdown", function (event) {
        touches[event.pointerId] = { x: event.clientX, y: event.clientY };
        pointer.suppress = false;   // a fresh press always allows the next click
        var rect = stage.getBoundingClientRect();
        pointer.x = (event.clientX - rect.left - view.tx) / view.k;
        pointer.y = (event.clientY - rect.top - view.ty) / view.k;
        pointer.inside = true;
        pointer.moved = false;
        pointer.down = true;
        pointer.originX = event.clientX;
        pointer.originY = event.clientY;
        if (Object.keys(touches).length === 2) {
          pointer.pinch = pinchState();
          pointer.panning = false;
        } else if (!event.target.closest(".an-node") && !event.target.closest(".an-topic")) {
          pointer.panning = true;
          svg.setPointerCapture(event.pointerId);
          stage.classList.add("is-panning");
        }
      });

      svg.addEventListener("pointermove", function (event) {
        var rect = stage.getBoundingClientRect();
        var localX = event.clientX - rect.left;
        var localY = event.clientY - rect.top;
        if (touches[event.pointerId]) { touches[event.pointerId].x = event.clientX; touches[event.pointerId].y = event.clientY; }

        var ids = Object.keys(touches);
        if (pointer.pinch && ids.length >= 2) {
          var pinch = pinchState();
          zoomAt(pinch.distance / pointer.pinch.distance, pinch.x - rect.left, pinch.y - rect.top);
          view.tx += pinch.x - pointer.pinch.x;
          view.ty += pinch.y - pointer.pinch.y;
          clampView();
          applyView();
          pointer.pinch = pinch;
          pointer.moved = true;
          return;
        }

        if (pointer.down) {
          var dx = event.clientX - pointer.originX;
          var dy = event.clientY - pointer.originY;
          if (Math.hypot(dx, dy) > 6) pointer.moved = true;
          if (pointer.panning && pointer.moved) {
            view.tx += event.clientX - (pointer.lastX || pointer.originX);
            view.ty += event.clientY - (pointer.lastY || pointer.originY);
            clampView();
            applyView();
          }
        }
        pointer.lastX = event.clientX;
        pointer.lastY = event.clientY;
        pointer.x = (localX - view.tx) / view.k;
        pointer.y = (localY - view.ty) / view.k;
        pointer.inside = true;
      });

      function releasePointer(event) {
        delete touches[event.pointerId];
        if (Object.keys(touches).length < 2) pointer.pinch = null;
        if (pointer.down) {
          pointer.down = false;
          pointer.panning = false;
          stage.classList.remove("is-panning");
          if (!pointer.moved && !event.target.closest(".an-node") && !event.target.closest(".an-topic")) {
            if (state.open) closeSheet();
            else if (state.topic) setExpanded(null, null);
            else {
              var pulseTarget = allBodies().filter(function (body) { return body.active; }).sort(function (a, b) {
                return Math.hypot(a.x - pointer.x, a.y - pointer.y) - Math.hypot(b.x - pointer.x, b.y - pointer.y);
              })[0];
              spark(clamp(pointer.x, 0, size.w), clamp(pointer.y, 0, size.h));
              if (pulseTarget) {
                links.forEach(function (link) {
                  if (link.a === pulseTarget || link.b === pulseTarget) spawnPulse(link, Math.random() * 0.2);
                });
              }
            }
          }
        }
        pointer.suppress = pointer.moved;
        pointer.moved = false;
        pointer.lastX = pointer.lastY = undefined;
      }

      svg.addEventListener("pointerup", releasePointer);
      svg.addEventListener("pointercancel", releasePointer);
      svg.addEventListener("pointerleave", function () {
        pointer.inside = false;
        state.hover = null;
        refreshClasses();
      });

      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape") {
          if (state.open) {
            event.preventDefault();
            closeSheet();
            return;
          }
          if (state.topic) {
            event.preventDefault();
            setExpanded(null, null);
            return;
          }
        }
        // Keyboard navigation reads the focused element: SVG groups do not
        // reliably fire focus events, but focus() always moves activeElement.
        var bodies = allBodies().filter(function (body) { return body.active; });
        var index = -1;
        for (var i = 0; i < bodies.length; i++) {
          if (bodies[i].el === document.activeElement) { index = i; break; }
        }
        if (index < 0 && state.hover) index = bodies.indexOf(state.hover);
        if (index < 0 || event.key.indexOf("Arrow") !== 0) return;
        var body = bodies[index];
        var best = null;
        var bestScore = Infinity;
        bodies.forEach(function (other) {
          if (other === body) return;
          var dx = other.x - body.x;
          var dy = other.y - body.y;
          var along = event.key === "ArrowRight" ? dx : event.key === "ArrowLeft" ? -dx : event.key === "ArrowDown" ? dy : -dy;
          var across = event.key === "ArrowRight" || event.key === "ArrowLeft" ? Math.abs(dy) : Math.abs(dx);
          if (along <= 0) return;
          var score = along + across * 1.6;
          if (score < bestScore) { bestScore = score; best = other; }
        });
        if (best) {
          event.preventDefault();
          best.el.focus({ preventScroll: true });
        }
      });

      var resizeTimer = 0;
      function scheduleResize() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function () {
          resizeTimer = 0;
          resize();
          if (reduce.matches) settle();
        }, 140);
      }

      window.addEventListener("resize", scheduleResize, { passive: true });

      document.addEventListener("visibilitychange", function () {
        if (document.hidden) {
          stopMotion();
          return;
        }
        // Hidden frames do not get resize notifications, so re-measure on return.
        resize();
        startMotion();
      });

      if (window.ResizeObserver) {
        var observer = new ResizeObserver(scheduleResize);
        observer.observe(stage);
        if (header) observer.observe(header);
        if (footer) observer.observe(footer);
        if (navbar) observer.observe(navbar);
      }

      reduce.addEventListener("change", function () {
        if (reduce.matches) { stopMotion(); settle(); } else startMotion();
      });
      fine.addEventListener("change", function () {
        if (!fine.matches) { pointer.inside = false; }
      });
    }

    function pinchState() {
      var ids = Object.keys(touches);
      var a = touches[ids[0]];
      var b = touches[ids[1]];
      return {
        x: (a.x + b.x) / 2,
        y: (a.y + b.y) / 2,
        distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y))
      };
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
