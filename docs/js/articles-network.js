// Articles network — the Articles page drawn as a field of article thumbnails.
//
// Data comes from docs/articles-network.json (generate_articles_network.py).
// "By topic" pulls every article toward the hubs of its topics; "by date" lays
// them along a timeline. Clicking an article breaks the layout — the rest float
// and bounce off the walls — and opens the summary sheet; closing rebuilds it.
// Drag pans, wheel or pinch zooms, and pulses run along the links.
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  // --- tuning -------------------------------------------------------------
  var HUB_MIN = 2;          // articles a topic needs before it earns a hub
  var CHIP_LIMIT = 12;      // topic chips before the "+N more" toggle
  var LINK_KEEP = 3;        // strongest relationships drawn per article
  var R_MIN = 30;           // radius of the oldest article
  var R_MAX = 44;           // radius of the newest article
  var HUB_R = 15;           // topic hub marker
  var ZOOM_MIN = 0.55;
  var ZOOM_MAX = 2.6;
  var REPULSION = 7.5e6;    // px³/s² between articles: keeps unlinked articles apart
  var MAX_FORCE = 3200;     // px/s² clamp on each force
  var DAMPING = 2.3;        // 1/s exponential velocity decay
  var SPRING_PAIR = 6.0;    // 1/s² per unit of topic similarity between articles
  var SPRING_HUB = 3.1;     // 1/s² pull toward the picked topic
  var SPRING_DATE = 4.2;    // 1/s² pull toward a timeline slot
  var SPRING_FOCUS = 8.0;   // 1/s² pull for the article held in the sheet
  var RING_PUSH = 2.6;      // 1/s² soft orbit radius around a hub
  var CENTER_PULL = 0.32;   // 1/s² pull keeping the field inside the stage
  var SPRING_SLOT = 1.1;    // 1/s² pull toward an article's slot in the ring
  var PAIR_SPACING = 0.5;   // share of the stage a similarity link spans at rest
  var PAIR_MIN = 160;       // px floor for that resting length
  var PAIR_MAX = 340;       // px ceiling for that resting length
  var LINK_FLOOR = 0.05;    // similarity below which two articles are not linked
  var WANDER = 30;          // px/s² drift once the graph is broken
  var HUB_GAP = 78;         // px of clearance a community label needs
  var LABEL_ROOM = 66;      // px kept clear of the sheet for captions
  var HUB_SPREAD = 0.42;    // member spread (share of the stage) that makes a topic too diffuse
  var RESTITUTION = 0.94;   // wall bounce
  var MOUSE_PULL = 200;     // px/s² at the pointer, fading over MOUSE_RANGE
  var MOUSE_RANGE = 260;
  var PULSE_EVERY = 1.2;    // seconds between ambient pulses
  var FLASH = 0.7;          // seconds a node stays lit after a pulse lands

  function init() {
    var root = document.querySelector("[data-network]");
    if (!root) return;

    var stage = root.querySelector("[data-network-stage]");
    var svg = root.querySelector(".network-canvas");
    var sheet = root.querySelector("[data-network-sheet]");
    var chipRow = root.querySelector("[data-network-topics]");
    var statusEl = root.querySelector("[data-network-status]");
    var modeButtons = Array.prototype.slice.call(root.querySelectorAll("[data-network-mode]"));
    var zoomInButton = root.querySelector('[data-network-zoom="in"]');
    var zoomOutButton = root.querySelector('[data-network-zoom="out"]');
    if (!stage || !svg || !sheet || !chipRow) return;

    var reduce = matchMedia("(prefers-reduced-motion: reduce)");
    var fine = matchMedia("(hover: hover) and (pointer: fine)");

    var data = null;
    var topicById = {};
    var nodes = [];
    var hubs = [];
    var links = [];
    var pulses = [];
    var view = { k: 1, tx: 0, ty: 0 };
    var size = { w: 0, h: 0 };
    var pairRest = 200;   // resting length of a similarity link, set from the stage size
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
    function wrapTitle(text) {
      var words = String(text).split(/\s+/);
      var lines = [""];
      for (var i = 0; i < words.length; i++) {
        var candidate = lines[lines.length - 1] ? lines[lines.length - 1] + " " + words[i] : words[i];
        if (candidate.length > 24 && lines.length < 2) lines.push(words[i]);
        else lines[lines.length - 1] = candidate;
      }
      if (lines.length === 2 && lines[1].length > 24) lines[1] = lines[1].slice(0, 23) + "…";
      return lines;
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
      layers.hubs = layer("an-hubs");
      layers.nodes = layer("an-nodes");

      buildChips(payload.topics || []);
      resize();               // measure the stage before anything is placed
      buildNodes();
      buildField();
      bindEvents();
      settle();   // lay the field out before the first frame is due
      updateStatus();
      startMotion();
      root.setAttribute("data-network-ready", "1");
    }

    // --- chips ------------------------------------------------------------
    function buildChips(topics) {
      var topicChips = [];
      chipRow.innerHTML = "";
      var all = document.createElement("button");
      all.type = "button";
      all.className = "network-chip";
      all.setAttribute("data-network-topic", "");
      all.setAttribute("aria-pressed", "true");
      all.textContent = "All topics";
      chipRow.appendChild(all);

      topics.forEach(function (topic, index) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "network-chip";
        chip.setAttribute("data-network-topic", topic.id);
        chip.setAttribute("aria-pressed", "false");
        chip.title = topic.count + (topic.count === 1 ? " article" : " articles");
        chip.appendChild(document.createTextNode(topic.label));
        var count = document.createElement("span");
        count.className = "network-chip-count";
        count.textContent = topic.count;
        chip.appendChild(count);
        if (index >= CHIP_LIMIT) chip.hidden = true;
        topicChips.push(chip);
        chipRow.appendChild(chip);
      });

      var hidden = topics.length - CHIP_LIMIT;
      if (hidden > 0) {
        var more = document.createElement("button");
        more.type = "button";
        more.className = "network-chip";
        more.setAttribute("data-network-more", "");
        more.setAttribute("aria-expanded", "false");
        more.textContent = "+" + hidden + " more";
        chipRow.appendChild(more);
      }

      chipRow.addEventListener("click", function (event) {
        var more = event.target.closest("[data-network-more]");
        if (more) {
          var open = more.getAttribute("aria-expanded") === "true";
          more.setAttribute("aria-expanded", open ? "false" : "true");
          more.textContent = open ? "+" + hidden + " more" : "Show fewer";
          topicChips.forEach(function (chip, index) {
            if (index >= CHIP_LIMIT) chip.hidden = open;
          });
          return;
        }
        var chip = event.target.closest("[data-network-topic]");
        if (!chip) return;
        setTopic(chip.getAttribute("data-network-topic") || null);
      });
    }

    function setTopic(id) {
      state.topic = id || null;
      // A filter that excludes the article on show closes its summary.
      if (state.topic && state.open) {
        var shown = nodes.filter(function (node) { return node.article.slug === state.open; })[0];
        if (shown && shown.hubIds.indexOf(state.topic) < 0) closeSheet(false);
      }
      Array.prototype.forEach.call(chipRow.querySelectorAll("[data-network-topic]"), function (chip) {
        chip.setAttribute("aria-pressed", (chip.getAttribute("data-network-topic") || null) === state.topic ? "true" : "false");
      });
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
      buildField();
      updateStatus();
      if (!reduce.matches) startMotion(); else settle();
    }

    // --- field construction ----------------------------------------------
    function radiusFor(node) {
      var count = nodes.length;
      if (count < 2) return R_MAX;
      return R_MAX - (node.index / (count - 1)) * (R_MAX - R_MIN);
    }

    function buildNodes() {
      var count = data.articles.length;
      data.articles.forEach(function (article, index) {
        var node = {
          article: article,
          index: index,
          hubIds: article.topics.filter(function (id) { return topicById[id]; }),
          x: 0, y: 0, vx: 0, vy: 0, fx: 0, fy: 0,
          phase: index * 1.7,
          scale: 1, flash: -1,
          dim: false, near: false
        };
        node.r = radiusFor(node);

        var group = element("g", "an-node", layers.nodes);
        group.setAttribute("tabindex", "0");
        group.setAttribute("role", "button");
        group.setAttribute("aria-label", article.title + " — " + article.month +
          (article.topics.length ? " · " + article.topics.map(topicLabel).join(", ") : ""));

        node.el = group;
        node.halo = element("circle", "an-node-halo", group);
        node.halo.setAttribute("r", node.r + 7);
        node.outline = element("circle", "an-node-outline", group);
        node.outline.setAttribute("r", node.r + 2.5);

        if (article.image) {
          var image = element("image", "an-node-img", group);
          image.setAttribute("x", -node.r);
          image.setAttribute("y", -node.r);
          image.setAttribute("width", node.r * 2);
          image.setAttribute("height", node.r * 2);
          image.setAttribute("preserveAspectRatio", "xMidYMid slice");
          image.setAttribute("clip-path", "url(#an-clip)");
          image.setAttribute("href", article.image);
          node.image = image;
        } else {
          var fallback = element("circle", "an-node-fallback", group);
          fallback.setAttribute("r", node.r);
        }

        node.ring = element("circle", "an-node-ring", group);
        node.ring.setAttribute("r", node.r);

        var lines = wrapTitle(article.shortTitle || article.title);
        node.label = element("text", "an-node-label", group);
        node.label.setAttribute("y", node.r + 20);
        lines.forEach(function (line, lineIndex) {
          var tspan = element("tspan", null, node.label);
          tspan.setAttribute("x", 0);
          if (lineIndex) tspan.setAttribute("dy", "13");
          tspan.textContent = lineIndex === 0 && lines.length > 1 ? line + " " : line;
        });
        node.labelHalf = Math.max.apply(null, lines.map(function (line) { return line.length; })) * 3.5;
        node.date = element("text", "an-node-date", group);
        node.date.setAttribute("y", node.r + 34 + (lines.length - 1) * 13);
        node.date.textContent = article.month;

        var hit = element("circle", "an-node-hit", group);
        hit.setAttribute("r", node.r + 6);

        // Start somewhere near the middle so the first frame is not a pile-up.
        node.x = size.w / 2 + (index % 2 ? 1 : -1) * (60 + index * 18);
        node.y = size.h / 2 + ((index % 3) - 1) * (70 + index * 12);

        nodes.push(node);
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

    function buildField() {
      clear(layers.links);
      clear(layers.hubs);
      clear(layers.years);
      clear(layers.pulses);
      links = [];
      pulses = [];
      hubs = [];
      nodes.forEach(function (node) { node.pulls = []; node.pairs = []; });

      if (state.mode === "topic") {
        // The links are the shared topics and the labels annotate where each
        // community settled, so the shape of the corpus comes from the field
        // itself rather than from fixed anchors.
        (data.topics || []).forEach(function (topic) {
          if (topic.count < HUB_MIN) return;
          var hub = {
            id: topic.id, label: topic.label, count: topic.count,
            x: 0, y: 0, target: null, hidden: true,
            fixed: state.topic === topic.id,
            el: element("g", "an-hub is-hidden", layers.hubs)
          };
          hub.dot = element("circle", "an-hub-dot", hub.el);
          hub.dot.setAttribute("r", 4.5);
          hub.labelEl = element("text", "an-hub-label", hub.el);
          hub.labelEl.setAttribute("y", -14);
          hub.labelEl.textContent = topic.label;
          hub.halfWidth = Math.max(HUB_R, topic.label.length * 3.7);
          hubs.push(hub);
        });

        var candidates = [];
        for (var i = 0; i < nodes.length; i++) {
          for (var j = i + 1; j < nodes.length; j++) {
            var weight = similarity(nodes[i], nodes[j]);
            if (weight >= LINK_FLOOR) candidates.push({ a: nodes[i], b: nodes[j], weight: weight });
          }
        }
        // Keep each article's strongest relationships only: a web you can read,
        // not the complete graph that shares one broad topic with everything.
        var kept = [];
        nodes.forEach(function (node) {
          candidates
            .filter(function (pair) { return pair.a === node || pair.b === node; })
            .sort(function (x, y) { return y.weight - x.weight; })
            .slice(0, LINK_KEEP)
            .forEach(function (pair) { if (kept.indexOf(pair) < 0) kept.push(pair); });
        });
        kept.forEach(function (pair) {
          var link = {
            a: pair.a, b: pair.b, weight: pair.weight,
            articles: [pair.a, pair.b],
            el: element("line", "an-link", layers.links)
          };
          link.el.setAttribute("opacity", (0.16 + 0.66 * pair.weight).toFixed(3));
          links.push(link);
          pair.a.pairs.push(link);
          pair.b.pairs.push(link);
        });

        // A ring of slots keeps the field evenly spread; the similarity links
        // only bend that base shape, so no corner of the stage stays empty.
        var ringX = size.w * 0.33;
        var ringY = size.h * 0.31;
        nodes.forEach(function (node, index) {
          var angle = (index / nodes.length) * Math.PI * 2 - Math.PI / 2;
          var depth = index % 2 ? 0.64 : 1;
          node.slot = {
            x: size.w / 2 + Math.cos(angle) * ringX * depth,
            y: size.h / 2 + Math.sin(angle) * ringY * depth
          };
        });

        // Picking a topic gathers its members around that topic's hub.
        hubs.forEach(function (hub) {
          if (!hub.fixed) return;
          nodes.forEach(function (node) {
            if (node.hubIds.indexOf(hub.id) >= 0) node.pulls.push(hub);
          });
        });
      } else {
        var pad = Math.min(160, size.w * 0.14);
        var span = Math.max(1, size.w - pad * 2);
        var previous = null;
        nodes.forEach(function (node, index) {
          node.slot = {
            x: pad + (nodes.length < 2 ? span / 2 : (index / (nodes.length - 1)) * span),
            y: size.h * 0.5 + Math.sin(index * 1.9) * size.h * 0.15
          };
          if (previous) {
            links.push({ a: previous, b: node, chain: true, articles: [previous, node], el: linkElement("an-link is-chain") });
          }
          previous = node;
        });
        var seen = {};
        nodes.forEach(function (node) {
          var year = node.article.year;
          if (seen[year]) return;
          seen[year] = true;
          var rule = element("line", "an-year-rule", layers.years);
          rule.setAttribute("x1", node.slot.x);
          rule.setAttribute("x2", node.slot.x);
          rule.setAttribute("y1", 30);
          rule.setAttribute("y2", size.h - 26);
          var text = element("text", "an-year-label", layers.years);
          text.setAttribute("x", node.slot.x);
          text.setAttribute("y", 22);
          text.textContent = year;
        });
      }

      function linkElement(className) {
        var line = element("line", className, layers.links);
        line.setAttribute("x1", 0);
        line.setAttribute("y1", 0);
        line.setAttribute("x2", 0);
        line.setAttribute("y2", 0);
        return line;
      }

      refreshClasses();
      updateHubs(0, true);
      if (reduce.matches) settle();
    }

    // Community labels settle over the centroid of their members. A topic whose
    // members are scattered across the field is too diffuse to be worth a label,
    // and two labels never crowd each other: the bigger community wins the spot.
    function updateHubs(dt, instant) {
      var i;
      var placed = [];
      hubs.forEach(function (hub) { hub.target = null; });

      hubs.slice().sort(function (a, b) { return b.count - a.count; }).forEach(function (hub) {
        var members = 0;
        var cx = 0;
        var cy = 0;
        var spread = 0;
        for (i = 0; i < nodes.length; i++) {
          if (nodes[i].dim || nodes[i].hubIds.indexOf(hub.id) < 0) continue;
          members++;
          cx += nodes[i].x;
          cy += nodes[i].y;
        }
        if (!members) return;
        cx /= members;
        cy /= members;
        for (i = 0; i < nodes.length; i++) {
          if (nodes[i].dim || nodes[i].hubIds.indexOf(hub.id) < 0) continue;
          spread += Math.hypot(nodes[i].x - cx, nodes[i].y - cy);
        }
        spread /= members;

        var lit = hub.id === state.topic;
        if (!lit && spread > Math.min(size.w, size.h) * HUB_SPREAD) return;
        var target = {
          x: hub.fixed ? size.w / 2 : cx,
          y: (hub.fixed ? size.h / 2 : cy) - 30
        };
        if (state.open && !hub.fixed && target.x > size.w - sheetLimit.right - 40) return;
        for (i = 0; i < placed.length; i++) {
          if (Math.hypot(placed[i].x - target.x, placed[i].y - target.y) < HUB_GAP) return;
        }
        // Never drop a label onto an article's caption: lift it clear instead.
        var attempts = 0;
        while (attempts < 3 && overlapsCaption(target, hub)) {
          target.y -= HUB_GAP;
          attempts++;
        }
        if (overlapsCaption(target, hub)) return;
        hub.target = target;
        placed.push(target);
      });

      hubs.forEach(function (hub) {
        hub.hidden = hub.target === null;
        hub.el.classList.toggle("is-hidden", hub.hidden);
        if (hub.hidden) return;
        if (instant) {
          hub.x = hub.target.x;
          hub.y = hub.target.y;
        } else {
          hub.x = ease(hub.x, hub.target.x, dt, 2.4);
          hub.y = ease(hub.y, hub.target.y, dt, 2.4);
        }
        hub.el.setAttribute("transform", "translate(" + hub.x.toFixed(1) + " " + hub.y.toFixed(1) + ")");
      });
    }

    function refreshClasses() {
      nodes.forEach(function (node) {
        node.dim = !!state.topic && node.hubIds.indexOf(state.topic) < 0;
        node.el.classList.toggle("is-dim", node.dim);
        node.el.classList.toggle("is-active", state.open === node.article.slug);
        node.el.classList.toggle("is-near", state.hover === node);
      });
      hubs.forEach(function (hub) {
        hub.el.classList.toggle("is-lit", !!state.topic && hub.id === state.topic);
        hub.el.classList.toggle("is-dim", !!state.topic && hub.id !== state.topic);
      });
      var spotlight = state.open || (state.hover ? state.hover.article.slug : null);
      links.forEach(function (link) {
        var lit = link.articles.some(function (article) { return article.article.slug === spotlight; });
        var dim = spotlight ? !lit : link.articles.some(function (article) { return article.dim; });
        link.el.classList.toggle("is-dim", dim);
        link.el.classList.toggle("is-lit", !!spotlight && lit);
      });
    }

    // --- physics ----------------------------------------------------------
    // Bounds are asymmetric: a caption hangs below its circle and can be wider
    // than it, so both are kept clear of the stage edges and of the sheet.
    function boundsFor(node) {
      var inset = Math.max(node.r + 12, Math.min(96, node.labelHalf + 12));
      var box = {
        left: inset,
        right: size.w - inset,
        top: node.r + 10,
        bottom: size.h - node.r - 56
      };
      if (state.open) {
        box.right = Math.max(box.left, size.w - sheetLimit.right - inset);
        box.bottom = Math.max(box.top, size.h - sheetLimit.bottom - inset);
      }
      return box;
    }

    // A topic label is a box, not a point: test it against each caption box.
    function overlapsCaption(target, hub) {
      var left = target.x - hub.halfWidth;
      var right = target.x + hub.halfWidth;
      var top = target.y - 30;
      var bottom = target.y + 12;
      for (var i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        var half = Math.max(node.r, Math.min(96, node.labelHalf)) + 8;
        if (right > node.x - half && left < node.x + half &&
            bottom > node.y - node.r - 10 && top < node.y + node.r + 56) {
          return true;
        }
      }
      return false;
    }

    function focusPoint() {
      var right = state.open ? sheetLimit.right : size.w;
      return { x: clamp(right / 2, 120, size.w - 120), y: size.h / 2 };
    }

    function step(dt) {
      var i, j, a, b, dx, dy, d2, d, force, ux, uy;
      // The field is laid out in a square metric and stretched onto the stage, so
      // a wide panel gets a wide network instead of a circle in the middle.
      // Distances are therefore measured in "field units" while positions stay
      // in stage pixels.
      var base = Math.min(size.w, size.h) || 1;
      var ax = size.w / base;
      var ay = size.h / base;

      for (i = 0; i < nodes.length; i++) { nodes[i].fx = 0; nodes[i].fy = 0; }

      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        for (j = i + 1; j < nodes.length; j++) {
          b = nodes[j];
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

          var rest = a.r + b.r + 26;   // includes room for the labels below
          if (d < rest) {
            force = (rest - d) * 7;
            a.fx -= ux * force / ax; a.fy -= uy * force / ay;
            b.fx += ux * force / ax; b.fy += uy * force / ay;
          }
        }
      }

      // Similarity springs: the more topics two articles share, the closer they sit.
      if (state.mode === "topic" && !state.broken) {
        for (i = 0; i < links.length; i++) {
          var pair = links[i];
          if (pair.a.dim || pair.b.dim) continue;
          dx = (pair.b.x - pair.a.x) / ax;
          dy = (pair.b.y - pair.a.y) / ay;
          d = Math.hypot(dx, dy) || 1;
          force = (d - pairRest) * SPRING_PAIR * pair.weight;
          ux = dx / d;
          uy = dy / d;
          pair.a.fx += ux * force / ax;
          pair.a.fy += uy * force / ay;
          pair.b.fx -= ux * force / ax;
          pair.b.fy -= uy * force / ay;
        }
      }

      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];

        if (state.broken) {
          if (state.open === a.article.slug) {
            var focus = focusPoint();
            a.fx += (focus.x - a.x) * SPRING_FOCUS;
            a.fy += (focus.y - a.y) * SPRING_FOCUS;
          } else if (!reduce.matches) {
            a.fx += Math.sin(clock * 0.8 + a.phase) * WANDER;
            a.fy += Math.cos(clock * 0.66 + a.phase * 1.3) * WANDER;
          }
        } else if (state.mode === "date") {
          if (a.slot) {
            a.fx += (a.slot.x - a.x) * SPRING_DATE;
            a.fy += (a.slot.y - a.y) * SPRING_DATE;
          }
        } else if (a.dim) {
          // Filtered out: drift to the periphery instead of piling up.
          dx = a.x - size.w / 2;
          dy = a.y - size.h / 2;
          d = Math.hypot(dx, dy) || 1;
          a.fx += (dx / d) * 26;
          a.fy += (dy / d) * 26;
        } else if (a.pulls.length) {
          // A picked topic pulls its members into orbit around its hub.
          var weight = 1 / a.pulls.length;
          for (var k = 0; k < a.pulls.length; k++) {
            var hub = a.pulls[k];
            a.fx += (hub.x - a.x) * SPRING_HUB * weight;
            a.fy += (hub.y - a.y) * SPRING_HUB * weight;
            dx = (a.x - hub.x) / ax;
            dy = (a.y - hub.y) / ay;
            d = Math.hypot(dx, dy) || 1;
            var orbit = HUB_R + a.r + 40;
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
          a.fy += (size.h / 2 - a.y) * CENTER_PULL;
        }

        if (pointer.inside && !state.broken) {
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
      var margin;
      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
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
        if (state.broken) {
          if (a.x < left) { a.x = left; a.vx = Math.abs(a.vx) * RESTITUTION + 14; }
          else if (a.x > right) { a.x = right; a.vx = -Math.abs(a.vx) * RESTITUTION - 14; }
          if (a.y < top) { a.y = top; a.vy = Math.abs(a.vy) * RESTITUTION + 14; }
          else if (a.y > bottom) { a.y = bottom; a.vy = -Math.abs(a.vy) * RESTITUTION - 14; }
        } else {
          a.x = clamp(a.x, left, Math.max(left, right));
          a.y = clamp(a.y, top, Math.max(top, bottom));
        }

        // Hover easing happens here so it shares the frame clock.
        a.scale = ease(a.scale, a.near || state.open === a.article.slug ? 1.09 : 1, dt, 10);
      }
    }

    function settle(steps) {
      var count = steps || 220;
      for (var i = 0; i < count; i++) step(1 / 60);
      for (var pass = 0; pass < 8; pass++) updateHubs(1 / 30, false);
      updateHubs(0, true);   // land the labels where they belong
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
        if (target && target.article) target.flash = clock + FLASH;
      });
    }

    // --- render -----------------------------------------------------------
    function draw() {
      var i;
      for (i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        node.el.setAttribute("transform",
          "translate(" + node.x.toFixed(2) + " " + node.y.toFixed(2) + ") scale(" + node.scale.toFixed(3) + ")");
        var lit = node.flash > clock;
        node.outline.setAttribute("opacity", lit ? "1" : "");
        node.outline.classList.toggle("is-lit", lit);
      }
      for (i = 0; i < links.length; i++) {
        var link = links[i];
        link.el.setAttribute("x1", link.a.x.toFixed(2));
        link.el.setAttribute("y1", link.a.y.toFixed(2));
        link.el.setAttribute("x2", link.b.x.toFixed(2));
        link.el.setAttribute("y2", link.b.y.toFixed(2));
      }
    }

    // --- view (zoom / pan) ------------------------------------------------
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
      var rect = stage.getBoundingClientRect();
      if (rect.width < 40 || rect.height < 40) return;
      var changed = Math.abs(size.w - rect.width) > 2 || Math.abs(size.h - rect.height) > 2;
      size.w = rect.width;
      size.h = rect.height;
      pairRest = clamp(Math.min(size.w, size.h) * PAIR_SPACING, PAIR_MIN, PAIR_MAX);
      svg.setAttribute("viewBox", "0 0 " + size.w.toFixed(1) + " " + size.h.toFixed(1));
      applyView();
      if (!changed || !nodes.length) return;
      if (state.mode === "date") {
        var pad = Math.min(160, size.w * 0.14);
        var span = Math.max(1, size.w - pad * 2);
        nodes.forEach(function (node, index) {
          node.slot = {
            x: pad + (nodes.length < 2 ? span / 2 : (index / (nodes.length - 1)) * span),
            y: size.h * 0.5 + Math.sin(index * 1.9) * size.h * 0.15
          };
        });
      } else {
        buildField();
      }
    }

    function measureSheet() {
      var rect = sheet.getBoundingClientRect();
      var stacked = window.innerWidth < 768;
      sheetLimit.right = stacked ? 0 : rect.width + 22;
      sheetLimit.bottom = stacked ? rect.height + 12 : 0;
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
      updateHubs(dt, false);
      draw();
      updatePulses(dt);

      pulseTimer += dt;
      if (pulseTimer > PULSE_EVERY && links.length) {
        pulseTimer = 0;
        var candidates = links.filter(function (link) { return !link.el.classList.contains("is-dim"); });
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
      renderSheet(article);
      measureSheet();
      nodes.forEach(function (other) {
        var box = boundsFor(other);
        other.x = clamp(other.x, box.left, box.right);
        other.y = clamp(other.y, box.top, box.bottom);
      });
      draw();
      sheet.classList.add("is-open");
      sheet.setAttribute("aria-hidden", "false");
      node.vx += (node.x - size.w / 2) * 0.4;
      node.vy += (node.y - size.h / 2) * 0.4;
      nodes.forEach(function (other) {
        if (other === node) return;
        var dx = other.x - node.x;
        var dy = other.y - node.y;
        var d = Math.hypot(dx, dy) || 1;
        var kick = Math.max(0, 1 - d / 420) * 520 + 90;
        other.vx += (dx / d) * kick;
        other.vy += (dy / d) * kick - 60;
      });
      refreshClasses();
      updateStatus();
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
      sheet.classList.remove("is-open");
      sheet.setAttribute("aria-hidden", "true");
      refreshClasses();
      updateStatus();
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

      var source = article.imageSource || article.image;
      if (source) {
        var media = document.createElement("div");
        media.className = "network-sheet-media";
        var img = document.createElement("img");
        img.src = source;
        img.alt = article.imageAlt || "";
        media.appendChild(img);
        sheet.appendChild(media);
      }

      var body = document.createElement("div");
      body.className = "network-sheet-body";

      var meta = document.createElement("p");
      meta.className = "network-sheet-meta";
      meta.textContent = article.month;
      body.appendChild(meta);

      var title = document.createElement("h3");
      title.className = "network-sheet-title";
      title.id = "network-sheet-title";
      title.tabIndex = -1;
      title.textContent = article.title;
      body.appendChild(title);

      var chips = document.createElement("div");
      chips.className = "network-sheet-topics";
      article.topics.forEach(function (id) {
        var chip = document.createElement("span");
        chip.className = "topic-chip";
        chip.textContent = topicLabel(id);
        chips.appendChild(chip);
      });
      body.appendChild(chips);

      if (article.description) {
        var summary = document.createElement("p");
        summary.className = "network-sheet-summary";
        summary.textContent = article.description;
        body.appendChild(summary);
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
      body.appendChild(actions);
      sheet.appendChild(body);
    }

    // --- status -----------------------------------------------------------
    function updateStatus() {
      var total = nodes.length;
      var shown = nodes.filter(function (node) { return !node.dim; }).length;
      var mode = state.mode === "topic" ? "grouped by topic" : "along the timeline";
      statusEl.textContent = (state.topic ? shown + " of " + total + " articles · " + topicLabel(state.topic) + " · " : total + " articles · ") + mode;
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
          var rect = stage.getBoundingClientRect();
          zoomAt(zoom.getAttribute("data-network-zoom") === "in" ? 1.35 : 1 / 1.35, rect.width / 2, rect.height / 2);
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
        node.el.addEventListener("focus", function () { state.hover = node; refreshClasses(); });
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

      svg.addEventListener("wheel", function (event) {
        event.preventDefault();
        var rect = stage.getBoundingClientRect();
        var step = event.deltaMode === 1 ? 0.04 : 0.0016;
        zoomAt(Math.exp(-event.deltaY * step), event.clientX - rect.left, event.clientY - rect.top);
      }, { passive: false });

      svg.addEventListener("pointerdown", function (event) {
        touches[event.pointerId] = { x: event.clientX, y: event.clientY };
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
        } else if (!event.target.closest(".an-node")) {
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
          if (!pointer.moved && !event.target.closest(".an-node")) {
            if (state.open) closeSheet();
            else {
              var pulseTarget = nodes.slice().sort(function (a, b) {
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
        if (event.key === "Escape" && state.open) {
          event.preventDefault();
          closeSheet();
          return;
        }
        // Keyboard navigation reads the focused element: SVG groups do not
        // reliably fire focus events, but focus() always moves activeElement.
        var index = -1;
        for (var i = 0; i < nodes.length; i++) {
          if (nodes[i].el === document.activeElement) { index = i; break; }
        }
        if (index < 0 && state.hover) index = nodes.indexOf(state.hover);
        if (index < 0 || event.key.indexOf("Arrow") !== 0) return;
        var node = nodes[index];
        var best = null;
        var bestScore = Infinity;
        nodes.forEach(function (other) {
          if (other === node) return;
          var dx = other.x - node.x;
          var dy = other.y - node.y;
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

      window.addEventListener("resize", function () {
        resize();
        measureSheet();
        if (reduce.matches) settle();
      }, { passive: true });

      document.addEventListener("visibilitychange", function () {
        if (document.hidden) stopMotion();
        else startMotion();
      });

      if (window.ResizeObserver) {
        new ResizeObserver(function () {
          resize();
          if (reduce.matches) settle();
        }).observe(stage);
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
