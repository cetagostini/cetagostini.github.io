// Career rail — the roles laid out as a small DAG.
//
// The nodes are real HTML buttons (a tablist); the graph itself (dots, wires,
// arrowheads) is drawn in SVG from measured DOM positions, so the same rail
// works as a horizontal timeline on wide screens and as a vertical one below
// 992px. Without JS every panel stays stacked in normal flow — the script only
// takes over once it is ready, hence the [data-career-ready] gate in CSS.
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var round = function (value) { return value.toFixed(1); };

  function init() {
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    var root = document.querySelector("[data-career]");
    if (!root) return;
    var svg = root.querySelector(".career-wires");
    var track = root.querySelector(".career-track");
    var panelsWrap = root.querySelector(".career-panels");
    if (!svg || !track || !panelsWrap) return;

    var tabs = Array.prototype.slice.call(track.querySelectorAll(".rail-node"));
    if (!tabs.length) return;

    function create(tag, className, parent) {
      var el = document.createElementNS(NS, tag);
      if (className) el.setAttribute("class", className);
      parent.appendChild(el);
      return el;
    }

    var defs = create("defs", null, svg);
    defs.innerHTML =
      '<marker id="rail-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" ' +
      'orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L7 4 L1 7" fill="none" ' +
      'stroke="var(--brown)" stroke-width="1.2"/></marker>' +
      '<marker id="rail-arrow-lit" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" ' +
      'orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L7 4 L1 7" fill="none" ' +
      'stroke="var(--green-strong)" stroke-width="1.4"/></marker>';

    var wires = create("g", "rail-wires", svg);
    var dotsLayer = create("g", "rail-dots", svg);

    var panels = new Map();
    Array.prototype.forEach.call(panelsWrap.querySelectorAll(".career-panel"), function (panel) {
      panels.set(panel.dataset.node, panel);
    });

    var nodes = new Map();
    tabs.forEach(function (tab) {
      var id = tab.dataset.node;
      var halo = create("circle", "rail-halo", dotsLayer);
      var dot = create("circle", "rail-dot", dotsLayer);
      var hit = create("circle", "rail-hit", dotsLayer);
      halo.setAttribute("r", "12");
      dot.setAttribute("r", "5");
      hit.setAttribute("r", "22");
      var node = {
        id: id, tab: tab, panel: panels.get(id), dot: dot, halo: halo, hit: hit,
        anchor: { x: 0, y: 0 }, isNow: tab.dataset.now === "1"
      };
      // The dot is a real target, not just a marker: clicking it picks the role.
      hit.addEventListener("click", function () {
        interactive = true;
        select(node, true);
      });
      hit.addEventListener("pointerenter", function () {
        node.dot.classList.add("is-hover");
        tab.classList.add("is-hover");
      });
      hit.addEventListener("pointerleave", function () {
        node.dot.classList.remove("is-hover");
        tab.classList.remove("is-hover");
      });
      nodes.set(id, node);
    });

    var edges = (root.dataset.edges || "").trim().split(/\s+/).filter(Boolean).map(function (pair) {
      var ends = pair.split(">");
      return { source: ends[0], target: ends[1] };
    }).filter(function (edge) {
      return nodes.has(edge.source) && nodes.has(edge.target);
    });
    edges.forEach(function (edge) { edge.el = create("path", "rail-wire", wires); });

    var vertical = false;
    var active = null;
    var interactive = false;

    var order = tabs.map(function (tab) { return tab.dataset.node; });

    function curve(edge) {
      var a = nodes.get(edge.source).anchor;
      var b = nodes.get(edge.target).anchor;
      // In the stacked layout an edge that skips a node (the 2024 fork) bows
      // sideways so it does not run straight through the node in between.
      var skip = vertical && Math.abs(order.indexOf(edge.target) - order.indexOf(edge.source)) > 1;
      if (vertical) {
        var ky = (b.y - a.y) * 0.5;
        var bow = skip ? 16 : 0;
        return "M" + round(a.x) + " " + round(a.y) +
          " C" + round(a.x + bow) + " " + round(a.y + ky) +
          " " + round(b.x + bow) + " " + round(b.y - ky) +
          " " + round(b.x) + " " + round(b.y);
      }
      var kx = (b.x - a.x) * 0.5;
      return "M" + round(a.x) + " " + round(a.y) +
        " C" + round(a.x + kx) + " " + round(a.y) +
        " " + round(b.x - kx) + " " + round(b.y) +
        " " + round(b.x) + " " + round(b.y);
    }

    function measure() {
      var box = root.getBoundingClientRect();
      var width = Math.max(1, box.width);
      var height = Math.max(1, box.height);
      // A degenerate box means the layout is not ready (hidden tab, first paint
      // before fonts): keep the previous drawing and retry shortly.
      if (width < 8 || height < 8) return false;
      svg.setAttribute("viewBox", "0 0 " + round(width) + " " + round(height));

      var trackBox = track.getBoundingClientRect();
      vertical = trackBox.height > trackBox.width * 0.8;

      nodes.forEach(function (node) {
        var rect = node.tab.getBoundingClientRect();
        var left = rect.left - box.left;
        var top = rect.top - box.top;
        // Wide layout: the dot sits on the rail axis, above the label.
        // Stacked layout: the dot sits in the left gutter, beside the label.
        node.anchor = vertical
          ? { x: left - 15, y: top + rect.height / 2 }
          : { x: left + 6, y: top - 15 };
        node.dot.setAttribute("cx", round(node.anchor.x));
        node.dot.setAttribute("cy", round(node.anchor.y));
        node.halo.setAttribute("cx", round(node.anchor.x));
        node.halo.setAttribute("cy", round(node.anchor.y));
        node.hit.setAttribute("cx", round(node.anchor.x));
        node.hit.setAttribute("cy", round(node.anchor.y));
      });

      edges.forEach(function (edge) {
        edge.el.setAttribute("d", curve(edge));
      });
      return true;
    }

    function place(node) {
      var panel = node.panel;
      if (!panel) return;
      var box = root.getBoundingClientRect();
      var panelWidth = panel.offsetWidth;
      var x = vertical ? 0 : Math.max(0, Math.min(node.anchor.x - 10, box.width - panelWidth));
      panelsWrap.style.setProperty("--panel-x", round(x) + "px");
      panelsWrap.style.setProperty("--panels-height", panel.offsetHeight + "px");
    }

    function select(node, focus) {
      if (!node) return;
      active = node;

      // The path leading to the selected role stays lit; the rest recedes.
      var lit = new Set([node.id]);
      var grew = true;
      while (grew) {
        grew = false;
        edges.forEach(function (edge) {
          if (lit.has(edge.target) && !lit.has(edge.source)) { lit.add(edge.source); grew = true; }
        });
      }

      nodes.forEach(function (candidate) {
        var isActive = candidate === node;
        candidate.tab.setAttribute("aria-selected", isActive ? "true" : "false");
        candidate.tab.tabIndex = isActive ? 0 : -1;
        candidate.dot.classList.toggle("is-lit", isActive);
        candidate.dot.classList.toggle("is-now", candidate.isNow);
        candidate.halo.classList.toggle("is-now", candidate.isNow);
        if (candidate.panel) candidate.panel.classList.toggle("is-active", isActive);
      });

      edges.forEach(function (edge) {
        var isLit = lit.has(edge.target);
        edge.el.classList.toggle("is-lit", isLit);
        edge.el.setAttribute("marker-end", isLit ? "url(#rail-arrow-lit)" : "url(#rail-arrow)");
      });

      if (focus) node.tab.focus();
      place(node);
      // Stacked layout: the panel sits below the whole rail, so bring it into
      // view when the reader picks a role themselves. Never on first paint.
      if (interactive && vertical && node.panel) {
        node.panel.scrollIntoView({ block: "nearest", behavior: reduce.matches ? "auto" : "smooth" });
      }
    }

    function draw() {
      if (!measure()) { resync(200); return; }
      select(active || current, false);
    }

    var current = nodes.get(tabs[tabs.length - 1].dataset.node);
    nodes.forEach(function (node) { if (node.isNow) current = node; });

    // Hand the layout over to the script only now: with JS the panels become a
    // single floating sheet, without it they remain stacked and readable.
    root.dataset.careerReady = "1";
    draw();

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        interactive = true;
        select(nodes.get(tab.dataset.node), false);
      });
    });

    track.addEventListener("keydown", function (event) {
      var index = tabs.indexOf(document.activeElement);
      if (index < 0) return;
      var next = index;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault();
      interactive = true;
      select(nodes.get(tabs[next].dataset.node), true);
    });

    // Geometry is measured from the DOM, so every reflow has to trigger a
    // redraw. requestAnimationFrame is throttled in background tabs — a queued
    // frame there may never run, so resync() clears the flag and redraws now.
    var queued = 0;
    var retry = 0;
    function schedule() {
      if (queued) return;
      queued = requestAnimationFrame(function () { queued = 0; draw(); });
    }
    function resync(delay) {
      clearTimeout(retry);
      retry = setTimeout(function () {
        if (queued) { cancelAnimationFrame(queued); queued = 0; }
        draw();
      }, delay || 0);
    }

    if (window.ResizeObserver) new ResizeObserver(schedule).observe(track);
    window.addEventListener("resize", schedule, { passive: true });
    window.addEventListener("load", schedule, { once: true });
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) resync();
    });
    // Late layout (fonts, images, the photo above the rail) still needs a pass.
    resync(400);

    if (document.fonts && document.fonts.ready) document.fonts.ready.then(schedule);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
