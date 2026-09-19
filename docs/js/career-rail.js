// Career rail — fixed-geometry DAG with native dialog for role details.
//
// Nodes are native buttons with .rail-dot markers measured from the DOM.
// Edges (SVG paths) are drawn relative to .career-track only — geometry
// never depends on panel height or selection.  Clicking a node opens a
// native <dialog> by moving the matching article into it; closing restores
// the article to .career-panels in source order.  No role is shown until
// the user explicitly activates one.
(function () {
  "use strict";

  function init() {
    var root = document.querySelector("[data-career]");
    if (!root) return;
    var track = root.querySelector(".career-track");
    var viewport = root.querySelector(".career-viewport");
    var svg = root.querySelector(".career-wires");
    var panelsWrap = root.querySelector(".career-panels");
    var scrollHint = root.querySelector(".career-scroll-hint");
    if (!track || !svg || !panelsWrap) return;

    var dialog = document.getElementById("career-dialog");
    var dialogBody = dialog ? dialog.querySelector(".career-dialog-body") : null;
    var dialogCount = dialog ? dialog.querySelector(".career-dialog-count") : null;
    var btnClose = dialog ? dialog.querySelector("[data-career-close]") : null;
    var btnPrev = dialog ? dialog.querySelector("[data-career-prev]") : null;
    var btnNext = dialog ? dialog.querySelector("[data-career-next]") : null;
    if (!dialogBody || !btnClose || !btnPrev || !btnNext ||
        typeof dialog.showModal !== "function") return;

    var nodes = Array.prototype.slice.call(track.querySelectorAll(".rail-node"));
    if (!nodes.length) return;

    // ── Panel map (original source order for DOM restoration) ──
    var panelMap = {};
    var panelOrder = [];
    Array.prototype.forEach.call(panelsWrap.querySelectorAll(".career-panel"),
      function (p) {
        var id = p.dataset.node;
        if (id) { panelMap[id] = p; panelOrder.push(id); }
      });

    // ── ARIA setup on buttons ──
    nodes.forEach(function (btn) {
      btn.setAttribute("aria-haspopup", "dialog");
      btn.setAttribute("aria-controls", "career-dialog");
      btn.setAttribute("aria-expanded", "false");
    });

    // ── Edge list from data-edges ──
    var edges = (root.dataset.edges || "").trim().split(/\s+/).filter(Boolean)
      .map(function (pair) {
        var p = pair.split(">");
        return { source: p[0], target: p[1] };
      });

    // ── SVG scaffolding ──
    var NS = "http://www.w3.org/2000/svg";

    var defs = document.createElementNS(NS, "defs");
    defs.innerHTML =
      '<marker id="rail-arrow" viewBox="0 0 8 8" refX="7" refY="4" ' +
      'markerWidth="6" markerHeight="6" orient="auto" markerUnits="userSpaceOnUse">' +
      '<path d="M1 1 L7 4 L1 7" fill="none" stroke="currentColor" stroke-width="1.2"/>' +
      '</marker>';
    svg.appendChild(defs);

    edges.forEach(function (e) {
      e.el = document.createElementNS(NS, "path");
      e.el.classList.add("rail-wire");
      e.el.setAttribute("marker-end", "url(#rail-arrow)");
      svg.appendChild(e.el);
    });

    // ── State ──
    var activeId = null;       // node id whose article is in the dialog
    var triggerNode = null;    // button that first opened the dialog (focus return)
    var measured = false;
    var nodeCenters = {};

    var nodeById = {};
    nodes.forEach(function (btn) { nodeById[btn.dataset.node] = btn; });

    // ── Geometry: track-only measurement ──
    function measure() {
      var tRect = track.getBoundingClientRect();
      if (tRect.width < 8 || tRect.height < 8) return false;

      svg.setAttribute("viewBox",
        "0 0 " + tRect.width.toFixed(1) + " " + tRect.height.toFixed(1));

      nodeCenters = {};
      nodes.forEach(function (btn) {
        var dot = btn.querySelector(".rail-dot");
        if (!dot) return;
        var dRect = dot.getBoundingClientRect();
        nodeCenters[btn.dataset.node] = {
          x: (dRect.left + dRect.width / 2) - tRect.left,
          y: (dRect.top + dRect.height / 2) - tRect.top
        };
      });
      measured = true;
      return true;
    }

    function drawEdges() {
      if (!measured) return;
      edges.forEach(function (e) {
        var a = nodeCenters[e.source];
        var b = nodeCenters[e.target];
        if (!a || !b) return;
        var startX = a.x + 14;
        var endX = b.x - 14;
        var middleX = (startX + endX) / 2;
        e.el.setAttribute("d",
          "M" + startX.toFixed(1) + " " + a.y.toFixed(1) +
          " C" + middleX.toFixed(1) + " " + a.y.toFixed(1) +
          " " + middleX.toFixed(1) + " " + b.y.toFixed(1) +
          " " + endX.toFixed(1) + " " + b.y.toFixed(1));
      });
    }

    // ── Highlight ancestor path (BFS backward, visited set) ──
    function highlightPath(nodeId) {
      // Clear
      nodes.forEach(function (btn) { btn.classList.remove("is-lit"); });
      edges.forEach(function (e) { e.el.classList.remove("is-lit"); });
      if (!nodeId || !measured) return;

      // Walk backward through edges to find ancestors
      var visited = {};
      var queue = [nodeId];
      visited[nodeId] = true;
      while (queue.length) {
        var cur = queue.shift();
        for (var i = 0; i < edges.length; i++) {
          if (edges[i].target === cur && !visited[edges[i].source]) {
            visited[edges[i].source] = true;
            queue.push(edges[i].source);
          }
        }
      }

      nodes.forEach(function (btn) {
        if (visited[btn.dataset.node]) btn.classList.add("is-lit");
      });
      edges.forEach(function (e) {
        if (visited[e.source] && visited[e.target]) e.el.classList.add("is-lit");
      });
    }

    // ── Scroll-hint state ──
    function updateScrollHint() {
      if (!viewport || !scrollHint) return;
      var canScroll = viewport.scrollWidth > viewport.clientWidth + 1;
      if (canScroll) {
        root.dataset.careerScrollable = "1";
      } else {
        delete root.dataset.careerScrollable;
      }
    }

    // ── Dialog: article move lifecycle ──
    // Restore the active article to .career-panels in original DOM order.
    function restorePanel() {
      if (!activeId || !panelMap[activeId]) return;
      var panel = panelMap[activeId];
      if (panel.parentNode !== dialogBody) return;

      // Find the first sibling in the original order that is still in panelsWrap
      // and comes after this panel's position; insert before it.
      var idx = panelOrder.indexOf(activeId);
      var inserted = false;
      for (var i = idx + 1; i < panelOrder.length; i++) {
        var ref = panelMap[panelOrder[i]];
        if (ref && ref.parentNode === panelsWrap) {
          panelsWrap.insertBefore(panel, ref);
          inserted = true;
          break;
        }
      }
      if (!inserted) panelsWrap.appendChild(panel);
    }

    function openDialog(nodeId, trigger) {
      if (!dialog || !dialogBody || !panelMap[nodeId]) return;

      // If an article is already in the dialog, put it back first
      restorePanel();

      activeId = nodeId;
      // Only set trigger on initial open; prev/next preserve the original
      if (trigger) triggerNode = trigger;

      // Move article into dialog body (single copy, no duplication)
      var panel = panelMap[nodeId];
      dialogBody.appendChild(panel);
      dialogBody.scrollTop = 0;

      // Accessible name: company + title
      var titleId = "role-title-" + nodeId;
      var companyId = "role-company-" + nodeId;
      dialog.setAttribute("aria-labelledby", companyId + " " + titleId);

      // Counter ("2 of 6")
      var idx = panelOrder.indexOf(nodeId);
      var _t = window.siteI18n ? window.siteI18n.t.bind(window.siteI18n) : function (k, v) { return k; };
      if (dialogCount) {
        dialogCount.textContent = _t("career.counter", { current: idx + 1, total: panelOrder.length });
      }

      // Prev/next: stop at endpoints, no wrap
      if (btnPrev) btnPrev.disabled = (idx === 0);
      if (btnNext) btnNext.disabled = (idx === panelOrder.length - 1);

      // aria-expanded: only the open node is true
      nodes.forEach(function (btn) {
        btn.setAttribute("aria-expanded",
          btn.dataset.node === nodeId ? "true" : "false");
      });

      // Highlight ancestor path
      highlightPath(nodeId);

      // Show dialog (only on first open; navigations keep it open)
      if (!dialog.open) dialog.showModal();

      // Focus title, prevent scroll
      var title = document.getElementById(titleId);
      if (title) title.focus({ preventScroll: true });
    }

    // ── Dialog close: single cleanup point ──
    // The native 'close' event fires after dialog.close() (whether triggered
    // by Escape/cancel, close button, or pointer-dismiss).
    if (dialog) {
      dialog.addEventListener("close", function () {
        // A queued close event may arrive after a new activation.
        if (dialog.open) return;
        var returnTo = triggerNode;
        restorePanel();
        activeId = null;
        triggerNode = null;
        nodes.forEach(function (btn) {
          btn.setAttribute("aria-expanded", "false");
        });
        highlightPath(null);
        if (returnTo) returnTo.focus({ preventScroll: true });
      });
    }

    // ── Click: open dialog ──
    nodes.forEach(function (btn) {
      btn.addEventListener("click", function () {
        openDialog(btn.dataset.node, btn);
      });
    });

    // ── Keyboard on track: arrows/Home/End focus only, never open ──
    track.addEventListener("keydown", function (e) {
      var idx = nodes.indexOf(document.activeElement);
      if (idx < 0) return;
      var next = idx;
      switch (e.key) {
        case "ArrowRight": case "ArrowDown":
          next = (idx + 1) % nodes.length; break;
        case "ArrowLeft": case "ArrowUp":
          next = (idx - 1 + nodes.length) % nodes.length; break;
        case "Home": next = 0; break;
        case "End": next = nodes.length - 1; break;
        default: return;
      }
      e.preventDefault();
      nodes[next].focus();
    });

    // ── Hover / focus highlight (not while dialog is open) ──
    nodes.forEach(function (btn) {
      btn.addEventListener("mouseenter", function () {
        if (!dialog || !dialog.open) highlightPath(btn.dataset.node);
      });
      btn.addEventListener("mouseleave", function () {
        if (!dialog || !dialog.open) highlightPath(null);
      });
      btn.addEventListener("focus", function () {
        if (!dialog || !dialog.open) highlightPath(btn.dataset.node);
      });
      btn.addEventListener("blur", function () {
        if (!dialog || !dialog.open) highlightPath(null);
      });
    });

    // ── Dialog controls ──
    if (btnClose) {
      btnClose.addEventListener("click", function () { dialog.close(); });
    }

    if (btnPrev) {
      btnPrev.addEventListener("click", function () {
        if (!activeId) return;
        var idx = panelOrder.indexOf(activeId);
        if (idx > 0) openDialog(panelOrder[idx - 1]);
      });
    }
    if (btnNext) {
      btnNext.addEventListener("click", function () {
        if (!activeId) return;
        var idx = panelOrder.indexOf(activeId);
        if (idx < panelOrder.length - 1) openDialog(panelOrder[idx + 1]);
      });
    }

    // ── Pointer-safe dismiss ──
    // Only a primary-pointer gesture beginning and ending outside the dialog
    // dismisses it; a drag from the text or a click on its border does not.
    if (dialog) {
      var outsidePointer = null;
      function outsideDialog(event) {
        var box = dialog.getBoundingClientRect();
        return event.clientX < box.left || event.clientX > box.right ||
          event.clientY < box.top || event.clientY > box.bottom;
      }
      dialog.addEventListener("pointerdown", function (event) {
        outsidePointer = event.isPrimary && event.button === 0 && outsideDialog(event)
          ? event.pointerId : null;
      });
      dialog.addEventListener("pointerup", function (event) {
        if (outsidePointer === event.pointerId && outsideDialog(event)) dialog.close();
        outsidePointer = null;
      });
      dialog.addEventListener("pointercancel", function () { outsidePointer = null; });
    }

    // ── Geometry redraw (independent from selection) ──
    function redraw() {
      if (measure()) drawEdges();
      updateScrollHint();
    }

    if (window.ResizeObserver) {
      var ro = new ResizeObserver(redraw);
      ro.observe(track);
      if (viewport) ro.observe(viewport);
    }
    window.addEventListener("resize", redraw, { passive: true });
    window.addEventListener("load", redraw, { once: true });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(redraw);
    }
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) redraw();
    });

    // ── Print: restore article so all six appear in source order ──
    var printActiveId = null;
    window.addEventListener("beforeprint", function () {
      if (dialog && dialog.open && activeId) {
        printActiveId = activeId;
        restorePanel();
      }
    });
    window.addEventListener("afterprint", function () {
      if (dialog.open && activeId === printActiveId) {
        dialogBody.appendChild(panelMap[printActiveId]);
        panelMap[printActiveId].querySelector(".panel-title").focus({ preventScroll: true });
      }
      printActiveId = null;
    });

    // ── Initial render ──
    redraw();
    root.dataset.careerReady = "1";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
