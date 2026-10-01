// reading-tree.js — retro reading tree + dithered progress meter for articles.
//
// Turns the margin-sidebar TOC (`nav#TOC`) into a monospaced "tree" of the
// article's headings, with a dithered progress bar, a percentage, and a
// keyboard hint. ArrowUp / ArrowDown jump to the previous / next heading.
//
// No-ops unless body carries `article-page` (articles/_metadata.yml) and the
// margin sidebar holds a TOC with heading links. UI strings come from
// js/site-i18n.js ("reading.*" keys), which loads before this script.
(function () {
  "use strict";

  var t = window.siteI18n
    ? window.siteI18n.t.bind(window.siteI18n)
    : function (key) { return key; };

  function init() {
    if (!document.body.classList.contains("article-page")) return;

    var toc = document.querySelector("#quarto-margin-sidebar nav#TOC");
    if (!toc) return;

    var links = Array.prototype.slice.call(
      toc.querySelectorAll("a.nav-link[data-scroll-target]")
    );
    if (!links.length) return;

    var content = document.getElementById("quarto-document-content");
    var targets = links
      .map(function (link) {
        return document.getElementById(
          link.getAttribute("data-scroll-target").replace(/^#/, "")
        );
      })
      .filter(Boolean);

    // --- retro chrome: title, dithered meter, keyboard hint -----------------
    var title = toc.querySelector("#toc-title");
    if (title) title.textContent = t("reading.tree");

    var meter = document.createElement("div");
    meter.className = "reading-meter";
    meter.setAttribute("role", "group");
    meter.setAttribute("aria-label", t("reading.progress", { pct: 0 }));

    var track = document.createElement("div");
    track.className = "reading-meter-track";
    track.setAttribute("aria-hidden", "true");
    var fill = document.createElement("div");
    fill.className = "reading-meter-fill";
    track.appendChild(fill);

    var value = document.createElement("span");
    value.className = "reading-meter-value";
    value.setAttribute("aria-hidden", "true");
    value.textContent = "0%";

    meter.appendChild(track);
    meter.appendChild(value);

    var hint = document.createElement("p");
    hint.className = "reading-hint";
    hint.textContent = t("reading.scrollHint");

    var actions = toc.querySelector(".toc-actions");
    toc.insertBefore(meter, actions || null);
    toc.insertBefore(hint, actions || null);

    // --- reading progress ---------------------------------------------------
    function progress() {
      if (!content) return 0;
      var top = content.getBoundingClientRect().top + window.scrollY;
      var span = content.offsetHeight - window.innerHeight;
      if (span <= 0) return 1;
      return Math.min(1, Math.max(0, (window.scrollY - top) / span));
    }

    var lastPct = -1;
    var raf = 0;
    function update() {
      raf = 0;
      var pct = Math.round(progress() * 100);
      if (pct === lastPct) return;
      lastPct = pct;
      fill.style.width = pct + "%";
      value.textContent = pct + "%";
      meter.setAttribute("aria-label", t("reading.progress", { pct: pct }));
    }
    function schedule() {
      if (!raf) raf = window.requestAnimationFrame(update);
    }
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    update();

    // --- ArrowUp / ArrowDown jump between headings --------------------------
    function anchorOffset() {
      var header = document.getElementById("quarto-header");
      return (header ? header.getBoundingClientRect().height : 0) + 12;
    }

    function behavior() {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth";
    }

    function jump(dir) {
      var offset = anchorOffset();
      var y = window.scrollY + offset;
      var best = null;
      for (var i = 0; i < targets.length; i++) {
        var top = targets[i].getBoundingClientRect().top + window.scrollY;
        if (dir > 0 && top > y + 2) { best = top; break; }
        if (dir < 0 && top < y - 2) best = top;
      }
      // Nothing in that direction: clamp to the end / the top of the page.
      if (best === null) {
        best = dir > 0
          ? document.documentElement.scrollHeight - window.innerHeight + offset
          : 0;
      }
      window.scrollTo({ top: Math.max(0, best - offset), behavior: behavior() });
    }

    function editableContext() {
      var el = document.activeElement;
      if (!el) return false;
      var tag = el.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
        el.isContentEditable;
    }

    window.addEventListener("keydown", function (e) {
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
      if (editableContext()) return;
      if (document.querySelector("dialog[open], .modal.show")) return;
      e.preventDefault();
      jump(e.key === "ArrowDown" ? 1 : -1);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
