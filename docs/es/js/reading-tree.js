// reading-tree.js — retro reading contents + dithered progress meter for articles.
//
// Turns the margin-sidebar TOC (`nav#TOC`) into a monospaced "Content" list of
// the article's headings, with a dithered progress bar and a percentage that
// track how much of the article lies ahead.
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

    var links = toc.querySelectorAll("a.nav-link[data-scroll-target]");
    if (!links.length) return;

    var content = document.getElementById("quarto-document-content");

    // --- retro chrome: title + dithered meter ------------------------------
    var title = toc.querySelector("#toc-title");
    if (title) title.textContent = t("reading.contents");

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

    var actions = toc.querySelector(".toc-actions");
    toc.insertBefore(meter, actions || null);

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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
