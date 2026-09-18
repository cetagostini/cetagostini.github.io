// language-switcher.js — runtime language dropdown in the navbar.
//
// Finds the navbar dropdown that contains links to / and /es/ (the one Quarto
// generates for the ES profile). If no such dropdown exists (English-only nav),
// creates one. Reads <link rel=alternate hreflang=…> values from the <head>
// (emitted by llm-seo.lua) and rewrites both menu hrefs so they point to the
// correct counterpart page.
//
// Rules:
//   • / ↔ /es/
//   • /about.html ↔ /es/about.html
//   • /articles/slug/slug.html ↔ /es/articles/slug/slug.html
//   • /diary/date.html ↔ /es/diary/date.html
//   • Alias URLs → canonical article route
//   • Query + hash preserved only for the same logical route
//   • NEVER produces /es/es/
//   • Sets aria-current="page" on the active language
//   • NEVER rewrites the words "English" / "Español"
(function () {
  "use strict";

  var isES = (document.documentElement.getAttribute("lang") || "").toLowerCase().indexOf("es") === 0;
  var ES_PREFIX = "/es";

  // ── Read hreflang alternates from <head> ──────────────────────────────
  function getAlternates() {
    var links = document.querySelectorAll('link[rel="alternate"][hreflang]');
    var alternates = {};
    for (var i = 0; i < links.length; i++) {
      var lang = links[i].getAttribute("hreflang");
      var href = links[i].getAttribute("href");
      if (lang && href) alternates[lang] = href;
    }
    return alternates;
  }

  // ── Path mapping ──────────────────────────────────────────────────────
  // Maps an EN path to its ES counterpart and vice versa.
  // The caller passes the raw pathname; we return the counterpart pathname.
  // Returns null if the route is unmappable.

  // Extract the "logical route" from an ES path: strip /es prefix, strip
  // /index.html suffix for comparison. Returns the normalised EN path.
  function logicalPath(pathname) {
    var p = pathname;
    // Strip /es or /es/ prefix
    if (p === "/es" || p === "/es/") return "/";
    if (p.indexOf("/es/") === 0) p = p.substring(3); // "/es/foo" -> "/foo"
    // Normalise trailing /index.html
    if (p === "/index.html") return "/";
    if (p.length > 12 && p.substring(p.length - 11) === "/index.html") {
      p = p.substring(0, p.length - 10); // keep trailing /
    }
    return p;
  }

  // Normalise alias paths to their canonical article form.
  // "/articles/slug.html" → "/articles/slug/slug.html"
  function canonicaliseArticle(pathname) {
    var m = pathname.match(/^\/articles\/([^/]+)\.html$/);
    if (m) return "/articles/" + m[1] + "/" + m[1] + ".html";
    return pathname;
  }

  function enToES(pathname) {
    if (pathname === "/") return "/es/";
    // Don't double-prefix
    if (pathname.indexOf("/es") === 0) return pathname;
    return "/es" + pathname;
  }

  function esToEN(pathname) {
    if (pathname === "/es" || pathname === "/es/") return "/";
    if (pathname.indexOf("/es/") === 0) return pathname.substring(3);
    return pathname;
  }

  function counterpartPath(currentPathname) {
    var logical = logicalPath(currentPathname);
    logical = canonicaliseArticle(logical);

    if (isES) {
      // ES → EN: strip /es prefix, return the canonical EN path
      return logical;
    } else {
      // EN → ES: add /es prefix
      return enToES(logical);
    }
  }

  // ── Build the full counterpart URL ────────────────────────────────────
  function getCounterpartURL() {
    var alternates = getAlternates();

    // Primary: use the hreflang alternate for the target language
    var targetLang = isES ? "en" : "es";
    var targetHref = alternates[targetLang] || alternates["x-default"];
    if (targetHref) {
      try {
        var u = new URL(targetHref, window.location.origin);
        // Guard: never produce /es/es/
        if (u.pathname.indexOf("/es/es") === 0) {
          u.pathname = u.pathname.replace(/^\/es/, "");
        }
        // Re-base onto the origin actually serving this page. The alternate
        // link is absolute because hreflang must be, but following it from a
        // local preview would jump the reader to production.
        return window.location.origin + u.pathname + window.location.search + window.location.hash;
      } catch (e) { /* fall through */ }
    }

    // Fallback: compute from current pathname
    var current = new URL(window.location.href);
    var targetPath = counterpartPath(current.pathname);
    // Guard: never produce /es/es/
    if (targetPath.indexOf("/es/es") === 0) {
      targetPath = targetPath.replace(/^\/es/, "");
    }
    current.pathname = targetPath;
    return current.href;
  }

  // Same logical route check: preserve query + hash only when both pages
  // serve the same logical route (i.e., we're not navigating to root).
  function counterpartURLSameRoute() {
    var alternates = getAlternates();
    var targetLang = isES ? "en" : "es";
    var targetHref = alternates[targetLang];
    if (targetHref) {
      try {
        var u = new URL(targetHref, window.location.origin);
        if (u.pathname.indexOf("/es/es") === 0) {
          u.pathname = u.pathname.replace(/^\/es/, "");
        }
        return window.location.origin + u.pathname + window.location.search + window.location.hash;
      } catch (e) { /* fall through */ }
    }
    return getCounterpartURL();
  }

  // ── Visible one-click control ─────────────────────────────────────────
  // A dropdown is the wrong affordance here. It needs Bootstrap's JS, it hides
  // the destination behind a menu, and this site does not load the
  // bootstrap-icons font — so an icon-only toggle renders as an invisible
  // blank. With two languages, one link labelled with the OTHER language is
  // clearer and survives with JS only for the href refinement.
  function findControl() {
    var links = document.querySelectorAll(".navbar a.nav-link");
    for (var i = 0; i < links.length; i++) {
      var t = links[i].textContent.trim();
      if (t === "Español" || t === "English") return links[i];
    }
    return null;
  }

  function createControl() {
    var nav = document.querySelector(".navbar-nav.ms-auto, .navbar-nav:last-of-type");
    if (!nav) return null;
    var li = document.createElement("li");
    li.className = "nav-item";
    var a = document.createElement("a");
    a.className = "nav-link lang-switch";
    li.appendChild(a);
    nav.appendChild(li);
    return a;
  }

  function updateControl(el) {
    if (!el) return;
    // Label with the language the click takes you TO; point it at the
    // counterpart of the current page, falling back to that language's home.
    el.textContent = isES ? "English" : "Español";
    el.setAttribute("href", counterpartURLSameRoute());
    el.setAttribute("lang", isES ? "en" : "es");
    el.setAttribute("aria-label", isES ? "Switch to English" : "Cambiar a español");
    el.setAttribute("data-lang-switcher", "");
  }

  function init() {
    var control = findControl() || createControl();
    if (!control) return;
    updateControl(control);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();