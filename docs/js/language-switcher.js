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
        return u.href;
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
        return u.href;
      } catch (e) { /* fall through */ }
    }
    return getCounterpartURL();
  }

  // ── Dropdown creation / discovery ─────────────────────────────────────
  function findExistingDropdown() {
    // Look for any navbar dropdown whose menu contains links to /es/ or /
    var dropdowns = document.querySelectorAll(".navbar .dropdown");
    for (var i = 0; i < dropdowns.length; i++) {
      var dd = dropdowns[i];
      var links = dd.querySelectorAll(".dropdown-menu a");
      for (var j = 0; j < links.length; j++) {
        var href = links[j].getAttribute("href") || "";
        if (href === "/es/" || href === "/es/index.html" ||
            href === "/" || href === "/index.html") {
          return dd;
        }
      }
    }
    return null;
  }

  function createDropdown() {
    var nav = document.querySelector(".navbar-nav.ms-auto, .navbar-nav:last-of-type");
    if (!nav) return null;

    var li = document.createElement("li");
    li.className = "nav-item dropdown";

    var toggle = document.createElement("a");
    toggle.className = "nav-link dropdown-toggle";
    toggle.href = "#";
    toggle.setAttribute("role", "button");
    toggle.setAttribute("data-bs-toggle", "dropdown");
    toggle.setAttribute("aria-expanded", "false");
    toggle.innerHTML = '<i class="bi bi-globe2"></i>';

    var menu = document.createElement("ul");
    menu.className = "dropdown-menu dropdown-menu-end";

    var enItem = document.createElement("li");
    var enLink = document.createElement("a");
    enLink.className = "dropdown-item";
    enLink.href = "/"; // placeholder
    enLink.textContent = "English";
    enItem.appendChild(enLink);

    var esItem = document.createElement("li");
    var esLink = document.createElement("a");
    esLink.className = "dropdown-item";
    esLink.href = "/es/"; // placeholder
    esLink.textContent = "Español";
    esItem.appendChild(esLink);

    menu.appendChild(enItem);
    menu.appendChild(esItem);
    li.appendChild(toggle);
    li.appendChild(menu);
    nav.appendChild(li);

    return li;
  }

  // ── Update hrefs in the dropdown ──────────────────────────────────────
  function updateDropdown(dropdown) {
    if (!dropdown) return;

    var links = dropdown.querySelectorAll(".dropdown-menu a");
    var enLink = null;
    var esLink = null;

    for (var i = 0; i < links.length; i++) {
      var text = links[i].textContent.trim();
      if (text === "English") enLink = links[i];
      else if (text === "Español") esLink = links[i];
    }

    if (!enLink || !esLink) return;

    // Compute counterpart URL
    var counterpart = counterpartURLSameRoute();

    if (isES) {
      // Current page is Spanish: "English" gets the EN URL, "Español" is current
      enLink.href = counterpart;
      esLink.href = window.location.href;
      enLink.removeAttribute("aria-current");
      esLink.setAttribute("aria-current", "page");
    } else {
      // Current page is English: "Español" gets the ES URL, "English" is current
      esLink.href = counterpart;
      enLink.href = window.location.href;
      esLink.removeAttribute("aria-current");
      enLink.setAttribute("aria-current", "page");
    }
  }

  // ── Init ──────────────────────────────────────────────────────────────
  function init() {
    // Find or create the language dropdown
    var dropdown = findExistingDropdown();
    if (!dropdown) {
      dropdown = createDropdown();
    }
    if (!dropdown) return;

    // Mark it for downstream code
    dropdown.setAttribute("data-lang-switcher", "");
    var toggle = dropdown.querySelector('[data-bs-toggle="dropdown"], [data-toggle="dropdown"]');
    if (toggle) toggle.setAttribute("data-lang-switcher", "");

    updateDropdown(dropdown);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();