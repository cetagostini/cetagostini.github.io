// language-switcher.js — refine the server-rendered language menu.
//
// _quarto.yml emits one navbar menu listing every language, each entry labelled
// in its own language (a language is never renamed) and pointing at that tree's
// absolute URL. Without JS that absolute link is already correct from every
// page; this script makes it *exact*:
//
//   • every entry's href becomes THIS page's counterpart in that language,
//     taken from the <link rel="alternate" hreflang=…> tags llm-seo.lua emits
//     and re-based onto the origin actually serving the page, so a local
//     preview stays local
//   • the entry for the current page's language gets aria-current="page"
//   • the toggle becomes a globe whose accessible name names the current
//     language, so the control reads the same on every tree
//
// The menu is identified by its labels, not by its hrefs: Quarto rewrites a
// navbar href relative to each tree's own output directory, so on the
// Portuguese tree the Spanish entry arrives as "./es/" — which resolves to
// /pt/es/. The labels are the one part of the menu the pipeline never rewrites.
(function () {
  "use strict";

  // Language names, in their own language — the menu's labels. Only the toggle
  // needs the extra word for "language".
  var LANG_NAMES = { en: "English", es: "Español", pt: "Português" };
  var LANG_WORD = { en: "Language", es: "Idioma", pt: "Idioma" };

  function nameToLang(name) {
    for (var code in LANG_NAMES) {
      if (LANG_NAMES.hasOwnProperty(code) && LANG_NAMES[code] === name) return code;
    }
    return null;
  }

  function currentLang() {
    var lang = (document.documentElement.getAttribute("lang") || "en").toLowerCase();
    if (lang.indexOf("es") === 0) return "es";
    if (lang.indexOf("pt") === 0) return "pt";
    return "en";
  }

  // ── Read hreflang alternates from <head> ──────────────────────────────
  // Keyed by the BCP47 tag llm-seo.lua wrote (en, es, pt-PT); `x-default` is a
  // duplicate of the English route and is ignored.
  function getAlternates() {
    var links = document.querySelectorAll('link[rel="alternate"][hreflang]');
    var alternates = {};
    for (var i = 0; i < links.length; i++) {
      var tag = links[i].getAttribute("hreflang");
      var href = links[i].getAttribute("href");
      if (tag && href && tag !== "x-default") alternates[tag] = href;
    }
    return alternates;
  }

  // The tag this tree appears under in the alternate set. The Portuguese tree
  // is lang="pt-PT" / hreflang="pt-PT", so a region suffix is matched by prefix.
  function tagFor(alternates, lang) {
    if (alternates[lang]) return lang;
    for (var tag in alternates) {
      if (alternates.hasOwnProperty(tag) && tag.indexOf(lang + "-") === 0) return tag;
    }
    return null;
  }

  // ── Find the language menu by its labels ──────────────────────────────
  function findMenu() {
    var anchors = document.querySelectorAll(".navbar a.dropdown-item");
    var found = [];
    for (var i = 0; i < anchors.length; i++) {
      var anchor = anchors[i];
      var label = anchor.querySelector(".dropdown-text") || anchor;
      var lang = nameToLang(label.textContent.trim());
      if (lang) found.push({ anchor: anchor, lang: lang });
    }
    return found;
  }

  // An alternate's path, on the origin actually serving this page. The href is
  // absolute because hreflang must be, but following it from a local preview
  // would jump the reader to production.
  function localHref(href) {
    try {
      var path = new URL(href, window.location.origin).pathname;
      return window.location.origin + path + window.location.search + window.location.hash;
    } catch (e) {
      return null;
    }
  }

  // ── Visible control ───────────────────────────────────────────────────
  // Inline SVG, not an icon font: this site does not load bootstrap-icons, so
  // any <i class="bi …"> renders as an invisible blank.
  var GLOBE =
    '<svg class="lang-switch-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<circle cx="12" cy="12" r="9"/>' +
    '<path d="M3 12h18"/>' +
    '<path d="M12 3c2.7 3.5 2.7 14.5 0 18-2.7-3.5-2.7-14.5 0-18z"/>' +
    "</svg>";

  function updateToggle(anchor, lang) {
    var toggle = anchor.closest(".dropdown");
    toggle = toggle && toggle.querySelector(".dropdown-toggle");
    if (!toggle) return;
    var name = LANG_NAMES[lang] || lang;
    toggle.innerHTML = GLOBE;
    toggle.setAttribute("lang", lang);
    toggle.setAttribute("aria-label", (LANG_WORD[lang] || "Language") + ": " + name);
    toggle.setAttribute("title", name);
    toggle.setAttribute("data-lang-switcher", "");
  }

  function init() {
    var links = findMenu();
    if (!links.length) return;

    var alternates = getAlternates();
    var lang = currentLang();
    var currentTag = tagFor(alternates, lang);

    for (var i = 0; i < links.length; i++) {
      var link = links[i];
      var tag = tagFor(alternates, link.lang);
      var href = tag && alternates[tag] ? localHref(alternates[tag]) : null;
      if (href) link.anchor.setAttribute("href", href);
      if (tag && tag === currentTag) {
        link.anchor.setAttribute("aria-current", "page");
      } else {
        link.anchor.removeAttribute("aria-current");
      }
    }

    updateToggle(links[0].anchor, lang);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
