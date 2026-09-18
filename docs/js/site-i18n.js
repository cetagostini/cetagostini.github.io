// site-i18n.js — tiny, dependency-free, synchronous translation helper.
// Exposes window.siteI18n.t(key, vars) for runtime i18n.
// Language detected from <html lang=…>: 'es' or 'es-*' → Spanish, else English.
// Missing keys return the English string.
(function () {
  "use strict";

  var isES = (function () {
    var lang = (document.documentElement.getAttribute("lang") || "").toLowerCase();
    return lang === "es" || lang.indexOf("es-") === 0;
  })();

  var dict = {
    // Footer prose
    "footer.connectWith": { en: "Connect with me:", es: "Conecta conmigo:" },

    // Articles network — topic count
    "network.articles.one":   { en: "{count} article",  es: "{count} artículo" },
    "network.articles.other": { en: "{count} articles", es: "{count} artículos" },
    "network.keywords.connections": { en: "{keywords} keywords · {connections} connections",
                                      es: "{keywords} palabras clave · {connections} conexiones" },
    "network.articles.timeline": { en: "{count} articles · along the timeline",
                                    es: "{count} artículos · a lo largo de la línea de tiempo" },
    "network.topic.open": { en: "{label} · {count} articles · keyword open",
                             es: "{label} · {count} artículos · palabra clave abierta" },
    "network.status.summaryOpen": { en: " · summary open", es: " · resumen abierto" },
    "network.error.load": { en: "The interactive network could not load — the index below has every article.",
                             es: "La red interactiva no pudo cargarse — el índice a continuación tiene todos los artículos." },
    "network.sheet.close":       { en: "Close the summary",     es: "Cerrar el resumen" },
    "network.sheet.preview":     { en: "Article preview",       es: "Vista previa del artículo" },
    "network.sheet.readMore":    { en: "Read more",             es: "Leer más" },
    "network.sheet.listen":      { en: "Listen to this article", es: "Escuchar este artículo" },
    "network.sheet.back":        { en: "Back to network",       es: "Volver a la red" },
    "network.hint.date":   { en: "Drag to pan · Scroll or pinch to zoom · Select an article for its summary · Esc to close",
                              es: "Arrastra para desplazar · Desplázate o pellizca para ampliar · Selecciona un artículo para ver su resumen · Esc para cerrar" },
    "network.hint.topic":  { en: "Select an article for its summary · Click the keyword again, Esc or an empty spot to fold it back",
                              es: "Selecciona un artículo para ver su resumen · Haz clic en la palabra clave de nuevo, Esc o en un espacio vacío para plegar" },
    "network.hint.pick":   { en: "Pick a keyword to open its articles · Drag to pan · Scroll or pinch to zoom",
                              es: "Elige una palabra clave para abrir sus artículos · Arrastra para desplazar · Desplázate o pellizca para ampliar" },

    // Articles network — topic ARIA
    "network.aria.article.one":   { en: "{count} article.",  es: "{count} artículo." },
    "network.aria.article.other": { en: "{count} articles.", es: "{count} artículos." },
    "network.aria.open":   { en: "Open. Activate to fold its articles back.",
                              es: "Abierto. Activa para plegar sus artículos." },
    "network.aria.closed": { en: "Activate to open its articles.",
                              es: "Activa para abrir sus artículos." },

    // Cookie consent
    "cookie.title":    { en: "Cookie Consent", es: "Consentimiento de cookies" },
    "cookie.body":     { en: "This website uses cookies to enhance your browsing experience and analyze site traffic. By clicking \u201CAccept\u201D, you consent to the use of cookies for analytics purposes.",
                         es: "Este sitio web utiliza cookies para mejorar tu experiencia de navegación y analizar el tráfico. Al hacer clic en \u201CAceptar\u201D, consientes el uso de cookies con fines analíticos." },
    "cookie.accept":   { en: "Accept", es: "Aceptar" },
    "cookie.decline":  { en: "Decline", es: "Rechazar" },

    // Hero DAG — motion controls
    "hero.pause":        { en: "Pause motion",    es: "Pausar movimiento" },
    "hero.resume":       { en: "Resume motion",   es: "Reanudar movimiento" },
    "hero.noMotion":     { en: "An illustrative causal field", es: "Un campo causal ilustrativo" },
    "hero.paused":       { en: "Motion paused",   es: "Movimiento pausado" },
    "hero.interactHint": { en: "Move anywhere · Make connections", es: "Muévete en cualquier lugar · Crea conexiones" },

    // Career rail — role counter
    "career.counter": { en: "{current} of {total}", es: "{current} de {total}" },

    // Video carousel
    "video.goTo":    { en: "Go to video {n}", es: "Ir al video {n}" },
    "video.default": { en: "Video",           es: "Video" },

    // Skip link
    "skip.toContent": { en: "Skip to content", es: "Ir al contenido" }
  };

  /**
   * Translate a key, interpolating {vars}.
   * Missing keys return the English string (or the raw key if no English exists).
   */
  function t(key, vars) {
    var entry = dict[key];
    var str = entry ? (isES ? entry.es || entry.en : entry.en) : key;
    if (vars) {
      for (var k in vars) {
        if (vars.hasOwnProperty(k)) {
          str = str.replace(new RegExp("\\{" + k + "\\}", "g"), vars[k]);
        }
      }
    }
    return str;
  }

  function lang() { return isES ? "es" : "en"; }

  window.siteI18n = { t: t, lang: lang };

  // --- data-i18n: replace textContent of any element with [data-i18n] ----
  function applyDataI18n() {
    var els = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var key = el.getAttribute("data-i18n");
      if (key) {
        var translated = t(key);
        if (translated !== key) el.textContent = translated;
      }
    }
  }

  // Apply data-i18n as soon as DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyDataI18n, { once: true });
  } else {
    applyDataI18n();
  }

  // Replace the footer "Connect with me:" prose at runtime
  function applyFooterProse() {
    var items = document.querySelectorAll(".page-footer .footer-items .nav-item");
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      if (item.children.length === 0) {
        var text = item.textContent.trim();
        if (text === "Connect with me:" || text === "Conecta conmigo:") {
          item.textContent = t("footer.connectWith");
        }
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyFooterProse, { once: true });
  } else {
    applyFooterProse();
  }
})();