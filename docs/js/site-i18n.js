// site-i18n.js — tiny, dependency-free, synchronous translation helper.
// Exposes window.siteI18n.t(key, vars) for runtime i18n.
// Language detected from <html lang=…>: 'es'/'es-*' → Spanish, 'pt'/'pt-*' →
// Portuguese (the pt tree is lang="pt-PT"), anything else → English.
// Missing keys return the English string.
(function () {
  "use strict";

  var LANG = (function () {
    var lang = (document.documentElement.getAttribute("lang") || "").toLowerCase();
    if (lang === "es" || lang.indexOf("es-") === 0) return "es";
    if (lang === "pt" || lang.indexOf("pt-") === 0) return "pt";
    return "en";
  })();

  var dict = {
    // Footer prose
    "footer.connectWith": { en: "Connect with me:", es: "Conecta conmigo:",
                            pt: "Contacte-me:" },

    // Articles network — topic count
    "network.articles.one":   { en: "{count} article",  es: "{count} artículo",
                                pt: "{count} artigo" },
    "network.articles.other": { en: "{count} articles", es: "{count} artículos",
                                pt: "{count} artigos" },
    "network.keywords.connections": { en: "{keywords} keywords · {connections} connections",
                                      es: "{keywords} palabras clave · {connections} conexiones",
                                      pt: "{keywords} palavras-chave · {connections} ligações" },
    "network.articles.timeline": { en: "{count} articles · along the timeline",
                                    es: "{count} artículos · a lo largo de la línea de tiempo",
                                    pt: "{count} artigos · ao longo da linha temporal" },
    "network.topic.open": { en: "{label} · {count} articles · keyword open",
                             es: "{label} · {count} artículos · palabra clave abierta",
                             pt: "{label} · {count} artigos · palavra-chave aberta" },
    "network.status.summaryOpen": { en: " · summary open", es: " · resumen abierto",
                                    pt: " · resumo aberto" },
    "network.error.load": { en: "The interactive network could not load — the index below has every article.",
                             es: "La red interactiva no pudo cargarse — el índice a continuación tiene todos los artículos.",
                             pt: "A rede interativa não pôde ser carregada — o índice abaixo tem todos os artigos." },
    "network.sheet.close":       { en: "Close the summary",     es: "Cerrar el resumen",
                                   pt: "Fechar o resumo" },
    "network.sheet.preview":     { en: "Article preview",       es: "Vista previa del artículo",
                                   pt: "Pré-visualização do artigo" },
    "network.sheet.readMore":    { en: "Read more",             es: "Leer más",
                                   pt: "Ler mais" },
    "network.sheet.listen":      { en: "Listen to this article", es: "Escuchar este artículo",
                                   pt: "Ouvir este artigo" },
    "network.sheet.back":        { en: "Back to network",       es: "Volver a la red",
                                   pt: "Voltar à rede" },
    "network.hint.date":   { en: "Drag to pan · Scroll or pinch to zoom · Select an article for its summary · Esc to close",
                              es: "Arrastra para desplazar · Desplázate o pellizca para ampliar · Selecciona un artículo para ver su resumen · Esc para cerrar",
                              pt: "Arraste para deslocar · Desloque ou faça pinça para ampliar · Selecione um artigo para ver o resumo · Esc para fechar" },
    "network.hint.topic":  { en: "Select an article for its summary · Click the keyword again, Esc or an empty spot to fold it back",
                              es: "Selecciona un artículo para ver su resumen · Haz clic en la palabra clave de nuevo, Esc o en un espacio vacío para plegar",
                              pt: "Selecione um artigo para ver o resumo · Clique novamente na palavra-chave, Esc ou um espaço vazio para fechar" },
    "network.hint.pick":   { en: "Pick a keyword to open its articles · Drag to pan · Scroll or pinch to zoom",
                              es: "Elige una palabra clave para abrir sus artículos · Arrastra para desplazar · Desplázate o pellizca para ampliar",
                              pt: "Escolha uma palavra-chave para abrir os seus artigos · Arraste para deslocar · Desloque ou faça pinça para ampliar" },

    // Articles network — topic ARIA
    "network.aria.article.one":   { en: "{count} article.",  es: "{count} artículo.",
                                    pt: "{count} artigo." },
    "network.aria.article.other": { en: "{count} articles.", es: "{count} artículos.",
                                    pt: "{count} artigos." },
    "network.aria.open":   { en: "Open. Activate to fold its articles back.",
                              es: "Abierto. Activa para plegar sus artículos.",
                              pt: "Aberto. Ative para fechar os seus artigos." },
    "network.aria.closed": { en: "Activate to open its articles.",
                              es: "Activa para abrir sus artículos.",
                              pt: "Ative para abrir os seus artigos." },

    // Cookie consent
    "cookie.title":    { en: "Cookie Consent", es: "Consentimiento de cookies",
                         pt: "Consentimento de cookies" },
    "cookie.body":     { en: "This website uses cookies to enhance your browsing experience and analyze site traffic. By clicking \u201CAccept\u201D, you consent to the use of cookies for analytics purposes.",
                         es: "Este sitio web utiliza cookies para mejorar tu experiencia de navegación y analizar el tráfico. Al hacer clic en \u201CAceptar\u201D, consientes el uso de cookies con fines analíticos.",
                         pt: "Este site utiliza cookies para melhorar a sua experiência de navegação e analisar o tráfego. Ao clicar em \u00ABAceitar\u00BB, consente a utilização de cookies para fins analíticos." },
    "cookie.accept":   { en: "Accept", es: "Aceptar", pt: "Aceitar" },
    "cookie.decline":  { en: "Decline", es: "Rechazar", pt: "Recusar" },

    // Hero DAG — motion controls
    "hero.pause":        { en: "Pause motion",    es: "Pausar movimiento",
                           pt: "Pausar movimento" },
    "hero.resume":       { en: "Resume motion",   es: "Reanudar movimiento",
                           pt: "Retomar movimento" },
    "hero.noMotion":     { en: "An illustrative causal field", es: "Un campo causal ilustrativo",
                           pt: "Um campo causal ilustrativo" },
    "hero.paused":       { en: "Motion paused",   es: "Movimiento pausado",
                           pt: "Movimento em pausa" },
    "hero.interactHint": { en: "Move anywhere · Make connections", es: "Muévete en cualquier lugar · Crea conexiones",
                           pt: "Mova-se em qualquer lugar · Crie ligações" },

    // Career rail — role counter
    "career.counter": { en: "{current} of {total}", es: "{current} de {total}",
                        pt: "{current} de {total}" },

    // Video carousel
    "video.goTo":    { en: "Go to video {n}", es: "Ir al video {n}",
                       pt: "Ir para o vídeo {n}" },
    "video.default": { en: "Video",           es: "Video", pt: "Vídeo" },

    // Skip link
    "skip.toContent": { en: "Skip to content", es: "Ir al contenido",
                        pt: "Saltar para o conteúdo" }
  };

  /**
   * Translate a key, interpolating {vars}.
   * Missing keys return the English string (or the raw key if no English exists).
   */
  function t(key, vars) {
    var entry = dict[key];
    var str = entry ? (entry[LANG] || entry.en) : key;
    if (vars) {
      for (var k in vars) {
        if (vars.hasOwnProperty(k)) {
          str = str.replace(new RegExp("\\{" + k + "\\}", "g"), vars[k]);
        }
      }
    }
    return str;
  }

  function lang() { return LANG; }

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

  // Replace the footer "Connect with me:" prose at runtime. The server-rendered
  // label is English on every tree (the footer is base config), so match it and
  // the already-translated forms.
  function applyFooterProse() {
    var items = document.querySelectorAll(".page-footer .footer-items .nav-item");
    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      if (item.children.length === 0) {
        var text = item.textContent.trim();
        if (text === "Connect with me:" || text === "Conecta conmigo:" ||
            text === "Contacte-me:") {
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
