// Single-card infinite video carousel + lightbox (Talks page)
(function () {
  function init() {
    var carousel = document.querySelector("[data-carousel]");
    if (!carousel) return;
    var viewport = carousel.querySelector(".carousel-viewport");
    var track = carousel.querySelector(".carousel-track");
    var cards = Array.prototype.slice.call(track.children);
    if (cards.length < 1) return;
    var N = cards.length;
    var prevBtn = carousel.querySelector(".carousel-prev");
    var nextBtn = carousel.querySelector(".carousel-next");
    var dotsWrap = carousel.querySelector(".carousel-dots");
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // Clones for a seamless infinite loop (first at end, last at start)
    var firstClone = cards[0].cloneNode(true);
    var lastClone = cards[N - 1].cloneNode(true);
    firstClone.setAttribute("aria-hidden", "true");
    firstClone.setAttribute("tabindex", "-1");
    lastClone.setAttribute("aria-hidden", "true");
    lastClone.setAttribute("tabindex", "-1");
    track.appendChild(firstClone);
    track.insertBefore(lastClone, track.firstChild);

    var total = N + 2;
    var index = 1; // first real card (after prepended last-clone)
    var animating = false;

    // Dots
    var dots = [];
    for (var i = 0; i < N; i++) {
      var d = document.createElement("button");
      d.className = "carousel-dot";
      d.type = "button";
      d.setAttribute("role", "tab");
      d.setAttribute("aria-label", "Go to video " + (i + 1));
      (function (idx) { d.addEventListener("click", function () { goTo(idx); }); })(i);
      dotsWrap.appendChild(d);
      dots.push(d);
    }

    function step() {
      // single-card: step = viewport width (one full card)
      return viewport.offsetWidth;
    }

    function realIndex(i) {
      if (i === 0) return N - 1;
      if (i === total - 1) return 0;
      return i - 1;
    }

    function update(animate) {
      animating = animate;
      track.style.transition = animate && !reduce
        ? "transform .5s cubic-bezier(.2,.7,.3,1)"
        : "none";
      track.style.transform = "translateX(" + (-index * step()) + "px)";
      var r = realIndex(index);
      dots.forEach(function (d, i) { d.classList.toggle("active", i === r); });
    }

    function goTo(i) { if (animating) return; index = i + 1; update(true); }
    function next() { if (animating) return; index++; update(true); }
    function prev() { if (animating) return; index--; update(true); }

    nextBtn.addEventListener("click", next);
    prevBtn.addEventListener("click", prev);

    track.addEventListener("transitionend", function () {
      animating = false;
      if (index === total - 1) { index = 1; update(false); }
      else if (index === 0) { index = N; update(false); }
    });

    // Lightbox
    var lb = document.getElementById("video-lightbox");
    var lbFrame = lb ? lb.querySelector(".lightbox-frame") : null;
    var lbCaption = lb ? lb.querySelector(".lightbox-caption") : null;

    function openLightbox(embed, caption) {
      if (!lb || !embed) return;
      lbFrame.innerHTML =
        '<iframe src="' + embed + '" title="' + (caption || "Video") +
        '" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>';
      lbCaption.textContent = caption || "";
      lb.classList.add("active");
      lb.setAttribute("aria-hidden", "false");
      document.body.style.overflow = "hidden";
    }

    function closeLightbox() {
      if (!lb) return;
      lb.classList.remove("active");
      lb.setAttribute("aria-hidden", "true");
      if (lbFrame) lbFrame.innerHTML = "";
      document.body.style.overflow = "";
    }

    Array.prototype.forEach.call(track.querySelectorAll(".video-card"), function (card) {
      card.addEventListener("click", function () {
        openLightbox(card.getAttribute("data-embed"), card.getAttribute("data-caption") || "");
      });
    });

    if (lb) {
      lb.querySelector(".lightbox-backdrop").addEventListener("click", closeLightbox);
      var cBtn = lb.querySelector(".lightbox-close");
      if (cBtn) cBtn.addEventListener("click", closeLightbox);
      document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeLightbox(); });
    }

    // Touch swipe
    var startX = null;
    viewport.addEventListener("touchstart", function (e) { startX = e.touches[0].clientX; }, { passive: true });
    viewport.addEventListener("touchend", function (e) {
      if (startX === null) return;
      var dx = e.changedTouches[0].clientX - startX;
      if (Math.abs(dx) > 40) { dx < 0 ? next() : prev(); }
      startX = null;
    });

    window.addEventListener("resize", function () { update(false); });
    window.addEventListener("load", function () { update(false); });
    update(false);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
