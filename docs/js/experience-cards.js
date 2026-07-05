// Experience cards: inline accordion (replaces the old modal/popup).
document.addEventListener("DOMContentLoaded", function () {
  var cards = document.querySelectorAll(".experience-card");

  cards.forEach(function (card) {
    var toggle = card.querySelector(".experience-toggle");
    var details = card.querySelector(".experience-details");
    if (!toggle || !details) return;

    // Ensure the toggle is a real button for a11y.
    if (toggle.tagName !== "BUTTON") {
      var btn = document.createElement("button");
      btn.className = toggle.className;
      btn.textContent = toggle.textContent;
      toggle.replaceWith(btn);
      toggle = btn;
    }
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("type", "button");

    if (!details.id) details.id = "exp-details-" + Math.random().toString(36).slice(2, 8);
    toggle.setAttribute("aria-controls", details.id);

    toggle.addEventListener("click", function () {
      var open = card.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.textContent = open ? "Show less" : "Read more";
    });
  });
});
