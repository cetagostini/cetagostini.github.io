// Hero DAG cursor micro-interaction
// Draws faint lines from the pointer to nearby DAG nodes on the home hero.
// Disabled under prefers-reduced-motion and on touch / no-fine-pointer devices.
(function () {
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  if (reduce || !fine) return;

  function init() {
    var net = document.querySelector(".dag-network");
    if (!net) return;
    var nodes = Array.prototype.slice.call(net.querySelectorAll(".dn-node"));
    if (!nodes.length) return;

    var home = document.querySelector(".home");
    if (!home) return;

    var ns = "http://www.w3.org/2000/svg";
    var group = document.createElementNS(ns, "g");
    group.setAttribute("class", "dn-cursor");
    net.appendChild(group);

    var R = 16; // radius of influence (viewBox units)
    var pending = null;
    var current = null; // last pointer in viewBox coords

    function toViewBox(clientX, clientY) {
      var m = net.getScreenCTM();
      if (!m) return null;
      var pt = net.createSVGPoint();
      pt.x = clientX; pt.y = clientY;
      return pt.matrixTransform(m.inverse());
    }

    function render() {
      pending = null;
      while (group.firstChild) group.removeChild(group.firstChild);
      if (!current) return;
      nodes.forEach(function (n) {
        var cx = parseFloat(n.getAttribute("cx"));
        var cy = parseFloat(n.getAttribute("cy"));
        var dx = current.x - cx, dy = current.y - cy;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < R) {
          var line = document.createElementNS(ns, "line");
          line.setAttribute("x1", current.x.toFixed(2));
          line.setAttribute("y1", current.y.toFixed(2));
          line.setAttribute("x2", cx);
          line.setAttribute("y2", cy);
          line.setAttribute("opacity", (0.35 * (1 - d / R)).toFixed(3));
          group.appendChild(line);
        }
      });
    }

    home.addEventListener("pointermove", function (e) {
      var p = toViewBox(e.clientX, e.clientY);
      if (!p) return;
      current = p;
      if (!pending) pending = requestAnimationFrame(render);
    });

    home.addEventListener("pointerleave", function () {
      current = null;
      if (!pending) pending = requestAnimationFrame(render);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
