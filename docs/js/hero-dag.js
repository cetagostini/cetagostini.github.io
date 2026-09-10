// The static SVG is the source of truth; motion only changes its presentation.
(function () {
  "use strict";

  function init() {
    const hero = document.querySelector(".home-shell");
    const stage = hero && hero.querySelector(".dag-stage");
    const svg = stage && stage.querySelector(".dag-network");
    if (!svg) return;
    const nodeLayer = svg.querySelector(".dn-nodes");
    const edgeLayer = svg.querySelector(".dn-edges");
    const aura = hero.querySelector(".hero-aura");
    const toggle = hero.querySelector("[data-dag-toggle]");
    const pulseButton = hero.querySelector("[data-dag-pulse]");
    const hint = hero.querySelector("[data-dag-hint]");
    const reduce = matchMedia("(prefers-reduced-motion: reduce)");
    const fine = matchMedia("(hover: hover) and (pointer: fine)");
    const width = svg.viewBox.baseVal.width;
    const height = svg.viewBox.baseVal.height;
    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
    const nodes = Array.from(nodeLayer.querySelectorAll(".dn-node-group"), (el, i) => {
      const circle = el.querySelector(".dn-node");
      const ring = el.querySelector(".dn-node-ring");
      return {
        el, ring, id: el.dataset.node, phase: i * 1.7,
        x: circle.cx.baseVal.value, y: circle.cy.baseVal.value,
        radius: ring.r.baseVal.value, dx: 0, dy: 0
      };
    });
    const byId = new Map(nodes.map(node => [node.id, node]));
    const edges = Array.from(edgeLayer.querySelectorAll(".dn-edge"), el => {
      const original = el.getAttribute("d");
      return {
        el, original, source: byId.get(el.dataset.source), target: byId.get(el.dataset.target),
        // Markup uses one cubic Bezier per edge. Keep its authored curvature.
        base: original.match(/-?\d*\.?\d+/g).map(Number), points: new Float64Array(8)
      };
    });
    if (!nodes.length || edges.some(edge => !edge.source || !edge.target || edge.base.length !== 8)) return;

    function element(tag, className, parent) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      el.setAttribute("class", className);
      parent.appendChild(el);
      return el;
    }
    function layer(className, before) {
      const el = element("g", className, svg);
      el.setAttribute("aria-hidden", "true");
      el.setAttribute("pointer-events", "none");
      el.style.visibility = "hidden";
      svg.insertBefore(el, before);
      return el;
    }
    function position(el, x, y) {
      el.setAttribute("cx", x.toFixed(2));
      el.setAttribute("cy", y.toFixed(2));
    }

    const field = layer("dn-field", edgeLayer);
    const wash = element("circle", "dn-cursor-wash", field);
    wash.setAttribute("r", "230");
    const mesh = element("g", "dn-mesh", field);
    const dust = element("g", "dn-dust", field);
    const specks = Array.from({ length: 120 }, (_, i) => {
      const el = element("circle", "dn-speck", dust);
      el.setAttribute("r", i % 5 === 0 ? "2.1" : "1.2");
      const x = 20 + Math.random() * (width - 40);
      const y = 30 + Math.random() * (height - 60);
      return { el, bx: x, by: y, x, y, phase: i * 2.4 };
    });
    // Build a sparse neighbourhood once, not an all-pairs network every frame.
    const links = [];
    for (let i = 0; i < specks.length; i++) {
      let nearest = -1;
      let distance = 220 * 220;
      for (let j = i + 1; j < specks.length; j++) {
        const d = (specks[i].x - specks[j].x) ** 2 + (specks[i].y - specks[j].y) ** 2;
        if (d < distance) { nearest = j; distance = d; }
      }
      if (nearest >= 0) links.push({ el: element("line", "dn-mesh-link", mesh), a: specks[i], b: specks[nearest] });
    }
    const ripple = element("circle", "dn-ripple", field);
    const cursorRing = element("circle", "dn-cursor-ring", field);
    cursorRing.setAttribute("r", "12");
    const travelers = layer("dn-particles", nodeLayer);
    const particles = [];
    for (const edge of edges) {
      for (let i = 0; i < 2; i++) {
        const tail = element("circle", "dn-particle-tail", travelers);
        const dot = element("circle", "dn-particle", travelers);
        tail.setAttribute("r", "4");
        dot.setAttribute("r", "2.5");
        particles.push({ edge, dot, tail, phase: i / 2 + Math.random() * .12, speed: .10 + Math.random() * .025 });
      }
    }

    let raf = 0;
    let previous = null;
    let clock = 0;
    let flow = 0;
    let paused = false;
    let pulseAge = 2;
    let nearest = null;
    let presence = 0;
    let halo = 0;
    const pointer = {
      active: false, inside: false, x: width / 2, y: height / 2,
      sx: width / 2, sy: height / 2, nx: 0, ny: 0,
      hx: stage.clientWidth * .75, hy: stage.clientHeight * .45,
      ax: stage.clientWidth * .75, ay: stage.clientHeight * .45
    };
    const allowed = () => !reduce.matches && fine.matches;
    const canRun = () => allowed() && !paused && !document.hidden;

    function clearPointer() {
      pointer.active = false;
      pointer.inside = false;
      pointer.nx = 0;
      pointer.ny = 0;
    }
    function highlight(node) {
      if (nearest === node) return;
      if (nearest) nearest.el.classList.remove("is-near");
      nearest = node;
      if (node) node.el.classList.add("is-near");
      for (const edge of edges) edge.el.classList.toggle("is-active", !!node && (edge.source === node || edge.target === node));
    }
    function staticGraph() {
      highlight(null);
      for (const node of nodes) {
        node.dx = node.dy = 0;
        node.el.removeAttribute("transform");
        node.ring.setAttribute("r", node.radius);
      }
      for (const edge of edges) edge.el.setAttribute("d", edge.original);
      field.style.visibility = travelers.style.visibility = "hidden";
      aura.hidden = true;
      presence = halo = 0;
      pulseAge = 2;
    }
    function placeOnCurve(el, points, t) {
      const u = 1 - t;
      const a = u * u * u;
      const b = 3 * u * u * t;
      const c = 3 * u * t * t;
      const d = t * t * t;
      position(el, a * points[0] + b * points[2] + c * points[4] + d * points[6],
        a * points[1] + b * points[3] + c * points[5] + d * points[7]);
    }

    function frame(now) {
      raf = 0;
      if (!canRun()) return;
      const dt = previous === null ? 0 : Math.min((now - previous) / 1000, .05);
      previous = now;
      clock += dt;
      pulseAge += dt;
      const pulse = Math.max(0, 1 - pulseAge / 1.8);
      const ease = 1 - Math.exp(-dt * 7);
      presence += ((pointer.active ? 1 : 0) - presence) * ease;
      halo += ((pointer.inside ? 1 : 0) - halo) * ease;
      pointer.sx += (pointer.x - pointer.sx) * ease;
      pointer.sy += (pointer.y - pointer.sy) * ease;
      pointer.ax += (pointer.hx - pointer.ax) * ease;
      pointer.ay += (pointer.hy - pointer.ay) * ease;
      aura.style.setProperty("--aura-position", `translate3d(${pointer.ax.toFixed(1)}px, ${pointer.ay.toFixed(1)}px, 0)`);
      aura.style.opacity = (.12 + presence * .62).toFixed(3);
      position(wash, pointer.sx, pointer.sy);
      position(cursorRing, pointer.sx, pointer.sy);
      wash.setAttribute("opacity", halo.toFixed(3));
      cursorRing.setAttribute("opacity", (halo * .65).toFixed(3));
      ripple.setAttribute("r", (16 + (1 - pulse) * 360).toFixed(2));
      ripple.setAttribute("opacity", (pulse * pulse * .65).toFixed(3));

      let closest = null;
      let distance = 140 * 140;
      for (const node of nodes) {
        const mx = pointer.x - node.x;
        const my = pointer.y - node.y;
        const influence = pointer.inside ? Math.max(0, 1 - Math.hypot(mx, my) / 290) : 0;
        const tx = clamp(Math.sin(clock * .8 + node.phase) * 7 + pointer.nx * 18 + mx * influence * .32, -38, 38);
        const ty = clamp(Math.cos(clock * .65 + node.phase) * 7 + pointer.ny * 14 + my * influence * .32, -38, 38);
        node.dx += (tx - node.dx) * ease;
        node.dy += (ty - node.dy) * ease;
        node.el.setAttribute("transform", `translate(${node.dx.toFixed(2)} ${node.dy.toFixed(2)})`);
        node.ring.setAttribute("r", (node.radius + Math.sin(clock * 1.8 + node.phase) * 2 + pulse * 5).toFixed(2));
        const d = (pointer.x - node.x - node.dx) ** 2 + (pointer.y - node.y - node.dy) ** 2;
        if (pointer.inside && d < distance) { closest = node; distance = d; }
      }
      highlight(closest);
      for (const edge of edges) {
        const p = edge.points;
        for (let i = 0; i < 8; i += 2) {
          const node = i < 4 ? edge.source : edge.target;
          p[i] = edge.base[i] + node.dx;
          p[i + 1] = edge.base[i + 1] + node.dy;
        }
        edge.el.setAttribute("d", `M${p[0]} ${p[1]} C${p[2]} ${p[3]} ${p[4]} ${p[5]} ${p[6]} ${p[7]}`);
      }
      flow += dt * (1 + presence * .8 + pulse * 3);
      for (const particle of particles) {
        const t = (flow * particle.speed + particle.phase) % 1;
        placeOnCurve(particle.dot, particle.edge.points, t);
        placeOnCurve(particle.tail, particle.edge.points, Math.max(0, t - .025));
      }
      for (const speck of specks) {
        const dx = speck.bx - pointer.sx;
        const dy = speck.by - pointer.sy;
        const d = Math.hypot(dx, dy);
        const push = halo * Math.max(0, 1 - d / 210) * 75;
        const tx = clamp(speck.bx + Math.sin(clock * .35 + speck.phase) * 16 + dx / Math.max(d, 1) * push, 8, width - 8);
        const ty = clamp(speck.by + Math.cos(clock * .3 + speck.phase) * 16 + dy / Math.max(d, 1) * push, 8, height - 8);
        speck.x += (tx - speck.x) * ease;
        speck.y += (ty - speck.y) * ease;
        position(speck.el, speck.x, speck.y);
        speck.el.setAttribute("opacity", (.35 + Math.max(0, 1 - d / 230) * halo * .6).toFixed(3));
      }
      for (const link of links) {
        const d = Math.hypot(link.a.x - link.b.x, link.a.y - link.b.y);
        link.el.setAttribute("x1", link.a.x.toFixed(2));
        link.el.setAttribute("y1", link.a.y.toFixed(2));
        link.el.setAttribute("x2", link.b.x.toFixed(2));
        link.el.setAttribute("y2", link.b.y.toFixed(2));
        link.el.setAttribute("opacity", (Math.max(0, 1 - d / 240) * (.3 + halo * .3)).toFixed(3));
      }
      field.style.visibility = travelers.style.visibility = "visible";
      raf = requestAnimationFrame(frame);
    }

    function sync() {
      const enabled = allowed();
      toggle.hidden = pulseButton.hidden = !enabled;
      toggle.textContent = paused ? "Resume motion" : "Pause motion";
      pulseButton.disabled = paused;
      hint.textContent = !enabled ? "An illustrative causal field" : paused ? "Motion paused" : "Move anywhere · Make connections";
      hero.classList.toggle("is-motion-paused", paused || !enabled);
      if (!enabled) { clearPointer(); staticGraph(); }
      else aura.hidden = false;
      if (canRun()) {
        if (!raf) { previous = null; raf = requestAnimationFrame(frame); }
      } else {
        cancelAnimationFrame(raf);
        raf = 0;
        previous = null;
      }
    }
    function point(event) {
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      const inverse = matrix.inverse();
      const bounds = stage.getBoundingClientRect();
      pointer.active = true;
      pointer.x = inverse.a * event.clientX + inverse.c * event.clientY + inverse.e;
      pointer.y = inverse.b * event.clientX + inverse.d * event.clientY + inverse.f;
      pointer.inside = pointer.x >= 0 && pointer.x <= width && pointer.y >= 0 && pointer.y <= height;
      pointer.hx = event.clientX - bounds.left;
      pointer.hy = event.clientY - bounds.top;
      pointer.nx = clamp(pointer.hx / bounds.width * 2 - 1, -1, 1);
      pointer.ny = clamp(pointer.hy / bounds.height * 2 - 1, -1, 1);
    }
    function sendPulse(x, y) {
      if (!canRun()) return;
      position(ripple, x, y);
      pulseAge = 0;
    }
    document.addEventListener("pointermove", event => {
      if (canRun() && event.pointerType !== "touch") point(event);
    }, { passive: true });
    document.addEventListener("pointerleave", clearPointer);
    document.addEventListener("click", event => {
      if (!canRun() || event.target.closest("a, button, input, select, textarea, summary, [role='button'], [contenteditable]")) return;
      point(event);
      sendPulse(clamp(pointer.x, 0, width), clamp(pointer.y, 0, height));
    });
    pulseButton.addEventListener("click", () => {
      const source = byId.get("x");
      sendPulse(source.x + source.dx, source.y + source.dy);
    });
    toggle.addEventListener("click", () => { paused = !paused; clearPointer(); sync(); });
    window.addEventListener("resize", clearPointer, { passive: true });
    window.addEventListener("blur", clearPointer);
    document.addEventListener("visibilitychange", () => { clearPointer(); sync(); });
    reduce.addEventListener("change", sync);
    fine.addEventListener("change", sync);
    sync();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
