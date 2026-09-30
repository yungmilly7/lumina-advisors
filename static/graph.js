/* Small self-contained force-directed graph renderer on <canvas>.
 * No external dependency (the CDN graph libs weren't reachable from every
 * network this had to be verified on, so this is a from-scratch
 * Fruchterman-Reingold-style layout + a hand-rolled renderer instead).
 * Exposes a single class, ForceGraph, used by app.js.
 */
class ForceGraph {
  constructor(container, canvas, tooltipEl) {
    this.container = container;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltipEl = tooltipEl;
    this.nodes = [];      // {id, label, title, x, y, vx, vy, radius, fill, stroke, strokeWidth, fixed}
    this.edges = [];      // {source, target, color, width, dashed:[..]|null, arrow:bool}
    this.nodeById = new Map();
    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.dpr = window.devicePixelRatio || 1;
    this.dragNode = null;
    this.dragMoved = false;
    this.panning = false;
    this.lastPointer = { x: 0, y: 0 };
    this.onNodeClick = null;
    this.hoverNode = null;

    this._resize = this._resize.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);

    window.addEventListener("resize", this._resize);
    canvas.addEventListener("wheel", this._onWheel, { passive: false });
    canvas.addEventListener("mousedown", this._onPointerDown);
    window.addEventListener("mousemove", this._onPointerMove);
    window.addEventListener("mouseup", this._onPointerUp);

    this._resize();
  }

  _resize() {
    const rect = this.container.getBoundingClientRect();
    this.width = rect.width;
    this.height = rect.height;
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.draw();
  }

  setData(nodes, edges) {
    const existing = this.nodeById;
    this.nodes = nodes.map((n) => {
      const prev = existing.get(n.id);
      return {
        ...n,
        x: prev ? prev.x : (Math.random() - 0.5) * 400,
        y: prev ? prev.y : (Math.random() - 0.5) * 400,
        vx: 0,
        vy: 0,
        fixed: prev ? prev.fixed : false,
      };
    });
    this.nodeById = new Map(this.nodes.map((n) => [n.id, n]));
    this.edges = edges
      .map((e) => ({ ...e, s: this.nodeById.get(e.source), t: this.nodeById.get(e.target) }))
      .filter((e) => e.s && e.t);
  }

  layout(iterations = 260) {
    const n = this.nodes.length;
    if (n === 0) return;
    const area = 560 * 560;
    const k = Math.sqrt(area / n) * 0.9;
    const repulsionCutoff = k * 5; // ignore repulsion beyond this so a big
    // connected cluster can't fling sparsely-connected nodes arbitrarily
    // far away -- centering force then wins and pulls them back to a
    // reasonable orbit instead of off into empty space
    let temp = 50;

    for (let iter = 0; iter < iterations; iter++) {
      // repulsion
      for (let i = 0; i < n; i++) {
        const a = this.nodes[i];
        a.fx = 0; a.fy = 0;
      }
      for (let i = 0; i < n; i++) {
        const a = this.nodes[i];
        for (let j = i + 1; j < n; j++) {
          const b = this.nodes[j];
          let dx = a.x - b.x, dy = a.y - b.y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
          if (dist > repulsionCutoff) continue;
          const force = (k * k) / dist;
          dx /= dist; dy /= dist;
          a.fx += dx * force; a.fy += dy * force;
          b.fx -= dx * force; b.fy -= dy * force;
        }
      }
      // attraction along edges
      for (const e of this.edges) {
        let dx = e.s.x - e.t.x, dy = e.s.y - e.t.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const force = (dist * dist) / k * (0.5 + 0.5 * (e.weight || 0.5));
        dx /= dist; dy /= dist;
        e.s.fx -= dx * force; e.s.fy -= dy * force;
        e.t.fx += dx * force; e.t.fy += dy * force;
      }
      // centering pull (also keeps weakly-connected/isolated nodes from
      // flying off arbitrarily far under pure repulsion)
      for (const a of this.nodes) {
        a.fx -= a.x * 0.02;
        a.fy -= a.y * 0.02;
      }
      // apply
      for (const a of this.nodes) {
        if (a.fixed) continue;
        const disp = Math.sqrt(a.fx * a.fx + a.fy * a.fy) || 0.01;
        const capped = Math.min(disp, temp);
        a.x += (a.fx / disp) * capped;
        a.y += (a.fy / disp) * capped;
      }
      temp *= 0.97;
    }
    this._fitView();
  }

  _fitView() {
    if (this.nodes.length === 0) return;
    const xs = this.nodes.map((n) => n.x);
    const ys = this.nodes.map((n) => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const w = maxX - minX || 1, h = maxY - minY || 1;
    const scale = Math.min((this.width - 60) / w, (this.height - 60) / h, 1.4);
    this.scale = scale;
    this.offsetX = this.width / 2 - ((minX + maxX) / 2) * scale;
    this.offsetY = this.height / 2 - ((minY + maxY) / 2) * scale;
  }

  worldToScreen(x, y) {
    return [x * this.scale + this.offsetX, y * this.scale + this.offsetY];
  }
  screenToWorld(x, y) {
    return [(x - this.offsetX) / this.scale, (y - this.offsetY) / this.scale];
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    ctx.save();
    for (const e of this.edges) {
      const [sx, sy] = this.worldToScreen(e.s.x, e.s.y);
      const [tx, ty] = this.worldToScreen(e.t.x, e.t.y);
      ctx.beginPath();
      ctx.strokeStyle = e.color;
      ctx.globalAlpha = this.hoverNode && !(e.s === this.hoverNode || e.t === this.hoverNode) ? 0.12 : 0.45;
      ctx.lineWidth = Math.max(0.6, e.width * this.scale);
      ctx.setLineDash(e.dashed || []);
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      if (e.arrow) {
        const angle = Math.atan2(ty - sy, tx - sx);
        const rr = (this.hoverNode === e.t ? e.t.radius : e.t.radius) * this.scale + 2;
        const ax = tx - Math.cos(angle) * rr;
        const ay = ty - Math.sin(angle) * rr;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - 6 * Math.cos(angle - 0.4), ay - 6 * Math.sin(angle - 0.4));
        ctx.lineTo(ax - 6 * Math.cos(angle + 0.4), ay - 6 * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = e.color;
        ctx.fill();
      }
    }
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.restore();

    const showLabels = this.scale > 0.55;
    for (const nd of this.nodes) {
      const [x, y] = this.worldToScreen(nd.x, nd.y);
      const r = nd.radius * this.scale;
      const dimmed = this.hoverNode && this._neighborSet && !this._neighborSet.has(nd.id) && this.hoverNode !== nd;

      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, r), 0, Math.PI * 2);
      ctx.fillStyle = nd.fill;
      ctx.globalAlpha = dimmed ? 0.25 : 1;
      ctx.fill();
      ctx.lineWidth = nd.strokeWidth || 1.5;
      ctx.strokeStyle = nd.stroke;
      ctx.stroke();

      if (showLabels || this.hoverNode === nd) {
        ctx.font = "600 10px -apple-system, Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.lineWidth = 3;
        ctx.strokeStyle = "rgba(255,255,255,0.75)";
        ctx.strokeText(nd.label, x, y + r + 9);
        ctx.fillStyle = "#222";
        ctx.fillText(nd.label, x, y + r + 9);
      }
      ctx.globalAlpha = 1;
    }
  }

  _pickNode(px, py) {
    const [wx, wy] = this.screenToWorld(px, py);
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const nd = this.nodes[i];
      const dx = nd.x - wx, dy = nd.y - wy;
      if (Math.sqrt(dx * dx + dy * dy) <= nd.radius + 3) return nd;
    }
    return null;
  }

  _canvasPos(evt) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
  }

  _onWheel(evt) {
    evt.preventDefault();
    const pos = this._canvasPos(evt);
    const [wx, wy] = this.screenToWorld(pos.x, pos.y);
    const factor = evt.deltaY < 0 ? 1.1 : 0.9;
    this.scale = Math.max(0.15, Math.min(4, this.scale * factor));
    this.offsetX = pos.x - wx * this.scale;
    this.offsetY = pos.y - wy * this.scale;
    this.draw();
  }

  _onPointerDown(evt) {
    const pos = this._canvasPos(evt);
    const nd = this._pickNode(pos.x, pos.y);
    this.dragMoved = false;
    if (nd) {
      this.dragNode = nd;
      nd.fixed = true;
    } else {
      this.panning = true;
    }
    this.lastPointer = pos;
  }

  _onPointerMove(evt) {
    const rect = this.canvas.getBoundingClientRect();
    const pos = { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
    const inside = pos.x >= 0 && pos.y >= 0 && pos.x <= rect.width && pos.y <= rect.height;

    if (this.dragNode) {
      const [wx, wy] = this.screenToWorld(pos.x, pos.y);
      this.dragNode.x = wx;
      this.dragNode.y = wy;
      this.dragMoved = true;
      this.draw();
      return;
    }
    if (this.panning) {
      const dx = pos.x - this.lastPointer.x, dy = pos.y - this.lastPointer.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) this.dragMoved = true;
      this.offsetX += dx;
      this.offsetY += dy;
      this.lastPointer = pos;
      this.draw();
      return;
    }
    if (inside) {
      const nd = this._pickNode(pos.x, pos.y);
      if (nd !== this.hoverNode) {
        this.hoverNode = nd;
        this._neighborSet = nd
          ? new Set(this.edges.filter((e) => e.s === nd || e.t === nd).flatMap((e) => [e.s.id, e.t.id]))
          : null;
        this.draw();
      }
      if (nd && this.tooltipEl) {
        this.tooltipEl.classList.remove("hidden");
        this.tooltipEl.style.left = pos.x + "px";
        this.tooltipEl.style.top = pos.y + "px";
        this.tooltipEl.innerHTML = nd.title || nd.label;
      } else if (this.tooltipEl) {
        this.tooltipEl.classList.add("hidden");
      }
    }
  }

  _onPointerUp(evt) {
    if (this.dragNode) {
      const nd = this.dragNode;
      if (!this.dragMoved && this.onNodeClick) this.onNodeClick(nd.id);
      this.dragNode = null;
    } else if (this.panning && !this.dragMoved) {
      // plain click on empty space: no-op
    }
    this.panning = false;
  }

  focusNode(id, targetScale = 1.3) {
    const nd = this.nodeById.get(id);
    if (!nd) return;
    this.scale = Math.max(this.scale, targetScale);
    this.offsetX = this.width / 2 - nd.x * this.scale;
    this.offsetY = this.height / 2 - nd.y * this.scale;
    this.draw();
  }

  destroy() {
    window.removeEventListener("resize", this._resize);
    window.removeEventListener("mousemove", this._onPointerMove);
    window.removeEventListener("mouseup", this._onPointerUp);
  }
}
