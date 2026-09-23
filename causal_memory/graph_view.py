"""
Live causal-graph viewer for HydraDB.

Renders the causal graph (Events + CAUSES/ENABLES edges) as it grows, in real
time. Small stdlib-only HTTP server:

  - GET /            -> the viewer page (self-contained HTML, no CDN)
  - GET /api/graph   -> {nodes, edges} polled live from HydraDB

Run:
    source .venv/bin/activate
    HYDRADB_URL=http://localhost:8443 \
    HYDRADB_TOKEN=local-dev-auth-token-32-characters-long \
    python -m causal_memory.graph_view           # serves on :8080

Open http://localhost:8080. The page polls /api/graph every second and repaints
the SVG force-directed graph, animating newly added nodes/edges.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .tracing import recent_traces


class CausalGraphAPI:
    """Reads the causal graph out of HydraDB (pure reads, no LLM)."""

    def __init__(self):
        self.url = os.environ.get("HYDRADB_URL", "http://localhost:18443").rstrip("/")
        token = os.environ.get("HYDRADB_TOKEN", "local-dev-auth-token-32-characters-long")
        namespace = os.environ.get("HYDRADB_NAMESPACE", "default")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "X-Graph-Namespace": namespace,
            "Content-Type": "application/json",
        }

    def query(self, q: str):
        body = {"cell_id": os.environ.get("HYDRADB_CELL", "cell-0"), "query": q}
        resp = requests.post(f"{self.url}/v1/graphs/default/query",
                             json=body, headers=self.headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload.get("rows", [])

    def rows_to_dicts(self, rows, aliases):
        return [dict(zip(aliases, (c.get("value") for c in row))) for row in rows]

    def graph(self):
        nodes = self.rows_to_dicts(
            self.query(
                "MATCH (e:Event) "
                "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic "
                "LIMIT 500"
            ),
            ["id", "text", "timestamp", "session_id", "type", "topic"],
        )
        edges = []
        for rel_type in ("CAUSES", "ENABLES", "OVERWRITES", "CONFLICTS"):
            try:
                for e in self.rows_to_dicts(
                    self.query(
                        f"MATCH (a:Event)-[r:{rel_type}]->(b:Event) "
                        "RETURN a.id AS source, b.id AS target, "
                        "r.confidence AS confidence "
                        "LIMIT 2000",
                    ),
                    ["source", "target", "confidence"],
                ):
                    e["type"] = rel_type
                    edges.append(e)
            except RuntimeError:
                continue
        return {"nodes": nodes, "edges": edges}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HydraDNA · causal memory</title>
<style>
  /* ---- theme tokens: dark default, light via [data-theme=light] or OS pref ---- */
  :root {
    --bg:#141519; --surface:#1B1C21; --raised:#22242B;
    --ink:#F4F2ED; --ink-dim:#ABAEB6; --ink-faint:#6C6F78;
    --line:#2C2E36; --line-strong:#3B3E48;
    --accent:#FF6A34; --accent-ink:#141519; --accent-soft:rgba(255,106,52,.14);
    --focus:#7DD3FC;
    --ok:#34D399;
    --font-display:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --font-body:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    --font-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  [data-theme="light"] {
    --bg:#FAF8F4; --surface:#FFFFFF; --raised:#F1EEE6;
    --ink:#1C1D21; --ink-dim:#5B5D66; --ink-faint:#8A8D96;
    --line:#E4E0D6; --line-strong:#D2CDBF;
    --accent:#C7430D; --accent-ink:#FFFFFF; --accent-soft:rgba(199,67,13,.10);
    --focus:#0369A1;
    --ok:#0D9668;
  }
  @media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) {
      --bg:#FAF8F4; --surface:#FFFFFF; --raised:#F1EEE6;
      --ink:#1C1D21; --ink-dim:#5B5D66; --ink-faint:#8A8D96;
      --line:#E4E0D6; --line-strong:#D2CDBF;
      --accent:#C7430D; --accent-ink:#FFFFFF; --accent-soft:rgba(199,67,13,.10);
      --focus:#0369A1;
      --ok:#0D9668;
    }
  }

  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; }
  body { background:var(--bg); color:var(--ink);
         font:15px/1.55 var(--font-body);
         display:flex; flex-direction:column; overflow:hidden; }
  button { font:inherit; color:inherit; }
  :focus-visible { outline:2px solid var(--focus); outline-offset:2px; border-radius:2px; }

  .skip-link { position:absolute; left:12px; top:-48px; z-index:100;
               background:var(--accent); color:var(--accent-ink);
               padding:10px 16px; border-radius:4px; font-size:14px; font-weight:600;
               transition:top .15s ease; text-decoration:none; }
  .skip-link:focus { top:12px; }

  /* ---- header: quiet, one line ---- */
  header { flex:0 0 auto; display:flex; align-items:center; gap:16px;
           padding:10px 18px; background:var(--surface); border-bottom:1px solid var(--line);
           position:relative; z-index:5; }
  .brand { display:flex; align-items:baseline; gap:6px; font-family:var(--font-display);
           font-size:17px; font-weight:600; letter-spacing:.01em; white-space:nowrap; }
  .brand em { font-style:normal; color:var(--accent); }
  .stats { color:var(--ink-dim); font-size:13px; white-space:nowrap; }
  .status { display:flex; align-items:center; gap:6px; font-size:13px; color:var(--ink-dim);
            margin-left:auto; white-space:nowrap; }
  .status .dot { width:8px; height:8px; border-radius:50%; background:var(--ok); flex:0 0 auto; }
  .status.off .dot { background:var(--ink-faint); }
  @media (prefers-reduced-motion: no-preference) {
    .status:not(.off) .dot { animation:pulse 2s ease-in-out infinite; }
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

  .iconbtn { display:inline-flex; align-items:center; gap:6px; background:none;
             border:1px solid var(--line); border-radius:6px; padding:6px 10px;
             cursor:pointer; font-size:13px; color:var(--ink-dim); }
  .iconbtn:hover { border-color:var(--line-strong); color:var(--ink); }
  .iconbtn[aria-pressed="true"] { background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }

  /* ---- layout: graph + collapsible side panel ---- */
  .layout { flex:1; display:flex; min-height:0; position:relative; }
  #view { flex:1; position:relative; overflow:hidden; background:var(--bg); min-width:0; }
  #view svg { width:100%; height:100%; display:block; cursor:grab; touch-action:none; }
  #view svg.panning { cursor:grabbing; }

  .empty-state { position:absolute; inset:0; display:flex; align-items:center;
                 justify-content:center; text-align:center; padding:24px; color:var(--ink-faint);
                 font-size:14px; pointer-events:none; }
  .empty-state.hidden { display:none; }

  /* legend: text-first, not color-only */
  #legend { position:absolute; left:16px; bottom:16px; z-index:4;
            display:flex; flex-direction:column; gap:6px; background:var(--surface);
            border:1px solid var(--line); border-radius:8px; padding:10px 12px;
            font-size:12px; color:var(--ink-dim); }
  #legend .row { display:flex; align-items:center; gap:8px; }
  #legend .swatch { width:20px; height:2px; flex:0 0 auto; }
  #legend .row[data-t="CAUSES"] .swatch { background:var(--accent); }
  #legend .row[data-t="ENABLES"] .swatch { background:var(--ink-faint); }
  #legend .row[data-t="OVERWRITES"] .swatch { background:var(--ink-faint);
    background-image:repeating-linear-gradient(90deg, var(--ink-faint) 0 5px, transparent 5px 9px); height:0; border-top:2px dashed var(--ink-faint); background:none; }
  #legend .row[data-t="CONFLICTS"] .swatch { border-top:2px dotted var(--accent); background:none; }

  /* node detail panel: replaces mouse-only tooltip, keyboard reachable */
  #detail { position:absolute; right:16px; top:16px; z-index:6; width:min(320px, calc(100% - 32px));
            background:var(--surface); border:1px solid var(--line); border-radius:8px;
            box-shadow:0 8px 28px rgba(0,0,0,.25); padding:14px 16px; }
  #detail.hidden { display:none; }
  #detail .d-kicker { font-size:11px; letter-spacing:.06em; text-transform:uppercase;
                      color:var(--accent); font-weight:700; margin-bottom:6px; }
  #detail .d-text { font-size:14px; line-height:1.5; margin-bottom:10px; }
  #detail .d-meta { font-size:12px; color:var(--ink-faint); }
  #detail .d-close { position:absolute; top:10px; right:10px; background:none; border:none;
                     color:var(--ink-faint); cursor:pointer; font-size:16px; line-height:1; padding:4px; }
  #detail .d-close:hover { color:var(--ink); }

  /* ---- side panel: closed by default, toggled ---- */
  #panel { flex:0 0 340px; display:flex; flex-direction:column; background:var(--surface);
           border-left:1px solid var(--line); overflow:hidden; }
  #panel.closed { display:none; }
  .panel-tabs { flex:0 0 auto; display:flex; border-bottom:1px solid var(--line); }
  .panel-tabs button { flex:1; padding:11px 8px; background:none; border:none;
                       border-bottom:2px solid transparent; color:var(--ink-faint);
                       font-size:13px; font-weight:600; cursor:pointer; }
  .panel-tabs button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--accent); }
  .panel-body { flex:1; overflow-y:auto; padding:12px 14px; }
  .panel-body[hidden] { display:none; }

  .card { margin-bottom:10px; padding:10px 12px; background:var(--raised);
          border:1px solid var(--line); border-radius:6px; font-size:13px; }
  .card.user { border-left:3px solid var(--accent); }
  .card.reasoning { border-left:3px solid var(--ok); }
  .card-top { display:flex; justify-content:space-between; gap:8px; margin-bottom:4px;
              color:var(--ink-faint); font-size:11px; letter-spacing:.03em; }
  .card-top b { color:var(--accent); text-transform:uppercase; }
  .card.user .card-top b { color:var(--ink); }
  .card-text { color:var(--ink-dim); line-height:1.45; word-break:break-word; }
  .card.user .card-text { color:var(--ink); }
  button.card { display:block; width:100%; text-align:left; cursor:pointer; }
  .card-text.full { white-space:pre-wrap; }

  .trace-card .trace-top { display:flex; justify-content:space-between; gap:8px;
                           margin-bottom:6px; font-size:12px; }
  .trace-card .trace-top b { color:var(--accent); text-transform:uppercase; letter-spacing:.03em; }
  .trace-card .trace-top span { color:var(--ink-faint); }
  .trace-q { color:var(--ink-dim); font-size:12px; margin-bottom:8px; line-height:1.4; }
  .trace-step { display:flex; gap:8px; padding:3px 0; font-size:12px; color:var(--ink-dim); }
  .trace-step .step-dot { flex:0 0 6px; height:6px; margin-top:6px; border-radius:50%; background:var(--accent); }
  .trace-step .step-name { color:var(--ink); }
  .trace-step .step-ms { color:var(--ink-faint); margin-left:6px; }
  .trace-step .step-det { color:var(--ink-faint); font-size:11px; margin-top:1px; }
  .trace-result { margin-top:6px; font-size:12px; color:var(--ok); }

  .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
             overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }

  footer { flex:0 0 auto; display:flex; align-items:center; gap:16px;
           padding:8px 18px; background:var(--surface); border-top:1px solid var(--line);
           color:var(--ink-faint); font-size:12px; }
  footer .sep { color:var(--line-strong); }

  @media (max-width:760px) {
    #panel { position:absolute; right:0; top:0; bottom:0; width:min(340px,90vw); z-index:8; }
    #legend { display:none; }
  }
</style>
</head>
<body>
<a class="skip-link" href="#panel-toggle">Skip to activity panel</a>
<header>
  <div class="brand">HYDRA<em>DNA</em></div>
  <div class="stats" id="stats" aria-live="polite">Connecting…</div>
  <div class="status off" id="status" role="status" aria-live="polite">
    <span class="dot" aria-hidden="true"></span><span id="statusText">Offline</span>
  </div>
  <button class="iconbtn" id="themeToggle" type="button" aria-pressed="false">
    <span id="themeIcon" aria-hidden="true">&#9789;</span><span id="themeLabel">Light</span>
  </button>
  <button class="iconbtn" id="panel-toggle" type="button" aria-pressed="false"
          aria-controls="panel" aria-expanded="false">
    Activity
  </button>
</header>
<div class="layout">
  <div id="view">
    <svg id="graphSvg" role="application"
         aria-label="Causal graph. Use Tab to move between events, Enter to open details, arrow keys to pan, plus and minus to zoom."></svg>
    <div class="empty-state" id="emptyState">No events yet. Observations will appear here as they're recorded.</div>
    <div id="legend" aria-hidden="true">
      <div class="row" data-t="CAUSES"><span class="swatch"></span>Causes</div>
      <div class="row" data-t="ENABLES"><span class="swatch"></span>Enables</div>
      <div class="row" data-t="OVERWRITES"><span class="swatch"></span>Overwrites</div>
      <div class="row" data-t="CONFLICTS"><span class="swatch"></span>Conflicts</div>
    </div>
    <div id="detail" class="hidden" role="dialog" aria-label="Event detail" aria-modal="false">
      <button class="d-close" id="detailClose" type="button" aria-label="Close detail">&times;</button>
      <div class="d-kicker" id="detailKicker"></div>
      <div class="d-text" id="detailText"></div>
      <div class="d-meta" id="detailMeta"></div>
    </div>
  </div>
  <aside id="panel" class="closed" aria-label="Live activity">
    <div class="panel-tabs" role="tablist">
      <button role="tab" id="tab-events" aria-selected="true" aria-controls="pane-events" type="button">Events</button>
      <button role="tab" id="tab-trace" aria-selected="false" aria-controls="pane-trace" type="button">Retrieval</button>
    </div>
    <div class="panel-body" id="pane-events" role="tabpanel" aria-labelledby="tab-events" aria-live="polite"></div>
    <div class="panel-body" id="pane-trace" role="tabpanel" aria-labelledby="tab-trace" hidden></div>
  </aside>
</div>
<footer>
  <span>hydradna</span><span class="sep">&middot;</span>
  <span>polling every 1s</span><span class="sep">&middot;</span>
  <span id="clock">--:--:--</span>
</footer>
<div class="sr-only" id="liveAnnounce" aria-live="polite"></div>
<script>
(function(){
  "use strict";
  const svg = document.getElementById("graphSvg");
  const view = document.getElementById("view");
  const emptyState = document.getElementById("emptyState");
  const SVGNS = "http://www.w3.org/2000/svg";
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- theme toggle ----
  const themeToggle = document.getElementById("themeToggle");
  const themeLabel = document.getElementById("themeLabel");
  const themeIcon = document.getElementById("themeIcon");
  function applyTheme(t){
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
    const isLight = t === "light" ||
      (!t && window.matchMedia("(prefers-color-scheme: light)").matches);
    themeToggle.setAttribute("aria-pressed", String(isLight));
    themeLabel.textContent = isLight ? "Dark" : "Light";
    themeIcon.textContent = isLight ? "\\u263D" : "\\u263C";
  }
  let saved = null;
  try { saved = localStorage.getItem("hydradna-theme"); } catch(e){}
  applyTheme(saved);
  themeToggle.addEventListener("click", () => {
    const isLight = themeToggle.getAttribute("aria-pressed") === "true";
    const next = isLight ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem("hydradna-theme", next); } catch(e){}
  });

  // ---- side panel toggle + tabs ----
  const panel = document.getElementById("panel");
  const panelToggle = document.getElementById("panel-toggle");
  function setPanelOpen(open){
    panel.classList.toggle("closed", !open);
    panelToggle.setAttribute("aria-pressed", String(open));
    panelToggle.setAttribute("aria-expanded", String(open));
  }
  panelToggle.addEventListener("click", () => {
    setPanelOpen(panel.classList.contains("closed"));
  });
  const tabs = [
    { btn: document.getElementById("tab-events"), pane: document.getElementById("pane-events") },
    { btn: document.getElementById("tab-trace"), pane: document.getElementById("pane-trace") },
  ];
  tabs.forEach((t, i) => t.btn.addEventListener("click", () => {
    tabs.forEach((o, j) => {
      o.btn.setAttribute("aria-selected", String(i === j));
      o.pane.hidden = i !== j;
    });
    if (!open) setPanelOpen(true);
    var open = true;
  }));

  // ---- graph model ----
  const nodes = new Map();   // id -> {id, text, session_id, type, topic, x, y, vx, vy}
  const edges = [];          // {source, target, type}
  let selected = null, focusIndex = -1, order = [];

  const EDGE_CLASS = { CAUSES:"e-causes", ENABLES:"e-enables", OVERWRITES:"e-overwrites", CONFLICTS:"e-conflicts" };

  function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
  function shortLabel(n){
    const t = n.text || ("Event #" + n.id);
    return t.length > 40 ? t.slice(0,40) + "\\u2026" : t;
  }

  // ---- simple force layout (no deps): repel all, spring on edges, centre pull ----
  let W = 900, H = 600;
  function resize(){
    W = view.clientWidth || 900; H = view.clientHeight || 600;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  }
  window.addEventListener("resize", resize);

  function seedPos(n){
    const a = Math.random() * Math.PI * 2, r = 40 + Math.random() * 120;
    n.x = W/2 + Math.cos(a) * r; n.y = H/2 + Math.sin(a) * r;
    n.vx = 0; n.vy = 0;
  }

  function step(){
    const arr = Array.from(nodes.values());
    const n = arr.length;
    if (!n) return;
    const REPEL = 2600, SPRING = 0.02, LEN = 90, CENTER = 0.0045, DAMP = 0.82;
    for (let i=0;i<n;i++){
      let fx=0, fy=0;
      for (let j=0;j<n;j++){
        if (i===j) continue;
        let dx = arr[i].x - arr[j].x, dy = arr[i].y - arr[j].y;
        let d2 = dx*dx + dy*dy || 0.01;
        let f = REPEL / d2;
        const d = Math.sqrt(d2);
        fx += (dx/d) * f; fy += (dy/d) * f;
      }
      fx += (W/2 - arr[i].x) * CENTER;
      fy += (H/2 - arr[i].y) * CENTER;
      arr[i].fx = fx; arr[i].fy = fy;
    }
    for (const e of edges){
      const a = nodes.get(e.source), b = nodes.get(e.target);
      if (!a || !b) continue;
      let dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.hypot(dx,dy) || 0.01;
      const f = (d - LEN) * SPRING;
      const ux = dx/d, uy = dy/d;
      a.fx += ux*f; a.fy += uy*f;
      b.fx -= ux*f; b.fy -= uy*f;
    }
    for (const nd of arr){
      nd.vx = (nd.vx + nd.fx) * DAMP; nd.vy = (nd.vy + nd.fy) * DAMP;
      nd.x += nd.vx * 0.06; nd.y += nd.vy * 0.06;
      nd.x = Math.max(24, Math.min(W-24, nd.x));
      nd.y = Math.max(24, Math.min(H-24, nd.y));
    }
  }

  // ---- render (rebuild DOM only for structural changes; positions each tick) ----
  let edgeEls = new Map(), nodeEls = new Map();
  const gEdges = document.createElementNS(SVGNS, "g");
  const gNodes = document.createElementNS(SVGNS, "g");
  svg.appendChild(gEdges); svg.appendChild(gNodes);

  function rebuildIfNeeded(){
    if (edgeEls.size !== edges.length){
      gEdges.innerHTML = "";
      edgeEls = new Map();
      edges.forEach((e, i) => {
        const line = document.createElementNS(SVGNS, "line");
        line.setAttribute("class", "edge " + (EDGE_CLASS[e.type] || "e-enables"));
        gEdges.appendChild(line);
        edgeEls.set(i, line);
      });
    }
    if (nodeEls.size !== nodes.size){
      const seen = new Set(nodeEls.keys());
      order = Array.from(nodes.keys());
      for (const id of order){
        seen.delete(id);
        if (nodeEls.has(id)) continue;
        const g = document.createElementNS(SVGNS, "g");
        g.setAttribute("class", "node");
        g.setAttribute("tabindex", "0");
        g.setAttribute("role", "button");
        g.dataset.id = id;
        const circle = document.createElementNS(SVGNS, "circle");
        circle.setAttribute("r", "7");
        const label = document.createElementNS(SVGNS, "text");
        label.setAttribute("class", "node-label");
        label.setAttribute("text-anchor", "middle");
        g.appendChild(circle); g.appendChild(label);
        g.addEventListener("click", () => selectNode(id));
        g.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter" || ev.key === " "){ ev.preventDefault(); selectNode(id); }
        });
        g.addEventListener("focus", () => { focusIndex = order.indexOf(id); highlight(id); });
        gNodes.appendChild(g);
        nodeEls.set(id, g);
        const nd = nodes.get(id);
        nd.ariaLabel = shortLabel(nd) + (nd.type ? ", " + nd.type : "") + (nd.topic ? ", topic " + nd.topic : "");
        g.setAttribute("aria-label", nd.ariaLabel);
      }
      for (const id of seen){
        const el = nodeEls.get(id);
        if (el) el.remove();
        nodeEls.delete(id);
      }
    }
    emptyState.classList.toggle("hidden", nodes.size > 0);
  }

  function highlight(id){
    const connected = new Set();
    edges.forEach(e => {
      if (e.source === id) connected.add(e.target);
      if (e.target === id) connected.add(e.source);
    });
    nodeEls.forEach((el, nid) => {
      el.classList.toggle("connected", connected.has(nid));
      el.classList.toggle("dimmed", id != null && nid !== id && !connected.has(nid));
    });
    edgeEls.forEach((el, i) => {
      const e = edges[i];
      const rel = e.source === id || e.target === id;
      el.classList.toggle("dimmed", id != null && !rel);
    });
  }

  function paint(){
    edges.forEach((e, i) => {
      const a = nodes.get(e.source), b = nodes.get(e.target);
      const el = edgeEls.get(i);
      if (!a || !b || !el) return;
      el.setAttribute("x1", a.x); el.setAttribute("y1", a.y);
      el.setAttribute("x2", b.x); el.setAttribute("y2", b.y);
    });
    nodeEls.forEach((el, id) => {
      const nd = nodes.get(id);
      if (!nd) return;
      el.setAttribute("transform", "translate(" + nd.x + "," + nd.y + ")");
      const label = el.querySelector(".node-label");
      const isFocused = document.activeElement === el;
      const isSel = selected === id;
      label.style.opacity = (isFocused || isSel) ? "1" : "0";
      label.textContent = shortLabel(nd);
      label.setAttribute("y", "20");
      el.classList.toggle("selected", isSel);
    });
  }

  function selectNode(id){
    selected = selected === id ? null : id;
    highlight(selected);
    showDetail(selected);
  }

  const detail = document.getElementById("detail");
  const detailKicker = document.getElementById("detailKicker");
  const detailText = document.getElementById("detailText");
  const detailMeta = document.getElementById("detailMeta");
  document.getElementById("detailClose").addEventListener("click", () => { selected = null; highlight(null); showDetail(null); });
  function showDetail(id){
    if (id == null){ detail.classList.add("hidden"); return; }
    const nd = nodes.get(id);
    if (!nd) return;
    detailKicker.textContent = (nd.type || "event") + (nd.topic ? " \\u00b7 " + nd.topic : "");
    detailText.textContent = nd.text || ("Event #" + nd.id);
    detailMeta.textContent = "session " + (nd.session_id || "\\u2014");
    detail.classList.remove("hidden");
  }

  // ---- pan/zoom on the SVG (mouse, wheel; keyboard arrows on the graph itself) ----
  let panX = 0, panY = 0, zoom = 1, dragging = null;
  function applyViewTransform(){
    gEdges.setAttribute("transform", "translate(" + panX + "," + panY + ") scale(" + zoom + ")");
    gNodes.setAttribute("transform", "translate(" + panX + "," + panY + ") scale(" + zoom + ")");
  }
  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    zoom = Math.max(0.3, Math.min(3, zoom * Math.exp(-ev.deltaY * 0.001)));
    applyViewTransform();
  }, {passive:false});
  svg.addEventListener("mousedown", (ev) => {
    if (ev.target.closest(".node")) return;
    dragging = { x: ev.clientX, y: ev.clientY, panX, panY };
    svg.classList.add("panning");
  });
  window.addEventListener("mousemove", (ev) => {
    if (!dragging) return;
    panX = dragging.panX + (ev.clientX - dragging.x);
    panY = dragging.panY + (ev.clientY - dragging.y);
    applyViewTransform();
  });
  window.addEventListener("mouseup", () => { dragging = null; svg.classList.remove("panning"); });
  svg.addEventListener("keydown", (ev) => {
    const STEP = 30;
    if (ev.key === "ArrowLeft"){ panX += STEP; applyViewTransform(); }
    else if (ev.key === "ArrowRight"){ panX -= STEP; applyViewTransform(); }
    else if (ev.key === "ArrowUp"){ panY += STEP; applyViewTransform(); }
    else if (ev.key === "ArrowDown"){ panY -= STEP; applyViewTransform(); }
    else if (ev.key === "+" || ev.key === "="){ zoom = Math.min(3, zoom*1.15); applyViewTransform(); }
    else if (ev.key === "-"){ zoom = Math.max(0.3, zoom*0.87); applyViewTransform(); }
    else return;
    ev.preventDefault();
  });

  // ---- data polling ----
  const stats = document.getElementById("stats");
  const statusEl = document.getElementById("status");
  const statusText = document.getElementById("statusText");

  async function pollGraph(){
    let data;
    try {
      const r = await fetch("/api/graph");
      data = await r.json();
    } catch(e){
      statusEl.classList.add("off"); statusText.textContent = "Offline";
      return;
    }
    statusEl.classList.remove("off"); statusText.textContent = "Live";
    stats.textContent = data.nodes.length + " event" + (data.nodes.length===1?"":"s") +
      " \\u00b7 " + data.edges.length + " causal edge" + (data.edges.length===1?"":"s");
    for (const n of data.nodes){
      const existing = nodes.get(n.id);
      if (existing){
        Object.assign(existing, n);
      } else {
        const nd = Object.assign({}, n);
        seedPos(nd);
        nodes.set(n.id, nd);
      }
    }
    edges.length = 0;
    for (const e of data.edges) edges.push(e);
    rebuildIfNeeded();
  }

  function loop(){
    if (!prefersReducedMotion) step();
    paint();
    requestAnimationFrame(loop);
  }

  // ---- side panels: events + retrieval trace ----
  const fmtTime = (ms) => {
    const d = new Date(ms);
    const p = (x)=>String(x).padStart(2,"0");
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  };

  let lastTraceId = null;
  async function pollActivity(){
    let traces;
    try { traces = await (await fetch("/api/activity")).json(); } catch(e){ return; }
    if (!traces.length) return;
    const fresh = traces[0];
    if (lastTraceId === fresh.id) return;
    lastTraceId = fresh.id;
    const pane = document.getElementById("pane-trace");
    let html = "";
    for (const tr of traces.slice(0, 6)){
      html += '<div class="card trace-card"><div class="trace-top"><b>' + esc(tr.tool) +
        '</b><span>' + fmtTime(tr.start) + ' \\u00b7 ' + tr.total_ms + 'ms</span></div>';
      if (tr.query){
        let q = tr.query;
        try { const obj = JSON.parse(q); q = obj.query || obj.text || obj.question || q; } catch(e){}
        html += '<div class="trace-q">' + esc(q) + '</div>';
      }
      for (const s of (tr.stages || [])){
        html += '<div class="trace-step"><span class="step-dot" aria-hidden="true"></span><span>' +
          '<span class="step-name">' + esc(s.name) + '</span><span class="step-ms">' + s.ms + 'ms</span>' +
          (s.detail ? '<div class="step-det">' + esc(s.detail) + '</div>' : '') + '</span></div>';
      }
      if (tr.result) html += '<div class="trace-result">\\u2192 ' + esc(tr.result) + '</div>';
      html += '</div>';
    }
    pane.innerHTML = html;
  }

  let lastEventId = null, lastEvents = [];
  const expanded = new Set();
  function renderEvents(evs){
    const pane = document.getElementById("pane-events");
    let html = "";
    for (const ev of evs.slice(0, 20)){
      const isUser = ev.type === "user", isReasoning = ev.type === "reasoning";
      let cls = "card" + (isUser ? " user" : (isReasoning ? " reasoning" : ""));
      const t = ev.timestamp ? fmtTime(ev.timestamp * 1000) : "--:--:--";
      const full = ev.text || "";
      const isOpen = expanded.has(ev.id);
      const short = full.length > 90 ? full.slice(0,90) + "\\u2026" : full;
      html += '<button type="button" class="' + cls + '" data-id="' + ev.id + '" aria-expanded="' + isOpen + '">' +
        '<div class="card-top"><b>' + esc(isUser ? "you" : (isReasoning ? "thought" : (ev.type || "event"))) +
        '</b><span>' + t + '</span></div>' +
        '<div class="card-text' + (isOpen ? ' full' : '') + '">' + esc(isOpen ? full : short) + '</div></button>';
    }
    pane.innerHTML = html;
  }
  async function pollEvents(){
    let evs;
    try { evs = await (await fetch("/api/events")).json(); } catch(e){ return; }
    if (!evs.length) return;
    lastEvents = evs;
    const fresh = evs[0];
    if (lastEventId === fresh.id) return;
    lastEventId = fresh.id;
    renderEvents(evs);
  }
  document.getElementById("pane-events").addEventListener("click", (e) => {
    const item = e.target.closest(".card");
    if (!item) return;
    const id = Number(item.dataset.id);
    if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
    if (lastEvents.length) renderEvents(lastEvents);
  });

  function clock(){
    const d = new Date();
    const p = (x)=>String(x).padStart(2,"0");
    document.getElementById("clock").textContent =
      p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  resize();
  clock(); setInterval(clock, 1000);
  setInterval(pollGraph, 1000); pollGraph();
  setInterval(pollActivity, 800); pollActivity();
  setInterval(pollEvents, 1000); pollEvents();
  requestAnimationFrame(loop);
})();
</script>
<style>
  .edge { stroke:var(--ink-faint); stroke-width:1.2; opacity:.55; transition:opacity .15s; }
  .edge.e-causes { stroke:var(--accent); stroke-width:1.6; opacity:.75; }
  .edge.e-overwrites { stroke-dasharray:5,4; }
  .edge.e-conflicts { stroke:var(--accent); stroke-dasharray:2,4; opacity:.5; }
  .edge.dimmed { opacity:.08; }
  .node circle { fill:var(--raised); stroke:var(--ink-faint); stroke-width:1.4; cursor:pointer; }
  .node:hover circle, .node:focus circle { stroke:var(--accent); fill:var(--accent-soft); }
  .node.selected circle { fill:var(--accent); stroke:var(--accent); }
  .node.connected circle { stroke:var(--accent); }
  .node.dimmed { opacity:.25; }
  .node-label { fill:var(--ink-dim); font:12px/1 var(--font-body); pointer-events:none; opacity:0; transition:opacity .1s; }
  .node.selected .node-label { fill:var(--accent); }
</style>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    api = None

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/graph":
            try:
                g = self.api.graph()
                self._send(200, json.dumps(g).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode())
        elif self.path == "/api/activity":
            try:
                self._send(200, json.dumps(recent_traces()).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode())
        elif self.path == "/api/events":
            try:
                rows = self.api.query(
                    "MATCH (e:Event) "
                    "RETURN e.id, e.text, e.timestamp, e.type, e.topic "
                    "ORDER BY e.timestamp DESC LIMIT 60"
                )
                evs = [dict(zip(["id", "text", "timestamp", "type", "topic"],
                                (c.get("value") for c in row))) for row in rows]
                self._send(200, json.dumps(evs).encode())
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode())
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write("[graph_view] %s\n" % (fmt % args))


def main():
    port = int(os.environ.get("GRAPH_VIEW_PORT", "8080"))
    Handler.api = CausalGraphAPI()
    try:
        Handler.api.graph()
    except Exception as e:
        print("HydraDB unreachable at %s: %s" % (Handler.api.url, e))
        print("Start it first, or set HYDRADB_URL.")
        sys.exit(1)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("Causal-graph viewer: http://localhost:%d  (HydraDB %s)"
          % (port, Handler.api.url))
    srv.serve_forever()


if __name__ == "__main__":
    main()