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
                edges += self.rows_to_dicts(
                    self.query(
                        f"MATCH (a:Event)-[r:{rel_type}]->(b:Event) "
                        "RETURN a.id AS source, b.id AS target, "
                        "r.confidence AS confidence "
                        "LIMIT 2000",
                    ),
                    ["source", "target", "confidence"],
                )
            except RuntimeError:
                continue
        for e in edges:
            e["type"] = rel_type
        return {"nodes": nodes, "edges": edges}


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HydraDB · causal memory</title>
<style>
  :root {
    --bg:#1C1D21; --bg2:#22242A; --panel:#16171B;
    --ink:#F2F0EA; --dim:#8A8C93; --faint:#56585F;
    --accent:#FF5719; --accent-dim:#B03F12;
    --line:#2A2C33;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html,body { height:100%; }
  body { background:var(--bg); color:var(--ink);
         font:12px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
         display:flex; flex-direction:column; overflow:hidden; }

  /* HydraDB signature: dot-grid + vignette over near-black */
  #view::before {
    content:""; position:absolute; inset:0; z-index:0; pointer-events:none;
    background-image:radial-gradient(rgba(255,255,255,.045) 1px, transparent 1px);
    background-size:22px 22px;
  }
  #view::after {
    content:""; position:absolute; inset:0; z-index:0; pointer-events:none;
    background:radial-gradient(120% 90% at 50% 40%, transparent 55%, rgba(0,0,0,.5));
  }
  #view svg { position:relative; z-index:1; }

  /* pixel-cut corners (hydradb.com pixel-corner motif) */
  .cut { clip-path:polygon(0 0, calc(100% - 10px) 0, 100% 10px,
          100% 100%, 10px 100%, 0 calc(100% - 10px)); }

  header { flex:0 0 auto; display:flex; align-items:stretch; gap:0;
           background:var(--panel); border-bottom:1px solid var(--line);
           position:relative; z-index:5; }
  .brand { display:flex; align-items:center; gap:12px; padding:0 18px;
           background:var(--accent); color:#1C1D21; }
  .brand .mark { font-size:17px; font-weight:800; letter-spacing:.12em;
                 line-height:1; }
  .brand .mark small { display:block; font-size:8px; font-weight:700;
                       letter-spacing:.3em; opacity:.7; }
  .mid { flex:1; display:flex; align-items:center; gap:18px; padding:0 18px;
         min-width:0; }
  .mid h1 { font-size:11px; font-weight:700; letter-spacing:.28em;
            white-space:nowrap; }
  .stats { color:var(--dim); letter-spacing:.06em; white-space:nowrap; }
  legend { display:flex; gap:16px; color:var(--dim); font-size:10px;
           letter-spacing:.14em; margin-left:auto; white-space:nowrap; }
  legend i { font-style:normal; display:inline-flex; align-items:center; gap:6px; }
  legend i::before { content:""; width:18px; height:0; border-top:2px solid currentColor; }
  legend i.c { color:var(--accent); }
  legend i.c::before { border-top-style:solid; }
  legend i.e::before { border-top-color:var(--faint); }
  legend i.o::before { border-top-style:dashed; border-top-color:var(--faint); }
  legend i.x::before { border-top-style:dotted; border-top-color:var(--accent-dim); }
  .live { align-self:stretch; display:flex; align-items:center; gap:8px;
          padding:0 18px; border-left:1px solid var(--line);
          font-size:10px; letter-spacing:.24em; color:var(--dim);
          background:var(--bg); white-space:nowrap; }
  .live .dot { width:7px; height:7px; background:var(--accent);
               animation:pulse 1.6s ease-in-out infinite; }
  .live.off .dot { background:var(--faint); animation:none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

  #view { flex:1; position:relative; overflow:hidden; background:var(--bg); }
  svg { width:100%; height:100%; display:block; cursor:grab; touch-action:none; }
  svg.panning { cursor:grabbing; }

  .edge { stroke:var(--faint); stroke-width:1; opacity:.5; transition:opacity .2s; }
  .edge.edge-CAUSES { stroke:var(--accent); stroke-width:1.4; opacity:.85; }
  .edge.edge-ENABLES { stroke:var(--faint); stroke-width:1; }
  .edge.edge-OVERWRITES { stroke:var(--faint); stroke-width:1; stroke-dasharray:5 4; }
  .edge.edge-CONFLICTS { stroke:var(--accent-dim); stroke-width:1;
                         stroke-dasharray:2 5; opacity:.7; }
  .edge.new { animation:drawIn .7s ease-out both; }
  @keyframes drawIn { from { opacity:0; stroke-width:3 } to { opacity:.85;
    stroke-width:inherit } }
  .edge:hover { opacity:1; stroke-width:2; }

  .node { cursor:move; }
  .node circle { fill:var(--bg2); stroke:var(--faint); stroke-width:1.4; }
  .node:hover circle { fill:var(--accent); stroke:var(--accent); }
  .node .lab { fill:var(--dim); font-size:8px; letter-spacing:.05em;
               text-anchor:middle; }
  .node:hover .lab { fill:var(--ink); }
  .node.new { animation:popIn .5s ease-out both; }
  .node.new circle { stroke:var(--accent); fill:#26100A; }
  @keyframes popIn { from { opacity:0 } to { opacity:1 } }

  #tooltip { position:absolute; display:none; z-index:20; pointer-events:none;
             max-width:360px; background:#16171B; border:1px solid var(--line);
             border-left:2px solid var(--accent); padding:10px 14px;
             box-shadow:0 12px 32px rgba(0,0,0,.55); font-size:11px; }
  #tooltip .tt-h { color:var(--accent); letter-spacing:.18em; font-size:9px;
                   font-weight:700; margin-bottom:6px; text-transform:uppercase; }
  #tooltip .tt-b { color:var(--ink); line-height:1.5; }

  footer { flex:0 0 auto; display:flex; align-items:center; gap:20px;
           padding:6px 18px; background:var(--panel); border-top:1px solid var(--line);
           color:var(--faint); font-size:9px; letter-spacing:.2em;
           position:relative; z-index:5; }
  footer .ok { color:var(--accent); }
</style>
</head>
<body>
<header>
  <div class="brand cut">
    <div class="mark">HYDRA<small>GRAPH DB</small></div>
  </div>
  <div class="mid">
    <h1>CAUSAL&nbsp;MEMORY</h1>
    <div class="stats" id="stats">CONNECTING…</div>
    <legend>
      <i class="c">CAUSES</i><i class="e">ENABLES</i>
      <i class="o">OVERWRITES</i><i class="x">CONFLICTS</i>
    </legend>
  </div>
  <div class="live" id="live"><span class="dot"></span>LIVE</div>
</header>
<div id="view"><svg id="svg"></svg></div>
<div id="tooltip"></div>
<footer>
  <span>HYDRADB</span><span class="ok">●</span>
  <span>POLL&nbsp;1S</span><span id="clock">--:--:--</span>
</footer>
<script>
(function(){
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.getElementById("svg");
  const view = document.getElementById("view");
  const W = () => svg.clientWidth || 900, H = () => svg.clientHeight || 700;
  const tick = { nodes:{}, edges:{} };   // id -> our anim state
  const textMap = {};                    // node id -> server node (label data)

  const gRoot = el("g",{class:"mains"});
  svg.appendChild(gRoot);
  let cam = {x:0, y:0, k:1};             // pan + zoom
  let drag = null;
  let fitted = false;

  function el(name, attrs){
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function applyCam(){
    gRoot.setAttribute("transform",
      "translate(" + cam.x + "," + cam.y + ") scale(" + cam.k + ")");
    for (const l of labels) l.setAttribute("opacity", cam.k < .5 ? "0" : "1");
  }

  function fitAll(){
    const ids = Object.keys(tick.nodes);
    if (ids.length < 2) return;
    let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
    for (const id of ids){
      const n = tick.nodes[id];
      if (n.x<x0)x0=n.x; if (n.x>x1)x1=n.x;
      if (n.y<y0)y0=n.y; if (n.y>y1)y1=n.y;
    }
    const w = (x1-x0)||1, h = (y1-y0)||1;
    cam.k = Math.max(.2, Math.min(1.5, Math.min(W()/w, H()/h)*.85));
    cam.x = W()/2 - (x0+x1)/2*cam.k;
    cam.y = H()/2 - (y0+y1)/2*cam.k;
    applyCam();
  }

  view.addEventListener("wheel", (ev)=>{
    ev.preventDefault();
    const rect = svg.getBoundingClientRect();
    const px = ev.clientX-rect.left, py = ev.clientY-rect.top;
    const k2 = cam.k * Math.exp(-ev.deltaY * .0012);
    cam.k = Math.max(.1, Math.min(8, k2));
    cam.x = px - (px - cam.x) * (cam.k / (cam.k*Math.exp(-ev.deltaY*.0012)));
    cam.y = py - (py - cam.y) * (cam.k / (cam.k*Math.exp(-ev.deltaY*.0012)));
    applyCam();
  }, {passive:false});

  view.addEventListener("mousedown", (ev)=>{
    const node = ev.target.closest ? ev.target.closest(".node") : null;
    if (node && node.__id != null){
      drag = {mode:"node", id:node.__id, sx:ev.clientX, sy:ev.clientY,
              ox:tick.nodes[node.__id].x, oy:tick.nodes[node.__id].y};
      ev.preventDefault();
    } else {
      drag = {mode:"pan", sx:ev.clientX, sy:ev.clientY, ox:cam.x, oy:cam.y};
      svg.classList.add("panning");
      ev.preventDefault();
    }
  });
  window.addEventListener("mousemove", (ev)=>{
    if (!drag) return;
    const dx = ev.clientX - drag.sx, dy = ev.clientY - drag.sy;
    if (drag.mode === "pan"){
      cam.x = drag.ox + dx; cam.y = drag.oy + dy;
      applyCam();
    } else {
      const n = tick.nodes[drag.id];
      if (n){
        n.x = drag.ox + dx/cam.k; n.y = drag.oy + dy/cam.k;
        n.vx = 0; n.vy = 0;
        n._n.setAttribute("transform","translate("+n.x+","+n.y+")");
        for (const e of Object.values(tick.edges)){
          if (e.source!==drag.id && e.target!==drag.id) continue;
          const a = tick.nodes[e.source], b = tick.nodes[e.target];
          if (!a || !b) continue;
          e._n.setAttribute("x1",a.x); e._n.setAttribute("y1",a.y);
          e._n.setAttribute("x2",b.x); e._n.setAttribute("y2",b.y);
        }
      }
    }
  });
  window.addEventListener("mouseup", ()=>{
    drag = null; svg.classList.remove("panning");
  });

  // static sunflower layout: nodes are placed once and never moved again.
  let placed = 0;
  const labels = [];
  const GOLDEN = 2.39996323;      // golden angle (radians)
  function layoutPos(){
    const i = placed++;
    const rad = 30 * Math.sqrt(i);
    return { x: W()/2 + rad*Math.cos(i*GOLDEN),
             y: H()/2 + rad*Math.sin(i*GOLDEN) };
  }

  function render(data){
    const nodes = data.nodes, edges = data.edges;
    document.getElementById("stats").textContent =
      nodes.length + " events · " + edges.length + " causal edges";

    for (const n of nodes){
      if (!tick.nodes[n.id]){
        const p = layoutPos();
        tick.nodes[n.id] = {
          id:n.id, x:p.x, y:p.y, vx:0, vy:0,
          _n: el("g",{class:"node new"}),
        };
      }
    }
    for (const e of edges){
      if (!tick.edges[e.source+"|"+e.target+"|"+e.type]){
        tick.edges[e.source+"|"+e.target+"|"+e.type] = {
          source:e.source, target:e.target, type:e.type,
          _n: el("line",{class:"edge edge-"+e.type+" new", x1:0, y1:0, x2:0, y2:0}),
        };
      }
    }

    for (const e of Object.values(tick.edges)){
      const a = tick.nodes[e.source], b = tick.nodes[e.target];
      if (!a || !b) continue;
      e._n.setAttribute("x1", a.x); e._n.setAttribute("y1", a.y);
      e._n.setAttribute("x2", b.x); e._n.setAttribute("y2", b.y);
      if (!e._n.parentNode) gRoot.appendChild(e._n);
    }
    for (const n of Object.values(tick.nodes)){
      const nn = n._n;
      nn.setAttribute("transform", "translate(" + n.x + "," + n.y + ")");
      if (!nn.parentNode) gRoot.appendChild(nn);
      nn.__id = n.id;
      if (nn._done) continue;
      nn._done = true;
      nn.appendChild(el("circle", {r:8}));
      const g0 = textMap[n.id];
      const text = (g0 && g0.text) || ("#" + n.id);
      const short = text.length > 30 ? text.slice(0,30) + "…" : text;
      const t1 = el("text", {class:"lab", x:0, y:24}); t1.textContent = short;
      nn.appendChild(t1);
      labels.push(t1);
    }
    if (!fitted && Object.keys(tick.nodes).length > 1){
      fitted = true; fitAll();
    }
  }

  function wireTooltips(data){
    const tt = document.getElementById("tooltip");
    view.onmousemove = (ev)=>{
      const g = document.elementFromPoint(ev.clientX, ev.clientY);
      const gEl = g && g.closest ? g.closest(".node") : null;
      if (!gEl){ tt.style.display = "none"; return; }
      const id = gEl.__id;
      if (id == null) return;
      const n = textMap[id];
      if (n){
        const head = (n.session_id || "?") + (n.type ? " · " + n.type : "") +
          (n.topic ? " · " + n.topic : "");
        const t = '<div class="tt-h">' + head + "</div>" +
          '<div class="tt-b">' + n.text + "</div>";
        tt.innerHTML = t;
        tt.style.display = "block";
        tt.style.left = (ev.clientX + 14) + "px";
        tt.style.top  = (ev.clientY + 14) + "px";
      }
    };
  }

  function clock(){
    const d = new Date();
    const p = (x)=>String(x).padStart(2,"0");
    document.getElementById("clock").textContent =
      p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  async function poll(){
    try {
      const r = await fetch("/api/graph");
      const data = await r.json();
      for (const n of data.nodes) if (!textMap[n.id]) textMap[n.id] = n;
      render(data); wireTooltips(data);
      document.getElementById("live").classList.remove("off");
      document.getElementById("live").lastChild.textContent = " LIVE";
    } catch(e){
      document.getElementById("live").classList.add("off");
      document.getElementById("live").lastChild.textContent = " OFFLINE";
    }
  }

  clock(); setInterval(clock, 1000);
  setInterval(poll, 1000); poll();
})();
</script>
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