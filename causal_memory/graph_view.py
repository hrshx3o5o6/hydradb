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
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HydraDNA · causal memory</title>
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

  #view { flex:1; position:relative; overflow:hidden; background:var(--bg);
          cursor:grab; touch-action:none; }
  #view.orbs { cursor:grabbing; }
  #cv { width:100%; height:100%; display:block; position:relative; z-index:1; }

  #tooltip { position:absolute; display:none; z-index:20; pointer-events:none;
             max-width:360px; background:#16171B; border:1px solid var(--line);
             border-left:2px solid var(--accent); padding:10px 14px;
             box-shadow:0 12px 32px rgba(0,0,0,.55); font-size:11px; }
  #tooltip .tt-h { color:var(--accent); letter-spacing:.18em; font-size:9px;
                   font-weight:700; margin-bottom:6px; text-transform:uppercase; }
  #tooltip .tt-b { color:var(--ink); line-height:1.5; }

  #trace { position:absolute; top:0; right:0; bottom:0; width:280px;
           z-index:10; background:rgba(22,23,27,.92); border-left:1px solid var(--line);
           display:flex; flex-direction:column; font-size:10px; overflow:hidden;
           pointer-events:none; }
  #trace .tr-h { flex:0 0 auto; padding:10px 14px; border-bottom:1px solid var(--line);
                 color:var(--accent); font-size:9px; letter-spacing:.2em; font-weight:700; }
  #trace .tr-h2 { flex:0 0 auto; color:var(--dim); }
  #trace .tr-h2 .tr-h2 { color:var(--faint); font-weight:400; }
  #trace .tr-l { flex:1; overflow-y:auto; padding:8px 10px; }
  #trace .tr-item { margin-bottom:10px; padding:8px 10px; background:var(--bg2);
                    border:1px solid var(--line); border-left:2px solid var(--accent); }
  #trace .tr-item.running { border-left-color:var(--green); }
  #trace .tr-top { display:flex; justify-content:space-between; gap:8px;
                   color:var(--ink); margin-bottom:6px; letter-spacing:.08em; }
  #trace .tr-top b { color:var(--accent); font-weight:700; text-transform:uppercase; }
  #trace .tr-top .ms { color:var(--faint); white-space:nowrap; }
  #trace .tr-q { color:var(--dim); margin-bottom:6px; line-height:1.4; word-break:break-word; }
  #trace .tr-s { display:flex; align-items:flex-start; gap:8px; position:relative;
                 padding:2px 0 2px 0; color:var(--dim); line-height:1.35; }
  #trace .tr-s::before { content:""; position:absolute; left:3px; top:14px; bottom:-4px;
                         width:1px; background:var(--line); }
  #trace .tr-s:last-child::before { display:none; }
  #trace .tr-s .dot { flex:0 0 7px; height:7px; margin-top:4px; border-radius:50%;
                      background:var(--accent); position:relative; z-index:1; }
  #trace .tr-s.running .dot { background:var(--green); animation:pulse 1s ease-in-out infinite; }
  #trace .tr-s .body { flex:1; min-width:0; }
  #trace .tr-s .body .name { color:var(--ink); letter-spacing:.06em; }
  #trace .tr-s .body .ms { color:var(--faint); margin-left:6px; }
  #trace .tr-s .body .det { margin-top:2px; color:var(--faint); font-size:9px;
                            word-break:break-word; }

  #trace .ev-item { margin-bottom:6px; padding:6px 10px; background:var(--bg2);
                    border:1px solid var(--line); border-left:2px solid var(--line); }
  #trace .ev-item.user { border-left-color:var(--accent); }
  #trace .ev-item.new { animation:evIn .5s ease-out both; }
  @keyframes evIn { from { background:rgba(255,87,25,.18); } to { background:var(--bg2); } }
  #trace .ev-top { display:flex; justify-content:space-between; gap:8px; margin-bottom:3px;
                   color:var(--faint); font-size:8px; letter-spacing:.1em; }
  #trace .ev-top b { color:var(--accent); }
  #trace .ev-item.user .ev-top b { color:var(--ink); }
  #trace .ev-text { color:var(--dim); font-size:9px; line-height:1.4; word-break:break-word; }
  #trace .ev-item.user .ev-text { color:var(--ink); }
  #trace .ev-text::before { content:"▸ "; color:var(--accent); }

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
    <div class="mark">HYDRA<small>DNA</small></div>
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
<div id="view"><canvas id="cv"></canvas></div>
<div id="trace">
  <div class="tr-h">LIVE <span class="tr-h2">EVENTS</span></div>
  <div class="tr-l" id="eventList"></div>
  <div class="tr-h tr-h2">RETRIEVAL <span class="tr-h2">TIMELINE</span></div>
  <div class="tr-l" id="traceList"></div>
</div>
<div id="tooltip"></div>
<footer>
  <span>HYDRADNA</span><span class="ok">●</span>
  <span>POLL&nbsp;1S</span><span id="clock">--:--:--</span>
</footer>
<script>
(function(){
  const cv = document.getElementById("cv");
  const view = document.getElementById("view");
  const ctx = cv.getContext("2d");
  const tick = { nodes:{}, edges:{} };   // id -> our state
  const textMap = {};                    // node id -> server node

  let W = 0, H = 0, DPR = 1;
  function resize(){
    DPR = window.devicePixelRatio || 1;
    W = cv.clientWidth  || 900; H = cv.clientHeight || 700;
    cv.width = W * DPR; cv.height = H * DPR;
    ctx.setTransform(DPR,0,0,DPR,0,0);
  }
  window.addEventListener("resize", resize);
  resize();

  // ---- camera: orbit around a target ----
  const FOV = 60 * Math.PI / 180;
  const cam = { yaw: 0.6, pitch: 0.5, dist: 460, tx: 0, ty: 0, tz: 0 };
  const NEAR = 1;
  const focal = () => (H/2) / Math.tan(FOV/2);

  const C = { f:[0,0,0], r:[1,0,0], u:[0,1,0] };  // camera basis
  function setBasis(){
    const cp = Math.cos(cam.pitch);
    const px = cam.tx + cam.dist * cp * Math.sin(cam.yaw);
    const py = cam.ty + cam.dist * Math.sin(cam.pitch);
    const pz = cam.tz + cam.dist * cp * Math.cos(cam.yaw);
    let fx = cam.tx - px, fy = cam.ty - py, fz = cam.tz - pz;
    const fl = Math.hypot(fx,fy,fz) || 1; fx/=fl; fy/=fl; fz/=fl;
    // right = normalize(cross(f, up(0,1,0)))
    let rx = fz, rz = -fx, rl = Math.hypot(rx,rz) || 1; rx/=rl; rz/=rl;
    // up = cross(right, f)
    const ux = rz*fy - 0*fz;   // = rz*fy
    const uy = fz*rx - fx*rz;
    const uz = fx*0 - rx*fy;   // = -rx*fy
    C.f = [fx,fy,fz]; C.r = [rx,0,rz]; C.u = [ux,uy,uz];
  }

  // returns {x,y,z} or null if behind camera
  function project(p){
    const fx=C.f[0], fy=C.f[1], fz=C.f[2];
    const rx=C.r[0], rz=C.r[2];
    const ux=C.u[0], uy=C.u[1], uz=C.u[2];
    const relx = p.x - (cam.tx + cam.dist*Math.cos(cam.pitch)*Math.sin(cam.yaw));
    const rely = p.y - (cam.ty + cam.dist*Math.sin(cam.pitch));
    const relz = p.z - (cam.tz + cam.dist*Math.cos(cam.pitch)*Math.cos(cam.yaw));
    const depth = relx*fx + rely*fy + relz*fz;
    if (depth < NEAR) return null;
    const sx = relx*rx + relz*rz;
    const sy = relx*ux + rely*uy + relz*uz;
    const f = focal();
    return { x: W/2 + sx*f/depth, y: H/2 - sy*f/depth, z: depth, s: f/depth };
  }

  // ---- static 3D layout: golden-angle spiral, outward with index ----
  let placed = 0;
  const GOLDEN = 2.39996323;
  function layoutPos(){
    const i = placed++;
    const R = 52 + 14 * Math.sqrt(i);
    const z = 1 - 2 * ((i * 0.618034) % 1);
    const a = i * GOLDEN;
    const rr = R * Math.sqrt(1 - z*z);
    return { x: rr * Math.cos(a), y: R * z * 0.9, z: rr * Math.sin(a) };
  }

  // ---- render state ----
  let hover = null, sel = null, drag = null, fitted = false;
  const EDGE_STYLE = {
    CAUSES:      { color:"#FF7A4D", width:1.4, dash:[4,3] },
    ENABLES:     { color:"#56585F", width:1,   dash:[] },
    OVERWRITES:  { color:"#56585F", width:1,   dash:[5,4] },
    CONFLICTS:   { color:"#B03F12", width:1,   dash:[2,5] },
  };
  const ORANGE = "#FF5719", DIM = "#8A8C93", FAINT = "#56585F";

  function render(){
    ctx.clearRect(0,0,W,H);
    setBasis();

    // edges, far-to-near by midpoint
    const now = performance.now();
    const eList = [];
    for (const e of Object.values(tick.edges)){
      const a = tick.nodes[e.source], b = tick.nodes[e.target];
      if (!a || !b) continue;
      const pa = project(a), pb = project(b);
      if (!pa || !pb) continue;
      eList.push({e, pa, pb, mid:(pa.z+pb.z)/2});
    }
    eList.sort((x,y)=>y.mid-x.mid);
    for (const it of eList){
      const st = EDGE_STYLE[it.e.type] || EDGE_STYLE.ENABLES;
      const age = (now - it.e.bornAt) / 1000;
      const alpha = Math.min(1, age / 0.5);
      ctx.beginPath();
      ctx.moveTo(it.pa.x, it.pa.y);
      ctx.lineTo(it.pb.x, it.pb.y);
      ctx.strokeStyle = st.color;
      ctx.globalAlpha = alpha * (it.e.type === "CAUSES" ? 0.7 : 0.55);
      ctx.lineWidth = st.width * Math.min(2, (it.pa.s + it.pb.s)/2);
      ctx.setLineDash(st.dash);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // nodes, far-to-near
    const nList = [];
    for (const id of Object.keys(tick.nodes)){
      const n = tick.nodes[id];
      const p = project(n);
      if (!p) continue;
      nList.push({n, p});
    }
    nList.sort((x,y)=>y.p.z-x.p.z);

    for (const it of nList){
      const n = it.n, p = it.p;
      const r = Math.max(1.6, Math.min(9, 7 * p.s / 20));
      const isSel = sel === n.id, isHov = hover === n.id;
      const born = (now - n.bornAt) / 1000;
      // connection highlight
      let conn = false;
      if (sel != null){
        for (const e of Object.values(tick.edges)){
          if (e.source === sel && e.target === n.id ||
              e.target === sel && e.source === n.id){ conn = true; break; }
        }
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, 6.2832);
      ctx.fillStyle = isSel ? ORANGE : (isHov ? "#2A1E14" : "#2C2E35");
      ctx.strokeStyle = isSel || isHov ? ORANGE : (conn ? ORANGE : FAINT);
      ctx.lineWidth = isSel ? 2 : (isHov ? 1.8 : 1.2);
      ctx.globalAlpha = (born < 2) ? 1 : 1;
      ctx.fill(); ctx.stroke();
      if (born < 2){
        // new-node ring fading out
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 3, 0, 6.2832);
        ctx.strokeStyle = ORANGE;
        ctx.lineWidth = 1.4;
        ctx.globalAlpha = Math.max(0, 1 - born/2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (isSel || isHov){
        ctx.font = "9px ui-monospace,Menlo,monospace";
        const label = shortLabel(n);
        ctx.fillStyle = isSel ? ORANGE : DIM;
        ctx.textAlign = "center";
        ctx.fillText(label, p.x, p.y + r + 11);
      }
    }
  }

  function shortLabel(n){
    const g = textMap[n.id];
    const t = (g && g.text) || ("#" + n.id);
    return t.length > 34 ? t.slice(0,34) + "…" : t;
  }

  // ---- interaction: orbit + zoom + hover/select ----
  view.addEventListener("wheel", (ev)=>{
    ev.preventDefault();
    cam.dist *= Math.exp(ev.deltaY * 0.0011);
    cam.dist = Math.max(40, Math.min(4000, cam.dist));
  }, {passive:false});

  view.addEventListener("mousedown", (ev)=>{
    if (ev.button !== 0) return;
    drag = { x: ev.clientX, y: ev.clientY, yaw: cam.yaw, pitch: cam.pitch };
    view.classList.add("orbs");
  });
  window.addEventListener("mousemove", (ev)=>{
    if (drag){
      cam.yaw   = drag.yaw   - (ev.clientX - drag.x) * 0.005;
      cam.pitch = Math.max(-1.4, Math.min(1.4,
                   drag.pitch + (ev.clientY - drag.y) * 0.005));
      return;
    }
    // hover pick
    const rect = cv.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    setBasis();
    let best = null, bestD = 14;
    for (const id of Object.keys(tick.nodes)){
      const p = project(tick.nodes[id]);
      if (!p) continue;
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestD){ bestD = d; best = id; }
    }
    hover = best;
    if (hover != null){
      const g = textMap[hover];
      const tt = document.getElementById("tooltip");
      if (g){
        const head = (g.session_id || "?") + (g.type ? " · " + g.type : "") +
          (g.topic ? " · " + g.topic : "");
        tt.innerHTML = '<div class="tt-h">' + head + "</div>" +
          '<div class="tt-b">' + g.text + "</div>";
        tt.style.display = "block";
        tt.style.left = (mx + 16) + "px";
        tt.style.top  = (my + 16) + "px";
      }
    } else if (!sel){
      document.getElementById("tooltip").style.display = "none";
    }
  });
  window.addEventListener("mouseup", (ev)=>{
    if (drag){
      const moved = Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y);
      if (moved < 5){
        // click = toggle selection
        const rect = cv.getBoundingClientRect();
        const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
        setBasis();
        let best = null, bestD = 14;
        for (const id of Object.keys(tick.nodes)){
          const p = project(tick.nodes[id]);
          if (!p) continue;
          const d = Math.hypot(p.x - mx, p.y - my);
          if (d < bestD){ bestD = d; best = id; }
        }
        sel = (best === sel) ? null : best;
      }
    }
    drag = null; view.classList.remove("orbs");
  });
  view.addEventListener("mouseleave", ()=>{
    if (!drag && !sel) document.getElementById("tooltip").style.display = "none";
  });
  view.addEventListener("dblclick", (ev)=>{
    // refocus on the middle of the graph
    const rect = cv.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    setBasis();
    // project target plane through origin
    cam.tx = cam.ty = cam.tz = 0;
    let x0=1e9,y0=1e9,z0=1e9,x1=-1e9,y1=-1e9,z1=-1e9;
    for (const n of Object.values(tick.nodes)){
      if (n.x<x0)x0=n.x; if (n.x>x1)x1=n.x;
      if (n.y<y0)y0=n.y; if (n.y>y1)y1=n.y;
      if (n.z<z0)z0=n.z; if (n.z>z1)z1=n.z;
    }
    if (x1 < 1e8){
      cam.tx = (x0+x1)/2; cam.ty = (y0+y1)/2; cam.tz = (z0+z1)/2;
      cam.dist = Math.hypot(x1-x0, y1-y0, z1-z0) * 1.6 + 60;
    }
  });

  function fitAll(){
    let rMax = 10;
    for (const n of Object.values(tick.nodes)){
      rMax = Math.max(rMax, Math.hypot(n.x, n.y, n.z));
    }
    cam.dist = Math.max(60, rMax * 2.1);
  }

  // ---- data ingest ----
  function renderData(data){
    document.getElementById("stats").textContent =
      data.nodes.length + " events · " + data.edges.length + " causal edges";
    const now = performance.now();
    for (const n of data.nodes){
      if (!tick.nodes[n.id]){
        const p = layoutPos();
        tick.nodes[n.id] = { id:n.id, x:p.x, y:p.y, z:p.z, bornAt:now };
      }
    }
    for (const e of data.edges){
      const k = e.source + "|" + e.target + "|" + e.type;
      if (!tick.edges[k]){
        tick.edges[k] = { source:e.source, target:e.target, type:e.type, bornAt:now };
      }
    }
    if (!fitted && Object.keys(tick.nodes).length > 1){
      fitted = true; fitAll();
    }
  }

  function clock(){
    const d = new Date();
    const p = (x)=>String(x).padStart(2,"0");
    document.getElementById("clock").textContent =
      p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  const fmtTime = (ms) => {
    const d = new Date(ms);
    const p = (x)=>String(x).padStart(2,"0");
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  };

  let lastTraceId = null;
  async function pollActivity(){
    let traces;
    try {
      const r = await fetch("/api/activity");
      traces = await r.json();
    } catch(e){ return; }
    if (!traces.length) return;
    const list = document.getElementById("traceList");
    const fresh = traces[0];
    if (lastTraceId === fresh.id) return;
    lastTraceId = fresh.id;
    let html = "";
    for (const tr of traces.slice(0, 6)){
      const st = tr.stages || [];
      html += '<div class="tr-item"><div class="tr-top"><b>' + tr.tool +
        '</b><span class="ms">' + fmtTime(tr.start) + ' · ' + tr.total_ms + 'ms</span></div>';
      if (tr.query){
        let q = tr.query;
        try { const obj = JSON.parse(q); q = obj.query || obj.text || obj.question || q; }
        catch(e){}
        html += '<div class="tr-q">' + q + '</div>';
      }
      for (const s of st){
        html += '<div class="tr-s"><span class="dot"></span><span class="body">' +
          '<span class="name">' + s.name + '</span><span class="ms">' + s.ms + 'ms</span>' +
          (s.detail ? '<div class="det">' + s.detail + '</div>' : '') +
          '</span></div>';
      }
      if (tr.result){
        html += '<div class="tr-q" style="margin-top:6px;color:var(--green)">→ ' +
          esc(tr.result) + '</div>';
      }
      html += '</div>';
    }
    list.innerHTML = html;
  }

  function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

  let lastEventId = null;
  async function pollEvents(){
    let evs;
    try {
      const r = await fetch("/api/events");
      evs = await r.json();
    } catch(e){ return; }
    if (!evs.length) return;
    const fresh = evs[0];
    if (lastEventId === fresh.id) return;
    lastEventId = fresh.id;
    const list = document.getElementById("eventList");
    let html = "";
    for (const ev of evs.slice(0, 14)){
      const isUser = ev.type === "user";
      const cls = isUser ? "ev-item user" : "ev-item";
      const t = ev.timestamp ? fmtTime(ev.timestamp * 1000) : "--:--:--";
      const txt = isUser ? (ev.text || "").slice(0, 140) : (ev.text || "").slice(0, 90);
      html += '<div class="' + cls + '"><div class="ev-top"><b>' +
        (isUser ? "YOU" : (ev.type || "event")) + '</b><span>' + t + '</span></div>' +
        '<div class="ev-text">' + esc(txt) + '</div></div>';
    }
    list.innerHTML = html;
  }

  async function poll(){
    try {
      const r = await fetch("/api/graph");
      const data = await r.json();
      for (const n of data.nodes) if (!textMap[n.id]) textMap[n.id] = n;
      renderData(data);
      document.getElementById("live").classList.remove("off");
      document.getElementById("live").lastChild.textContent = " LIVE";
    } catch(e){
      document.getElementById("live").classList.add("off");
      document.getElementById("live").lastChild.textContent = " OFFLINE";
    }
  }

  function loop(){ render(); requestAnimationFrame(loop); }

  clock(); setInterval(clock, 1000);
  setInterval(poll, 1000); poll();
  setInterval(pollActivity, 800); pollActivity();
  setInterval(pollEvents, 1000); pollEvents();
  requestAnimationFrame(loop);
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