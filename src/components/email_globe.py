"""
components/email_globe.py — Globo 3D interattivo del percorso email.

Visualizza la catena Received hop-by-hop su un mappamondo 3D rotante
costruito con D3.js (geoOrthographic) senza dipendenze Python aggiuntive.
Il globo ruota automaticamente, si può trascinare con il mouse e mostra
archi animati tra gli hop con popup dettagliati.

Utilizzo in app.py:
    from src.components.email_globe import render_email_globe

    with st.expander("🌍 Percorso geografico email", expanded=True):
        render_email_globe(soc, validator)
"""

import ipaddress
import json
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import streamlit.components.v1 as components


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_geolocatable_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address((ip or "").strip("[]")).is_global
    except ValueError:
        return False


def _score_to_risk(score) -> str:
    if score is None:
        return "unknown"
    s = int(score)
    if s >= 75:  return "high"
    if s >= 25:  return "medium"
    return "low"


def _risk_color(risk: str) -> str:
    return {
        "high":    "#E24B4A",
        "medium":  "#EF9F27",
        "low":     "#1D9E75",
        "unknown": "#888780",
    }[risk]


def _geo_coords(geo: dict):
    if geo.get("status") != "ok":
        return None
    lat, lon = geo.get("lat"), geo.get("lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


# ── HTML del globo ─────────────────────────────────────────────────────────

def _build_globe_html(hops_data: list[dict]) -> str:
    """
    Genera l'HTML completo del globo D3 con i dati degli hop iniettati
    come JSON inline. Nessuna dipendenza Python oltre a streamlit.
    """

    # Serializza i dati hop per JavaScript
    js_hops = []
    total_hops = len(hops_data)
    for route_index, h in enumerate(hops_data):
        coords = h["coords"]
        if coords is None:
            continue
        geo  = h["geo"]
        rep  = h["rep"]
        score = rep.get("abuseConfidenceScore") if rep.get("status") == "ok" else None
        risk  = _score_to_risk(score)
        hop   = h["hop"]

        js_hops.append({
            "lat":       coords[0],
            "lon":       coords[1],
            "role":      h["role"],
            "color":     _risk_color(risk),
            "risk":      risk,
            "ip":        hop.get("sender_ip") or "—",
            "fromHost":  hop.get("from_host") or "—",
            "byHost":    hop.get("by_host") or "—",
            "tls":       " ".join(part for part in (hop.get("tls_version"), hop.get("tls_cipher")) if part),
            "hopNumber":  total_hops - route_index,
            "routeIndex": route_index + 1,
            "senderDomain": hop.get("sender_domain") or "",
            "forAddress": hop.get("for_address") or "",
            "allIps":     hop.get("all_ips") or [],
            "raw":        hop.get("raw") or "",
            "city":      geo.get("city", ""),
            "country":   geo.get("country", ""),
            "region":    geo.get("region", ""),
            "timezone":  geo.get("timezone", ""),
            "isp":       geo.get("isp", ""),
            "org":       geo.get("org", ""),
            "asn":       geo.get("asn", ""),
            "isProxy":   bool(geo.get("is_proxy")),
            "isHosting": bool(geo.get("is_hosting")),
            "score":     score,
            "reports":   rep.get("totalReports", 0) if rep.get("status") == "ok" else None,
            "usageType": rep.get("usageType", "") if rep.get("status") == "ok" else "",
            "domain":    rep.get("domain", "") if rep.get("status") == "ok" else "",
            "lastReport": rep.get("lastReportedAt", "") if rep.get("status") == "ok" else "",
            "roleLabel": {
                "sender":    "Closest to sender",
                "injection": "Injection server",
                "relay":     "Relay intermedio",
                "recipient": "Closest to recipient",
            }.get(h["role"], h["role"]),
        })

    # I Received header sono in ordine inverso: [0]=recipient … [-1]=sender.
    # Invertiamo così gli archi sul globo vanno da sender → recipient.
    js_hops = list(reversed(js_hops))

    # Se più hop hanno la stessa città/coordinate, i marker si coprono.
    # Li separiamo solo graficamente; il tooltip conserva IP e dati reali.
    coordinate_groups: dict[tuple[float, float], list[int]] = {}
    for idx, hop in enumerate(js_hops):
        key = (round(hop["lat"], 3), round(hop["lon"], 3))
        coordinate_groups.setdefault(key, []).append(idx)

    for indexes in coordinate_groups.values():
        if len(indexes) == 1:
            continue
        spread = 0.45
        for pos, idx in enumerate(indexes):
            offset = (pos - (len(indexes) - 1) / 2) * spread
            js_hops[idx]["lat"] = js_hops[idx]["lat"] + offset * 0.35
            js_hops[idx]["lon"] = js_hops[idx]["lon"] + offset

    hops_json = json.dumps(js_hops)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; font-family:sans-serif; overflow:hidden; }}
  #globe-wrap {{
    width:100%; height:520px;
    display:flex; align-items:center; justify-content:center;
    position:relative;
  }}
  canvas {{ cursor:grab; }}
  canvas:active {{ cursor:grabbing; }}

  #tooltip {{
    position:absolute; pointer-events:auto;
    background:rgba(13,17,23,.92); border:1px solid rgba(255,255,255,.12);
    border-radius:8px; padding:10px 14px;
    font-size:12px; color:#e6edf3; line-height:1.7;
    max-width:280px; display:none; z-index:99;
  }}
  #tooltip .tt-title {{
    font-weight:600; font-size:13px;
    border-bottom:1px solid rgba(255,255,255,.12);
    padding-bottom:5px; margin-bottom:6px;
  }}
  #tooltip .tt-row {{ display:flex; gap:8px; }}
  #tooltip .tt-label {{ color:#8b949e; min-width:60px; }}
  #tooltip .risk-high    {{ color:#E24B4A; font-weight:600; }}
  #tooltip .risk-medium  {{ color:#EF9F27; font-weight:600; }}
  #tooltip .risk-low     {{ color:#1D9E75; font-weight:600; }}
  #tooltip .risk-unknown {{ color:#888780; }}

  #legend {{
    position:absolute; bottom:14px; left:14px;
    background:rgba(13,17,23,.8); border:1px solid rgba(255,255,255,.1);
    border-radius:7px; padding:8px 12px;
    font-size:11px; color:#8b949e; line-height:2;
  }}
  #legend span {{
    display:inline-block; width:9px; height:9px;
    border-radius:50%; margin-right:5px; vertical-align:middle;
  }}

  #controls {{
    position:absolute; bottom:14px; right:14px;
    display:flex; flex-direction:column; gap:6px;
  }}
  #controls button {{
    background:rgba(13,17,23,.8); border:1px solid rgba(255,255,255,.15);
    border-radius:6px; color:#c9d1d9; font-size:12px;
    padding:5px 10px; cursor:pointer; transition:background .15s;
  }}
  #controls button:hover {{ background:rgba(255,255,255,.08); }}
</style>
</head>
<body>
<div id="globe-wrap">
  <canvas id="globe"></canvas>
  <div id="tooltip"></div>
  <div id="legend">
    <div><span style="background:#E24B4A"></span>Alto rischio / origine</div>
    <div><span style="background:#EF9F27"></span>Medio rischio / relay</div>
    <div><span style="background:#1D9E75"></span>Pulito / destinatario</div>
    <div><span style="background:#888780"></span>Sconosciuto</div>
  </div>
  <div id="controls">
    <button id="btn-play" title="Avvia/ferma rotazione">&#9654;</button>
    <button id="btn-fit"  title="Centra sul percorso">&#x2316;</button>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
const HOPS = {hops_json};

const H = 520;

const wrap = document.getElementById('globe-wrap');
const canvas = document.getElementById('globe');
const ctx = canvas.getContext('2d');

let W = 720;
let R = Math.min(W, H) / 2 - 20;

const proj = d3.geoOrthographic()
  .scale(R)
  .translate([W/2, H/2])
  .clipAngle(90);

const path = d3.geoPath(proj, ctx);

let world = null;
let rotating = false;
let rotateSpeed = 0.18;
let lambda = 0, phi = 0;
let dragStart = null, dragLambda, dragPhi;
let hoverIdx = null;
let pinnedIdx = null;
let animFrame = null;

const tooltip  = document.getElementById('tooltip');
const btnPlay  = document.getElementById('btn-play');
const btnFit   = document.getElementById('btn-fit');

function toRad(d) {{ return d * Math.PI / 180; }}
function toDeg(r) {{ return r * 180 / Math.PI; }}

function resizeGlobe() {{
  const measured = wrap.getBoundingClientRect().width || wrap.offsetWidth || 720;
  W = Math.max(320, Math.floor(measured));
  R = Math.min(W, H) / 2 - 20;
  canvas.width = W;
  canvas.height = H;
  proj.scale(R).translate([W / 2, H / 2]);
}}

function centerOnRoute() {{
  if (HOPS.length === 0) return;
  const avgLon = HOPS.reduce((s,h)=>s+h.lon,0)/HOPS.length;
  const avgLat = HOPS.reduce((s,h)=>s+h.lat,0)/HOPS.length;
  lambda = -avgLon;
  phi    = -avgLat;
  proj.rotate([lambda, phi]);
}}

function greatCirclePoints(lon1, lat1, lon2, lat2, n) {{
  const pts = [];
  for (let i = 0; i <= n; i++) {{
    const t = i / n;
    const p = d3.geoInterpolate([lon1, lat1], [lon2, lat2])(t);
    pts.push(p);
  }}
  return pts;
}}

function drawGlobe() {{
  ctx.clearRect(0, 0, W, H);

  ctx.beginPath();
  path({{type:'Sphere'}});
  ctx.fillStyle = '#1a2332';
  ctx.fill();

  ctx.beginPath();
  path({{type:'Sphere'}});
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth = 0.8;
  ctx.stroke();

  if (world) {{
    ctx.beginPath();
    path(topojson.feature(world, world.objects.land));
    ctx.fillStyle = '#243447';
    ctx.fill();

    ctx.beginPath();
    path(topojson.mesh(world, world.objects.countries, (a,b) => a !== b));
    ctx.strokeStyle = 'rgba(255,255,255,.08)';
    ctx.lineWidth = 0.4;
    ctx.stroke();
  }}

  ctx.beginPath();
  path(d3.geoGraticule()());
  ctx.strokeStyle = 'rgba(255,255,255,.04)';
  ctx.lineWidth = 0.3;
  ctx.stroke();

  // Archi tra hop consecutivi
  for (let i = 0; i < HOPS.length - 1; i++) {{
    const a = HOPS[i], b = HOPS[i+1];
    const pts = greatCirclePoints(a.lon, a.lat, b.lon, b.lat, 60);
    const geo = {{type:'LineString', coordinates: pts}};

    ctx.beginPath();
    path(geo);
    ctx.strokeStyle = a.color;
    ctx.lineWidth = 1.8;
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([6, 10]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }}

  // Marker hop
  HOPS.forEach((h, i) => {{
    const px = proj([h.lon, h.lat]);
    if (!px) return;

    // Verifica se è sul lato visibile
    const visible = d3.geoContains(
      {{type:'Sphere', coordinates:[]}},
      [h.lon, h.lat]
    );
    if (!visible && dotProduct(h.lon, h.lat) < 0) return;

    const isHover = hoverIdx === i || pinnedIdx === i;
    const r = isHover ? 12 : (HOPS.length === 1 ? 10 : 8);

    ctx.beginPath();
    ctx.arc(px[0], px[1], r + 3, 0, 2*Math.PI);
    ctx.fillStyle = h.color + '30';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px[0], px[1], r, 0, 2*Math.PI);
    ctx.fillStyle = h.color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,.8)';
    ctx.lineWidth = isHover ? 2 : 1.5;
    ctx.stroke();

    // Etichetta
    const roleToLbl = {{sender:'S', injection:'I', relay:'R', recipient:'D'}};
    const lbl = roleToLbl[h.role] || String(i+1);

    ctx.fillStyle = '#fff';
    ctx.font = `bold ${{isHover ? 11 : 10}}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(lbl, px[0], px[1]);

    // City label
    if (h.city && isHover) {{
      const label = h.city + (h.country ? ', '+h.country : '');
      ctx.font = '11px sans-serif';
      ctx.fillStyle = '#e6edf3';
      ctx.textAlign = 'center';
      ctx.fillText(label, px[0], px[1] - r - 8);
    }}
  }});
}}

function dotProduct(lon, lat) {{
  const r = toRad;
  const [lam, ph] = proj.rotate();
  return Math.cos(r(lat)) * Math.cos(r(ph)) * Math.cos(r(lon) - r(-lam))
       + Math.sin(r(lat)) * Math.sin(r(ph));
}}

let lastTime = null;
let arcOffset = 0;

function animate(ts) {{
  if (!lastTime) lastTime = ts;
  const dt = ts - lastTime;
  lastTime = ts;

  if (rotating) {{
    lambda += rotateSpeed * dt / 16;
    proj.rotate([lambda, phi]);
  }}

  drawGlobe();
  animFrame = requestAnimationFrame(animate);
}}

function escapeHtml(value) {{
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

function shortRaw(value) {{
  const raw = String(value || '').replace(/\\s+/g, ' ').trim();
  return raw.length > 260 ? raw.slice(0, 260) + '…' : raw;
}}

function renderHopTooltip(h, px, sticky=false) {{
  const score = h.score !== null ? h.score + '/100' : 'N/D';
  const reports = h.reports !== null && h.reports !== undefined ? h.reports : 'N/D';
  const riskCls = 'risk-' + h.risk;
  const loc = [h.city, h.region, h.country].filter(Boolean).join(', ') || '—';
  const tlsRow = h.tls ? `<div class="tt-row"><span class="tt-label">TLS</span><span>${{escapeHtml(h.tls)}}</span></div>` : '';
  const domainRow = h.senderDomain ? `<div class="tt-row"><span class="tt-label">HELO</span><span>${{escapeHtml(h.senderDomain)}}</span></div>` : '';
  const forRow = h.forAddress ? `<div class="tt-row"><span class="tt-label">For</span><span>${{escapeHtml(h.forAddress)}}</span></div>` : '';
  const orgRow = h.org ? `<div class="tt-row"><span class="tt-label">Org</span><span>${{escapeHtml(h.org)}}</span></div>` : '';
  const asnRow = h.asn ? `<div class="tt-row"><span class="tt-label">ASN</span><span>${{escapeHtml(h.asn)}}</span></div>` : '';
  const usageRow = h.usageType ? `<div class="tt-row"><span class="tt-label">Uso</span><span>${{escapeHtml(h.usageType)}}</span></div>` : '';
  const lastRow = h.lastReport ? `<div class="tt-row"><span class="tt-label">Ultima</span><span>${{escapeHtml(h.lastReport)}}</span></div>` : '';
  const allIps = (h.allIps || []).length ? escapeHtml((h.allIps || []).join(', ')) : '—';
  const badges = (h.isProxy ? '<span style="color:#E24B4A"> ⚠ Proxy/VPN</span>' : '')
               + (h.isHosting ? '<span style="color:#EF9F27"> ☁ Datacenter</span>' : '');
  const rawRow = h.raw ? `<details style="margin-top:6px"><summary style="cursor:pointer;color:#8b949e">Raw Received</summary><div style="font-family:monospace;font-size:11px;line-height:1.45;margin-top:4px">${{escapeHtml(shortRaw(h.raw))}}</div></details>` : '';

  tooltip.innerHTML = `
    <div class="tt-title">${{sticky ? '[fissato] ' : ''}}${{escapeHtml(h.roleLabel)}} · Hop ${{h.hopNumber}}</div>
    <div class="tt-row"><span class="tt-label">IP</span><span style="font-family:monospace">${{escapeHtml(h.ip)}}</span></div>
    <div class="tt-row"><span class="tt-label">From</span><span>${{escapeHtml(h.fromHost)}}</span></div>
    <div class="tt-row"><span class="tt-label">By</span><span>${{escapeHtml(h.byHost)}}</span></div>
    ${{domainRow}}
    ${{forRow}}
    ${{tlsRow}}
    <div class="tt-row"><span class="tt-label">Luogo</span><span>${{escapeHtml(loc)}}</span></div>
    <div class="tt-row"><span class="tt-label">ISP</span><span>${{escapeHtml(h.isp || '—')}}</span></div>
    ${{orgRow}}
    ${{asnRow}}
    ${{usageRow}}
    <div class="tt-row"><span class="tt-label">Abuse</span><span class="${{riskCls}}">${{escapeHtml(score)}} · ${{escapeHtml(reports)}} report</span></div>
    ${{lastRow}}
    <div class="tt-row"><span class="tt-label">Tutti IP</span><span style="font-family:monospace">${{allIps}}</span></div>
    ${{badges}}
    ${{rawRow}}
  `;
  let tx = px[0] + 16, ty = px[1] - 10;
  if (tx + 280 > W) tx = px[0] - 280;
  tooltip.style.left = tx + 'px';
  tooltip.style.top = ty + 'px';
  tooltip.style.display = 'block';
}}

// Drag
canvas.addEventListener('mousedown', e => {{
  rotating = false;
  dragStart = [e.clientX, e.clientY];
  const r = proj.rotate();
  dragLambda = r[0]; dragPhi = r[1];
}});

window.addEventListener('mousemove', e => {{
  if (dragStart) {{
    const dx = e.clientX - dragStart[0];
    const dy = e.clientY - dragStart[1];
    lambda = dragLambda + dx * 0.3;
    phi    = Math.max(-60, Math.min(60, dragPhi - dy * 0.3));
    proj.rotate([lambda, phi]);
  }} else {{
    // Hover detection
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let found = -1;
    HOPS.forEach((h, i) => {{
      const px = proj([h.lon, h.lat]);
      if (!px) return;
      if (dotProduct(h.lon, h.lat) < 0) return;
      const d = Math.hypot(px[0]-mx, px[1]-my);
      if (d < 14) found = i;
    }});
    hoverIdx = found;
    if (pinnedIdx === null) {{
      if (found >= 0) {{
        renderHopTooltip(HOPS[found], proj([HOPS[found].lon, HOPS[found].lat]));
      }} else {{
        tooltip.style.display = 'none';
      }}
    }}
  }}
}});

window.addEventListener('mouseup', e => {{
  if (dragStart) {{
    dragStart = null;
  }}
}});

canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let found = -1;
  HOPS.forEach((h, i) => {{
    const px = proj([h.lon, h.lat]);
    if (!px) return;
    if (dotProduct(h.lon, h.lat) < 0) return;
    if (Math.hypot(px[0] - mx, px[1] - my) < 16) found = i;
  }});

  pinnedIdx = found >= 0 ? found : null;
  if (pinnedIdx !== null) {{
    const h = HOPS[pinnedIdx];
    renderHopTooltip(h, proj([h.lon, h.lat]), true);
  }} else {{
    tooltip.style.display = 'none';
  }}
}});

// Touch support
canvas.addEventListener('touchstart', e => {{
  e.preventDefault();
  rotating = false;
  dragStart = [e.touches[0].clientX, e.touches[0].clientY];
  const r = proj.rotate();
  dragLambda = r[0]; dragPhi = r[1];
}}, {{passive:false}});

canvas.addEventListener('touchmove', e => {{
  e.preventDefault();
  if (!dragStart) return;
  const dx = e.touches[0].clientX - dragStart[0];
  const dy = e.touches[0].clientY - dragStart[1];
  lambda = dragLambda + dx * 0.4;
  phi    = Math.max(-60, Math.min(60, dragPhi - dy * 0.4));
  proj.rotate([lambda, phi]);
}}, {{passive:false}});

canvas.addEventListener('touchend', () => {{
  dragStart = null;
}});

// Pulsanti
btnPlay.addEventListener('click', () => {{
  rotating = !rotating;
  btnPlay.textContent = rotating ? '❚❚' : '▶';
}});

btnFit.addEventListener('click', () => {{
  centerOnRoute();
}});

// Centra inizialmente sul percorso
resizeGlobe();
centerOnRoute();

window.addEventListener('resize', () => {{
  resizeGlobe();
  centerOnRoute();
}});

setTimeout(() => {{ resizeGlobe(); centerOnRoute(); }}, 150);
setTimeout(() => {{ resizeGlobe(); centerOnRoute(); }}, 650);
setTimeout(() => {{ resizeGlobe(); centerOnRoute(); }}, 1500);

// Carica topologia mondo
d3.json('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
  .then(w => {{ world = w; }})
  .catch(() => {{ world = null; }});

requestAnimationFrame(animate);
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────

def render_email_globe(soc: dict, validator) -> None:
    """
    Renderizza il globo 3D interattivo del percorso email in Streamlit.

    Sostituisce (o affianca) render_email_path_map in app.py:

        with st.expander("🌍 Percorso geografico email", expanded=True):
            render_email_globe(soc, validator)
    """
    hops = soc.get("received_hops", [])
    if not hops:
        st.info("Nessun header Received trovato: impossibile costruire il globo.")
        return

    # Assegna ruolo a ogni hop
    n = len(hops)
    roles = []
    for i in range(n):
        if i == 0:          roles.append("recipient")
        elif i == n - 1:    roles.append("sender")
        elif i == 1:        roles.append("injection")
        else:               roles.append("relay")

    # Geolocalizza + reputazione in parallelo
    def _fetch(hop: dict):
        ip = hop.get("sender_ip") or ""
        if not _is_geolocatable_ip(ip):
            return {"status": "skipped"}, {"status": "skipped"}
        return validator.geolocate_ip(ip), validator.check_ip_reputation(ip)

    with st.spinner("Geolocalizzazione hop in corso…"):
        with ThreadPoolExecutor(max_workers=min(n, 6)) as ex:
            results = list(ex.map(_fetch, hops))

    hops_data = [
        {
            "hop":    hop,
            "role":   role,
            "geo":    geo,
            "rep":    rep,
            "coords": _geo_coords(geo),
        }
        for hop, role, (geo, rep) in zip(hops, roles, results)
    ]

    located = [h for h in hops_data if h["coords"] is not None]

    if not located:
        st.info(
            "Tutti gli IP sono privati o non risolvibili. "
            "Il globo richiede almeno un IP pubblico geolocalizzabile."
        )
        return

    # Riepilogo card sopra il globo — ordine sender→recipient, coerente col globo
    hops_data_display = list(reversed(hops_data))
    cols = st.columns(max(n, 1))
    risk_icon = {"high": "🔴", "medium": "🟠", "low": "🟢", "unknown": "⚪"}
    role_label = {
        "sender": "Origine", "injection": "Iniezione",
        "relay": "Relay",    "recipient": "Destinatario",
    }
    for col, h in zip(cols, hops_data_display):
        ip     = h["hop"].get("sender_ip") or "—"
        city   = h["geo"].get("city", "")
        country= h["geo"].get("country", "")
        score  = h["rep"].get("abuseConfidenceScore") if h["rep"].get("status") == "ok" else None
        risk   = _score_to_risk(score)
        loc    = ", ".join(p for p in [city, country] if p) or ("IP privato" if not h["coords"] else "—")
        with col:
            st.markdown(
                f"**{risk_icon[risk]} {role_label.get(h['role'], h['role'])}**  \n"
                f"`{ip}`  \n"
                f"<span style='font-size:12px;color:gray'>{loc}</span>",
                unsafe_allow_html=True,
            )

    skipped = len(hops_data) - len(located)
    if skipped:
        st.caption(
            f"{skipped} hop non sono sul globo perché hanno IP privati, riservati "
            "o non geolocalizzabili; restano visibili nel dettaglio Routing sotto."
        )

    st.markdown("")
    # Render globo
    globe_html = _build_globe_html(hops_data)
    components.html(globe_html, height=530, scrolling=False)

    st.caption("Trascina per ruotare. Passa su un marker per l'anteprima, cliccalo per fissare tutti i dettagli.")
