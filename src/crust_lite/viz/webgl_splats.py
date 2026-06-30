from __future__ import annotations

import json
import math
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector
from crust_lite.paths import ProjectPaths
from crust_lite.viz.japan_outline import local_context_outlines


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _type_code(value: str) -> int:
    return {"direct": 0, "reflected": 1, "scattered": 2, "residual": 3}.get(value, 3)


def _round(value: Any, digits: int = 3) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(number, digits)



def _catmull_rom(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], t: float) -> tuple[float, float]:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2.0 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2.0 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    return x, y


def _densify_outline(coords: list[tuple[float, float]], samples_per_segment: int = 18) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return coords
    closed = coords[0] == coords[-1]
    base = coords[:-1] if closed else coords
    if len(base) < 3:
        return coords
    dense: list[tuple[float, float]] = []
    n = len(base)
    for i in range(n if closed else n - 1):
        p0 = base[(i - 1) % n] if closed or i > 0 else base[0]
        p1 = base[i]
        p2 = base[(i + 1) % n]
        p3 = base[(i + 2) % n] if closed or i + 2 < n else base[-1]
        segment_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        samples = max(samples_per_segment, int(math.ceil(segment_len / 0.006)))
        for step in range(samples):
            dense.append(_catmull_rom(p0, p1, p2, p3, step / samples))
    if closed:
        dense.append(dense[0])
    else:
        dense.append(base[-1])
    return dense


def _terrain_payload(config: AppConfig, is_sample: bool) -> dict[str, Any]:
    """Return outline-only context geometry.

    The splat preview intentionally avoids filled land/sea/terrain surfaces.
    Filled surfaces hide subsurface splats and make the view look like a flat
    map plate. The 3D context is therefore only high-density coastline traces
    projected into the configured local CRS at z=0.
    """
    projector = LocalProjector(config.region)
    target_segment_km = 5.0 if is_sample else 1.0
    outlines = local_context_outlines(
        config.region.bbox,
        margin_deg=3.0,
        target_segment_km=target_segment_km,
        prefer_high_resolution=True,
    )
    samples_per_segment = 12 if is_sample else 200
    dense_outlines = []
    for outline in outlines:
        source = str(outline.get("source", "offline_coarse_context"))
        if source == "natural_earth_10m_admin_0_japan":
            coordinates = outline["coordinates"]
        else:
            coordinates = _densify_outline(outline["coordinates"], samples_per_segment=samples_per_segment)
        dense_outlines.append(
            {
                "name": outline["name"],
                "coordinates": coordinates,
                "source": source,
                "target_segment_km": outline.get("target_segment_km", target_segment_km),
            }
        )
    outline_payload = []
    for outline in dense_outlines:
        flat: list[float] = []
        for lon, lat in outline["coordinates"]:
            x_m, y_m = projector.lonlat_to_xy(lon, lat)
            flat.extend([_round(x_m), _round(y_m), 0.0])
        outline_payload.append({"name": outline["name"], "source": outline["source"], "positions": flat})
    outline_sources = sorted({str(outline["source"]) for outline in dense_outlines})
    high_resolution = "natural_earth_10m_admin_0_japan" in outline_sources
    return {
        "nx": 0,
        "ny": 0,
        "positions": [],
        "colors": [],
        "indices": [],
        "outlines": outline_payload,
        "outline_vertices": sum(len(outline["positions"]) // 3 for outline in outline_payload),
        "outline_sources": outline_sources,
        "outline_target_segment_km": target_segment_km,
        "outline_resolution": (
            "natural_earth_10m_admin_0_japan_densified_to_1km"
            if high_resolution and not is_sample
            else "natural_earth_10m_admin_0_japan_densified_to_5km"
            if high_resolution
            else "catmull_rom_densified_offline_outline_fallback"
        ),
        "surface_enabled": False,
    }


def _splat_payload(config: AppConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions: list[float] = []
    colors: list[float] = []
    sizes: list[float] = []
    opacities: list[float] = []
    types: list[float] = []
    amplitudes: list[float] = []
    line_positions: list[float] = []
    vertical = config.visualization_3d.vertical_exaggeration
    line_rows = sorted(
        rows,
        key=lambda row: (float(row.get("beam_power", 0.0) or 0.0), float(row.get("amplitude", 0.0) or 0.0)),
        reverse=True,
    )[: min(1600, len(rows))]
    line_ids = {id(row) for row in line_rows}
    for row in rows:
        x_m = float(row.get("x_m", 0.0) or 0.0)
        y_m = float(row.get("y_m", 0.0) or 0.0)
        z_plot = -1.0 * float(row.get("z_m", 0.0) or 0.0) * vertical
        positions.extend([_round(x_m), _round(y_m), _round(z_plot)])
        colors.extend(
            [
                _round(float(row.get("color_r", 180) or 180) / 255.0, 5),
                _round(float(row.get("color_g", 120) or 120) / 255.0, 5),
                _round(float(row.get("color_b", 80) or 80) / 255.0, 5),
            ]
        )
        sigma_xy = max(float(row.get("sigma_x_m", 1.0) or 1.0), float(row.get("sigma_y_m", 1.0) or 1.0))
        sigma_z = max(float(row.get("sigma_z_m", 1.0) or 1.0) * vertical, 1.0)
        sizes.append(_round(max(sigma_xy, 0.35 * sigma_z), 3))
        opacities.append(_round(float(row.get("opacity", 0.6) or 0.6), 5))
        primitive_type = str(row.get("primitive_type", "direct") or "direct")
        types.append(float(_type_code(primitive_type)))
        amplitudes.append(_round(float(row.get("amplitude", 0.0) or 0.0), 5))
        if id(row) in line_ids:
            sx = float(row.get("source_event_x_m", x_m) or x_m)
            sy = float(row.get("source_event_y_m", y_m) or y_m)
            sz = -1.0 * float(row.get("source_event_z_m", row.get("z_m", 0.0)) or 0.0) * vertical
            line_positions.extend([_round(sx), _round(sy), _round(sz), _round(x_m), _round(y_m), _round(z_plot)])
    return {
        "positions": positions,
        "colors": colors,
        "sizes": sizes,
        "opacities": opacities,
        "types": types,
        "amplitudes": amplitudes,
        "source_lines": line_positions,
        "line_segments": len(line_positions) // 6,
    }


def _bounds_from_payload(splats: dict[str, Any], terrain: dict[str, Any]) -> dict[str, float]:
    values = [splats["positions"], terrain["positions"]]
    values.extend(outline["positions"] for outline in terrain.get("outlines", []))
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for flat in values:
        if not flat:
            continue
        xs.extend(flat[0::3])
        ys.extend(flat[1::3])
        zs.extend(flat[2::3])
    if not xs:
        xs = ys = zs = [0.0]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "min_z": min(zs),
        "max_z": max(zs),
    }


def _webgl_html(payload: dict[str, Any]) -> str:
    json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>crust-lite WebGL Gaussian splats</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #071015; color: #e5eef5; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #gl {{ width: 100vw; height: 100vh; display: block; touch-action: none; }}
    #hud {{ position: fixed; left: 12px; top: 10px; max-width: 520px; background: rgba(7, 16, 21, 0.78); border: 1px solid rgba(180, 205, 220, 0.32); padding: 10px 12px; font-size: 13px; line-height: 1.35; backdrop-filter: blur(5px); }}
    #hud h1 {{ font-size: 16px; margin: 0 0 6px; }}
    #hud label {{ margin-right: 10px; white-space: nowrap; }}
    #hud input[type=range] {{ width: 140px; vertical-align: middle; }}
    .notice {{ color: #ffd29d; font-weight: 600; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 3px; border-radius: 50%; }}
  </style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="hud">
  <h1>WebGL Gaussian splats + outline-only Japan context</h1>
  <div class="notice">研究用の状態表示です。地震の発生日・場所・規模を断定的に予測するものではありません。</div>
  <div id="stats"></div>
  <div>
    <label><input type="checkbox" data-type="0" checked><span class="swatch" style="background:#48a0d8"></span>direct</label>
    <label><input type="checkbox" data-type="1" checked><span class="swatch" style="background:#f0a640"></span>reflected</label>
    <label><input type="checkbox" data-type="2" checked><span class="swatch" style="background:#ad67df"></span>scattered</label>
    <label><input type="checkbox" data-type="3" checked><span class="swatch" style="background:#a0a0a0"></span>residual</label>
  </div>
  <div>
    <label><input id="outlineToggle" type="checkbox" checked>Japan outline</label>
    <label><input id="lineToggle" type="checkbox" checked>source lines</label>
  </div>
  <div>
    splat scale <input id="scaleSlider" type="range" min="0.25" max="8" step="0.05" value="1.45">
    opacity <input id="opacitySlider" type="range" min="0.15" max="2.5" step="0.05" value="1.0">
  </div>
  <div>drag: rotate / wheel or pinch: zoom / shift+drag or two-finger drag: pan</div>
</div>
<script id="payload" type="application/json">{json_text}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', {{antialias: true, alpha: false}});
if (!gl) throw new Error('WebGL2 is required');
document.getElementById('stats').textContent =
  `splats=${{payload.metadata.displayed_splats}} / surface=off / outline vertices=${{payload.terrain.outline_vertices}} / renderer=${{payload.metadata.renderer}}`;

function shader(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
function program(vs, fs) {{
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}}
const splatVS = `#version 300 es
precision highp float;
in vec3 a_pos; in vec3 a_color; in float a_size; in float a_opacity; in float a_type;
uniform mat4 u_mvp; uniform float u_pointScale;
out vec3 v_color; out float v_opacity; out float v_type;
void main() {{
  vec4 clip = u_mvp * vec4(a_pos, 1.0);
  gl_Position = clip;
  float perspectiveScale = clamp(1.0 / max(0.25, clip.w), 0.35, 3.0);
  gl_PointSize = clamp(a_size * u_pointScale * perspectiveScale, 2.0, 384.0);
  v_color = a_color; v_opacity = a_opacity; v_type = a_type;
}}`;
const splatFS = `#version 300 es
precision highp float;
in vec3 v_color; in float v_opacity; in float v_type;
uniform vec4 u_visible; uniform float u_opacityScale;
out vec4 outColor;
void main() {{
  float vis = v_type < 0.5 ? u_visible.x : (v_type < 1.5 ? u_visible.y : (v_type < 2.5 ? u_visible.z : u_visible.w));
  if (vis < 0.5) discard;
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;
  float gaussian = exp(-3.25 * r2);
  float alpha = clamp(v_opacity * u_opacityScale * gaussian, 0.0, 0.92);
  outColor = vec4(v_color, alpha);
}}`;
const meshVS = `#version 300 es
precision highp float;
in vec3 a_pos; in vec3 a_color; uniform mat4 u_mvp; out vec3 v_color;
void main() {{ gl_Position = u_mvp * vec4(a_pos, 1.0); v_color = a_color; }}`;
const meshFS = `#version 300 es
precision highp float;
in vec3 v_color; uniform float u_alpha; out vec4 outColor;
void main() {{ outColor = vec4(v_color, u_alpha); }}`;
const lineVS = `#version 300 es
precision highp float;
in vec3 a_pos; uniform mat4 u_mvp;
void main() {{ gl_Position = u_mvp * vec4(a_pos, 1.0); }}`;
const lineFS = `#version 300 es
precision highp float;
uniform vec4 u_color; out vec4 outColor;
void main() {{ outColor = u_color; }}`;
const splatProg = program(splatVS, splatFS), meshProg = program(meshVS, meshFS), lineProg = program(lineVS, lineFS);

const b = payload.bounds;
const center = [(b.min_x+b.max_x)/2, (b.min_y+b.max_y)/2, (b.min_z+b.max_z)/2];
const span = Math.max(b.max_x-b.min_x, b.max_y-b.min_y, b.max_z-b.min_z, 1);
function normPositions(src) {{
  const out = new Float32Array(src.length);
  for (let i=0; i<src.length; i+=3) {{
    out[i] = (src[i] - center[0]) / span * 2.0;
    out[i+1] = (src[i+1] - center[1]) / span * 2.0;
    out[i+2] = (src[i+2] - center[2]) / span * 2.0;
  }}
  return out;
}}
function normSizes(src) {{
  const out = new Float32Array(src.length);
  for (let i=0; i<src.length; i++) out[i] = Math.max(3.0, Math.min(140.0, src[i] / span * 2600.0));
  return out;
}}
function buf(data, target=gl.ARRAY_BUFFER) {{
  const b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b;
}}
const splat = {{
  n: payload.splats.positions.length / 3,
  pos: buf(normPositions(payload.splats.positions)),
  color: buf(new Float32Array(payload.splats.colors)),
  size: buf(normSizes(payload.splats.sizes)),
  opacity: buf(new Float32Array(payload.splats.opacities)),
  type: buf(new Float32Array(payload.splats.types)),
}};
const terrain = {{
  n: payload.terrain.indices.length,
  pos: buf(normPositions(payload.terrain.positions)),
  color: buf(new Float32Array(payload.terrain.colors)),
  idx: buf(new Uint32Array(payload.terrain.indices), gl.ELEMENT_ARRAY_BUFFER),
}};
const sourceLines = {{ n: payload.splats.source_lines.length / 3, pos: buf(normPositions(payload.splats.source_lines)) }};
const outlineBuffers = payload.terrain.outlines.map(o => ({{ name:o.name, n:o.positions.length/3, pos:buf(normPositions(o.positions)) }}));

function attrib(p, name, buffer, size) {{
  const loc = gl.getAttribLocation(p, name);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
}}
function mat4mul(a,b) {{
  const o = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];
  return o;
}}
function perspective(fovy, aspect, near, far) {{
  const f=1/Math.tan(fovy/2), nf=1/(near-far), o=new Float32Array(16);
  o[0]=f/aspect; o[5]=f; o[10]=(far+near)*nf; o[11]=-1; o[14]=2*far*near*nf; return o;
}}
function lookAt(eye, target, up) {{
  let zx=eye[0]-target[0], zy=eye[1]-target[1], zz=eye[2]-target[2];
  let zl=1/Math.hypot(zx,zy,zz); zx*=zl; zy*=zl; zz*=zl;
  let xx=up[1]*zz-up[2]*zy, xy=up[2]*zx-up[0]*zz, xz=up[0]*zy-up[1]*zx;
  let xl=1/Math.hypot(xx,xy,xz); xx*=xl; xy*=xl; xz*=xl;
  const yx=zy*xz-zz*xy, yy=zz*xx-zx*xz, yz=zx*xy-zy*xx;
  const o=new Float32Array(16);
  o[0]=xx; o[1]=yx; o[2]=zx; o[4]=xy; o[5]=yy; o[6]=zy; o[8]=xz; o[9]=yz; o[10]=zz; o[15]=1;
  o[12]=-(xx*eye[0]+xy*eye[1]+xz*eye[2]); o[13]=-(yx*eye[0]+yy*eye[1]+yz*eye[2]); o[14]=-(zx*eye[0]+zy*eye[1]+zz*eye[2]);
  return o;
}}
let yaw=0.72, pitch=0.46, dist=3.2, pan=[0,0,0];
let visible=[1,1,1,1], showTerrain=false, showOutlines=true, showLines=true, splatScale=1.45, opacityScale=1.0;
const pointers = new Map();
let lastCentroid = null, lastPinchDistance = 0, lastPointer = null, panning = false;
function mvp() {{
  const eye=[dist*Math.cos(pitch)*Math.sin(yaw)+pan[0], dist*Math.cos(pitch)*Math.cos(yaw)+pan[1], dist*Math.sin(pitch)+pan[2]];
  return mat4mul(perspective(45*Math.PI/180, canvas.width/canvas.height, 0.01, 100.0), lookAt(eye, pan, [0,0,1]));
}}
function resize() {{ const dpr=Math.min(devicePixelRatio||1,4); canvas.width=Math.floor(innerWidth*dpr); canvas.height=Math.floor(innerHeight*dpr); gl.viewport(0,0,canvas.width,canvas.height); render(); }}
function pointerCentroid() {{
  let x=0, y=0;
  for (const p of pointers.values()) {{ x += p.x; y += p.y; }}
  const n = Math.max(1, pointers.size);
  return [x/n, y/n];
}}
function pointerDistance() {{
  const pts = Array.from(pointers.values());
  if (pts.length < 2) return 0;
  return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
}}
function panBy(dx, dy) {{
  pan[0] -= dx / 420;
  pan[2] += dy / 420;
}}
function render() {{
  gl.clearColor(0.027,0.063,0.082,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const matrix=mvp();
  if (showTerrain) {{
    gl.useProgram(meshProg); gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,'u_mvp'), false, matrix); gl.uniform1f(gl.getUniformLocation(meshProg,'u_alpha'), 0.0);
    attrib(meshProg,'a_pos',terrain.pos,3); attrib(meshProg,'a_color',terrain.color,3); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, terrain.idx); gl.drawElements(gl.TRIANGLES, terrain.n, gl.UNSIGNED_INT, 0);
  }}
  gl.useProgram(lineProg); gl.uniformMatrix4fv(gl.getUniformLocation(lineProg,'u_mvp'), false, matrix); gl.lineWidth(1);
  if (showOutlines) {{
    gl.uniform4f(gl.getUniformLocation(lineProg,'u_color'), 0.82, 1.0, 0.78, 0.95);
    for (const o of outlineBuffers) {{ attrib(lineProg,'a_pos',o.pos,3); gl.drawArrays(gl.LINE_STRIP,0,o.n); }}
  }}
  if (showLines) {{
    gl.uniform4f(gl.getUniformLocation(lineProg,'u_color'), 0.65, 0.78, 0.90, 0.23);
    attrib(lineProg,'a_pos',sourceLines.pos,3); gl.drawArrays(gl.LINES,0,sourceLines.n);
  }}
  gl.depthMask(false);
  gl.useProgram(splatProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(splatProg,'u_mvp'), false, matrix);
  gl.uniform1f(gl.getUniformLocation(splatProg,'u_pointScale'), splatScale);
  gl.uniform1f(gl.getUniformLocation(splatProg,'u_opacityScale'), opacityScale);
  gl.uniform4f(gl.getUniformLocation(splatProg,'u_visible'), visible[0], visible[1], visible[2], visible[3]);
  attrib(splatProg,'a_pos',splat.pos,3); attrib(splatProg,'a_color',splat.color,3); attrib(splatProg,'a_size',splat.size,1); attrib(splatProg,'a_opacity',splat.opacity,1); attrib(splatProg,'a_type',splat.type,1);
  gl.drawArrays(gl.POINTS,0,splat.n); gl.depthMask(true);
}}
canvas.addEventListener('pointerdown', e => {{
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, {{x:e.clientX, y:e.clientY}});
  panning = e.shiftKey || pointers.size >= 2;
  lastPointer = [e.clientX, e.clientY];
  lastCentroid = pointerCentroid();
  lastPinchDistance = pointerDistance();
}}, {{passive:false}});
canvas.addEventListener('pointermove', e => {{
  if (!pointers.has(e.pointerId)) return;
  e.preventDefault();
  pointers.set(e.pointerId, {{x:e.clientX, y:e.clientY}});
  if (pointers.size >= 2) {{
    const centroid = pointerCentroid();
    const pinchDistance = pointerDistance();
    if (lastCentroid) panBy(centroid[0] - lastCentroid[0], centroid[1] - lastCentroid[1]);
    if (lastPinchDistance > 0 && pinchDistance > 0) dist = Math.max(0.55, Math.min(12, dist * (lastPinchDistance / pinchDistance)));
    lastCentroid = centroid;
    lastPinchDistance = pinchDistance;
  }} else if (lastPointer) {{
    const dx = e.clientX - lastPointer[0], dy = e.clientY - lastPointer[1];
    lastPointer = [e.clientX, e.clientY];
    if (panning || e.shiftKey) panBy(dx, dy);
    else {{
      yaw += dx * 0.006;
      pitch = Math.max(-1.25, Math.min(1.25, pitch + dy * 0.006));
    }}
  }}
  render();
}}, {{passive:false}});
function endPointer(e) {{
  if (pointers.has(e.pointerId)) pointers.delete(e.pointerId);
  lastPointer = null;
  lastCentroid = pointers.size ? pointerCentroid() : null;
  lastPinchDistance = pointerDistance();
  panning = pointers.size >= 2;
}}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('lostpointercapture', endPointer);
canvas.addEventListener('wheel', e => {{ e.preventDefault(); dist=Math.max(0.55,Math.min(12,dist*Math.exp(e.deltaY*0.001))); render(); }}, {{passive:false}});
document.querySelectorAll('input[data-type]').forEach(el => el.addEventListener('change', e => {{ visible[Number(e.target.dataset.type)] = e.target.checked ? 1 : 0; render(); }}));
document.getElementById('outlineToggle').addEventListener('change', e => {{ showOutlines=e.target.checked; render(); }});
document.getElementById('lineToggle').addEventListener('change', e => {{ showLines=e.target.checked; render(); }});
document.getElementById('scaleSlider').addEventListener('input', e => {{ splatScale=Number(e.target.value); render(); }});
document.getElementById('opacitySlider').addEventListener('input', e => {{ opacityScale=Number(e.target.value); render(); }});
addEventListener('resize', resize); resize();
</script>
</body>
</html>
"""


def write_webgl_splat_preview(config: AppConfig, paths: ProjectPaths, rows: list[dict[str, Any]], is_sample: bool) -> None:
    paths.outputs_3d.mkdir(parents=True, exist_ok=True)
    limit_rows = rows[: min(len(rows), 250_000)]
    splats = _splat_payload(config, limit_rows)
    terrain = _terrain_payload(config, is_sample=is_sample)
    metadata = {
        "html": str(paths.outputs_3d / "array_projection_splats.html"),
        "renderer": "webgl2_gaussian_point_sprite",
        "gaussian_shader": "fragment_alpha=opacity*exp(-3.25*r2)",
        "displayed_splats": len(limit_rows),
        "total_splats": len(rows),
        "line_segments": splats["line_segments"],
        "is_sample_data": is_sample,
        "vertical_exaggeration": config.visualization_3d.vertical_exaggeration,
        "synthetic_aperture_enabled": config.waveform_array.synthetic_aperture_enabled,
        "uses_phase": config.waveform_array.use_phase,
        "uses_group_delay": config.waveform_array.use_group_delay,
        "primitive_type_counts": _count_values(rows, "primitive_type"),
        "path_family_counts": _count_values(rows, "path_family"),
        "terrain_overlay": "disabled_surface_outline_only",
        "terrain_grid": {"nx": terrain["nx"], "ny": terrain["ny"]},
        "canvas_device_pixel_ratio_max": 4,
        "point_sprite_max_px": 384,
        "touch_controls": "pointer_events_one_finger_rotate_two_finger_pan_pinch_zoom",
        "japan_outline_vertices": terrain["outline_vertices"],
        "japan_outline_sources": terrain["outline_sources"],
        "japan_outline_target_segment_km": terrain["outline_target_segment_km"],
        "japan_outline_resolution": terrain["outline_resolution"],
        "surface_rendering": "disabled_to_avoid_hiding_subsurface_splats",
        "sample_lightweight_rendering": is_sample,
        "rendering": "WebGL2 high-density point-sprite Gaussian splats with outline-only Japan context; not Plotly mesh ellipsoids",
        "not_prediction": True,
    }
    payload = {
        "metadata": metadata,
        "bounds": _bounds_from_payload(splats, terrain),
        "splats": splats,
        "terrain": terrain,
    }
    out = paths.outputs_3d / "array_projection_splats.html"
    out.write_text(_webgl_html(payload), encoding="utf-8")
    (paths.outputs_3d / "array_projection_splats.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
