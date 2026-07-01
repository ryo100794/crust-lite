from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, fault_rectangle_vertices
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import read_table
from crust_lite.paths import ProjectPaths
from crust_lite.viz.html_timeseries import NOTICE_3D
from crust_lite.viz.japan_outline import local_context_outlines


def _plot_z_m(z_m: float, vertical_exaggeration: float) -> float:
    return -1.0 * float(z_m) * vertical_exaggeration


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _select_events(events: list[dict[str, Any]], max_events: int) -> tuple[list[dict[str, Any]], str]:
    if len(events) <= max_events:
        return sorted(events, key=lambda row: str(row.get("time_utc", ""))), "none"
    half = max_events // 2
    by_mag = sorted(events, key=lambda row: _safe_float(row.get("magnitude")), reverse=True)[:half]
    by_time = sorted(events, key=lambda row: str(row.get("time_utc", "")), reverse=True)[: max_events - half]
    selected = {str(row.get("event_id", index)): row for index, row in enumerate(by_mag + by_time)}
    return sorted(selected.values(), key=lambda row: str(row.get("time_utc", ""))), "magnitude_top_and_latest"


def _limit_faults(features: list[dict[str, Any]], max_faults: int) -> tuple[list[dict[str, Any]], str]:
    if len(features) <= max_faults:
        return features, "none"
    ranked = sorted(
        features,
        key=lambda feature: _safe_float(
            feature.get("properties", {}).get("fault_score", feature.get("properties", {}).get("confidence", 0.0))
        ),
        reverse=True,
    )
    return ranked[:max_faults], "fault_score_or_confidence_top"


def _ramp_color(value: float, low: float, high: float) -> list[float]:
    span = max(1e-12, high - low)
    t = max(0.0, min(1.0, (value - low) / span))
    stops = [
        (0.0, (0.14, 0.18, 0.44)),
        (0.35, (0.08, 0.52, 0.72)),
        (0.70, (0.64, 0.78, 0.25)),
        (1.0, (0.96, 0.54, 0.16)),
    ]
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:], strict=False):
        if x0 <= t <= x1:
            f = (t - x0) / max(1e-12, x1 - x0)
            return [c0[i] + (c1[i] - c0[i]) * f for i in range(3)]
    return list(stops[-1][1])


def _events_payload(events: list[dict[str, Any]], cfg: AppConfig, display_mode: str, frame_days: int) -> dict[str, Any]:
    viz = cfg.visualization_3d
    frame_days = max(1, int(frame_days))
    if not events:
        return {
            "positions": [],
            "colors": [],
            "sizes": [],
            "opacities": [],
            "times": [],
            "magnitudes": [],
            "depths_km": [],
            "ids": [],
            "frame_labels": ["no_events"],
            "frame_indices": [0],
            "frame_days": frame_days,
            "mode": display_mode,
            "count": 0,
            "magnitude_range": [0.0, 0.0],
            "depth_range_km": [0.0, 0.0],
        }
    event_times = [_parse_dt(str(row["time_utc"])) for row in events]
    start = min(event_times).replace(hour=0, minute=0, second=0, microsecond=0)
    event_day_indices = [
        int((event_time.replace(hour=0, minute=0, second=0, microsecond=0) - start).days // frame_days)
        for event_time in event_times
    ]
    # Keep the scientific time bin at one day, but animate only non-empty daily
    # bins. Otherwise a long sparse catalog spends most frames showing no visual
    # change, which looks like the screen is not updating.
    frame_indices = sorted(set(event_day_indices))
    frame_labels = [(start + timedelta(days=index * frame_days)).date().isoformat() for index in frame_indices]
    magnitudes = [_safe_float(row.get("magnitude"), 0.0) for row in events]
    depths = [_safe_float(row.get("depth_km"), _safe_float(row.get("z_m"), 0.0) / 1000.0) for row in events]
    mag_min, mag_max = min(magnitudes), max(magnitudes)
    depth_min, depth_max = min(depths), max(depths)
    mag_span = max(1e-12, mag_max - mag_min)
    positions: list[float] = []
    colors: list[float] = []
    sizes: list[float] = []
    opacities: list[float] = []
    time_indices: list[float] = []
    ids: list[str] = []
    for row, event_time, mag, depth_km in zip(events, event_times, magnitudes, depths, strict=False):
        z_m = _safe_float(row.get("z_m"), depth_km * 1000.0)
        positions.extend([_safe_float(row.get("x_m")), _safe_float(row.get("y_m")), _plot_z_m(z_m, viz.vertical_exaggeration)])
        colors.extend(_ramp_color(mag, mag_min, mag_max))
        normalized = (mag - mag_min) / mag_span
        sizes.append(
            float(
                np.clip(
                    viz.event_marker_size_min + normalized * (viz.event_marker_size_max - viz.event_marker_size_min),
                    viz.event_marker_size_min,
                    viz.event_marker_size_max,
                )
            )
            * 2.3
        )
        opacities.append(0.86)
        event_day = event_time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_indices.append(float((event_day - start).days // frame_days))
        ids.append(str(row.get("event_id", "")))
    return {
        "positions": positions,
        "colors": colors,
        "sizes": sizes,
        "opacities": opacities,
        "times": time_indices,
        "magnitudes": magnitudes,
        "depths_km": depths,
        "ids": ids,
        "frame_labels": frame_labels,
        "frame_indices": frame_indices,
        "frame_days": frame_days,
        "mode": display_mode,
        "count": len(events),
        "magnitude_range": [mag_min, mag_max],
        "depth_range_km": [depth_min, depth_max],
    }


def _fault_color(props: dict[str, Any], cfg: AppConfig, is_inferred: bool) -> list[float]:
    if not is_inferred:
        return [0.52, 0.66, 0.78]
    value = _safe_float(props.get(cfg.visualization_3d.color_faults_by, props.get("confidence", 0.5)), 0.5)
    return _ramp_color(value, 0.0, 1.0)


def _faults_payload(features: list[dict[str, Any]], cfg: AppConfig) -> dict[str, Any]:
    known_tri: list[float] = []
    known_colors: list[float] = []
    known_lines: list[float] = []
    inferred_tri: list[float] = []
    inferred_colors: list[float] = []
    inferred_lines: list[float] = []
    shown = 0
    for feature in features:
        props = feature.get("properties", {})
        is_inferred = str(props.get("is_inferred", "")).lower() == "true" or props.get("is_inferred") is True
        if is_inferred and not cfg.visualization_3d.show_inferred_faults:
            continue
        if not is_inferred and not cfg.visualization_3d.show_known_faults:
            continue
        center_depth = _safe_float(
            props.get("center_depth_km"),
            (_safe_float(props.get("top_depth_km"), 0.0) + _safe_float(props.get("bottom_depth_km"), 10.0)) / 2.0,
        )
        verts = fault_rectangle_vertices(
            _safe_float(props.get("center_x_m")),
            _safe_float(props.get("center_y_m")),
            center_depth,
            _safe_float(props.get("strike")),
            _safe_float(props.get("dip"), 70.0),
            max(0.5, _safe_float(props.get("length_km"), 5.0)),
            max(0.5, _safe_float(props.get("width_km"), 5.0)),
        )
        display_verts = [[float(v[0]), float(v[1]), _plot_z_m(float(v[2]), cfg.visualization_3d.vertical_exaggeration)] for v in verts]
        triangles = [display_verts[0], display_verts[1], display_verts[2], display_verts[0], display_verts[2], display_verts[3]]
        line_pairs = [
            display_verts[0], display_verts[1], display_verts[1], display_verts[2],
            display_verts[2], display_verts[3], display_verts[3], display_verts[0],
        ]
        color = _fault_color(props, cfg, is_inferred)
        target_tri = inferred_tri if is_inferred else known_tri
        target_colors = inferred_colors if is_inferred else known_colors
        target_lines = inferred_lines if is_inferred else known_lines
        for vertex in triangles:
            target_tri.extend(vertex)
            target_colors.extend(color)
        for vertex in line_pairs:
            target_lines.extend(vertex)
        shown += 1
    return {
        "known_positions": known_tri,
        "known_colors": known_colors,
        "known_line_positions": known_lines,
        "inferred_positions": inferred_tri,
        "inferred_colors": inferred_colors,
        "inferred_line_positions": inferred_lines,
        "displayed_fault_count": shown,
    }


def _context_payload(cfg: AppConfig) -> dict[str, Any]:
    projector = LocalProjector(cfg.region)
    bbox = (float(cfg.region.bbox[0]), float(cfg.region.bbox[1]), float(cfg.region.bbox[2]), float(cfg.region.bbox[3]))
    outlines: list[dict[str, Any]] = []
    for outline in local_context_outlines(bbox, margin_deg=3.0, target_segment_km=1.0):
        positions: list[float] = []
        for lon, lat in outline["coordinates"]:
            x, y = projector.lonlat_to_xy(lon, lat)
            positions.extend([x, y, 0.0])
        outlines.append({"name": outline["name"], "positions": positions})
    min_lon, min_lat, max_lon, max_lat = cfg.region.bbox
    bbox_positions: list[float] = []
    for lon, lat in [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat)]:
        x, y = projector.lonlat_to_xy(lon, lat)
        bbox_positions.extend([x, y, 1000.0])
    return {
        "outlines": outlines,
        "bbox_positions": bbox_positions,
        "outline_vertices": sum(len(item["positions"]) // 3 for item in outlines),
    }


def _bounds(payloads: list[list[float]]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for flat in payloads:
        if not flat:
            continue
        xs.extend(flat[0::3])
        ys.extend(flat[1::3])
        zs.extend(flat[2::3])
    if not xs:
        xs, ys, zs = [0.0], [0.0], [0.0]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "min_z": min(zs), "max_z": max(zs)}


def _html(payload: dict[str, Any]) -> str:
    json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>crust-lite WebGL events and faults</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #071015; color: #e5eef5; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    #gl {{ width: 100vw; height: 100vh; display: block; touch-action: none; }}
    #hud {{ position: fixed; left: 12px; top: 10px; max-width: 650px; background: rgba(7, 16, 21, 0.80); border: 1px solid rgba(180, 205, 220, 0.32); padding: 10px 12px; font-size: 13px; line-height: 1.35; backdrop-filter: blur(5px); }}
    #hud h1 {{ font-size: 16px; margin: 0 0 6px; }}
    #hud label {{ margin-right: 10px; white-space: nowrap; }}
    #timeSlider {{ width: min(58vw, 520px); vertical-align: middle; }}
    .notice {{ color: #ffd29d; font-weight: 600; }}
    button {{ margin-right: 6px; }}
  </style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="hud">
  <h1>WebGL events + faults daily time series</h1>
  <div class="notice">{NOTICE_3D}</div>
  <div id="stats"></div>
  <div>map overlay: outline-only Japan context + target bbox. is_sample_data={str(payload['metadata'].get('is_sample_data', False)).lower()}</div>
  <div>
    <button id="playBtn">再生</button><button id="pauseBtn">一時停止</button>
    <input id="timeSlider" type="range" min="0" max="0" step="1" value="0">
    <span id="frameLabel"></span>
  </div>
  <div>
    <label>mode <select id="modeSelect"><option value="cumulative">cumulative</option><option value="window">window</option></select></label>
    <label>frame interval ms <input id="speedSlider" type="range" min="20" max="500" step="10" value="80"></label><span id="speedLabel">80</span>
  </div>
  <div>
    <label><input id="eventsToggle" type="checkbox" checked>events</label>
    <label><input id="knownToggle" type="checkbox" checked>known faults</label>
    <label><input id="inferredToggle" type="checkbox" checked>inferred faults</label>
    <label><input id="outlineToggle" type="checkbox" checked>Japan outline</label>
    <label><input id="bboxToggle" type="checkbox" checked>bbox</label>
  </div>
  <div>
    <label>event color <select id="colorMode"><option value="0" selected>magnitude</option><option value="1">depth</option><option value="2">time</option></select></label>
    event scale <input id="scaleSlider" type="range" min="0.4" max="8" step="0.05" value="2.4">
    opacity <input id="opacitySlider" type="range" min="0.2" max="2.4" step="0.05" value="1.2">
    trail days <input id="trailSlider" type="range" min="0" max="90" step="1" value="14"><span id="trailLabel">14</span>
  </div>
  <div>drag: rotate / wheel or pinch: zoom / shift+drag or two-finger drag: pan</div>
</div>
<script id="payload" type="application/json">{json_text}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', {{antialias: true, alpha: false}});
if (!gl) throw new Error('WebGL2 is required');
const labels = payload.events.frame_labels;
const frameIndices = payload.events.frame_indices || labels.map((_, i) => i);
let frame = 0, timer = null, frameIntervalMs = payload.metadata.playback_frame_interval_ms || 80;
let mode = payload.events.mode === 'window' ? 1 : 0;
let showEvents = true, showKnown = true, showInferred = true, showOutlines = true, showBbox = true;
let pointScale = 2.4, opacityScale = 1.2, colorMode = 0, trailDays = payload.metadata.event_trail_days || 14;
document.getElementById('stats').textContent = `events=${{payload.events.count}} / faults=${{payload.faults.displayed_fault_count}} / frames=${{labels.length}} / step=${{payload.events.frame_days}} day(s) / renderer=${{payload.metadata.renderer}}`;
const slider = document.getElementById('timeSlider'); slider.max = Math.max(0, labels.length - 1);
const frameLabel = document.getElementById('frameLabel');
const modeSelect = document.getElementById('modeSelect'); modeSelect.value = payload.events.mode === 'window' ? 'window' : 'cumulative';
function shader(type, src) {{ const s=gl.createShader(type); gl.shaderSource(s,src); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }}
function program(vs, fs) {{ const p=gl.createProgram(); gl.attachShader(p,shader(gl.VERTEX_SHADER,vs)); gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs)); gl.linkProgram(p); if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; }}
const eventVS = `#version 300 es
precision highp float;
in vec3 a_pos; in vec3 a_color; in float a_size; in float a_opacity; in float a_time; in float a_mag; in float a_depth;
uniform mat4 u_mvp; uniform float u_frame; uniform int u_mode; uniform float u_pointScale; uniform float u_trailDays;
out vec3 v_color; out float v_opacity; out float v_visible; out float v_mag; out float v_depth; out float v_time; out float v_timeDelta; out float v_current;
void main() {{
  vec4 clip = u_mvp * vec4(a_pos, 1.0);
  gl_Position = clip;
  float delta = abs(a_time - u_frame);
  float current = 1.0 - step(0.5, delta);
  float inWindow = u_trailDays <= 0.0 ? current : 1.0 - step(u_trailDays + 0.5, delta);
  float cumulative = step(a_time, u_frame + 0.01);
  v_visible = u_mode == 0 ? cumulative : inWindow;
  float perspectiveScale = clamp(1.0 / max(0.25, clip.w), 0.35, 3.0);
  float fade = u_trailDays <= 0.0 ? current : 1.0 - clamp(delta / max(u_trailDays, 1.0), 0.0, 1.0);
  float highlight = mix(1.0, 2.6, current);
  gl_PointSize = clamp(a_size * u_pointScale * perspectiveScale * highlight * mix(0.65, 1.0, fade), 2.5, 240.0);
  v_color = a_color; v_opacity = a_opacity; v_mag = a_mag; v_depth = a_depth; v_time = a_time; v_timeDelta = delta; v_current = current;
}}`;
const eventFS = `#version 300 es
precision highp float;
in vec3 v_color; in float v_opacity; in float v_visible; in float v_mag; in float v_depth; in float v_time; in float v_timeDelta; in float v_current;
uniform float u_opacityScale; uniform int u_colorMode; uniform vec2 u_depthRange; uniform vec2 u_timeRange; uniform float u_trailDays;
out vec4 outColor;
vec3 ramp(float t) {{
  t = clamp(t, 0.0, 1.0);
  vec3 a = vec3(0.14,0.18,0.44), b = vec3(0.08,0.52,0.72), c = vec3(0.64,0.78,0.25), d = vec3(0.96,0.54,0.16);
  if (t < 0.35) return mix(a,b,t/0.35);
  if (t < 0.70) return mix(b,c,(t-0.35)/0.35);
  return mix(c,d,(t-0.70)/0.30);
}}
void main() {{
  if (v_visible < 0.5) discard;
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;
  float gaussian = exp(-3.25 * r2);
  vec3 color = v_color;
  if (u_colorMode == 1) color = ramp((v_depth - u_depthRange.x) / max(1e-6, u_depthRange.y - u_depthRange.x));
  if (u_colorMode == 2) color = ramp((v_time - u_timeRange.x) / max(1e-6, u_timeRange.y - u_timeRange.x));
  float fade = u_trailDays <= 0.0 ? 1.0 : 1.0 - clamp(v_timeDelta / max(u_trailDays, 1.0), 0.0, 1.0);
  color = mix(color, vec3(1.0, 0.96, 0.42), 0.68 * v_current);
  float alpha = clamp(v_opacity * u_opacityScale * gaussian * mix(0.25, 1.0, fade), 0.0, 0.98);
  outColor = vec4(color, alpha);
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
const eventProg=program(eventVS,eventFS), meshProg=program(meshVS,meshFS), lineProg=program(lineVS,lineFS);
const b = payload.bounds;
const center = [(b.min_x+b.max_x)/2, (b.min_y+b.max_y)/2, (b.min_z+b.max_z)/2];
const span = Math.max(b.max_x-b.min_x, b.max_y-b.min_y, b.max_z-b.min_z, 1);
function normPositions(src) {{ const out = new Float32Array(src.length); for (let i=0;i<src.length;i+=3) {{ out[i]=(src[i]-center[0])/span*2.0; out[i+1]=(src[i+1]-center[1])/span*2.0; out[i+2]=(src[i+2]-center[2])/span*2.0; }} return out; }}
function buf(data, target=gl.ARRAY_BUFFER) {{ const b=gl.createBuffer(); gl.bindBuffer(target,b); gl.bufferData(target,data,gl.STATIC_DRAW); return b; }}
function makeMesh(positions, colors) {{ return {{ n: positions.length/3, pos: buf(normPositions(positions)), color: buf(new Float32Array(colors)) }}; }}
function makeLine(positions) {{ return {{ n: positions.length/3, pos: buf(normPositions(positions)) }}; }}
const events = {{
  n: payload.events.positions.length/3,
  pos: buf(normPositions(payload.events.positions)), color: buf(new Float32Array(payload.events.colors)), size: buf(new Float32Array(payload.events.sizes)),
  opacity: buf(new Float32Array(payload.events.opacities)), time: buf(new Float32Array(payload.events.times)), mag: buf(new Float32Array(payload.events.magnitudes)), depth: buf(new Float32Array(payload.events.depths_km))
}};
const knownMesh = makeMesh(payload.faults.known_positions, payload.faults.known_colors);
const inferredMesh = makeMesh(payload.faults.inferred_positions, payload.faults.inferred_colors);
const knownLines = makeLine(payload.faults.known_line_positions);
const inferredLines = makeLine(payload.faults.inferred_line_positions);
const bboxLine = makeLine(payload.context.bbox_positions);
const outlineLines = payload.context.outlines.map(o => ({{name:o.name, ...makeLine(o.positions)}}));
function attrib(p,name,buffer,size) {{ const loc=gl.getAttribLocation(p,name); gl.bindBuffer(gl.ARRAY_BUFFER,buffer); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0); }}
function mat4mul(a,b) {{ const o=new Float32Array(16); for(let c=0;c<4;c++) for(let r=0;r<4;r++) o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3]; return o; }}
function perspective(fovy,aspect,near,far) {{ const f=1/Math.tan(fovy/2), nf=1/(near-far), o=new Float32Array(16); o[0]=f/aspect; o[5]=f; o[10]=(far+near)*nf; o[11]=-1; o[14]=2*far*near*nf; return o; }}
function lookAt(eye,target,up) {{
  let zx=eye[0]-target[0], zy=eye[1]-target[1], zz=eye[2]-target[2]; let zl=1/Math.hypot(zx,zy,zz); zx*=zl; zy*=zl; zz*=zl;
  let xx=up[1]*zz-up[2]*zy, xy=up[2]*zx-up[0]*zz, xz=up[0]*zy-up[1]*zx; let xl=1/Math.hypot(xx,xy,xz); xx*=xl; xy*=xl; xz*=xl;
  const yx=zy*xz-zz*xy, yy=zz*xx-zx*xz, yz=zx*xy-zy*xx; const o=new Float32Array(16);
  o[0]=xx; o[1]=yx; o[2]=zx; o[4]=xy; o[5]=yy; o[6]=zy; o[8]=xz; o[9]=yz; o[10]=zz; o[15]=1;
  o[12]=-(xx*eye[0]+xy*eye[1]+xz*eye[2]); o[13]=-(yx*eye[0]+yy*eye[1]+yz*eye[2]); o[14]=-(zx*eye[0]+zy*eye[1]+zz*eye[2]); return o;
}}
let yaw=0.72, pitch=0.46, dist=3.2, pan=[0,0,0];
const pointers = new Map(); let lastCentroid=null, lastPinchDistance=0, lastPointer=null, panning=false;
function mvp() {{ const eye=[dist*Math.cos(pitch)*Math.sin(yaw)+pan[0], dist*Math.cos(pitch)*Math.cos(yaw)+pan[1], dist*Math.sin(pitch)+pan[2]]; return mat4mul(perspective(45*Math.PI/180, canvas.width/canvas.height, 0.01, 100.0), lookAt(eye, pan, [0,0,1])); }}
function drawLine(line, color) {{ if (line.n <= 0) return; gl.useProgram(lineProg); gl.uniformMatrix4fv(gl.getUniformLocation(lineProg,'u_mvp'), false, mvp()); gl.uniform4f(gl.getUniformLocation(lineProg,'u_color'), color[0],color[1],color[2],color[3]); attrib(lineProg,'a_pos',line.pos,3); gl.drawArrays(gl.LINES,0,line.n); }}
function drawLineStrip(line, color) {{ if (line.n <= 0) return; gl.useProgram(lineProg); gl.uniformMatrix4fv(gl.getUniformLocation(lineProg,'u_mvp'), false, mvp()); gl.uniform4f(gl.getUniformLocation(lineProg,'u_color'), color[0],color[1],color[2],color[3]); attrib(lineProg,'a_pos',line.pos,3); gl.drawArrays(gl.LINE_STRIP,0,line.n); }}
function drawMesh(mesh, alpha) {{ if (mesh.n <= 0) return; gl.useProgram(meshProg); gl.uniformMatrix4fv(gl.getUniformLocation(meshProg,'u_mvp'), false, mvp()); gl.uniform1f(gl.getUniformLocation(meshProg,'u_alpha'), alpha); attrib(meshProg,'a_pos',mesh.pos,3); attrib(meshProg,'a_color',mesh.color,3); gl.drawArrays(gl.TRIANGLES,0,mesh.n); }}
function render() {{
  gl.clearColor(0.027,0.063,0.082,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  if (showOutlines) for (const o of outlineLines) drawLineStrip(o, [0.82,1.0,0.78,0.95]);
  if (showBbox) drawLineStrip(bboxLine, [1.0,0.37,0.18,0.95]);
  gl.depthMask(false);
  if (showKnown) {{ drawMesh(knownMesh,0.14); drawLine(knownLines,[0.76,0.90,1.0,0.72]); }}
  if (showInferred) {{ drawMesh(inferredMesh,0.18); drawLine(inferredLines,[1.0,0.72,0.18,0.78]); }}
  if (showEvents && events.n > 0) {{
    gl.disable(gl.DEPTH_TEST);
    const matrix = mvp(); gl.useProgram(eventProg); gl.uniformMatrix4fv(gl.getUniformLocation(eventProg,'u_mvp'), false, matrix);
    const currentFrameTime = frameIndices[frame] ?? frame;
    gl.uniform1f(gl.getUniformLocation(eventProg,'u_frame'), currentFrameTime); gl.uniform1i(gl.getUniformLocation(eventProg,'u_mode'), mode);
    gl.uniform1f(gl.getUniformLocation(eventProg,'u_pointScale'), pointScale); gl.uniform1f(gl.getUniformLocation(eventProg,'u_opacityScale'), opacityScale); gl.uniform1i(gl.getUniformLocation(eventProg,'u_colorMode'), colorMode); gl.uniform1f(gl.getUniformLocation(eventProg,'u_trailDays'), mode == 1 ? trailDays : 0.0);
    gl.uniform2f(gl.getUniformLocation(eventProg,'u_depthRange'), payload.events.depth_range_km[0], payload.events.depth_range_km[1]); gl.uniform2f(gl.getUniformLocation(eventProg,'u_timeRange'), Math.min(...frameIndices), Math.max(1, Math.max(...frameIndices)));
    attrib(eventProg,'a_pos',events.pos,3); attrib(eventProg,'a_color',events.color,3); attrib(eventProg,'a_size',events.size,1); attrib(eventProg,'a_opacity',events.opacity,1); attrib(eventProg,'a_time',events.time,1); attrib(eventProg,'a_mag',events.mag,1); attrib(eventProg,'a_depth',events.depth,1);
    gl.drawArrays(gl.POINTS,0,events.n);
    gl.enable(gl.DEPTH_TEST);
  }}
  gl.depthMask(true);
}}
function updateFrameLabel() {{ frameLabel.textContent = `${{labels[frame] || 'no_events'}}  ${{frame+1}}/${{labels.length}} / dayIndex=${{frameIndices[frame] ?? frame}}`; slider.value = frame; }}
function setFrame(value) {{ frame = Math.max(0, Math.min(labels.length - 1, Number(value) || 0)); updateFrameLabel(); render(); }}
function play() {{ if (timer) clearInterval(timer); timer = setInterval(() => setFrame((frame + 1) % labels.length), frameIntervalMs); }}
function pause() {{ if (timer) clearInterval(timer); timer = null; }}
function resize() {{ const dpr=Math.min(devicePixelRatio||1,4); canvas.width=Math.floor(innerWidth*dpr); canvas.height=Math.floor(innerHeight*dpr); gl.viewport(0,0,canvas.width,canvas.height); render(); }}
function pointerCentroid() {{ let x=0,y=0; for (const p of pointers.values()) {{ x+=p.x; y+=p.y; }} const n=Math.max(1,pointers.size); return [x/n,y/n]; }}
function pointerDistance() {{ const pts=Array.from(pointers.values()); if(pts.length<2) return 0; return Math.hypot(pts[0].x-pts[1].x, pts[0].y-pts[1].y); }}
function panBy(dx,dy) {{ pan[0]-=dx/420; pan[2]+=dy/420; }}
canvas.addEventListener('pointerdown', e => {{ e.preventDefault(); canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId,{{x:e.clientX,y:e.clientY}}); panning=e.shiftKey||pointers.size>=2; lastPointer=[e.clientX,e.clientY]; lastCentroid=pointerCentroid(); lastPinchDistance=pointerDistance(); }}, {{passive:false}});
canvas.addEventListener('pointermove', e => {{
  if(!pointers.has(e.pointerId)) return; e.preventDefault(); pointers.set(e.pointerId,{{x:e.clientX,y:e.clientY}});
  if(pointers.size>=2) {{ const c=pointerCentroid(), d=pointerDistance(); if(lastCentroid) panBy(c[0]-lastCentroid[0], c[1]-lastCentroid[1]); if(lastPinchDistance>0&&d>0) dist=Math.max(0.55,Math.min(12,dist*(lastPinchDistance/d))); lastCentroid=c; lastPinchDistance=d; }}
  else if(lastPointer) {{ const dx=e.clientX-lastPointer[0], dy=e.clientY-lastPointer[1]; lastPointer=[e.clientX,e.clientY]; if(panning||e.shiftKey) panBy(dx,dy); else {{ yaw+=dx*0.006; pitch=Math.max(-1.25,Math.min(1.25,pitch+dy*0.006)); }} }}
  render();
}}, {{passive:false}});
function endPointer(e) {{ if(pointers.has(e.pointerId)) pointers.delete(e.pointerId); lastPointer=null; lastCentroid=pointers.size?pointerCentroid():null; lastPinchDistance=pointerDistance(); panning=pointers.size>=2; }}
canvas.addEventListener('pointerup', endPointer); canvas.addEventListener('pointercancel', endPointer); canvas.addEventListener('lostpointercapture', endPointer);
canvas.addEventListener('wheel', e => {{ e.preventDefault(); dist=Math.max(0.55,Math.min(12,dist*Math.exp(e.deltaY*0.001))); render(); }}, {{passive:false}});
slider.addEventListener('input', e => setFrame(e.target.value));
document.getElementById('playBtn').addEventListener('click', play); document.getElementById('pauseBtn').addEventListener('click', pause);
modeSelect.addEventListener('change', e => {{ mode = e.target.value === 'window' ? 1 : 0; render(); }});
document.getElementById('speedSlider').addEventListener('input', e => {{ frameIntervalMs = Number(e.target.value); document.getElementById('speedLabel').textContent = String(frameIntervalMs); if(timer) play(); }});
document.getElementById('eventsToggle').addEventListener('change', e => {{ showEvents=e.target.checked; render(); }});
document.getElementById('knownToggle').addEventListener('change', e => {{ showKnown=e.target.checked; render(); }});
document.getElementById('inferredToggle').addEventListener('change', e => {{ showInferred=e.target.checked; render(); }});
document.getElementById('outlineToggle').addEventListener('change', e => {{ showOutlines=e.target.checked; render(); }});
document.getElementById('bboxToggle').addEventListener('change', e => {{ showBbox=e.target.checked; render(); }});
document.getElementById('colorMode').addEventListener('change', e => {{ colorMode=Number(e.target.value); render(); }});
document.getElementById('scaleSlider').addEventListener('input', e => {{ pointScale=Number(e.target.value); render(); }});
document.getElementById('opacitySlider').addEventListener('input', e => {{ opacityScale=Number(e.target.value); render(); }});
document.getElementById('trailSlider').addEventListener('input', e => {{ trailDays=Number(e.target.value); document.getElementById('trailLabel').textContent=String(trailDays); render(); }});
addEventListener('resize', resize); updateFrameLabel(); resize();
</script>
</body>
</html>
'''


def write_webgl_events_faults(
    cfg: AppConfig,
    paths: ProjectPaths,
    metadata: dict[str, Any],
    mode: str | None = None,
    time_bin_days: int | None = None,
    max_events: int | None = None,
) -> None:
    events = read_table(paths.data_interim / "event_qc.parquet")
    selected_events, event_decimation = _select_events(events, max_events or cfg.visualization_3d.max_events)
    known = read_features(paths.data_processed / "fault_segment.gpkg") if (paths.data_processed / "fault_segment.gpkg").exists() else []
    inferred = read_features(paths.data_processed / "inferred_faults.gpkg") if (paths.data_processed / "inferred_faults.gpkg").exists() else []
    faults, fault_decimation = _limit_faults(known + inferred, cfg.visualization_3d.max_fault_segments)
    display_mode = mode or cfg.visualization_3d.mode
    # Events/faults use one-day indexed frames by default. Unlike Plotly frames,
    # the WebGL renderer stores each event once and filters by time in the shader.
    frame_days = max(1, int(time_bin_days)) if time_bin_days is not None else 1
    events_payload = _events_payload(selected_events, cfg, display_mode, frame_days)
    faults_payload = _faults_payload(faults, cfg)
    context_payload = _context_payload(cfg)
    bounds = _bounds(
        [
            events_payload["positions"],
            faults_payload["known_positions"],
            faults_payload["known_line_positions"],
            faults_payload["inferred_positions"],
            faults_payload["inferred_line_positions"],
            context_payload["bbox_positions"],
            *[item["positions"] for item in context_payload["outlines"]],
        ]
    )
    local_meta = {
        **metadata,
        "renderer": "webgl2_event_fault_point_sprite",
        "gaussian_shader": "fragment_alpha=opacity*exp(-3.25*r2)",
        "event_frame_strategy": "single_event_buffer_shader_time_filter_no_frame_duplication_active_event_days_only",
        "event_frame_selection": "active_event_days_only",
        "event_time_step_days": frame_days,
        "actual_time_bin_days": frame_days,
        "playback_frame_interval_ms": 80,
        "event_trail_days": 14,
        "event_animation_visibility": "window mode highlights the current day and fades nearby event days so playback visibly changes",
        "event_depth_test": "disabled for event points so static fault/context geometry cannot hide the time animation",
        "camera_persistence": "single WebGL camera state; slider changes do not recreate the scene",
        "old_frame_residue_prevention": "events are filtered by a uniform frame index in one draw call; previous frame geometry is not appended",
        "original_event_count": len(events),
        "displayed_event_count": len(selected_events),
        "original_fault_count": len(known) + len(inferred),
        "displayed_fault_count": faults_payload["displayed_fault_count"],
        "original_frame_count": len(events_payload["frame_indices"]),
        "displayed_frame_count": len(events_payload["frame_labels"]),
        "decimation_method": ", ".join(sorted({event_decimation, "none_webgl_daily_frames", fault_decimation})),
        "events_faults_map_overlay": "outline-only Japan context + target bbox in local CRS",
        "japan_outline_vertices": context_payload["outline_vertices"],
        "not_prediction": True,
    }
    payload = {
        "metadata": local_meta,
        "events": events_payload,
        "faults": faults_payload,
        "context": context_payload,
        "bounds": bounds,
    }
    out = paths.outputs_3d / "events_faults_timeseries.html"
    out.write_text(_html(payload), encoding="utf-8")
    metadata.update(local_meta)
