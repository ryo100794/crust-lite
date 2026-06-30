from __future__ import annotations

import json
import math
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector
from crust_lite.paths import ProjectPaths
from crust_lite.viz.japan_outline import JapanOutline, local_context_outlines
from crust_lite.viz.tectonics import TectonicLine, tectonic_context_from_config


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_existing_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if key not in row:
            continue
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _type_code(value: str) -> int:
    return {"direct": 0, "reflected": 1, "scattered": 2, "residual": 3}.get(value, 3)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_integer_km_depth(z_m: float) -> bool:
    depth_km = z_m / 1000.0
    return abs(depth_km - round(depth_km)) < 1.0e-6


def _late_delay_clipped(row: dict[str, Any], max_delay_s: float) -> bool:
    if "late_delay_clipped" in row:
        return _boolish(row.get("late_delay_clipped"))
    primitive_type = str(row.get("primitive_type", "direct") or "direct")
    if primitive_type == "direct":
        return False
    delay_s = float(row.get("late_phase_delay_s", 0.0) or 0.0)
    return delay_s >= max(0.0, max_delay_s - 1.0e-6)


def _depth_flag_code(row: dict[str, Any], max_delay_s: float) -> int:
    primitive_type = str(row.get("primitive_type", "direct") or "direct")
    if primitive_type == "direct":
        z_m = float(row.get("source_event_z_m", row.get("z_m", 0.0)) or 0.0)
        return 1 if _is_integer_km_depth(z_m) else 0
    if _late_delay_clipped(row, max_delay_s):
        return 2
    return 3


def _top_depth_bins(rows: list[dict[str, Any]], key: str, bin_km: float, limit: int = 12) -> list[dict[str, Any]]:
    counts: dict[float, int] = {}
    for row in rows:
        if key not in row:
            continue
        depth_km = float(row.get(key, 0.0) or 0.0) / 1000.0
        binned = round(round(depth_km / bin_km) * bin_km, 3)
        counts[binned] = counts.get(binned, 0) + 1
    return [
        {"depth_km": depth, "count": count}
        for depth, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _depth_center_m(row: dict[str, Any]) -> float:
    p50_km = _finite_float(row.get("depth_p50_km"))
    if p50_km is not None:
        return p50_km * 1000.0
    return float(row.get("z_m", 0.0) or 0.0)


def _summary(values: list[float], digits: int = 6) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0}
    n = len(finite)

    def quantile(q: float) -> float:
        if n == 1:
            return finite[0]
        pos = (n - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return finite[lo]
        return finite[lo] * (hi - pos) + finite[hi] * (pos - lo)

    return {
        "count": n,
        "min": round(finite[0], digits),
        "p05": round(quantile(0.05), digits),
        "p50": round(quantile(0.50), digits),
        "mean": round(sum(finite) / n, digits),
        "p95": round(quantile(0.95), digits),
        "max": round(finite[-1], digits),
    }


def _depth_uncertainty_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    percentile_rows = 0
    complete_percentile_rows = 0
    interval_widths: list[float] = []
    p05_values: list[float] = []
    p50_values: list[float] = []
    p95_values: list[float] = []
    velocity_spans: list[float] = []
    velocity_samples: list[float] = []
    refinement_offsets: list[float] = []
    refinement_score_gains: list[float] = []
    for row in rows:
        p05 = _finite_float(row.get("depth_p05_km"))
        p50 = _finite_float(row.get("depth_p50_km"))
        p95 = _finite_float(row.get("depth_p95_km"))
        if p05 is not None or p50 is not None or p95 is not None:
            percentile_rows += 1
        if p05 is not None:
            p05_values.append(p05)
        if p50 is not None:
            p50_values.append(p50)
        if p95 is not None:
            p95_values.append(p95)
        if p05 is not None and p50 is not None and p95 is not None:
            complete_percentile_rows += 1
            interval_widths.append(max(0.0, p95 - p05))

        velocity_min = _finite_float(row.get("depth_velocity_min_km_s"))
        velocity_max = _finite_float(row.get("depth_velocity_max_km_s"))
        if velocity_min is not None and velocity_max is not None:
            velocity_spans.append(max(0.0, velocity_max - velocity_min))
        samples = _finite_float(row.get("depth_velocity_samples"))
        if samples is not None:
            velocity_samples.append(samples)

        dx_m = _finite_float(row.get("projection_refinement_dx_m"))
        dy_m = _finite_float(row.get("projection_refinement_dy_m"))
        if dx_m is not None and dy_m is not None:
            refinement_offsets.append(math.hypot(dx_m, dy_m))
        score_gain = _finite_float(row.get("projection_refinement_score_gain"))
        if score_gain is not None:
            refinement_score_gains.append(score_gain)

    return {
        "available": percentile_rows > 0,
        "rows_with_any_depth_percentile": percentile_rows,
        "rows_with_complete_p05_p50_p95": complete_percentile_rows,
        "depth_p05_km": _summary(p05_values),
        "depth_p50_km": _summary(p50_values),
        "depth_p95_km": _summary(p95_values),
        "p05_p95_width_km": _summary(interval_widths),
        "depth_velocity_span_km_s": _summary(velocity_spans),
        "depth_velocity_samples": _summary(velocity_samples, digits=3),
        "depth_uncertainty_method_counts": _count_existing_values(rows, "depth_uncertainty_method"),
        "projection_refinement_offset_m": _summary(refinement_offsets, digits=3),
        "projection_refinement_score_gain": _summary(refinement_score_gains),
        "projection_refinement_method_counts": _count_existing_values(rows, "projection_refinement_method"),
        "display_semantics": "WebGL z is the depth center: depth_p50_km when available, otherwise legacy z_m. The p05-p95 interval remains metadata for uncertainty interpretation rather than a visual post-processing filter.",
    }


def _depth_diagnostics(config: AppConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_delay_s = float(config.waveform_array.late_phase_max_delay_s)
    flag_counts = {"continuous_or_unflagged": 0, "catalog_integer_km_direct": 0, "late_delay_window_clipped": 0, "late_model_derived": 0}
    source_integer = 0
    source_count = 0
    late_count = 0
    clipped_count = 0
    for row in rows:
        flag = _depth_flag_code(row, max_delay_s)
        if flag == 1:
            flag_counts["catalog_integer_km_direct"] += 1
        elif flag == 2:
            flag_counts["late_delay_window_clipped"] += 1
        elif flag == 3:
            flag_counts["late_model_derived"] += 1
        else:
            flag_counts["continuous_or_unflagged"] += 1

        if "source_event_z_m" in row:
            source_count += 1
            if _is_integer_km_depth(float(row.get("source_event_z_m", 0.0) or 0.0)):
                source_integer += 1
        if str(row.get("primitive_type", "direct") or "direct") != "direct":
            late_count += 1
            if _late_delay_clipped(row, max_delay_s):
                clipped_count += 1

    return {
        "depth_coordinate_note": "WebGL z is plotted as the depth center (depth_p50_km when present, otherwise legacy z_m). Depth is not directly observed by the waveform array; p05-p95 metadata represents computational uncertainty.",
        "late_phase_max_delay_s": max_delay_s,
        "source_depth_integer_km_fraction": round(source_integer / source_count, 6) if source_count else 0.0,
        "late_delay_clipped_count": clipped_count,
        "late_delay_clipped_fraction_of_late": round(clipped_count / late_count, 6) if late_count else 0.0,
        "depth_flag_counts": flag_counts,
        "top_splat_depth_bins_1km": _top_depth_bins(rows, "z_m", 1.0),
        "top_source_depth_bins_1km": _top_depth_bins(rows, "source_event_z_m", 1.0),
        "uncertainty": _depth_uncertainty_summary(rows),
        "interpretation": "Layer-like bands should be read through the computational depth uncertainty model, not as display smoothing artifacts. Catalog-depth quantization, late-delay window clipping, velocity-range sampling, and projection refinement are retained as diagnostic metadata so apparent layers can be checked against the p05-p95 interval and independent velocity/plate constraints.",
    }


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
    dense_outlines: list[JapanOutline] = []
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
    for dense_outline in dense_outlines:
        flat: list[float] = []
        for lon, lat in dense_outline["coordinates"]:
            x_m, y_m = projector.lonlat_to_xy(lon, lat)
            flat.extend([_round(x_m), _round(y_m), 0.0])
        outline_payload.append({"name": dense_outline["name"], "source": dense_outline["source"], "positions": flat})
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


def _tectonics_payload(config: AppConfig) -> dict[str, Any]:
    projector = LocalProjector(config.region)
    vertical = config.visualization_3d.vertical_exaggeration
    context = tectonic_context_from_config(config)

    def project_line(line: TectonicLine) -> dict[str, Any]:
        flat: list[float] = []
        for lon, lat, depth_km in line["coordinates"]:
            x_m, y_m = projector.lonlat_to_xy(float(lon), float(lat))
            z_plot = -1.0 * float(depth_km) * 1000.0 * vertical
            flat.extend([_round(x_m), _round(y_m), _round(z_plot)])
        return {
            "name": line["name"],
            "plate": line["plate"],
            "kind": line["kind"],
            "color": list(line["color"]),
            "positions": flat,
            "vertices": len(flat) // 3,
        }

    boundaries = [project_line(line) for line in context["boundaries"]]
    interfaces = [project_line(line) for line in context["interfaces"]]
    return {
        "boundaries": boundaries,
        "interfaces": interfaces,
        "boundary_vertices": sum(line["vertices"] for line in boundaries),
        "interface_vertices": sum(line["vertices"] for line in interfaces),
        "source": context["source"],
        "note": context["note"],
        "literature_based": context["literature_based"],
        "model_source": context["model_source"],
        "source_files": context["source_files"],
        "fallback_used": context["fallback_used"],
        "default_show": context["default_show"],
    }


def _splat_payload(config: AppConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions: list[float] = []
    colors: list[float] = []
    sizes: list[float] = []
    opacities: list[float] = []
    types: list[float] = []
    depth_flags: list[float] = []
    amplitudes: list[float] = []
    depth_centers_km: list[float] = []
    depth_p05_km: list[float | None] = []
    depth_p50_km: list[float | None] = []
    depth_p95_km: list[float | None] = []
    depth_velocity_samples: list[float | None] = []
    projection_refinement_offsets_m: list[float | None] = []
    projection_refinement_score_gains: list[float | None] = []
    line_positions: list[float] = []
    vertical = config.visualization_3d.vertical_exaggeration
    max_delay_s = float(config.waveform_array.late_phase_max_delay_s)
    amplitude_values = [float(row.get("amplitude", 0.0) or 0.0) for row in rows]
    min_amplitude = min(amplitude_values) if amplitude_values else 0.0
    max_amplitude = max(amplitude_values) if amplitude_values else 1.0
    amplitude_span = max(max_amplitude - min_amplitude, 1.0e-12)
    line_rows = sorted(
        rows,
        key=lambda row: (float(row.get("beam_power", 0.0) or 0.0), float(row.get("amplitude", 0.0) or 0.0)),
        reverse=True,
    )[: min(1600, len(rows))]
    line_ids = {id(row) for row in line_rows}
    for row in rows:
        x_m = float(row.get("x_m", 0.0) or 0.0)
        y_m = float(row.get("y_m", 0.0) or 0.0)
        depth_center_m = _depth_center_m(row)
        z_plot = -1.0 * depth_center_m * vertical
        positions.extend([_round(x_m), _round(y_m), _round(z_plot)])
        amplitude = float(row.get("amplitude", 0.0) or 0.0)
        relative_intensity = max(0.0, min(1.0, (amplitude - min_amplitude) / amplitude_span))
        grayscale = 0.10 + 0.90 * math.sqrt(relative_intensity)
        colors.extend([_round(grayscale, 5), _round(grayscale, 5), _round(grayscale, 5)])
        sigma_xy = max(float(row.get("sigma_x_m", 1.0) or 1.0), float(row.get("sigma_y_m", 1.0) or 1.0))
        sizes.append(_round(sigma_xy, 3))
        opacities.append(_round(float(row.get("opacity", 0.6) or 0.6), 5))
        primitive_type = str(row.get("primitive_type", "direct") or "direct")
        types.append(float(_type_code(primitive_type)))
        depth_flags.append(float(_depth_flag_code(row, max_delay_s)))
        amplitudes.append(_round(float(row.get("amplitude", 0.0) or 0.0), 5))
        depth_centers_km.append(_round(depth_center_m / 1000.0, 6))
        p05 = _finite_float(row.get("depth_p05_km"))
        p50 = _finite_float(row.get("depth_p50_km"))
        p95 = _finite_float(row.get("depth_p95_km"))
        depth_p05_km.append(round(p05, 6) if p05 is not None else None)
        depth_p50_km.append(round(p50, 6) if p50 is not None else None)
        depth_p95_km.append(round(p95, 6) if p95 is not None else None)
        samples = _finite_float(row.get("depth_velocity_samples"))
        depth_velocity_samples.append(round(samples, 3) if samples is not None else None)
        dx_m = _finite_float(row.get("projection_refinement_dx_m"))
        dy_m = _finite_float(row.get("projection_refinement_dy_m"))
        projection_refinement_offsets_m.append(round(math.hypot(dx_m, dy_m), 3) if dx_m is not None and dy_m is not None else None)
        score_gain = _finite_float(row.get("projection_refinement_score_gain"))
        projection_refinement_score_gains.append(round(score_gain, 6) if score_gain is not None else None)
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
        "depth_flags": depth_flags,
        "amplitudes": amplitudes,
        "depth_centers_km": depth_centers_km,
        "depth_p05_km": depth_p05_km,
        "depth_p50_km": depth_p50_km,
        "depth_p95_km": depth_p95_km,
        "depth_velocity_samples": depth_velocity_samples,
        "projection_refinement_offsets_m": projection_refinement_offsets_m,
        "projection_refinement_score_gains": projection_refinement_score_gains,
        "source_lines": line_positions,
        "line_segments": len(line_positions) // 6,
    }


def _bounds_from_payload(splats: dict[str, Any], terrain: dict[str, Any], tectonics: dict[str, Any] | None = None) -> dict[str, float]:
    values = [splats["positions"], terrain["positions"]]
    values.extend(outline["positions"] for outline in terrain.get("outlines", []))
    if tectonics is not None:
        values.extend(line["positions"] for line in tectonics.get("boundaries", []))
        values.extend(line["positions"] for line in tectonics.get("interfaces", []))
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
    <label>color <select id="colorMode"><option value="0" selected>grayscale intensity</option><option value="1">path type overlay</option><option value="2">depth diagnostics</option></select></label>
    <span>classification overlay: direct / reflected / scattered / residual</span>
  </div>
  <div>
    <label><input type="checkbox" data-depth-flag="0" checked>continuous/unflagged depth</label>
    <label><input type="checkbox" data-depth-flag="1" checked>catalog integer-km direct</label>
    <label><input type="checkbox" data-depth-flag="2" checked>delay-window clipped</label>
    <label><input type="checkbox" data-depth-flag="3" checked>late model depth</label>
  </div>
  <div>depth diagnostics: amber=catalog-rounded direct, red=delay-window clipped, cyan=late model-derived.</div>
  <div>depth uncertainty: z uses median/center depth; p05-p95 stays metadata and no longer enlarges point size.</div>
  <div>
    <label><input type="checkbox" data-type="0" checked>direct</label>
    <label><input type="checkbox" data-type="1" checked>reflected</label>
    <label><input type="checkbox" data-type="2" checked>scattered</label>
    <label><input type="checkbox" data-type="3" checked>residual</label>
  </div>
  <div>
    <label><input id="outlineToggle" type="checkbox" checked>Japan outline</label>
    <label><input id="plateBoundaryToggle" type="checkbox">plate boundaries</label>
    <label><input id="plateInterfaceToggle" type="checkbox">slab/interface lines</label>
    <label><input id="lineToggle" type="checkbox">source-projection guides</label>
  </div>
  <div id="plateOverlayNote"></div>
  <div>z note: waveform data do not directly observe depth; direct/late z is an uncertainty-aware computational center.</div>
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
const depthUncertainty = payload.metadata.depth_diagnostics.uncertainty || {{rows_with_complete_p05_p50_p95: 0}};
document.getElementById('stats').textContent =
  `splats=${{payload.metadata.displayed_splats}} / depth p05-p95 rows=${{depthUncertainty.rows_with_complete_p05_p50_p95}} / clipped late=${{payload.metadata.depth_diagnostics.late_delay_clipped_count}} / outline vertices=${{payload.terrain.outline_vertices}}`;
document.getElementById('plateOverlayNote').textContent = payload.metadata.tectonic_overlay_note;

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
in vec3 a_pos; in vec3 a_color; in float a_size; in float a_opacity; in float a_type; in float a_depthFlag;
uniform mat4 u_mvp; uniform float u_pointScale;
out vec3 v_color; out float v_opacity; out float v_type; out float v_depthFlag;
void main() {{
  vec4 clip = u_mvp * vec4(a_pos, 1.0);
  gl_Position = clip;
  float perspectiveScale = clamp(1.0 / max(0.25, clip.w), 0.35, 3.0);
  gl_PointSize = clamp(a_size * u_pointScale * perspectiveScale, 2.0, 384.0);
  v_color = a_color; v_opacity = a_opacity; v_type = a_type; v_depthFlag = a_depthFlag;
}}`;
const splatFS = `#version 300 es
precision highp float;
in vec3 v_color; in float v_opacity; in float v_type; in float v_depthFlag;
uniform vec4 u_visible; uniform vec4 u_depthFlagVisible; uniform float u_opacityScale; uniform int u_colorMode;
out vec4 outColor;
vec3 pathTypeColor(float t) {{
  if (t < 0.5) return vec3(0.30, 0.72, 1.00);
  if (t < 1.5) return vec3(1.00, 0.62, 0.16);
  if (t < 2.5) return vec3(0.72, 0.42, 0.95);
  return vec3(0.68, 0.68, 0.68);
}}
vec3 depthFlagColor(float f) {{
  if (f < 0.5) return vec3(0.82, 0.82, 0.82);
  if (f < 1.5) return vec3(1.00, 0.72, 0.18);
  if (f < 2.5) return vec3(1.00, 0.22, 0.18);
  return vec3(0.20, 0.92, 1.00);
}}
void main() {{
  float vis = v_type < 0.5 ? u_visible.x : (v_type < 1.5 ? u_visible.y : (v_type < 2.5 ? u_visible.z : u_visible.w));
  float depthVis = v_depthFlag < 0.5 ? u_depthFlagVisible.x : (v_depthFlag < 1.5 ? u_depthFlagVisible.y : (v_depthFlag < 2.5 ? u_depthFlagVisible.z : u_depthFlagVisible.w));
  if (vis < 0.5 || depthVis < 0.5) discard;
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;
  float gaussian = exp(-3.25 * r2);
  float alpha = clamp(v_opacity * u_opacityScale * gaussian, 0.0, 0.92);
  vec3 color = u_colorMode == 1 ? pathTypeColor(v_type) : (u_colorMode == 2 ? depthFlagColor(v_depthFlag) : v_color);
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
  depthFlag: buf(new Float32Array(payload.splats.depth_flags)),
}};
const terrain = {{
  n: payload.terrain.indices.length,
  pos: buf(normPositions(payload.terrain.positions)),
  color: buf(new Float32Array(payload.terrain.colors)),
  idx: buf(new Uint32Array(payload.terrain.indices), gl.ELEMENT_ARRAY_BUFFER),
}};
const sourceLines = {{ n: payload.splats.source_lines.length / 3, pos: buf(normPositions(payload.splats.source_lines)) }};
const outlineBuffers = payload.terrain.outlines.map(o => ({{ name:o.name, n:o.positions.length/3, pos:buf(normPositions(o.positions)) }}));
const plateBoundaryBuffers = payload.tectonics.boundaries.map(o => ({{ name:o.name, color:o.color, n:o.positions.length/3, pos:buf(normPositions(o.positions)) }}));
const plateInterfaceBuffers = payload.tectonics.interfaces.map(o => ({{ name:o.name, color:o.color, n:o.positions.length/3, pos:buf(normPositions(o.positions)) }}));

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
let visible=[1,1,1,1], depthVisible=[1,1,1,1], showTerrain=false, showOutlines=true, showPlateBoundaries=Boolean(payload.tectonics.default_show), showPlateInterfaces=Boolean(payload.tectonics.default_show), showLines=false, splatScale=1.45, opacityScale=1.0, colorMode=0;
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
  if (showPlateInterfaces) {{
    for (const line of plateInterfaceBuffers) {{
      const c = line.color;
      gl.uniform4f(gl.getUniformLocation(lineProg,'u_color'), c[0], c[1], c[2], c[3]);
      attrib(lineProg,'a_pos',line.pos,3); gl.drawArrays(gl.LINE_STRIP,0,line.n);
    }}
  }}
  if (showPlateBoundaries) {{
    for (const line of plateBoundaryBuffers) {{
      const c = line.color;
      gl.uniform4f(gl.getUniformLocation(lineProg,'u_color'), c[0], c[1], c[2], c[3]);
      attrib(lineProg,'a_pos',line.pos,3); gl.drawArrays(gl.LINE_STRIP,0,line.n);
    }}
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
  gl.uniform1i(gl.getUniformLocation(splatProg,'u_colorMode'), colorMode);
  gl.uniform4f(gl.getUniformLocation(splatProg,'u_visible'), visible[0], visible[1], visible[2], visible[3]);
  gl.uniform4f(gl.getUniformLocation(splatProg,'u_depthFlagVisible'), depthVisible[0], depthVisible[1], depthVisible[2], depthVisible[3]);
  attrib(splatProg,'a_pos',splat.pos,3); attrib(splatProg,'a_color',splat.color,3); attrib(splatProg,'a_size',splat.size,1); attrib(splatProg,'a_opacity',splat.opacity,1); attrib(splatProg,'a_type',splat.type,1); attrib(splatProg,'a_depthFlag',splat.depthFlag,1);
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
document.querySelectorAll('input[data-depth-flag]').forEach(el => el.addEventListener('change', e => {{ depthVisible[Number(e.target.dataset.depthFlag)] = e.target.checked ? 1 : 0; render(); }}));
document.getElementById('colorMode').addEventListener('change', e => {{ colorMode=Number(e.target.value); render(); }});
document.getElementById('outlineToggle').addEventListener('change', e => {{ showOutlines=e.target.checked; render(); }});
document.getElementById('plateBoundaryToggle').checked = showPlateBoundaries;
document.getElementById('plateInterfaceToggle').checked = showPlateInterfaces;
document.getElementById('plateBoundaryToggle').addEventListener('change', e => {{ showPlateBoundaries=e.target.checked; render(); }});
document.getElementById('plateInterfaceToggle').addEventListener('change', e => {{ showPlateInterfaces=e.target.checked; render(); }});
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
    depth_diagnostics = _depth_diagnostics(config, limit_rows)
    terrain = _terrain_payload(config, is_sample=is_sample)
    tectonics = _tectonics_payload(config)
    metadata = {
        "html": str(paths.outputs_3d / "array_projection_splats.html"),
        "renderer": "webgl2_gaussian_point_sprite",
        "gaussian_shader": "fragment_alpha=opacity*exp(-3.25*r2)",
        "visual_resolution_policy": "point sprite size uses horizontal resolution only; depth uncertainty is metadata and does not blur the WebGL splat by default",
        "displayed_splats": len(limit_rows),
        "total_splats": len(rows),
        "line_segments": splats["line_segments"],
        "source_projection_guides_default_visible": False,
        "splat_color_default": "grayscale_relative_amplitude",
        "splat_color_modes": ["grayscale_relative_amplitude", "path_type_overlay", "depth_diagnostics"],
        "splat_color_note": "Default grayscale encodes relative amplitude. Path-type colors and depth diagnostics are optional overlays, not intensity.",
        "depth_quality_handling": {
            "display_filtering_default": "none",
            "z_display_policy": "The plotted z coordinate is the computational depth center: depth_p50_km when available, otherwise legacy z_m. The p05-p95 interval is preserved in metadata and does not enlarge point sprites by default.",
            "computation_policy": "All splat candidates are retained by default. Direct-wave catalog-depth anchors, clipped late-delay candidates, velocity-sampling ranges, and projection-refinement offsets are represented as computational uncertainty/diagnostic metadata. Downstream structure density uses structure_amplitude and resolution sigma_z_m; depth uncertainty does not blur density unless explicitly enabled.",
        },
        "depth_diagnostics": depth_diagnostics,
        "is_sample_data": is_sample,
        "vertical_exaggeration": config.visualization_3d.vertical_exaggeration,
        "synthetic_aperture_enabled": config.waveform_array.synthetic_aperture_enabled,
        "uses_phase": config.waveform_array.use_phase,
        "uses_group_delay": config.waveform_array.use_group_delay,
        "primitive_type_counts": _count_values(rows, "primitive_type"),
        "path_family_counts": _count_values(rows, "path_family"),
        "splat_role_counts": _count_existing_values(rows, "splat_role"),
        "displayed_splat_role_counts": _count_existing_values(limit_rows, "splat_role"),
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
        "tectonic_overlay_source": tectonics["source"],
        "tectonic_overlay_note": tectonics["note"],
        "tectonic_boundary_count": len(tectonics["boundaries"]),
        "tectonic_interface_line_count": len(tectonics["interfaces"]),
        "tectonic_overlay_default_visible": bool(tectonics["default_show"]),
        "tectonic_overlay_literature_based": bool(tectonics["literature_based"]),
        "tectonic_overlay_model_source": tectonics["model_source"],
        "tectonic_overlay_source_files": tectonics["source_files"],
        "tectonic_overlay_fallback_used": bool(tectonics["fallback_used"]),
        "tectonic_overlay_warning": (
            "Schematic fallback context only. It is not calibrated to Slab2, GSI, JMA, JAMSTEC, or other published plate-interface datasets and should not be used for analytical comparison."
            if tectonics["fallback_used"]
            else "Local plate model overlay loaded from configured files; validate source provenance and preprocessing before analytical comparison."
            if tectonics["literature_based"]
            else "No external plate-interface model was loaded; schematic plate fallback is disabled to avoid a misleading overlay."
        ),
        "sample_lightweight_rendering": is_sample,
        "rendering": "WebGL2 high-density point-sprite Gaussian splats with outline-only Japan context; not Plotly mesh ellipsoids",
        "not_prediction": True,
    }
    payload = {
        "metadata": metadata,
        "bounds": _bounds_from_payload(splats, terrain, tectonics),
        "splats": splats,
        "terrain": terrain,
        "tectonics": tectonics,
    }
    out = paths.outputs_3d / "array_projection_splats.html"
    out.write_text(_webgl_html(payload), encoding="utf-8")
    (paths.outputs_3d / "array_projection_splats.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
