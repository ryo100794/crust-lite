from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, fault_rectangle_vertices
from crust_lite.io.database import failure_values_for_years, stress_time_bins
from crust_lite.io.geopackage import read_features
from crust_lite.io.metadata import write_metadata
from crust_lite.io.parquet import read_sidecar, read_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths
from crust_lite.viz.html_timeseries import fallback_plotly_html, wrap_plotly_html, write_index
from crust_lite.viz.japan_outline import JAPAN_ARCHIPELAGO_OUTLINES, local_context_outlines

LOGGER = get_logger(__name__)


def plot_z_m(z_m: float, vertical_exaggeration: float) -> float:
    return -1.0 * float(z_m) * vertical_exaggeration


def _load_plotly() -> Any | None:
    try:
        import plotly.graph_objects as go  # type: ignore

        return go
    except Exception:
        return None


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _select_events(events: list[dict[str, Any]], max_events: int) -> tuple[list[dict[str, Any]], str]:
    if len(events) <= max_events:
        return events, "none"
    half = max_events // 2
    by_mag = sorted(events, key=lambda row: float(row["magnitude"]), reverse=True)[:half]
    by_time = sorted(events, key=lambda row: str(row["time_utc"]), reverse=True)[: max_events - half]
    selected: dict[str, dict[str, Any]] = {str(row["event_id"]): row for row in by_mag + by_time}
    return sorted(selected.values(), key=lambda row: str(row["time_utc"])), "magnitude_top_and_latest"


def _bin_events(
    events: list[dict[str, Any]], time_bin_days: int, max_frames: int
) -> tuple[list[str], dict[str, list[dict[str, Any]]], int, str]:
    if not events:
        return ["no_events"], {"no_events": []}, time_bin_days, "no_events"
    times = [_parse_dt(str(row["time_utc"])) for row in events]
    start, end = min(times), max(times)
    days = max(1, (end - start).days + 1)
    actual_bin = max(1, time_bin_days)
    original_frames = max(1, int(np.ceil(days / actual_bin)))
    method = "none"
    if original_frames > max_frames:
        actual_bin = int(np.ceil(days / max_frames))
        method = "time_bin_widened"
    bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, event_time in zip(events, times, strict=False):
        idx = int((event_time - start).days // actual_bin)
        bin_start = start + timedelta(days=idx * actual_bin)
        label = bin_start.date().isoformat()
        bins[label].append(row)
    labels = sorted(bins)
    return labels, bins, actual_bin, method


def _event_trace(go: Any, rows: list[dict[str, Any]], cfg: AppConfig, name: str, visible: bool) -> Any:
    viz = cfg.visualization_3d
    magnitudes = np.array([float(row["magnitude"]) for row in rows], dtype=float)
    if len(magnitudes) == 0:
        sizes: list[float] = []
    else:
        lo, hi = float(np.min(magnitudes)), float(np.max(magnitudes))
        span = max(1e-9, hi - lo)
        sizes = [
            float(
                np.clip(
                    viz.event_marker_size_min
                    + (mag - lo) / span * (viz.event_marker_size_max - viz.event_marker_size_min),
                    viz.event_marker_size_min,
                    viz.event_marker_size_max,
                )
            )
            for mag in magnitudes
        ]
    color_values = [float(row.get(viz.color_events_by, row.get("magnitude", 0.0))) for row in rows]
    hover = [
        "<br>".join(
            [
                f"event_id={row.get('event_id')}",
                f"time_utc={row.get('time_utc')}",
                f"magnitude={row.get('magnitude')}",
                f"depth_km={row.get('depth_km')}",
                f"lat={row.get('lat')}",
                f"lon={row.get('lon')}",
                f"catalog_source={row.get('catalog_source')}",
            ]
        )
        for row in rows
    ]
    return go.Scatter3d(
        x=[float(row["x_m"]) for row in rows],
        y=[float(row["y_m"]) for row in rows],
        z=[plot_z_m(float(row["z_m"]), viz.vertical_exaggeration) for row in rows],
        mode="markers",
        marker={
            "size": sizes,
            "color": color_values,
            "colorscale": "Viridis",
            "colorbar": {"title": viz.color_events_by},
            "opacity": 0.82,
        },
        text=hover,
        hoverinfo="text",
        name=name,
        visible=visible,
    )


def _limit_faults(features: list[dict[str, Any]], max_faults: int) -> tuple[list[dict[str, Any]], str]:
    if len(features) <= max_faults:
        return features, "none"
    ranked = sorted(
        features,
        key=lambda feature: float(feature.get("properties", {}).get("fault_score", feature.get("properties", {}).get("confidence", 0.0))),
        reverse=True,
    )
    return ranked[:max_faults], "fault_score_or_confidence_top"


def _fault_mesh_trace(
    go: Any,
    feature: dict[str, Any],
    cfg: AppConfig,
    color_value: float,
    name: str,
    visible: bool,
    cmin: float | None = None,
    cmax: float | None = None,
    colorbar_title: str | None = None,
    opacity: float = 0.55,
    colorscale: str = "Turbo",
    z_offset_m: float = 0.0,
    note: str | None = None,
    showscale: bool = False,
) -> Any:
    props = feature.get("properties", {})
    center_x = float(props.get("center_x_m", 0.0))
    center_y = float(props.get("center_y_m", 0.0))
    center_depth = float(props.get("center_depth_km", (float(props.get("top_depth_km", 0.0)) + float(props.get("bottom_depth_km", 10.0))) / 2.0))
    verts = fault_rectangle_vertices(
        center_x,
        center_y,
        center_depth,
        float(props.get("strike", 0.0)),
        float(props.get("dip", 70.0)),
        float(props.get("length_km", 5.0)),
        float(props.get("width_km", 5.0)),
    )
    z = [plot_z_m(v[2], cfg.visualization_3d.vertical_exaggeration) + z_offset_m for v in verts]
    hover = "<br>".join(
        [
            f"segment_id={props.get('segment_id')}",
            f"cluster_id={props.get('cluster_id', '')}",
            f"n_events={props.get('n_events', '')}",
            f"strike={props.get('strike')}",
            f"dip={props.get('dip')}",
            f"rake={props.get('rake')}",
            f"length_km={props.get('length_km')}",
            f"width_km={props.get('width_km')}",
            f"fault_score={props.get('fault_score', '')}",
            f"display_color_value={color_value}",
            f"display_z_offset_m={z_offset_m}",
            *([f"note={note}"] if note else []),
            f"confidence={props.get('confidence')}",
            f"is_inferred={props.get('is_inferred')}",
        ]
    )
    mesh_kwargs: dict[str, Any] = {
        "x": [float(v[0]) for v in verts],
        "y": [float(v[1]) for v in verts],
        "z": z,
        "i": [0, 0],
        "j": [1, 2],
        "k": [2, 3],
        "intensity": [color_value] * 4,
        "colorscale": colorscale,
        "opacity": opacity,
        "showscale": showscale,
        "text": [hover] * 4,
        "hoverinfo": "text",
        "name": name,
        "visible": visible,
    }
    if cmin is not None and cmax is not None:
        mesh_kwargs["cmin"] = cmin
        mesh_kwargs["cmax"] = cmax
    if colorbar_title:
        mesh_kwargs["colorbar"] = {"title": colorbar_title}
    return go.Mesh3d(**mesh_kwargs)


def _fault_center(feature: dict[str, Any]) -> tuple[float, float]:
    props = feature.get("properties", {})
    return float(props.get("center_x_m", 0.0)), float(props.get("center_y_m", 0.0))


def _failure_indicator_trace(
    go: Any,
    features: list[dict[str, Any]],
    cfg: AppConfig,
    values_by_seg: dict[str, float],
    baseline_by_seg: dict[str, float],
    year: int,
    value_column: str,
    cmin: float,
    cmax: float,
    max_delta: float,
) -> Any:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    sizes: list[float] = []
    values: list[float] = []
    hover: list[str] = []
    scale = max(max_delta, 1e-9)
    for feature in features:
        props = feature.get("properties", {})
        seg = str(props.get("segment_id"))
        value = float(values_by_seg.get(seg, 0.0))
        baseline = float(baseline_by_seg.get(seg, value))
        delta = max(0.0, value - baseline)
        x, y = _fault_center(feature)
        # The marker height is a display-only delta indicator. Fault geometry stays fixed;
        # the hover text keeps the scientific value and the display mapping separate.
        display_z = 25_000.0 + min(delta / scale, 1.0) * 135_000.0
        xs.append(x)
        ys.append(y)
        zs.append(display_z)
        sizes.append(5.0 + min(delta / scale, 1.0) * 18.0)
        values.append(value)
        hover.append(
            "<br>".join(
                [
                    f"segment_id={seg}",
                    f"year={year}",
                    f"{value_column}={value:.4f}",
                    f"baseline_year0={baseline:.4f}",
                    f"increase_from_year0={delta:.4f}",
                    "marker_height_is_display_only=true",
                    "日時を示すものではありません",
                ]
            )
        )
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker={
            "size": sizes,
            "color": values,
            "colorscale": "Inferno",
            "cmin": cmin,
            "cmax": cmax,
            "opacity": 0.92,
            "line": {"color": "#111827", "width": 1},
            "colorbar": {"title": f"{value_column} [-]"},
        },
        text=hover,
        hoverinfo="text",
        name=f"animated failure-index markers year {year}",
        visible=True,
    )


def _failure_delta_columns_trace(
    go: Any,
    features: list[dict[str, Any]],
    values_by_seg: dict[str, float],
    baseline_by_seg: dict[str, float],
    year: int,
    max_delta: float,
) -> Any:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    scale = max(max_delta, 1e-9)
    for feature in features:
        seg = str(feature.get("properties", {}).get("segment_id"))
        value = float(values_by_seg.get(seg, 0.0))
        baseline = float(baseline_by_seg.get(seg, value))
        delta = max(0.0, value - baseline)
        x, y = _fault_center(feature)
        display_z = 25_000.0 + min(delta / scale, 1.0) * 135_000.0
        xs.extend([x, x, None])
        ys.extend([y, y, None])
        zs.extend([0.0, display_z, None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#334155", "width": 3},
        name=f"increase columns year {year}",
        hoverinfo="skip",
        visible=True,
    )

def _gnss_traces(go: Any, cfg: AppConfig, paths: ProjectPaths) -> list[Any]:
    path = paths.data_processed / "gnss_features.parquet"
    if not path.exists() or not cfg.visualization_3d.show_gnss_vectors:
        return []
    rows = read_table(path)
    traces = []
    scale = cfg.visualization_3d.gnss_vector_scale
    for row in rows:
        x = float(row["x_m"])
        y = float(row["y_m"])
        xe = x + float(row["east_velocity_m_per_yr"]) * scale
        ye = y + float(row["north_velocity_m_per_yr"]) * scale
        text = (
            f"station_id={row.get('station_id')}<br>"
            f"east_velocity_m_per_yr={row.get('east_velocity_m_per_yr')}<br>"
            f"north_velocity_m_per_yr={row.get('north_velocity_m_per_yr')}<br>"
            f"display_scale={scale}"
        )
        traces.append(
            go.Scatter3d(
                x=[x, xe],
                y=[y, ye],
                z=[0, 0],
                mode="lines+markers",
                line={"color": "#1f77b4", "width": 4},
                marker={"size": 3},
                text=[text, text],
                hoverinfo="text",
                name=f"GNSS {row.get('station_id')}",
                visible=True,
            )
        )
    return traces


def _japan_outline_traces(go: Any, cfg: AppConfig, context: bool = False) -> list[Any]:
    outlines = JAPAN_ARCHIPELAGO_OUTLINES if context else local_context_outlines(cfg.region.bbox)
    traces = []
    for outline in outlines:
        coords = outline["coordinates"]
        if context:
            xs = [lon for lon, _lat in coords]
            ys = [lat for _lon, lat in coords]
        else:
            projector = LocalProjector(cfg.region)
            projected = [projector.lonlat_to_xy(lon, lat) for lon, lat in coords]
            xs = [x for x, _y in projected]
            ys = [y for _x, y in projected]
        zs = [35.0] * len(xs)
        hover = (
            f"Japan archipelago outline<br>island={outline['name']}<br>"
            "simplified offline cartographic context; not analytical coastline"
        )
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": "#1f5132", "width": 5 if context else 4},
                name=f"Japan archipelago - {outline['name']}",
                text=[hover] * len(xs),
                hoverinfo="text",
                visible=True,
            )
        )
        if len(coords) >= 4 and coords[0] == coords[-1]:
            mesh_x = xs[:-1]
            mesh_y = ys[:-1]
            mesh_z = [15.0] * len(mesh_x)
            n = len(mesh_x)
            traces.append(
                go.Mesh3d(
                    x=mesh_x,
                    y=mesh_y,
                    z=mesh_z,
                    i=[0] * max(0, n - 2),
                    j=list(range(1, max(1, n - 1))),
                    k=list(range(2, n)),
                    color="#b7d4c4",
                    opacity=0.20 if context else 0.16,
                    name=f"Japan land fill - {outline['name']}",
                    text=[hover] * n,
                    hoverinfo="text",
                    showlegend=False,
                )
            )
    return traces


def _map_overlay_traces(go: Any, cfg: AppConfig) -> list[Any]:
    projector = LocalProjector(cfg.region)
    min_lon, min_lat, max_lon, max_lat = cfg.region.bbox
    corners = [
        projector.lonlat_to_xy(min_lon, min_lat),
        projector.lonlat_to_xy(max_lon, min_lat),
        projector.lonlat_to_xy(max_lon, max_lat),
        projector.lonlat_to_xy(min_lon, max_lat),
    ]
    hover = (
        f"map overlay<br>region={cfg.region.name}<br>"
        f"bbox={cfg.region.bbox}<br>local_crs={cfg.region.crs_local}<br>"
        "surface z=0 m; depths are plotted below the surface"
    )
    plane = go.Mesh3d(
        x=[point[0] for point in corners],
        y=[point[1] for point in corners],
        z=[0.0, 0.0, 0.0, 0.0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="#d7eef7",
        opacity=0.18,
        name="map overlay ground plane",
        text=[hover] * 4,
        hoverinfo="text",
        showlegend=True,
    )

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for lon in np.linspace(min_lon, max_lon, 6):
        for lat in np.linspace(min_lat, max_lat, 32):
            x, y = projector.lonlat_to_xy(float(lon), float(lat))
            xs.append(x)
            ys.append(y)
            zs.append(0.0)
        xs.append(None)
        ys.append(None)
        zs.append(None)
    for lat in np.linspace(min_lat, max_lat, 6):
        for lon in np.linspace(min_lon, max_lon, 32):
            x, y = projector.lonlat_to_xy(float(lon), float(lat))
            xs.append(x)
            ys.append(y)
            zs.append(0.0)
        xs.append(None)
        ys.append(None)
        zs.append(None)
    closed = [*corners, corners[0]]
    for x, y in closed:
        xs.append(x)
        ys.append(y)
        zs.append(0.0)
    grid = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": "#6b7280", "width": 2},
        name="map overlay lon/lat grid",
        text=[hover] * len(xs),
        hoverinfo="text",
        visible=True,
    )
    return [plane, grid, *_japan_outline_traces(go, cfg, context=False)]


def _layout(title: str, sliders: list[dict[str, Any]] | None = None, updatemenus: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "title": title,
        "scene": {
            "xaxis_title": "Easting in local CRS [m]",
            "yaxis_title": "Northing in local CRS [m]",
            "zaxis_title": "Elevation-like depth display [m], depth exaggerated",
        },
        "legend": {"orientation": "h"},
        "sliders": sliders or [],
        "updatemenus": updatemenus or [],
        "margin": {"l": 0, "r": 0, "b": 0, "t": 48},
    }


def _animation_controls(labels: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "表示時点/期間: "},
            "steps": [
                {
                    "label": label,
                    "method": "animate",
                    "args": [[label], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                }
                for label in labels
            ],
        }
    ]
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "再生",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "一時停止",
                    "method": "animate",
                    "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                },
            ],
        }
    ]
    return sliders, updatemenus


def _write_figure(path: Path, fig: Any, title: str, cfg: AppConfig, metadata: dict[str, Any]) -> None:
    include = True if cfg.visualization_3d.include_plotlyjs else "cdn"
    html = fig.to_html(full_html=False, include_plotlyjs=include)
    path.write_text(wrap_plotly_html(title, html, metadata), encoding="utf-8")


def write_events_faults_3d(cfg: AppConfig, paths: ProjectPaths, metadata: dict[str, Any], mode: str | None = None, time_bin_days: int | None = None, max_events: int | None = None) -> None:
    go = _load_plotly()
    out = paths.outputs_3d / "events_faults_timeseries.html"
    if go is None:
        out.write_text(fallback_plotly_html("Events and faults 3D time series", metadata), encoding="utf-8")
        return
    events = read_table(paths.data_interim / "event_qc.parquet")
    selected_events, event_decimation = _select_events(events, max_events or cfg.visualization_3d.max_events)
    labels, bins, actual_bin, frame_decimation = _bin_events(
        selected_events, time_bin_days or cfg.visualization_3d.time_bin_days, cfg.visualization_3d.max_frames
    )
    display_mode = mode or cfg.visualization_3d.mode
    traces = []
    for idx, label in enumerate(labels):
        initial_visible = idx == 0
        traces.append(_event_trace(go, bins[label], cfg, f"events {label}", initial_visible))
    known = read_features(paths.data_processed / "fault_segment.gpkg") if (
        paths.data_processed / "fault_segment.gpkg"
    ).exists() else []
    inferred = read_features(paths.data_processed / "inferred_faults.gpkg") if (
        paths.data_processed / "inferred_faults.gpkg"
    ).exists() else []
    faults, fault_decimation = _limit_faults(known + inferred, cfg.visualization_3d.max_fault_segments)
    for feature in faults:
        props = feature.get("properties", {})
        is_inferred = str(props.get("is_inferred", "")).lower() == "true" or props.get("is_inferred") is True
        if is_inferred and not cfg.visualization_3d.show_inferred_faults:
            continue
        if not is_inferred and not cfg.visualization_3d.show_known_faults:
            continue
        value = float(props.get(cfg.visualization_3d.color_faults_by, props.get("confidence", 0.5)))
        traces.append(
            _fault_mesh_trace(
                go,
                feature,
                cfg,
                value,
                f"{'inferred' if is_inferred else 'known'} {props.get('segment_id')}",
                True,
            )
        )
    traces.extend(_gnss_traces(go, cfg, paths))
    traces.extend(_map_overlay_traces(go, cfg))
    frames = []
    event_trace_count = len(labels)
    for idx, label in enumerate(labels):
        trace_visibility: list[bool] = []
        for trace_idx in range(len(traces)):
            if trace_idx < event_trace_count:
                trace_visibility.append(trace_idx <= idx if display_mode == "cumulative" else trace_idx == idx)
            else:
                trace_visibility.append(True)
        frame_data = [
            {"type": getattr(trace, "type", "scatter3d"), "visible": value}
            for trace, value in zip(traces, trace_visibility, strict=False)
        ]
        frames.append(go.Frame(name=label, data=frame_data, traces=list(range(len(traces)))))
    sliders, buttons = _animation_controls(labels)
    fig = go.Figure(data=traces, frames=frames, layout=_layout("Events, known faults, inferred faults", sliders, buttons))
    _write_figure(out, fig, "Events and faults 3D time series", cfg, metadata)
    metadata.update(
        {
            "original_event_count": len(events),
            "displayed_event_count": len(selected_events),
            "original_fault_count": len(known) + len(inferred),
            "displayed_fault_count": len(faults),
            "original_frame_count": max(1, len(labels)),
            "displayed_frame_count": len(labels),
            "actual_time_bin_days": actual_bin,
            "decimation_method": ", ".join(sorted({event_decimation, frame_decimation, fault_decimation})),
        }
    )


def write_stress_3d(cfg: AppConfig, paths: ProjectPaths, metadata: dict[str, Any]) -> None:
    go = _load_plotly()
    out = paths.outputs_3d / "stress_timeseries_3d.html"
    if go is None:
        out.write_text(fallback_plotly_html("Stress 3D time series", metadata), encoding="utf-8")
        return
    features = []
    for path in (paths.data_processed / "fault_segment.gpkg", paths.data_processed / "inferred_faults.gpkg"):
        if path.exists():
            features.extend(read_features(path))
    features, fault_decimation = _limit_faults(features, cfg.visualization_3d.max_fault_segments)
    labels, stress_by_label_seg, method, actual_bin = stress_time_bins(
        paths, cfg.visualization_3d.time_bin_days, cfg.visualization_3d.max_frames
    )
    if not labels:
        labels = ["no_stress"]
        stress_by_label_seg = {}
        method = "no_stress"
        actual_bin = cfg.visualization_3d.time_bin_days
    traces = []
    for feature in features:
        props = feature.get("properties", {})
        seg = str(props.get("segment_id"))
        traces.append(_fault_mesh_trace(go, feature, cfg, stress_by_label_seg.get((labels[0], seg), 0.0), f"stress {seg}", True))
    traces.extend(_map_overlay_traces(go, cfg))
    frames = []
    fault_trace_count = len(features)
    for label in labels:
        frame_data = []
        for feature in features:
            seg = str(feature.get("properties", {}).get("segment_id"))
            value = stress_by_label_seg.get((label, seg), 0.0)
            frame_data.append(_fault_mesh_trace(go, feature, cfg, value, f"stress {seg}", True))
        # Keep map overlay traces static; animation updates only fault traces.
        frames.append(go.Frame(name=label, data=frame_data, traces=list(range(fault_trace_count))))
    sliders, buttons = _animation_controls(labels)
    title = "Stress state (fallback relative score)" if method == "fallback_approximation" else "Coulomb stress change [Pa]"
    fig = go.Figure(data=traces, frames=frames, layout=_layout(title, sliders, buttons))
    local_meta = {
        **metadata,
        "stress_method": method,
        "stress_color_value": "cfs_score_approx" if method == "fallback_approximation" else "cfs_pa",
        "stress_time_bin_days": actual_bin,
        "stress_displayed_frames": len(labels),
        "stress_fault_decimation": fault_decimation,
        "database_engine": "duckdb",
    }
    metadata.update(
        {
            "stress_method": method,
            "stress_displayed_frames": len(labels),
            "stress_time_bin_days": actual_bin,
        }
    )
    _write_figure(out, fig, "Stress 3D time series", cfg, local_meta)


def write_failure_3d(cfg: AppConfig, paths: ProjectPaths, metadata: dict[str, Any]) -> None:
    go = _load_plotly()
    out = paths.outputs_3d / "failure_scenarios_3d.html"
    if go is None:
        out.write_text(fallback_plotly_html("Failure scenario 3D", metadata), encoding="utf-8")
        return
    scenarios = read_table(paths.outputs_tables / "failure_scenarios.parquet")
    features = []
    for path in (paths.data_processed / "fault_segment.gpkg", paths.data_processed / "inferred_faults.gpkg"):
        if path.exists():
            features.extend(read_features(path))
    features, fault_decimation = _limit_faults(features, cfg.visualization_3d.max_fault_segments)
    desired = [0, 10, 30, 50, 100]
    available = sorted({int(row["year"]) for row in scenarios})
    years = [year for year in desired if year in available] or available[: min(10, len(available))]
    value_by_year_seg = failure_values_for_years(paths, years, cfg.visualization_3d.color_failure_by)
    if value_by_year_seg is None:
        value_by_year_seg = {
            (int(row["year"]), str(row["segment_id"])): float(row.get(cfg.visualization_3d.color_failure_by, row.get("failure_index_p50", 0.0)))
            for row in scenarios
            if int(row["year"]) in years
        }
    selected_segments = [str(feature.get("properties", {}).get("segment_id")) for feature in features]
    displayed_values = [
        float(value_by_year_seg.get((year, seg), 0.0))
        for year in years
        for seg in selected_segments
    ]
    cmin = min(displayed_values + [0.0])
    cmax = max(displayed_values + [1.0])
    if abs(cmax - cmin) < 1e-9:
        cmax = cmin + 1.0
    baseline_year = years[0]
    baseline_by_seg = {seg: float(value_by_year_seg.get((baseline_year, seg), 0.0)) for seg in selected_segments}
    max_delta = max(
        [
            max(0.0, float(value_by_year_seg.get((year, seg), 0.0)) - baseline_by_seg.get(seg, 0.0))
            for year in years
            for seg in selected_segments
        ]
        + [1e-9]
    )

    def values_for(year: int) -> dict[str, float]:
        return {seg: float(value_by_year_seg.get((year, seg), 0.0)) for seg in selected_segments}

    def delta_for(seg: str, values_by_seg: dict[str, float]) -> float:
        return max(0.0, float(values_by_seg.get(seg, 0.0)) - baseline_by_seg.get(seg, 0.0))

    def overlay_opacity(delta: float) -> float:
        if max_delta <= 1e-9:
            return 0.03
        return 0.04 + 0.58 * min(delta / max_delta, 1.0)

    def frame_fault_surfaces(year: int) -> list[Any]:
        values_by_seg = values_for(year)
        frame_data: list[Any] = []
        for idx, feature in enumerate(features):
            seg = str(feature.get("properties", {}).get("segment_id"))
            frame_data.append(
                _fault_mesh_trace(
                    go,
                    feature,
                    cfg,
                    values_by_seg.get(seg, 0.0),
                    f"failure_index surface {seg}",
                    True,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar_title=f"{cfg.visualization_3d.color_failure_by} [-]",
                    opacity=0.50,
                    colorscale="Turbo",
                    note="absolute failure_index on fixed fault surface",
                    showscale=idx == 0,
                )
            )
        for feature in features:
            seg = str(feature.get("properties", {}).get("segment_id"))
            delta = delta_for(seg, values_by_seg)
            frame_data.append(
                _fault_mesh_trace(
                    go,
                    feature,
                    cfg,
                    delta,
                    f"increase overlay {seg}",
                    True,
                    cmin=0.0,
                    cmax=max_delta,
                    colorbar_title="increase from year 0 [-]",
                    opacity=overlay_opacity(delta),
                    colorscale="Inferno",
                    z_offset_m=1800.0,
                    note="transparent overlay encodes increase_from_year0; geometry is fixed",
                    showscale=False,
                )
            )
        return frame_data

    traces = frame_fault_surfaces(years[0])
    dynamic_trace_count = len(traces)
    traces.extend(_map_overlay_traces(go, cfg))

    frames = [
        go.Frame(name=str(year), data=frame_fault_surfaces(year), traces=list(range(dynamic_trace_count)))
        for year in years
    ]
    labels = [str(year) for year in years]
    sliders, buttons = _animation_controls(labels)
    layout = _layout(
        "Relative failure index on fault surfaces",
        sliders,
        buttons,
    )
    layout["annotations"] = [
        {
            "text": "断層面の位置は固定。主面の色はfailure_index、少し浮いた半透明面は0年からの増分。日時を示すものではありません。",
            "xref": "paper",
            "yref": "paper",
            "x": 0.01,
            "y": 0.99,
            "showarrow": False,
            "align": "left",
            "bgcolor": "rgba(255,255,255,0.84)",
            "bordercolor": "#cbd5e1",
        }
    ]
    fig = go.Figure(data=traces, frames=frames, layout=layout)
    local_meta = {
        **metadata,
        "failure_color_value": cfg.visualization_3d.color_failure_by,
        "failure_interpretation": "Dimensionless relative index; above 1 means model-threshold exceedance only.",
        "failure_fault_decimation": fault_decimation,
        "failure_display_years": years,
        "failure_color_cmin": cmin,
        "failure_color_cmax": cmax,
        "failure_baseline_year": baseline_year,
        "failure_max_increase_from_baseline": max_delta,
        "animated_failure_index_indicator": "fault-surface color updates plus transparent increase overlay; no vertical bar traces",
        "failure_overlay_z_offset_m": 1800.0,
        "database_engine": "duckdb",
    }
    metadata.update(
        {
            "failure_display_years": years,
            "failure_color_cmin": cmin,
            "failure_color_cmax": cmax,
            "failure_baseline_year": baseline_year,
            "failure_max_increase_from_baseline": max_delta,
            "animated_failure_index_indicator": "fault-surface color updates plus transparent increase overlay",
            "failure_overlay_z_offset_m": 1800.0,
        }
    )
    _write_figure(out, fig, "Failure scenario 3D", cfg, local_meta)

def write_japan_context_map_3d(cfg: AppConfig, paths: ProjectPaths, metadata: dict[str, Any]) -> None:
    go = _load_plotly()
    out = paths.outputs_3d / "japan_archipelago_context.html"
    local_meta = {
        **metadata,
        "map_overlay": "simplified_offline_japan_archipelago_outline",
        "japan_outline_note": "Coarse cartographic context only; not analytical coastline data.",
    }
    if go is None:
        out.write_text(fallback_plotly_html("Japan archipelago context map", local_meta), encoding="utf-8")
        return
    traces = _japan_outline_traces(go, cfg, context=True)
    min_lon, min_lat, max_lon, max_lat = cfg.region.bbox
    region_x = [min_lon, max_lon, max_lon, min_lon, min_lon]
    region_y = [min_lat, min_lat, max_lat, max_lat, min_lat]
    traces.append(
        go.Scatter3d(
            x=region_x,
            y=region_y,
            z=[80.0] * len(region_x),
            mode="lines",
            line={"color": "#c2410c", "width": 7},
            name=f"target bbox - {cfg.region.name}",
            text=[f"target bbox<br>region={cfg.region.name}<br>bbox={cfg.region.bbox}"] * len(region_x),
            hoverinfo="text",
        )
    )
    layout = _layout("Japan archipelago context map")
    layout["scene"] = {
        "xaxis_title": "Longitude [deg]",
        "yaxis_title": "Latitude [deg]",
        "zaxis_title": "Context display height [m]",
        "aspectmode": "data",
    }
    fig = go.Figure(data=traces, layout=layout)
    _write_figure(out, fig, "Japan archipelago context map", cfg, local_meta)


def generate_3d_visualizations(
    config: AppConfig,
    paths: ProjectPaths,
    mode: str | None = None,
    time_bin_days: int | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    if not config.visualization_3d.enabled:
        LOGGER.info("Skipping 3D visualization because visualization_3d.enabled=false")
        return {"enabled": False}
    paths.outputs_3d.mkdir(parents=True, exist_ok=True)
    event_meta = read_sidecar(paths.data_interim / "event_qc.parquet")
    metadata: dict[str, Any] = {
        "enabled": True,
        "region": config.region.name,
        "start_date": config.region.start_date.isoformat(),
        "end_date": config.region.end_date.isoformat(),
        "is_sample_data": bool(event_meta.get("is_sample_data", False)),
        "vertical_exaggeration": config.visualization_3d.vertical_exaggeration,
        "mode": mode or config.visualization_3d.mode,
        "time_bin_days": time_bin_days or config.visualization_3d.time_bin_days,
        "map_overlay": "local_crs_bbox_graticule_surface_with_japan_archipelago_outline",
    }
    write_events_faults_3d(config, paths, metadata, mode=mode, time_bin_days=time_bin_days, max_events=max_events)
    write_stress_3d(config, paths, metadata)
    write_failure_3d(config, paths, metadata)
    write_japan_context_map_3d(config, paths, metadata)
    metadata["japan_context_file"] = "japan_archipelago_context.html"
    metadata["japan_archipelago_layer"] = "simplified_offline_outline"
    write_index(config, paths, metadata)
    write_metadata(paths.outputs_3d / "metadata.json", metadata)
    LOGGER.info("Wrote 3D HTML outputs to %s", paths.outputs_3d)
    return metadata
