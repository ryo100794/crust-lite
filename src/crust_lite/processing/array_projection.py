"""Phase-aware waveform array projection and Gaussian splat export.

This module converts event-window spectra into a compact geometric
representation for later GPU-side volumetric experiments.  Phase and group
delay are kept as first-class inputs so spectra do not discard timing
information that is needed for array-style triangulation.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, clamp01
from crust_lite.io.database import materialize_file, materialize_known_tables, write_rows_csv_stream
from crust_lite.io.parquet import read_sidecar, read_table, write_sidecar, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths
from crust_lite.processing.transfer_function import estimate_transfer_functions

LOGGER = get_logger(__name__)

# Keep a narrow schema: these rows are the CPU-prepared handoff to later
# GPU experiments, so every field should be inexpensive to scan from Parquet.
PROJECTION_COLUMNS = [
    "event_id",
    "time_utc",
    "time_bin_index",
    "time_bin_start_utc",
    "magnitude",
    "depth_km",
    "x_m",
    "y_m",
    "z_m",
    "projection_x_m",
    "projection_y_m",
    "projection_z_m",
    "frequency_hz",
    "beam_energy",
    "phase_coherence",
    "delay_fit",
    "mean_amplitude_log",
    "n_stations",
    "velocity_km_s",
    "phase_resultant_rad",
    "projection_rank",
    "projection_method",
    "is_sample_data",
]

SPLAT_COLUMNS = [
    "primitive_id",
    "event_id",
    "time_utc",
    "time_bin_index",
    "x_m",
    "y_m",
    "z_m",
    "sigma_x_m",
    "sigma_y_m",
    "sigma_z_m",
    "amplitude",
    "opacity",
    "phase_rad",
    "color_r",
    "color_g",
    "color_b",
    "source_projection_rank",
    "source_frequency_hz",
    "source_projection_method",
    "interpretation",
    "is_sample_data",
]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _time_bin(value: str, start: datetime, days: int) -> tuple[int, str]:
    current = _parse_time(value)
    delta_days = max(0, (current - start).days)
    idx = delta_days // max(1, days)
    bin_start = start.timestamp() + idx * max(1, days) * 86400
    return int(idx), _format_time(datetime.fromtimestamp(bin_start, timezone.utc))


def _wrap_phase(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _phase_result(values: list[tuple[float, float]]) -> tuple[float, float]:
    """Return circular coherence and mean phase for weighted phase samples."""
    if not values:
        return 0.0, 0.0
    sx = sum(weight * math.cos(phase) for phase, weight in values)
    sy = sum(weight * math.sin(phase) for phase, weight in values)
    weight_sum = max(sum(weight for _phase, weight in values), 1e-30)
    return min(1.0, math.hypot(sx, sy) / weight_sum), math.atan2(sy, sx)


def _candidate_offsets(radius_m: float, grid_m: float) -> list[tuple[float, float]]:
    """Build a small horizontal search stencil around the catalog hypocenter."""
    steps = int(math.ceil(radius_m / grid_m))
    offsets: list[tuple[float, float]] = []
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            dx = ix * grid_m
            dy = iy * grid_m
            if math.hypot(dx, dy) <= radius_m + 1e-6:
                offsets.append((dx, dy))
    offsets.sort(key=lambda item: (math.hypot(item[0], item[1]), item[0], item[1]))
    return offsets or [(0.0, 0.0)]


def _read_events(paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    source = paths.data_interim / "event_qc.parquet"
    if not source.exists():
        source = paths.data_processed / "event.parquet"
    rows = read_table(source)
    return {str(row["event_id"]): row for row in rows if row.get("event_id")}


def _event_time_index(events: dict[str, dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for event_id, event in events.items():
        raw = str(event.get("time_utc", ""))
        if not raw:
            continue
        try:
            dt = _parse_time(raw)
            index[dt.strftime("%Y-%m-%dT%H:%M")] = event_id
            index[dt.strftime("%Y-%m-%dT%H:%M:%S")] = event_id
        except Exception:
            index[raw[:16]] = event_id
    return index


def _resolve_spectrum_event_id(row: dict[str, Any], events: dict[str, dict[str, Any]], time_index: dict[str, str]) -> str | None:
    """Match spectra to catalog events, tolerating sources that omit event_id."""
    event_id = str(row.get("event_id", ""))
    if event_id in events:
        return event_id
    raw_time = str(row.get("time_utc", ""))
    if raw_time:
        try:
            dt = _parse_time(raw_time)
            return time_index.get(dt.strftime("%Y-%m-%dT%H:%M:%S")) or time_index.get(dt.strftime("%Y-%m-%dT%H:%M"))
        except Exception:
            return time_index.get(raw_time[:19]) or time_index.get(raw_time[:16])
    return None


def _read_spectra(config: AppConfig, paths: ProjectPaths, sample: bool) -> tuple[list[dict[str, Any]], bool, str]:
    spectra_path = paths.data_processed / "waveform_spectrum.parquet"
    if not spectra_path.exists() and (sample or config.data_sources.waveform_spectra_csv):
        estimate_transfer_functions(config, paths, sample=sample)
    if not spectra_path.exists():
        return [], False, "waveform_spectrum_not_available"
    meta = read_sidecar(spectra_path)
    rows = read_table(spectra_path)
    return rows, bool(meta.get("is_sample_data", sample)), str(spectra_path)


def _station_rows(
    rows: list[dict[str, Any]],
    projector: LocalProjector,
    max_stations: int,
) -> list[dict[str, float]]:
    # Collapse many frequency samples per station into one robust station row;
    # the projection grid should be driven by station geometry, not CSV volume.
    by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_station[str(row["station_id"])].append(row)
    stations: list[dict[str, float]] = []
    for station_id, items in by_station.items():
        amplitudes = [max(float(item.get("amplitude", 0.0) or 0.0), 1e-30) for item in items]
        phases = [float(item.get("phase_rad", 0.0) or 0.0) for item in items]
        delays = [float(item.get("group_delay_s", 0.0) or 0.0) for item in items]
        item0 = items[0]
        x_raw = item0.get("x_m", "")
        y_raw = item0.get("y_m", "")
        if x_raw not in (None, "") and y_raw not in (None, ""):
            x_m = float(x_raw)
            y_m = float(y_raw)
        else:
            lat = float(item0.get("lat", 0.0) or 0.0)
            lon = float(item0.get("lon", 0.0) or 0.0)
            x_m, y_m = projector.lonlat_to_xy(lon, lat) if lat and lon else (0.0, 0.0)
        phase_weights = list(zip(phases, amplitudes, strict=False))
        sx = sum(weight * math.cos(phase) for phase, weight in phase_weights)
        sy = sum(weight * math.sin(phase) for phase, weight in phase_weights)
        stations.append(
            {
                "station_id_hash": float(abs(hash(station_id)) % 1000000),
                "x_m": x_m,
                "y_m": y_m,
                "amplitude": median(amplitudes),
                "log_amplitude": math.log(max(median(amplitudes), 1e-30)),
                "phase_rad": math.atan2(sy, sx),
                "group_delay_s": median(delays),
            }
        )
    stations.sort(key=lambda row: float(row["amplitude"]), reverse=True)
    return stations[:max_stations]


def _score_candidate(
    stations: list[dict[str, float]],
    gx: float,
    gy: float,
    frequency_hz: float,
    velocity_m_s: float,
    delay_sigma_s: float,
    use_phase: bool,
    use_group_delay: bool,
) -> tuple[float, float, float, float]:
    # Delay-and-sum scoring is intentionally approximate here. It ranks likely
    # horizontal source projections without asserting a solved velocity model.
    distances = [math.hypot(float(st["x_m"]) - gx, float(st["y_m"]) - gy) for st in stations]
    reference_distance = median(distances)
    predicted = [(dist - reference_distance) / velocity_m_s for dist in distances]
    observed_delay = [float(st["group_delay_s"]) for st in stations]
    delay_reference = median(observed_delay)
    observed_delay_rel = [value - delay_reference for value in observed_delay]
    weights = [max(float(st["amplitude"]), 1e-30) for st in stations]
    max_weight = max(weights or [1.0])
    weights = [0.25 + 0.75 * (weight / max_weight) for weight in weights]

    phase_coherence = 0.5
    phase_result = 0.0
    if use_phase:
        plus: list[tuple[float, float]] = []
        minus: list[tuple[float, float]] = []
        for st, pred, weight in zip(stations, predicted, weights, strict=False):
            shift = 2.0 * math.pi * frequency_hz * pred
            phase = float(st["phase_rad"])
            plus.append((_wrap_phase(phase - shift), weight))
            minus.append((_wrap_phase(phase + shift), weight))
        plus_score, plus_phase = _phase_result(plus)
        minus_score, minus_phase = _phase_result(minus)
        if plus_score >= minus_score:
            phase_coherence = plus_score
            phase_result = plus_phase
        else:
            phase_coherence = minus_score
            phase_result = minus_phase

    delay_fit = 0.5
    if use_group_delay:
        fits = []
        for obs, pred in zip(observed_delay_rel, predicted, strict=False):
            fits.append(math.exp(-0.5 * ((obs - pred) / delay_sigma_s) ** 2))
        delay_fit = sum(fits) / max(len(fits), 1)

    if use_phase and use_group_delay:
        energy = 0.65 * phase_coherence + 0.35 * delay_fit
    elif use_phase:
        energy = phase_coherence
    elif use_group_delay:
        energy = delay_fit
    else:
        energy = 0.5
    mean_amp = sum(float(st["log_amplitude"]) for st in stations) / max(len(stations), 1)
    return clamp01(energy), clamp01(phase_coherence), clamp01(delay_fit), phase_result if math.isfinite(phase_result) else 0.0


def build_waveform_array_projection(
    config: AppConfig,
    paths: ProjectPaths,
    sample: bool = False,
) -> dict[str, Any]:
    # This is the public CPU preprocessing step used before any GPU splatting
    # work: spectra -> projection candidates -> compact splat primitives.
    paths.ensure()
    projection_path = paths.data_processed / "waveform_array_projection.parquet"
    splat_path = paths.data_processed / "gaussian_splat_primitive.parquet"
    if not config.waveform_array.enabled:
        return _write_empty(paths, "waveform_array.enabled=false", sample)

    spectra_rows, is_sample, source_note = _read_spectra(config, paths, sample)
    if not spectra_rows:
        return _write_empty(paths, source_note, is_sample)

    events = _read_events(paths)
    if not events:
        return _write_empty(paths, "event_table_not_available", is_sample)

    start = datetime.combine(config.region.start_date, datetime.min.time(), tzinfo=timezone.utc)
    projector = LocalProjector(config.region)
    by_event_freq: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    time_index = _event_time_index(events)
    for row in spectra_rows:
        event_id = _resolve_spectrum_event_id(row, events, time_index)
        if event_id is not None:
            by_event_freq[(event_id, float(row.get("frequency_hz", 0.0) or 0.0))].append(row)

    event_order = sorted(
        {event_id for event_id, _freq in by_event_freq},
        key=lambda event_id: (
            -float(events[event_id].get("magnitude", 0.0) or 0.0),
            str(events[event_id].get("time_utc", "")),
        ),
    )
    if config.waveform_array.max_events:
        event_order = event_order[: config.waveform_array.max_events]
    allowed_events = set(event_order)

    offsets = _candidate_offsets(
        config.waveform_array.projection_radius_km * 1000.0,
        config.waveform_array.projection_grid_km * 1000.0,
    )
    velocity_m_s = config.waveform_array.velocity_km_s * 1000.0
    projection_rows: list[dict[str, Any]] = []

    grouped: dict[str, list[tuple[float, list[dict[str, Any]]]]] = defaultdict(list)
    for (event_id, freq), rows in by_event_freq.items():
        if event_id in allowed_events and freq > 0.0:
            grouped[event_id].append((freq, rows))

    for event_id in event_order:
        event = events[event_id]
        event_candidates: list[dict[str, Any]] = []
        event_x = float(event.get("x_m", 0.0) or 0.0)
        event_y = float(event.get("y_m", 0.0) or 0.0)
        depth_km = float(event.get("depth_km", 0.0) or 0.0)
        z_m = float(event.get("z_m", depth_km * 1000.0) or 0.0)
        time_utc = str(event.get("time_utc", ""))
        time_bin_index, time_bin_start = _time_bin(time_utc, start, config.waveform_array.time_bin_days)
        for freq, rows in sorted(grouped[event_id], key=lambda item: item[0]):
            stations = _station_rows(rows, projector, config.waveform_array.max_stations_per_event)
            if len(stations) < config.waveform_array.min_stations:
                continue
            mean_amp = sum(float(st["log_amplitude"]) for st in stations) / len(stations)
            for dx, dy in offsets:
                gx = event_x + dx
                gy = event_y + dy
                energy, phase_coherence, delay_fit, phase_result = _score_candidate(
                    stations,
                    gx,
                    gy,
                    freq,
                    velocity_m_s,
                    config.waveform_array.delay_sigma_s,
                    config.waveform_array.use_phase,
                    config.waveform_array.use_group_delay,
                )
                event_candidates.append(
                    {
                        "event_id": event_id,
                        "time_utc": time_utc,
                        "time_bin_index": time_bin_index,
                        "time_bin_start_utc": time_bin_start,
                        "magnitude": float(event.get("magnitude", 0.0) or 0.0),
                        "depth_km": depth_km,
                        "x_m": event_x,
                        "y_m": event_y,
                        "z_m": z_m,
                        "projection_x_m": gx,
                        "projection_y_m": gy,
                        "projection_z_m": z_m,
                        "frequency_hz": freq,
                        "beam_energy": energy,
                        "phase_coherence": phase_coherence,
                        "delay_fit": delay_fit,
                        "mean_amplitude_log": mean_amp,
                        "n_stations": len(stations),
                        "velocity_km_s": config.waveform_array.velocity_km_s,
                        "phase_resultant_rad": phase_result,
                        "projection_rank": 0,
                        "projection_method": "delay_and_sum_phase_group_delay_projection",
                        "is_sample_data": is_sample,
                    }
                )
        event_candidates.sort(key=lambda row: float(row["beam_energy"]), reverse=True)
        for rank, row in enumerate(event_candidates[: config.waveform_array.top_projections_per_event], start=1):
            row["projection_rank"] = rank
            projection_rows.append(row)
            if len(projection_rows) >= config.waveform_array.max_projection_rows:
                break
        if len(projection_rows) >= config.waveform_array.max_projection_rows:
            break

    write_table(
        projection_rows,
        projection_path,
        {
            "is_sample_data": is_sample,
            "source_path": source_note,
            "method": "delay-and-sum array projection using phase and group delay",
            "interpretation": "relative coherent array energy image, not an earthquake prediction or unique subsurface inversion",
            "event_count_with_spectra": len(allowed_events),
            "projection_rows": len(projection_rows),
        },
    )
    materialize_file(paths, "waveform_array_projection", projection_path)

    splat_rows = _build_splats(config, projection_rows, is_sample)
    write_table(
        splat_rows,
        splat_path,
        {
            "is_sample_data": is_sample,
            "source_path": str(projection_path),
            "method": "gaussian_splat_primitives_from_array_projection",
            "interpretation": "rendering primitives for relative waveform-derived structure indicators; not a deterministic subsurface model",
            "splat_rows": len(splat_rows),
        },
    )
    materialize_file(paths, "gaussian_splat_primitive", splat_path)
    if config.waveform_array.output_ply:
        _write_splat_ply(paths.outputs_3d / "gaussian_splat_primitives.ply", splat_rows)
    if config.waveform_array.output_html_preview:
        _write_splat_preview(config, paths, splat_rows, is_sample)
    materialize_known_tables(paths)
    meta = {
        "is_sample_data": is_sample,
        "source_path": source_note,
        "projection_rows": len(projection_rows),
        "splat_rows": len(splat_rows),
        "projection_method": "delay_and_sum_phase_group_delay_projection",
        "uses_phase": config.waveform_array.use_phase,
        "uses_group_delay": config.waveform_array.use_group_delay,
        "not_prediction": True,
    }
    write_sidecar(projection_path, {**read_sidecar(projection_path), **meta})
    LOGGER.info("Built %d waveform array projections and %d splat primitives", len(projection_rows), len(splat_rows))
    return meta


def _write_empty(paths: ProjectPaths, reason: str, is_sample: bool) -> dict[str, Any]:
    metadata = {
        "is_sample_data": is_sample,
        "method": "not_generated",
        "reason": reason,
        "projection_rows": 0,
        "splat_rows": 0,
        "not_prediction": True,
    }
    write_rows_csv_stream([], paths.data_processed / "waveform_array_projection.parquet", PROJECTION_COLUMNS, metadata)
    write_rows_csv_stream([], paths.data_processed / "gaussian_splat_primitive.parquet", SPLAT_COLUMNS, metadata)
    return {"skipped": True, **metadata}


def _build_splats(config: AppConfig, projection_rows: list[dict[str, Any]], is_sample: bool) -> list[dict[str, Any]]:
    """Convert projection rows into Gaussian-splat-like primitives.

    These are not rendered as the final scientific result; they are compact
    seeds for GPU experiments where multiple 2D projections can be fused.
    """
    rows = sorted(projection_rows, key=lambda row: float(row.get("beam_energy", 0.0) or 0.0), reverse=True)
    rows = rows[: config.waveform_array.max_splats]
    splats: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        amplitude = clamp01(float(row.get("beam_energy", 0.0) or 0.0))
        color_r, color_g, color_b = _energy_rgb(amplitude)
        splats.append(
            {
                "primitive_id": f"gs_{idx:08d}",
                "event_id": row.get("event_id", ""),
                "time_utc": row.get("time_utc", ""),
                "time_bin_index": row.get("time_bin_index", 0),
                "x_m": row.get("projection_x_m", row.get("x_m", 0.0)),
                "y_m": row.get("projection_y_m", row.get("y_m", 0.0)),
                "z_m": row.get("projection_z_m", row.get("z_m", 0.0)),
                "sigma_x_m": config.waveform_array.splat_sigma_horizontal_m,
                "sigma_y_m": config.waveform_array.splat_sigma_horizontal_m,
                "sigma_z_m": config.waveform_array.splat_sigma_vertical_m,
                "amplitude": amplitude,
                "opacity": clamp01(0.10 + 0.90 * amplitude),
                "phase_rad": row.get("phase_resultant_rad", 0.0),
                "color_r": color_r,
                "color_g": color_g,
                "color_b": color_b,
                "source_projection_rank": row.get("projection_rank", 0),
                "source_frequency_hz": row.get("frequency_hz", 0.0),
                "source_projection_method": row.get("projection_method", ""),
                "interpretation": "relative coherent-energy Gaussian primitive; not a claim of rupture timing or unique structure",
                "is_sample_data": is_sample,
            }
        )
    return splats


def _energy_rgb(value: float) -> tuple[int, int, int]:
    v = clamp01(value)
    r = int(40 + 215 * v)
    g = int(90 + 110 * (1.0 - abs(v - 0.5) * 2.0))
    b = int(230 - 190 * v)
    return r, max(0, min(255, g)), max(0, min(255, b))


def _write_splat_ply(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an ASCII PLY preview that common 3D tools can inspect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        "comment crust-lite Gaussian splat primitive export",
        "comment coordinates use local CRS meters; z_m is positive downward depth",
        f"element vertex {len(rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float opacity",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float amplitude",
        "property float phase_rad",
        "end_header",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for row in rows:
            fh.write(
                "{x:.6f} {y:.6f} {z:.6f} {sx:.6f} {sy:.6f} {sz:.6f} {op:.6f} {r:d} {g:d} {b:d} {amp:.6f} {phase:.6f}\n".format(
                    x=float(row.get("x_m", 0.0) or 0.0),
                    y=float(row.get("y_m", 0.0) or 0.0),
                    z=float(row.get("z_m", 0.0) or 0.0),
                    sx=float(row.get("sigma_x_m", 0.0) or 0.0),
                    sy=float(row.get("sigma_y_m", 0.0) or 0.0),
                    sz=float(row.get("sigma_z_m", 0.0) or 0.0),
                    op=float(row.get("opacity", 0.0) or 0.0),
                    r=int(row.get("color_r", 0) or 0),
                    g=int(row.get("color_g", 0) or 0),
                    b=int(row.get("color_b", 0) or 0),
                    amp=float(row.get("amplitude", 0.0) or 0.0),
                    phase=float(row.get("phase_rad", 0.0) or 0.0),
                )
            )
    write_sidecar(path, {"row_count": len(rows), "format": "ascii_ply_gaussian_splat_primitives"})


def _write_splat_preview(config: AppConfig, paths: ProjectPaths, rows: list[dict[str, Any]], is_sample: bool) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return
    paths.outputs_3d.mkdir(parents=True, exist_ok=True)
    limit_rows = rows[: min(len(rows), 20000)]
    z_values = [
        -1.0 * float(row.get("z_m", 0.0) or 0.0) * config.visualization_3d.vertical_exaggeration
        for row in limit_rows
    ]
    colors = [float(row.get("amplitude", 0.0) or 0.0) for row in limit_rows]
    customdata = [
        [
            row.get("primitive_id", ""),
            row.get("event_id", ""),
            row.get("time_utc", ""),
            row.get("source_frequency_hz", ""),
            row.get("amplitude", ""),
            row.get("is_sample_data", is_sample),
        ]
        for row in limit_rows
    ]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[float(row.get("x_m", 0.0) or 0.0) for row in limit_rows],
                y=[float(row.get("y_m", 0.0) or 0.0) for row in limit_rows],
                z=z_values,
                mode="markers",
                marker={
                    "size": [max(2.0, 10.0 * float(row.get("amplitude", 0.0) or 0.0)) for row in limit_rows],
                    "color": colors,
                    "colorscale": "Turbo",
                    "colorbar": {"title": "coherent energy [-]"},
                    "opacity": 0.65,
                },
                customdata=customdata,
                hovertemplate=(
                    "primitive=%{customdata[0]}<br>event=%{customdata[1]}<br>"
                    "time=%{customdata[2]}<br>frequency=%{customdata[3]} Hz<br>"
                    "energy=%{customdata[4]:.3f}<br>is_sample_data=%{customdata[5]}<extra></extra>"
                ),
                name="Gaussian splat primitives",
            )
        ]
    )
    fig.update_layout(
        title=(
            "Waveform array projection Gaussian primitives "
            f"(vertical exaggeration {config.visualization_3d.vertical_exaggeration:g}x; is_sample_data={str(is_sample).lower()})"
        ),
        scene={
            "xaxis_title": "Easting in local CRS [m]",
            "yaxis_title": "Northing in local CRS [m]",
            "zaxis_title": "Elevation-like depth display [m], depth exaggerated",
            "aspectmode": "data",
        },
        annotations=[
            {
                "text": "Research state display from phase/group-delay array projection; not an earthquake forecast.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.07,
                "showarrow": False,
                "align": "left",
            }
        ],
    )
    include = True if config.visualization_3d.include_plotlyjs else "cdn"
    out = paths.outputs_3d / "array_projection_splats.html"
    fig.write_html(out, include_plotlyjs=include, full_html=True)
    index = paths.outputs_3d / "array_projection_splats.metadata.json"
    index.write_text(
        json.dumps(
            {
                "html": str(out),
                "displayed_splats": len(limit_rows),
                "total_splats": len(rows),
                "is_sample_data": is_sample,
                "vertical_exaggeration": config.visualization_3d.vertical_exaggeration,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
