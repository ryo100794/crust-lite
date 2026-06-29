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
from datetime import UTC, datetime
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
from crust_lite.viz.japan_outline import local_context_outlines

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
    "path_family",
    "primitive_type",
    "late_phase_delay_s",
    "excess_path_km",
    "residual_spread_s",
    "positive_residual_fraction",
    "scatter_weight",
    "frequency_hz",
    "beam_energy",
    "phase_coherence",
    "delay_fit",
    "array_coherence",
    "beam_power",
    "mean_amplitude_log",
    "n_stations",
    "aperture_km",
    "velocity_km_s",
    "velocity_model",
    "slowness_x_s_per_km",
    "slowness_y_s_per_km",
    "frequency_band",
    "phase_resultant_rad",
    "gaussian_splat_sigma_m",
    "dominant_source",
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
    "source_event_x_m",
    "source_event_y_m",
    "source_event_z_m",
    "primitive_type",
    "path_family",
    "late_phase_delay_s",
    "excess_path_km",
    "residual_spread_s",
    "positive_residual_fraction",
    "scatter_weight",
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
    "array_coherence",
    "beam_power",
    "aperture_km",
    "frequency_band",
    "velocity_model",
    "slowness_x_s_per_km",
    "slowness_y_s_per_km",
    "dominant_source",
    "interpretation",
    "is_sample_data",
]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _time_bin(value: str, start: datetime, days: int) -> tuple[int, str]:
    current = _parse_time(value)
    delta_days = max(0, (current - start).days)
    idx = delta_days // max(1, days)
    bin_start = start.timestamp() + idx * max(1, days) * 86400
    return int(idx), _format_time(datetime.fromtimestamp(bin_start, UTC))


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


def _source_family(row: dict[str, Any]) -> str:
    """Classify spectrum provenance without storing long credential-path strings."""
    text = " ".join(str(row.get(key, "")) for key in ("source", "network", "station_id", "channel")).lower()
    if "network=0101" in text or "hi-net" in text or "hinet" in text:
        return "hinet_0101"
    if "network=0103a" in text:
        return "nied_fnet_strong_motion_0103a"
    if "network=0103" in text or "f-net" in text or "fnet" in text:
        return "nied_fnet_0103"
    if "nied" in text:
        return "nied"
    if "fdsn" in text or "iris" in text or "earthscope" in text:
        return "fdsn"
    return "unknown"


def _dominant_source(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[_source_family(row)] += 1
    if not counts:
        return "unknown"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) == 1:
        return ordered[0][0]
    return "mixed:" + ",".join(f"{name}:{count}" for name, count in ordered[:3])


def _source_weight(config: AppConfig, dominant_source: str) -> float:
    if not config.waveform_array.synthetic_aperture_enabled:
        return 1.0
    source = dominant_source.lower()
    if "hinet_0101" in source:
        return float(config.waveform_array.hinet_source_boost)
    return 1.0


def _station_aperture_km(stations: list[dict[str, float]]) -> float:
    if len(stations) < 2:
        return 0.0
    xs = [float(st["x_m"]) for st in stations]
    ys = [float(st["y_m"]) for st in stations]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 1000.0


def _slowness_vector_s_per_km(event_x: float, event_y: float, gx: float, gy: float, velocity_km_s: float) -> tuple[float, float]:
    dx_km = (gx - event_x) / 1000.0
    dy_km = (gy - event_y) / 1000.0
    distance_km = math.hypot(dx_km, dy_km)
    if distance_km <= 1e-9:
        return 0.0, 0.0
    slowness = 1.0 / max(velocity_km_s, 1e-9)
    return slowness * dx_km / distance_km, slowness * dy_km / distance_km


def _resolution_sigma_m(config: AppConfig, frequency_hz: float, aperture_km: float) -> float:
    if not config.waveform_array.synthetic_aperture_enabled:
        return float(config.waveform_array.splat_sigma_horizontal_m)
    frequency = max(float(frequency_hz), 1e-6)
    wavelength_m = config.waveform_array.velocity_km_s * 1000.0 / frequency
    aperture_m = max(float(aperture_km) * 1000.0, 1.0)
    aperture_gain = max(1.0, aperture_m / max(wavelength_m, 1.0))
    grid_floor_m = 0.5 * config.waveform_array.projection_grid_km * 1000.0
    sigma = max(grid_floor_m, 0.5 * wavelength_m / math.sqrt(aperture_gain))
    return max(
        float(config.waveform_array.resolution_sigma_min_m),
        min(float(config.waveform_array.resolution_sigma_max_m), sigma),
    )


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
    return clamp01(energy), clamp01(phase_coherence), clamp01(delay_fit), phase_result if math.isfinite(phase_result) else 0.0


def _relative_delay_terms(
    stations: list[dict[str, float]],
    gx: float,
    gy: float,
    velocity_m_s: float,
) -> tuple[list[float], list[float], list[float]]:
    distances = [math.hypot(float(st["x_m"]) - gx, float(st["y_m"]) - gy) for st in stations]
    reference_distance = median(distances)
    predicted = [(dist - reference_distance) / velocity_m_s for dist in distances]
    observed_delay = [float(st["group_delay_s"]) for st in stations]
    delay_reference = median(observed_delay)
    observed_delay_rel = [value - delay_reference for value in observed_delay]
    weights = [max(float(st["amplitude"]), 1e-30) for st in stations]
    max_weight = max(weights or [1.0])
    weights = [0.25 + 0.75 * (weight / max_weight) for weight in weights]
    return predicted, observed_delay_rel, weights


def _weighted_median(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, key=lambda item: item[0])
    half = 0.5 * sum(max(weight, 0.0) for _value, weight in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += max(weight, 0.0)
        if acc >= half:
            return value
    return ordered[-1][0]


def _late_phase_metrics(
    stations: list[dict[str, float]],
    gx: float,
    gy: float,
    frequency_hz: float,
    velocity_m_s: float,
    delay_sigma_s: float,
    use_phase: bool,
) -> tuple[float, float, float, float, float, float]:
    """Score coherent late energy after removing the direct-arrival trend.

    The result is a relative reflection/scattering indicator. It is not a
    solved reflector location or a migrated seismic image.
    """
    predicted, observed_delay_rel, weights = _relative_delay_terms(stations, gx, gy, velocity_m_s)
    residuals = [obs - pred for obs, pred in zip(observed_delay_rel, predicted, strict=False)]
    positive_threshold = max(0.05, 0.5 * delay_sigma_s)
    positive = [
        (residual, weight)
        for residual, weight in zip(residuals, weights, strict=False)
        if residual > positive_threshold
    ]
    total_weight = max(sum(weights), 1e-12)
    positive_weight = sum(weight for _residual, weight in positive)
    positive_fraction = clamp01(positive_weight / total_weight)
    if len(positive) < 2 or positive_fraction < 0.20:
        return 0.0, 0.0, 0.0, 0.0, positive_fraction, 0.0

    late_delay_s = max(positive_threshold, min(30.0, _weighted_median(positive)))
    window_s = max(0.25, 2.0 * delay_sigma_s)
    weighted_fits = [
        math.exp(-0.5 * ((residual - late_delay_s) / window_s) ** 2) * weight
        for residual, weight in zip(residuals, weights, strict=False)
        if residual > 0.0
    ]
    late_fit = clamp01(sum(weighted_fits) / total_weight)

    late_phase = 0.5
    phase_result = 0.0
    if use_phase:
        plus: list[tuple[float, float]] = []
        minus: list[tuple[float, float]] = []
        for st, pred, weight in zip(stations, predicted, weights, strict=False):
            shift = 2.0 * math.pi * frequency_hz * (pred + late_delay_s)
            phase = float(st["phase_rad"])
            plus.append((_wrap_phase(phase - shift), weight))
            minus.append((_wrap_phase(phase + shift), weight))
        plus_score, plus_phase = _phase_result(plus)
        minus_score, minus_phase = _phase_result(minus)
        if plus_score >= minus_score:
            late_phase = plus_score
            phase_result = plus_phase
        else:
            late_phase = minus_score
            phase_result = minus_phase

    mean_positive = sum(residual * weight for residual, weight in positive) / max(positive_weight, 1e-12)
    spread_s = math.sqrt(
        sum(((residual - mean_positive) ** 2) * weight for residual, weight in positive) / max(positive_weight, 1e-12)
    )
    energy = clamp01((0.55 * late_phase + 0.45 * late_fit) * (0.35 + 0.65 * positive_fraction))
    return energy, late_phase, late_delay_s, spread_s, positive_fraction, phase_result


def _late_path_family(late_energy: float, spread_s: float, delay_sigma_s: float, positive_fraction: float) -> tuple[str, str]:
    if late_energy >= 0.35 and positive_fraction >= 0.45 and spread_s <= max(1.0, 4.0 * delay_sigma_s):
        return "late_phase_reflection", "reflected"
    if late_energy >= 0.22:
        return "late_phase_scattering", "scattered"
    return "late_phase_residual", "residual"


def _late_projection_depth_m(event_z_m: float, late_delay_s: float, velocity_km_s: float, max_depth_km: float) -> float:
    # Single-bounce reflection is approximated as half the extra path length;
    # scattering/conversion can deviate, so this is stored as an indicator.
    extra_path_m = max(0.0, late_delay_s * velocity_km_s * 1000.0)
    max_depth_m = max(1_000.0, max_depth_km * 1000.0)
    return min(max_depth_m, max(0.0, event_z_m + 0.5 * extra_path_m))


def _projection_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return float(row.get("beam_power", 0.0) or 0.0), float(row.get("beam_energy", 0.0) or 0.0)


def _select_projection_rows(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    direct = sorted(
        [row for row in candidates if row.get("primitive_type") == "direct"],
        key=_projection_sort_key,
        reverse=True,
    )
    late = sorted(
        [row for row in candidates if row.get("primitive_type") != "direct"],
        key=_projection_sort_key,
        reverse=True,
    )
    if not late:
        return direct[:limit]
    direct_quota = max(1, min(len(direct), limit // 2))
    late_quota = max(1, limit - direct_quota)
    selected = [*direct[:direct_quota], *late[:late_quota]]
    selected_ids = {id(row) for row in selected}
    if len(selected) < limit:
        for row in sorted(candidates, key=_projection_sort_key, reverse=True):
            if id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))
                if len(selected) >= limit:
                    break
    return sorted(selected, key=_projection_sort_key, reverse=True)[:limit]


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, "unknown"))] += 1
    return dict(sorted(counts.items()))


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

    start = datetime.combine(config.region.start_date, datetime.min.time(), tzinfo=UTC)
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
            dominant_source = _dominant_source(rows)
            source_weight = _source_weight(config, dominant_source)
            aperture_km = _station_aperture_km(stations)
            splat_sigma_m = _resolution_sigma_m(config, freq, aperture_km)
            frequency_band = f"{freq:g} Hz point spectrum"
            velocity_model = f"homogeneous_{config.waveform_array.velocity_km_s:g}_km_s"
            method = (
                "synthetic_aperture_delay_sum_phase_group_delay_projection"
                if config.waveform_array.synthetic_aperture_enabled
                else "delay_and_sum_phase_group_delay_projection"
            )
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
                if config.waveform_array.use_phase and config.waveform_array.use_group_delay:
                    array_coherence = clamp01(phase_coherence * delay_fit)
                elif config.waveform_array.use_phase:
                    array_coherence = clamp01(phase_coherence)
                elif config.waveform_array.use_group_delay:
                    array_coherence = clamp01(delay_fit)
                else:
                    array_coherence = clamp01(energy)
                beam_power = max(0.0, energy) * math.log1p(len(stations)) * source_weight
                slowness_x, slowness_y = _slowness_vector_s_per_km(
                    event_x, event_y, gx, gy, config.waveform_array.velocity_km_s
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
                        "path_family": "direct",
                        "primitive_type": "direct",
                        "late_phase_delay_s": 0.0,
                        "excess_path_km": 0.0,
                        "residual_spread_s": 0.0,
                        "positive_residual_fraction": 0.0,
                        "scatter_weight": 0.0,
                        "frequency_hz": freq,
                        "beam_energy": clamp01(energy * source_weight),
                        "phase_coherence": phase_coherence,
                        "delay_fit": delay_fit,
                        "array_coherence": array_coherence,
                        "beam_power": beam_power,
                        "mean_amplitude_log": mean_amp,
                        "n_stations": len(stations),
                        "aperture_km": aperture_km,
                        "velocity_km_s": config.waveform_array.velocity_km_s,
                        "velocity_model": velocity_model,
                        "slowness_x_s_per_km": slowness_x,
                        "slowness_y_s_per_km": slowness_y,
                        "frequency_band": frequency_band,
                        "phase_resultant_rad": phase_result,
                        "gaussian_splat_sigma_m": splat_sigma_m,
                        "dominant_source": dominant_source,
                        "projection_rank": 0,
                        "projection_method": method,
                        "is_sample_data": is_sample,
                    }
                )
                if config.waveform_array.synthetic_aperture_enabled and config.waveform_array.use_group_delay:
                    late_energy, late_phase, late_delay_s, spread_s, positive_fraction, late_phase_result = _late_phase_metrics(
                        stations,
                        gx,
                        gy,
                        freq,
                        velocity_m_s,
                        config.waveform_array.delay_sigma_s,
                        config.waveform_array.use_phase,
                    )
                    if late_energy >= 0.08:
                        path_family, primitive_type = _late_path_family(
                            late_energy, spread_s, config.waveform_array.delay_sigma_s, positive_fraction
                        )
                        late_z_m = _late_projection_depth_m(
                            z_m,
                            late_delay_s,
                            config.waveform_array.velocity_km_s,
                            config.filters.max_depth_km,
                        )
                        excess_path_km = max(0.0, late_delay_s * config.waveform_array.velocity_km_s)
                        late_sigma = max(
                            splat_sigma_m,
                            min(config.waveform_array.resolution_sigma_max_m, splat_sigma_m + 0.25 * excess_path_km * 1000.0),
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
                                "projection_z_m": late_z_m,
                                "path_family": path_family,
                                "primitive_type": primitive_type,
                                "late_phase_delay_s": late_delay_s,
                                "excess_path_km": excess_path_km,
                                "residual_spread_s": spread_s,
                                "positive_residual_fraction": positive_fraction,
                                "scatter_weight": clamp01(late_energy * positive_fraction),
                                "frequency_hz": freq,
                                "beam_energy": clamp01(late_energy * source_weight),
                                "phase_coherence": late_phase,
                                "delay_fit": clamp01(1.0 / (1.0 + spread_s / max(config.waveform_array.delay_sigma_s, 1e-9))),
                                "array_coherence": clamp01(late_phase * (0.5 + 0.5 * positive_fraction)),
                                "beam_power": late_energy * math.log1p(len(stations)) * source_weight * 0.9,
                                "mean_amplitude_log": mean_amp,
                                "n_stations": len(stations),
                                "aperture_km": aperture_km,
                                "velocity_km_s": config.waveform_array.velocity_km_s,
                                "velocity_model": velocity_model,
                                "slowness_x_s_per_km": slowness_x,
                                "slowness_y_s_per_km": slowness_y,
                                "frequency_band": f"{freq:g} Hz late-phase window",
                                "phase_resultant_rad": late_phase_result,
                                "gaussian_splat_sigma_m": late_sigma,
                                "dominant_source": dominant_source,
                                "projection_rank": 0,
                                "projection_method": "synthetic_aperture_late_phase_reflection_scattering_projection",
                                "is_sample_data": is_sample,
                            }
                        )
        for rank, row in enumerate(_select_projection_rows(event_candidates, config.waveform_array.top_projections_per_event), start=1):
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
            "method": "synthetic-aperture direct plus late-phase reflection/scattering projection using phase and group delay",
            "interpretation": "relative coherent array energy image including direct, reflected, scattered, and residual late-phase indicators; not an earthquake prediction or unique subsurface inversion",
            "event_count_with_spectra": len(allowed_events),
            "projection_rows": len(projection_rows),
            "projection_type_counts": _count_values(projection_rows, "primitive_type"),
            "path_family_counts": _count_values(projection_rows, "path_family"),
            "synthetic_aperture_enabled": config.waveform_array.synthetic_aperture_enabled,
            "resolution_sigma_min_m": config.waveform_array.resolution_sigma_min_m,
            "resolution_sigma_max_m": config.waveform_array.resolution_sigma_max_m,
            "hinet_source_boost": config.waveform_array.hinet_source_boost,
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
            "interpretation": "rendering primitives for relative waveform-derived direct and late-phase structure indicators; not a deterministic subsurface model",
            "splat_rows": len(splat_rows),
            "primitive_type_counts": _count_values(splat_rows, "primitive_type"),
            "path_family_counts": _count_values(splat_rows, "path_family"),
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
        "projection_method": (
            "synthetic_aperture_direct_late_phase_reflection_scattering_projection"
            if config.waveform_array.synthetic_aperture_enabled
            else "delay_and_sum_phase_group_delay_projection"
        ),
        "projection_type_counts": _count_values(projection_rows, "primitive_type"),
        "path_family_counts": _count_values(projection_rows, "path_family"),
        "splat_type_counts": _count_values(splat_rows, "primitive_type"),
        "synthetic_aperture_enabled": config.waveform_array.synthetic_aperture_enabled,
        "resolution_sigma_min_m": config.waveform_array.resolution_sigma_min_m,
        "resolution_sigma_max_m": config.waveform_array.resolution_sigma_max_m,
        "hinet_source_boost": config.waveform_array.hinet_source_boost,
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
    rows = sorted(
        projection_rows,
        key=lambda row: (float(row.get("beam_power", 0.0) or 0.0), float(row.get("beam_energy", 0.0) or 0.0)),
        reverse=True,
    )
    rows = rows[: config.waveform_array.max_splats]
    splats: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        amplitude = clamp01(float(row.get("beam_energy", 0.0) or 0.0))
        array_coherence = clamp01(float(row.get("array_coherence", amplitude) or 0.0))
        sigma_xy = float(row.get("gaussian_splat_sigma_m", config.waveform_array.splat_sigma_horizontal_m) or 0.0)
        sigma_z = max(
            float(config.waveform_array.splat_sigma_vertical_m),
            min(float(config.waveform_array.resolution_sigma_max_m), 0.6 * sigma_xy),
        )
        primitive_type = str(row.get("primitive_type", "direct") or "direct")
        color_r, color_g, color_b = _primitive_rgb(amplitude, primitive_type)
        splats.append(
            {
                "primitive_id": f"gs_{idx:08d}",
                "event_id": row.get("event_id", ""),
                "time_utc": row.get("time_utc", ""),
                "time_bin_index": row.get("time_bin_index", 0),
                "x_m": row.get("projection_x_m", row.get("x_m", 0.0)),
                "y_m": row.get("projection_y_m", row.get("y_m", 0.0)),
                "z_m": row.get("projection_z_m", row.get("z_m", 0.0)),
                "source_event_x_m": row.get("x_m", 0.0),
                "source_event_y_m": row.get("y_m", 0.0),
                "source_event_z_m": row.get("z_m", 0.0),
                "primitive_type": primitive_type,
                "path_family": row.get("path_family", "direct"),
                "late_phase_delay_s": row.get("late_phase_delay_s", 0.0),
                "excess_path_km": row.get("excess_path_km", 0.0),
                "residual_spread_s": row.get("residual_spread_s", 0.0),
                "positive_residual_fraction": row.get("positive_residual_fraction", 0.0),
                "scatter_weight": row.get("scatter_weight", 0.0),
                "sigma_x_m": sigma_xy,
                "sigma_y_m": sigma_xy,
                "sigma_z_m": sigma_z,
                "amplitude": amplitude,
                "opacity": clamp01(0.10 + 0.60 * amplitude + 0.30 * array_coherence),
                "phase_rad": row.get("phase_resultant_rad", 0.0),
                "color_r": color_r,
                "color_g": color_g,
                "color_b": color_b,
                "source_projection_rank": row.get("projection_rank", 0),
                "source_frequency_hz": row.get("frequency_hz", 0.0),
                "source_projection_method": row.get("projection_method", ""),
                "array_coherence": array_coherence,
                "beam_power": row.get("beam_power", 0.0),
                "aperture_km": row.get("aperture_km", 0.0),
                "frequency_band": row.get("frequency_band", ""),
                "velocity_model": row.get("velocity_model", ""),
                "slowness_x_s_per_km": row.get("slowness_x_s_per_km", 0.0),
                "slowness_y_s_per_km": row.get("slowness_y_s_per_km", 0.0),
                "dominant_source": row.get("dominant_source", "unknown"),
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


def _primitive_rgb(value: float, primitive_type: str) -> tuple[int, int, int]:
    v = clamp01(value)
    if primitive_type == "reflected":
        return int(210 + 45 * v), int(100 + 80 * v), int(20 + 70 * (1.0 - v))
    if primitive_type == "scattered":
        return int(125 + 95 * v), int(60 + 80 * (1.0 - v)), int(190 + 45 * v)
    if primitive_type == "residual":
        return int(90 + 120 * v), int(90 + 110 * v), int(90 + 80 * v)
    return _energy_rgb(value)


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


def _hex_color(row: dict[str, Any], alpha_scale: float = 1.0) -> str:
    r = max(0, min(255, int(row.get("color_r", 180) or 180)))
    g = max(0, min(255, int(row.get("color_g", 120) or 120)))
    b = max(0, min(255, int(row.get("color_b", 80) or 80)))
    return f"rgb({r},{g},{b})"


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _synthetic_relief_m(lon: float, lat: float, is_land: bool) -> float:
    """Lightweight context relief used when no DEM is bundled with the run.

    The surface is cartographic context only. It keeps land above sea level and
    offshore areas slightly below zero so splats can be inspected against a
    recognizable terrain-like reference without loading a large DEM into HTML.
    """
    ridge = 850.0 * math.exp(-((lon - 138.0) / 3.8) ** 2 - ((lat - 36.7) / 2.5) ** 2)
    northeast = 420.0 * math.exp(-((lon - 140.6) / 2.6) ** 2 - ((lat - 39.3) / 2.2) ** 2)
    southwest = 360.0 * math.exp(-((lon - 131.2) / 2.7) ** 2 - ((lat - 33.0) / 1.8) ** 2)
    roughness = 120.0 * math.sin(lon * 1.7) * math.cos(lat * 1.3)
    if is_land:
        return max(20.0, ridge + northeast + southwest + roughness)
    trench = -900.0 * math.exp(-((lon - 143.5) / 2.6) ** 2 - ((lat - 38.2) / 4.4) ** 2)
    ocean = -120.0 - 180.0 * max(0.0, lon - 137.0) / 18.0
    return min(-20.0, ocean + trench)


def _terrain_overlay_traces(go: Any, config: AppConfig) -> list[Any]:
    projector = LocalProjector(config.region)
    outlines = local_context_outlines(config.region.bbox, margin_deg=3.0)
    min_lon, min_lat, max_lon, max_lat = config.region.bbox
    nx = 96
    ny = 80
    lons = [min_lon + (max_lon - min_lon) * i / (nx - 1) for i in range(nx)]
    lats = [min_lat + (max_lat - min_lat) * j / (ny - 1) for j in range(ny)]
    x_grid: list[list[float]] = []
    y_grid: list[list[float]] = []
    z_grid: list[list[float]] = []
    color_grid: list[list[float]] = []
    polygons = [outline["coordinates"] for outline in outlines if len(outline["coordinates"]) >= 3]
    for lat in lats:
        x_row: list[float] = []
        y_row: list[float] = []
        z_row: list[float] = []
        color_row: list[float] = []
        for lon in lons:
            x_m, y_m = projector.lonlat_to_xy(lon, lat)
            is_land = any(_point_in_polygon(lon, lat, polygon) for polygon in polygons)
            relief = _synthetic_relief_m(lon, lat, is_land)
            x_row.append(x_m)
            y_row.append(y_m)
            z_row.append(relief)
            color_row.append(relief)
        x_grid.append(x_row)
        y_grid.append(y_row)
        z_grid.append(z_row)
        color_grid.append(color_row)

    hover = (
        "terrain overlay<br>"
        "source=offline synthetic context relief<br>"
        "not analytical DEM; z is display elevation [m]"
    )
    terrain = go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=color_grid,
        colorscale=[
            [0.00, "#0b4f71"],
            [0.32, "#3a8fb7"],
            [0.38, "#b7d7bf"],
            [0.62, "#628c4f"],
            [0.82, "#9a7b4f"],
            [1.00, "#f3efe2"],
        ],
        cmin=-1200,
        cmax=1800,
        opacity=0.44,
        showscale=True,
        colorbar={"title": "terrain display elevation [m]"},
        name="terrain overlay",
        text=[[hover for _lon in lons] for _lat in lats],
        hoverinfo="text",
    )

    outline_traces: list[Any] = []
    for outline in outlines:
        coords = outline["coordinates"]
        projected = [projector.lonlat_to_xy(lon, lat) for lon, lat in coords]
        outline_traces.append(
            go.Scatter3d(
                x=[x for x, _y in projected],
                y=[y for _x, y in projected],
                z=[180.0] * len(projected),
                mode="lines",
                line={"color": "#16351f", "width": 5},
                name=f"Japan terrain outline - {outline['name']}",
                text=[f"Japan outline<br>island={outline['name']}<br>terrain overlay context"] * len(projected),
                hoverinfo="text",
            )
        )
    return [terrain, *outline_traces]


def _ellipsoid_mesh(row: dict[str, Any], vertical_exaggeration: float, n_theta: int = 18, n_phi: int = 10) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    cx = float(row.get("x_m", 0.0) or 0.0)
    cy = float(row.get("y_m", 0.0) or 0.0)
    cz = -1.0 * float(row.get("z_m", 0.0) or 0.0) * vertical_exaggeration
    sx = max(float(row.get("sigma_x_m", 0.0) or 0.0), 1.0)
    sy = max(float(row.get("sigma_y_m", 0.0) or 0.0), 1.0)
    sz = max(float(row.get("sigma_z_m", 0.0) or 0.0) * vertical_exaggeration, 1.0)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for ip in range(n_phi + 1):
        phi = -0.5 * math.pi + math.pi * ip / n_phi
        for it in range(n_theta):
            theta = 2.0 * math.pi * it / n_theta
            xs.append(cx + sx * math.cos(phi) * math.cos(theta))
            ys.append(cy + sy * math.cos(phi) * math.sin(theta))
            zs.append(cz + sz * math.sin(phi))
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    for ip in range(n_phi):
        for it in range(n_theta):
            a = ip * n_theta + it
            b = ip * n_theta + (it + 1) % n_theta
            c = (ip + 1) * n_theta + it
            d = (ip + 1) * n_theta + (it + 1) % n_theta
            ii.extend([a, b])
            jj.extend([b, d])
            kk.extend([c, c])
    return xs, ys, zs, ii, jj, kk


def _write_splat_preview(config: AppConfig, paths: ProjectPaths, rows: list[dict[str, Any]], is_sample: bool) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return
    paths.outputs_3d.mkdir(parents=True, exist_ok=True)
    limit_rows = rows[: min(len(rows), 20000)]
    ellipsoid_rows = sorted(
        limit_rows,
        key=lambda row: (float(row.get("beam_power", 0.0) or 0.0), float(row.get("amplitude", 0.0) or 0.0)),
        reverse=True,
    )[: min(360, len(limit_rows))]
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
            row.get("array_coherence", ""),
            row.get("aperture_km", ""),
            row.get("sigma_x_m", ""),
            row.get("dominant_source", ""),
            row.get("is_sample_data", is_sample),
        ]
        for row in limit_rows
    ]
    traces: list[Any] = [*_terrain_overlay_traces(go, config)]
    traces.append(
        go.Scatter3d(
            x=[float(row.get("x_m", 0.0) or 0.0) for row in limit_rows],
            y=[float(row.get("y_m", 0.0) or 0.0) for row in limit_rows],
            z=z_values,
            mode="markers",
            marker={
                "size": [max(2.0, min(18.0, 3.0 + 14.0 * float(row.get("amplitude", 0.0) or 0.0))) for row in limit_rows],
                "color": colors,
                "colorscale": "Turbo",
                "colorbar": {"title": "coherent energy [-]"},
                "opacity": 0.50,
            },
            customdata=customdata,
            hovertemplate=(
                "primitive=%{customdata[0]}<br>event=%{customdata[1]}<br>"
                "time=%{customdata[2]}<br>primitive_type=%{customdata[3]}<br>"
                "path_family=%{customdata[4]}<br>frequency=%{customdata[5]} Hz<br>"
                "energy=%{customdata[6]:.3f}<br>array_coherence=%{customdata[7]:.3f}<br>"
                "late_delay=%{customdata[8]:.2f} s<br>excess_path=%{customdata[9]:.2f} km<br>"
                "aperture=%{customdata[10]:.1f} km<br>sigma=%{customdata[11]:.0f} m<br>"
                "source=%{customdata[12]}<br>is_sample_data=%{customdata[13]}<extra></extra>"
            ),
            name="splat centers",
        )
    )
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for row in ellipsoid_rows:
        sx = float(row.get("source_event_x_m", row.get("x_m", 0.0)) or 0.0)
        sy = float(row.get("source_event_y_m", row.get("y_m", 0.0)) or 0.0)
        sz = -1.0 * float(row.get("source_event_z_m", row.get("z_m", 0.0)) or 0.0) * config.visualization_3d.vertical_exaggeration
        line_x.extend([sx, float(row.get("x_m", 0.0) or 0.0), None])
        line_y.extend([sy, float(row.get("y_m", 0.0) or 0.0), None])
        line_z.extend([sz, -1.0 * float(row.get("z_m", 0.0) or 0.0) * config.visualization_3d.vertical_exaggeration, None])
    traces.append(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": "#334155", "width": 2},
            name="hypocenter to projected splat",
            hoverinfo="skip",
        )
    )
    for idx, row in enumerate(ellipsoid_rows, start=1):
        xs, ys, zs, ii, jj, kk = _ellipsoid_mesh(row, config.visualization_3d.vertical_exaggeration)
        hover = (
            f"ellipsoid_splat={row.get('primitive_id')}<br>"
            f"event={row.get('event_id')}<br>"
            f"sigma_x_m={row.get('sigma_x_m')}<br>"
            f"sigma_z_m={row.get('sigma_z_m')}<br>"
            f"primitive_type={row.get('primitive_type')}<br>"
            f"path_family={row.get('path_family')}<br>"
            f"late_phase_delay_s={row.get('late_phase_delay_s')}<br>"
            f"array_coherence={row.get('array_coherence')}<br>"
            "ellipsoid_is_resolution_kernel=true"
        )
        traces.append(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=ii,
                j=jj,
                k=kk,
                color=_hex_color(row),
                opacity=0.11 + 0.28 * float(row.get("array_coherence", 0.0) or 0.0),
                name="top Gaussian splat ellipsoids" if idx == 1 else f"splat ellipsoid {idx}",
                text=[hover] * len(xs),
                hoverinfo="text",
                showlegend=idx == 1,
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            "Waveform array projection Gaussian primitives "
            f"(high-resolution terrain overlay; top {len(ellipsoid_rows)} rendered as ellipsoids; vertical exaggeration {config.visualization_3d.vertical_exaggeration:g}x; "
            f"is_sample_data={str(is_sample).lower()})"
        ),
        height=1120,
        autosize=True,
        uirevision="crust-lite-camera",
        scene={
            "xaxis_title": "Easting in local CRS [m]",
            "yaxis_title": "Northing in local CRS [m]",
            "zaxis_title": "Elevation-like depth display [m], depth exaggerated",
            "aspectmode": "data",
            "uirevision": "crust-lite-camera",
            "dragmode": "orbit",
        },
        annotations=[
            {
                "text": "Terrain is a lightweight display overlay, not an analytical DEM. Ellipsoids are resolution kernels from array-projection sigma. Lines connect catalog hypocenters to projected coherent-energy centers.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.07,
                "showarrow": False,
                "align": "left",
                "bgcolor": "rgba(255,255,255,0.86)",
                "bordercolor": "#cbd5e1",
            }
        ],
        margin={"l": 0, "r": 0, "b": 0, "t": 54},
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
                "displayed_ellipsoid_splats": len(ellipsoid_rows),
                "total_splats": len(rows),
                "is_sample_data": is_sample,
                "vertical_exaggeration": config.visualization_3d.vertical_exaggeration,
                "synthetic_aperture_enabled": config.waveform_array.synthetic_aperture_enabled,
                "uses_phase": config.waveform_array.use_phase,
                "uses_group_delay": config.waveform_array.use_group_delay,
                "primitive_type_counts": _count_values(rows, "primitive_type"),
                "path_family_counts": _count_values(rows, "path_family"),
                "terrain_overlay": "synthetic_context_surface_with_japan_outline",
                "terrain_grid": {"nx": 96, "ny": 80},
                "ellipsoid_mesh_resolution": {"n_theta": 18, "n_phi": 10},
                "rendering": "high-resolution splat centers plus top translucent ellipsoid kernels over a lightweight terrain context surface; includes direct, reflected, scattered, and residual late-phase primitives",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
