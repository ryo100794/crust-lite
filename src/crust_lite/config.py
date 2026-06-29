from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RegionConfig:
    name: str
    bbox: tuple[float, float, float, float]
    crs_local: str
    start_date: date
    end_date: date


@dataclass(frozen=True)
class FilterConfig:
    min_magnitude: float
    max_depth_km: float
    mechanism_min_magnitude: float
    max_events_for_waveforms: int


@dataclass(frozen=True)
class SimulationConfig:
    dt_days: int
    years: int
    n_ensemble: int
    shear_modulus_pa: float
    poisson_ratio: float
    effective_friction_range: tuple[float, float]
    cohesion_pa_range: tuple[float, float]
    effective_normal_stress_pa_range: tuple[float, float]
    random_seed: int


@dataclass(frozen=True)
class OutputConfig:
    write_maps: bool
    write_dashboard: bool
    write_report: bool


@dataclass(frozen=True)
class DataSourceConfig:
    use_fdsn: bool
    use_jshis: bool
    use_gnss: bool
    use_active_faults: bool
    use_waveforms: bool
    fdsn_client: str = "IRIS"
    catalog_source: str = "IRIS"
    event_csv: str | None = None
    mechanism_csv: str | None = None
    gnss_csv: str | None = None
    active_fault_file: str | None = None
    waveform_spectra_csv: str | None = None
    waveform_feature_csv: str | None = None


@dataclass(frozen=True)
class WaveformArrayConfig:
    enabled: bool = True
    time_bin_days: int = 30
    max_events: int = 2000
    max_stations_per_event: int = 256
    min_stations: int = 4
    projection_grid_km: float = 20.0
    projection_radius_km: float = 80.0
    velocity_km_s: float = 3.5
    delay_sigma_s: float = 0.35
    top_projections_per_event: int = 4
    max_projection_rows: int = 100_000
    splat_sigma_horizontal_m: float = 20_000.0
    splat_sigma_vertical_m: float = 12_000.0
    max_splats: int = 100_000
    use_phase: bool = True
    use_group_delay: bool = True
    output_ply: bool = True
    output_html_preview: bool = True


@dataclass(frozen=True)
class ResourceConfig:
    memory_mode: str = "auto"
    db_engine: str = "auto"
    max_memory_fraction: float = 0.35
    in_memory_row_limit: int = 100_000
    db_bulk_load_row_threshold: int = 100_000
    batch_rows_min: int = 10_000
    batch_rows_max: int = 250_000


@dataclass(frozen=True)
class PreprocessingConfig:
    compact_enabled: bool = True
    time_bin_days: int = 30
    spatial_bin_km: float = 10.0
    depth_bin_km: float = 5.0
    magnitude_bin: float = 0.5
    retain_event_level_compact: bool = True
    materialize_database: bool = True
    prefer_compact_tables: bool = True


@dataclass(frozen=True)
class Visualization3DConfig:
    enabled: bool = True
    engine: str = "plotly"
    html_standalone: bool = True
    include_plotlyjs: bool = True
    time_bin_days: int = 30
    mode: str = "cumulative"
    max_events: int = 20000
    max_fault_segments: int = 1000
    max_frames: int = 300
    vertical_exaggeration: float = 2.0
    event_marker_size_min: float = 2.0
    event_marker_size_max: float = 12.0
    color_events_by: str = "magnitude"
    color_faults_by: str = "fault_score"
    color_stress_by: str = "cfs_pa"
    color_failure_by: str = "failure_index_p50"
    show_known_faults: bool = True
    show_inferred_faults: bool = True
    show_gnss_vectors: bool = True
    show_stress: bool = True
    show_failure_index: bool = True
    gnss_vector_scale: float = 50000.0


@dataclass(frozen=True)
class AppConfig:
    region: RegionConfig
    filters: FilterConfig
    simulation: SimulationConfig
    outputs: OutputConfig
    data_sources: DataSourceConfig
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    visualization_3d: Visualization3DConfig = field(default_factory=Visualization3DConfig)
    waveform_array: WaveformArrayConfig = field(default_factory=WaveformArrayConfig)
    path: Path | None = None


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"{field_name} must be an ISO date string")


def _tuple_float(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return tuple(float(v) for v in value)


def _get(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    cfg = parse_config(raw)
    return AppConfig(
        region=cfg.region,
        filters=cfg.filters,
        simulation=cfg.simulation,
        outputs=cfg.outputs,
        data_sources=cfg.data_sources,
        resources=cfg.resources,
        preprocessing=cfg.preprocessing,
        visualization_3d=cfg.visualization_3d,
        waveform_array=cfg.waveform_array,
        path=config_path,
    )


def parse_config(raw: dict[str, Any]) -> AppConfig:
    region_raw = _get(raw, "region")
    filters_raw = _get(raw, "filters")
    sim_raw = _get(raw, "simulation")
    outputs_raw = _get(raw, "outputs")
    sources_raw = _get(raw, "data_sources")
    viz_raw = raw.get("visualization_3d", {})
    resources_raw = raw.get("resources", {})
    preprocessing_raw = raw.get("preprocessing", {})
    waveform_array_raw = raw.get("waveform_array", {})

    bbox = _tuple_float(region_raw.get("bbox"), 4, "region.bbox")
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("region.bbox must be [min_lon, min_lat, max_lon, max_lat]")
    crs_local = str(_get(region_raw, "crs_local"))
    if not crs_local.upper().startswith("EPSG:"):
        raise ValueError("region.crs_local must look like EPSG:6670")

    start = _parse_date(_get(region_raw, "start_date"), "region.start_date")
    end = _parse_date(_get(region_raw, "end_date"), "region.end_date")
    if start > end:
        raise ValueError("region.start_date must be before region.end_date")

    friction_range = _tuple_float(
        sim_raw.get("effective_friction_range"), 2, "simulation.effective_friction_range"
    )
    cohesion_range = _tuple_float(sim_raw.get("cohesion_pa_range"), 2, "simulation.cohesion_pa_range")
    normal_range = _tuple_float(
        sim_raw.get("effective_normal_stress_pa_range"),
        2,
        "simulation.effective_normal_stress_pa_range",
    )
    for name, value in {
        "effective_friction_range": friction_range,
        "cohesion_pa_range": cohesion_range,
        "effective_normal_stress_pa_range": normal_range,
    }.items():
        if value[0] > value[1]:
            raise ValueError(f"simulation.{name} lower bound must be <= upper bound")

    resources = ResourceConfig(**{**ResourceConfig().__dict__, **resources_raw})
    if resources.memory_mode not in {"auto", "low", "balanced", "high"}:
        raise ValueError("resources.memory_mode must be auto, low, balanced, or high")
    if resources.db_engine not in {"auto", "duckdb", "sqlite"}:
        raise ValueError("resources.db_engine must be auto, duckdb, or sqlite")
    if not 0.05 <= resources.max_memory_fraction <= 0.90:
        raise ValueError("resources.max_memory_fraction must be between 0.05 and 0.90")
    if resources.in_memory_row_limit < 0:
        raise ValueError("resources.in_memory_row_limit must be non-negative")

    preprocessing = PreprocessingConfig(**{**PreprocessingConfig().__dict__, **preprocessing_raw})
    if preprocessing.time_bin_days <= 0:
        raise ValueError("preprocessing.time_bin_days must be positive")
    if preprocessing.spatial_bin_km <= 0:
        raise ValueError("preprocessing.spatial_bin_km must be positive")
    if preprocessing.depth_bin_km <= 0:
        raise ValueError("preprocessing.depth_bin_km must be positive")
    if preprocessing.magnitude_bin <= 0:
        raise ValueError("preprocessing.magnitude_bin must be positive")

    viz = Visualization3DConfig(**{**Visualization3DConfig().__dict__, **viz_raw})
    if viz.mode not in {"cumulative", "window"}:
        raise ValueError("visualization_3d.mode must be cumulative or window")
    if viz.engine != "plotly":
        raise ValueError("visualization_3d.engine currently supports only plotly")
    if viz.vertical_exaggeration <= 0:
        raise ValueError("visualization_3d.vertical_exaggeration must be positive")

    waveform_array = WaveformArrayConfig(**{**WaveformArrayConfig().__dict__, **waveform_array_raw})
    if waveform_array.time_bin_days <= 0:
        raise ValueError("waveform_array.time_bin_days must be positive")
    if waveform_array.max_events < 0:
        raise ValueError("waveform_array.max_events must be non-negative")
    if waveform_array.max_stations_per_event <= 0:
        raise ValueError("waveform_array.max_stations_per_event must be positive")
    if waveform_array.min_stations <= 1:
        raise ValueError("waveform_array.min_stations must be greater than 1")
    if waveform_array.projection_grid_km <= 0 or waveform_array.projection_radius_km <= 0:
        raise ValueError("waveform_array projection grid and radius must be positive")
    if waveform_array.velocity_km_s <= 0:
        raise ValueError("waveform_array.velocity_km_s must be positive")
    if waveform_array.delay_sigma_s <= 0:
        raise ValueError("waveform_array.delay_sigma_s must be positive")

    return AppConfig(
        region=RegionConfig(
            name=str(_get(region_raw, "name")),
            bbox=(float(min_lon), float(min_lat), float(max_lon), float(max_lat)),
            crs_local=crs_local,
            start_date=start,
            end_date=end,
        ),
        filters=FilterConfig(
            min_magnitude=float(_get(filters_raw, "min_magnitude")),
            max_depth_km=float(_get(filters_raw, "max_depth_km")),
            mechanism_min_magnitude=float(_get(filters_raw, "mechanism_min_magnitude")),
            max_events_for_waveforms=int(_get(filters_raw, "max_events_for_waveforms")),
        ),
        simulation=SimulationConfig(
            dt_days=int(_get(sim_raw, "dt_days")),
            years=int(_get(sim_raw, "years")),
            n_ensemble=int(_get(sim_raw, "n_ensemble")),
            shear_modulus_pa=float(_get(sim_raw, "shear_modulus_pa")),
            poisson_ratio=float(_get(sim_raw, "poisson_ratio")),
            effective_friction_range=(float(friction_range[0]), float(friction_range[1])),
            cohesion_pa_range=(float(cohesion_range[0]), float(cohesion_range[1])),
            effective_normal_stress_pa_range=(float(normal_range[0]), float(normal_range[1])),
            random_seed=int(_get(sim_raw, "random_seed")),
        ),
        outputs=OutputConfig(
            write_maps=bool(_get(outputs_raw, "write_maps")),
            write_dashboard=bool(_get(outputs_raw, "write_dashboard")),
            write_report=bool(_get(outputs_raw, "write_report")),
        ),
        resources=resources,
        preprocessing=preprocessing,
        waveform_array=waveform_array,
        data_sources=DataSourceConfig(
            use_fdsn=bool(_get(sources_raw, "use_fdsn")),
            use_jshis=bool(_get(sources_raw, "use_jshis")),
            use_gnss=bool(_get(sources_raw, "use_gnss")),
            use_active_faults=bool(_get(sources_raw, "use_active_faults")),
            use_waveforms=bool(_get(sources_raw, "use_waveforms")),
            fdsn_client=str(sources_raw.get("fdsn_client", "IRIS")),
            catalog_source=str(sources_raw.get("catalog_source", sources_raw.get("fdsn_client", "IRIS"))),
            event_csv=sources_raw.get("event_csv"),
            mechanism_csv=sources_raw.get("mechanism_csv"),
            gnss_csv=sources_raw.get("gnss_csv"),
            active_fault_file=sources_raw.get("active_fault_file"),
            waveform_spectra_csv=sources_raw.get("waveform_spectra_csv"),
            waveform_feature_csv=sources_raw.get("waveform_feature_csv"),
        ),
        visualization_3d=viz,
    )
