from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector
from crust_lite.io.database import latest_stress_by_segment
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import read_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def _import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _plot_fault_lines(ax: Any, features: list[dict[str, Any]], color: str, label: str) -> None:
    plotted = False
    for feature in features:
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        xs = props.get("trace_x_m")
        ys = props.get("trace_y_m")
        if xs and ys:
            ax.plot(xs, ys, color=color, linewidth=2.0, label=label if not plotted else None)
            plotted = True
            continue
        local = geom.get("local_trace_m")
        if local:
            ax.plot([p[0] for p in local], [p[1] for p in local], color=color, linewidth=2.0)


def _draw_map_overlay(ax: Any, config: AppConfig) -> None:
    projector = LocalProjector(config.region)
    min_lon, min_lat, max_lon, max_lat = config.region.bbox
    lons = np.linspace(min_lon, max_lon, 6)
    lats = np.linspace(min_lat, max_lat, 6)

    corners = [
        projector.lonlat_to_xy(min_lon, min_lat),
        projector.lonlat_to_xy(max_lon, min_lat),
        projector.lonlat_to_xy(max_lon, max_lat),
        projector.lonlat_to_xy(min_lon, max_lat),
        projector.lonlat_to_xy(min_lon, min_lat),
    ]
    ax.fill(
        [point[0] for point in corners],
        [point[1] for point in corners],
        color="#eef5f9",
        alpha=0.38,
        zorder=-20,
        label="map overlay",
    )
    ax.plot(
        [point[0] for point in corners],
        [point[1] for point in corners],
        color="#4b5563",
        linewidth=1.2,
        zorder=-10,
    )

    for lon in lons:
        line = [projector.lonlat_to_xy(float(lon), float(lat)) for lat in np.linspace(min_lat, max_lat, 32)]
        ax.plot(
            [point[0] for point in line],
            [point[1] for point in line],
            color="#9aa6b2",
            linewidth=0.55,
            alpha=0.7,
            zorder=-9,
        )
        ax.text(line[0][0], line[0][1], f"{lon:.2f}E", fontsize=7, color="#52606d")

    for lat in lats:
        line = [projector.lonlat_to_xy(float(lon), float(lat)) for lon in np.linspace(min_lon, max_lon, 32)]
        ax.plot(
            [point[0] for point in line],
            [point[1] for point in line],
            color="#9aa6b2",
            linewidth=0.55,
            alpha=0.7,
            zorder=-9,
        )
        ax.text(line[0][0], line[0][1], f"{lat:.2f}N", fontsize=7, color="#52606d")

    ax.set_aspect("equal", adjustable="box")


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.clf()


def write_static_maps(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    plt = _import_pyplot()
    event_rows = read_table(paths.data_interim / "event_qc.parquet") if (
        paths.data_interim / "event_qc.parquet"
    ).exists() else []
    known = read_features(paths.data_processed / "fault_segment.gpkg") if (
        paths.data_processed / "fault_segment.gpkg"
    ).exists() else []
    inferred = read_features(paths.data_processed / "inferred_faults.gpkg") if (
        paths.data_processed / "inferred_faults.gpkg"
    ).exists() else []
    gnss = read_table(paths.data_processed / "gnss_features.parquet") if (
        paths.data_processed / "gnss_features.parquet"
    ).exists() else []
    stress_latest = latest_stress_by_segment(paths) or {}
    scenarios = read_table(paths.outputs_tables / "failure_scenarios.parquet") if (
        paths.outputs_tables / "failure_scenarios.parquet"
    ).exists() else []

    outputs = []
    for name, title in [
        ("event_map.png", "Event catalog with map overlay"),
        ("known_faults_map.png", "Known faults with map overlay"),
        ("inferred_faults_map.png", "Inferred faults with map overlay"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_map_overlay(ax, config)
        if event_rows:
            ax.scatter(
                [float(row["x_m"]) for row in event_rows],
                [float(row["y_m"]) for row in event_rows],
                s=[max(8.0, float(row["magnitude"]) ** 2.0) for row in event_rows],
                c=[float(row["depth_km"]) for row in event_rows],
                cmap="viridis_r",
                alpha=0.75,
                label="events",
            )
        _plot_fault_lines(ax, known, "#444444", "known faults")
        _plot_fault_lines(ax, inferred, "#c44e52", "inferred faults")
        if gnss:
            scale = 50_000.0
            ax.quiver(
                [float(row["x_m"]) for row in gnss],
                [float(row["y_m"]) for row in gnss],
                [float(row["east_velocity_m_per_yr"]) * scale for row in gnss],
                [float(row["north_velocity_m_per_yr"]) * scale for row in gnss],
                color="#4c72b0",
                label="GNSS velocity",
            )
        ax.set_title(title)
        ax.set_xlabel("Easting in local CRS [m]")
        ax.set_ylabel("Northing in local CRS [m]")
        ax.legend(loc="best")
        out = paths.outputs_maps / name
        _save(fig, out)
        outputs.append(str(out))

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_map_overlay(ax, config)
    if inferred:
        values_by_segment = stress_latest
        for feature in inferred:
            props = feature.get("properties", {})
            value = values_by_segment.get(str(props.get("segment_id")), 0.0)
            local = feature.get("geometry", {}).get("local_trace_m")
            if local:
                ax.plot(
                    [p[0] for p in local],
                    [p[1] for p in local],
                    color=plt.cm.magma(value),
                    linewidth=3,
                )
    _plot_fault_lines(ax, known, "#777777", "known faults")
    ax.set_title("Latest fallback stress score with map overlay")
    ax.set_xlabel("Easting in local CRS [m]")
    ax.set_ylabel("Northing in local CRS [m]")
    out = paths.outputs_maps / "stress_map_latest.png"
    _save(fig, out)
    outputs.append(str(out))

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_map_overlay(ax, config)
    scenario_100 = {
        str(row["segment_id"]): float(row["failure_index_p50"])
        for row in scenarios
        if str(row.get("year")) == "100"
    }
    for feature in inferred + known:
        props = feature.get("properties", {})
        value = scenario_100.get(str(props.get("segment_id")), 0.0)
        local = feature.get("geometry", {}).get("local_trace_m")
        if local:
            ax.plot(
                [p[0] for p in local],
                [p[1] for p in local],
                color=plt.cm.plasma(min(value, 1.0)),
                linewidth=3,
            )
        elif props.get("trace_x_m"):
            ax.plot(
                props["trace_x_m"],
                props["trace_y_m"],
                color=plt.cm.plasma(min(value, 1.0)),
                linewidth=3,
            )
    ax.set_title("100-year relative failure index p50 with map overlay")
    ax.set_xlabel("Easting in local CRS [m]")
    ax.set_ylabel("Northing in local CRS [m]")
    out = paths.outputs_maps / "failure_index_100yr_map.png"
    _save(fig, out)
    outputs.append(str(out))
    LOGGER.info("Wrote %d static maps with map overlays", len(outputs))
    return {"map_files": outputs, "map_overlay": "local_crs_bbox_graticule"}
