from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, polyline_length_km
from crust_lite.io.geopackage import write_features
from crust_lite.paths import ProjectPaths, resolve_input


def read_fault_geojson(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Fault file must be GeoJSON FeatureCollection: {path}")
    return list(data.get("features", []))


def _trace_strike_deg(xy: list[tuple[float, float]]) -> float:
    if len(xy) < 2:
        return 0.0
    x0, y0 = xy[0]
    x1, y1 = xy[-1]
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def _augment_feature(feature: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    props = dict(feature.get("properties", {}))
    geom = dict(feature.get("geometry") or {})
    projector = LocalProjector(config.region)
    coords = geom.get("coordinates") or []
    xy = projector.line_lonlat_to_xy(coords) if geom.get("type") == "LineString" else []
    if xy:
        xs, ys = zip(*xy, strict=False)
        props["center_x_m"] = sum(xs) / len(xs)
        props["center_y_m"] = sum(ys) / len(ys)
        props["trace_x_m"] = [x for x, _ in xy]
        props["trace_y_m"] = [y for _, y in xy]
        geom["local_trace_m"] = [[x, y] for x, y in xy]
    props.setdefault("segment_id", f"known_fault_{abs(hash(json.dumps(geom))) % 1_000_000}")
    props.setdefault("source", "local_geojson")
    props.setdefault("fault_type", "unknown")
    props.setdefault("strike", _trace_strike_deg(xy))
    props.setdefault("dip", 70.0)
    props.setdefault("rake", 0.0)
    props.setdefault("length_km", polyline_length_km(xy))
    props.setdefault("width_km", 10.0)
    props.setdefault("top_depth_km", 0.0)
    props.setdefault("bottom_depth_km", float(props.get("width_km", 10.0)))
    props.setdefault("is_inferred", False)
    props.setdefault("confidence", 0.5)
    props.setdefault("notes", "")
    props["is_sample_data"] = props.get("source") == "synthetic_sample"
    return {"type": "Feature", "geometry": geom, "properties": props}


def fetch_active_faults(
    config: AppConfig,
    paths: ProjectPaths,
    sample: bool = False,
) -> dict[str, Any]:
    paths.ensure()
    if not config.data_sources.use_active_faults and not sample:
        write_features([], paths.data_processed / "fault_segment.gpkg", {"is_sample_data": False})
        return {"is_sample_data": False, "fault_count": 0}
    fallback = paths.data_raw / "sample" / "sample_known_faults.geojson"
    source_path = resolve_input(paths.root, config.data_sources.active_fault_file, fallback)
    if sample:
        source_path = fallback
    features = [_augment_feature(feature, config) for feature in read_fault_geojson(source_path)]
    write_features(
        features,
        paths.data_processed / "fault_segment.gpkg",
        {"is_sample_data": source_path == fallback, "source_path": str(source_path)},
    )
    return {"is_sample_data": source_path == fallback, "fault_count": len(features)}
