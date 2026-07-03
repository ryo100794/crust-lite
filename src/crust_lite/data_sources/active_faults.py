from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.geo import LocalProjector, distance_to_polyline_km, polyline_length_km
from crust_lite.io.geopackage import read_features, write_features
from crust_lite.io.parquet import write_table
from crust_lite.paths import ProjectPaths, resolve_input

Feature = dict[str, Any]
LineXY = list[tuple[float, float]]


def read_fault_geojson(path: Path) -> list[Feature]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Fault file must be GeoJSON FeatureCollection: {path}")
    return list(data.get("features", []))


def _read_geopandas_features(path: Path) -> list[Feature]:
    try:
        import geopandas as gpd  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional geospatial stack
        raise ValueError(f"Reading {path.suffix} active-fault data requires geopandas: {exc}") from exc
    gdf = gpd.read_file(path)
    if gdf.empty:
        return []
    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:4326")
    features: list[Feature] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {str(k): _jsonable(v) for k, v in row.items() if k != "geometry"}
        props.setdefault("source_feature_index", int(idx) if isinstance(idx, int) else str(idx))
        features.append({"type": "Feature", "geometry": geom.__geo_interface__, "properties": props})
    return features


def _read_csv_features(path: Path) -> list[Feature]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if not rows:
        return []
    if {"segment_id", "lon", "lat"}.issubset(rows[0]):
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(str(row["segment_id"]), []).append(row)
        features: list[Feature] = []
        for segment_id, group in grouped.items():
            group.sort(key=lambda row: float(row.get("sequence", row.get("point_index", len(group))) or 0.0))
            coords = [[float(row["lon"]), float(row["lat"])] for row in group]
            props = dict(group[0])
            for key in ("lon", "lat", "sequence", "point_index"):
                props.pop(key, None)
            props["segment_id"] = segment_id
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": props,
                }
            )
        return features
    raise ValueError(
        f"CSV active-fault input must contain segment_id, lon, lat columns for trace vertices: {path}"
    )


def read_fault_features(path: Path) -> list[Feature]:
    suffix = path.suffix.lower()
    if suffix in {".geojson", ".json"}:
        return read_fault_geojson(path)
    if suffix == ".csv":
        return _read_csv_features(path)
    if suffix in {".gpkg", ".shp", ".zip"}:
        return _read_geopandas_features(path)
    raise ValueError(f"Unsupported active-fault input format: {path}")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _geometry_lines_lonlat(geom: dict[str, Any]) -> list[list[list[float]]]:
    if geom.get("type") == "LineString":
        return [[list(map(float, coord[:2])) for coord in geom.get("coordinates", [])]]
    if geom.get("type") == "MultiLineString":
        return [
            [list(map(float, coord[:2])) for coord in line]
            for line in geom.get("coordinates", [])
        ]
    return []


def _feature_local_lines(feature: Feature, projector: LocalProjector) -> list[LineXY]:
    geom = feature.get("geometry") or {}
    local = geom.get("local_trace_m")
    if local:
        return [[(float(x), float(y)) for x, y in local]]
    return [projector.line_lonlat_to_xy(line) for line in _geometry_lines_lonlat(geom)]


def _trace_strike_deg(xy: LineXY) -> float | None:
    if len(xy) < 2:
        return None
    x0, y0 = xy[0]
    x1, y1 = xy[-1]
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360.0) % 360.0


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _source_quality(props: dict[str, Any], source_path: Path, is_sample: bool) -> str:
    text = " ".join(
        str(props.get(key, ""))
        for key in ("source", "geometry_accuracy", "notes", "source_agency_reference")
    ).lower()
    path_text = str(source_path).lower()
    if is_sample or "synthetic_sample" in text:
        return "synthetic_sample"
    if "coarse_seed" in text or "coarse_seed" in path_text or "not_official_trace" in text:
        return "coarse_reference_not_official"
    return "user_supplied_data_source"


def _is_accepted_known_fault(config: AppConfig, props: dict[str, Any], source_path: Path, is_sample: bool) -> bool:
    quality = _source_quality(props, source_path, is_sample)
    if quality == "synthetic_sample":
        return is_sample
    if quality == "coarse_reference_not_official":
        return bool(config.known_fault_detail.allow_coarse_reference_seed)
    return True


def _segment_id(props: dict[str, Any], source_path: Path, index: int) -> str:
    for key in ("segment_id", "fault_id", "id", "ID", "code", "name", "Name", "fault_name"):
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{source_path.stem}_{index:06d}"


def _augment_feature(feature: Feature, config: AppConfig, source_path: Path, index: int, is_sample: bool) -> Feature:
    props = dict(feature.get("properties", {}))
    geom = dict(feature.get("geometry") or {})
    projector = LocalProjector(config.region)
    lines = _feature_local_lines({"geometry": geom, "properties": props}, projector)
    xy = max(lines, key=polyline_length_km) if lines else []
    source_quality = _source_quality(props, source_path, is_sample)
    segment_id = _segment_id(props, source_path, index)

    if xy:
        xs, ys = zip(*xy, strict=False)
        props["center_x_m"] = sum(xs) / len(xs)
        props["center_y_m"] = sum(ys) / len(ys)
        props["trace_x_m"] = [x for x, _ in xy]
        props["trace_y_m"] = [y for _, y in xy]
        geom["local_trace_m"] = [[x, y] for x, y in xy]

    if props.get("strike") in (None, ""):
        strike = _trace_strike_deg(xy)
        if strike is not None:
            props["strike"] = strike
            props["strike_source"] = "source_geometry_derived"
    if props.get("length_km") in (None, ""):
        props["length_km"] = polyline_length_km(xy)
        props["length_source"] = "source_geometry_derived"

    props["segment_id"] = segment_id
    props.setdefault("source", source_path.stem if not is_sample else "synthetic_sample")
    props.setdefault("fault_type", props.get("type", "unknown"))
    props["is_inferred"] = False
    props["is_sample_data"] = is_sample
    props["source_path"] = str(source_path)
    props["source_feature_index"] = props.get("source_feature_index", index)
    props["source_quality"] = source_quality
    props["known_fault_data_policy"] = (
        "accepted_source_feature"
        if _is_accepted_known_fault(config, props, source_path, is_sample)
        else "reference_only_not_known_fault"
    )

    # Do not invent scientific geometry attributes. Missing dip/rake/depth/width
    # remain null; model stages may use explicit numerical fallbacks internally.
    for key in ("dip", "rake", "width_km", "top_depth_km", "bottom_depth_km", "confidence"):
        props[key] = _coerce_float(props.get(key))
    props.setdefault("notes", "")
    return {"type": "Feature", "geometry": geom, "properties": props}


def _line_point_at_distance(line: LineXY, distance_m: float) -> tuple[float, float]:
    if not line:
        return (0.0, 0.0)
    if distance_m <= 0.0 or len(line) == 1:
        return line[0]
    travelled = 0.0
    for a, b in zip(line[:-1], line[1:], strict=False):
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if travelled + seg_len >= distance_m:
            t = (distance_m - travelled) / max(seg_len, 1.0e-12)
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        travelled += seg_len
    return line[-1]


def _sample_distances(length_m: float, step_m: float) -> list[float]:
    if length_m <= 0.0:
        return [0.0]
    count = max(1, int(math.floor(length_m / step_m)))
    values = [idx * step_m for idx in range(count + 1)]
    if values[-1] < length_m:
        values.append(length_m)
    return values


def build_known_fault_reference_layers(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    if not config.known_fault_detail.enabled:
        return {"skipped": True, "reason": "known_fault_detail.enabled=false"}
    fault_path = paths.data_processed / "fault_segment.gpkg"
    features = read_features(fault_path) if fault_path.exists() else []
    projector = LocalProjector(config.region)
    trace_rows: list[dict[str, Any]] = []
    subsegment_rows: list[dict[str, Any]] = []
    point_limit = int(config.known_fault_detail.max_trace_points)
    trace_step_m = float(config.known_fault_detail.trace_sample_spacing_km) * 1000.0
    sub_step_m = float(config.known_fault_detail.subsegment_length_km) * 1000.0

    for feature in features:
        props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
        segment_id = str(props.get("segment_id", "unknown"))
        for line_index, line in enumerate(_feature_local_lines(feature, projector)):
            length_m = polyline_length_km(line) * 1000.0
            for point_index, distance_m in enumerate(_sample_distances(length_m, trace_step_m)):
                if len(trace_rows) >= point_limit:
                    break
                x_m, y_m = _line_point_at_distance(line, distance_m)
                trace_rows.append(
                    {
                        "source_segment_id": segment_id,
                        "line_index": line_index,
                        "point_index": point_index,
                        "distance_along_km": distance_m / 1000.0,
                        "x_m": x_m,
                        "y_m": y_m,
                        "source": props.get("source", "unknown"),
                        "source_quality": props.get("source_quality", "unknown"),
                        "source_path": props.get("source_path", ""),
                        "geometry_accuracy": props.get("geometry_accuracy", ""),
                        "derivative_role": "trace_sample_from_source_geometry",
                        "is_known_fault_segment": False,
                        "not_prediction": True,
                    }
                )
            sub_distances = _sample_distances(length_m, sub_step_m)
            for sub_index, (start_m, end_m) in enumerate(zip(sub_distances[:-1], sub_distances[1:], strict=False)):
                start = _line_point_at_distance(line, start_m)
                end = _line_point_at_distance(line, end_m)
                center = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
                strike = _trace_strike_deg([start, end])
                subsegment_rows.append(
                    {
                        "source_segment_id": segment_id,
                        "subsegment_id": f"{segment_id}_srcseg_{line_index:02d}_{sub_index:05d}",
                        "line_index": line_index,
                        "subsegment_index": sub_index,
                        "start_distance_km": start_m / 1000.0,
                        "end_distance_km": end_m / 1000.0,
                        "length_km": math.hypot(end[0] - start[0], end[1] - start[1]) / 1000.0,
                        "x0_m": start[0],
                        "y0_m": start[1],
                        "x1_m": end[0],
                        "y1_m": end[1],
                        "center_x_m": center[0],
                        "center_y_m": center[1],
                        "strike": strike,
                        "source": props.get("source", "unknown"),
                        "source_quality": props.get("source_quality", "unknown"),
                        "source_path": props.get("source_path", ""),
                        "geometry_accuracy": props.get("geometry_accuracy", ""),
                        "derivative_role": "subsegment_from_source_geometry",
                        "is_known_fault_segment": False,
                        "not_prediction": True,
                    }
                )

    comparison_rows = _known_inferred_comparison_rows(config, paths, features, projector)
    metadata = {
        "method": "source_geometry_reference_sampling",
        "input_known_fault_feature_count": len(features),
        "derived_from_source_geometry_only": True,
        "does_not_create_known_faults": True,
        "trace_sample_spacing_km": config.known_fault_detail.trace_sample_spacing_km,
        "subsegment_length_km": config.known_fault_detail.subsegment_length_km,
        "not_prediction": True,
    }
    write_table(
        trace_rows,
        paths.data_processed / "known_fault_trace_point.parquet",
        {**metadata, "table_role": "trace_points"},
    )
    write_table(
        subsegment_rows,
        paths.data_processed / "known_fault_subsegment.parquet",
        {**metadata, "table_role": "subsegments"},
    )
    write_table(
        comparison_rows,
        paths.data_processed / "known_fault_inferred_comparison.parquet",
        {**metadata, "table_role": "comparison"},
    )
    return {
        "known_fault_count": len(features),
        "trace_point_count": len(trace_rows),
        "subsegment_count": len(subsegment_rows),
        "comparison_count": len(comparison_rows),
    }


def _feature_center_xy(feature: Feature, projector: LocalProjector) -> tuple[float, float]:
    props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
    if props.get("center_x_m") not in (None, "") and props.get("center_y_m") not in (None, ""):
        return float(props["center_x_m"]), float(props["center_y_m"])
    lines = _feature_local_lines(feature, projector)
    points = [point for line in lines for point in line]
    if not points:
        return (0.0, 0.0)
    return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))


def _known_inferred_comparison_rows(
    config: AppConfig,
    paths: ProjectPaths,
    known_features: list[Feature],
    projector: LocalProjector,
) -> list[dict[str, Any]]:
    inferred_path = paths.data_processed / "inferred_faults.gpkg"
    if not known_features or not inferred_path.exists():
        return []
    inferred = read_features(inferred_path)
    known_lines = []
    for feature in known_features:
        props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
        lines = _feature_local_lines(feature, projector)
        if lines:
            known_lines.append((str(props.get("segment_id", "unknown")), props, lines))
    rows: list[dict[str, Any]] = []
    max_distance = float(config.known_fault_detail.comparison_max_distance_km)
    for feature in inferred:
        props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
        center = _feature_center_xy(feature, projector)
        best = (float("inf"), "", {})
        for segment_id, known_props, lines in known_lines:
            distance = min(distance_to_polyline_km(center, line) for line in lines if len(line) >= 2)
            if distance < best[0]:
                best = (distance, segment_id, known_props)
        if math.isfinite(best[0]) and best[0] <= max_distance:
            rows.append(
                {
                    "inferred_segment_id": props.get("segment_id", ""),
                    "known_segment_id": best[1],
                    "nearest_distance_km": best[0],
                    "known_source": best[2].get("source", "unknown"),
                    "known_source_quality": best[2].get("source_quality", "unknown"),
                    "inferred_fault_score": props.get("fault_score", None),
                    "comparison_role": "nearest_inferred_candidate_to_source_known_fault_geometry",
                    "not_prediction": True,
                }
            )
    return rows


def fetch_active_faults(
    config: AppConfig,
    paths: ProjectPaths,
    sample: bool = False,
) -> dict[str, Any]:
    paths.ensure()
    empty_meta = {"is_sample_data": False, "not_prediction": True, "does_not_create_known_faults": True}
    if not config.data_sources.use_active_faults and not sample:
        write_features([], paths.data_processed / "fault_segment.gpkg", empty_meta)
        detail = build_known_fault_reference_layers(config, paths)
        return {"is_sample_data": False, "fault_count": 0, "reference_layers": detail}

    fallback = paths.data_raw / "sample" / "sample_known_faults.geojson"
    if sample:
        source_path = fallback
    elif not config.data_sources.active_fault_file:
        write_features(
            [],
            paths.data_processed / "fault_segment.gpkg",
            {**empty_meta, "reason": "active_fault_file_not_configured"},
        )
        detail = build_known_fault_reference_layers(config, paths)
        return {
            "is_sample_data": False,
            "fault_count": 0,
            "reference_only_count": 0,
            "reason": "active_fault_file_not_configured",
            "reference_layers": detail,
        }
    else:
        source_path = resolve_input(paths.root, config.data_sources.active_fault_file, fallback)
        if not source_path.exists():
            write_features(
                [],
                paths.data_processed / "fault_segment.gpkg",
                {**empty_meta, "reason": f"active_fault_file_missing:{source_path}"},
            )
            detail = build_known_fault_reference_layers(config, paths)
            return {
                "is_sample_data": False,
                "fault_count": 0,
                "reference_only_count": 0,
                "reason": f"active_fault_file_missing:{source_path}",
                "reference_layers": detail,
            }
    raw_features = read_fault_features(source_path)
    augmented = [
        _augment_feature(feature, config, source_path, idx, source_path == fallback)
        for idx, feature in enumerate(raw_features)
    ]
    accepted = [
        feature
        for feature in augmented
        if _is_accepted_known_fault(config, feature.get("properties", {}), source_path, source_path == fallback)
    ]
    reference_only = [feature for feature in augmented if feature not in accepted]

    metadata = {
        "is_sample_data": source_path == fallback,
        "source_path": str(source_path),
        "raw_feature_count": len(raw_features),
        "accepted_known_fault_count": len(accepted),
        "reference_only_feature_count": len(reference_only),
        "does_not_create_known_faults": True,
        "coarse_reference_seed_allowed": config.known_fault_detail.allow_coarse_reference_seed,
        "not_prediction": True,
    }
    write_features(accepted, paths.data_processed / "fault_segment.gpkg", metadata)
    if reference_only:
        write_features(
            reference_only,
            paths.data_processed / "reference_fault_segment.gpkg",
            {**metadata, "table_role": "reference_only_not_known_fault"},
        )
    detail = build_known_fault_reference_layers(config, paths)
    return {
        "is_sample_data": source_path == fallback,
        "fault_count": len(accepted),
        "reference_only_count": len(reference_only),
        "source_path": str(source_path),
        "reference_layers": detail,
    }
