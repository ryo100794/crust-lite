from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Literal, TypedDict


class TectonicLine(TypedDict):
    name: str
    plate: str
    kind: str
    color: tuple[float, float, float, float]
    coordinates: list[tuple[float, float, float]]


class TectonicContext(TypedDict):
    boundaries: list[TectonicLine]
    interfaces: list[TectonicLine]
    source: str
    note: str
    literature_based: bool
    model_source: str
    source_files: list[str]
    fallback_used: bool
    default_show: bool


BoundaryKind = Literal["trench", "trough", "plate_interface"]


PACIFIC_COLOR = (0.10, 0.70, 1.00, 0.58)
PHILIPPINE_SEA_COLOR = (0.95, 0.82, 0.20, 0.58)
BOUNDARY_COLOR = (0.35, 0.95, 1.00, 0.86)
TROUGH_COLOR = (1.00, 0.82, 0.25, 0.86)
MODEL_BOUNDARY_COLOR = (0.32, 0.86, 1.00, 0.90)
MODEL_INTERFACE_COLOR = (1.00, 0.74, 0.26, 0.72)


JAPAN_TRENCH = [
    (145.4, 44.4),
    (144.8, 43.2),
    (144.2, 42.0),
    (143.9, 40.5),
    (143.5, 39.0),
    (143.0, 37.5),
    (142.3, 36.0),
    (141.6, 34.7),
]


IZU_BONIN_TRENCH = [
    (141.6, 34.7),
    (142.0, 33.0),
    (142.4, 31.2),
    (142.8, 29.5),
    (143.3, 27.7),
    (143.9, 25.8),
]


NANKAI_TROUGH = [
    (131.2, 31.8),
    (132.6, 32.3),
    (134.1, 32.8),
    (135.6, 33.1),
    (137.1, 33.5),
    (138.8, 34.2),
    (140.0, 34.9),
]


RYUKYU_TRENCH = [
    (122.8, 23.8),
    (124.7, 24.4),
    (126.5, 25.3),
    (128.3, 26.4),
    (130.0, 28.1),
    (131.2, 31.8),
]


def _line_with_depth(coords: list[tuple[float, float]], depth_km: float = 0.0) -> list[tuple[float, float, float]]:
    return [(lon, lat, depth_km) for lon, lat in coords]


def _offset_lonlat(lon: float, lat: float, east_km: float, north_km: float) -> tuple[float, float]:
    cos_lat = max(0.1, math.cos(math.radians(lat)))
    return lon + east_km / (111.32 * cos_lat), lat + north_km / 110.57


def _densify_polyline(coords: list[tuple[float, float]], samples_per_segment: int) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    dense: list[tuple[float, float]] = []
    for start, end in zip(coords, coords[1:], strict=False):
        lon1, lat1 = start
        lon2, lat2 = end
        for step in range(samples_per_segment):
            f = step / samples_per_segment
            dense.append((lon1 + (lon2 - lon1) * f, lat1 + (lat2 - lat1) * f))
    dense.append(coords[-1])
    return dense


def _surface_wire_from_trench(
    name: str,
    plate: str,
    trench: list[tuple[float, float]],
    color: tuple[float, float, float, float],
    *,
    dip_east_km_per_step: float,
    dip_north_km_per_step: float,
    depth_start_km: float,
    depth_step_km: float,
    down_dip_steps: int,
    along_samples_per_segment: int,
) -> list[TectonicLine]:
    along = _densify_polyline(trench, along_samples_per_segment)
    grid: list[list[tuple[float, float, float]]] = []
    for i in range(down_dip_steps + 1):
        row: list[tuple[float, float, float]] = []
        for lon, lat in along:
            xlon, xlat = _offset_lonlat(lon, lat, dip_east_km_per_step * i, dip_north_km_per_step * i)
            row.append((xlon, xlat, depth_start_km + depth_step_km * i))
        grid.append(row)

    lines: list[TectonicLine] = []
    for i, row in enumerate(grid):
        lines.append(
            {
                "name": f"{name} depth-contour-{i:02d}",
                "plate": plate,
                "kind": "plate_interface",
                "color": color,
                "coordinates": row,
            }
        )
    column_step = max(1, len(along) // 18)
    for j in range(0, len(along), column_step):
        lines.append(
            {
                "name": f"{name} downdip-{j:03d}",
                "plate": plate,
                "kind": "plate_interface",
                "color": color,
                "coordinates": [grid[i][j] for i in range(len(grid))],
            }
        )
    return lines


def _with_metadata(context: dict[str, Any], *, default_show: bool) -> TectonicContext:
    return {
        "boundaries": context["boundaries"],
        "interfaces": context["interfaces"],
        "source": context["source"],
        "note": context["note"],
        "literature_based": bool(context.get("literature_based", False)),
        "model_source": str(context.get("model_source", context["source"])),
        "source_files": [str(path) for path in context.get("source_files", [])],
        "fallback_used": bool(context.get("fallback_used", False)),
        "default_show": bool(default_show),
    }


def japan_tectonic_context() -> TectonicContext:
    """Return a lightweight Japan plate-tectonic context for visualization.

    This is a schematic overlay for orientation. It is not a Slab2/J-SHIS/GSI
    analytical plate model and must not be used as a quantitative interface.
    """
    boundaries: list[TectonicLine] = [
        {
            "name": "Japan Trench",
            "plate": "Pacific plate subduction boundary",
            "kind": "trench",
            "color": BOUNDARY_COLOR,
            "coordinates": _line_with_depth(JAPAN_TRENCH),
        },
        {
            "name": "Izu-Bonin Trench",
            "plate": "Pacific plate subduction boundary",
            "kind": "trench",
            "color": BOUNDARY_COLOR,
            "coordinates": _line_with_depth(IZU_BONIN_TRENCH),
        },
        {
            "name": "Nankai Trough",
            "plate": "Philippine Sea plate subduction boundary",
            "kind": "trough",
            "color": TROUGH_COLOR,
            "coordinates": _line_with_depth(NANKAI_TROUGH),
        },
        {
            "name": "Ryukyu Trench",
            "plate": "Philippine Sea plate subduction boundary",
            "kind": "trench",
            "color": TROUGH_COLOR,
            "coordinates": _line_with_depth(RYUKYU_TRENCH),
        },
    ]
    interfaces = []
    interfaces.extend(
        _surface_wire_from_trench(
            "Pacific slab schematic",
            "Pacific plate",
            [*JAPAN_TRENCH, *IZU_BONIN_TRENCH[1:]],
            PACIFIC_COLOR,
            dip_east_km_per_step=-45.0,
            dip_north_km_per_step=4.0,
            depth_start_km=12.0,
            depth_step_km=18.0,
            down_dip_steps=13,
            along_samples_per_segment=5,
        )
    )
    interfaces.extend(
        _surface_wire_from_trench(
            "Philippine Sea slab schematic",
            "Philippine Sea plate",
            [*RYUKYU_TRENCH, *NANKAI_TROUGH[1:]],
            PHILIPPINE_SEA_COLOR,
            dip_east_km_per_step=-23.0,
            dip_north_km_per_step=18.0,
            depth_start_km=8.0,
            depth_step_km=9.0,
            down_dip_steps=10,
            along_samples_per_segment=5,
        )
    )
    return _with_metadata(
        {
            "boundaries": boundaries,
            "interfaces": interfaces,
            "source": "schematic_japan_plate_context_v0",
            "note": "Schematic plate boundaries and slab-interface wireframes for visual context only; not a quantitative plate model.",
            "literature_based": False,
            "model_source": "schematic_japan_plate_context_v0",
            "source_files": [],
            "fallback_used": True,
        },
        default_show=False,
    )


def _empty_context(source: str, source_files: list[str], note: str, *, default_show: bool) -> TectonicContext:
    return _with_metadata(
        {
            "boundaries": [],
            "interfaces": [],
            "source": source,
            "note": note,
            "literature_based": False,
            "model_source": source,
            "source_files": source_files,
            "fallback_used": False,
        },
        default_show=default_show,
    )


def _resolve_existing(path_value: str | None, config_path: Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    if config_path is not None and not path.is_absolute():
        candidates.append(config_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _string_prop(properties: dict[str, Any], names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = properties.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _color_for(kind: str, plate: str, *, interface: bool) -> tuple[float, float, float, float]:
    text = f"{kind} {plate}".lower()
    if "philippine" in text or "nankai" in text or "ryukyu" in text:
        return PHILIPPINE_SEA_COLOR if interface else TROUGH_COLOR
    if "pacific" in text or "japan" in text or "kuril" in text or "izu" in text:
        return PACIFIC_COLOR if interface else BOUNDARY_COLOR
    return MODEL_INTERFACE_COLOR if interface else MODEL_BOUNDARY_COLOR


def _geojson_positions(coords: Any) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    if not isinstance(coords, list):
        return positions
    for item in coords:
        if not isinstance(item, list | tuple) or len(item) < 2:
            continue
        lon = _float_or_none(item[0])
        lat = _float_or_none(item[1])
        depth = _float_or_none(item[2]) if len(item) >= 3 else 0.0
        if lon is None or lat is None:
            continue
        positions.append((lon, lat, depth or 0.0))
    return positions


def _load_boundary_geojson(path: Path) -> list[TectonicLine]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", []) if isinstance(data, dict) and data.get("type") == "FeatureCollection" else [data]
    lines: list[TectonicLine] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", feature)
        properties = feature.get("properties", {}) if isinstance(feature.get("properties", {}), dict) else {}
        if not isinstance(geometry, dict):
            continue
        geom_type = geometry.get("type")
        coord_groups = []
        if geom_type == "LineString":
            coord_groups = [geometry.get("coordinates", [])]
        elif geom_type == "MultiLineString":
            coord_groups = geometry.get("coordinates", [])
        else:
            continue
        plate = _string_prop(properties, ("plate", "plate_name", "subducting_plate", "upper_plate"), "unknown")
        kind = _string_prop(properties, ("kind", "type", "boundary_type"), "plate_boundary")
        name = _string_prop(properties, ("name", "Name", "title", "id"), f"boundary-{index:04d}")
        for part, coords in enumerate(coord_groups):
            positions = _geojson_positions(coords)
            if len(positions) < 2:
                continue
            part_name = name if len(coord_groups) == 1 else f"{name} part-{part + 1}"
            lines.append(
                {
                    "name": part_name,
                    "plate": plate,
                    "kind": kind,
                    "color": _color_for(kind, plate, interface=False),
                    "coordinates": positions,
                }
            )
    return lines


def _field(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    lowered = {key.lower().strip(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _load_slab_csv(path: Path, *, default_kind: str) -> list[TectonicLine]:
    groups: dict[tuple[str, str, str], list[tuple[int, float, float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader):
            lon = _float_or_none(_field(row, ("lon", "longitude", "x")))
            lat = _float_or_none(_field(row, ("lat", "latitude", "y")))
            depth_km = _float_or_none(_field(row, ("depth_km", "depth", "z_km", "interface_depth_km")))
            if lon is None or lat is None:
                continue
            plate = _field(row, ("plate", "plate_name", "subducting_plate")) or "unknown"
            kind = _field(row, ("kind", "type", "contour", "class")) or default_kind
            name = _field(row, ("name", "line", "line_id", "contour_name", "segment"))
            if name is None:
                depth_label = "surface" if depth_km is None else f"{depth_km:g}km"
                name = f"{plate} {kind} {depth_label}"
            groups.setdefault((name, plate, kind), []).append((row_number, lon, lat, depth_km or 0.0))
    lines: list[TectonicLine] = []
    for (name, plate, kind), rows in groups.items():
        rows.sort(key=lambda item: item[0])
        coords = [(lon, lat, depth_km) for _, lon, lat, depth_km in rows]
        if len(coords) < 2:
            continue
        lines.append(
            {
                "name": name,
                "plate": plate,
                "kind": kind,
                "color": _color_for(kind, plate, interface=True),
                "coordinates": coords,
            }
        )
    return lines


def tectonic_context_from_config(config: Any) -> TectonicContext:
    model = config.tectonic_model
    source = str(model.source)
    requested_files = [
        value
        for value in (model.boundary_geojson, model.slab_depth_grid_csv, model.slab_contour_csv)
        if value
    ]
    default_show = bool(model.default_show)
    if not bool(model.enabled):
        return _empty_context(source, [str(value) for value in requested_files], "Tectonic model overlay disabled by config.", default_show=False)

    config_path = getattr(config, "path", None)
    boundary_path = _resolve_existing(model.boundary_geojson, config_path)
    depth_grid_path = _resolve_existing(model.slab_depth_grid_csv, config_path)
    contour_path = _resolve_existing(model.slab_contour_csv, config_path)

    boundaries: list[TectonicLine] = []
    interfaces: list[TectonicLine] = []
    source_files: list[str] = []
    if boundary_path is not None:
        boundaries.extend(_load_boundary_geojson(boundary_path))
        source_files.append(str(boundary_path))
    if depth_grid_path is not None:
        interfaces.extend(_load_slab_csv(depth_grid_path, default_kind="slab_depth_grid"))
        source_files.append(str(depth_grid_path))
    if contour_path is not None:
        interfaces.extend(_load_slab_csv(contour_path, default_kind="slab_contour"))
        source_files.append(str(contour_path))

    if boundaries or interfaces:
        return _with_metadata(
            {
                "boundaries": boundaries,
                "interfaces": interfaces,
                "source": source,
                "note": "Plate model loaded from local configured files. Validate the source, license, and preprocessing before analytical comparison.",
                "literature_based": True,
                "model_source": source,
                "source_files": source_files,
                "fallback_used": False,
            },
            default_show=default_show,
        )

    if bool(model.fallback_to_schematic):
        context = japan_tectonic_context()
        return _with_metadata(
            {
                "boundaries": context["boundaries"],
                "interfaces": context["interfaces"],
                "source": "schematic_japan_plate_context_v0",
                "note": "Configured plate-model files were missing or empty; using schematic context for orientation only.",
                "literature_based": False,
                "model_source": source,
                "source_files": [str(value) for value in requested_files],
                "fallback_used": True,
            },
            default_show=False,
        )

    return _empty_context(
        source,
        [str(value) for value in requested_files],
        "Configured plate-model files were missing or empty and schematic fallback is disabled.",
        default_show=False,
    )
