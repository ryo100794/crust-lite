#!/usr/bin/env python3
"""Freeze exact shared station isochron parameters by bounded tile streaming."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import national_depth_consumer_guard_v1144 as depth_guard


R = 6_378_137.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def parse_tiles(entries: list[str]) -> dict[str, Path]:
    result = {}
    for entry in entries:
        tile_id, separator, value = entry.partition("=")
        if not separator or not tile_id or not value or tile_id in result:
            raise ValueError(f"invalid or duplicate --tile {entry!r}")
        result[tile_id] = Path(value)
    return result


def mercator_x(lon: float) -> float:
    return R * math.radians(lon)


def core_mask(specification: dict, xyz: np.ndarray, position: int, count: int) -> np.ndarray:
    if "core_lon" in specification:
        west, east = map(float, specification["core_lon"])
        left, right = mercator_x(west), mercator_x(east)
        return (xyz[:, 0] >= left) & (
            (xyz[:, 0] < right) if position < count - 1 else (xyz[:, 0] <= right)
        )
    if "core_xy_bounds_m" in specification:
        west, east, south, north = map(float, specification["core_xy_bounds_m"])
        return (
            (xyz[:, 0] >= west)
            & (xyz[:, 0] <= east)
            & (xyz[:, 1] >= south)
            & (xyz[:, 1] <= north)
        )
    raise RuntimeError(f"tile {specification.get('id')} lacks a supported core definition")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def exact_parameters(values: np.memmap, codes: np.memmap, count: int) -> dict:
    if count < 32:
        return {
            "usable": False,
            "valid_core_nodes": count,
            "global": [0.0, 1.0],
            "groups": {},
            "differential_time_groups": 0,
            "groups_with_robust_local_stats": 0,
            "group_count_min_median_max": [0, 0.0, 0],
        }
    value_view = np.asarray(values[:count])
    code_view = np.asarray(codes[:count])
    global_median = float(np.median(value_view))
    global_scale = max(float(np.median(np.abs(value_view - global_median))) * 1.4826, 1.0e-6)
    order = np.argsort(code_view, kind="stable")
    sorted_codes = code_view[order]
    sorted_values = value_view[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [count]))
    groups = {}
    group_counts = []
    for start, stop in zip(starts, stops):
        size = int(stop - start)
        group_counts.append(size)
        if size < 32:
            continue
        group_values = sorted_values[start:stop]
        median = float(np.median(group_values))
        scale = max(
            float(np.median(np.abs(group_values - median))) * 1.4826,
            0.10 * max(median, 0.0),
            0.25 * global_scale,
            1.0e-6,
        )
        groups[str(int(sorted_codes[start]))] = [median, scale]
    return {
        "usable": True,
        "valid_core_nodes": count,
        "global": [global_median, global_scale],
        "groups": groups,
        "differential_time_groups": len(group_counts),
        "groups_with_robust_local_stats": len(groups),
        "group_count_min_median_max": [
            int(np.min(group_counts)),
            float(np.median(group_counts)),
            int(np.max(group_counts)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--phase", choices=("P", "S"), required=True)
    parser.add_argument("--tile", action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--grid-audit", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()

    depth_guard.enforce_consumer(args.project.resolve(), "parameter-freeze", args.contract)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    grid_audit = json.loads(args.grid_audit.read_text(encoding="utf-8"))
    if contract.get("known_structure_used") is not False or grid_audit.get("pass") is not True:
        raise RuntimeError("a non-geological passing grid contract is required")
    tile_paths = parse_tiles(args.tile)
    specifications = {tile["id"]: tile for tile in contract["tiles"]}
    ordered_ids = [tile["id"] for tile in contract["tiles"]]
    if set(tile_paths) != set(ordered_ids):
        raise RuntimeError("tile IDs differ from contract")

    migration = load_module("streaming_isochron_migration_v472", args.migration.resolve())
    stream = load_module("streaming_isochron_kernel_v472", args.stream.resolve())
    project = args.project.resolve()
    features = pd.read_parquet(args.features)
    features["event_id"] = features.event_id.astype(str)
    events = pd.read_parquet(args.events)
    events["event_id"] = events.event_id.astype(str)
    event_rows = events[events.event_id == str(args.event_id)].drop_duplicates("event_id")
    if len(event_rows) != 1:
        raise RuntimeError(f"event row count is {len(event_rows)}")
    event = event_rows.iloc[0].to_dict()
    rows_event = features[features.event_id == str(args.event_id)]
    s_picks = (
        rows_event[rows_event.phase_family == "S"]
        .groupby("base_station_id")
        .picked_arrival_s.median().astype(float).to_dict()
    )
    rows = rows_event[
        (rows_event.phase_family == args.phase) & (rows_event.frequency_hz == 1.0)
    ].drop_duplicates(["raw_path", "channel", "base_station_id"])
    components = migration.load_components(project, rows, args.phase, s_picks)
    stations = migration.balanced_station_subset(migration.group_stations(components), 16)
    if len(stations) < 4:
        raise RuntimeError(f"insufficient stations: {len(stations)}")

    tile_data = {}
    for position, tile_id in enumerate(ordered_ids):
        frame = pd.read_parquet(tile_paths[tile_id], columns=["x_m", "y_m", "z_km"])
        xyz = frame[["x_m", "y_m", "z_km"]].to_numpy(np.float64)
        core = core_mask(specifications[tile_id], xyz, position, len(ordered_ids))
        tile_data[tile_id] = {"xyz": xyz, "core": core}
    core_nodes = int(sum(int(row["core"].sum()) for row in tile_data.values()))
    if core_nodes < 32:
        raise RuntimeError("empty core union")

    scratch_parent = args.scratch.resolve() if args.scratch else args.output.parent.resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    station_reports = []
    for station_index, station in enumerate(stations):
        atomic_json(args.progress, {
            "schema": "shared-isochron-streaming-progress-v472",
            "state": "running",
            "event_id": str(args.event_id),
            "phase": args.phase,
            "station_current": station_index + 1,
            "stations_total": len(stations),
            "fraction": station_index / len(stations),
            "station_id": str(station["id"]),
            "publication_allowed": False,
        })
        with tempfile.TemporaryDirectory(prefix="isochron-v472-", dir=scratch_parent) as temporary:
            temporary_path = Path(temporary)
            values = np.memmap(temporary_path / "values.f64", dtype=np.float64, mode="w+", shape=(core_nodes,))
            codes = np.memmap(temporary_path / "codes.i32", dtype=np.int32, mode="w+", shape=(core_nodes,))
            count = 0
            for tile_id in ordered_ids:
                xyz = tile_data[tile_id]["xyz"]
                core = tile_data[tile_id]["core"]
                _, amplitude_ratio, differential_time, valid = stream.station_intermediate(
                    migration, event, args.phase, station, xyz
                )
                selected = core & valid & np.isfinite(amplitude_ratio)
                size = int(selected.sum())
                values[count:count + size] = np.log1p(np.maximum(amplitude_ratio[selected], 0.0))
                codes[count:count + size] = np.floor(differential_time[selected]).astype(np.int32)
                count += size
            values.flush()
            codes.flush()
            parameters = exact_parameters(values, codes, count)
        station_reports.append({"station_id": str(station["id"]), **parameters})

    checks = {
        "known_structure_absent": True,
        "grid_audit_pass": grid_audit.get("pass") is True,
        "all_contract_tiles_streamed": len(tile_data) == len(ordered_ids),
        "core_union_nonempty": core_nodes > 0,
        "minimum_station_count": len(stations) >= 4,
        "independent_views_have_support": sum(row["usable"] for row in station_reports) >= 4,
        "all_station_counts_bounded_by_core_union": all(
            0 <= row["valid_core_nodes"] <= core_nodes for row in station_reports
        ),
    }
    payload = {
        "schema": "shared-isochron-parameters-streaming-v472",
        "known_structure_used": False,
        "event_id": str(args.event_id),
        "phase": args.phase,
        "contract": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "grid_audit": str(args.grid_audit),
        "grid_audit_sha256": sha256(args.grid_audit),
        "features": str(args.features),
        "features_sha256": sha256(args.features),
        "events": str(args.events),
        "events_sha256": sha256(args.events),
        "migration_sha256": sha256(args.migration),
        "stream_sha256": sha256(args.stream),
        "tile_sha256": {tile_id: sha256(path) for tile_id, path in tile_paths.items()},
        "tile_nodes": {tile_id: len(tile_data[tile_id]["xyz"]) for tile_id in ordered_ids},
        "core_nodes": core_nodes,
        "stations": len(stations),
        "station_parameters": station_reports,
        "checks": checks,
        "pass": all(checks.values()),
        "publication_allowed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "waveform_io_workers": int(os.environ.get("EQUAKE_WAVEFORM_IO_WORKERS", "8")),
    }
    atomic_json(args.output, payload)
    atomic_json(args.progress, {
        "schema": "shared-isochron-streaming-progress-v472",
        "state": "complete" if payload["pass"] else "failed",
        "event_id": str(args.event_id),
        "phase": args.phase,
        "station_current": len(stations),
        "stations_total": len(stations),
        "fraction": 1.0,
        "output": str(args.output),
        "pass": payload["pass"],
        "publication_allowed": False,
    })
    print(json.dumps({"event_id": args.event_id, "phase": args.phase, "stations": len(stations), "core_nodes": core_nodes, "checks": checks, "pass": payload["pass"]}, ensure_ascii=False))
    raise SystemExit(0 if payload["pass"] else 2)


if __name__ == "__main__":
    main()
