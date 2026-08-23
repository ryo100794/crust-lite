#!/usr/bin/env python3
"""Build one corrected event-phase across tiles without a dense event cube."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import national_depth_consumer_guard_v1144 as depth_guard


R = 6_378_137.0
SIGNALS = ("full", "view_a", "view_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    if not result:
        raise ValueError(value)
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_npz(path: Path, compressed: bool = True, **value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    writer = np.savez_compressed if compressed else np.savez
    writer(temporary, **value)
    temporary.replace(path)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(path.parent))
    specification.loader.exec_module(module)
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
    raise RuntimeError(f"tile {specification.get('id')} lacks a core definition")


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, np.float64)
    median = float(np.median(values))
    scale = max(
        float(np.median(np.abs(values - median))) * 1.4826,
        float(np.quantile(np.abs(values), 0.25)) * 0.10,
        1.0e-8,
    )
    return median, scale


def normalized(values: np.ndarray, xyz: np.ndarray, stats: dict[float, tuple[float, float]]) -> np.ndarray:
    target = np.zeros(len(values), np.float32)
    for depth, (median, scale) in stats.items():
        selected = xyz[:, 2] == depth
        target[selected] = np.clip(
            (np.asarray(values[selected], np.float64) - median) / scale, -4.0, 4.0
        )
    return np.maximum(target, 0.0)


def decode_parameters(row: dict) -> dict:
    return {
        "usable": bool(row["usable"]),
        "valid_core_nodes": int(row["valid_core_nodes"]),
        "global": tuple(map(float, row["global"])),
        "groups": {int(code): tuple(map(float, values)) for code, values in row["groups"].items()},
        "differential_time_groups": int(row["differential_time_groups"]),
        "groups_with_robust_local_stats": int(row["groups_with_robust_local_stats"]),
        "group_count_min_median_max": row["group_count_min_median_max"],
    }


def exact_p95(values: np.ndarray, illumination: np.ndarray) -> tuple[float, int]:
    active = (values > 1.0e-15) & (illumination > 0.05)
    selected = np.asarray(values)[active]
    return (float(np.quantile(selected, 0.95)) if len(selected) else 0.0, int(active.sum()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--phase", choices=("P", "S"), required=True)
    parser.add_argument("--tile", action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()

    depth_guard.enforce_consumer(args.project.resolve(), "response-chunks", args.contract)

    started = time.monotonic()

    def progress(stage: str, completed: int, total: int, current_tile: str | None = None) -> None:
        if args.progress is None:
            return
        atomic_json(args.progress, {
            "schema": "streamed-event-phase-progress-v517",
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_id": str(args.event_id),
            "phase": str(args.phase),
            "stage": stage,
            "completed_units": int(completed),
            "total_units": int(total),
            "percent": round(100.0 * completed / max(total, 1), 3),
            "current_tile": current_tile,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "publication_allowed": False,
        })

    progress("loading_and_provenance", 0, 100)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        contract.get("known_structure_used") is not False
        or parameters.get("known_structure_used") is not False
        or parameters.get("pass") is not True
        or protocol.get("known_structure_used") is not False
        or protocol.get("selected_statistic") != "median"
        or protocol.get("parameter_retuned_after_sanriku") is not False
        or protocol.get("shell_amplification_allowed") is not False
        or protocol.get("near_source_fraction_may_increase") is not False
    ):
        raise RuntimeError("passing fixed-median non-geological provenance required")
    event_id = str(args.event_id)
    phase = str(args.phase)
    if str(parameters.get("event_id")) != event_id or parameters.get("phase") != phase:
        raise RuntimeError("parameter event or phase differs")
    if parameters.get("contract_sha256") != sha256(args.contract):
        raise RuntimeError("parameter contract hash differs")

    tile_paths = parse_tiles(args.tile)
    ordered_ids = [tile["id"] for tile in contract["tiles"]]
    specifications = {tile["id"]: tile for tile in contract["tiles"]}
    if set(tile_paths) != set(ordered_ids):
        raise RuntimeError("tile IDs differ from contract")
    migration = load_module("production_event_phase_migration_v517", args.migration.resolve())
    stream = load_module("production_event_phase_stream_v517", args.stream.resolve())

    features = pd.read_parquet(args.features)
    features["event_id"] = features.event_id.astype(str)
    events = pd.read_parquet(args.events)
    events["event_id"] = events.event_id.astype(str)
    event_rows = events[events.event_id == event_id].drop_duplicates("event_id")
    if len(event_rows) != 1:
        raise RuntimeError(f"event row count is {len(event_rows)}")
    event = event_rows.iloc[0].to_dict()
    rows_event = features[features.event_id == event_id]
    s_picks = (
        rows_event[rows_event.phase_family == "S"]
        .groupby("base_station_id")
        .picked_arrival_s.median().astype(float).to_dict()
    )
    rows = rows_event[
        (rows_event.phase_family == phase) & (rows_event.frequency_hz == 1.0)
    ].drop_duplicates(["raw_path", "channel", "base_station_id"])
    components = migration.load_components(args.project.resolve(), rows, phase, s_picks)
    stations = migration.balanced_station_subset(migration.group_stations(components), 16)
    parameter_rows = {
        str(row["station_id"]): decode_parameters(row)
        for row in parameters["station_parameters"]
    }
    station_ids = [str(station["id"]) for station in stations]
    if station_ids != list(parameter_rows):
        raise RuntimeError("station selection or ordering changed")

    tile_data = {}
    for position, tile_id in enumerate(ordered_ids):
        frame = pd.read_parquet(tile_paths[tile_id], columns=["x_m", "y_m", "z_km"])
        xyz = frame[["x_m", "y_m", "z_km"]].to_numpy(np.float64)
        core = core_mask(specifications[tile_id], xyz, position, len(ordered_ids))
        tile_data[tile_id] = {"xyz": xyz, "core": core}
    core_nodes = sum(int(value["core"].sum()) for value in tile_data.values())
    if core_nodes != int(parameters.get("core_nodes", -1)):
        raise RuntimeError("core node count differs from station parameter manifest")
    depths = np.asarray(contract["required_depths_km"], float)
    scratch_parent = args.scratch.resolve() if args.scratch else args.manifest.parent.resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    progress("station_migration", 8, 100)

    with tempfile.TemporaryDirectory(prefix=f"event-phase-v517-{safe_name(event_id)}-{phase}-", dir=scratch_parent) as temporary_name:
        temporary = Path(temporary_name)
        depth_codes = np.memmap(temporary / "depth.f64", np.float64, "w+", shape=(core_nodes,))
        depth_values = np.memmap(temporary / "depth_values.f64", np.float64, "w+", shape=(3, core_nodes))
        core_offset = 0
        view_a_ids, view_b_ids = stream.base.split_views(stations)
        for tile_index, tile_id in enumerate(ordered_ids):
            xyz = tile_data[tile_id]["xyz"]
            core = tile_data[tile_id]["core"]
            station_values = []
            station_evidence = []
            valid_arrays = []
            for station in stations:
                combined, amplitude_ratio, differential_time, valid = stream.station_intermediate(
                    migration, event, phase, station, xyz
                )
                station_values.append(combined)
                station_evidence.append(stream.apply_isochron_parameters(
                    amplitude_ratio, differential_time, valid, parameter_rows[str(station["id"])]
                ))
                valid_arrays.append(valid)
            values = np.asarray(station_values)
            evidence = np.asarray(station_evidence)
            valid = np.asarray(valid_arrays)
            full, _, _ = stream.previous.aggregate_views(
                migration, values, evidence, valid, stations, None, phase
            )
            score_a, coherence_a, valid_a = stream.previous.aggregate_views(
                migration, values, evidence, valid, stations, view_a_ids, phase
            )
            score_b, coherence_b, valid_b = stream.previous.aggregate_views(
                migration, values, evidence, valid, stations, view_b_ids, phase
            )
            raw_path = temporary / f"raw_{safe_name(tile_id)}.npz"
            atomic_npz(
                raw_path,
                compressed=False,
                full=full,
                view_a=score_a,
                view_b=score_b,
                coherence_a=coherence_a,
                coherence_b=coherence_b,
                valid_a=valid_a,
                valid_b=valid_b,
            )
            size = int(core.sum())
            depth_codes[core_offset:core_offset + size] = xyz[core, 2]
            for signal_index, signal in enumerate((full, score_a, score_b)):
                depth_values[signal_index, core_offset:core_offset + size] = signal[core]
            core_offset += size
            progress("station_migration", 8 + round(37 * (tile_index + 1) / len(ordered_ids)), 100, tile_id)
        depth_codes.flush()
        depth_values.flush()
        if core_offset != core_nodes:
            raise RuntimeError("core append count differs")

        depth_stats: list[dict[float, tuple[float, float]]] = []
        for signal_index in range(3):
            signal_stats = {}
            for depth in depths:
                selected = np.asarray(depth_codes) == depth
                if not selected.any():
                    raise RuntimeError(f"depth has no core nodes: {depth}")
                signal_stats[float(depth)] = robust_stats(np.asarray(depth_values[signal_index])[selected])
            depth_stats.append(signal_stats)
        progress("shared_depth_normalization", 50, 100)

        source_distance = np.memmap(temporary / "source_distance.f64", np.float64, "w+", shape=(core_nodes,))
        source_response = np.memmap(temporary / "source_response.f64", np.float64, "w+", shape=(core_nodes,))
        source_illumination = np.memmap(temporary / "source_illumination.f64", np.float64, "w+", shape=(core_nodes,))
        core_offset = 0
        for tile_index, tile_id in enumerate(ordered_ids):
            xyz = tile_data[tile_id]["xyz"]
            core = tile_data[tile_id]["core"]
            with np.load(temporary / f"raw_{safe_name(tile_id)}.npz", allow_pickle=False) as raw:
                zf = normalized(raw["full"], xyz, depth_stats[0])
                za = normalized(raw["view_a"], xyz, depth_stats[1])
                zb = normalized(raw["view_b"], xyz, depth_stats[2])
                coherence_a = np.asarray(raw["coherence_a"])
                coherence_b = np.asarray(raw["coherence_b"])
                valid_a = np.asarray(raw["valid_a"])
                valid_b = np.asarray(raw["valid_b"])
            common = np.sqrt(za * zb) * np.sqrt(zf)
            common *= 0.20 + 0.80 * np.minimum(coherence_a, coherence_b)
            common *= np.sqrt(np.minimum(valid_a, valid_b))
            common[(za <= 0) | (zb <= 0) | (zf <= 0)] = 0.0
            value_a = za * (0.20 + 0.80 * coherence_a) * np.sqrt(valid_a)
            value_b = zb * (0.20 + 0.80 * coherence_b) * np.sqrt(valid_b)
            value_a[(za <= 0) | (valid_a <= 0)] = 0.0
            value_b[(zb <= 0) | (valid_b <= 0)] = 0.0
            normalized_path = temporary / f"normalized_{safe_name(tile_id)}.npz"
            atomic_npz(
                normalized_path,
                compressed=False,
                response=np.clip(common, 0.0, 8.0).astype(np.float32),
                illumination=np.sqrt(np.minimum(valid_a, valid_b)).astype(np.float32),
                view_a=np.clip(value_a, 0.0, 8.0).astype(np.float32),
                view_b=np.clip(value_b, 0.0, 8.0).astype(np.float32),
                view_a_illumination=np.asarray(valid_a, np.float32),
                view_b_illumination=np.asarray(valid_b, np.float32),
            )
            distance = np.sqrt(
                ((xyz[:, 0] - float(event["x_m"])) / 1000.0) ** 2
                + ((xyz[:, 1] - float(event["y_m"])) / 1000.0) ** 2
                + (xyz[:, 2] - float(event["depth_km"])) ** 2
            )
            with np.load(normalized_path, allow_pickle=False) as normalized_values:
                size = int(core.sum())
                source_distance[core_offset:core_offset + size] = distance[core]
                source_response[core_offset:core_offset + size] = np.asarray(normalized_values["response"], np.float64)[core]
                source_illumination[core_offset:core_offset + size] = np.asarray(normalized_values["illumination"], np.float64)[core]
            core_offset += size
            progress("source_shell_accumulation", 50 + round(20 * (tile_index + 1) / len(ordered_ids)), 100, tile_id)
        source_distance.flush()
        source_response.flush()
        source_illumination.flush()

        shell_width = float(protocol["shell_width_km"])
        distance_all = np.asarray(source_distance)
        response_all = np.asarray(source_response)
        illumination_all = np.asarray(source_illumination)
        shell = np.floor(distance_all / shell_width).astype(np.int32)
        valid = illumination_all > float(protocol["valid_illumination_threshold"])
        raw_weights = np.where(valid, response_all, 0.0)
        raw_total = float(raw_weights.sum())
        if raw_total <= 0:
            raise RuntimeError("no positive response")
        masses = np.bincount(shell[valid], weights=response_all[valid], minlength=int(shell.max()) + 1)
        positive_masses = masses[masses > 0]
        cap = float(np.quantile(positive_masses, 0.50))
        shell_factors = np.ones_like(masses, np.float64)
        active_shells = masses > 0
        shell_factors[active_shells] = np.minimum(1.0, cap / masses[active_shells])
        factors = shell_factors[shell]
        near = distance_all < float(protocol["near_source_radius_km"])
        before_near = float(raw_weights[near].sum() / raw_total)
        corrected = raw_weights * factors
        near_mass = float(corrected[near].sum())
        far_mass = float(corrected[~near].sum())
        guard = 1.0
        after_pre_guard = near_mass / max(near_mass + far_mass, 1.0e-30)
        if after_pre_guard > before_near + 1.0e-12:
            if before_near <= 0:
                guard = 0.0
            else:
                guard = float(np.clip(
                    before_near * far_mass / (near_mass * (1.0 - before_near)), 0.0, 1.0
                ))
        final_core_factors = factors.copy()
        final_core_factors[near] *= guard
        final_weights = raw_weights * final_core_factors
        after_near = float(final_weights[near].sum() / max(float(final_weights.sum()), 1.0e-30))
        progress("fixed_median_source_correction", 75, 100)

        scale_values = np.memmap(temporary / "scale_values.f32", np.float32, "w+", shape=(3, core_nodes))
        scale_illumination = np.memmap(temporary / "scale_illumination.f32", np.float32, "w+", shape=(3, core_nodes))
        chunk_rows = []
        core_offset = 0
        maximum_chunk_uncompressed_bytes = 0
        for tile_index, tile_id in enumerate(ordered_ids):
            xyz = tile_data[tile_id]["xyz"]
            core = tile_data[tile_id]["core"]
            tile_distance = np.sqrt(
                ((xyz[:, 0] - float(event["x_m"])) / 1000.0) ** 2
                + ((xyz[:, 1] - float(event["y_m"])) / 1000.0) ** 2
                + (xyz[:, 2] - float(event["depth_km"])) ** 2
            )
            tile_shell = np.floor(tile_distance / shell_width).astype(np.int32)
            tile_factor = np.ones(len(tile_distance), np.float64)
            represented = tile_shell < len(shell_factors)
            tile_factor[represented] = shell_factors[tile_shell[represented]]
            tile_factor[tile_distance < float(protocol["near_source_radius_km"])] *= guard
            with np.load(temporary / f"normalized_{safe_name(tile_id)}.npz", allow_pickle=False) as raw:
                response = (np.asarray(raw["response"], np.float64) * tile_factor).astype(np.float32)
                view_a = (np.asarray(raw["view_a"], np.float64) * tile_factor).astype(np.float32)
                view_b = (np.asarray(raw["view_b"], np.float64) * tile_factor).astype(np.float32)
                illumination = np.asarray(raw["illumination"], np.float32)
                illumination_a = np.asarray(raw["view_a_illumination"], np.float32)
                illumination_b = np.asarray(raw["view_b_illumination"], np.float32)
            grid_path = args.output_dir / tile_id / "grid_exact_xyz_v517.npz"
            if grid_path.exists():
                with np.load(grid_path, allow_pickle=False) as grid:
                    if not np.array_equal(np.asarray(grid["xyz"]), xyz):
                        raise RuntimeError(f"existing grid differs: {tile_id}")
            else:
                atomic_npz(
                    grid_path,
                    schema=np.asarray("event-phase-chunk-grid-v517"),
                    known_structure_used=np.asarray(False),
                    xyz=xyz,
                )
            chunk_path = args.output_dir / tile_id / phase / f"{safe_name(event_id)}.npz"
            atomic_npz(
                chunk_path,
                schema=np.asarray("event-phase-response-chunk-v492"),
                known_structure_used=np.asarray(False),
                event_id=np.asarray(event_id),
                phase=np.asarray(phase),
                response=response,
                illumination=illumination,
                view_a=view_a,
                view_b=view_b,
                view_a_illumination=illumination_a,
                view_b_illumination=illumination_b,
            )
            arrays = (response, illumination, view_a, view_b, illumination_a, illumination_b)
            maximum_chunk_uncompressed_bytes = max(
                maximum_chunk_uncompressed_bytes, sum(value.nbytes for value in arrays)
            )
            size = int(core.sum())
            for signal_index, (value, light) in enumerate((
                (response, illumination),
                (view_a, illumination_a),
                (view_b, illumination_b),
            )):
                scale_values[signal_index, core_offset:core_offset + size] = value[core]
                scale_illumination[signal_index, core_offset:core_offset + size] = light[core]
            chunk_rows.append({
                "tile_id": tile_id,
                "path": str(chunk_path),
                "sha256": sha256(chunk_path),
                "bytes": chunk_path.stat().st_size,
                "nodes": len(xyz),
                "grid": str(grid_path),
                "grid_sha256": sha256(grid_path),
                "core_nodes": size,
            })
            core_offset += size
            progress("writing_corrected_chunks", 75 + round(20 * (tile_index + 1) / len(ordered_ids)), 100, tile_id)
        scale_values.flush()
        scale_illumination.flush()
        scales = {}
        detected = {}
        for signal_index, name in enumerate(SIGNALS):
            scales[name], detected[name] = exact_p95(
                np.asarray(scale_values[signal_index]),
                np.asarray(scale_illumination[signal_index]),
            )

    checks = {
        "known_structure_absent": True,
        "parameter_manifest_pass": parameters.get("pass") is True,
        "station_selection_and_order_exact": station_ids == list(parameter_rows),
        "all_contract_tiles_written": len(chunk_rows) == len(ordered_ids),
        "core_union_exact": sum(row["core_nodes"] for row in chunk_rows) == core_nodes,
        "coordinate_dtype_float64": all(value["xyz"].dtype == np.float64 for value in tile_data.values()),
        "fixed_median_not_retuned": True,
        "no_shell_amplification": float(shell_factors.max(initial=1.0)) <= 1.0,
        "near_source_fraction_not_increased": after_near <= before_near + 2.0e-12,
        "all_scales_positive": all(value > 0 for value in scales.values()),
        "all_views_detect_nodes": all(value > 0 for value in detected.values()),
        "one_event_phase_chunk_memory_bound": maximum_chunk_uncompressed_bytes == max(row["nodes"] for row in chunk_rows) * 6 * 4,
    }
    manifest = {
        "schema": "streamed-corrected-event-phase-chunk-manifest-v517",
        "known_structure_used": False,
        "event_id": event_id,
        "phase": phase,
        "inputs": {
            "features": str(args.features),
            "events": str(args.events),
            "contract": str(args.contract),
            "parameters": str(args.parameters),
            "protocol": str(args.protocol),
            "migration": str(args.migration),
            "stream": str(args.stream),
        },
        "input_sha256": {
            "features": sha256(args.features),
            "events": sha256(args.events),
            "contract": sha256(args.contract),
            "parameters": sha256(args.parameters),
            "protocol": sha256(args.protocol),
            "migration": sha256(args.migration),
            "stream": sha256(args.stream),
            **{f"tile_{tile_id}": sha256(path) for tile_id, path in tile_paths.items()},
        },
        "tiles": chunk_rows,
        "core_nodes": core_nodes,
        "stations": len(stations),
        "depth_statistics": {
            name: {str(depth): {"median": stats[depth][0], "scale": stats[depth][1]} for depth in stats}
            for name, stats in zip(SIGNALS, depth_stats)
        },
        "source_correction": {
            "statistic": "median",
            "shell_width_km": shell_width,
            "cap_mass": cap,
            "near_source_guard_factor": guard,
            "raw_near_source_fraction": before_near,
            "corrected_near_source_fraction": after_near,
            "positive_shells": int((masses > 0).sum()),
        },
        "event_scale": {**scales, **{f"detected_{key}": value for key, value in detected.items()}},
        "maximum_chunk_uncompressed_bytes": maximum_chunk_uncompressed_bytes,
        "checks": checks,
        "pass": all(checks.values()),
        "publication_allowed": False,
    }
    atomic_json(args.manifest, manifest)
    progress("complete" if manifest["pass"] else "failed_quality_gate", 100, 100)
    print(json.dumps({
        "event_id": event_id,
        "phase": phase,
        "tiles": len(chunk_rows),
        "core_nodes": core_nodes,
        "stations": len(stations),
        "source_correction": manifest["source_correction"],
        "event_scale": manifest["event_scale"],
        "checks": checks,
        "pass": manifest["pass"],
    }, ensure_ascii=False))
    raise SystemExit(0 if manifest["pass"] else 2)


if __name__ == "__main__":
    main()
