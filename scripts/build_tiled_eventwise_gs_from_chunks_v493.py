#!/usr/bin/env python3
"""Build halo-tile eventwise GS while loading only one event-phase chunk at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import build_eventwise_gs_fusion_deterministic_v433 as reference
import national_depth_consumer_guard_v1144 as depth_guard


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    depth_guard.enforce_consumer(Path(__file__).resolve().parents[1], "tiled-gs-fusion", args.contract)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = json.loads(args.chunks.read_text(encoding="utf-8"))
    scales = json.loads(args.scales.read_text(encoding="utf-8"))
    if (
        contract.get("known_structure_used") is not False
        or manifest.get("known_structure_used") is not False
        or scales.get("known_structure_used") is not False
        or manifest.get("pass") is not True
        or scales.get("pass") is not True
    ):
        raise RuntimeError("passing non-geological chunk and scale manifests required")
    event_ids = np.asarray(manifest["event_ids"]).astype(str)
    if not np.array_equal(event_ids, np.asarray(scales["event_ids"]).astype(str)):
        raise RuntimeError("event IDs differ from scale manifest")
    grid_path = Path(manifest["grid"])
    if sha256(grid_path) != manifest["grid_sha256"]:
        raise RuntimeError("grid hash differs")
    with np.load(grid_path, allow_pickle=False) as grid:
        if bool(np.asarray(grid["known_structure_used"]).item()):
            raise RuntimeError("grid provenance invalid")
        xyz = np.asarray(grid["xyz"], np.float64)
    sources = reference.event_sources(args.events, event_ids.tolist())
    dx = np.unique(np.round(np.diff(np.unique(xyz[:, 0])), 6))
    dy = np.unique(np.round(np.diff(np.unique(xyz[:, 1])), 6))
    horizontal_spacing_m = float(min(dx.min(), dy.min()))
    output = {
        "schema": np.asarray("tiled-eventwise-gs-independent-v456"),
        "known_structure_used": np.asarray(False),
        "tile_id": np.asarray(args.tile_id),
        "event_ids": event_ids,
        "region_node_indices": np.arange(len(xyz), dtype=np.uint32),
        "region_xyz": xyz.astype(np.float64),
        "horizontal_spacing_m": np.asarray(horizontal_spacing_m),
    }
    phase_audits = {}
    maximum_chunk_uncompressed_bytes = 0
    verified_chunks = 0
    for phase in ("P", "S"):
        chunk_rows = manifest["phases"][phase]
        if [row["event_id"] for row in chunk_rows] != event_ids.tolist():
            raise RuntimeError(f"{phase} chunk event order differs")
        scale_map = {
            row["event_id"]: row for row in scales["phases"][phase]
        }
        offsets = [0]
        node_parts = []
        intensity_parts = []
        reliability_parts = []
        fused_sum = np.zeros(len(xyz), np.float64)
        view_sum = np.zeros(len(xyz), np.float64)
        exposure_count = np.zeros(len(xyz), np.uint16)
        detection_count = np.zeros(len(xyz), np.uint16)
        sectors = np.zeros(len(xyz), np.uint8)
        for event_index, (event_id, (source_x, source_y), chunk_row) in enumerate(
            zip(event_ids, sources, chunk_rows)
        ):
            chunk_path = Path(chunk_row["path"])
            if sha256(chunk_path) != chunk_row["sha256"]:
                raise RuntimeError(f"chunk hash differs: {chunk_path}")
            with np.load(chunk_path, allow_pickle=False) as chunk:
                if (
                    str(np.asarray(chunk["event_id"]).item()) != event_id
                    or str(np.asarray(chunk["phase"]).item()) != phase
                    or bool(np.asarray(chunk["known_structure_used"]).item())
                ):
                    raise RuntimeError(f"chunk identity differs: {chunk_path}")
                response = np.asarray(chunk["response"], np.float32)
                illumination = np.asarray(chunk["illumination"], np.float32)
                view_a = np.asarray(chunk["view_a"], np.float32)
                view_b = np.asarray(chunk["view_b"], np.float32)
                illumination_a = np.asarray(chunk["view_a_illumination"], np.float32)
                illumination_b = np.asarray(chunk["view_b_illumination"], np.float32)
            arrays = (
                response, illumination, view_a, view_b, illumination_a, illumination_b
            )
            if any(array.shape != (len(xyz),) for array in arrays):
                raise RuntimeError(f"chunk node shape differs: {chunk_path}")
            maximum_chunk_uncompressed_bytes = max(
                maximum_chunk_uncompressed_bytes,
                sum(array.nbytes for array in arrays),
            )
            verified_chunks += 1
            scale = scale_map[event_id]
            exposed = illumination > 0.05
            detected = (response > 1.0e-15) & exposed
            norm = reference.normalize(response, float(scale["full"]))
            norm_a = reference.normalize(view_a, float(scale["view_a"]))
            norm_b = reference.normalize(view_b, float(scale["view_b"]))
            norm_a *= illumination_a > 0.05
            norm_b *= illumination_b > 0.05
            reliability = np.sqrt(norm_a * norm_b).astype(np.float32)
            local_nodes = np.flatnonzero(detected)
            node_parts.append(local_nodes.astype(np.uint32))
            intensity_parts.append(norm[local_nodes])
            reliability_parts.append(reliability[local_nodes])
            offsets.append(offsets[-1] + len(local_nodes))
            fused_sum += norm * detected
            view_sum += reliability * detected
            exposure_count += exposed
            detection_count += detected
            azimuth = (
                np.degrees(np.arctan2(source_x - xyz[:, 0], source_y - xyz[:, 1]))
                + 360.0
            ) % 360.0
            sector = np.floor(azimuth / 45.0).astype(np.uint8)
            sectors[detected] |= (1 << sector[detected]).astype(np.uint8)

        fused = np.divide(
            fused_sum,
            exposure_count,
            out=np.zeros(len(xyz)),
            where=exposure_count > 0,
        )
        view_reliability = np.divide(
            view_sum,
            detection_count,
            out=np.zeros(len(xyz)),
            where=detection_count > 0,
        )
        directions = reference.bit_count(sectors)
        eligible = (
            (exposure_count >= 2)
            & (detection_count >= 2)
            & (directions >= 2)
            & (fused > 0)
        )
        selected = np.flatnonzero(eligible)
        centers, axes_u, axes_v, _ = reference.fused_geometry(
            xyz, selected, fused, horizontal_spacing_m
        )
        output[f"{phase}_event_offsets"] = np.asarray(offsets, np.uint64)
        output[f"{phase}_event_node_index"] = (
            np.concatenate(node_parts) if node_parts else np.empty(0, np.uint32)
        )
        output[f"{phase}_event_intensity"] = (
            np.concatenate(intensity_parts) if intensity_parts else np.empty(0, np.float32)
        )
        output[f"{phase}_event_view_reliability"] = (
            np.concatenate(reliability_parts)
            if reliability_parts else np.empty(0, np.float32)
        )
        output[f"{phase}_fused_node_index"] = selected.astype(np.uint32)
        output[f"{phase}_fused_center_xyz_m"] = centers
        output[f"{phase}_fused_axis_u_m"] = axes_u
        output[f"{phase}_fused_axis_v_m"] = axes_v
        output[f"{phase}_fused_intensity"] = fused[selected].astype(np.float32)
        output[f"{phase}_fused_view_reliability"] = view_reliability[selected].astype(np.float32)
        output[f"{phase}_exposure_count"] = exposure_count[selected]
        output[f"{phase}_detection_count"] = detection_count[selected]
        output[f"{phase}_direction_sectors"] = directions[selected]
        phase_audits[phase] = {
            "events": len(event_ids),
            "eventwise_splats": int(offsets[-1]),
            "fused_splats": int(len(selected)),
            "global_scales_recomputed_per_tile": False,
            "one_event_phase_chunk_loaded_at_a_time": True,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(args.output)
    checks = {
        "known_structure_absent": True,
        "chunk_manifest_pass": manifest.get("pass") is True,
        "global_scale_manifest_pass": scales.get("pass") is True,
        "minimum_24_events": len(event_ids) >= 24,
        "exact_1p875_spacing": dx.tolist() == [1875.0] and dy.tolist() == [1875.0],
        "all_chunks_hash_verified": verified_chunks == 2 * len(event_ids),
        "one_event_phase_chunk_memory_bound": (
            maximum_chunk_uncompressed_bytes == len(xyz) * 6 * 4
        ),
        "both_phases_have_splats": all(
            report["fused_splats"] > 0 for report in phase_audits.values()
        ),
    }
    audit = {
        "schema": "chunk-streamed-tiled-eventwise-gs-build-audit-v493",
        "known_structure_used": False,
        "tile_id": args.tile_id,
        "chunks": str(args.chunks),
        "chunks_sha256": sha256(args.chunks),
        "scales": str(args.scales),
        "scales_sha256": sha256(args.scales),
        "maximum_chunk_uncompressed_bytes": maximum_chunk_uncompressed_bytes,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "phases": phase_audits,
        "checks": checks,
        "pass": all(checks.values()),
        "publication_allowed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tile_id": args.tile_id,
        "maximum_chunk_uncompressed_bytes": maximum_chunk_uncompressed_bytes,
        "phases": phase_audits,
        "checks": checks,
        "pass": audit["pass"],
    }, ensure_ascii=False))
    raise SystemExit(0 if audit["pass"] else 2)


if __name__ == "__main__":
    main()
