#!/usr/bin/env python3
"""Run one blind national event-phase through freeze and corrected chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import national_depth_consumer_guard_v1144 as depth_guard


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--phase", choices=("P", "S"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--grid-audit", type=Path, required=True)
    parser.add_argument("--execution-gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()

    depth_guard.enforce_consumer(args.project.resolve(), "event-phase-orchestrator", args.contract)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    grid = json.loads(args.grid_audit.read_text(encoding="utf-8"))
    gate = json.loads(args.execution_gate.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not (
        contract.get("pass") is True
        and grid.get("pass") is True
        and gate.get("pass") is True
        and gate.get("national_execution_allowed") is True
        and protocol.get("pass") is True
        and all(item.get("known_structure_used") is False for item in (contract, grid, gate, protocol))
        and all(item.get("publication_allowed") is False for item in (contract, grid, gate, protocol))
    ):
        raise RuntimeError("passing nonpublic blind national inputs required")
    if protocol.get("tile_contract_sha256") != sha256(args.contract):
        raise RuntimeError("source protocol does not pin this tile contract")

    grid_by_id = {row["id"]: row for row in grid["tiles"]}
    tile_arguments: list[str] = []
    for row in contract["tiles"]:
        materialized = grid_by_id.get(row["id"])
        if materialized is None or materialized["path"] != row["grid_path"] or materialized["sha256"] != row["grid_sha256"]:
            raise RuntimeError(f"grid provenance differs: {row['id']}")
        tile_arguments.extend(["--tile", f"{row['id']}={row['grid_path']}"])

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    stem = f"{args.event_id}_{args.phase}"
    parameters = args.output_root / "parameters" / f"{stem}_shared_isochron_v472.json"
    freeze_progress = args.output_root / "progress" / f"{stem}_freeze_v472.json"
    build_progress = args.output_root / "progress" / f"{stem}_build_v517.json"
    manifest = args.output_root / "manifests" / f"{stem}_chunks_v517.json"
    audit = args.output_root / "audits" / f"{stem}_runner_v520.json"
    runner_progress = args.output_root / "progress" / f"{stem}_runner_v520.json"

    common_provenance = {
        "schema": "national-one-event-phase-runner-progress-v520",
        "event_id": args.event_id,
        "phase": args.phase,
        "known_structure_used": False,
        "publication_allowed": False,
    }
    atomic_json(runner_progress, {**common_provenance, "updated_at_utc": utc_now(), "stage": "freezing_station_parameters", "stage_index": 1, "stages_total": 2})
    freeze_command = [
        str(args.project / ".venv/bin/python"), str(args.freeze),
        "--project", str(args.project), "--features", str(args.features), "--events", str(args.events),
        "--event-id", args.event_id, "--phase", args.phase,
        *tile_arguments,
        "--contract", str(args.contract), "--grid-audit", str(args.grid_audit),
        "--migration", str(args.migration), "--stream", str(args.stream),
        "--output", str(parameters), "--progress", str(freeze_progress), "--scratch", str(args.scratch),
    ]
    freeze_result = subprocess.run(freeze_command, cwd=args.project, check=False)
    if freeze_result.returncode != 0 or not parameters.is_file():
        atomic_json(runner_progress, {**common_provenance, "updated_at_utc": utc_now(), "stage": "freeze_failed", "returncode": freeze_result.returncode})
        raise SystemExit(freeze_result.returncode or 2)
    parameter_payload = json.loads(parameters.read_text(encoding="utf-8"))
    if parameter_payload.get("pass") is not True or parameter_payload.get("known_structure_used") is not False:
        raise RuntimeError("station parameter quality gate failed")

    atomic_json(runner_progress, {**common_provenance, "updated_at_utc": utc_now(), "stage": "building_corrected_chunks", "stage_index": 2, "stages_total": 2})
    build_command = [
        str(args.project / ".venv/bin/python"), str(args.builder),
        "--project", str(args.project), "--features", str(args.features), "--events", str(args.events),
        "--event-id", args.event_id, "--phase", args.phase,
        *tile_arguments,
        "--contract", str(args.contract), "--parameters", str(parameters), "--protocol", str(args.protocol),
        "--migration", str(args.migration), "--stream", str(args.stream),
        "--output-dir", str(args.output_root / "chunks"), "--manifest", str(manifest),
        "--progress", str(build_progress), "--scratch", str(args.scratch),
    ]
    build_result = subprocess.run(build_command, cwd=args.project, check=False)
    if build_result.returncode != 0 or not manifest.is_file():
        atomic_json(runner_progress, {**common_provenance, "updated_at_utc": utc_now(), "stage": "build_failed", "returncode": build_result.returncode})
        raise SystemExit(build_result.returncode or 2)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

    checks = {
        "blind_contract_pass": contract.get("pass") is True and contract.get("known_structure_used") is False,
        "national_execution_gate_pass": gate.get("pass") is True and gate.get("national_execution_allowed") is True,
        "all_contract_tiles_frozen": len(parameter_payload.get("tile_nodes", {})) == int(contract.get("tile_count", -1)),
        "all_contract_tiles_built": len(manifest_payload.get("tiles", [])) == int(contract.get("tile_count", -1)),
        "core_union_exact": parameter_payload.get("core_nodes") == manifest_payload.get("core_nodes") == contract.get("source_grid_expected_nodes"),
        "station_parameters_pass": parameter_payload.get("pass") is True,
        "corrected_chunk_manifest_pass": manifest_payload.get("pass") is True,
        "parameter_contract_hash_exact": parameter_payload.get("contract_sha256") == sha256(args.contract),
        "builder_used_frozen_parameters": manifest_payload.get("input_sha256", {}).get("parameters") == sha256(parameters),
        "known_structure_absent": parameter_payload.get("known_structure_used") is False and manifest_payload.get("known_structure_used") is False,
        "publication_remains_prohibited": manifest_payload.get("publication_allowed") is False,
    }
    payload = {
        "schema": "national-one-event-phase-runner-audit-v520",
        "completed_at_utc": utc_now(),
        "event_id": args.event_id,
        "phase": args.phase,
        "inputs": {
            "features": str(args.features), "events": str(args.events), "contract": str(args.contract),
            "grid_audit": str(args.grid_audit), "execution_gate": str(args.execution_gate),
            "protocol": str(args.protocol), "migration": str(args.migration), "stream": str(args.stream),
            "freeze": str(args.freeze), "builder": str(args.builder),
        },
        "outputs": {
            "parameters": str(parameters), "freeze_progress": str(freeze_progress),
            "manifest": str(manifest), "build_progress": str(build_progress),
        },
        "input_sha256": {
            "features": sha256(args.features), "events": sha256(args.events), "contract": sha256(args.contract),
            "grid_audit": sha256(args.grid_audit), "execution_gate": sha256(args.execution_gate),
            "protocol": sha256(args.protocol), "migration": sha256(args.migration), "stream": sha256(args.stream),
            "freeze": sha256(args.freeze), "builder": sha256(args.builder),
        },
        "output_sha256": {"parameters": sha256(parameters), "manifest": sha256(manifest)},
        "checks": checks,
        "pass": all(checks.values()),
        "known_structure_used": False,
        "publication_allowed": False,
    }
    atomic_json(audit, payload)
    atomic_json(runner_progress, {**common_provenance, "updated_at_utc": utc_now(), "stage": "complete" if payload["pass"] else "failed_quality_gate", "stage_index": 2, "stages_total": 2, "audit": str(audit), "pass": payload["pass"]})
    print(json.dumps({"event_id": args.event_id, "phase": args.phase, "checks": checks, "pass": payload["pass"]}, ensure_ascii=False))
    raise SystemExit(0 if payload["pass"] else 2)


if __name__ == "__main__":
    main()
