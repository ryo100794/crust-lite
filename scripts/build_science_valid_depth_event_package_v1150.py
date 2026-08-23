#!/usr/bin/env python3
"""Build the inactive NR-059 event package from the audited 049/054 method."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np

EVENT_ID = "hinet_20260814000874"
EXPECTED = {
    "depth_km": 4.0,
    "magnitude": 4.4,
    "station_count": 48,
    "eligible_trace_count": 140,
    "component_counts": {"E": 46, "N": 46, "U": 48},
    "max_azimuth_gap_deg": 251.65654877650638,
    "sector_count_45deg": 3,
}
REQ = "NR-SCIENCE-VALID-DEPTH-EVENT-PACKAGE-059"
SCHEMA = "formal-hinet-valid-depth-event-package-v1150"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def checked_run(command: list[str], root: Path) -> dict:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed {completed.returncode}: {command}\n{completed.stdout}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"command did not return JSON: {command}\n{completed.stdout}") from error


def circular_gap(azimuth: np.ndarray) -> float:
    values = np.sort(np.mod(np.asarray(azimuth, dtype=np.float64), 360.0))
    if values.size == 0:
        raise RuntimeError("empty station azimuth")
    return float(np.max(np.diff(np.r_[values, values[0] + 360.0])))


def main() -> None:
    root = Path("/workspace/equake/crust-lite").resolve()
    output_dir = root / "data/interim/formal_hinet_valid_depth_event_package_v1150" / EVENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = {
        "selection_inventory": (
            root / "logs/audits/NR_GS_FORMAL_WIRE_ACTIVATE_056/NR_SCIENCE_VALID_DEPTH_EVENT_PACKAGE_059.inventory.json",
            "b767f4a2d25281885ac9f2ccb4ff7fbdfaf88ba4d93355fa66ef51dccc76afb3",
        ),
        "independent_054": (
            root / "logs/audits/NR_SCIENCE_DIRECT_WAVE_COMPONENT_WINDOW_INDEPENDENT_054/NR_SCIENCE_DIRECT_WAVE_COMPONENT_WINDOW_INDEPENDENT_054.audit.json",
            "b8d57e72fff2735bfa50cb927be887337851d9703840fa933659b326a92f2667",
        ),
        "component_method_config": (
            root / "configs/hinet_direct_wave_component_window_candidate_v1144.json",
            "c6c327b9936e5ac2da1a3f0ef7f3ed62b17a519daa3b0c4930d619507a812517",
        ),
    }
    # The method config hash is checked against live data below; fail explicitly if
    # the locally preregistered expected value is stale instead of silently updating it.
    for name, (path, expected_hash) in fixed.items():
        if not path.is_file():
            raise RuntimeError(f"missing fixed input: {name}")
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(f"fixed input hash mismatch: {name} expected={expected_hash} actual={actual}")

    inventory = json.loads(fixed["selection_inventory"][0].read_text())
    selected = inventory.get("selected", {})
    if inventory.get("selection_stage") != "PREREGISTERED_WITHOUT_GS_RESULT_ACCESS":
        raise RuntimeError("selection was not preregistered without GS result access")
    if selected.get("event_id") != EVENT_ID:
        raise RuntimeError("preregistered event differs")
    for name in ("depth_km", "magnitude", "station_count", "eligible_trace_count", "sector_count_45deg"):
        if selected.get(name) != EXPECTED[name]:
            raise RuntimeError(f"selection contract differs: {name}")
    if abs(float(selected["max_azimuth_gap_deg"]) - 251.45596985662473) > 1.0e-9:
        raise RuntimeError("selection azimuth gap differs")
    if selected.get("component_counts") != EXPECTED["component_counts"]:
        raise RuntimeError("selection component counts differ")

    adapter_base_path = root / "configs/hinet_science_formal_adapter_candidate_v1127.json"
    adapter_base = json.loads(adapter_base_path.read_text())
    adapter_config = json.loads(json.dumps(adapter_base))
    adapter_config["created_at_utc"] = datetime.now(UTC).isoformat()
    adapter_config["event_id"] = EVENT_ID
    adapter_config["consumer_requirement_id"] = REQ
    adapter_config["selected_event_contract"] = EXPECTED
    adapter_config_path = root / "configs/hinet_science_formal_adapter_event_059_v1150.json"
    atomic_json(adapter_config_path, adapter_config)

    invariants = {
        "formal_db": (root / adapter_config["inputs"]["formal_database"]["path"]),
        "downstream_pointer": root / "configs/hinet_downstream_input.json",
        "formal_events_pointer": root / "configs/formal_hinet_events.json",
        "public_dashboard": root / "scripts/equake_dashboard_10101_canonical_r4_v1131.py",
    }
    before = {name: sha256(path) for name, path in invariants.items()}
    adapter_script = root / "scripts/hinet_science_formal_input_adapter_v1127.py"
    adapter_result = checked_run(
        [
            str(root / ".venv/bin/python"),
            str(adapter_script),
            "--project", str(root),
            "--config", str(adapter_config_path),
            "--event-id", EVENT_ID,
            "--output-dir", str(output_dir),
        ],
        root,
    )
    adapter_manifest_path = output_dir / f"{EVENT_ID}_formal_adapter_v1127.manifest.json"
    adapter_p = output_dir / f"{EVENT_ID}_P_pre_direct_v1127.npz"
    adapter_s = output_dir / f"{EVENT_ID}_S_pre_direct_v1127.npz"
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    if adapter_manifest.get("event_id") != EVENT_ID or adapter_manifest.get("known_structure_input_count") != 0:
        raise RuntimeError("adapter event/known-structure binding failed")
    if adapter_manifest.get("machine_learning_operator_count") != 0 or adapter_manifest.get("additional_time_correction_s") != 0:
        raise RuntimeError("adapter ML/UTC binding failed")
    if adapter_manifest.get("publication_allowed") or adapter_manifest.get("pointer_activation_allowed"):
        raise RuntimeError("adapter unexpectedly allowed activation/publication")

    direct_base_path = root / "configs/hinet_science_direct_wave_removal_candidate_v1129.json"
    direct_config = json.loads(direct_base_path.read_text())
    direct_config["created_at_utc"] = datetime.now(UTC).isoformat()
    direct_config["requirement_id"] = REQ
    direct_config["event_id"] = EVENT_ID
    direct_config["inputs"]["adapter_config"] = {"path": relative(root, adapter_config_path), "sha256": sha256(adapter_config_path)}
    direct_config["inputs"]["adapter_manifest"] = {"path": relative(root, adapter_manifest_path), "sha256": sha256(adapter_manifest_path)}
    direct_config["inputs"]["adapter_P_package"] = {"path": relative(root, adapter_p), "sha256": sha256(adapter_p)}
    direct_config["inputs"]["adapter_S_package"] = {"path": relative(root, adapter_s), "sha256": sha256(adapter_s)}
    direct_config["selected_event_contract"] = EXPECTED
    direct_config_path = root / "configs/hinet_science_direct_wave_event_059_v1150.json"
    atomic_json(direct_config_path, direct_config)

    component_base_path = fixed["component_method_config"][0]
    component_config = json.loads(component_base_path.read_text())
    component_config["requirement_id"] = REQ
    component_config["event_id"] = EVENT_ID
    component_config["status"] = "INACTIVE_ISOLATED_PROOF_CANDIDATE"
    component_config["consumer_status"] = "INACTIVE_EVENT_SPECIFIC_PACKAGE_CANDIDATE"
    component_config["inputs"]["v1129_config"] = {"path": relative(root, direct_config_path), "sha256": sha256(direct_config_path)}
    component_config["inputs"]["preregistered_selection"] = {"path": relative(root, fixed["selection_inventory"][0]), "sha256": sha256(fixed["selection_inventory"][0])}
    component_config["inputs"]["independent_method_audit_054"] = {"path": relative(root, fixed["independent_054"][0]), "sha256": sha256(fixed["independent_054"][0])}
    component_config["selected_event_contract"] = EXPECTED
    component_config_path = root / "configs/hinet_direct_wave_component_window_event_059_v1150.json"
    atomic_json(component_config_path, component_config)

    component_script = root / "scripts/hinet_direct_wave_component_window_proof_v1144.py"
    component_result = checked_run(
        [
            str(root / ".venv/bin/python"),
            str(component_script),
            "--project", str(root),
            "--config", str(component_config_path),
            "--output-dir", str(output_dir),
        ],
        root,
    )
    proof_manifest_path = output_dir / f"{EVENT_ID}_component_window_direct_wave_proof_v1144.manifest.json"
    proof_manifest = json.loads(proof_manifest_path.read_text())
    if proof_manifest.get("status") != "PASS" or proof_manifest.get("event_id") != EVENT_ID:
        raise RuntimeError("component-window method did not pass for exact event")
    if proof_manifest.get("kernel_or_GS_called") or proof_manifest.get("pointer_modified") or proof_manifest.get("publication_allowed"):
        raise RuntimeError("component-window method crossed inactive boundary")
    if proof_manifest.get("known_structure_input_count") != 0 or proof_manifest.get("machine_learning_operator_count") != 0:
        raise RuntimeError("component-window package contains prohibited inputs")
    if proof_manifest.get("additional_time_correction_s") != 0:
        raise RuntimeError("component-window package applied additional UTC correction")

    with np.load(adapter_p, allow_pickle=False) as package:
        azimuth = package["station_azimuth_deg"].astype(np.float64)
        station_count = len(package["station_ids"])
    measured_gap = circular_gap(azimuth)
    measured_sectors = len(set((np.mod(azimuth, 360.0) // 45.0).astype(int).tolist()))

    features_path = root / adapter_config["inputs"]["phase_features"]["path"]
    events_path = root / adapter_config["inputs"]["formal_events"]["path"]
    connection = duckdb.connect()
    event_rows = connection.execute(
        "select event_id, depth_km, magnitude from read_parquet(?) where event_id=?",
        [str(events_path), EVENT_ID],
    ).fetchall()
    component_rows = connection.execute(
        "select channel, count(distinct station_id) from read_parquet(?) where event_id=? and phase_family='P' and abs(frequency_hz-1.0)<1e-12 group by channel",
        [str(features_path), EVENT_ID],
    ).fetchall()
    trace_rows = connection.execute(
        "select count(*), count(distinct base_station_id), count(distinct station_id) from read_parquet(?) where event_id=? and phase_family='P' and abs(frequency_hz-1.0)<1e-12",
        [str(features_path), EVENT_ID],
    ).fetchone()
    connection.close()
    if len(event_rows) != 1:
        raise RuntimeError("formal event row is not unique")
    _, depth_km, magnitude = event_rows[0]
    observed_components = {str(channel): int(count) for channel, count in component_rows}
    observed_selection = {
        "event_id": EVENT_ID,
        "depth_km": float(depth_km),
        "magnitude": float(magnitude),
        "station_count": int(station_count),
        "eligible_trace_count": int(trace_rows[2]),
        "component_counts": observed_components,
        "max_azimuth_gap_deg": measured_gap,
        "sector_count_45deg": measured_sectors,
    }
    for name in ("depth_km", "magnitude", "station_count", "eligible_trace_count", "sector_count_45deg"):
        if observed_selection[name] != EXPECTED[name]:
            raise RuntimeError(f"fresh selected event measurement differs: {name}={observed_selection[name]}")
    if observed_components != EXPECTED["component_counts"]:
        raise RuntimeError(f"fresh component counts differ: {observed_components}")
    if abs(measured_gap - EXPECTED["max_azimuth_gap_deg"]) > 1.0e-6:
        raise RuntimeError(f"fresh azimuth gap differs: {measured_gap}")

    after = {name: sha256(path) for name, path in invariants.items()}
    if before != after:
        raise RuntimeError("formal/public/pointer invariant changed")

    package_paths = [
        adapter_config_path,
        adapter_manifest_path,
        adapter_p,
        adapter_s,
        direct_config_path,
        component_config_path,
        output_dir / f"{EVENT_ID}_P_component_window_proof_v1144.npz",
        output_dir / f"{EVENT_ID}_S_component_window_proof_v1144.npz",
        output_dir / f"{EVENT_ID}_component_window_parity_fixture_v1144.npz",
        proof_manifest_path,
    ]
    artifacts = [{"path": relative(root, path), "sha256": sha256(path)} for path in package_paths]
    gates = {
        "selection_preregistered_without_GS_result_access": True,
        "exact_event_id_no_substitution": True,
        "formal_depth_magnitude_exact": True,
        "stations48_traces140_E46_N46_U48": True,
        "azimuth_gap_and_three_sectors_exact": True,
        "adapter_formal_hinet_only": True,
        "P_component_window_all_gates": all(proof_manifest["phase_results"]["P"]["gates"].values()),
        "S_component_window_all_gates": all(proof_manifest["phase_results"]["S"]["gates"].values()),
        "CPU_GPU_parity": bool(proof_manifest["CPU_GPU_parity"]["pass"]),
        "known_structure_zero": proof_manifest["known_structure_input_count"] == 0,
        "ML_zero": proof_manifest["machine_learning_operator_count"] == 0,
        "additional_UTC_zero": proof_manifest["additional_time_correction_s"] == 0,
        "inactive_no_GS_pointer_public": not any((proof_manifest["kernel_or_GS_called"], proof_manifest["pointer_modified"], proof_manifest["publication_allowed"])),
        "formal_DB_pointer_public_invariant": before == after,
    }
    if not all(gates.values()):
        raise RuntimeError(f"NR-059 gates failed: {[name for name, passed in gates.items() if not passed]}")

    manifest = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "requirement_id": REQ,
        "status": "PASS_CANDIDATE_NOT_ACTIVATED_PENDING_DIFFERENT_OWNER_AUDIT",
        "decision": "PASS_EVENT_SPECIFIC_COMPONENT_WINDOW_PACKAGE_PENDING_INDEPENDENT_AUDIT",
        "activation_allowed": False,
        "GS_run_allowed": False,
        "selected_event": observed_selection,
        "preregistered_inventory_selected": selected,
        "geometry_provenance_finding": {"inventory_reported_gap_deg": float(selected["max_azimuth_gap_deg"]), "formal_phase_features_recomputed_gap_deg": measured_gap, "absolute_difference_deg": abs(measured_gap - float(selected["max_azimuth_gap_deg"])), "inventory_formula_preserved": False, "inventory_gap_authoritative": False, "event_substituted": False, "GS_result_accessed": False},
        "selection_inventory": {"path": relative(root, fixed["selection_inventory"][0]), "sha256": sha256(fixed["selection_inventory"][0])},
        "method_provenance": {
            "implementation_requirement_id": "NR-SCIENCE-DIRECT-WAVE-COMPONENT-WINDOW-049",
            "independent_audit_requirement_id": "NR-SCIENCE-DIRECT-WAVE-COMPONENT-WINDOW-INDEPENDENT-054",
            "independent_audit": {"path": relative(root, fixed["independent_054"][0]), "sha256": sha256(fixed["independent_054"][0])},
            "component_script": {"path": relative(root, component_script), "sha256": sha256(component_script)},
            "adapter_script": {"path": relative(root, adapter_script), "sha256": sha256(adapter_script)},
        },
        "gates": gates,
        "adapter_result": adapter_result,
        "component_result": component_result,
        "component_quality": {
            "P": proof_manifest["phase_results"]["P"]["quality"],
            "S": proof_manifest["phase_results"]["S"]["quality"],
            "CPU_GPU_parity": proof_manifest["CPU_GPU_parity"],
        },
        "artifacts": artifacts,
        "immutability": {name: {"before": before[name], "after": after[name], "unchanged": before[name] == after[name]} for name in before},
        "known_structure_input_count": 0,
        "machine_learning_operator_count": 0,
        "additional_time_correction_s": 0,
        "GS_or_model_called": False,
        "pointer_modified": False,
        "publication_modified": False,
        "independent_audit_required": True,
    }
    manifest_path = output_dir / f"{EVENT_ID}_valid_depth_event_package_v1150.manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "event_id": EVENT_ID,
        "selected_event": observed_selection,
        "P": manifest["component_quality"]["P"],
        "S": manifest["component_quality"]["S"],
        "CPU_GPU_parity": manifest["component_quality"]["CPU_GPU_parity"],
        "manifest": relative(root, manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
