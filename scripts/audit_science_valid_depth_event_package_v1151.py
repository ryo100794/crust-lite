#!/usr/bin/env python3
"""Formal self-audit for the inactive NR-059 event package."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/equake/crust-lite")
EVENT = "hinet_20260814000874"
REQ = "NR-SCIENCE-VALID-DEPTH-EVENT-PACKAGE-059"
PKG = ROOT / "data/interim/formal_hinet_valid_depth_event_package_v1150" / EVENT
MANIFEST = PKG / f"{EVENT}_valid_depth_event_package_v1150.manifest.json"
OUT = ROOT / "logs/audits/NR_SCIENCE_VALID_DEPTH_EVENT_PACKAGE_059"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def active_commands() -> list[str]:
    commands = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
        except (OSError, PermissionError):
            continue
        if command and any(token in command for token in (
            "build_science_valid_depth_event_package_v1150.py",
            "hinet_direct_wave_component_window_proof_v1144.py",
            "hinet_science_formal_input_adapter_v1127.py",
        )):
            commands.append(command)
    return commands


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    live_artifacts = []
    artifact_mismatch = []
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        actual = sha256(path) if path.is_file() else None
        live_artifacts.append({**artifact, "actual_sha256": actual, "match": actual == artifact["sha256"]})
        if actual != artifact["sha256"]:
            artifact_mismatch.append(artifact["path"])

    phase_checks = {}
    phase_summary = {}
    for phase in ("P", "S"):
        path = PKG / f"{EVENT}_{phase}_component_window_proof_v1144.npz"
        with np.load(path, allow_pickle=False) as package:
            channel_valid = package["channel_valid"].astype(bool)
            starts = package["component_formal_window_start_s"].astype(float)
            ends = package["component_formal_window_end_s"].astype(float)
            picks = package["component_picked_arrival_s"].astype(float)
            supports = package["component_support_radius_s"].astype(float)
            valid = channel_valid
            outside = ((picks - supports < starts - 1.0e-12) | (picks + supports > ends + 1.0e-12)) & valid
            phase_checks[phase] = {
                "schema_exact": str(package["schema"].item()) == "formal-hinet-component-window-direct-wave-proof-package-v1144",
                "method_requirement_049_preserved": str(package["requirement_id"].item()) == "NR-SCIENCE-DIRECT-WAVE-COMPONENT-WINDOW-049",
                "event_exact": str(package["event_id"].item()) == EVENT,
                "phase_exact": str(package["phase"].item()) == phase,
                "stations48_valid140": channel_valid.shape == (48, 3) and int(valid.sum()) == 140,
                "full3C46_vertical_only2": int((valid.sum(axis=1) == 3).sum()) == 46 and int((valid.sum(axis=1) == 1).sum()) == 2,
                "component_support_escape0": int(outside.sum()) == 0,
                "UTC_additional_zero": bool(np.all(package["additional_time_correction_s"] == 0)),
                "known_zero": int(package["known_structure_input_count"].item()) == 0,
                "ML_zero": int(package["machine_learning_operator_count"].item()) == 0,
                "kernel_GS_false": not bool(package["kernel_or_GS_called"].item()),
                "publication_false": not bool(package["publication_allowed"].item()),
                "finite_response": bool(np.isfinite(package["station_response_direct_removed_spectral"]).all()),
            }
            phase_summary[phase] = {
                "stations": int(channel_valid.shape[0]),
                "valid_components": int(valid.sum()),
                "full3C": int((valid.sum(axis=1) == 3).sum()),
                "vertical_only": int((valid.sum(axis=1) == 1).sum()),
                "support_escape": int(outside.sum()),
                "package_sha256": sha256(path),
            }

    proof_path = PKG / f"{EVENT}_component_window_direct_wave_proof_v1144.manifest.json"
    proof = json.loads(proof_path.read_text())
    geometry = manifest["geometry_provenance_finding"]
    running = active_commands()
    checks = {
        "outer_status_pass_candidate_inactive": manifest.get("status") == "PASS_CANDIDATE_NOT_ACTIVATED_PENDING_DIFFERENT_OWNER_AUDIT" and manifest.get("activation_allowed") is False,
        "event_exact_no_substitution": manifest["selected_event"]["event_id"] == EVENT and geometry["event_substituted"] is False,
        "depth4_magnitude4p4": manifest["selected_event"]["depth_km"] == 4.0 and manifest["selected_event"]["magnitude"] == 4.4,
        "stations48_traces140_components": manifest["selected_event"]["station_count"] == 48 and manifest["selected_event"]["eligible_trace_count"] == 140 and manifest["selected_event"]["component_counts"] == {"E": 46, "N": 46, "U": 48},
        "formal_gap_recomputed_sectors3": abs(manifest["selected_event"]["max_azimuth_gap_deg"] - 251.65654877650638) <= 1.0e-9 and manifest["selected_event"]["sector_count_45deg"] == 3,
        "056_nonreproducible_gap_not_authoritative": geometry["inventory_reported_gap_deg"] == 251.45596985662473 and abs(geometry["absolute_difference_deg"] - 0.2005789198816501) <= 1.0e-12 and geometry["inventory_formula_preserved"] is False and geometry["inventory_gap_authoritative"] is False,
        "selection_no_GS_result_access": geometry["GS_result_accessed"] is False and manifest["GS_run_allowed"] is False,
        "artifact_hashes_all_live": not artifact_mismatch,
        "P_package_contract_all": all(phase_checks["P"].values()),
        "S_package_contract_all": all(phase_checks["S"].values()),
        "proof_P_all_gates": all(proof["phase_results"]["P"]["gates"].values()),
        "proof_S_all_gates": all(proof["phase_results"]["S"]["gates"].values()),
        "proof_CPU_GPU92_pass": proof["CPU_GPU_parity"]["pass"] is True and proof["CPU_GPU_parity"]["cases"] == 92,
        "unused_and_injection_independence": proof["phase_results"]["P"]["quality"]["unused_window_max_abs_change"] == 0.0 and proof["phase_results"]["S"]["quality"]["unused_window_max_abs_change"] == 0.0 and proof["phase_results"]["P"]["quality"]["injected_retention_abs_error_max"] <= 1.0e-10 and proof["phase_results"]["S"]["quality"]["injected_retention_abs_error_max"] <= 1.0e-10,
        "known_ML_UTC_zero": manifest["known_structure_input_count"] == 0 and manifest["machine_learning_operator_count"] == 0 and manifest["additional_time_correction_s"] == 0,
        "no_GS_model_pointer_public": manifest["GS_or_model_called"] is False and manifest["pointer_modified"] is False and manifest["publication_modified"] is False,
        "formal_DB_pointer_public_invariant": all(item["unchanged"] and item["before"] == item["after"] for item in manifest["immutability"].values()),
        "no_generation_process_remaining": not running,
        "different_owner_audit_required": manifest["independent_audit_required"] is True,
    }
    passed = all(checks.values())
    result = {
        "schema": "nr-science-valid-depth-event-package-059-formal-audit-v1151",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "requirement_id": REQ,
        "status": "PASS_CANDIDATE_INACTIVE_PENDING_DIFFERENT_OWNER_AUDIT_WITH_GEOMETRY_PROVENANCE_FINDING" if passed else "FAIL",
        "decision": "PASS_PACKAGE_NOT_ACTIVATED_INDEPENDENT_AUDIT_REQUIRED" if passed else "FAIL_PACKAGE_REMAINS_BLOCKED",
        "activation_allowed": False,
        "GS_run_allowed": False,
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": [name for name, value in checks.items() if not value],
        "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST)},
        "generator": {"path": "scripts/build_science_valid_depth_event_package_v1150.py", "sha256": sha256(ROOT / "scripts/build_science_valid_depth_event_package_v1150.py")},
        "live_artifacts": live_artifacts,
        "artifact_mismatch": artifact_mismatch,
        "phase_checks": phase_checks,
        "phase_summary": phase_summary,
        "component_quality": manifest["component_quality"],
        "geometry_provenance_finding": geometry,
        "immutability": manifest["immutability"],
        "active_generation_processes": running,
        "known_structure_input_count": 0,
        "machine_learning_operator_count": 0,
        "additional_time_correction_s": 0,
        "GS_or_model_called": False,
        "pointer_modified": False,
        "public_modified": False,
        "different_owner_independent_audit_required": True,
        "new_requirement_reported_not_implemented": {
            "proposed_id": "NR-SCIENCE-EVENT-GEOMETRY-PROVENANCE-064",
            "reason": "056 preregistration persisted a non-reproducible 251.4559698566 degree gap without SQL/formula/source; formal phase-feature geometry independently gives 251.6565487765 degrees. Preserve event selection but replace geometry evidence with a versioned formula/query/source hash before reuse.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "NR_SCIENCE_VALID_DEPTH_EVENT_PACKAGE_059.audit.json"
    md_path = OUT / "NR_SCIENCE_VALID_DEPTH_EVENT_PACKAGE_059.audit.md"
    sums_path = OUT / "SHA256SUMS"
    atomic(audit_path, json.dumps(result, indent=2) + "\n")
    md = f"""# NR-SCIENCE-VALID-DEPTH-EVENT-PACKAGE-059 formal audit

- Decision: **{result['decision']}**
- Checks: **{sum(checks.values())}/{len(checks)}**
- Event: `{EVENT}`; depth 4.0 km; M4.4; 48 stations / 140 traces; E46 N46 U48.
- P/S component support escape: 0/0; unused-window change: 0/0.
- CPU/GPU parity: 92 cases PASS.
- Formal geodesic max gap: 251.6565487765 deg; sectors: 3.
- 056 inventory's 251.4559698566 deg value is non-reproducible (difference 0.2005789199 deg) and is recorded as a provenance finding, not silently accepted.
- Known structure / ML / extra UTC / GS / pointer / public changes: all zero.
- Candidate remains inactive and requires a different-owner independent audit before 056 may rerun.
"""
    atomic(md_path, md)
    entries = [
        (audit_path, sha256(audit_path)),
        (md_path, sha256(md_path)),
        (MANIFEST, sha256(MANIFEST)),
        (ROOT / "scripts/build_science_valid_depth_event_package_v1150.py", sha256(ROOT / "scripts/build_science_valid_depth_event_package_v1150.py")),
    ]
    atomic(sums_path, "".join(f"{digest}  {path.relative_to(ROOT)}\n" for path, digest in entries))
    print(json.dumps({
        "decision": result["decision"],
        "checks": f"{sum(checks.values())}/{len(checks)}",
        "audit": str(audit_path.relative_to(ROOT)),
        "audit_sha256": sha256(audit_path),
        "md_sha256": sha256(md_path),
        "manifest_sha256": sha256(MANIFEST),
        "ledger_sha256": sha256(sums_path),
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
