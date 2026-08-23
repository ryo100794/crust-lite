"""Single fail-closed resolver for the normal Hi-net-only data plane.

The normal CLI never downloads or materializes data.  It verifies the immutable
formal database and the two audited downstream pointers, then returns a
redacted manifest.  Legacy FDSN/ComCat, mixed CSV, active-fault, J-SHIS and
sample inputs are outside this resolver by construction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig, load_config
from crust_lite.io.database_selection import database_status
from crust_lite.paths import ProjectPaths


DOWNSTREAM_POINTER = Path("configs/hinet_downstream_input.json")
EVENTS_POINTER = Path("configs/formal_hinet_events.json")
FORMAL_POLICY = "NIED_HINET_ONLY"
FORMAL_CATALOG = "NIED Hi-net authenticated event catalog"


class FormalInputError(RuntimeError):
    """A formal pointer, source policy, or immutable artifact failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalInputError(f"formal pointer unavailable: {path}") from exc
    if not isinstance(payload, dict):
        raise FormalInputError(f"formal pointer root is not an object: {path}")
    return payload


def _artifact(root: Path, spec: dict[str, Any], label: str) -> dict[str, Any]:
    path_value = str(spec.get("path", ""))
    expected = str(spec.get("sha256", ""))
    if not path_value or not expected:
        raise FormalInputError(f"{label} has no path/SHA-256 contract")
    path = (root / path_value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise FormalInputError(f"{label} escapes project root") from exc
    if not path.is_file():
        raise FormalInputError(f"{label} is missing: {path_value}")
    actual = _sha256(path)
    if actual != expected:
        raise FormalInputError(f"{label} SHA-256 mismatch")
    return {"path": path_value, "sha256": actual, "bytes": path.stat().st_size}


def _validate_normal_config(config: AppConfig) -> None:
    source = config.data_sources
    forbidden_flags = {
        "use_fdsn": source.use_fdsn,
        "use_jshis": source.use_jshis,
        "use_gnss": source.use_gnss,
        "use_active_faults": source.use_active_faults,
        "use_waveforms": source.use_waveforms,
    }
    enabled = sorted(name for name, value in forbidden_flags.items() if value)
    forbidden_paths = {
        "event_csv": source.event_csv,
        "mechanism_csv": source.mechanism_csv,
        "gnss_csv": source.gnss_csv,
        "active_fault_file": source.active_fault_file,
        "waveform_spectra_csv": source.waveform_spectra_csv,
        "waveform_feature_csv": source.waveform_feature_csv,
    }
    configured = sorted(name for name, value in forbidden_paths.items() if value)
    if enabled or configured or source.catalog_source != FORMAL_CATALOG:
        raise FormalInputError(
            "normal config is not pointer-only Hi-net: "
            f"enabled={enabled}, configured_paths={configured}, "
            f"catalog_source={source.catalog_source!r}"
        )


def resolve_formal_inputs(config_path: str | Path) -> dict[str, Any]:
    """Verify all normal inputs without downloading or writing any artifact."""
    config = load_config(config_path)
    _validate_normal_config(config)
    paths = ProjectPaths.from_config(config)
    root = paths.root.resolve()
    db = database_status(paths)
    if not db.get("verified") or db.get("source_policy") != FORMAL_POLICY:
        raise FormalInputError("formal database is not verified Hi-net-only")

    downstream_path = root / DOWNSTREAM_POINTER
    downstream = _load_json(downstream_path)
    if (
        downstream.get("schema") != "hinet-downstream-formal-input-pointer-v1"
        or downstream.get("status") != "ACTIVE_PASS"
        or downstream.get("source_policy") != FORMAL_POLICY
        or downstream.get("known_structure_input_count") != 0
        or downstream.get("additional_time_correction_s") != 0
        or downstream.get("timezone_heuristic_allowed") is not False
    ):
        raise FormalInputError("downstream pointer policy/status contract failed")
    db_spec = downstream.get("formal_database") or {}
    if db_spec.get("sha256") != db.get("database_sha256") or db_spec.get("read_only") is not True:
        raise FormalInputError("downstream pointer database identity mismatch")
    features = _artifact(root, downstream.get("features") or {}, "P/S phase features")
    projections = _artifact(root, downstream.get("projections") or {}, "P/S projections")
    _artifact(root, {"path": (downstream.get("features") or {}).get("audit_path"), "sha256": (downstream.get("features") or {}).get("audit_sha256")}, "phase feature audit")
    _artifact(root, {"path": (downstream.get("projections") or {}).get("audit_path"), "sha256": (downstream.get("projections") or {}).get("audit_sha256")}, "phase projection audit")

    events_path = root / EVENTS_POINTER
    events = _load_json(events_path)
    if (
        events.get("schema") != "formal-hinet-v472-adapter-pointer-v1"
        or events.get("status") != "ACTIVE_PASS"
        or events.get("source_policy") != FORMAL_POLICY
        or events.get("known_structure_input_count") != 0
        or events.get("machine_learning_used") is not False
        or events.get("additional_time_correction_s") != 0
        or events.get("v493_minimum_24_gate_unchanged") is not True
    ):
        raise FormalInputError("formal event pointer policy/status contract failed")
    event_artifact = _artifact(root, events.get("events") or {}, "formal Hi-net events")
    event_features = events.get("features") or {}
    if event_features.get("path") != features["path"] or event_features.get("sha256") != features["sha256"]:
        raise FormalInputError("event/downstream feature identity mismatch")
    _artifact(root, events.get("independent_audit") or {}, "formal event independent audit")

    return {
        "schema": "formal-hinet-normal-input-status-v1120",
        "status": "PASS",
        "source_policy": FORMAL_POLICY,
        "database": {
            "engine": db["engine"],
            "role": db["role"],
            "read_only": db["read_only"],
            "sha256": db["database_sha256"],
        },
        "events": {**event_artifact, "rows": int((events.get("events") or {}).get("rows", 0))},
        "phase_features": {**features, "rows": int((downstream.get("features") or {}).get("rows", 0))},
        "phase_projections": {**projections, "rows": int((downstream.get("projections") or {}).get("rows", 0))},
        "eligible_trace_count": int(downstream.get("eligible_trace_count", 0)),
        "quarantined_horizontal_trace_count": int(downstream.get("quarantined_horizontal_trace_count", 0)),
        "time_basis": "UTC_ALREADY_CORRECTED",
        "additional_time_correction_s": 0,
        "known_structure_input_count": 0,
        "machine_learning_used": False,
        "network_requests": 0,
        "artifacts_created": 0,
    }
