#!/usr/bin/env python3
"""Canonical queue/model status derived only from durable operations stores."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_queue_store_v1093 import DEFAULT_DB, read_snapshot
from agent_runtime_store_v1098 import snapshot as runtime_snapshot


ACTIVE_RUNTIME_STATES = {"claimed", "running", "checkpoint"}
SCIENCE_PREFIXES = (
    "NR-GS-", "NR-PIPELINE-", "NR-SCIENCE-", "NR-HINET-",
    "NR-LIGHTWEIGHT-EVENT-GS-",
)
EVENT_PATTERN = re.compile(r"hinet_[0-9]{14}")
FULL64 = re.compile(r"^[0-9a-f]{64}$")
HASH_AND_PATH = re.compile(r"^([0-9a-f]{64})\s+(.+)$")


class ArtifactHashConflictError(RuntimeError):
    """Raised when one artifact path is bound to different valid digests."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _dispatch_class(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    status = row.get("status")
    if status == "in_progress":
        return "running"
    if status == "complete_pass":
        return "completed"
    if status in {"closed_fail", "closed_fail_superseded", "cancelled"}:
        return "failed"
    reason = str(row.get("blocked_reason") or "").lower()
    if "approval" in reason or "承認" in reason:
        return "approval-wait"
    dependencies = row.get("dependencies") or []
    if any(by_id.get(dep, {}).get("status") != "complete_pass" for dep in dependencies):
        return "dependency-wait"
    if reason or status == "blocked":
        return "dependency-wait"
    if row.get("assignee") and status == "queued":
        return "slot-wait"
    return "ready"


def queue_payload(db_path: Path = DEFAULT_DB) -> dict[str, Any]:
    try:
        data = read_snapshot(db_path)
        runtime = runtime_snapshot(db_path)
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as error:
        return {
            "ok": False,
            "schema": "agent-queue-canonical-error-v1154",
            "source": "operations-db",
            "database": str(db_path),
            "error": f"{type(error).__name__}: {error}",
        }
    rows = data.get("requirements") or []
    by_id = {row["requirement_id"]: row for row in rows}
    counts = {key: 0 for key in (
        "total", "running", "ready", "dependency-wait", "slot-wait",
        "approval-wait", "failed", "completed",
    )}
    counts["total"] = len(rows)
    for row in rows:
        state = _dispatch_class(row, by_id)
        row["dispatch_state"] = state
        counts[state] += 1
    normalized_runtime = []
    for row in runtime.get("agents") or []:
        item = dict(row)
        item.update(
            created_utc=row.get("state_changed_at"),
            updated_utc=row.get("heartbeat_at"),
            scope=row.get("stage") or row.get("state"),
            result=(f"{row.get('progress') or '—'} · state={row.get('state')} "
                    f"· heartbeat_age={row.get('heartbeat_age_s')}s · stale={row.get('stale')} "
                    f"· slot_releasable={row.get('slot_releasable')}"),
            status=("in_progress" if row.get("state") in ACTIVE_RUNTIME_STATES
                    and not row.get("slot_releasable") else row.get("state")),
        )
        normalized_runtime.append(item)
    data.update(
        ok=True,
        schema="agent-queue-runtime-canonical-v1154",
        source="operations-db",
        database=str(db_path),
        dispatch_counts=counts,
        runtime_schema=runtime.get("schema"),
        runtime_agents=normalized_runtime,
        next_ready_claim=runtime.get("next_ready_claim"),
        heartbeat_contract_seconds=runtime.get("heartbeat_contract_seconds"),
        stale_after_seconds=runtime.get("stale_after_seconds"),
        push_constraint=runtime.get("push_constraint"),
    )
    return data


def _is_science_requirement(row: dict[str, Any]) -> bool:
    return str(row.get("requirement_id") or "").startswith(SCIENCE_PREFIXES)


def _event_id(*values: Any) -> str | None:
    for value in values:
        match = EVENT_PATTERN.search(str(value or ""))
        if match:
            return match.group(0)
    return None


def _timestamp(value: Any) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp()
    except (TypeError, ValueError):
        return 0.0


def _select_runtime(runtime_rows: list[dict[str, Any]],
                    by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Fresh first, then requirement priority, newest update, stable ID."""
    if not runtime_rows:
        return None
    fresh = [row for row in runtime_rows if not row.get("stale")]
    pool = fresh or runtime_rows

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        requirement = by_id.get(str(row.get("requirement_id")), {})
        return (
            int(requirement.get("priority", 9)),
            -_timestamp(requirement.get("updated_utc")),
            str(requirement.get("requirement_id") or row.get("requirement_id") or ""),
            str(row.get("assignee") or ""),
        )

    return sorted(pool, key=key)[0]


def _normalize_artifact_hashes(paths: list[Any], values: list[Any]) -> dict[str, str]:
    """Normalize hashes, rejecting any conflicting same-path full64 binding."""
    result: dict[str, str] = {}
    bare: list[str] = []

    def bind(path: str, digest: str) -> None:
        prior = result.get(path)
        if prior is not None and prior != digest:
            raise ArtifactHashConflictError(
                f"artifact hash conflict for {path}: {prior} != {digest}")
        result[path] = digest

    for value in values or []:
        if isinstance(value, dict):
            path = str(value.get("path") or "")
            digest = str(value.get("sha256") or "")
            if path and FULL64.fullmatch(digest):
                bind(path, digest)
        elif isinstance(value, str):
            match = HASH_AND_PATH.fullmatch(value.strip())
            if match:
                bind(match.group(2), match.group(1))
            elif FULL64.fullmatch(value.strip()):
                bare.append(value.strip())
    bare_index = 0
    for path_value in paths or []:
        path = str(path_value)
        if path not in result and bare_index < len(bare):
            bind(path, bare[bare_index])
            bare_index += 1
    return result


def build_status(*, db_path: Path = DEFAULT_DB, **_: Any) -> dict[str, Any]:
    queue = queue_payload(db_path)
    if not queue.get("ok"):
        return {
            "ok": False, "schema": "model-run-canonical-error-v1154",
            "generated_utc": _utc_now(), "status": "failed", "status_label": "FAILED",
            "error": queue.get("error"),
        }
    requirements = [row for row in queue.get("requirements") or [] if _is_science_requirement(row)]
    by_id = {row["requirement_id"]: row for row in requirements}
    runtime_rows = [
        row for row in queue.get("runtime_agents") or []
        if row.get("requirement_id") in by_id
        and row.get("state") in ACTIVE_RUNTIME_STATES
        and not row.get("slot_releasable")
    ]
    selected_runtime = _select_runtime(runtime_rows, by_id)
    selected_requirement = by_id.get(selected_runtime.get("requirement_id")) if selected_runtime else None
    fresh = bool(selected_runtime and not selected_runtime.get("stale"))
    if fresh:
        status, stop_reason = "running", None
    elif selected_runtime:
        status = "stale"
        stop_reason = "A science runtime row exists, but its authoritative heartbeat is stale."
    else:
        status = "paused"
        stop_reason = "No authoritative science runtime is currently active."
    blockers = [{"requirement_id": row.get("requirement_id"),
                 "text": row.get("blocked_reason") or row.get("result") or row.get("title")}
                for row in requirements if row.get("status") in {"blocked", "closed_fail"}]
    ready = [row for row in requirements if row.get("dispatch_state") == "ready"]
    in_progress = [row for row in requirements if row.get("status") == "in_progress"]
    completed = [row for row in requirements if row.get("status") == "complete_pass"]
    failed = [row for row in requirements if row.get("status") in {"closed_fail", "closed_fail_superseded"}]
    total = len(requirements)
    active_id = selected_requirement.get("requirement_id") if selected_requirement else None
    event_id = _event_id(active_id,
                         selected_requirement.get("title") if selected_requirement else None,
                         selected_requirement.get("scope") if selected_requirement else None,
                         selected_requirement.get("result") if selected_requirement else None)
    paths = list(selected_requirement.get("artifact_paths") or []) if selected_requirement else []
    try:
        hashes = _normalize_artifact_hashes(
            paths, list(selected_requirement.get("artifact_hashes") or []) if selected_requirement else [])
    except ArtifactHashConflictError as error:
        return {
            "ok": False, "schema": "model-run-canonical-error-v1154",
            "generated_utc": _utc_now(), "status": "failed", "status_label": "FAILED",
            "error": str(error), "evidence": [], "active_process": False,
            "historical_complete_is_active": False,
        }
    evidence = [{"path": str(path), "sha256": hashes[str(path)]}
                for path in paths if str(path) in hashes]
    return {
        "ok": True, "schema": "model-run-authoritative-operations-v1154",
        "generated_utc": _utc_now(), "status": status, "status_label": status.upper(),
        "active_process": fresh, "processes": [],
        "heartbeat": {
            "last_utc": selected_runtime.get("heartbeat_at") if selected_runtime else None,
            "age_seconds": selected_runtime.get("heartbeat_age_s") if selected_runtime else None,
            "ttl_seconds": queue.get("stale_after_seconds"), "fresh": fresh,
        },
        "work": {
            "development": {"completed": len(completed), "total": total},
            "event_id": event_id or "none",
            "phase": selected_runtime.get("stage") if selected_runtime else "none",
            "stage": selected_runtime.get("stage") if selected_runtime else "paused",
            "tile": {"completed": 0, "total": 0},
            "unit": {"completed": len(completed), "total": total,
                     "remaining": max(0, total - len(completed))},
        },
        "versions": {
            "queue": queue.get("schema"), "active_requirement": active_id,
            "active_revision": selected_requirement.get("revision") if selected_requirement else None,
        },
        "stages": [
            {"id": "complete", "label": "quality-gated complete", "state": "complete", "count": len(completed)},
            {"id": "running", "label": "active science requirements", "state": status, "count": len(in_progress)},
            {"id": "failed", "label": "closed scientific failures", "state": "failed" if failed else "complete", "count": len(failed)},
            {"id": "ready", "label": "ready scientific requirements", "state": "pending", "count": len(ready)},
        ],
        "blockers": blockers, "stop_reason": stop_reason,
        "resume_conditions": [row.get("requirement_id") for row in ready[:8]]
        or (["Wait for the active requirement heartbeat."] if runtime_rows
            else ["No science requirement is currently ready."]),
        "eta": {"available": False,
                "reason": "ETA is not synthesized from free-form runtime text; a machine-readable estimate is required."},
        "last_stage_durations_seconds": [], "evidence": evidence,
        "raw_audit_url": "/api/model-run/audit", "historical_complete_is_active": False,
    }
