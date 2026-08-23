#!/usr/bin/env python3
"""Authoritative model-run status derived from live processes and heartbeat evidence.

Operational metadata only.  Historical complete progress is evidence, never proof
of a currently running process.  The module is importable for API use and fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path("/workspace/equake/crust-lite")
EVENT = "hinet_20260814000874"
TTL_SECONDS = 180
RUN_MARKERS = (
    "run_event_evidence_pipeline", "full48_ps", "radial_p100_gs",
    "fuse_event_ps_gaussian_splats", "phase_aware_gs",
)
EXCLUDE_MARKERS = ("equake_dashboard", "model_run_status_v1097.py", "pytest")
PROGRESS_PATHS = (
    PROJECT / f"logs/{EVENT}_full48_ps_v1053.progress.json",
    PROJECT / f"logs/{EVENT}_radial_p100_gs_v1060.progress.json",
    PROJECT / f"logs/{EVENT}_evidence_pipeline_v1079.progress.json",
)
AUDIT_PATHS = (
    PROJECT / f"logs/{EVENT}_full48_ps_v1053.audit.json",
    PROJECT / f"logs/{EVENT}_radial_p100_gs_v1060.audit.json",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def discover_processes(project: Path = PROJECT) -> list[dict[str, Any]]:
    rows = []
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    boot = None
    try:
        boot = float(Path("/proc/stat").read_text().split("btime ", 1)[1].splitlines()[0])
    except (OSError, ValueError, IndexError):
        pass
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            cwd = str(proc.joinpath("cwd").resolve())
            stat = proc.joinpath("stat").read_text().split()
        except (OSError, IndexError):
            continue
        if str(project) not in (cmd + " " + cwd):
            continue
        if not any(marker in cmd for marker in RUN_MARKERS) or any(marker in cmd for marker in EXCLUDE_MARKERS):
            continue
        started = iso(datetime.fromtimestamp(boot + float(stat[21]) / ticks, timezone.utc)) if boot else None
        rows.append({"pid": int(proc.name), "started_utc": started, "command": cmd[:600]})
    return sorted(rows, key=lambda x: x["pid"])


def evidence(path: Path, now: datetime) -> dict[str, Any]:
    data = read_json(path)
    try:
        mt = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        age = max(0, int((now - mt).total_seconds()))
    except OSError:
        mt, age = None, None
    return {
        "path": str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else path.name,
        "exists": path.is_file(), "mtime_utc": iso(mt) if mt else None,
        "age_seconds": age, "sha256": sha256(path),
        "schema": data.get("schema"), "state": data.get("state"), "pass": data.get("pass"),
    }


def duration_samples(audits: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for audit in audits:
        for task in audit.get("tasks", []):
            if not isinstance(task, dict):
                continue
            for key in ("persistent_builder_wall_seconds", "wall_seconds", "duration_seconds"):
                value = task.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    values.append(float(value)); break
    return values


def eta_for(status: str, completed: int, total: int, samples: list[float]) -> dict[str, Any]:
    if status != "running":
        return {"available": False, "reason": "ETA is calculated only for a live process with a fresh heartbeat."}
    if total <= completed:
        return {"available": False, "reason": "No remaining units."}
    if len(samples) < 3:
        return {"available": False, "reason": "Fewer than three comparable stage/tile durations."}
    median = statistics.median(samples)
    deviations = [abs(x - median) for x in samples]
    mad = statistics.median(deviations) or median * 0.15
    remaining = total - completed
    center = median * remaining
    low = max(0.0, (median - 1.4826 * mad) * remaining)
    high = (median + 1.4826 * mad) * remaining
    confidence = "high" if len(samples) >= 10 and mad / median < 0.35 else "medium" if len(samples) >= 6 else "low"
    return {"available": True, "seconds": round(center), "range_seconds": [round(low), round(high)], "confidence": confidence, "sample_count": len(samples), "rolling_median_seconds": round(median, 2)}


def build_status(*, now: datetime | None = None, processes: list[dict[str, Any]] | None = None,
                 fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    now = now or utc_now()
    fixture = fixture or {}
    processes = discover_processes() if processes is None else processes
    progress_docs = [read_json(p) for p in PROGRESS_PATHS]
    audit_docs = [read_json(p) for p in AUDIT_PATHS]
    ev = [evidence(p, now) for p in (*PROGRESS_PATHS, *AUDIT_PATHS)]
    if fixture.get("heartbeat_age_seconds") is not None:
        heartbeat_age = int(fixture["heartbeat_age_seconds"])
        heartbeat_utc = iso(datetime.fromtimestamp(now.timestamp() - heartbeat_age, timezone.utc))
    else:
        ages = [x["age_seconds"] for x in ev[:len(PROGRESS_PATHS)] if x["age_seconds"] is not None]
        heartbeat_age = min(ages) if ages else None
        heartbeat_utc = next((x["mtime_utc"] for x in sorted(ev[:len(PROGRESS_PATHS)], key=lambda x: x["age_seconds"] if x["age_seconds"] is not None else 10**18)), None)
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= TTL_SECONDS
    active = bool(processes)
    explicit = fixture.get("explicit_state")
    if explicit in {"failed", "complete", "blocked", "paused"}:
        status = explicit
    elif active and heartbeat_fresh:
        status = "running"
    elif active:
        status = "stale"
    else:
        status = "paused"
    completed_units = int(fixture.get("completed_units", 24))
    total_units = int(fixture.get("total_units", 26))
    samples = fixture.get("durations")
    if samples is None:
        samples = duration_samples(audit_docs)
    failed = next((doc for doc in progress_docs if doc.get("state") == "failed"), {})
    blockers = [
        {"requirement_id": None, "text": "Official calibrated input switch is not complete."},
        {"requirement_id": "NR-KNOWN-LEAK-008", "text": "Known-structure leakage audit/remediation is pending."},
    ]
    stop_reason = fixture.get("stop_reason") or (
        "No authoritative model process is running; P/S and 12 tile/event GS are historical completed stages, fusion failed, and final is pending."
    )
    return {
        "ok": True, "schema": "model-run-authoritative-status-v1", "generated_utc": iso(now),
        "status": status, "status_label": status.upper(), "active_process": active,
        "processes": processes, "heartbeat": {"last_utc": heartbeat_utc, "age_seconds": heartbeat_age, "ttl_seconds": TTL_SECONDS, "fresh": heartbeat_fresh},
        "work": {"development": {"completed": 4, "total": 7}, "event_id": EVENT, "phase": "P/S", "stage": "fusion", "tile": {"completed": 12, "total": 12}, "unit": {"completed": completed_units, "total": total_units, "remaining": max(0, total_units - completed_units)}},
        "versions": {"input": "Hi-net official calibrated input switch pending", "calibration": "Hi-net documented normalization/correction audit pending", "model": "event-wise GS v1060 + covariance fusion v1075", "gate": "frozen development gate v1077"},
        "stages": [
            {"id": "phase-manifest", "label": "P/S", "state": "complete", "historical": True},
            {"id": "event-gs", "label": "12 tile/event GS", "state": "complete", "historical": True},
            {"id": "fusion", "label": "P/S covariance fusion", "state": "failed", "historical": True, "error_type": failed.get("error_type"), "error_summary": (failed.get("error") or "").splitlines()[-1][:500]},
            {"id": "final", "label": "final quality", "state": "pending", "historical": False},
        ],
        "blockers": blockers, "stop_reason": stop_reason,
        "resume_conditions": ["Switch to the official calibrated Hi-net input and record its evidence hash.", "Resolve NR-KNOWN-LEAK-008 and invalidate/recompute contaminated outputs before resuming."],
        "eta": eta_for(status, completed_units, total_units, [float(x) for x in samples]),
        "last_stage_durations_seconds": [round(float(x), 3) for x in samples[-12:]],
        "evidence": ev, "raw_audit_url": "/api/model-run/audit",
        "historical_complete_is_active": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fixture = read_json(args.fixture) if args.fixture else None
    payload = build_status(fixture=fixture, processes=fixture.get("processes", []) if fixture and "processes" in fixture else None)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
