#!/usr/bin/env python3
"""Append-only agent dispatch/heartbeat state beside, never inside, science data."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_queue_store_v1093 import DEFAULT_DB, canonical_json, connect

STATES = {"queued", "claimed", "running", "checkpoint", "completed", "failed", "paused"}
STALE_SECONDS = 90


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def init(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS ops_agent_runtime_event(
      event_id INTEGER PRIMARY KEY AUTOINCREMENT,
      assignee TEXT NOT NULL,
      requirement_id TEXT NOT NULL REFERENCES ops_requirement(requirement_id),
      state TEXT NOT NULL CHECK(state IN ('queued','claimed','running','checkpoint','completed','failed','paused')),
      occurred_utc TEXT NOT NULL,
      heartbeat_utc TEXT,
      stage TEXT,
      progress_text TEXT,
      eta_text TEXT,
      evidence_paths_json TEXT NOT NULL DEFAULT '[]',
      evidence_hashes_json TEXT NOT NULL DEFAULT '[]',
      actor TEXT NOT NULL,
      reason TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_runtime_event_agent ON ops_agent_runtime_event(assignee,event_id DESC);
    CREATE TABLE IF NOT EXISTS ops_agent_runtime_current(
      assignee TEXT PRIMARY KEY,
      requirement_id TEXT NOT NULL REFERENCES ops_requirement(requirement_id),
      state TEXT NOT NULL,
      state_changed_at TEXT NOT NULL,
      heartbeat_at TEXT,
      stage TEXT,
      progress_text TEXT,
      eta_text TEXT,
      evidence_paths_json TEXT NOT NULL DEFAULT '[]',
      evidence_hashes_json TEXT NOT NULL DEFAULT '[]',
      updated_utc TEXT NOT NULL
    );
    """)
    con.commit()


def transition(con: sqlite3.Connection, *, assignee: str, requirement_id: str, state: str,
               actor: str, reason: str, stage: str | None = None, progress: str | None = None,
               eta: str | None = None, evidence_paths: list[str] | None = None,
               evidence_hashes: list[str] | None = None, at: str | None = None) -> None:
    if state not in STATES: raise ValueError(f"invalid runtime state {state}")
    paths, hashes = evidence_paths or [], evidence_hashes or []
    if state == "completed" and (not paths or len(paths) != len(hashes) or any(len(x) != 64 for x in hashes)):
        raise ValueError("completed requires one SHA-256 per evidence path")
    if con.execute("SELECT 1 FROM ops_requirement WHERE requirement_id=?", (requirement_id,)).fetchone() is None:
        raise ValueError(f"unknown requirement {requirement_id}")
    at = at or now_utc(); heartbeat = at if state in {"claimed", "running", "checkpoint"} else None
    with con:
        con.execute("INSERT INTO ops_agent_runtime_event(assignee,requirement_id,state,occurred_utc,heartbeat_utc,stage,progress_text,eta_text,evidence_paths_json,evidence_hashes_json,actor,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (assignee, requirement_id, state, at, heartbeat, stage, progress, eta, canonical_json(paths), canonical_json(hashes), actor, reason))
        con.execute("INSERT INTO ops_agent_runtime_current(assignee,requirement_id,state,state_changed_at,heartbeat_at,stage,progress_text,eta_text,evidence_paths_json,evidence_hashes_json,updated_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(assignee) DO UPDATE SET requirement_id=excluded.requirement_id,state=excluded.state,state_changed_at=excluded.state_changed_at,heartbeat_at=excluded.heartbeat_at,stage=excluded.stage,progress_text=excluded.progress_text,eta_text=excluded.eta_text,evidence_paths_json=excluded.evidence_paths_json,evidence_hashes_json=excluded.evidence_hashes_json,updated_utc=excluded.updated_utc",
                    (assignee, requirement_id, state, at, heartbeat, stage, progress, eta, canonical_json(paths), canonical_json(hashes), at))


def heartbeat(con: sqlite3.Connection, assignee: str, actor: str, stage: str | None, progress: str | None, eta: str | None) -> None:
    row = con.execute("SELECT * FROM ops_agent_runtime_current WHERE assignee=?", (assignee,)).fetchone()
    if row is None: raise ValueError(f"no runtime row for {assignee}")
    if row["state"] not in {"claimed", "running", "checkpoint"}: raise ValueError(f"cannot heartbeat state {row['state']}")
    transition(con, assignee=assignee, requirement_id=row["requirement_id"], state="running", actor=actor,
               reason="heartbeat", stage=stage or row["stage"], progress=progress or row["progress_text"], eta=eta or row["eta_text"])


def _age(value: str | None, now: datetime) -> int | None:
    if not value: return None
    try: return max(0, int((now - datetime.fromisoformat(value.replace("Z", "+00:00"))).total_seconds()))
    except ValueError: return None


def snapshot(db_path: Path = DEFAULT_DB) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    con = connect(db_path, readonly=True); con.row_factory = sqlite3.Row
    try:
        try: rows = [dict(x) for x in con.execute("SELECT c.*,r.status requirement_status,r.artifact_paths_json requirement_artifacts,r.artifact_hashes_json requirement_hashes FROM ops_agent_runtime_current c JOIN ops_requirement r USING(requirement_id) ORDER BY c.assignee")]
        except sqlite3.OperationalError: rows = []
        requirements = [dict(x) for x in con.execute("SELECT requirement_id,status,priority,created_utc,dependencies_json,blocked_reason,assignee FROM ops_requirement ORDER BY priority,created_utc,requirement_id")]
    finally: con.close()
    agents = []
    for r in rows:
        age = _age(r["heartbeat_at"], now); stale = r["state"] in {"claimed","running","checkpoint"} and (age is None or age > STALE_SECONDS)
        paths, hashes = json.loads(r["requirement_artifacts"]), json.loads(r["requirement_hashes"])
        evidence_valid = bool(paths) and len(paths) == len(hashes) and all(isinstance(x, str) and len(x) == 64 for x in hashes)
        releasable = r["state"] in {"completed","failed","paused"} or (r["requirement_status"] == "complete_pass" and evidence_valid)
        agents.append({"assignee":r["assignee"],"requirement_id":r["requirement_id"],"state":r["state"],"state_changed_at":r["state_changed_at"],"heartbeat_at":r["heartbeat_at"],"heartbeat_age_s":age,"stale":stale,"slot_releasable":releasable,"stage":r["stage"],"progress":r["progress_text"],"eta":r["eta_text"],"requirement_status":r["requirement_status"]})
    by = {r["requirement_id"]: r for r in requirements}
    ready = []
    for r in requirements:
        if r["status"] != "queued" or r["blocked_reason"]: continue
        deps = json.loads(r["dependencies_json"])
        if all(by.get(dep, {}).get("status") == "complete_pass" for dep in deps): ready.append(r)
    next_ready = ready[0]["requirement_id"] if ready else None
    return {"schema":"agent-runtime-status-v1098","generated_utc":now_utc(),"heartbeat_contract_seconds":60,"stale_after_seconds":STALE_SECONDS,"agents":agents,"next_ready_claim":next_ready,"push_constraint":"RunPod cannot push directly to Codex collaboration mailbox. Active-turn notification uses the parent mailbox; persistent DB/API is authoritative between turns."}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("action",choices=["init","transition","heartbeat","snapshot"]); p.add_argument("--assignee"); p.add_argument("--requirement-id"); p.add_argument("--state"); p.add_argument("--actor",default="runtime-cli"); p.add_argument("--reason",default="explicit runtime update"); p.add_argument("--stage"); p.add_argument("--progress"); p.add_argument("--eta"); p.add_argument("--evidence-path",action="append",default=[]); p.add_argument("--evidence-hash",action="append",default=[]); a=p.parse_args()
    if a.action=="snapshot": print(json.dumps(snapshot(),ensure_ascii=False,indent=2)); return
    con=connect(DEFAULT_DB); init(con)
    try:
        if a.action=="transition": transition(con,assignee=a.assignee,requirement_id=a.requirement_id,state=a.state,actor=a.actor,reason=a.reason,stage=a.stage,progress=a.progress,eta=a.eta,evidence_paths=a.evidence_path,evidence_hashes=a.evidence_hash)
        elif a.action=="heartbeat": heartbeat(con,a.assignee,a.actor,a.stage,a.progress,a.eta)
    finally: con.close()
    print(json.dumps({"ok":True,"action":a.action,"at":now_utc()}))


if __name__=="__main__": main()
