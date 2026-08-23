#!/usr/bin/env python3
"""Durable operations requirement/agent queue store, isolated from science data.

The database created by this module contains operational metadata only.  It does
not join to, mutate, or infer state from the earthquake science RDB.  Imports are
explicit and append an audit record on every run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path("/workspace/equake/crust-lite")
DEFAULT_DB = PROJECT / "data/operations/agent_queue.sqlite"
DEFAULT_SEED = PROJECT / "configs/agent_queue_seed_v1093.json"
SCHEMA_VERSION = "agent-queue-v1093"
VALID_STATUSES = {
    "queued", "in_progress", "blocked", "complete_pass",
    "closed_fail", "closed_fail_superseded", "cancelled",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"expected list, got {type(value).__name__}")


def connect(db_path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db_path, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    if not readonly:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=FULL")
        con.execute("PRAGMA busy_timeout=5000")
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS ops_metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL,
          updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ops_requirement (
          requirement_id TEXT PRIMARY KEY,
          parent_id TEXT REFERENCES ops_requirement(requirement_id),
          title TEXT NOT NULL,
          evidence_json TEXT NOT NULL DEFAULT '[]',
          scope TEXT NOT NULL,
          priority INTEGER NOT NULL CHECK(priority BETWEEN 0 AND 9),
          status TEXT NOT NULL CHECK(status IN (
            'queued','in_progress','blocked','complete_pass','closed_fail',
            'closed_fail_superseded','cancelled'
          )),
          assignee TEXT,
          dependencies_json TEXT NOT NULL DEFAULT '[]',
          blocked_reason TEXT,
          created_utc TEXT NOT NULL,
          updated_utc TEXT NOT NULL,
          completed_utc TEXT,
          artifact_paths_json TEXT NOT NULL DEFAULT '[]',
          artifact_hashes_json TEXT NOT NULL DEFAULT '[]',
          acceptance TEXT NOT NULL,
          result TEXT,
          source TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_ops_requirement_queue
          ON ops_requirement(status, priority, created_utc, requirement_id);
        CREATE INDEX IF NOT EXISTS idx_ops_requirement_parent
          ON ops_requirement(parent_id);
        CREATE TABLE IF NOT EXISTS ops_requirement_history (
          history_id INTEGER PRIMARY KEY AUTOINCREMENT,
          requirement_id TEXT NOT NULL REFERENCES ops_requirement(requirement_id),
          from_status TEXT,
          to_status TEXT NOT NULL,
          changed_utc TEXT NOT NULL,
          actor TEXT NOT NULL,
          reason TEXT NOT NULL,
          revision INTEGER NOT NULL,
          snapshot_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ops_history_requirement
          ON ops_requirement_history(requirement_id, history_id DESC);
        CREATE TABLE IF NOT EXISTS ops_agent_snapshot (
          assignee TEXT PRIMARY KEY,
          state TEXT NOT NULL,
          current_requirement_id TEXT REFERENCES ops_requirement(requirement_id),
          observed_utc TEXT NOT NULL,
          imported_utc TEXT NOT NULL,
          source TEXT NOT NULL,
          detail_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS ops_progress_snapshot (
          progress_path TEXT PRIMARY KEY,
          observed_utc TEXT NOT NULL,
          imported_utc TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          state TEXT,
          requirement_id TEXT REFERENCES ops_requirement(requirement_id),
          payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ops_import_history (
          import_id INTEGER PRIMARY KEY AUTOINCREMENT,
          imported_utc TEXT NOT NULL,
          kind TEXT NOT NULL,
          source TEXT NOT NULL,
          source_sha256 TEXT,
          records_seen INTEGER NOT NULL,
          records_changed INTEGER NOT NULL,
          result TEXT NOT NULL,
          details_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    now = utc_now()
    con.execute(
        "INSERT INTO ops_metadata(key,value,updated_utc) VALUES('schema_version',?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_utc=excluded.updated_utc",
        (SCHEMA_VERSION, now),
    )
    con.commit()


def normalized_requirement(raw: dict[str, Any], now: str) -> dict[str, Any]:
    requirement_id = str(raw["requirement_id"]).strip()
    status = str(raw["status"]).strip()
    if not requirement_id:
        raise ValueError("requirement_id is empty")
    if status not in VALID_STATUSES:
        raise ValueError(f"{requirement_id}: invalid status {status!r}")
    completed = raw.get("completed_utc")
    if status in {"complete_pass", "closed_fail", "closed_fail_superseded", "cancelled"}:
        completed = completed or raw.get("updated_utc") or now
    else:
        completed = None
    return {
        "requirement_id": requirement_id,
        "parent_id": raw.get("parent_id") or None,
        "title": str(raw.get("title") or requirement_id),
        "evidence_json": canonical_json(json_list(raw.get("evidence", []))),
        "scope": str(raw.get("scope") or "operations"),
        "priority": int(raw.get("priority", 5)),
        "status": status,
        "assignee": raw.get("assignee") or None,
        "dependencies_json": canonical_json(json_list(raw.get("dependencies", []))),
        "blocked_reason": raw.get("blocked_reason") or None,
        "created_utc": raw.get("created_utc") or now,
        "updated_utc": raw.get("updated_utc") or now,
        "completed_utc": completed,
        "artifact_paths_json": canonical_json(json_list(raw.get("artifact_paths", []))),
        "artifact_hashes_json": canonical_json(json_list(raw.get("artifact_hashes", []))),
        "acceptance": str(raw.get("acceptance") or "Evidence and result must be recorded."),
        "result": raw.get("result") or None,
        "source": str(raw.get("source") or "explicit-import"),
    }


def requirement_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("evidence_json", "dependencies_json", "artifact_paths_json", "artifact_hashes_json"):
        out[key.removesuffix("_json")] = json.loads(out.pop(key))
    return out


def upsert_requirements(
    con: sqlite3.Connection, records: Iterable[dict[str, Any]], actor: str, reason: str
) -> tuple[int, int]:
    now = utc_now()
    normalized = [normalized_requirement(raw, now) for raw in records]
    identifiers = {row["requirement_id"] for row in normalized}
    existing_ids = {r[0] for r in con.execute("SELECT requirement_id FROM ops_requirement")}
    all_ids = existing_ids | identifiers
    for row in normalized:
        if row["parent_id"] and row["parent_id"] not in all_ids:
            raise ValueError(f"{row['requirement_id']}: unknown parent {row['parent_id']}")
        missing = [dep for dep in json.loads(row["dependencies_json"]) if dep not in all_ids]
        if missing:
            raise ValueError(f"{row['requirement_id']}: unknown dependencies {missing}")
    changed = 0
    with con:
        # Parents must be inserted before their children.
        pending = list(normalized)
        ordered: list[dict[str, Any]] = []
        known = set(existing_ids)
        while pending:
            ready = [row for row in pending if not row["parent_id"] or row["parent_id"] in known]
            if not ready:
                raise ValueError("parent cycle in requirement import")
            for row in ready:
                pending.remove(row)
                ordered.append(row)
                known.add(row["requirement_id"])
        for item in ordered:
            old = con.execute(
                "SELECT * FROM ops_requirement WHERE requirement_id=?", (item["requirement_id"],)
            ).fetchone()
            if old is None:
                revision, from_status = 1, None
                cols = list(item)
                con.execute(
                    f"INSERT INTO ops_requirement({','.join(cols)},revision) "
                    f"VALUES({','.join('?' for _ in cols)},?)",
                    [item[col] for col in cols] + [revision],
                )
                is_changed = True
            else:
                old_dict = dict(old)
                # Preserve original creation time unless explicitly supplied by a newer source.
                item["created_utc"] = old_dict["created_utc"]
                # updated_utc is store-owned; replaying the same import must not
                # manufacture a new revision.
                item["updated_utc"] = old_dict["updated_utc"]
                if item["status"] == old_dict["status"] and old_dict["completed_utc"]:
                    item["completed_utc"] = old_dict["completed_utc"]
                comparable = {key: item[key] for key in item}
                is_changed = any(old_dict[key] != value for key, value in comparable.items())
                revision, from_status = int(old_dict["revision"]), old_dict["status"]
                if is_changed:
                    revision += 1
                    item["updated_utc"] = now
                    assignments = ",".join(f"{key}=?" for key in item)
                    con.execute(
                        f"UPDATE ops_requirement SET {assignments},revision=? WHERE requirement_id=?",
                        [item[key] for key in item] + [revision, item["requirement_id"]],
                    )
            if is_changed:
                changed += 1
                current = dict(con.execute(
                    "SELECT * FROM ops_requirement WHERE requirement_id=?", (item["requirement_id"],)
                ).fetchone())
                con.execute(
                    "INSERT INTO ops_requirement_history(requirement_id,from_status,to_status,changed_utc,actor,reason,revision,snapshot_json) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (item["requirement_id"], from_status, current["status"], now, actor, reason,
                     current["revision"], canonical_json(requirement_snapshot(current))),
                )
    return len(normalized), changed


def load_document(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def import_requirements(con: sqlite3.Connection, path: Path, actor: str, reason: str) -> tuple[int, int]:
    document, digest = load_document(path)
    rows = document.get("requirements", document) if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("requirements document must be a list or {requirements:[...]}")
    seen, changed = upsert_requirements(con, rows, actor, reason)
    with con:
        con.execute(
            "INSERT INTO ops_import_history(imported_utc,kind,source,source_sha256,records_seen,records_changed,result,details_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (utc_now(), "requirements", str(path), digest, seen, changed, "pass", "{}"),
        )
    return seen, changed


def import_agents(con: sqlite3.Connection, path: Path) -> tuple[int, int]:
    document, digest = load_document(path)
    rows = document.get("agents", document) if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("agents document must be a list or {agents:[...]}")
    now, changed = utc_now(), 0
    with con:
        for raw in rows:
            assignee = str(raw.get("assignee") or raw.get("agent_id") or raw.get("task_name") or "").strip()
            if not assignee:
                raise ValueError("agent record has no assignee/agent_id/task_name")
            requirement_id = raw.get("current_requirement_id") or None
            if requirement_id and not con.execute(
                "SELECT 1 FROM ops_requirement WHERE requirement_id=?", (requirement_id,)
            ).fetchone():
                raise ValueError(f"{assignee}: unknown requirement {requirement_id}")
            values = (
                assignee, str(raw.get("state") or "unknown"), requirement_id,
                str(raw.get("observed_utc") or now), now, str(raw.get("source") or path),
                canonical_json(raw.get("detail", raw)),
            )
            before = con.execute("SELECT * FROM ops_agent_snapshot WHERE assignee=?", (assignee,)).fetchone()
            con.execute(
                "INSERT INTO ops_agent_snapshot(assignee,state,current_requirement_id,observed_utc,imported_utc,source,detail_json) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(assignee) DO UPDATE SET "
                "state=excluded.state,current_requirement_id=excluded.current_requirement_id,observed_utc=excluded.observed_utc," 
                "imported_utc=excluded.imported_utc,source=excluded.source,detail_json=excluded.detail_json",
                values,
            )
            changed += int(before is None or any(dict(before)[key] != value for key, value in zip(
                ("assignee", "state", "current_requirement_id", "observed_utc", "source", "detail_json"),
                (values[0], values[1], values[2], values[3], values[5], values[6]),
            )))
        con.execute(
            "INSERT INTO ops_import_history(imported_utc,kind,source,source_sha256,records_seen,records_changed,result,details_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (now, "agents", str(path), digest, len(rows), changed, "pass", "{}"),
        )
    return len(rows), changed


def import_progress(con: sqlite3.Connection, paths: Iterable[Path], mapping: dict[str, str] | None = None) -> tuple[int, int]:
    mapping, now, changed, seen = mapping or {}, utc_now(), 0, 0
    with con:
        for path in sorted(set(paths)):
            if not path.is_file():
                continue
            seen += 1
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            payload = json.loads(raw)
            requirement_id = mapping.get(str(path)) or mapping.get(path.name)
            if requirement_id and not con.execute(
                "SELECT 1 FROM ops_requirement WHERE requirement_id=?", (requirement_id,)
            ).fetchone():
                raise ValueError(f"{path}: unknown mapped requirement {requirement_id}")
            observed = str(payload.get("updated_utc") or payload.get("timestamp_utc") or payload.get("timestamp") or now)
            state = payload.get("state") or payload.get("status")
            before = con.execute("SELECT sha256 FROM ops_progress_snapshot WHERE progress_path=?", (str(path),)).fetchone()
            con.execute(
                "INSERT INTO ops_progress_snapshot(progress_path,observed_utc,imported_utc,sha256,state,requirement_id,payload_json) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(progress_path) DO UPDATE SET "
                "observed_utc=excluded.observed_utc,imported_utc=excluded.imported_utc,sha256=excluded.sha256," 
                "state=excluded.state,requirement_id=excluded.requirement_id,payload_json=excluded.payload_json",
                (str(path), observed, now, digest, state, requirement_id, canonical_json(payload)),
            )
            changed += int(before is None or before["sha256"] != digest)
        con.execute(
            "INSERT INTO ops_import_history(imported_utc,kind,source,source_sha256,records_seen,records_changed,result,details_json) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (now, "progress", "explicit-paths", None, seen, changed, "pass", canonical_json({"mapping": mapping})),
        )
    return seen, changed


def decode_row(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in tuple(out):
        if key.endswith("_json"):
            try:
                out[key.removesuffix("_json")] = json.loads(out.pop(key))
            except json.JSONDecodeError:
                out[key.removesuffix("_json")] = None
    return out


def read_snapshot(db_path: Path, history_limit: int = 200) -> dict[str, Any]:
    con = connect(db_path, readonly=True)
    try:
        requirements = [decode_row(row) for row in con.execute(
            "SELECT * FROM ops_requirement ORDER BY "
            "CASE status WHEN 'in_progress' THEN 0 WHEN 'blocked' THEN 1 WHEN 'queued' THEN 2 ELSE 3 END," 
            "priority,created_utc,requirement_id"
        )]
        agents = [decode_row(row) for row in con.execute(
            "SELECT * FROM ops_agent_snapshot ORDER BY assignee"
        )]
        history = [decode_row(row) for row in con.execute(
            "SELECT * FROM ops_requirement_history ORDER BY history_id DESC LIMIT ?", (history_limit,)
        )]
        progress_rows = [decode_row(row) for row in con.execute(
            "SELECT progress_path,observed_utc,imported_utc,sha256,state,requirement_id "
            "FROM ops_progress_snapshot ORDER BY imported_utc DESC,progress_path LIMIT 100"
        )]
        imports = [decode_row(row) for row in con.execute(
            "SELECT * FROM ops_import_history ORDER BY import_id DESC LIMIT 50"
        )]
        counts = {row["status"]: row["n"] for row in con.execute(
            "SELECT status,count(*) AS n FROM ops_requirement GROUP BY status"
        )}
        schema = con.execute("SELECT value FROM ops_metadata WHERE key='schema_version'").fetchone()
        return {
            "schema": schema["value"] if schema else None,
            "generated_utc": utc_now(),
            "requirements": requirements,
            "agents": agents,
            "history": history,
            "progress": progress_rows,
            "imports": imports,
            "counts": counts,
        }
    finally:
        con.close()


def audit(db_path: Path) -> dict[str, Any]:
    snapshot = read_snapshot(db_path, history_limit=10000)
    ids = {row["requirement_id"] for row in snapshot["requirements"]}
    errors: list[str] = []
    for row in snapshot["requirements"]:
        if row["parent_id"] and row["parent_id"] not in ids:
            errors.append(f"{row['requirement_id']}: missing parent {row['parent_id']}")
        for dep in row["dependencies"]:
            if dep not in ids:
                errors.append(f"{row['requirement_id']}: missing dependency {dep}")
        if row["status"] == "blocked" and not row["blocked_reason"]:
            errors.append(f"{row['requirement_id']}: blocked without blocked_reason")
        if row["status"] in {"complete_pass", "closed_fail", "closed_fail_superseded", "cancelled"} and not row["completed_utc"]:
            errors.append(f"{row['requirement_id']}: terminal status without completed_utc")
    history_ids = {row["requirement_id"] for row in snapshot["history"]}
    for missing in sorted(ids - history_ids):
        errors.append(f"{missing}: no history row")
    return {
        "schema": "agent-queue-audit-v1093",
        "audited_utc": utc_now(),
        "database": str(db_path),
        "requirements": len(snapshot["requirements"]),
        "history_rows": len(snapshot["history"]),
        "agent_snapshots": len(snapshot["agents"]),
        "progress_snapshots": len(snapshot["progress"]),
        "errors": errors,
        "pass": not errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    seed = sub.add_parser("seed")
    seed.add_argument("--input", type=Path, default=DEFAULT_SEED)
    seed.add_argument("--actor", default="/root/agent_queue_webui")
    seed.add_argument("--reason", default="NR-OPS-001 initial import")
    req = sub.add_parser("import-requirements")
    req.add_argument("--input", type=Path, required=True)
    req.add_argument("--actor", required=True)
    req.add_argument("--reason", required=True)
    agents = sub.add_parser("import-agents")
    agents.add_argument("--input", type=Path, required=True)
    progress = sub.add_parser("import-progress")
    progress.add_argument("paths", nargs="+", type=Path)
    progress.add_argument("--mapping", type=Path)
    sub.add_parser("export")
    sub.add_parser("audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "init":
        con = connect(args.db)
        init_schema(con)
        con.close()
        print(canonical_json({"pass": True, "database": str(args.db), "schema": SCHEMA_VERSION}))
        return 0
    con = connect(args.db)
    init_schema(con)
    if args.command in {"seed", "import-requirements"}:
        seen, changed = import_requirements(con, args.input, args.actor, args.reason)
        result = {"pass": True, "seen": seen, "changed": changed, "database": str(args.db)}
    elif args.command == "import-agents":
        seen, changed = import_agents(con, args.input)
        result = {"pass": True, "seen": seen, "changed": changed, "database": str(args.db)}
    elif args.command == "import-progress":
        mapping = load_document(args.mapping)[0] if args.mapping else {}
        seen, changed = import_progress(con, args.paths, mapping)
        result = {"pass": True, "seen": seen, "changed": changed, "database": str(args.db)}
    elif args.command == "export":
        con.close()
        print(json.dumps(read_snapshot(args.db), ensure_ascii=False, indent=2))
        return 0
    elif args.command == "audit":
        con.close()
        result = audit(args.db)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["pass"] else 1
    else:
        raise AssertionError(args.command)
    con.close()
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
