#!/usr/bin/env python3
"""Standalone progressive-viewer selector with live authority revalidation.

This module is deliberately self-contained. Immutable authority, manifest and
evidence files are read-only; only ``runtime_dir`` stores mutable last-good
state and byte-for-byte cache copies.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "viewer-progressive-quality-artifact-v1148"
AUTH_SCHEMA = "viewer-progressive-authority-v1148"
STATE_SCHEMA = "viewer-progressive-last-good-state-flat-v1158"
STATUS_SCHEMA = "viewer-progressive-status-flat-v1158"
REQUIRED = ("meta", "P", "S", "L", "poster")
FULL64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("empty path")
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError("path escapes project root")
    return result


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def artifact_map(paths_raw: str, hashes_raw: str) -> dict[str, str]:
    paths = json.loads(paths_raw)
    hashes = json.loads(hashes_raw)
    if not isinstance(paths, list) or not isinstance(hashes, list):
        raise ValueError("ops artifact arrays invalid")
    if len(paths) != len(set(paths)):
        raise ValueError("ops artifact paths duplicated")
    if hashes and all(isinstance(item, dict) for item in hashes):
        grouped: dict[str, set[str]] = {}
        for item in hashes:
            path, digest = item.get("path"), item.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ValueError("ops artifact path/hash invalid")
            grouped.setdefault(path, set()).add(digest)
        if any(len(values) != 1 for values in grouped.values()):
            raise ValueError("ops artifact path has conflicting hashes")
        mapping = {path: next(iter(values)) for path, values in grouped.items()}
        if set(mapping) != set(paths):
            raise ValueError("ops artifact path/hash mapping differs")
    else:
        if len(paths) != len(hashes):
            raise ValueError("ops artifact path/hash lengths differ")
        mapping = dict(zip(map(str, paths), map(str, hashes)))
    if not mapping or any(not FULL64.fullmatch(value) for value in mapping.values()):
        raise ValueError("ops artifact SHA-256 invalid")
    return mapping


@dataclass(frozen=True)
class Authority:
    path: str
    sha256: str
    requirement_id: str
    ops_revision: int
    issued_at_utc: str
    max_age_seconds: int
    max_future_skew_seconds: int
    approved: dict[str, dict]


@dataclass(frozen=True)
class Snapshot:
    manifest_path: str
    manifest_sha256: str
    artifact_id: str
    generated_at_utc: str
    coverage: dict
    meta: dict
    packets: dict[str, bytes]
    poster: bytes
    authority_requirement_id: str
    authority_sha256: str
    authority_revision: int
    restored_from_cache: bool = False


@dataclass(frozen=True)
class Validated:
    snapshot: Snapshot
    files: dict[str, Path]


def _verify_live_artifacts(root: Path, mapping: dict[str, str]) -> None:
    for relative, expected in mapping.items():
        path = inside(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"registered ops artifact invalid:{relative}")


def load_authority(root: Path, authority_path: Path, ops_db: Path, now: datetime) -> Authority:
    root = root.resolve()
    raw = authority_path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != AUTH_SCHEMA:
        raise ValueError("authority schema invalid")
    authority_sha = hashlib.sha256(raw).hexdigest()
    requirement_id = document.get("authority_requirement_id")
    if not isinstance(requirement_id, str) or not requirement_id:
        raise ValueError("authority requirement missing")
    connection = sqlite3.connect(f"file:{ops_db}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "select * from ops_requirement where requirement_id=?", (requirement_id,)
        ).fetchone()
        history = connection.execute(
            "select * from ops_requirement_history where requirement_id=? order by history_id desc limit 1",
            (requirement_id,),
        ).fetchone()
        expected_prerequisites = document.get("prerequisite_statuses", {})
        if expected_prerequisites:
            placeholders = ",".join("?" for _ in expected_prerequisites)
            prerequisite_rows = {
                item["requirement_id"]: item["status"]
                for item in connection.execute(
                    f"select requirement_id,status from ops_requirement where requirement_id in ({placeholders})",
                    tuple(expected_prerequisites),
                )
            }
        else:
            prerequisite_rows = {}
    finally:
        connection.close()
    if row is None or history is None or row["status"] != "complete_pass":
        raise ValueError("authority ops requirement is not complete_pass")
    if history["to_status"] != "complete_pass" or int(history["revision"]) != int(row["revision"]):
        raise ValueError("authority latest history mismatch")
    snapshot = json.loads(history["snapshot_json"])
    if snapshot.get("status") != "complete_pass" or int(snapshot.get("revision", -1)) != int(row["revision"]):
        raise ValueError("authority history snapshot mismatch")
    if snapshot.get("artifact_paths") != json.loads(row["artifact_paths_json"]):
        raise ValueError("authority history artifact paths mismatch")
    if snapshot.get("artifact_hashes") != json.loads(row["artifact_hashes_json"]):
        raise ValueError("authority history artifact hashes mismatch")
    if prerequisite_rows != expected_prerequisites:
        raise ValueError("authority prerequisite status mismatch")
    artifacts = artifact_map(row["artifact_paths_json"], row["artifact_hashes_json"])
    authority_relative = str(authority_path.resolve().relative_to(root))
    if artifacts.get(authority_relative) != authority_sha:
        raise ValueError("authority file is not exact registered ops artifact")
    _verify_live_artifacts(root, artifacts)
    future = int(document.get("max_future_skew_seconds", -1))
    age = int(document.get("max_age_seconds", -1))
    if not 0 <= future <= 900 or not 1 <= age <= 31 * 86400:
        raise ValueError("authority time bounds invalid")
    issued = utc(document["issued_at_utc"])
    delta = (now - issued).total_seconds()
    if delta < -future:
        raise ValueError("authority issued in future")
    if delta > age:
        raise ValueError("authority stale")
    completed = utc(row["completed_utc"])
    if (completed - now).total_seconds() > future:
        raise ValueError("authority ops completion in future")
    approved_items = document.get("approved_manifests")
    if not isinstance(approved_items, list) or not approved_items:
        raise ValueError("authority approval list empty")
    approved: dict[str, dict] = {}
    for item in approved_items:
        relative, digest = item.get("path"), item.get("sha256")
        if not isinstance(relative, str) or not FULL64.fullmatch(str(digest)) or relative in approved:
            raise ValueError("authority approval invalid")
        approved[relative] = item
    return Authority(
        authority_relative, authority_sha, requirement_id, int(row["revision"]),
        document["issued_at_utc"], age, future, approved,
    )


def _read_file(
    root: Path,
    relative: str,
    expected: str,
    expected_bytes: int | None,
    override: Path | None,
) -> tuple[bytes, Path]:
    if not FULL64.fullmatch(str(expected)):
        raise ValueError("file SHA-256 invalid")
    path = override or inside(root, relative)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"file hash invalid:{relative}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"file size invalid:{relative}")
    return path.read_bytes(), path


def validate_manifest(
    root: Path,
    manifest_path: Path,
    authority: Authority,
    now: datetime,
    overrides: dict[str, Path] | None = None,
    logical_path: str | None = None,
    restored: bool = False,
) -> Validated:
    overrides = overrides or {}
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    relative = logical_path or str(manifest_path.resolve().relative_to(root))
    digest = hashlib.sha256(raw).hexdigest()
    approval = authority.approved.get(relative)
    if approval is None or approval.get("sha256") != digest:
        raise ValueError("manifest is not authority-approved")
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete" or manifest.get("quality_pass") is not True:
        raise ValueError("manifest not complete quality-pass")
    generated = utc(manifest["generated_at_utc"])
    delta = (now - generated).total_seconds()
    if delta < -authority.max_future_skew_seconds:
        raise ValueError("manifest generated in future")
    if delta > authority.max_age_seconds:
        raise ValueError("manifest stale")
    for key in ("artifact_id", "generated_at_utc", "coverage", "source_model", "quality_audit"):
        if manifest.get(key) != approval.get(key):
            raise ValueError(f"manifest authority binding mismatch:{key}")
    coverage = manifest["coverage"]
    if not isinstance(coverage, dict) or coverage.get("event_count") != len(coverage.get("event_ids", [])):
        raise ValueError("coverage invalid")
    if coverage.get("nationwide") is not False:
        raise ValueError("fabricated nationwide coverage forbidden")
    assets = manifest.get("assets", {})
    if set(assets) != set(REQUIRED) or assets != approval.get("assets"):
        raise ValueError("asset authority binding mismatch")
    blobs: dict[str, bytes] = {}
    files: dict[str, Path] = {"manifest": manifest_path}
    for key in REQUIRED:
        item = assets[key]
        blobs[key], files[key] = _read_file(
            root, item["path"], item["sha256"], item["bytes"], overrides.get(key)
        )
    quality = manifest["quality_audit"]
    quality_raw, files["quality_audit"] = _read_file(
        root, quality["path"], quality["sha256"], None, overrides.get("quality_audit")
    )
    audit = json.loads(quality_raw)
    checks = audit.get("checks", {})
    if audit.get("pass") is not True or not checks or not all(value is True for value in checks.values()):
        raise ValueError("quality audit not pass")
    if checks.get("machine_learning_absent") is not True or checks.get("known_structure_absent") is not True:
        raise ValueError("quality audit forbidden input")
    source = manifest["source_model"]
    _, files["source_model"] = _read_file(
        root, source["path"], source["sha256"], None, overrides.get("source_model")
    )
    meta = json.loads(blobs["meta"])
    if meta.get("quality_pass") is not True or meta.get("event_id") not in coverage["event_ids"]:
        raise ValueError("meta quality/coverage mismatch")
    snapshot = Snapshot(
        relative, digest, manifest["artifact_id"], manifest["generated_at_utc"], coverage,
        meta, {key: blobs[key] for key in ("P", "S", "L")}, blobs["poster"],
        authority.requirement_id, authority.sha256, authority.ops_revision, restored,
    )
    return Validated(snapshot, files)


class Selector:
    """Two-phase refresh selector with a live authority gate on every read."""

    def __init__(
        self,
        root,
        manifest_dir,
        authority_path,
        ops_db,
        runtime_dir,
        now=None,
    ):
        self.root = Path(root).resolve()
        self.manifest_dir = Path(manifest_dir)
        self.authority_path = Path(authority_path)
        self.ops_db = Path(ops_db)
        self.runtime_dir = Path(runtime_dir)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.lock = threading.RLock()
        self.snapshot: Snapshot | None = None
        self.rejections: list[dict] = []
        self._last_gate_time: datetime | None = None
        self._state_document: dict | None = None
        self._restart_clock_invalid = False
        try:
            self._restore()
        except Exception as error:
            if "clock rollback across restart" in str(error):
                self._restart_clock_invalid = True
            self.rejections.append({
                "path": "mutable:last_good_state",
                "reason": f"{type(error).__name__}:{error}",
            })
        self.refresh(required=True)

    @property
    def state_path(self) -> Path:
        return self.runtime_dir / "last_good_state.json"

    def _clock(self) -> datetime:
        value = self._now()
        if value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
            raise ValueError("runtime clock must be UTC")
        return value.astimezone(timezone.utc)

    def _authority_at(self, now: datetime) -> Authority:
        return load_authority(self.root, self.authority_path, self.ops_db, now)

    def _authority(self) -> Authority:
        return self._authority_at(self._clock())

    def _invalidate_locked(self, path: str, error: Exception) -> None:
        self.snapshot = None
        self.rejections.append({"path": path, "reason": f"{type(error).__name__}:{error}"})

    def _write_state_locked(self, state: dict) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name + ".tmp-" + uuid.uuid4().hex)
        temporary.write_text(canonical(state) + "\n")
        os.replace(temporary, self.state_path)
        self._state_document = dict(state)

    def _record_gate_time_locked(self, now: datetime) -> None:
        if not self._state_document or self.snapshot is None:
            return
        state = dict(self._state_document)
        if state.get("manifest_sha256") != self.snapshot.manifest_sha256 or state.get("authority_revision") != self.snapshot.authority_revision:
            return
        state["last_validated_at_utc"] = now.isoformat().replace("+00:00", "Z")
        try:
            self._write_state_locked(state)
        except Exception as error:
            self.rejections.append({
                "path": "mutable:last_good_state",
                "reason": f"{type(error).__name__}:{error}",
            })

    def _gate_locked(self, expected: Snapshot | None = None) -> tuple[Snapshot, Authority, datetime]:
        try:
            now = self._clock()
            if self._last_gate_time is not None and now < self._last_gate_time:
                raise ValueError("runtime clock rollback")
            authority = self._authority_at(now)
            snapshot = self.snapshot
            if snapshot is None:
                raise RuntimeError("blank state forbidden")
            if expected is not None and snapshot is not expected:
                raise RuntimeError("snapshot changed during authorized read")
            if (
                snapshot.authority_requirement_id != authority.requirement_id
                or snapshot.authority_sha256 != authority.sha256
                or snapshot.authority_revision != authority.ops_revision
            ):
                raise ValueError("active snapshot authority binding mismatch")
            delta = (now - utc(snapshot.generated_at_utc)).total_seconds()
            if delta < -authority.max_future_skew_seconds:
                raise ValueError("active snapshot generated in future")
            if delta > authority.max_age_seconds:
                raise ValueError("active snapshot stale")
            self._last_gate_time = now
            self._record_gate_time_locked(now)
            return snapshot, authority, now
        except Exception as error:
            self._invalidate_locked(str(self.authority_path), error)
            raise RuntimeError("active authority invalid; snapshot atomically invalidated") from error

    def _restore(self) -> None:
        state = json.loads(self.state_path.read_text())
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("last-good state schema invalid")
        now = self._clock()
        last_validated = utc(state["last_validated_at_utc"])
        if now < last_validated:
            raise ValueError("runtime clock rollback across restart")
        authority = self._authority_at(now)
        if (
            state.get("authority_sha256") != authority.sha256
            or state.get("authority_requirement_id") != authority.requirement_id
            or int(state.get("authority_revision", -1)) != authority.ops_revision
        ):
            raise ValueError("last-good authority mismatch")
        cache_dir = inside(self.root, state["cache_dir"])
        manifest = cache_dir / "manifest.json"
        names = {key: cache_dir / f"{key}.cache" for key in (*REQUIRED, "quality_audit", "source_model")}
        validated = validate_manifest(
            self.root, manifest, authority, now, names, state["manifest_path"], True
        )
        snapshot = validated.snapshot
        if (
            snapshot.manifest_sha256 != state.get("manifest_sha256")
            or snapshot.artifact_id != state.get("artifact_id")
            or snapshot.generated_at_utc != state.get("generated_at_utc")
        ):
            raise ValueError("last-good state binding mismatch")
        self.snapshot = snapshot
        self._last_gate_time = last_validated
        self._state_document = dict(state)

    def _persist(self, validated: Validated, authority: Authority, validated_at: datetime) -> None:
        snapshot = validated.snapshot
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        cache_root = self.runtime_dir / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cache_root / (".tmp-" + uuid.uuid4().hex)
        final = cache_root / (snapshot.manifest_sha256 + "-" + uuid.uuid4().hex)
        temporary.mkdir()
        shutil.copyfile(validated.files["manifest"], temporary / "manifest.json")
        for key in (*REQUIRED, "quality_audit", "source_model"):
            shutil.copyfile(validated.files[key], temporary / f"{key}.cache")
        os.replace(temporary, final)
        state = {
            "schema": STATE_SCHEMA,
            "authority_requirement_id": authority.requirement_id,
            "authority_sha256": authority.sha256,
            "authority_revision": authority.ops_revision,
            "last_validated_at_utc": validated_at.isoformat().replace("+00:00", "Z"),
            "manifest_path": snapshot.manifest_path,
            "manifest_sha256": snapshot.manifest_sha256,
            "artifact_id": snapshot.artifact_id,
            "generated_at_utc": snapshot.generated_at_utc,
            "cache_dir": str(final.relative_to(self.root)),
        }
        self._write_state_locked(state)

    def candidates(self) -> list[Path]:
        return sorted(self.manifest_dir.glob("*.json"))

    def refresh(self, required: bool = False) -> Snapshot | None:
        rejected: list[dict] = []
        valid: list[Validated] = []
        try:
            if self._restart_clock_invalid:
                raise ValueError("runtime clock rollback across restart")
            scan_now = self._clock()
            scan_authority = self._authority_at(scan_now)
            for path in self.candidates():
                try:
                    valid.append(validate_manifest(self.root, path, scan_authority, scan_now))
                except Exception as error:
                    rejected.append({"path": str(path), "reason": f"{type(error).__name__}:{error}"})
        except Exception as error:
            with self.lock:
                self._invalidate_locked(str(self.authority_path), error)
                self.rejections.extend(rejected)
            if required:
                raise RuntimeError("no authoritative fresh artifact or restart-readable last-good") from error
            return None
        valid.sort(key=lambda item: (
            utc(item.snapshot.generated_at_utc), item.snapshot.artifact_id, item.snapshot.manifest_sha256,
        ))
        with self.lock:
            try:
                commit_now = self._clock()
                if commit_now < scan_now:
                    raise ValueError("runtime clock rollback during refresh")
                if self._last_gate_time is not None and commit_now < self._last_gate_time:
                    raise ValueError("runtime clock rollback")
                commit_authority = self._authority_at(commit_now)
                if commit_authority != scan_authority:
                    raise ValueError("authority changed during refresh")
                chosen = valid[-1] if valid else None
                if chosen is not None:
                    delta = (commit_now - utc(chosen.snapshot.generated_at_utc)).total_seconds()
                    if delta < -commit_authority.max_future_skew_seconds:
                        raise ValueError("candidate generated in future at commit")
                    if delta > commit_authority.max_age_seconds:
                        raise ValueError("candidate stale at commit")
            except Exception as error:
                self._invalidate_locked(str(self.authority_path), error)
                self.rejections.extend(rejected)
                if required:
                    raise RuntimeError("no authoritative fresh artifact or restart-readable last-good") from error
                return None
            current = self.snapshot
            if chosen and (
                current is None or utc(chosen.snapshot.generated_at_utc) >= utc(current.generated_at_utc)
            ):
                changed = (
                    current is None
                    or chosen.snapshot.manifest_sha256 != current.manifest_sha256
                    or current.authority_revision != commit_authority.ops_revision
                )
                self.snapshot = chosen.snapshot
                self._last_gate_time = commit_now
                if changed:
                    self._persist(chosen, commit_authority, commit_now)
            elif current is not None:
                try:
                    self._gate_locked(current)
                except RuntimeError:
                    pass
            self.rejections.extend(rejected)
            if self.snapshot is None and required:
                raise RuntimeError("no authoritative fresh artifact or restart-readable last-good")
            return self.snapshot

    def get(self) -> Snapshot:
        with self.lock:
            snapshot, _, _ = self._gate_locked()
            return snapshot

    def confirm(self, expected: Snapshot) -> None:
        with self.lock:
            self._gate_locked(expected)

    @contextmanager
    def authorized_snapshot(self):
        with self.lock:
            snapshot, _, _ = self._gate_locked()
            yield snapshot

    def status_for(self, snapshot: Snapshot) -> dict:
        return {
            "schema": STATUS_SCHEMA,
            "state": "ready-last-good",
            "artifact_id": snapshot.artifact_id,
            "generated_at_utc": snapshot.generated_at_utc,
            "coverage": snapshot.coverage,
            "manifest_path": snapshot.manifest_path,
            "manifest_sha256": snapshot.manifest_sha256,
            "authority_requirement_id": snapshot.authority_requirement_id,
            "authority_sha256": snapshot.authority_sha256,
            "authority_revision": snapshot.authority_revision,
            "restored_from_cache": snapshot.restored_from_cache,
            "rejections": list(self.rejections),
            "blank": False,
        }

    def status(self) -> dict:
        with self.authorized_snapshot() as snapshot:
            result = self.status_for(snapshot)
            self.confirm(snapshot)
            return result

    def preload(self, interval: float = 5.0, stop=None) -> None:
        stop = stop or threading.Event()
        while not stop.wait(interval):
            try:
                self.refresh()
            except Exception:
                pass


__all__ = [
    "Selector", "Authority", "Snapshot", "Validated", "canonical",
    "load_authority", "validate_manifest",
]
