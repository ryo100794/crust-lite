#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import viewer_progressive_selector_flat_v1158 as implementation
from viewer_progressive_selector_flat_v1158 import Selector, canonical
import equake_dashboard_10101_progressive_flat_v1158 as dashboard

NOW = datetime(2026, 8, 23, 17, 15, tzinfo=timezone.utc)
REQ = "NR-VIEWER-PROGRESSIVE-AUTHORITY-057"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root: Path, generated="2026-08-23T16:00:00Z", nationwide=False):
        self.root = root
        self.manifest_dir = root / "manifests"; self.manifest_dir.mkdir(parents=True)
        bundle = root / "bundle"; bundle.mkdir()
        self.files = {
            "meta": bundle / "meta.json", "P": bundle / "P.bin", "S": bundle / "S.bin",
            "L": bundle / "L.bin", "poster": bundle / "poster.png",
        }
        self.files["meta"].write_text(json.dumps({"event_id": "hinet_event", "quality_pass": True}))
        for key in ("P", "S", "L", "poster"):
            self.files[key].write_bytes((key + "-trusted").encode())
        self.source = bundle / "source.npz"; self.source.write_bytes(b"formal-model")
        self.audit = bundle / "quality.audit.json"
        self.audit.write_text(json.dumps({
            "pass": True,
            "checks": {"finite": True, "machine_learning_absent": True, "known_structure_absent": True},
        }))
        coverage = {
            "scope": "test-single-event", "geographic": "bounded test",
            "event_ids": ["hinet_event"], "event_count": 1, "station_count": 48,
            "phase_packets": ["P", "S", "L"], "nationwide": nationwide,
        }
        assets = {
            key: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for key, path in self.files.items()
        }
        manifest = {
            "schema": "viewer-progressive-quality-artifact-v1148",
            "artifact_id": "trusted-artifact", "generated_at_utc": generated,
            "state": "complete", "quality_pass": True, "coverage": coverage,
            "source_model": {"path": str(self.source.relative_to(root)), "sha256": digest(self.source)},
            "quality_audit": {"path": str(self.audit.relative_to(root)), "sha256": digest(self.audit)},
            "assets": assets,
        }
        self.manifest = self.manifest_dir / "trusted.json"
        self.manifest.write_text(canonical(manifest) + "\n")
        approval = {
            "path": str(self.manifest.relative_to(root)), "sha256": digest(self.manifest),
            "artifact_id": manifest["artifact_id"], "generated_at_utc": generated,
            "coverage": coverage, "source_model": manifest["source_model"],
            "quality_audit": manifest["quality_audit"], "assets": assets,
        }
        authority = {
            "schema": "viewer-progressive-authority-v1148",
            "authority_requirement_id": REQ,
            "issued_at_utc": "2026-08-23T17:00:00Z",
            "max_age_seconds": 604800,
            "max_future_skew_seconds": 300,
            "prerequisite_statuses": {
                "NR-VIEWER-PROGRESSIVE-015": "complete_pass",
                "NR-VIEWER-PROGRESSIVE-INDEPENDENT-055": "closed_fail",
            },
            "approved_manifests": [approval],
        }
        self.authority = root / "authority.json"
        self.authority.write_text(canonical(authority) + "\n")
        self.ops = root / "ops.sqlite"
        self._ops("complete_pass")
        self.runtime = root / "runtime"

    def _ops(self, status):
        if self.ops.exists():
            self.ops.unlink()
        connection = sqlite3.connect(self.ops)
        connection.executescript("""
        CREATE TABLE ops_requirement(requirement_id TEXT PRIMARY KEY,parent_id TEXT,title TEXT,evidence_json TEXT,scope TEXT,priority INTEGER,status TEXT,assignee TEXT,dependencies_json TEXT,blocked_reason TEXT,created_utc TEXT,updated_utc TEXT,completed_utc TEXT,artifact_paths_json TEXT,artifact_hashes_json TEXT,acceptance TEXT,result TEXT,source TEXT,revision INTEGER);
        CREATE TABLE ops_requirement_history(history_id INTEGER PRIMARY KEY,requirement_id TEXT,from_status TEXT,to_status TEXT,changed_utc TEXT,actor TEXT,reason TEXT,revision INTEGER,snapshot_json TEXT);
        """)
        prerequisites = [
            ("NR-VIEWER-PROGRESSIVE-015", "complete_pass"),
            ("NR-VIEWER-PROGRESSIVE-INDEPENDENT-055", "closed_fail"),
        ]
        for requirement_id, state in prerequisites:
            connection.execute(
                "INSERT INTO ops_requirement VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (requirement_id, None, requirement_id, "[]", "test", 1, state, "other", "[]", None,
                 "2026-08-23T16:00:00Z", "2026-08-23T16:01:00Z", "2026-08-23T16:01:00Z",
                 "[]", "[]", "x", "x", "test", 1),
            )
        paths = [str(self.authority.relative_to(self.root))]
        hashes = [digest(self.authority)]
        connection.execute(
            "INSERT INTO ops_requirement VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (REQ, None, REQ, "[]", "test", 1, status, "implementation-owner", "[]", None,
             "2026-08-23T16:00:00Z", "2026-08-23T17:01:00Z", "2026-08-23T17:01:00Z",
             json.dumps(paths), json.dumps(hashes), "x", "x", "test", 3),
        )
        snapshot = {"status": status, "revision": 3, "artifact_paths": paths, "artifact_hashes": hashes}
        connection.execute(
            "INSERT INTO ops_requirement_history VALUES(?,?,?,?,?,?,?,?,?)",
            (1, REQ, "in_progress", status, "2026-08-23T17:01:00Z", "/root", "formal import", 3, canonical(snapshot)),
        )
        connection.commit(); connection.close()

    def selector(self, clock=None):
        clock = clock or [NOW]
        return Selector(
            self.root, self.manifest_dir, self.authority, self.ops, self.runtime,
            now=lambda: clock[0],
        )


class ValidationTests(unittest.TestCase):
    def test_authoritative_exact_bundle_and_status(self):
        with tempfile.TemporaryDirectory() as value:
            current = Fixture(Path(value)).selector()
            self.assertEqual(current.get().artifact_id, "trusted-artifact")
            self.assertEqual(current.status()["schema"], "viewer-progressive-status-flat-v1158")
            self.assertFalse(current.status()["blank"])

    def test_fabricated_nationwide_future_stale_partial_and_low_quality_reject(self):
        for case in ("nationwide", "future", "stale", "partial", "low_quality"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as value:
                generated = "2026-08-23T18:00:01Z" if case == "future" else "2026-08-01T00:00:00Z" if case == "stale" else "2026-08-23T16:00:00Z"
                fixture = Fixture(Path(value), generated=generated, nationwide=case == "nationwide")
                if case == "partial":
                    fixture.files["S"].unlink()
                if case == "low_quality":
                    fixture.audit.write_text(json.dumps({"pass": False, "checks": {"finite": True}}))
                with self.assertRaisesRegex(RuntimeError, "no authoritative"):
                    fixture.selector()

    def test_authority_tamper_and_failed_ops_fail_closed(self):
        for tamper in (True, False):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as value:
                fixture = Fixture(Path(value))
                if tamper:
                    fixture.authority.write_text(fixture.authority.read_text() + " ")
                else:
                    fixture._ops("closed_fail")
                with self.assertRaisesRegex(RuntimeError, "no authoritative"):
                    fixture.selector()

    def test_conflicting_duplicate_ops_hashes_reject(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value))
            connection = sqlite3.connect(fixture.ops)
            hashes = [
                {"path": str(fixture.authority.relative_to(fixture.root)), "sha256": "0" * 64},
                {"path": str(fixture.authority.relative_to(fixture.root)), "sha256": digest(fixture.authority)},
            ]
            connection.execute("update ops_requirement set artifact_hashes_json=? where requirement_id=?", (json.dumps(hashes), REQ))
            snapshot = json.loads(connection.execute("select snapshot_json from ops_requirement_history where requirement_id=?", (REQ,)).fetchone()[0])
            snapshot["artifact_hashes"] = hashes
            connection.execute("update ops_requirement_history set snapshot_json=? where requirement_id=?", (json.dumps(snapshot), REQ))
            connection.commit(); connection.close()
            with self.assertRaisesRegex(RuntimeError, "no authoritative"):
                fixture.selector()


class RuntimeAuthorityTests(unittest.TestCase):
    def test_refresh_and_get_revocation_atomically_invalidate(self):
        for route in ("refresh", "get"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as value:
                fixture = Fixture(Path(value)); current = fixture.selector(); fixture._ops("closed_fail")
                with self.assertRaises(RuntimeError):
                    current.refresh(required=True) if route == "refresh" else current.get()
                self.assertIsNone(current.snapshot)

    def test_expiry_and_clock_rollback_invalidate(self):
        for target in (NOW + timedelta(days=8), NOW - timedelta(microseconds=1)):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as value:
                fixture = Fixture(Path(value)); clock = [NOW]; current = fixture.selector(clock); current.get(); clock[0] = target
                with self.assertRaisesRegex(RuntimeError, "atomically invalidated"):
                    current.get()
                self.assertIsNone(current.snapshot)

    def test_clock_rollback_across_restart_fails_closed(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); fixture.selector([NOW]).get()
            with self.assertRaisesRegex(RuntimeError, "no authoritative"):
                fixture.selector([NOW - timedelta(microseconds=1)])

    def test_revision_history_mismatch_invalidates(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); current = fixture.selector()
            connection = sqlite3.connect(fixture.ops)
            connection.execute("update ops_requirement set revision=revision+1 where requirement_id=?", (REQ,))
            connection.commit(); connection.close()
            with self.assertRaisesRegex(RuntimeError, "atomically invalidated"):
                current.get()

    def test_response_commit_closes_toctou(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); current = fixture.selector()
            with current.authorized_snapshot() as snapshot:
                self.assertEqual(snapshot.packets["P"], b"P-trusted")
                fixture._ops("closed_fail")
                with self.assertRaisesRegex(RuntimeError, "atomically invalidated"):
                    current.confirm(snapshot)
            self.assertIsNone(current.snapshot)

    def test_restart_origin_then_cache_last_good(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); first = fixture.selector(); expected = first.get().packets["P"]
            fixture.manifest.write_text("{corrupt"); fixture.files["P"].write_bytes(b"tampered")
            restarted = fixture.selector()
            self.assertEqual(restarted.get().packets["P"], expected)
            self.assertTrue(restarted.status()["restored_from_cache"])

    def test_invalid_authority_forbids_cached_last_good(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); fixture.selector().get()
            fixture.manifest.write_text("{corrupt"); fixture.files["P"].write_bytes(b"tampered"); fixture._ops("closed_fail")
            with self.assertRaisesRegex(RuntimeError, "no authoritative"):
                fixture.selector()

    def test_corrupt_mutable_state_uses_valid_origin_and_ledger_excludes_runtime(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); fixture.selector().get()
            fixture.runtime.joinpath("last_good_state.json").write_text("{corrupt")
            restarted = fixture.selector()
            self.assertFalse(restarted.status()["restored_from_cache"])
            connection = sqlite3.connect(fixture.ops)
            paths = json.loads(connection.execute("select artifact_paths_json from ops_requirement where requirement_id=?", (REQ,)).fetchone()[0])
            connection.close()
            self.assertFalse(any("runtime" in path or "last_good" in path for path in paths))


class ConcurrencyTests(unittest.TestCase):
    def test_heavy_validation_off_lock_and_commit_authority_recheck(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); current = fixture.selector()
            original = implementation.validate_manifest
            entered = threading.Event(); release = threading.Event(); errors = []
            def delayed(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("release timeout")
                return original(*args, **kwargs)
            implementation.validate_manifest = delayed
            worker = threading.Thread(target=lambda: current.refresh(required=True), name="flat-heavy-refresh")
            try:
                worker.start(); self.assertTrue(entered.wait(2))
                self.assertEqual(current.get().artifact_id, "trusted-artifact")
            except Exception as error:
                errors.append(error)
            finally:
                release.set(); worker.join(5); implementation.validate_manifest = original
            self.assertFalse(worker.is_alive())
            if errors:
                raise errors[0]

    def test_preload_background_preserves_nonblank_reader(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); current = fixture.selector(); stop = threading.Event()
            worker = threading.Thread(target=current.preload, kwargs={"interval": 0.01, "stop": stop})
            worker.start()
            try:
                for _ in range(10):
                    self.assertFalse(current.status()["blank"]); time.sleep(0.005)
            finally:
                stop.set(); worker.join(2)
            self.assertFalse(worker.is_alive())


class DashboardTests(unittest.TestCase):
    def test_http_routes_and_revocation_503(self):
        with tempfile.TemporaryDirectory() as value:
            fixture = Fixture(Path(value)); dashboard.SELECTOR = fixture.selector()
            server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
            worker = threading.Thread(target=server.serve_forever); worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                for route, expected in (
                    ("/api/single-event-gs/meta", 200),
                    ("/api/single-event-gs/display?phase=P", 200),
                    ("/api/single-event-gs/display?phase=S", 200),
                    ("/api/single-event-gs/display?phase=L", 200),
                    ("/api/viewer-progressive-status", 200),
                    ("/artifacts/japan-v987-reference.png", 200),
                ):
                    with urllib.request.urlopen(base + route, timeout=3) as response:
                        self.assertEqual(response.status, expected)
                fixture._ops("closed_fail")
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + "/api/viewer-progressive-status", timeout=3)
                self.assertEqual(caught.exception.code, 503)
            finally:
                server.shutdown(); server.server_close(); worker.join(3); dashboard.SELECTOR = None


if __name__ == "__main__":
    unittest.main(verbosity=2)
