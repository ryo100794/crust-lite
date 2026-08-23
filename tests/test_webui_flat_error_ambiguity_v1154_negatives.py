from __future__ import annotations

import importlib
import json
import socket
import sqlite3
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/workspace/equake/crust-lite")
sys.path.insert(0, str(ROOT / "scripts"))


class HashConflictTests(unittest.TestCase):
    def setUp(self):
        self.ops = importlib.import_module("operations_status_canonical_v1154")

    def test_object_and_string_conflict_fails_closed_explicitly(self):
        a, b = "a" * 64, "b" * 64
        with self.assertRaisesRegex(self.ops.ArtifactHashConflictError, "one.json"):
            self.ops._normalize_artifact_hashes(
                ["one.json"],
                [{"path": "one.json", "sha256": a}, f"{b}  one.json"],
            )

    def test_duplicate_same_hash_is_unambiguous(self):
        digest = "a" * 64
        self.assertEqual(
            self.ops._normalize_artifact_hashes(
                ["one.json"],
                [{"path": "one.json", "sha256": digest}, f"{digest}  one.json"],
            ),
            {"one.json": digest},
        )

    def test_malformed_hash_is_excluded(self):
        self.assertEqual(
            self.ops._normalize_artifact_hashes(
                ["one.json"], [{"path": "one.json", "sha256": "not-a-digest"}]),
            {},
        )

    def test_model_conflict_returns_failed_without_evidence(self):
        fixture = {
            "ok": True, "schema": "fixture", "stale_after_seconds": 90,
            "requirements": [{
                "requirement_id": "NR-HINET-CONFLICT", "priority": 1,
                "updated_utc": "2026-08-23T12:00:00Z", "status": "in_progress",
                "dispatch_state": "running", "revision": 3,
                "artifact_paths": ["one.json"],
                "artifact_hashes": [
                    {"path": "one.json", "sha256": "a" * 64},
                    {"path": "one.json", "sha256": "b" * 64},
                ],
            }],
            "runtime_agents": [{
                "requirement_id": "NR-HINET-CONFLICT", "state": "running",
                "stale": False, "slot_releasable": False,
                "heartbeat_at": "2026-08-23T12:00:00Z", "heartbeat_age_s": 1,
                "stage": "fixture", "assignee": "z",
            }],
        }
        original = self.ops.queue_payload
        self.ops.queue_payload = lambda *_args, **_kwargs: fixture
        try:
            result = self.ops.build_status()
        finally:
            self.ops.queue_payload = original
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("artifact hash conflict", result["error"])
        self.assertEqual(result["evidence"], [])


class ArtifactErrorPropagationTests(unittest.TestCase):
    def setUp(self):
        self.ops = importlib.import_module("operations_status_canonical_v1154")
        self.dashboard = importlib.import_module("equake_dashboard_10101_canonical_r4_v1154")

    def test_queue_db_error_propagates_to_artifact_payload(self):
        original = self.dashboard.queue_payload
        self.dashboard.queue_payload = lambda: {
            "ok": False, "schema": "fixture-error", "error": "fixture-corrupt"}
        try:
            payload = self.dashboard.artifact_payload("/artifact/current/final")
        finally:
            self.dashboard.queue_payload = original
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "failed")
        self.assertIn("fixture-corrupt", payload["error"])
        self.assertEqual(payload["requirements_with_artifacts"], [])

    def test_conflicting_artifacts_emit_no_rows(self):
        fixture = {
            "ok": True,
            "requirements": [{
                "requirement_id": "NR-X", "status": "complete_pass",
                "artifact_paths": ["one.json"],
                "artifact_hashes": [
                    {"path": "one.json", "sha256": "a" * 64},
                    {"path": "one.json", "sha256": "b" * 64},
                ],
            }],
        }
        original = self.dashboard.queue_payload
        self.dashboard.queue_payload = lambda: fixture
        try:
            payload = self.dashboard.artifact_payload("/artifact/current/final")
        finally:
            self.dashboard.queue_payload = original
        self.assertFalse(payload["ok"])
        self.assertIn("artifact hash conflict", payload["error"])
        self.assertEqual(payload["requirements_with_artifacts"], [])


class IsolatedHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sock = socket.socket(); sock.bind(("127.0.0.1", 0)); cls.port = sock.getsockname()[1]; sock.close()
        cls.process = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"),
             str(ROOT / "scripts/equake_dashboard_10101_canonical_r4_v1154.py"),
             "--host", "127.0.0.1", "--port", str(cls.port)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(200):
            if cls.process.poll() is not None:
                raise RuntimeError(cls.process.stderr.read())
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/api/viewer-diagnostics", timeout=1)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("isolated v1154 dashboard failed to start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate(); cls.process.wait(timeout=5)

    def fetch(self, path: str):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
                return response.status, response.headers.get_content_type(), response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get_content_type(), error.read()

    def test_artifact_api_and_html_are_explicit_normal_payloads(self):
        status, kind, body = self.fetch("/api/artifacts")
        self.assertEqual((status, kind), (200, "application/json"))
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        self.assertIn("requirements_with_artifacts", payload)
        status, kind, body = self.fetch("/artifact/current/final")
        self.assertEqual((status, kind), (200, "text/html"))
        self.assertIn(b'&quot;ok&quot;: true', body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
