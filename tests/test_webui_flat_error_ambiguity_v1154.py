from __future__ import annotations

import ast
import importlib
import json
import re
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path("/workspace/equake/crust-lite")
sys.path.insert(0, str(ROOT / "scripts"))
import operations_status_canonical_v1154 as ops


FORBIDDEN_OVERLAYS = (
    "agent_queue_webui_v1093", "agent_queue_webui_v1097", "agent_queue_webui_v1098",
    "agent_queue_webui_v1100", "agent_queue_webui_v1102", "agent_queue_webui_v1106",
    "model_run_status_v1097", "model_run_status_v1099", "model_run_status_v1105",
)


class FlatStatusTests(unittest.TestCase):
    def test_production_import_graph_has_no_legacy_overlays(self):
        dashboard_path = ROOT / "scripts/equake_dashboard_10101_canonical_r4_v1154.py"
        status_path = ROOT / "scripts/operations_status_canonical_v1154.py"
        source = dashboard_path.read_text() + status_path.read_text()
        for forbidden in FORBIDDEN_OVERLAYS:
            self.assertNotIn(forbidden, source)
        dashboard_tree = ast.parse(dashboard_path.read_text())
        modules = {node.module for node in ast.walk(dashboard_tree)
                   if isinstance(node, ast.ImportFrom) and node.module}
        self.assertIn("operations_status_canonical_v1154", modules)
        self.assertFalse(any("agent_queue_webui" in module or "model_run_status" in module
                             for module in modules))

    def test_no_hardcoded_old_event_or_known_leak_state(self):
        source = (ROOT / "scripts/equake_dashboard_10101_canonical_r4_v1154.py").read_text()
        self.assertIsNone(re.search(r"hinet_[0-9]{14}", source))
        self.assertNotIn("NR-KNOWN-LEAK-008", source)
        self.assertNotIn("DEVELOPMENT_BASE", source)

    def test_actual_queue_and_model_top_level_schema_compatible(self):
        old_queue = importlib.import_module("agent_queue_webui_v1106").api_payload()
        old_model = importlib.import_module("model_run_status_v1105").build_status()
        new_queue = ops.queue_payload()
        new_model = ops.build_status()
        self.assertEqual(set(old_queue), set(new_queue))
        self.assertEqual(set(old_model), set(new_model))
        self.assertTrue(new_queue["ok"])
        self.assertTrue(new_model["ok"])

    def test_fresh_runtime_precedes_stale_even_with_lower_requirement_priority(self):
        requirements = [
            {"requirement_id": "NR-HINET-A", "priority": 0, "updated_utc": "2026-08-23T12:00:00Z"},
            {"requirement_id": "NR-HINET-B", "priority": 9, "updated_utc": "2026-08-23T11:00:00Z"},
        ]
        runtimes = [
            {"requirement_id": "NR-HINET-A", "assignee": "a", "stale": True},
            {"requirement_id": "NR-HINET-B", "assignee": "z", "stale": False},
        ]
        selected = ops._select_runtime(runtimes, {row["requirement_id"]: row for row in requirements})
        self.assertEqual(selected["requirement_id"], "NR-HINET-B")

    def test_multiple_fresh_selects_priority_then_newest_update_not_assignee(self):
        requirements = [
            {"requirement_id": "NR-HINET-A", "priority": 4, "updated_utc": "2026-08-23T12:00:00Z"},
            {"requirement_id": "NR-HINET-B", "priority": 2, "updated_utc": "2026-08-23T10:00:00Z"},
            {"requirement_id": "NR-HINET-C", "priority": 2, "updated_utc": "2026-08-23T13:00:00Z"},
        ]
        runtimes = [
            {"requirement_id": "NR-HINET-A", "assignee": "000", "stale": False},
            {"requirement_id": "NR-HINET-B", "assignee": "111", "stale": False},
            {"requirement_id": "NR-HINET-C", "assignee": "999", "stale": False},
        ]
        selected = ops._select_runtime(runtimes, {row["requirement_id"]: row for row in requirements})
        self.assertEqual(selected["requirement_id"], "NR-HINET-C")

    def test_artifact_hash_objects_strings_and_bare_normalize_by_path(self):
        a, b, c = "a" * 64, "b" * 64, "c" * 64
        paths = ["one.json", "two.json", "three.json"]
        values = [{"path": "two.json", "sha256": b}, f"{a}  one.json", c]
        self.assertEqual(ops._normalize_artifact_hashes(paths, values),
                         {"one.json": a, "two.json": b, "three.json": c})

    def test_build_status_evidence_is_digest_string_and_path_matched(self):
        digest = "d" * 64
        fixture = {
            "ok": True, "schema": "fixture", "stale_after_seconds": 90,
            "requirements": [{
                "requirement_id": "NR-HINET-TEST", "priority": 1,
                "updated_utc": "2026-08-23T12:00:00Z", "status": "in_progress",
                "dispatch_state": "running", "revision": 3, "title": "test", "scope": "test",
                "result": "test", "artifact_paths": ["evidence.json"],
                "artifact_hashes": [{"path": "evidence.json", "sha256": digest}],
            }],
            "runtime_agents": [{
                "requirement_id": "NR-HINET-TEST", "state": "running", "stale": False,
                "slot_releasable": False, "heartbeat_at": "2026-08-23T12:00:00Z",
                "heartbeat_age_s": 1, "stage": "fixture", "assignee": "z",
            }],
        }
        original = ops.queue_payload
        ops.queue_payload = lambda *_args, **_kwargs: fixture
        try:
            result = ops.build_status()
        finally:
            ops.queue_payload = original
        self.assertEqual(result["evidence"], [{"path": "evidence.json", "sha256": digest}])


class BrowserBranchTests(unittest.TestCase):
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
        raise RuntimeError("isolated dashboard failed to start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=5)

    def fetch(self, path: str):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
                return response.status, response.headers.get_content_type(), response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get_content_type(), error.read()

    def test_public_browser_route_matrix(self):
        html_routes = ["/", "/index.html", "/agents-queue", "/viewer-single-event",
                       "/viewer-single-event-audit", "/artifact/current/final"]
        json_routes = ["/api/agent-queue", "/api/model-run", "/api/model-run-status",
                       "/api/model-run/audit", "/api/development-progress",
                       "/api/single-event-gs/meta", "/api/viewer-diagnostics",
                       "/api/viewer-merge-contract", "/api/viewer-lighting-contract",
                       "/api/viewer-background-contract", "/api/viewer-canonical-contract"]
        for path in html_routes:
            status, content_type, body = self.fetch(path)
            self.assertEqual(status, 200, path); self.assertEqual(content_type, "text/html", path)
            self.assertTrue(body, path)
        for path in json_routes:
            status, content_type, body = self.fetch(path)
            self.assertEqual(status, 200, path); self.assertEqual(content_type, "application/json", path)
            json.loads(body)
        for phase in "PSL":
            status, content_type, body = self.fetch(f"/api/single-event-gs/display?phase={phase}")
            self.assertEqual((status, content_type), (200, "application/octet-stream")); self.assertTrue(body)
        self.assertEqual(self.fetch("/api/single-event-gs/display?phase=X")[0], 400)
        self.assertEqual(self.fetch("/not-a-route")[0], 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
