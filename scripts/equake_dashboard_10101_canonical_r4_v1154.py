#!/usr/bin/env python3
"""Isolated canonical dashboard with a flat durable-operations dependency graph."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from operations_status_canonical_v1154 import (
    ArtifactHashConflictError, build_status, queue_payload, _normalize_artifact_hashes,
)
from viewer_10101_packet_adapter_v1120 import load_meta, packet


ROOT_PAGE = HERE / "root_10101_canonical_v1120.html"
QUEUE_PAGE = HERE / "queue_10101_canonical_v1120.html"
VIEWER_PAGE = HERE / "viewer_10101_canonical_v1120_r4.html"
VIEWER_PAGE_EXPECTED_SHA256 = "621162ff53781a4314d66d477ecc590475d1d7c99d80de3ef9053f2e973a9d36"
POSTER = PROJECT / "outputs/3d/japan_v987_check.png"
ARTIFACT_ROUTES = {
    "/artifact/storage/rdb-inventory", "/artifact/input/acquisition",
    "/artifact/input/completeness", "/artifact/input/phase",
    "/artifact/input/catalog-metadata", "/artifact/input/station-groups",
    "/artifact/design/split", "/artifact/design/grid", "/artifact/gpu/validation",
    "/artifact/current/full48", "/artifact/current/radial", "/artifact/current/core",
    "/artifact/current/fusion", "/artifact/current/evidence", "/artifact/current/final",
}

MERGE_CONTRACT = {
    "schema": "viewer-merge-contract-canonical-v1132",
    "lighting": {"model": "two-sided-wrap-diffuse", "defaults": {
        "ambient": 0.46, "wrap": 0.55, "roughness": 0.78, "azimuth_deg": -35.0,
        "elevation_deg": 50.0, "specular_intensity": 0.055,
        "rim_intensity": 0.055, "basis_weight": 0.22,
    }},
    "background": {"default": "light", "persistent_key": "equake-viewer-background",
                   "light_clear_color": [0.82, 0.88, 0.91, 1.0],
                   "dark_clear_color": [0.008, 0.025, 0.04, 1.0]},
    "unchanged": ["phase packets", "strength", "opacity expression", "geometry",
                  "centres", "covariance vectors", "P/S/L colours", "threshold",
                  "camera", "land opacity", "gesture", "static poster bytes"],
}

CAMERA_DIAGNOSTICS = {
    "schema": "viewer-diagnostics-contract-canonical-v1132",
    "privacy": {"client_telemetry_collected": False, "request_headers_recorded": False,
                "user_agent_recorded": False, "ip_recorded": False,
                "diagnostics_location": "browser-local window.__viewerDiagnostics"},
    "renderer_order": ["webgl2", "webgl1+ANGLE_instanced_arrays", "static-audit-image"],
    "forced_test_paths": {"webgl2": "/viewer-single-event?renderer=webgl2",
                          "webgl1": "/viewer-single-event?renderer=webgl1",
                          "context_none": "/viewer-single-event?renderer=none",
                          "auto": "/viewer-single-event?renderer=auto"},
    "static_fallback": "/artifacts/japan-v987-reference.png",
    "data_apis": ["/api/single-event-gs/meta",
                  "/api/single-event-gs/display?model=single&phase=P",
                  "/api/single-event-gs/display?model=single&phase=S",
                  "/api/single-event-gs/display?model=single&phase=L"],
    "interaction": {"one_pointer": "arcball rotation",
                    "two_pointer": "centroid pan plus pinch zoom",
                    "pinch_out": "projected screen bbox grows",
                    "pinch_in": "projected screen bbox shrinks",
                    "wheel_negative_delta": "projected screen bbox grows",
                    "mode_transition": "rebase on every pointer add/remove; no state mutation"},
    "mobile_controls": ["display settings", "quality", "top", "oblique", "reset"],
    "camera": {"ui_quantity": "screen magnification", "default_magnification": 1.0,
               "maximum_magnification": 15.2,
               "internal_projection_denominator": {"min": 0.05, "max": 5.0, "default": 0.76},
               "single_clamp_function": "clampZoom"},
    "land_layer": {"control_label": "日本列島 不透明度", "default_opacity": 0.08,
                   "independent_uniform": "uLayerAlpha", "P_alpha": 1.0, "S_alpha": 1.0,
                   "draw_order": ["L", "P", "S"], "land_depth_write": False},
    "viewer_merge": MERGE_CONTRACT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_viewer_bytes() -> bytes:
    body = VIEWER_PAGE.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != VIEWER_PAGE_EXPECTED_SHA256:
        raise RuntimeError(f"canonical r4 viewer SHA-256 mismatch: expected={VIEWER_PAGE_EXPECTED_SHA256} actual={actual}")
    return body


def authoritative_progress() -> dict:
    model = build_status()
    work = model.get("work") or {}
    units = work.get("unit") or {"completed": 0, "total": 0, "remaining": 0}
    status = model.get("status", "failed")
    active = model.get("active_process") is True
    return {
        "schema": "operations-derived-development-progress-v1132",
        "development_completed": units.get("completed", 0),
        "development_total": units.get("total", 0),
        "completed_events": [],
        "current_event": work.get("event_id", "none"),
        "P_isochron": {"state": status, "fraction": None, "station_current": None,
                       "stations_total": None},
        "S_isochron": {"state": status, "fraction": None, "station_current": None,
                       "stations_total": None},
        "manifests_ready": bool(model.get("evidence")),
        "gpu_pipeline": {"schema": "operations-derived-runtime", "state": status,
                         "event_id": work.get("event_id", "none"), "tasks_total": None,
                         "fraction": None, "audit": model.get("raw_audit_url"),
                         "pass": status == "running", "quality_pass_tiles": None,
                         "publication_allowed": False, "historical": not active},
        "stage_pass": {"phase_manifests": None, "full48_gpu": None, "event_gs": None,
                       "ps_covariance_fusion": None, "final_quality": None},
        "latest_preview_event": "none", "latest_preview_quality_pass": None,
        "last_completed_event": "none", "blind_events_touched": None,
        "retuning_on_blind_allowed": False, "publication_allowed": False,
        "model_run": {"schema": model.get("schema"), "status": status,
                      "active_process": active, "heartbeat": model.get("heartbeat"),
                      "eta": model.get("eta"), "historical_complete_is_active": False},
    }


def artifact_payload(route: str) -> dict:
    queue = queue_payload()
    if not queue.get("ok"):
        return {
            "ok": False, "status": "failed",
            "schema": "operations-artifact-browser-error-v1154", "route": route,
            "source": "operations-db",
            "error": queue.get("error") or "authoritative operations DB unavailable",
            "requirements_with_artifacts": [], "legacy_static_path_selected": False,
        }
    rows = []
    for requirement in queue.get("requirements") or []:
        paths = [str(path) for path in requirement.get("artifact_paths") or []]
        hashes = list(requirement.get("artifact_hashes") or [])
        if paths:
            try:
                normalized = _normalize_artifact_hashes(paths, hashes)
            except ArtifactHashConflictError as error:
                return {
                    "ok": False, "status": "failed",
                    "schema": "operations-artifact-browser-error-v1154", "route": route,
                    "source": "operations-db", "error": str(error),
                    "requirements_with_artifacts": [], "legacy_static_path_selected": False,
                }
            rows.append({
                "requirement_id": requirement.get("requirement_id"),
                "status": requirement.get("status"), "artifact_paths": paths,
                "artifact_hashes": [{"path": path, "sha256": normalized[path]}
                                    for path in paths if path in normalized],
            })
    return {
        "ok": True, "status": "ok", "schema": "operations-artifact-browser-v1154",
        "route": route, "source": "operations-db",
        "requirements_with_artifacts": rows, "legacy_static_path_selected": False,
    }


def artifact_page(route: str, payload: dict | None = None) -> bytes:
    payload = payload or artifact_payload(route)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    body = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Operations evidence</title><style>:root{{font-family:system-ui;color:#13313b;background:#f6fbfc}}body{{margin:0;padding:18px}}main{{max-width:1050px;margin:auto}}header{{position:sticky;top:8px;padding:12px;background:#fff;border:2px solid #00b8d8}}nav a{{margin-right:12px;min-height:44px;display:inline-flex;align-items:center}}pre{{padding:16px;background:#fff;border-left:6px solid #8b78ff;white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body><main><header><h1>運用DBのハッシュ付き成果物</h1><nav><a href="/">研究ワークベンチ</a><a href="/viewer-single-event">3D成果物</a></nav></header><pre>{html.escape(text)}</pre></main></body></html>'''
    return body.encode("utf-8")


def canonical_contract() -> dict:
    viewer = exact_viewer_bytes()
    assets = [ROOT_PAGE, QUEUE_PAGE, VIEWER_PAGE, HERE / "viewer_10101_canonical_meta_v1120.json"]
    return {
        "schema": "viewer-canonical-entrypoint-flat-operations-v1154",
        "requirement_id": "NR-WEBUI-FLAT-ERROR-AMBIGUITY-GATE-070",
        "review_only": True, "public_activation": False,
        "handler_base": "BaseHTTPRequestHandler", "runtime_string_rewrites": 0,
        "runtime_monkey_patches": 0, "historical_dashboard_imports": 0,
        "operations_dependency": "operations_status_canonical_v1154.py",
        "viewer_binding": {"path": VIEWER_PAGE.name,
                           "expected_sha256": VIEWER_PAGE_EXPECTED_SHA256,
                           "actual_sha256": hashlib.sha256(viewer).hexdigest(),
                           "hash_bound": True, "fail_closed": True},
        "assets": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in assets],
        "packet_adapter": "viewer_10101_packet_adapter_v1120.py",
        "viewer_merge": MERGE_CONTRACT,
    }


def audit_page() -> bytes:
    contract = html.escape(json.dumps(canonical_contract(), ensure_ascii=False, indent=2))
    return (f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Canonical review</title><style>body{{margin:auto;max-width:1000px;padding:16px;font-family:system-ui;background:#eef5f7;color:#123}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px;border:2px solid #38a7b8}}</style><h1>Canonical review</h1><p>Review-only. Public activation is frozen.</p><pre>{contract}</pre></html>').encode()


class Handler(BaseHTTPRequestHandler):
    server_version = "EquakeCanonicalFlatOperations/1154"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(self, body: bytes, content_type: str, status: int = 200,
                   cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, pretty: bool = False, status: int = 200) -> None:
        options = {"ensure_ascii": False, "indent": 2} if pretty else {
            "ensure_ascii": False, "separators": (",", ":")}
        self.send_bytes(json.dumps(payload, **options).encode(),
                        "application/json; charset=utf-8", status=status)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.send_bytes(ROOT_PAGE.read_bytes(), "text/html; charset=utf-8"); return
        if path == "/agents-queue":
            self.send_bytes(QUEUE_PAGE.read_bytes(), "text/html; charset=utf-8"); return
        if path in {"/viewer-single-event", "/viewer-single-event/"}:
            try:
                body = exact_viewer_bytes()
            except (OSError, RuntimeError) as error:
                self.send_json({"status": "fail_closed", "stage": "viewer-asset-sha256",
                                "error": str(error)}, pretty=True, status=503); return
            self.send_bytes(body, "text/html; charset=utf-8"); return
        if path == "/viewer-single-event-audit":
            self.send_bytes(audit_page(), "text/html; charset=utf-8"); return
        if path == "/api/agent-queue": self.send_json(queue_payload()); return
        if path in {"/api/model-run", "/api/model-run-status"}:
            self.send_json(build_status()); return
        if path == "/api/model-run/audit": self.send_json(build_status(), pretty=True); return
        if path == "/api/development-progress": self.send_json(authoritative_progress()); return
        if path == "/api/single-event-gs/meta": self.send_json(load_meta()); return
        if path == "/api/single-event-gs/display":
            phase = parse_qs(parsed.query).get("phase", ["P"])[0]
            if phase not in {"P", "S", "L"}:
                self.send_bytes(b"bad request\n", "text/plain; charset=utf-8", 400); return
            self.send_bytes(packet(phase), "application/octet-stream"); return
        if path == "/api/viewer-diagnostics": self.send_json(CAMERA_DIAGNOSTICS); return
        if path in {"/api/viewer-merge-contract", "/api/viewer-lighting-contract",
                    "/api/viewer-background-contract"}:
            self.send_json(MERGE_CONTRACT); return
        if path == "/api/viewer-canonical-contract": self.send_json(canonical_contract()); return
        if path in {"/api/artifacts", "/api/artifact-browser"}:
            route = parse_qs(parsed.query).get("route", ["/artifact/current/final"])[0]
            payload = artifact_payload(route)
            self.send_json(payload, status=200 if payload.get("ok") else 503); return
        if path == "/artifacts/japan-v987-reference.png":
            if not POSTER.is_file():
                self.send_bytes(b"poster unavailable\n", "text/plain; charset=utf-8", 404); return
            self.send_bytes(POSTER.read_bytes(), "image/png", cache="public,max-age=300"); return
        if path in ARTIFACT_ROUTES:
            payload = artifact_payload(path)
            self.send_bytes(artifact_page(path, payload), "text/html; charset=utf-8",
                            status=200 if payload.get("ok") else 503); return
        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", 404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18178)
    args = parser.parse_args()
    exact_viewer_bytes()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
