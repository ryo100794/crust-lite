#!/usr/bin/env python3
"""Review-only canonical 10101 dashboard with an exact r4 viewer binding.

The live service is intentionally not switched to this module.  All UI routes
are explicit and no historical dashboard Handler, HTML wrapper, runtime source
rewrite, or fallback server is inherited.
"""
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

from agent_queue_webui_v1106 import api_payload as queue_payload
from model_run_status_v1105 import build_status
from viewer_10101_packet_adapter_v1120 import load_meta, packet


ROOT_PAGE = HERE / "root_10101_canonical_v1120.html"
QUEUE_PAGE = HERE / "queue_10101_canonical_v1120.html"
VIEWER_PAGE = HERE / "viewer_10101_canonical_v1120_r4.html"
VIEWER_PAGE_EXPECTED_SHA256 = (
    "621162ff53781a4314d66d477ecc590475d1d7c99d80de3ef9053f2e973a9d36"
)
POSTER = PROJECT / "outputs/3d/japan_v987_check.png"

ARTIFACTS = {
    "/artifact/storage/rdb-inventory": (
        "RDB・データ源棚卸し",
        PROJECT / "logs/RDB_INVENTORY_20260823.md",
    ),
    "/artifact/input/acquisition": (
        "Hi-net取得・P/S分離監査",
        PROJECT / "logs/hinet_maintenance_20260822.audit.json",
    ),
    "/artifact/input/completeness": (
        "13地震×48局 完全性監査",
        PROJECT / "logs/hinet_event_completeness_20260822.audit.json",
    ),
    "/artifact/input/phase": (
        "P/S分離・直接波core抑制監査",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822"
        / "latest_hinet_phase_spectra_all13_20260822.audit.json",
    ),
    "/artifact/input/catalog-metadata": (
        "Hi-netカタログ取得条件",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822"
        / "catalog_20260801_20260822_m4.csv.metadata.json",
    ),
    "/artifact/input/station-groups": (
        "48観測点・24局×2独立群",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/north_tohoku_direction_hinet_v581"
        / "tohoku_groups24x2_targeted_v601.txt",
    ),
    "/artifact/design/split": (
        "development / blind 固定分割",
        PROJECT / "logs/hinet_48station_cohort_split_v1024.json",
    ),
    "/artifact/design/grid": (
        "北東北1.875 km・6タイル契約",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/recovery_v360/national_event_phase_v520"
        / "north_tohoku_resource_contract_v521.json",
    ),
    "/artifact/gpu/validation": (
        "48観測点P/S GPU数値同値・速度監査",
        PROJECT / "logs/GPU_48STATION_PS_WORKER_VIEW_DESIGN_v1029.md",
    ),
    "/artifact/current/full48": (
        "現event・full48 P/S 12タイル監査",
        PROJECT / "logs/hinet_20260802000524_full48_ps_v1053.audit.json",
    ),
    "/artifact/current/radial": (
        "現event・震源/radial nuisance除去監査",
        PROJECT / "logs/hinet_20260802000524_radial_p100_gs_v1060.audit.json",
    ),
    "/artifact/current/core": (
        "現event・地震別P/S core GS監査",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/recovery_v360"
        / "hinet_event_ps_core_mutual_p100_dev_v1063"
        / "hinet_20260802000524_PS_core_v1063.audit.json",
    ),
    "/artifact/current/fusion": (
        "現event・v1075完全共分散P/S融合監査",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/recovery_v360"
        / "hinet_event_ps_fused_overlap_dev_v1075"
        / "hinet_20260802000524_PS_fused_overlap_v1075.audit.json",
    ),
    "/artifact/current/evidence": (
        "現event・v1076エビデンス強度監査",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/recovery_v360"
        / "hinet_event_ps_fused_overlap_evidence_dev_v1076"
        / "hinet_20260802000524_PS_fused_overlap_evidence_v1076.audit.json",
    ),
    "/artifact/current/final": (
        "現段階の最終出力・v1077病理監査",
        PROJECT
        / "data/interim/phase_aware_gs_20260811/recovery_v360"
        / "hinet_event_ps_fused_overlap_evidence_dev_v1076"
        / "hinet_20260802000524_PS_fused_overlap_pathology_v1077.audit.json",
    ),
}

MERGE_CONTRACT = {
    "schema": "viewer-merge-contract-v1107",
    "lighting": {
        "model": "two-sided-wrap-diffuse-v1096",
        "defaults": {
            "ambient": 0.46,
            "wrap": 0.55,
            "roughness": 0.78,
            "azimuth_deg": -35.0,
            "elevation_deg": 50.0,
            "specular_intensity": 0.055,
            "rim_intensity": 0.055,
            "basis_weight": 0.22,
        },
    },
    "background": {
        "default": "light",
        "persistent_key": "equake-viewer-background-v1107",
        "light_clear_color": [0.82, 0.88, 0.91, 1.0],
        "dark_clear_color": [0.008, 0.025, 0.04, 1.0],
    },
    "unchanged": [
        "phase packets",
        "strength",
        "opacity expression",
        "geometry",
        "centres",
        "covariance vectors",
        "P/S/L colours",
        "threshold",
        "camera v1096",
        "land opacity 8%",
        "gesture v1095",
        "static poster bytes",
    ],
}

CAMERA_DIAGNOSTICS = {
    "schema": "viewer-diagnostics-contract-v1107-merge",
    "privacy": {
        "client_telemetry_collected": False,
        "request_headers_recorded": False,
        "user_agent_recorded": False,
        "ip_recorded": False,
        "diagnostics_location": "browser-local window.__viewerDiagnostics",
    },
    "renderer_order": [
        "webgl2",
        "webgl1+ANGLE_instanced_arrays",
        "static-audit-image",
    ],
    "forced_test_paths": {
        "webgl2": "/viewer-single-event?renderer=webgl2",
        "webgl1": "/viewer-single-event?renderer=webgl1",
        "context_none": "/viewer-single-event?renderer=none",
        "auto": "/viewer-single-event?renderer=auto",
    },
    "static_fallback": "/artifacts/japan-v987-reference.png",
    "data_apis": [
        "/api/single-event-gs/meta",
        "/api/single-event-gs/display?model=single&phase=P",
        "/api/single-event-gs/display?model=single&phase=S",
        "/api/single-event-gs/display?model=single&phase=L",
    ],
    "interaction": {
        "one_pointer": "arcball rotation",
        "two_pointer": "centroid pan plus pinch zoom",
        "pinch_out": "projected screen bbox grows",
        "pinch_in": "projected screen bbox shrinks",
        "wheel_negative_delta": "projected screen bbox grows",
        "mode_transition": "rebase on every pointer add/remove; no state mutation",
    },
    "mobile_controls": ["display settings", "quality", "top", "oblique", "reset"],
    "camera": {
        "ui_quantity": "screen magnification",
        "default_magnification": 1.0,
        "maximum_magnification": 15.2,
        "internal_projection_denominator": {"min": 0.05, "max": 5.0, "default": 0.76},
        "single_clamp_function": "clampZoom",
    },
    "land_layer": {
        "control_label": "日本列島 不透明度",
        "default_opacity": 0.08,
        "independent_uniform": "uLayerAlpha",
        "P_alpha": 1.0,
        "S_alpha": 1.0,
        "draw_order": ["L", "P", "S"],
        "land_depth_write": False,
    },
    "viewer_merge": MERGE_CONTRACT,
}

DEVELOPMENT_BASE = {
    "schema": "hinet-development-workbench-progress-v1120-canonical-review",
    "development_completed": 4,
    "development_total": 7,
    "completed_events": [
        "hinet_20260804000046",
        "hinet_20260807000101",
        "hinet_20260815000360",
        "hinet_20260802000524",
    ],
    "current_event": "hinet_20260814000874",
    "P_isochron": {"state": "complete", "fraction": 1.0, "station_current": 48, "stations_total": 48},
    "S_isochron": {"state": "complete", "fraction": 1.0, "station_current": 48, "stations_total": 48},
    "manifests_ready": True,
    "gpu_pipeline": {
        "schema": "full48-ps-tiles-persistent-gpu-progress-v1053",
        "state": "complete",
        "event_id": "hinet_20260814000874",
        "tasks_total": 12,
        "fraction": 1.0,
        "audit": "logs/hinet_20260814000874_full48_ps_v1053.audit.json",
        "pass": True,
        "quality_pass_tiles": 10,
        "publication_allowed": False,
        "historical_state": "complete",
        "historical": True,
    },
    "stage_pass": {
        "phase_manifests": True,
        "full48_gpu": True,
        "event_gs": True,
        "ps_covariance_fusion": False,
        "final_quality": False,
    },
    "latest_preview_event": "hinet_20260802000524",
    "latest_preview_quality_pass": True,
    "last_completed_event": "hinet_20260802000524",
    "blind_events_touched": 0,
    "retuning_on_blind_allowed": False,
    "publication_allowed": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_viewer_bytes() -> bytes:
    """Read the sole canonical viewer and fail closed on any asset drift."""
    body = VIEWER_PAGE.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != VIEWER_PAGE_EXPECTED_SHA256:
        raise RuntimeError(
            "canonical r4 viewer SHA-256 mismatch: "
            f"expected={VIEWER_PAGE_EXPECTED_SHA256} actual={actual}"
        )
    return body


def authoritative_progress() -> dict:
    payload = json.loads(json.dumps(DEVELOPMENT_BASE))
    model = build_status()
    payload["model_run"] = {
        "schema": model["schema"],
        "status": model["status"],
        "active_process": model["active_process"],
        "heartbeat": model["heartbeat"],
        "eta": model["eta"],
        "historical_complete_is_active": False,
    }
    payload["gpu_pipeline"]["execution_state"] = model["status"]
    return payload


def artifact_page(title: str, path: Path) -> bytes:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        state, accent = "FOUND", "#50d59f"
    except (OSError, json.JSONDecodeError) as error:
        text = f"{type(error).__name__}: {error}"
        state, accent = "MISSING", "#ff6b6b"
    body = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>:root{{font-family:system-ui;color:#13313b;background:#f6fbfc}}body{{margin:0;padding:18px;background:#f6fbfc}}main{{max-width:1050px;margin:auto}}header{{position:sticky;top:8px;padding:12px 14px;background:#fff;border:2px solid #00b8d8}}h1{{font-size:18px;margin:0 0 5px}}.state{{display:inline-block;padding:3px 7px;background:{accent};font:900 10px ui-monospace}}.path{{font:10px ui-monospace;color:#58737d;overflow-wrap:anywhere}}nav a{{color:#006d82;font-weight:800;margin-right:12px;min-height:44px;display:inline-flex;align-items:center}}pre{{margin-top:18px;padding:16px;background:#fff;border-left:6px solid #8b78ff;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.55 ui-monospace}}@media(max-width:700px){{body{{padding:9px}}pre{{padding:11px;font-size:10px}}}}</style></head><body><main><header><h1>{html.escape(title)}</h1><span class="state">{state}</span><div class="path">{html.escape(str(path))}</div><nav><a href="/">研究ワークベンチ</a><a href="/viewer-single-event">3D成果物</a></nav></header><pre>{html.escape(text)}</pre></main></body></html>'''
    return body.encode("utf-8")


def canonical_contract() -> dict:
    viewer_bytes = exact_viewer_bytes()
    assets = [ROOT_PAGE, QUEUE_PAGE, VIEWER_PAGE, HERE / "viewer_10101_canonical_meta_v1120.json"]
    return {
        "schema": "viewer-canonical-entrypoint-wire-v1130",
        "requirement_id": "NR-VIEWER-ENTRYPOINT-WIRE-032",
        "review_only": True,
        "public_activation": False,
        "handler_base": "BaseHTTPRequestHandler",
        "runtime_string_rewrites": 0,
        "runtime_monkey_patches": 0,
        "historical_viewer_imports": 0,
        "viewer_binding": {
            "path": VIEWER_PAGE.name,
            "expected_sha256": VIEWER_PAGE_EXPECTED_SHA256,
            "actual_sha256": hashlib.sha256(viewer_bytes).hexdigest(),
            "hash_bound": True,
            "fail_closed": True,
        },
        "assets": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in assets
        ],
        "packet_adapter": "viewer_10101_packet_adapter_v1120.py",
        "viewer_merge": MERGE_CONTRACT,
    }


def audit_page() -> bytes:
    contract = json.dumps(canonical_contract(), ensure_ascii=False, indent=2)
    return (
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Viewer canonical review</title>'
        '<style>body{margin:auto;max-width:1000px;padding:16px;font-family:system-ui;background:#eef5f7;color:#123}'
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px;border:2px solid #38a7b8}</style>'
        '<h1>Viewer canonical review v1120</h1><p>Review-only. Public activation is frozen.</p><pre>'
        + html.escape(contract)
        + "</pre></html>"
    ).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "EquakeCanonicalR4Review/1130"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_bytes(
        self, body: bytes, content_type: str, status: int = 200, cache: str = "no-store"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, pretty: bool = False, status: int = 200) -> None:
        options = {"ensure_ascii": False}
        if pretty:
            options["indent"] = 2
        else:
            options["separators"] = (",", ":")
        self.send_bytes(
            json.dumps(payload, **options).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.send_bytes(ROOT_PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/agents-queue":
            self.send_bytes(QUEUE_PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path in {"/viewer-single-event", "/viewer-single-event/"}:
            try:
                body = exact_viewer_bytes()
            except (OSError, RuntimeError) as error:
                self.send_json(
                    {
                        "status": "fail_closed",
                        "stage": "viewer-asset-sha256",
                        "error": str(error),
                    },
                    pretty=True,
                    status=503,
                )
                return
            self.send_bytes(body, "text/html; charset=utf-8")
            return
        if path == "/viewer-single-event-audit":
            self.send_bytes(audit_page(), "text/html; charset=utf-8")
            return
        if path == "/api/agent-queue":
            self.send_json(queue_payload())
            return
        if path in {"/api/model-run", "/api/model-run-status"}:
            self.send_json(build_status())
            return
        if path == "/api/model-run/audit":
            self.send_json(build_status(), pretty=True)
            return
        if path == "/api/development-progress":
            self.send_json(authoritative_progress())
            return
        if path == "/api/single-event-gs/meta":
            self.send_json(load_meta())
            return
        if path == "/api/single-event-gs/display":
            phase = parse_qs(parsed.query).get("phase", ["P"])[0]
            if phase not in {"P", "S", "L"}:
                self.send_bytes(b"bad request\n", "text/plain; charset=utf-8", 400)
                return
            self.send_bytes(packet(phase), "application/octet-stream")
            return
        if path == "/api/viewer-diagnostics":
            self.send_json(CAMERA_DIAGNOSTICS)
            return
        if path in {
            "/api/viewer-merge-contract",
            "/api/viewer-lighting-contract",
            "/api/viewer-background-contract",
        }:
            self.send_json(MERGE_CONTRACT)
            return
        if path == "/api/viewer-canonical-contract":
            self.send_json(canonical_contract())
            return
        if path == "/artifacts/japan-v987-reference.png":
            if not POSTER.is_file():
                self.send_bytes(b"poster unavailable\n", "text/plain; charset=utf-8", 404)
                return
            self.send_bytes(POSTER.read_bytes(), "image/png", cache="public,max-age=300")
            return
        if path in ARTIFACTS:
            title, artifact = ARTIFACTS[path]
            self.send_bytes(artifact_page(title, artifact), "text/html; charset=utf-8")
            return
        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", 404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18178)
    args = parser.parse_args()
    exact_viewer_bytes()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

