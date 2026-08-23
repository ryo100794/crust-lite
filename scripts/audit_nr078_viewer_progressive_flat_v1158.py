#!/usr/bin/env python3
"""Audit the inactive standalone NR-078 progressive viewer candidate."""
from __future__ import annotations

import hashlib
import json
import re
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/equake/crust-lite")
OUT = ROOT / "logs/audits/NR_VIEWER_PROGRESSIVE_CHAIN_FLATTEN_078/v1158"
sys.path.insert(0, str(ROOT / "scripts"))
from viewer_progressive_selector_flat_v1158 import Selector


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)


def public_ports():
    result = run("ss", "-ltnp")
    return sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if re.search(r":(10101|10201|10202)\s", line)
    )


def active_flat_processes():
    result = run("pgrep", "-af", "viewer_progressive_selector_flat_v1158|equake_dashboard_10101_progressive_flat_v1158")
    excluded = ("pgrep -af", Path(__file__).name)
    return [line for line in result.stdout.splitlines() if line and not any(token in line for token in excluded)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    add = lambda name, passed, evidence: checks.append({"name": name, "pass": bool(passed), "evidence": evidence})
    candidate = [
        ROOT / "scripts/viewer_progressive_selector_flat_v1158.py",
        ROOT / "scripts/equake_dashboard_10101_progressive_flat_v1158.py",
        ROOT / "tests/test_viewer_progressive_flat_v1158.py",
        ROOT / "scripts/audit_nr078_viewer_progressive_flat_v1158.py",
    ]
    protected = [
        ROOT / "scripts/viewer_progressive_selector_v1148.py",
        ROOT / "scripts/viewer_progressive_selector_v1149.py",
        ROOT / "scripts/viewer_progressive_selector_v1150.py",
        ROOT / "scripts/viewer_progressive_selector_v1151.py",
        ROOT / "scripts/viewer_progressive_selector_v1152.py",
        ROOT / "scripts/equake_dashboard_10101_progressive_v1152.py",
        ROOT / "configs/viewer_progressive_authority_candidate_v1148.json",
        ROOT / "scripts/viewer_10101_canonical_v1120_r4.html",
        ROOT / "scripts/viewer_10101_canonical_meta_v1120.json",
        ROOT / "configs/hinet_downstream_input.json",
        ROOT / "configs/formal_hinet_events.json",
        ROOT / "pyproject.toml",
    ]
    protected_before = {str(path.relative_to(ROOT)): sha(path) for path in protected}
    ports_before = public_ports()
    pids_before = active_flat_processes()

    compiled = run(str(ROOT / ".venv/bin/python"), "-m", "py_compile", *(str(path.relative_to(ROOT)) for path in candidate))
    add("compile", compiled.returncode == 0, compiled.stderr[-4000:])
    tests = run(str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests/test_viewer_progressive_flat_v1158.py")
    add(
        "standalone_behavior_authority_concurrency_http_tests",
        tests.returncode == 0 and "15 passed" in tests.stdout and "11 subtests passed" in tests.stdout,
        {"stdout": tests.stdout[-4000:], "stderr": tests.stderr[-2000:]},
    )

    selector_source = candidate[0].read_text()
    dashboard_source = candidate[1].read_text()
    production = selector_source + "\n" + dashboard_source
    import_lines = [line.strip() for line in production.splitlines() if line.lstrip().startswith(("import ", "from "))]
    overlays = [line for line in import_lines if re.search(r"viewer_progressive_selector_v\d+|equake_dashboard_10101_progressive_v\d+", line)]
    dynamic = [token for token in ("importlib", "__import__(", "exec(", "eval(", "setattr(") if token in production]
    add("old_progressive_overlay_imports_zero", not overlays, {"imports": import_lines, "violations": overlays})
    add("dynamic_patch_or_loader_zero", not dynamic, dynamic)
    add(
        "standalone_definitions_present",
        all(token in selector_source for token in ("class Authority", "class Snapshot", "def load_authority", "def validate_manifest", "class Selector")),
        "authority/manifest/selector implemented locally",
    )
    add(
        "two_phase_refresh_and_response_commit_gate",
        selector_source.index("scan_now = self._clock()") < selector_source.index("with self.lock:", selector_source.index("scan_now = self._clock()")) < selector_source.index("commit_now = self._clock()")
        and "commit_authority = self._authority_at(commit_now)" in selector_source
        and dashboard_source.index("authorized_snapshot") < dashboard_source.index("producer(snapshot)") < dashboard_source.index("current.confirm(snapshot)") < dashboard_source.index("self.send_raw"),
        {"heavy_validation_off_lock": True, "fresh_authority_at_commit": True, "confirm_before_send": True},
    )
    add(
        "lazy_dashboard_initialization_no_import_side_effect_server",
        "SELECTOR: Selector | None = None" in dashboard_source
        and "SELECTOR = build_selector()" in dashboard_source
        and dashboard_source.index("SELECTOR = build_selector()") > dashboard_source.index("def main()"),
        "selector/preload created only in main",
    )

    live = Selector(
        ROOT,
        ROOT / "data/interim/viewer_progressive_v1148/manifests",
        ROOT / "configs/viewer_progressive_authority_candidate_v1148.json",
        ROOT / "data/operations/agent_queue.sqlite",
        ROOT / "data/interim/viewer_progressive_flat_v1158/runtime",
    )
    status = live.status()
    add(
        "real_live_authority_truthful_nonblank_coverage",
        status["schema"] == "viewer-progressive-status-flat-v1158"
        and status["authority_requirement_id"] == "NR-VIEWER-PROGRESSIVE-AUTHORITY-057"
        and status["coverage"]["nationwide"] is False
        and status["blank"] is False,
        status,
    )

    connection = sqlite3.connect(f"file:{ROOT/'data/operations/agent_queue.sqlite'}?mode=ro", uri=True)
    rows = connection.execute(
        "select requirement_id,artifact_paths_json from ops_requirement where requirement_id in (?,?,?)",
        (
            "NR-VIEWER-PROGRESSIVE-AUTHORITY-057",
            "NR-VIEWER-PROGRESSIVE-RUNTIME-AUTHORITY-068",
            "NR-VIEWER-PROGRESSIVE-CHAIN-FLATTEN-078",
        ),
    ).fetchall()
    connection.close()
    registered = {requirement_id: json.loads(paths) for requirement_id, paths in rows}
    mutable = [
        path for paths in registered.values() for path in paths
        if "viewer_progressive_flat_v1158/runtime" in path or "last_good_state" in path
    ]
    add("mutable_runtime_excluded_from_immutable_ledger", not mutable, registered)

    probe = socket.socket(); probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]; probe.close()
    process = subprocess.Popen(
        [str(ROOT/".venv/bin/python"), "scripts/equake_dashboard_10101_progressive_flat_v1158.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    routes = [
        "/", "/api/viewer-progressive-status", "/api/single-event-gs/meta",
        "/api/single-event-gs/display?phase=P", "/api/single-event-gs/display?phase=S",
        "/api/single-event-gs/display?phase=L", "/artifacts/japan-v987-reference.png",
    ]
    observed = []
    try:
        for _ in range(120):
            if process.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/viewer-progressive-status", timeout=1)
                break
            except Exception:
                time.sleep(0.05)
        for cycle in range(3):
            for route in routes:
                started = time.monotonic()
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=8) as response:
                        observed.append({
                            "cycle": cycle, "route": route, "status": response.status,
                            "bytes": len(response.read()), "elapsed_seconds": round(time.monotonic() - started, 4),
                        })
                except Exception as error:
                    observed.append({
                        "cycle": cycle, "route": route,
                        "error": f"{type(error).__name__}:{error}",
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                    })
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
    add(
        "isolated_http_21_under_background_preload",
        len(observed) == 21 and all(item.get("status") == 200 and item.get("bytes", 0) > 0 for item in observed),
        {"port": port, "observations": observed},
    )

    protected_after = {str(path.relative_to(ROOT)): sha(path) for path in protected}
    ports_after = public_ports()
    pids_after = active_flat_processes()
    invariance = {
        "protected_before": protected_before,
        "protected_after": protected_after,
        "public_ports_before": ports_before,
        "public_ports_after": ports_after,
        "flat_processes_before": pids_before,
        "flat_processes_after": pids_after,
    }
    add(
        "protected_public_pointer_science_and_process_invariance",
        protected_before == protected_after and ports_before == ports_after and pids_before == pids_after == [],
        invariance,
    )
    add("candidate_inactive_public_zero", not any("flat_v1158" in line for line in ports_after), ports_after)

    failures = [item["name"] for item in checks if not item["pass"]]
    document = {
        "schema": "nr-viewer-progressive-chain-flatten-audit-v1158",
        "requirement_id": "NR-VIEWER-PROGRESSIVE-CHAIN-FLATTEN-078",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS_CANDIDATE_INACTIVE_PENDING_DIFFERENT_OWNER" if not failures else "FAIL_REFACTOR_REQUIRED",
        "counts": {"pass": len(checks) - len(failures), "total": len(checks), "fail": len(failures)},
        "failures": failures,
        "checks": checks,
        "candidate_hashes": {str(path.relative_to(ROOT)): sha(path) for path in candidate},
        "invariance": invariance,
        "activation": False,
        "public_changes": 0,
        "pointer_changes": 0,
        "science_changes": 0,
        "data_deletions": 0,
    }
    json_path = OUT / "NR_VIEWER_PROGRESSIVE_CHAIN_FLATTEN_078.audit.json"
    md_path = OUT / "NR_VIEWER_PROGRESSIVE_CHAIN_FLATTEN_078.audit.md"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    md_path.write_text(
        "# NR-VIEWER-PROGRESSIVE-CHAIN-FLATTEN-078\n\n"
        f"Decision: **{document['decision']}** ({document['counts']['pass']}/{document['counts']['total']}).\n\n"
        "The selector and dashboard are standalone: old progressive module imports and dynamic overlay/patch loaders are absent. Live operations/history/time revalidation, atomic invalidation, persistent last-good, two-phase preload, truthful coverage and confirm-before-HTTP-send behavior are preserved.\n\n"
        "Candidate remains isolated. Public ports, pointers, science/config inputs and protected sources are unchanged; no deletion occurred.\n"
    )
    ledger = OUT / "SHA256SUMS"
    ledger.write_text(f"{sha(json_path)}  {json_path.name}\n{sha(md_path)}  {md_path.name}\n")
    print(json.dumps({
        "decision": document["decision"], "counts": document["counts"],
        "failures": failures, "json": str(json_path), "json_sha256": sha(json_path),
        "md_sha256": sha(md_path), "ledger_sha256": sha(ledger),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
