#!/usr/bin/env python3
"""Formal inactive-candidate audit for NR-WEBUI-FLAT-ERROR-AMBIGUITY-GATE-070."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/equake/crust-lite")
REQ = "NR-WEBUI-FLAT-ERROR-AMBIGUITY-GATE-070"
OUT = ROOT / "logs/audits/NR_WEBUI_FLAT_ERROR_AMBIGUITY_GATE_070/v1154"
AUDIT = OUT / "NR_WEBUI_FLAT_ERROR_AMBIGUITY_GATE_070.audit.json"
MD = OUT / "NR_WEBUI_FLAT_ERROR_AMBIGUITY_GATE_070.audit.md"
LEDGER = OUT / "SHA256SUMS"
ARTIFACTS = [
    "scripts/operations_status_canonical_v1154.py",
    "scripts/equake_dashboard_10101_canonical_r4_v1154.py",
    "tests/test_webui_flat_error_ambiguity_v1154.py",
    "tests/test_webui_flat_error_ambiguity_v1154_negatives.py",
    "scripts/audit_nr070_webui_flat_error_ambiguity_v1154.py",
]
INVARIANTS = [
    "scripts/equake_dashboard_10101_canonical_r4_v1131.py",
    "scripts/viewer_10101_canonical_v1120_r4.html",
    "scripts/viewer_10101_packet_adapter_v1120.py",
    "configs/hinet_downstream_input.json",
    "configs/analysis_database.json",
    "data/processed/crust_lite_hinet_only_20260823_v2.duckdb",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    ops = importlib.import_module("operations_status_canonical_v1154")
    dash = importlib.import_module("equake_dashboard_10101_canonical_r4_v1154")
    OUT.mkdir(parents=True, exist_ok=True)
    before = {name: sha(ROOT / name) for name in INVARIANTS}
    checks: list[dict[str, Any]] = []

    missing = [name for name in ARTIFACTS + INVARIANTS if not (ROOT / name).is_file()]
    checks.append(check("immutable_inputs_present", not missing, missing))
    source_hashes = {name: sha(ROOT / name) for name in ARTIFACTS}
    checks.append(check("source_test_auditor_hashes_full64", all(HEX64.fullmatch(x) for x in source_hashes.values()), source_hashes))

    a, b = "a" * 64, "b" * 64
    conflict_error = None
    try:
        ops._normalize_artifact_hashes(
            ["same.json"],
            [{"path": "same.json", "sha256": a}, f"{b}  same.json"],
        )
    except ops.ArtifactHashConflictError as error:
        conflict_error = str(error)
    checks.append(check(
        "same_path_different_full64_explicit_fail_closed",
        conflict_error is not None and "same.json" in conflict_error
        and "artifact hash conflict" in conflict_error,
        conflict_error,
    ))
    duplicate = ops._normalize_artifact_hashes(
        ["same.json"], [{"path": "same.json", "sha256": a}, f"{a}  same.json"])
    malformed = ops._normalize_artifact_hashes(
        ["same.json"], [{"path": "same.json", "sha256": "not-a-digest"}])
    checks.append(check("same_digest_duplicate_allowed_and_malformed_excluded", duplicate == {"same.json": a} and malformed == {}, {"duplicate": duplicate, "malformed": malformed}))

    original_queue = dash.queue_payload
    dash.queue_payload = lambda: {"ok": False, "schema": "fixture-error", "error": "fixture-corrupt-db"}
    try:
        queue_error = dash.artifact_payload("/artifact/current/final")
    finally:
        dash.queue_payload = original_queue
    checks.append(check(
        "authoritative_db_error_propagates_never_empty_normal",
        queue_error.get("ok") is False and queue_error.get("status") == "failed"
        and "fixture-corrupt-db" in queue_error.get("error", "")
        and queue_error.get("requirements_with_artifacts") == [],
        queue_error,
    ))

    fixture_conflict = {
        "ok": True, "requirements": [{
            "requirement_id": "NR-X", "status": "complete_pass",
            "artifact_paths": ["same.json"],
            "artifact_hashes": [
                {"path": "same.json", "sha256": a},
                {"path": "same.json", "sha256": b},
            ],
        }],
    }
    dash.queue_payload = lambda: fixture_conflict
    try:
        conflict_payload = dash.artifact_payload("/artifact/current/final")
    finally:
        dash.queue_payload = original_queue
    checks.append(check(
        "artifact_conflict_emits_no_evidence_rows",
        conflict_payload.get("ok") is False and conflict_payload.get("status") == "failed"
        and "artifact hash conflict" in conflict_payload.get("error", "")
        and conflict_payload.get("requirements_with_artifacts") == [],
        conflict_payload,
    ))

    source = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ARTIFACTS[:2])
    forbidden = [name for name in (
        "agent_queue_webui_v1093", "agent_queue_webui_v1097", "agent_queue_webui_v1098",
        "agent_queue_webui_v1100", "agent_queue_webui_v1102", "agent_queue_webui_v1106",
        "model_run_status_v1097", "model_run_status_v1099", "model_run_status_v1105",
        "NR-KNOWN-LEAK-008",
    ) if name in source]
    checks.append(check(
        "flat_canonical_import_graph_and_no_old_hardcodes",
        not forbidden and "operations_status_canonical_v1154" in source
        and "operations_status_canonical_v1132 import" not in source,
        {"forbidden_found": forbidden},
    ))
    checks.append(check(
        "artifact_http_semantics_static_contract",
        'path in {"/api/artifacts", "/api/artifact-browser"}' in source
        and 'status=200 if payload.get("ok") else 503' in source
        and "artifact_page(path, payload)" in source,
        "JSON API and HTML route use the same authoritative payload and HTTP503 on failure",
    ))

    run = subprocess.run(
        [str(ROOT / ".venv/bin/pytest"), "-q",
         "tests/test_webui_flat_error_ambiguity_v1154.py",
         "tests/test_webui_flat_error_ambiguity_v1154_negatives.py"],
        cwd=ROOT, text=True, capture_output=True, timeout=240,
    )
    test_output = (run.stdout + run.stderr).strip()
    checks.append(check("old060_contract_plus_v1154_negative_browser_tests", run.returncode == 0 and "15 passed" in test_output, test_output))

    process = subprocess.run(
        ["pgrep", "-af", "equake_dashboard_10101_canonical_r4_v1154.py"],
        text=True, capture_output=True,
    )
    isolated = [line for line in process.stdout.splitlines() if line.strip()]
    checks.append(check("isolated_v1154_processes_stopped", not isolated, isolated))

    current = subprocess.run(
        ["pgrep", "-af", "equake_dashboard_10101_canonical_r4_v1131.py.*10101"],
        text=True, capture_output=True,
    )
    public_lines = [line for line in current.stdout.splitlines() if line.strip()]
    checks.append(check("public10101_remains_v1131", len(public_lines) == 1 and "--port 10101" in public_lines[0], public_lines))

    after = {name: sha(ROOT / name) for name in INVARIANTS}
    checks.append(check("public_viewer_packet_pointer_db_invariant", before == after, {"before": before, "after": after}))

    failed = [item["name"] for item in checks if not item["pass"]]
    decision = "PASS_CANDIDATE_INACTIVE_PUBLIC_ZERO_PENDING_DIFFERENT_OWNER_AUDIT" if not failed else "FAIL_AUDIT"
    generated = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "schema": "nr-webui-flat-error-ambiguity-gate-audit-v1154",
        "requirement_id": REQ, "generated_at_utc": generated,
        "decision": decision, "activation_allowed": False, "public_switch": False,
        "checks": checks, "checks_total": len(checks),
        "checks_passed": len(checks) - len(failed), "failed_checks": failed,
        "test_command": ".venv/bin/pytest -q tests/test_webui_flat_error_ambiguity_v1154.py tests/test_webui_flat_error_ambiguity_v1154_negatives.py",
        "test_output": test_output,
        "http_contract": {
            "success": {"status": 200, "ok": True},
            "operations_db_or_hash_conflict": {"status": 503, "ok": False, "status_label": "failed", "requirements_with_artifacts": []},
            "html_and_json_share_payload": True,
        },
        "immutable_artifacts": [{"path": name, "sha256": value} for name, value in source_hashes.items()],
        "invariants": after,
        "prohibitions": {"public_change": 0, "pointer_change": 0, "science_change": 0, "data_db_change": 0, "delete": 0},
        "next_gate": "A different owner must independently audit exact v1154 before any activation.",
    }
    AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {REQ} formal audit", "", f"- Decision: `{decision}`",
        f"- Generated UTC: `{generated}`", f"- Checks: `{len(checks)-len(failed)}/{len(checks)}`",
        f"- Tests: `{test_output}`", "- Candidate activation: `false`", "- Public switch: `0`", "",
        "Same-path conflicting full64 hashes now fail with an explicit error and emit no evidence rows. Authoritative operations DB errors propagate through JSON and HTML as ok=false/failed/HTTP503 and are never converted to a normal empty payload.",
        "", "## Checks", "",
    ]
    lines.extend(f"- [{'PASS' if item['pass'] else 'FAIL'}] `{item['name']}`" for item in checks)
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger_items = ARTIFACTS + [str(AUDIT.relative_to(ROOT)), str(MD.relative_to(ROOT))]
    LEDGER.write_text("".join(f"{sha(ROOT / name)}  {name}\n" for name in ledger_items), encoding="utf-8")
    print(json.dumps({"decision": decision, "checks": f"{len(checks)-len(failed)}/{len(checks)}", "audit_sha256": sha(AUDIT), "md_sha256": sha(MD), "ledger_sha256": sha(LEDGER)}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
