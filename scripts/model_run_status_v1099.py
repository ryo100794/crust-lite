#!/usr/bin/env python3
"""Evidence guard layered on v1097 model-run state derivation."""
from __future__ import annotations

from typing import Any
import model_run_status_v1097 as base


def build_status(*, now=None, processes=None, fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = base.build_status(now=now, processes=processes, fixture=fixture)
    fixture = fixture or {}
    if fixture.get("explicit_state") == "complete":
        digest = str(fixture.get("completion_evidence_sha256") or "")
        valid = fixture.get("completion_pass") is True and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())
        payload["completion_evidence_valid"] = valid
        if not valid:
            payload["status"] = "paused"
            payload["status_label"] = "PAUSED"
            payload["stop_reason"] = "Complete was rejected: explicit passing audit evidence and SHA-256 are required."
            payload["eta"] = {"available": False, "reason": "Unverified completion cannot produce an active or complete state."}
    return payload
