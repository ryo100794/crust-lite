#!/usr/bin/env python3
"""v1100: align compact dispatch summaries with explicit wait reasons."""
from __future__ import annotations

import agent_queue_webui_v1098 as v1098


def api_payload() -> dict:
    data = v1098.api_payload()
    counts = {k: 0 for k in ("total", "running", "ready", "dependency-wait", "slot-wait", "approval-wait", "failed", "completed")}
    rows = data.get("requirements") or []; counts["total"] = len(rows)
    for row in rows:
        state = row.get("dispatch_state")
        reason = (row.get("blocked_reason") or "").lower()
        if row.get("status") == "queued" and reason:
            state = "approval-wait" if ("approval" in reason or "承認" in reason) else "dependency-wait"
            row["dispatch_state"] = state
        counts[state] = counts.get(state, 0) + 1
    data["dispatch_counts"] = counts
    data["schema"] = "agent-queue-runtime-compact-v1100"
    return data


QUEUE_PAGE = v1098.QUEUE_PAGE
