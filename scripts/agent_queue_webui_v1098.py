#!/usr/bin/env python3
"""v1098 compact queue plus authoritative event-driven agent runtime fields."""
from __future__ import annotations

import agent_queue_webui_v1097 as v1097
from agent_runtime_store_v1098 import snapshot as runtime_snapshot


def api_payload() -> dict:
    data = v1097.api_payload()
    runtime = runtime_snapshot()
    normalized = []
    for row in runtime["agents"]:
        item = dict(row)
        item.update({
            "created_utc": row["state_changed_at"], "updated_utc": row["heartbeat_at"],
            "scope": row.get("stage") or row["state"],
            "result": f"{row['progress'] or '—'} · state={row['state']} · heartbeat_age={row['heartbeat_age_s']}s · stale={row['stale']} · slot_releasable={row['slot_releasable']}",
            "status": "in_progress" if not row["slot_releasable"] else row["state"],
        })
        normalized.append(item)
    data["runtime_schema"] = runtime["schema"]
    data["runtime_agents"] = normalized
    data["next_ready_claim"] = runtime["next_ready_claim"]
    data["heartbeat_contract_seconds"] = runtime["heartbeat_contract_seconds"]
    data["stale_after_seconds"] = runtime["stale_after_seconds"]
    data["push_constraint"] = runtime["push_constraint"]
    data["schema"] = "agent-queue-runtime-compact-v1098"
    return data


QUEUE_PAGE = v1097.QUEUE_PAGE.replace(
    "const running=(q.requirements||[]).filter(r=>r.status==='in_progress');",
    "const running=(q.runtime_agents||[]);",
    1,
).replace(
    "timer=setTimeout(load,15000)", "timer=setTimeout(load,5000)", 1
).replace(
    "next 15s", "next 5s", 1
).replace(
    "</style>", ".changed{animation:opsFlash 1.2s ease-out}@keyframes opsFlash{0%{outline:4px solid #ffd34e}100%{outline:0 solid transparent}}</style>", 1
).replace(
    "function render(q,m,stale){",
    "let previousFingerprint='';function render(q,m,stale){const fingerprint=JSON.stringify([q.generated_utc,q.next_ready_claim,(q.runtime_agents||[]).map(x=>[x.assignee,x.state,x.heartbeat_at,x.stale,x.slot_releasable]),m.status,m.heartbeat?.last_utc]);const changed=previousFingerprint&&previousFingerprint!==fingerprint;previousFingerprint=fingerprint;setTimeout(()=>{for(const x of document.querySelectorAll('.changed'))x.classList.remove('changed')},1300);",
    1,
).replace(
    "$('foot').textContent=`${q.schema}",
    "if(changed){$('summary').classList.add('changed');$('agents').classList.add('changed')}$('foot').textContent=`next-ready ${q.next_ready_claim||'none'} / heartbeat contract ${q.heartbeat_contract_seconds}s / stale>${q.stale_after_seconds}s / ${q.push_constraint} / ${q.schema}",
    1,
)
