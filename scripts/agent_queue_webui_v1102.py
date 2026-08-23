#!/usr/bin/env python3
"""v1102: keep remaining/ETA and resume conditions visible at 390 px."""
from __future__ import annotations
import agent_queue_webui_v1100 as v1100

def api_payload()->dict:
    data=v1100.api_payload();data["schema"]="agent-queue-runtime-compact-v1102";return data

QUEUE_PAGE=v1100.QUEUE_PAGE.replace(
    ".model .cell:nth-of-type(4),.model .cell:nth-of-type(5){display:none}","",1
).replace(
    "${esc(m.stop_reason)}</div><a class=\"raw\"",
    "${esc(m.stop_reason)}<br>resume: ${esc((m.resume_conditions||[]).join(' / '))}</div><a class=\"raw\"",1
)
