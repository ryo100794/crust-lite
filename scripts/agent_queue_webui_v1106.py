#!/usr/bin/env python3
"""v1106: RUNNING AGENTS contains active, non-releasable runtime rows only."""
from __future__ import annotations
import agent_queue_webui_v1102 as v1102
def api_payload()->dict:
 d=v1102.api_payload();d['schema']='agent-queue-runtime-compact-v1106';return d
QUEUE_PAGE=v1102.QUEUE_PAGE.replace("const running=(q.runtime_agents||[]);","const running=(q.runtime_agents||[]).filter(x=>['claimed','running','checkpoint'].includes(x.state)&&!x.slot_releasable);",1)
