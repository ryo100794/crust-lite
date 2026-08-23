#!/usr/bin/env python3
"""v1105: keep stop/resume reasons consistent with derived live/stale state."""
from __future__ import annotations
from typing import Any
import model_run_status_v1099 as base
def build_status(*,now=None,processes=None,fixture:dict[str,Any]|None=None)->dict[str,Any]:
 d=base.build_status(now=now,processes=processes,fixture=fixture)
 if d['status']=='stale' and d['active_process']:
  d['stop_reason']='A matching model process exists, but no fresh authoritative heartbeat is available within the configured TTL; progress and ETA are stale.'
  d['resume_conditions']=['The running process must publish a heartbeat/progress record tied to its event, phase/tile/stage and versions.','If the process is no longer valid, stop it through its owning requirement and record a paused/failed checkpoint.']
 elif d['status']=='running':d['stop_reason']=None
 return d
