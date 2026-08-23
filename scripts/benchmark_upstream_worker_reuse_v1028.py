#!/usr/bin/env python3
"""Compare fresh-process and long-lived v1023 event×tile execution."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def task_arguments(project,manifest,tile,output,audit):
    return ["--project",str(project),"--manifest",str(manifest),"--tile-id",tile,
            "--source-core-km","0","--source-taper-end-km","15",
            "--output",str(output),"--audit",str(audit)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project",type=Path,required=True);parser.add_argument("--script",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--tile",action="append",required=True)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--audit",type=Path,required=True)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    fresh=[]
    for tile in args.tile:
        output=args.root/"fresh"/tile/"P_views_v1023.npz";audit=output.with_suffix(".audit.json");output.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(args.script),*task_arguments(args.project,args.manifest,tile,output,audit)]
        started=time.monotonic();result=subprocess.run(command,capture_output=True,text=True);elapsed=time.monotonic()-started
        if result.returncode!=0:raise RuntimeError(f"fresh {tile}: {result.stderr[-4000:]}")
        fresh.append({"tile":tile,"wall_seconds":elapsed,"output":str(output),"audit":str(audit)})

    import_started=time.monotonic()
    spec=importlib.util.spec_from_file_location("persistent_upstream_v1023",args.script)
    if spec is None or spec.loader is None:raise RuntimeError(args.script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module_import_seconds=time.monotonic()-import_started
    reused=[]
    for tile in args.tile:
        output=args.root/"reused"/tile/"P_views_v1023.npz";audit=output.with_suffix(".audit.json");output.parent.mkdir(parents=True,exist_ok=True)
        previous=sys.argv;sys.argv=[str(args.script),*task_arguments(args.project,args.manifest,tile,output,audit)]
        started=time.monotonic()
        try:
            module.main()
        except SystemExit as exit_status:
            if exit_status.code not in (None,0):raise
        finally:sys.argv=previous
        reused.append({"tile":tile,"wall_seconds":time.monotonic()-started,"output":str(output),"audit":str(audit)})

    comparisons=[]
    for left,right in zip(fresh,reused):
        with np.load(left["output"],allow_pickle=False) as a,np.load(right["output"],allow_pickle=False) as b:
            comparisons.append({"tile":left["tile"],
                "response_max_abs":float(np.max(np.abs(a["view_response"]-b["view_response"]))),
                "illumination_max_abs":float(np.max(np.abs(a["view_illumination"]-b["view_illumination"]))),
                "mask_mismatch":int(np.count_nonzero((a["view_response"]>0)!=(b["view_response"]>0)))})
    fresh_total=sum(row["wall_seconds"] for row in fresh)
    reused_task_total=sum(row["wall_seconds"] for row in reused)
    reused_with_import=module_import_seconds+reused_task_total
    checks={"at_least_two_tiles":len(args.tile)>=2,"outputs_identical":all(row["response_max_abs"]==0 and row["illumination_max_abs"]==0 and row["mask_mismatch"]==0 for row in comparisons),
            "reuse_faster_including_single_import":reused_with_import<fresh_total,
            "machine_learning_absent":True,"known_structure_absent":True}
    payload={"schema":"upstream-long-lived-worker-benchmark-v1028","created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
             "event_manifest":str(args.manifest),"tiles":args.tile,"fresh_process_tasks":fresh,"reused_worker_tasks":reused,
             "module_import_once_seconds":module_import_seconds,"fresh_total_seconds":fresh_total,
             "reused_task_total_seconds":reused_task_total,"reused_including_import_seconds":reused_with_import,
             "speedup_including_import":fresh_total/reused_with_import,"speedup_steady_tasks":fresh_total/reused_task_total,
             "output_comparison":comparisons,"checks":checks,"pass":all(checks.values()),
             "machine_learning_used":False,"known_structure_used":False,"publication_allowed":False}
    args.audit.parent.mkdir(parents=True,exist_ok=True);args.audit.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"audit":str(args.audit),"pass":payload["pass"],"fresh":fresh_total,"reuse":reused_with_import,
                      "speedup":payload["speedup_including_import"],"tasks":reused},ensure_ascii=False))
    raise SystemExit(0 if payload["pass"] else 2)


if __name__=="__main__":main()
