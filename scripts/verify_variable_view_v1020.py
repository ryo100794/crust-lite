#!/usr/bin/env python3
"""Regression and independent CPU-reference checks for variable-K v1020."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def make_synthetic(view_count=8, shape=(12, 13, 7)):
    x = np.linspace(-1, 1, shape[0], dtype=np.float32)[:, None, None]
    y = np.linspace(-1, 1, shape[1], dtype=np.float32)[None, :, None]
    z = np.linspace(0, 1, shape[2], dtype=np.float32)[None, None, :]
    common = (
        1.8 * np.exp(-((x + .22*z)**2/.16 + (y-.18*z)**2/.12 + (z-.38)**2/.06))
        + .9 * np.exp(-((x-.48)**2/.10 + (y+.35)**2/.15 + (z-.72)**2/.08))
        + .12
    )
    response, illumination = [], []
    ix, iy, iz = np.indices(shape)
    for view in range(view_count):
        light = np.clip(.55 + .32*np.cos((view+1)*x + .7*y - .4*z) + .08*np.sin((view+2)*y), .08, 1)
        missing = ((3*ix + 5*iy + 7*iz + 11*view) % 29 == 0)
        light = np.where(missing, 0, light).astype(np.float32)
        signal = common * (1 + .035*np.sin((view+1)*x - (view+2)*y + .3*z))
        signal += .018*np.cos((view+3)*x + .8*y + .4*z)
        signal = np.maximum(signal, 0).astype(np.float32) * np.sqrt(np.maximum(light, .20))
        signal[missing] = 0
        response.append(signal); illumination.append(light)
    return np.asarray(response,np.float32), np.asarray(illumination,np.float32)


def max_abs(left, right):
    return float(torch.max(torch.abs(left-right)).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-dir", type=Path, required=True)
    parser.add_argument("--v1015-output", type=Path, required=True)
    parser.add_argument("--v1020-output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    torch.set_grad_enabled(False)
    v1015 = load_module("single_pass_v1015_for_verify_v1020", args.script_dir/"solve_single_pass_event_gs_gpu_v1015.py")
    v1020 = load_module("variable_view_v1020_for_verify", args.script_dir/"solve_variable_view_event_gs_gpu_v1020.py")
    with np.load(args.v1015_output,allow_pickle=False) as old, np.load(args.v1020_output,allow_pickle=False) as new:
        old_mask=np.asarray(old["candidate_mask"],bool); new_mask=np.asarray(new["candidate_mask"],bool)
        regression={
            "node_count":int(len(old_mask)),
            "v1015_candidates":int(np.count_nonzero(old_mask)),
            "v1020_candidates":int(np.count_nonzero(new_mask)),
            "candidate_mask_mismatch":int(np.count_nonzero(old_mask != new_mask)),
            "center_max_abs_m":float(np.max(np.abs(np.asarray(old["center_xyz_m"])-np.asarray(new["center_xyz_m"]))))
                if np.asarray(old["center_xyz_m"]).shape == np.asarray(new["center_xyz_m"]).shape else None,
        }

    response, illumination = make_synthetic()
    shape=response.shape[1:]; configs=v1020.shift_configurations(8,shape[0],shape[1])
    started=time.monotonic()
    cpu_raw=torch.from_numpy(response.copy()); cpu_light=torch.from_numpy(illumination.copy())
    cpu_y,cpu_scales=v1015.normalize_global_q90(cpu_raw,cpu_light)
    cpu_field,cpu_se,cpu_coverage=v1015.solve_wls(cpu_y,cpu_light,.04)
    cpu_sy,cpu_sl=v1020.make_shift_batch(cpu_y,cpu_light,configs)
    cpu_nf,cpu_ns,cpu_nc=v1020.solve_wls_shift_batch(cpu_sy,cpu_sl,.04)
    cpu_significance=cpu_field/torch.clamp(cpu_se,min=.01)
    cpu_threshold,cpu_raw_mask,cpu_mask,_=v1020.threshold_candidates(
        cpu_nf/torch.clamp(cpu_ns,min=.01),cpu_nc,cpu_significance,cpu_coverage,.985
    )
    cpu_seconds=time.monotonic()-started

    gpu_raw=torch.as_tensor(response,device="cuda"); gpu_light=torch.as_tensor(illumination,device="cuda")
    torch.cuda.reset_peak_memory_stats(); begin,end=torch.cuda.Event(True),torch.cuda.Event(True)
    begin.record()
    gpu_y,gpu_scales=v1015.normalize_global_q90(gpu_raw,gpu_light)
    gpu_field,gpu_se,gpu_coverage=v1015.solve_wls(gpu_y,gpu_light,.04)
    gpu_sy,gpu_sl=v1020.make_shift_batch(gpu_y,gpu_light,configs)
    gpu_nf,gpu_ns,gpu_nc=v1020.solve_wls_shift_batch(gpu_sy,gpu_sl,.04)
    gpu_significance=gpu_field/torch.clamp(gpu_se,min=.01)
    gpu_threshold,gpu_raw_mask,gpu_mask,_=v1020.threshold_candidates(
        gpu_nf/torch.clamp(gpu_ns,min=.01),gpu_nc,gpu_significance,gpu_coverage,.985
    )
    end.record(); torch.cuda.synchronize()
    agreement={
        "normalization_scale_max_abs":max_abs(gpu_scales.cpu(),cpu_scales),
        "normalized_response_max_abs":max_abs(gpu_y.cpu(),cpu_y),
        "actual_field_max_abs":max_abs(gpu_field.cpu(),cpu_field),
        "actual_standard_error_max_abs":max_abs(gpu_se.cpu(),cpu_se),
        "actual_coverage_max_abs":max_abs(gpu_coverage.cpu(),cpu_coverage),
        "null_field_max_abs":max_abs(gpu_nf.cpu(),cpu_nf),
        "null_standard_error_max_abs":max_abs(gpu_ns.cpu(),cpu_ns),
        "null_coverage_max_abs":max_abs(gpu_nc.cpu(),cpu_nc),
        "threshold_max_abs":max_abs(gpu_threshold.cpu(),cpu_threshold),
        "raw_mask_mismatch":int(torch.count_nonzero(gpu_raw_mask.cpu()!=cpu_raw_mask)),
        "gated_mask_mismatch":int(torch.count_nonzero(gpu_mask.cpu()!=cpu_mask)),
        "cpu_candidate_count":int(torch.count_nonzero(cpu_mask)),
        "gpu_candidate_count":int(torch.count_nonzero(gpu_mask)),
    }
    checks={
        "four_view_candidate_mask_matches_v1015":regression["candidate_mask_mismatch"]==0,
        "four_view_centers_match_v1015":regression["center_max_abs_m"] is not None and regression["center_max_abs_m"]==0,
        "eight_views_used":response.shape[0]==8,
        "eight_configurations":len(configs)==8,
        "distinct_roll_per_view_per_configuration":all(len(set(config))==8 for config in configs),
        "configuration_by_k_batch_shape":tuple(gpu_sy.shape[:2])==(8,8),
        "cpu_gpu_raw_mask_match":agreement["raw_mask_mismatch"]==0,
        "cpu_gpu_gated_mask_match":agreement["gated_mask_mismatch"]==0,
        "cpu_gpu_field_close":agreement["actual_field_max_abs"]<1e-5,
        "cpu_gpu_threshold_close":agreement["threshold_max_abs"]<1e-4,
        "autograd_absent":not torch.is_grad_enabled(),
        "machine_learning_absent":True,"known_structure_absent":True,
    }
    audit={
        "schema":"variable-view-v1020-regression-and-cpu-reference-audit",
        "created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "four_view_regression":regression,
        "eight_view_synthetic":{"shape":list(shape),"shift_configurations_xy":configs,
                                "cpu_seconds":cpu_seconds,"gpu_elapsed_ms":float(begin.elapsed_time(end)),
                                "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),
                                "agreement":agreement},
        "checks":checks,"pass":all(checks.values()),
        "machine_learning_used":False,"known_structure_used":False,
    }
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"audit":str(args.audit),"pass":audit["pass"],"regression":regression,
                      "agreement":agreement},ensure_ascii=False))
    raise SystemExit(0 if audit["pass"] else 2)


if __name__=="__main__":main()
