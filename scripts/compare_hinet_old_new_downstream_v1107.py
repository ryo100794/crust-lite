#!/usr/bin/env python3
"""Stage E: complete old/new waveform and phase-window impact audit."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from obspy import UTCDateTime, read
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel


ROOT = Path("/workspace/equake/crust-lite")
sys.path.insert(0, str(ROOT / "scripts"))
from phase_aware_gs_pipeline import exact_spectrum, filtered, phase_arrivals, phase_window, refine_pick  # noqa: E402

STAGE_C = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_C_v1103.json"
STAGE_D = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_D_v1106.json"
CATALOG = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/catalog_20260801_20260822_m4.csv"
NEW_ROOT = ROOT / "data/interim/hinet_official_calibrated_v1106/analysis_ready_nmps"
OUT_JSON = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_E_v1107.json"
OUT_MD = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_E_v1107.md"
PROGRESS = ROOT / "logs/hinet_official_calibration_stage_e_v1107.progress.json"
FREQUENCIES = [0.5, 1.0, 2.0, 4.0, 8.0]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_events() -> dict[str, dict[str, Any]]:
    out = {}
    with CATALOG.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["lat"] = float(row["lat"])
            row["lon"] = float(row["lon"])
            row["depth_km"] = float(row["depth_km"])
            out[str(row["event_id"])] = row
    return out


def alignment_correction(t0_raw: float, duration_s: float, arrivals: dict[str, tuple[float, float]]) -> float:
    expected_midpoint = 0.5 * (arrivals["P"][0] + arrivals["S"][0])

    def key(correction_s: float) -> tuple[int, float]:
        start_s = t0_raw + correction_s
        end_s = start_s + duration_s
        contained = int(start_s <= arrivals["P"][0] <= end_s) + int(start_s <= arrivals["S"][0] <= end_s)
        return contained, -abs(0.5 * (start_s + end_s) - expected_midpoint)

    return max((0.0, -32400.0, 32400.0), key=key)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = math.sqrt(float(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 0.0 else 0.0


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)), "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)), "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    stage_d = json.loads(STAGE_D.read_text(encoding="utf-8"))
    if not str(stage_d.get("decision", "")).startswith("PASS"):
        raise RuntimeError("Stage D is not complete/PASS")
    stage_c = json.loads(STAGE_C.read_text(encoding="utf-8"))
    old_records = stage_c["records"]
    if len(old_records) != 1872:
        raise RuntimeError(f"expected 1872 old records, got {len(old_records)}")
    events = load_events()
    model = TauPyModel(model="iasp91")
    arrival_cache: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    records: list[dict[str, Any]] = []
    short_windows: list[str] = []
    started = datetime.now(UTC)

    for index, item in enumerate(old_records, 1):
        event_id = str(item["event_id"])
        group = int(item["group"])
        station = str(item["station"])
        component = str(item["component"])
        old_path = ROOT / str(item["sac_path"])
        new_path = NEW_ROOT / event_id / f"group{group}" / f"{station}.{component}.SAC"
        old_trace = read(str(old_path))[0]
        new_trace = read(str(new_path))[0]
        old_data = np.asarray(old_trace.data, dtype=np.float64)
        new_data = np.asarray(new_trace.data, dtype=np.float64)
        event = events[event_id]
        origin = UTCDateTime(str(event["time_utc"]))
        cache_key = (event_id, station)
        if cache_key not in arrival_cache:
            distance_deg = locations2degrees(float(event["lat"]), float(event["lon"]), float(old_trace.stats.sac.stla), float(old_trace.stats.sac.stlo))
            arrival_cache[cache_key] = phase_arrivals(model, float(event["depth_km"]), float(distance_deg))
        arrivals = arrival_cache[cache_key]
        duration_s = float((old_trace.stats.npts - 1) / old_trace.stats.sampling_rate)
        old_t0_raw = float(old_trace.stats.starttime.timestamp) - float(origin.timestamp)
        new_t0_raw = float(new_trace.stats.starttime.timestamp) - float(origin.timestamp)
        old_correction = alignment_correction(old_t0_raw, duration_s, arrivals)
        new_correction = alignment_correction(new_t0_raw, duration_s, arrivals)
        old_filtered = filtered(old_data, float(old_trace.stats.sampling_rate))
        new_filtered = filtered(new_data, float(new_trace.stats.sampling_rate))
        old_times = old_t0_raw + old_correction + np.arange(old_data.size) / float(old_trace.stats.sampling_rate)
        new_times = new_t0_raw + new_correction + np.arange(new_data.size) / float(new_trace.stats.sampling_rate)
        old_p, old_pq = refine_pick(old_times, old_filtered, arrivals["P"][0], 2.0, 3.0)
        old_s, old_sq = refine_pick(old_times, old_filtered, arrivals["S"][0], 3.0, 5.0)
        new_p, new_pq = refine_pick(new_times, new_filtered, arrivals["P"][0], 2.0, 3.0)
        new_s, new_sq = refine_pick(new_times, new_filtered, arrivals["S"][0], 3.0, 5.0)
        if old_s <= old_p + 0.5:
            old_s = max(arrivals["S"][0], old_p + 0.5)
        if new_s <= new_p + 0.5:
            new_s = max(arrivals["S"][0], new_p + 0.5)
        phase_metrics: dict[str, Any] = {}
        for phase in ["P", "S"]:
            old_pick = old_p if phase == "P" else old_s
            new_pick = new_p if phase == "P" else new_s
            old_start, old_end, old_sigma = phase_window(phase, old_p, old_s, float(old_times[0]), float(old_times[-1]))
            new_start, new_end, new_sigma = phase_window(phase, new_p, new_s, float(new_times[0]), float(new_times[-1]))
            try:
                old_z, old_centroid, old_n = exact_spectrum(old_times, old_filtered, old_start, old_end, old_pick, old_sigma)
                new_z, new_centroid, new_n = exact_spectrum(new_times, new_filtered, new_start, new_end, new_pick, new_sigma)
                ratios = np.abs(new_z) / np.maximum(np.abs(old_z), np.finfo(np.float64).tiny)
                phase_delta = np.angle(new_z * np.conj(old_z))
                phase_metrics[phase] = {
                    "old_pick_s": old_pick, "new_pick_s": new_pick, "pick_delta_s": new_pick - old_pick,
                    "old_quality": old_pq if phase == "P" else old_sq, "new_quality": new_pq if phase == "P" else new_sq,
                    "old_centroid_s": old_centroid, "new_centroid_s": new_centroid,
                    "old_n_samples": old_n, "new_n_samples": new_n,
                    "amplitude_ratio_new_over_old": {str(f): float(v) for f, v in zip(FREQUENCIES, ratios, strict=True)},
                    "phase_delta_rad_new_minus_old": {str(f): float(v) for f, v in zip(FREQUENCIES, phase_delta, strict=True)},
                }
            except ValueError:
                short_windows.append(f"{event_id}/{group}/{station}/{component}/{phase}")
                phase_metrics[phase] = {"error": "short_window"}

        old_rms = float(np.sqrt(np.mean(old_data * old_data)))
        new_rms = float(np.sqrt(np.mean(new_data * new_data)))
        record = {
            "event_id": event_id, "group": group, "station": station, "component": component,
            "orientation_status": "vertical_not_rotated" if component == "U" else ("quarantined_official_orientation_unavailable" if station in {"N.SSGH", "N.SSWH"} else "official_true_NE"),
            "old_path": str(old_path.relative_to(ROOT)), "new_path": str(new_path.relative_to(ROOT)),
            "shape_equal": old_data.shape == new_data.shape,
            "both_finite": bool(np.isfinite(old_data).all() and np.isfinite(new_data).all()),
            "starttime_shift_s": float(new_trace.stats.starttime - old_trace.stats.starttime),
            "old_alignment_correction_s": old_correction, "new_alignment_correction_s": new_correction,
            "new_raw_window_contains_P": new_t0_raw <= arrivals["P"][0] <= new_t0_raw + duration_s,
            "new_raw_window_contains_S": new_t0_raw <= arrivals["S"][0] <= new_t0_raw + duration_s,
            "old_rms_nmps": old_rms, "new_rms_nmps": new_rms,
            "rms_ratio_new_over_old": new_rms / max(old_rms, np.finfo(np.float64).tiny),
            "whole_trace_correlation": correlation(old_data, new_data),
            "phase_metrics": phase_metrics,
        }
        records.append(record)
        if index % 72 == 0:
            atomic_json(PROGRESS, {
                "schema": "hinet-official-calibration-stage-e-progress-v1107",
                "updated_at_utc": datetime.now(UTC).isoformat(), "status": "running" if index < len(old_records) else "complete",
                "completed_sac": index, "expected_sac": len(old_records), "current": {"event_id": event_id, "group": group},
            })
            print(f"{index}/{len(old_records)}", flush=True)

    amplitude_by_phase_frequency: dict[str, list[float]] = defaultdict(list)
    phase_delta_by_phase_frequency: dict[str, list[float]] = defaultdict(list)
    pick_delta: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for phase, metrics in record["phase_metrics"].items():
            if "error" in metrics:
                continue
            pick_delta[phase].append(abs(float(metrics["pick_delta_s"])))
            for frequency, value in metrics["amplitude_ratio_new_over_old"].items():
                amplitude_by_phase_frequency[f"{phase}_{frequency}Hz"].append(float(value))
            for frequency, value in metrics["phase_delta_rad_new_minus_old"].items():
                phase_delta_by_phase_frequency[f"{phase}_{frequency}Hz"].append(abs(float(value)))
    rms_ratios = [float(r["rms_ratio_new_over_old"]) for r in records]
    correlations = [float(r["whole_trace_correlation"]) for r in records]
    summary = {
        "records": len(records),
        "all_shapes_equal": all(bool(r["shape_equal"]) for r in records),
        "all_finite": all(bool(r["both_finite"]) for r in records),
        "all_starttime_shifts_minus_32400": all(math.isclose(float(r["starttime_shift_s"]), -32400.0, abs_tol=1e-6) for r in records),
        "old_alignment_correction_counts": dict(Counter(str(r["old_alignment_correction_s"]) for r in records)),
        "new_alignment_correction_counts": dict(Counter(str(r["new_alignment_correction_s"]) for r in records)),
        "new_raw_windows_contain_P": sum(bool(r["new_raw_window_contains_P"]) for r in records),
        "new_raw_windows_contain_S": sum(bool(r["new_raw_window_contains_S"]) for r in records),
        "short_phase_windows": len(short_windows),
        "rms_ratio_new_over_old": percentiles(rms_ratios),
        "whole_trace_correlation": percentiles(correlations),
        "absolute_pick_delta_s": {phase: percentiles(values) for phase, values in pick_delta.items()},
        "amplitude_ratio_new_over_old": {key: percentiles(values) for key, values in sorted(amplitude_by_phase_frequency.items())},
        "absolute_phase_delta_rad": {key: percentiles(values) for key, values in sorted(phase_delta_by_phase_frequency.items())},
        "quarantined_horizontal_records": sum(str(r["orientation_status"]).startswith("quarantined") for r in records),
        "elapsed_s": (datetime.now(UTC) - started).total_seconds(),
    }
    pass_gate = (
        len(records) == 1872 and summary["all_shapes_equal"] and summary["all_finite"]
        and summary["all_starttime_shifts_minus_32400"]
        and summary["new_alignment_correction_counts"] == {"0.0": 1872}
        and summary["new_raw_windows_contain_P"] == 1872
        and summary["new_raw_windows_contain_S"] == 1872
        and summary["short_phase_windows"] == 0
    )
    audit = {
        "schema": "hinet-official-calibration-audit-stage-e-v1107",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": "E_old_new_difference_and_downstream_phase_impact",
        "inputs": {"stage_c": str(STAGE_C.relative_to(ROOT)), "stage_d": str(STAGE_D.relative_to(ROOT)), "catalog": str(CATALOG.relative_to(ROOT))},
        "summary": summary,
        "short_windows": short_windows,
        "records": records,
        "decision": "PASS_WITH_ORIENTATION_QUARANTINE" if pass_gate else "FAIL",
        "downstream_switch_proposal": {
            "allowed_now": False,
            "eligible": "All verticals and N/E from the 46 stations with official orientation, only after parent review.",
            "excluded": "N.SSGH and N.SSWH N/E until NR-HINET-ORIENT-001 is resolved.",
            "required_code_change": "Use UTC header directly and remove the arrival-dependent +/-9h branch; retain research filter/window configuration as a separately versioned choice."
        },
        "known_structure_used": False,
        "machine_learning_used": False,
        "existing_products_modified": False,
        "new_requirements": json.loads(STAGE_D.read_text(encoding="utf-8")).get("new_requirements", []),
    }
    atomic_json(OUT_JSON, audit)
    OUT_MD.write_text(f"""# Hi-net公式補正監査 段階E（v1107）

判定: **{audit['decision']}**

- 旧新比較: {summary['records']}/1872、shape一致={summary['all_shapes_equal']}、finite={summary['all_finite']}、開始時刻-32400 s={summary['all_starttime_shifts_minus_32400']}。
- 新UTC波形で現行到着時刻ヒューリスティックが0秒を選択: `{json.dumps(summary['new_alignment_correction_counts'], ensure_ascii=False)}`。
- 補正なしの新波形窓にP/S予測到着を含む: P={summary['new_raw_windows_contain_P']}/1872、S={summary['new_raw_windows_contain_S']}/1872。短すぎるP/S窓={summary['short_phase_windows']}。
- RMS比（新/旧）: `{json.dumps(summary['rms_ratio_new_over_old'], ensure_ascii=False)}`。
- P/Sの周波数別振幅比・位相差、pick差はJSONに全件保存した。
- 公式方位欠測で隔離する水平成分: {summary['quarantined_horizontal_records']}件。

旧成果物・Web・後段入力は変更していない。親レビュー前の入力切替は禁止し、切替時は予測到着依存±9時間分岐を廃止してUTCヘッダを直接使う。
""", encoding="utf-8")
    print(json.dumps({"decision": audit["decision"], "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
