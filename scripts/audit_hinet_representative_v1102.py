#!/usr/bin/env python3
"""Value-level audit of representative Hi-net CNT/CH/SAC conversion.

Creates only isolated audit outputs. It never modifies the source or current SAC.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from HinetPy import win32
from obspy import read


ROOT = Path("/workspace/equake/crust-lite")
RAW = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/raw/group0_raw/2026/hinet_20260804000046/stations_N_NYOH_N_ASBH_N_SSGH_N_KJSH_N_KGRH_N_OGAH/raw"
CURRENT_SAC = RAW.parent / "sac"
CH = RAW / "0101_20260804.ch"
CNT = RAW / "0101_202608040053_5.cnt"
OFFICIAL_BIN = ROOT / "logs/nied_official_audit_v1096/sources/win32tools_official/win32tools/win2sac.src/win2sac_32"
CURRENT_BIN = ROOT / ".deps/hinet-win32tools/win32tools/win2sac.src/win2sac_32"
ORIENTATION_CSV = ROOT / "logs/nied_official_audit_v1100/sources/national_csv.csv"
OUT = ROOT / "logs/nied_official_audit_v1102/representative"
AUDIT_JSON = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_B_v1102.json"
AUDIT_MD = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_B_v1102.md"
EVENT_TIME = datetime(2026, 8, 4, tzinfo=UTC)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def version(binary: Path) -> str:
    result = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    text = result.stdout + result.stderr
    return text.splitlines()[0] if text.splitlines() else ""


def applicable_orientations(names: set[str]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    with ORIENTATION_CSV.open("r", encoding="euc_jp", newline="") as fh:
        for row in csv.reader(fh, quotechar="'"):
            if len(row) < 7 or row[2] not in names:
                continue
            start = datetime.fromisoformat(row[4]).replace(tzinfo=UTC) if row[4] else None
            end = datetime.fromisoformat(row[5]).replace(tzinfo=UTC) if row[5] else None
            if start and EVENT_TIME < start:
                continue
            if end and EVENT_TIME > end:
                continue
            selected[row[2]] = {
                "theta_deg": float(row[3]),
                "valid_start": row[4] or None,
                "valid_end": row[5] or None,
                "method_flag": row[6] or None,
            }
    return selected


def sac_summary(path: Path) -> tuple[dict[str, object], np.ndarray]:
    trace = read(str(path), headonly=False)[0]
    sac = trace.stats.sac
    data = np.asarray(trace.data, dtype=np.float64)
    keys = ["stla", "stlo", "stel", "stdp", "cmpaz", "cmpinc", "calib", "scale", "depmin", "depmax", "depmen"]
    headers = {}
    for key in keys:
        value = sac.get(key, None)
        headers[key] = None if value is None else float(value)
    summary: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "starttime_header": str(trace.stats.starttime),
        "starttime_if_header_is_JST_then_UTC": str(trace.stats.starttime - 9 * 3600),
        "delta_s": float(trace.stats.delta),
        "sampling_rate_hz": float(trace.stats.sampling_rate),
        "npts": int(trace.stats.npts),
        "duration_s": float((trace.stats.npts - 1) * trace.stats.delta),
        "station": str(trace.stats.station),
        "channel": str(trace.stats.channel),
        "headers": headers,
        "data": {
            "dtype_read": str(data.dtype),
            "finite": bool(np.isfinite(data).all()),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "rms": float(np.sqrt(np.mean(data * data))),
        },
    }
    return summary, data


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    if a.shape != b.shape:
        return {"shape_equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = a - b
    denom = max(float(np.max(np.abs(b))), np.finfo(np.float64).tiny)
    return {
        "shape_equal": True,
        "array_equal": bool(np.array_equal(a, b)),
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_error": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error_over_reference_peak": float(np.max(np.abs(diff)) / denom),
    }


def run_conversion(binary: Path, label: str, channels: list[object]) -> dict[str, Path]:
    outdir = OUT / label
    outdir.mkdir(parents=True, exist_ok=True)
    prm = outdir / "win.prm"
    prm.write_text(f".\n{CH}\n.\n.\n", encoding="utf-8")
    results: dict[str, Path] = {}
    for channel in channels:
        subprocess.run(
            [str(binary), str(CNT), channel.id, "SAC", str(outdir), "-e", f"-p{prm}", "-m8640000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        path = outdir / f"{channel.name}.{channel.component}.SAC"
        if not path.exists():
            raise RuntimeError(f"missing conversion output {path}")
        results[f"{channel.name}.{channel.component}"] = path
    return results


def main() -> None:
    for path in [CH, CNT, OFFICIAL_BIN, CURRENT_BIN, ORIENTATION_CSV]:
        if not path.exists():
            raise FileNotFoundError(path)
    all_channels = win32.read_ctable(str(CH))
    names = {c.name for c in all_channels}
    orientations = applicable_orientations(names)
    candidates = [(abs(float(meta["theta_deg"])), name) for name, meta in orientations.items()]
    directional_station = max(candidates)[1] if candidates else "N.ASBH"
    selected_names = {"N.ASBH", directional_station}
    channels = [c for c in all_channels if c.name in selected_names and c.component in {"U", "N", "E"}]
    channels.sort(key=lambda c: (c.name, c.component))
    if len(channels) != 3 * len(selected_names):
        raise RuntimeError(f"expected three components per selected station, got {len(channels)}")

    official_files = run_conversion(OFFICIAL_BIN, "official_download_binary", channels)
    current_build_files = run_conversion(CURRENT_BIN, "current_build_binary", channels)
    records = []
    for channel in channels:
        key = f"{channel.name}.{channel.component}"
        existing = CURRENT_SAC / f"{key}.SAC"
        official_summary, official_data = sac_summary(official_files[key])
        build_summary, build_data = sac_summary(current_build_files[key])
        existing_summary, existing_data = sac_summary(existing)
        step_nm_per_unit = float(channel.lsb_value / (channel.gain * 10.0 ** (channel.preamplification / 20.0)) * 1.0e9)
        ratios = official_data / step_nm_per_unit
        orientation = orientations.get(channel.name)
        nominal_az = 0.0 if channel.component in {"U", "N"} else 90.0
        nominal_inc = 0.0 if channel.component == "U" else 90.0
        actual_az = nominal_az
        if orientation and channel.component == "N":
            actual_az = float(orientation["theta_deg"]) % 360.0
        elif orientation and channel.component == "E":
            actual_az = (float(orientation["theta_deg"]) + 90.0) % 360.0
        records.append({
            "station": channel.name,
            "component": channel.component,
            "channel_id": channel.id,
            "ch": {
                "latitude": float(channel.latitude),
                "longitude": float(channel.longitude),
                "unit": channel.unit,
                "sensor_gain_V_per_physical_unit": float(channel.gain),
                "period_s": float(channel.period),
                "damping": float(channel.damping),
                "preamplification_db": float(channel.preamplification),
                "lsb_V_per_count": float(channel.lsb_value),
                "expected_output_step_nm_per_s_per_count": step_nm_per_unit,
            },
            "official_orientation": orientation,
            "nominal_cmpaz_deg": nominal_az,
            "nominal_cmpinc_deg": nominal_inc,
            "actual_cmpaz_deg_for_directional_analysis": actual_az,
            "official_download_binary_output": official_summary,
            "current_build_binary_output": build_summary,
            "existing_output": existing_summary,
            "official_vs_current_build": compare(official_data, build_data),
            "official_vs_existing": compare(official_data, existing_data),
            "count_quantization": {
                "max_abs_fractional_count_residual": float(np.max(np.abs(ratios - np.rint(ratios)))),
                "p99_abs_fractional_count_residual": float(np.quantile(np.abs(ratios - np.rint(ratios)), 0.99)),
            },
            "header_checks": {
                "station_coordinates_match_CH": bool(
                    math.isclose(float(official_summary["headers"]["stla"]), float(channel.latitude), abs_tol=1e-4)
                    and math.isclose(float(official_summary["headers"]["stlo"]), float(channel.longitude), abs_tol=1e-4)
                ),
                "nominal_cmpaz_match": math.isclose(float(official_summary["headers"]["cmpaz"]), nominal_az, abs_tol=1e-6),
                "nominal_cmpinc_match": math.isclose(float(official_summary["headers"]["cmpinc"]), nominal_inc, abs_tol=1e-6),
                "actual_orientation_already_applied": math.isclose(float(official_summary["headers"]["cmpaz"]), actual_az, abs_tol=1e-6),
            },
        })

    comparisons = [r["official_vs_existing"] for r in records]
    all_arrays_equal = all(bool(c.get("array_equal")) for c in comparisons)
    all_nominal_headers = all(
        bool(r["header_checks"]["station_coordinates_match_CH"])
        and bool(r["header_checks"]["nominal_cmpaz_match"])
        and bool(r["header_checks"]["nominal_cmpinc_match"])
        for r in records
    )
    nonzero_directional = [r for r in records if r["component"] in {"N", "E"} and not math.isclose(float(r["actual_cmpaz_deg_for_directional_analysis"]), float(r["nominal_cmpaz_deg"]), abs_tol=1e-6)]
    raw_stamp_match = re.search(r"(20\d{10})", CNT.name)
    raw_header = records[0]["official_download_binary_output"]["starttime_header"]
    audit = {
        "schema": "hinet-official-calibration-audit-stage-b-v1102",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": "B_representative_CH_CNT_SAC_values",
        "official_primary_sources_only_for_normative_requirements": True,
        "known_structure_used": False,
        "machine_learning_used": False,
        "existing_products_modified": False,
        "inputs": {
            "ch": str(CH.relative_to(ROOT)), "ch_sha256": sha256(CH),
            "cnt": str(CNT.relative_to(ROOT)), "cnt_sha256": sha256(CNT),
            "current_sac_dir": str(CURRENT_SAC.relative_to(ROOT)),
            "official_orientation_csv": str(ORIENTATION_CSV.relative_to(ROOT)),
        },
        "binaries": {
            "official_download": {"path": str(OFFICIAL_BIN.relative_to(ROOT)), "sha256": sha256(OFFICIAL_BIN), "version": version(OFFICIAL_BIN)},
            "current_build": {"path": str(CURRENT_BIN.relative_to(ROOT)), "sha256": sha256(CURRENT_BIN), "version": version(CURRENT_BIN)},
            "binary_hash_equal": sha256(OFFICIAL_BIN) == sha256(CURRENT_BIN),
            "interpretation": "Different ELF build hashes are acceptable only if output equivalence is demonstrated."
        },
        "selection": {"baseline_station": "N.ASBH", "nonzero_orientation_demo_station": directional_station, "components": ["U", "N", "E"]},
        "time_check": {
            "raw_cnt_filename_timestamp": raw_stamp_match.group(1) if raw_stamp_match else None,
            "official_SAC_header_start": raw_header,
            "deterministic_UTC_if_source_header_is_JST": str(read(str(official_files[next(iter(official_files))]))[0].stats.starttime - timedelta(hours=9)),
            "current_pipeline_arrival_dependent_time_selection_is_official_conversion": False
        },
        "records": records,
        "summary": {
            "record_count": len(records),
            "official_download_vs_existing_all_arrays_equal": all_arrays_equal,
            "official_download_vs_existing_max_abs_error": max(float(c.get("max_abs_error", math.inf)) for c in comparisons),
            "CH_coordinates_and_nominal_orientation_headers_all_match": all_nominal_headers,
            "nonzero_actual_orientation_components_not_applied_count": len(nonzero_directional),
            "full_instrument_response_removed": False,
            "sampling_and_data_conversion": "PASS" if all_arrays_equal and all_nominal_headers else "FAIL",
            "actual_orientation": "FAIL" if nonzero_directional else "NOT_DEMONSTRATED",
            "deterministic_UTC": "FAIL",
            "full_response_for_amplitude_phase": "FAIL"
        },
        "decision": "FAIL_REQUIRES_STAGE_C_AND_ISOLATED_STAGE_D",
        "old_version_overwritten": False,
        "downstream_switch_allowed": False,
        "new_requirements": []
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = f"""# Hi-net公式補正監査 段階B（v1102）

判定: **{audit['decision']}**

- 代表: N.ASBH と、同一raw内で公式設置方位差が最大の {directional_station}、各U/N/E。
- 公式配布win2sac_32と既存SACの配列完全一致: **{all_arrays_equal}**（最大絶対誤差 {audit['summary']['official_download_vs_existing_max_abs_error']:.9g}）。
- CH座標・名目CMPAZ/CMPINC: **{'PASS' if all_nominal_headers else 'FAIL'}**。
- 公式実設置方位が未反映の水平成分: **{len(nonzero_directional)}件**。
- SACヘッダ時刻はCNTファイル名時刻と同じ基準で保存された。UTC解析入力は決定論的に9時間減算すべきで、予測到着時刻による選択は公式変換ではない。
- 感度換算は一致したが、センサ/A-D/デジタルフィルタを含む完全装置応答除去は未実施。

隔離出力: `logs/nied_official_audit_v1102/representative/`。既存SAC・Web・後段入力は変更していない。
"""
    AUDIT_MD.write_text(md, encoding="utf-8")
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
