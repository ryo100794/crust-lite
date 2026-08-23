#!/usr/bin/env python3
"""Isolated full reprocessing of the 13x48x3 Hi-net cohort from CNT/CH.

Layer 1 is regenerated transiently with the official NIED win2sac_32 binary.
Layer 2 is retained as response-corrected, UTC-standardized, geographically
oriented nm/s SAC. Stations without an official azimuth are explicitly
quarantined for horizontal directional use and are never silently imputed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from HinetPy import win32
from obspy import read, read_inventory


ROOT = Path("/workspace/equake/crust-lite")
COHORT = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/strict_completion_20260822.json"
RAW_ROOT = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/raw"
ORIENTATION_CSV = ROOT / "logs/nied_official_audit_v1100/sources/national_csv.csv"
OFFICIAL_BIN = ROOT / "logs/nied_official_audit_v1096/sources/win32tools_official/win32tools/win2sac.src/win2sac_32"
RESP_PATHS = {
    "type3": ROOT / "logs/nied_official_audit_v1104/sources/seed_type3.txt",
    "type4": ROOT / "logs/nied_official_audit_v1098/sources/seed_type4.txt",
}
DEFAULT_OUT = ROOT / "data/interim/hinet_official_calibrated_v1106/analysis_ready_nmps"
AUDIT_JSON = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_D_v1106.json"
AUDIT_MD = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_D_v1106.md"
PROGRESS = ROOT / "logs/hinet_official_calibration_v1106.progress.json"
PREFILT = [0.05, 0.10, 35.0, 45.0]
WATER_LEVEL_DB = 60.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def response_type(lsb: float) -> str:
    if math.isclose(lsb, 1.021e-7, rel_tol=0.0, abs_tol=5e-12):
        return "type4"
    if math.isclose(lsb, 1.023e-7, rel_tol=0.0, abs_tol=5e-12):
        return "type3"
    raise ValueError(f"unsupported LSB for this cohort: {lsb}")


def load_orientations() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ORIENTATION_CSV.open("r", encoding="euc_jp", newline="") as fh:
        for row in csv.reader(fh, quotechar="'"):
            if len(row) < 7:
                continue
            try:
                theta = float(row[3])
            except ValueError:
                continue
            out[row[2]].append({"theta_deg": theta, "valid_start": row[4] or None, "valid_end": row[5] or None, "method_flag": row[6] or None})
    return out


def applicable_orientation(rows: list[dict[str, Any]], when: datetime) -> dict[str, Any] | None:
    for row in rows:
        start = datetime.fromisoformat(row["valid_start"]) if row["valid_start"] else None
        end = datetime.fromisoformat(row["valid_end"]) if row["valid_end"] else None
        if start and when < start:
            continue
        if end and when > end:
            continue
        return row
    return None


def choose_event_dir(group: int, event_id: str) -> tuple[Path, str]:
    normal = RAW_ROOT / f"group{group}_raw/2026/{event_id}"
    retry = RAW_ROOT / f"group{group}_retry_latest_raw/2026/{event_id}"
    for path, label in [(retry, "retry_latest"), (normal, "normal")]:
        if len(list(path.glob("**/sac/*.SAC"))) == 72:
            return path, label
    raise RuntimeError(f"no complete source for {event_id}/group{group}")


def cnt_time(path: Path) -> datetime:
    match = re.search(r"(20\d{10})", path.name)
    if not match:
        raise ValueError(f"CNT timestamp unavailable: {path}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M")


def build_response_templates() -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    templates = {}
    metadata = {}
    for kind, path in RESP_PATHS.items():
        response = read_inventory(str(path), format="RESP")[0][0][0].response
        templates[kind] = response
        metadata[kind] = {
            "A0": float(response.response_stages[0].normalization_factor),
            "template_stage1_gain": float(response.response_stages[0].stage_gain),
            "digitizer_gain": float(response.response_stages[1].stage_gain),
            "template_final_sensitivity": float(response.instrument_sensitivity.value),
        }
    return templates, metadata


def configured_response(template: Any, gain: float) -> Any:
    response = copy.deepcopy(template)
    a0 = float(response.response_stages[0].normalization_factor)
    response.response_stages[0].stage_gain = gain / a0
    response.instrument_sensitivity.value = response.response_stages[0].stage_gain * float(response.response_stages[1].stage_gain)
    return response


def official_convert_one(binary: Path, cnt: Path, channel: Any, outdir: Path, prm: Path) -> Path:
    result = subprocess.run(
        [str(binary), str(cnt), channel.id, "SAC", str(outdir), "-e", f"-p{prm}", "-m8640000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    output = outdir / f"{channel.name}.{channel.component}.SAC"
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"official conversion failed for {channel.name}.{channel.component}: rc={result.returncode}")
    return output


def response_correct(trace: Any, channel: Any, response: Any) -> tuple[np.ndarray, float]:
    original_nmps = np.asarray(trace.data, dtype=np.float64)
    counts_per_mps_ch = float(channel.gain) * 10.0 ** (float(channel.preamplification) / 20.0) / float(channel.lsb_value)
    reconstructed_counts = original_nmps * 1.0e-9 * counts_per_mps_ch
    quantization_residual = float(np.max(np.abs(reconstructed_counts - np.rint(reconstructed_counts))))
    trace.data = reconstructed_counts.astype(np.float64)
    trace.stats.response = response
    trace.remove_response(
        output="VEL", pre_filt=PREFILT, water_level=WATER_LEVEL_DB,
        zero_mean=True, taper=True, taper_fraction=0.05,
    )
    corrected_nmps = np.asarray(trace.data, dtype=np.float64) * 1.0e9
    if not np.isfinite(corrected_nmps).all():
        raise RuntimeError(f"non-finite response output {channel.name}.{channel.component}")
    return corrected_nmps, quantization_residual


def set_output_headers(trace: Any, component: str, orientation_status: str, response_kind: str, gain: float, lsb: float) -> None:
    trace.stats.starttime -= 9 * 3600.0
    trace.stats.channel = component
    trace.stats.sac.cmpaz = 0.0 if component in {"U", "N"} else 90.0
    trace.stats.sac.cmpinc = 0.0 if component == "U" else 90.0
    trace.stats.sac.kuser0 = "RESPVEL"
    trace.stats.sac.kuser1 = response_kind.upper()
    trace.stats.sac.kuser2 = "ORIENTED" if orientation_status == "official_true_NE" else ("VERTICAL" if component == "U" else "ORNMISS")
    trace.stats.sac.user0 = float(gain)
    trace.stats.sac.user1 = float(lsb)


def process_group(
    event_id: str, group: int, out_root: Path, orientations: dict[str, list[dict[str, Any]]],
    templates: dict[str, Any], response_cache: dict[tuple[str, float], Any], convert_workers: int,
) -> dict[str, Any]:
    event_dir, source_variant = choose_event_dir(group, event_id)
    ch_path = next(iter(event_dir.glob("**/raw/*.ch")))
    cnt_path = next(iter(event_dir.glob("**/raw/*.cnt")))
    existing_sac_dir = next(iter(event_dir.glob("**/sac")))
    when = cnt_time(cnt_path)
    channels = win32.read_ctable(str(ch_path))
    channels.sort(key=lambda c: (c.name, c.component))
    if len(channels) != 72:
        raise RuntimeError(f"expected 72 channels {event_id}/group{group}, got {len(channels)}")
    outdir = out_root / event_id / f"group{group}"
    marker = outdir / "_group_complete.audit.json"
    if marker.exists() and len(list(outdir.glob("*.SAC"))) == 72:
        return json.loads(marker.read_text(encoding="utf-8"))
    if outdir.exists() and list(outdir.glob("*.SAC")):
        raise RuntimeError(f"partial output requires manual review, will not overwrite: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"hinet_v1106_{event_id}_g{group}_", dir="/tmp") as tmp_name:
        tmp = Path(tmp_name)
        prm = tmp / "win.prm"
        prm.write_text(f".\n{ch_path}\n.\n.\n", encoding="utf-8")
        with ThreadPoolExecutor(max_workers=convert_workers) as executor:
            converted = list(executor.map(lambda c: official_convert_one(OFFICIAL_BIN, cnt_path, c, tmp, prm), channels))
        converted_map = {(c.name, c.component): p for c, p in zip(channels, converted)}
        channel_map = {(c.name, c.component): c for c in channels}

        for station in sorted({c.name for c in channels}):
            traces: dict[str, Any] = {}
            corrected: dict[str, np.ndarray] = {}
            quant_residuals: dict[str, float] = {}
            kinds: dict[str, str] = {}
            official_hashes: dict[str, str] = {}
            existing_hashes: dict[str, str] = {}
            for component in ["U", "N", "E"]:
                channel = channel_map[(station, component)]
                path = converted_map[(station, component)]
                official_hashes[component] = sha256(path)
                current_path = existing_sac_dir / f"{station}.{component}.SAC"
                existing_hashes[component] = sha256(current_path)
                trace = read(str(path))[0]
                kind = response_type(float(channel.lsb_value))
                cache_key = (kind, float(channel.gain))
                if cache_key not in response_cache:
                    response_cache[cache_key] = configured_response(templates[kind], float(channel.gain))
                values, residual = response_correct(trace, channel, response_cache[cache_key])
                traces[component] = trace
                corrected[component] = values
                quant_residuals[component] = residual
                kinds[component] = kind

            orientation = applicable_orientation(orientations.get(station, []), when)
            if orientation is not None:
                theta_rad = math.radians(float(orientation["theta_deg"]))
                n_inst, e_inst = corrected["N"], corrected["E"]
                corrected["N"] = math.cos(theta_rad) * n_inst - math.sin(theta_rad) * e_inst
                corrected["E"] = math.sin(theta_rad) * n_inst + math.cos(theta_rad) * e_inst
                horizontal_status = "official_true_NE"
            else:
                horizontal_status = "quarantined_official_orientation_unavailable"

            for component in ["U", "N", "E"]:
                channel = channel_map[(station, component)]
                trace = traces[component]
                status = "vertical_not_rotated" if component == "U" else horizontal_status
                trace.data = corrected[component].astype(np.float32)
                set_output_headers(trace, component, "official_true_NE" if status == "official_true_NE" else status, kinds[component], float(channel.gain), float(channel.lsb_value))
                output = outdir / f"{station}.{component}.SAC"
                trace.write(str(output), format="SAC", byteorder="<")
                output_trace = read(str(output), headonly=True)[0]
                records.append({
                    "event_id": event_id, "group": group, "station": station, "component": component,
                    "source_variant": source_variant,
                    "source_CH": str(ch_path.relative_to(ROOT)), "source_CNT": str(cnt_path.relative_to(ROOT)),
                    "official_transient_SAC_sha256": official_hashes[component],
                    "existing_SAC_sha256": existing_hashes[component],
                    "official_transient_equals_existing": official_hashes[component] == existing_hashes[component],
                    "response_type": kinds[component], "CH_gain": float(channel.gain), "CH_lsb": float(channel.lsb_value),
                    "max_fractional_count_reconstruction_error": quant_residuals[component],
                    "UTC_shift_s": -32400.0,
                    "orientation_status": status,
                    "official_orientation": orientation,
                    "research_deconvolution": {"pre_filt_hz": PREFILT, "water_level_db": WATER_LEVEL_DB, "zero_mean": True, "taper": True, "taper_fraction": 0.05},
                    "output": str(output.relative_to(ROOT)), "output_sha256": sha256(output),
                    "output_starttime": str(output_trace.stats.starttime), "sampling_rate_hz": float(output_trace.stats.sampling_rate), "npts": int(output_trace.stats.npts),
                    "finite": bool(np.isfinite(trace.data).all()), "output_rms_nmps": float(np.sqrt(np.mean(np.asarray(trace.data, dtype=np.float64) ** 2))),
                })

    group_summary = {
        "schema": "hinet-official-calibrated-group-v1106",
        "event_id": event_id, "group": group, "source_variant": source_variant,
        "record_count": len(records),
        "official_transient_equals_existing_count": sum(bool(r["official_transient_equals_existing"]) for r in records),
        "finite_count": sum(bool(r["finite"]) for r in records),
        "official_true_NE_horizontal_count": sum(r["orientation_status"] == "official_true_NE" for r in records),
        "quarantined_horizontal_count": sum(str(r["orientation_status"]).startswith("quarantined") for r in records),
        "max_fractional_count_reconstruction_error": max(float(r["max_fractional_count_reconstruction_error"]) for r in records),
        "records": records,
        "complete": len(records) == 72 and all(bool(r["finite"]) for r in records),
    }
    atomic_json(marker, group_summary)
    return group_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--convert-workers", type=int, default=4)
    parser.add_argument("--max-event-groups", type=int, default=0)
    args = parser.parse_args()
    required = [COHORT, ORIENTATION_CSV, OFFICIAL_BIN, *RESP_PATHS.values()]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    events = list(cohort["complete_event_ids"])
    orientations = load_orientations()
    templates, response_metadata = build_response_templates()
    response_cache: dict[tuple[str, float], Any] = {}
    groups: list[dict[str, Any]] = []
    targets = [(event_id, group) for event_id in events for group in [0, 1]]
    if args.max_event_groups > 0:
        targets = targets[:args.max_event_groups]
    started = datetime.now(UTC)
    for index, (event_id, group) in enumerate(targets, 1):
        result = process_group(event_id, group, args.output_root, orientations, templates, response_cache, max(1, args.convert_workers))
        groups.append(result)
        progress = {
            "schema": "hinet-official-calibration-progress-v1106",
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "status": "running" if index < len(targets) else "complete",
            "completed_event_groups": index, "total_event_groups": len(targets),
            "completed_sac": sum(int(g["record_count"]) for g in groups),
            "expected_sac": len(targets) * 72,
            "current": {"event_id": event_id, "group": group},
            "output_root": str(args.output_root),
        }
        atomic_json(PROGRESS, progress)
        print(f"{index}/{len(targets)} {event_id} group{group} records={result['record_count']}", flush=True)

    records = [record for group in groups for record in group["records"]]
    expected = len(targets) * 72
    orientation_counts = Counter(str(r["orientation_status"]) for r in records)
    response_counts = Counter(str(r["response_type"]) for r in records)
    missing_stations = sorted({str(r["station"]) for r in records if str(r["orientation_status"]).startswith("quarantined")})
    summary = {
        "expected_sac": expected, "generated_sac": len(records),
        "complete_and_finite": len(records) == expected and all(bool(r["finite"]) for r in records),
        "official_transient_equals_existing": sum(bool(r["official_transient_equals_existing"]) for r in records),
        "response_type_counts": dict(response_counts),
        "orientation_status_counts": dict(orientation_counts),
        "quarantined_horizontal_stations": missing_stations,
        "max_fractional_count_reconstruction_error": max(float(r["max_fractional_count_reconstruction_error"]) for r in records),
        "sampling_rate_counts": dict(Counter(str(r["sampling_rate_hz"]) for r in records)),
        "npts_counts": dict(Counter(str(r["npts"]) for r in records)),
        "elapsed_s": (datetime.now(UTC) - started).total_seconds(),
    }
    full_cohort = len(targets) == 26
    decision = "PASS_WITH_52_HORIZONTAL_TRACES_QUARANTINED" if full_cohort and summary["complete_and_finite"] and len(missing_stations) == 2 else ("PASS" if summary["complete_and_finite"] else "FAIL")
    audit = {
        "schema": "hinet-official-calibration-audit-stage-d-v1106",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": "D_isolated_corrected_regeneration",
        "normative_sources": {
            "win2sac": "logs/nied_official_audit_v1096/sources/auth_manual_dlDialogue.php_4fb5dfa628.bin",
            "orientation": str(ORIENTATION_CSV.relative_to(ROOT)),
            "response_type3": str(RESP_PATHS["type3"].relative_to(ROOT)),
            "response_type4": str(RESP_PATHS["type4"].relative_to(ROOT)),
            "time_basis": "https://www.hinet.bosai.go.jp/strace/ (official page labels Date and Time as JST)",
        },
        "official_processing": {
            "source": "original CNT plus matching CH",
            "conversion": "NIED official downloaded win2sac_32 v2.50, -e",
            "response_selection": "CH LSB: 1.023e-7=type3, 1.021e-7=type4",
            "response_gain_replacement": "stage1 gain=CH sensor gain/A0; final sensitivity=stage1 gain*official digitizer gain",
            "orientation": "official station and validity-period theta; rotate instrument N/E into true geographic N/E",
            "time": "SAC source header JST minus 32400 s to UTC",
        },
        "research_processing_not_official_spec": {
            "response_removal": {"output_unit": "nm/s", "pre_filt_hz": PREFILT, "water_level_db": WATER_LEVEL_DB, "zero_mean": True, "taper": True, "taper_fraction": 0.05},
            "note": "These stabilization settings are versioned research choices, not NIED-prescribed values."
        },
        "official_response_template_metadata": response_metadata,
        "output_root": str(args.output_root.relative_to(ROOT) if args.output_root.is_relative_to(ROOT) else args.output_root),
        "summary": summary,
        "groups": [{key: value for key, value in group.items() if key != "records"} for group in groups],
        "decision": decision,
        "old_version_overwritten": False,
        "downstream_switch_allowed": False,
        "known_structure_used": False,
        "machine_learning_used": False,
        "new_requirements": [
            {
                "proposal_id": "NR-HINET-ORIENT-001",
                "evidence": "NIED national orientation CSV has no applicable theta for N.SSGH and N.SSWH in the 2026 cohort.",
                "impact_scope": "52 horizontal traces (2 stations x 13 events x N/E); vertical traces remain valid.",
                "urgency": "high_before_directional_downstream_switch",
                "dependencies": "NIED clarification or independently measured orientation with provenance; no imputation allowed in this official audit.",
                "recommended_separate_owner": "waveform-metadata/orientation specialist"
            }
        ] if missing_stations else [],
    }
    atomic_json(AUDIT_JSON, audit)
    md = f"""# Hi-net公式補正監査 段階D（v1106）

判定: **{decision}**

- 元CNT/CHから公式配布`win2sac_32`を再実行し、既存SACと一致したもの: {summary['official_transient_equals_existing']}/{expected}。
- 公式Type3/4 RESP、CH成分別ゲイン置換、決定論的JST→UTC、公式方位回転後の有限SAC: {len(records)}/{expected}。
- 応答タイプ: `{json.dumps(summary['response_type_counts'], ensure_ascii=False)}`。
- 方位状態: `{json.dumps(summary['orientation_status_counts'], ensure_ascii=False)}`。
- 公式方位欠測のため方向解析から隔離: `{', '.join(missing_stations) if missing_stations else 'なし'}`。
- 最大カウント再構成端数誤差: {summary['max_fractional_count_reconstruction_error']:.6g} count。
- 所要時間: {summary['elapsed_s']:.1f} s。

研究上の追加処理は、応答逆フィルタ安定化のpre-filter={PREFILT} Hz、water level={WATER_LEVEL_DB} dB、demean/taperであり、NIED公式仕様と分離して記録した。旧成果物、Web、後段入力は変更していない。方位欠測52水平成分が解決し、段階Eの旧新差分・後段影響がPASSするまでは入力切替不可。
"""
    AUDIT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"decision": decision, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
