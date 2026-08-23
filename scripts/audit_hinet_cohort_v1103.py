#!/usr/bin/env python3
"""Complete header/completeness audit for the 13x48x3 Hi-net cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from HinetPy import win32
from obspy import read


ROOT = Path("/workspace/equake/crust-lite")
COHORT = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/strict_completion_20260822.json"
RAW_ROOT = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/raw"
ORIENTATION_CSV = ROOT / "logs/nied_official_audit_v1100/sources/national_csv.csv"
OUT_JSON = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_C_v1103.json"
OUT_MD = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_C_v1103.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_orientation_rows() -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    with ORIENTATION_CSV.open("r", encoding="euc_jp", newline="") as fh:
        for row in csv.reader(fh, quotechar="'"):
            if len(row) < 7:
                continue
            try:
                theta = float(row[3])
            except ValueError:
                continue
            out[row[2]].append({
                "theta_deg": theta,
                "valid_start": row[4] or None,
                "valid_end": row[5] or None,
                "method_flag": row[6] or None,
            })
    return out


def applicable_orientation(rows: list[dict[str, object]], when: datetime) -> dict[str, object] | None:
    for row in rows:
        start = datetime.fromisoformat(str(row["valid_start"])) if row["valid_start"] else None
        end = datetime.fromisoformat(str(row["valid_end"])) if row["valid_end"] else None
        if start and when < start:
            continue
        if end and when > end:
            continue
        return row
    return None


def response_type(lsb: float) -> str:
    values = [(1.021e-7, "type4"), (1.023e-7, "type3"), (1.000e-7, "type2"), (1.192e-7, "type1")]
    for value, label in values:
        if math.isclose(lsb, value, rel_tol=0.0, abs_tol=5e-12):
            return label
    return "unknown"


def choose_event_dir(group: int, event_id: str) -> tuple[Path, str]:
    normal = RAW_ROOT / f"group{group}_raw/2026/{event_id}"
    retry = RAW_ROOT / f"group{group}_retry_latest_raw/2026/{event_id}"
    for path, label in [(retry, "retry_latest"), (normal, "normal")]:
        if len(list(path.glob("**/sac/*.SAC"))) == 72:
            return path, label
    return normal, "incomplete"


def parse_cnt_stamp(path: Path) -> datetime | None:
    match = re.search(r"(20\d{10})", path.name)
    return datetime.strptime(match.group(1), "%Y%m%d%H%M") if match else None


def special_caution(station: str, when: datetime) -> dict[str, object] | None:
    if station == "N.KNHH" and datetime(2003, 3, 19) <= when <= datetime(2008, 2, 19, 23, 59, 59):
        return {"kind": "velocity_EW_possible_polarity_reversal", "official_source": "NIED orientation_results"}
    if station == "N.FBRH" and when >= datetime(2008, 7, 28):
        return {"kind": "velocity_horizontal_components_swapped_requires_one_polarity_flip_then_90deg_rotation", "official_source": "NIED orientation_results"}
    if station in {"N.NGTH", "N.KAHH"}:
        return {"kind": "strong_motion_orientation_period_caution_not_Hi-net_velocity", "official_source": "NIED orientation_results"}
    if station == "N.YKSH":
        return {"kind": "orientation_may_be_unavailable_due_to_missing_high_sensitivity_accelerometer", "official_source": "NIED orientation_results"}
    return None


def main() -> None:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    events = list(cohort["complete_event_ids"])
    orientations = load_orientation_rows()
    records: list[dict[str, object]] = []
    event_summaries: list[dict[str, object]] = []
    station_sets_by_group: dict[int, set[str]] = {0: set(), 1: set()}
    duplicate_keys: list[str] = []
    missing_keys: list[str] = []

    for event_id in events:
        group_stations: dict[int, set[str]] = {}
        event_record_start = len(records)
        for group in [0, 1]:
            event_dir, source_variant = choose_event_dir(group, event_id)
            ch_files = list(event_dir.glob("**/raw/*.ch"))
            cnt_files = list(event_dir.glob("**/raw/*.cnt"))
            sac_files = list(event_dir.glob("**/sac/*.SAC"))
            if len(ch_files) != 1 or len(cnt_files) != 1:
                missing_keys.append(f"{event_id}/group{group}: CH={len(ch_files)} CNT={len(cnt_files)}")
                continue
            ch_path, cnt_path = ch_files[0], cnt_files[0]
            cnt_stamp = parse_cnt_stamp(cnt_path)
            if cnt_stamp is None:
                missing_keys.append(f"{event_id}/group{group}: CNT timestamp unavailable")
                continue
            channels = win32.read_ctable(str(ch_path))
            channel_map = {(c.name, c.component): c for c in channels}
            sac_map: dict[tuple[str, str], list[Path]] = defaultdict(list)
            for path in sac_files:
                bits = path.name.split(".")
                if len(bits) >= 4:
                    sac_map[(".".join(bits[:2]), bits[2])].append(path)
            group_stations[group] = {name for name, _component in channel_map}
            station_sets_by_group[group].update(group_stations[group])
            expected_keys = set(channel_map)
            for key, paths in sac_map.items():
                if len(paths) > 1:
                    duplicate_keys.append(f"{event_id}/group{group}/{key[0]}.{key[1]}")
            for key in sorted(expected_keys):
                paths = sac_map.get(key, [])
                if len(paths) != 1:
                    missing_keys.append(f"{event_id}/group{group}/{key[0]}.{key[1]}: SAC={len(paths)}")
                    continue
                path = paths[0]
                channel = channel_map[key]
                trace = read(str(path), headonly=True)[0]
                sac = trace.stats.sac
                component = channel.component
                nominal_az = 0.0 if component in {"U", "N"} else 90.0
                nominal_inc = 0.0 if component == "U" else 90.0
                orientation = applicable_orientation(orientations.get(channel.name, []), cnt_stamp)
                actual_az: float | None = None
                if component == "U":
                    actual_az = 0.0
                elif orientation and component == "N":
                    actual_az = float(orientation["theta_deg"]) % 360.0
                elif orientation and component == "E":
                    actual_az = (float(orientation["theta_deg"]) + 90.0) % 360.0
                cmpaz = float(sac.get("cmpaz"))
                cmpinc = float(sac.get("cmpinc"))
                stla = float(sac.get("stla"))
                stlo = float(sac.get("stlo"))
                stel = float(sac.get("stel"))
                start_naive = trace.stats.starttime.datetime.replace(tzinfo=None)
                caution = special_caution(channel.name, cnt_stamp)
                records.append({
                    "event_id": event_id,
                    "group": group,
                    "source_variant": source_variant,
                    "station": channel.name,
                    "component": component,
                    "channel_id": channel.id,
                    "sac_path": str(path.relative_to(ROOT)),
                    "sac_sha256": sha256(path),
                    "ch_path": str(ch_path.relative_to(ROOT)),
                    "cnt_path": str(cnt_path.relative_to(ROOT)),
                    "unit": channel.unit,
                    "gain": float(channel.gain),
                    "period_s": float(channel.period),
                    "damping": float(channel.damping),
                    "preamplification_db": float(channel.preamplification),
                    "lsb_V_per_count": float(channel.lsb_value),
                    "response_type": response_type(float(channel.lsb_value)),
                    "sampling_rate_hz": float(trace.stats.sampling_rate),
                    "delta_s": float(trace.stats.delta),
                    "npts": int(trace.stats.npts),
                    "starttime_header": str(trace.stats.starttime),
                    "starttime_if_JST_then_UTC": str(trace.stats.starttime - timedelta(hours=9)),
                    "cnt_filename_time": cnt_stamp.isoformat(),
                    "header_start_matches_CNT_minute": start_naive.replace(second=0, microsecond=0) == cnt_stamp,
                    "stla": stla,
                    "stlo": stlo,
                    "stel": stel,
                    "CH_coordinates_match": math.isclose(stla, float(channel.latitude), abs_tol=1e-4) and math.isclose(stlo, float(channel.longitude), abs_tol=1e-4),
                    "cmpaz": cmpaz,
                    "cmpinc": cmpinc,
                    "nominal_orientation_match": math.isclose(cmpaz, nominal_az, abs_tol=1e-6) and math.isclose(cmpinc, nominal_inc, abs_tol=1e-6),
                    "official_orientation": orientation,
                    "actual_cmpaz_expected": actual_az,
                    "actual_orientation_match": actual_az is not None and math.isclose(cmpaz, actual_az, abs_tol=1e-6),
                    "official_special_caution": caution,
                })
        group0 = group_stations.get(0, set())
        group1 = group_stations.get(1, set())
        event_summaries.append({
            "event_id": event_id,
            "group0_station_count": len(group0),
            "group1_station_count": len(group1),
            "union_station_count": len(group0 | group1),
            "overlap_station_count": len(group0 & group1),
            "records": len(records) - event_record_start,
        })
        print(event_id, len(records) - event_record_start, flush=True)

    expected = len(events) * 48 * 3
    horizontal = [r for r in records if r["component"] in {"N", "E"}]
    horizontal_with_official = [r for r in horizontal if r["official_orientation"] is not None]
    horizontal_actual_mismatch = [r for r in horizontal_with_official if not r["actual_orientation_match"]]
    nonzero_orientation_stations = sorted({str(r["station"]) for r in horizontal_with_official if not math.isclose(float(r["official_orientation"]["theta_deg"]), 0.0, abs_tol=1e-6)})
    missing_orientation_stations = sorted({str(r["station"]) for r in horizontal if r["official_orientation"] is None})
    caution_records = [r for r in records if r["official_special_caution"] is not None]
    applicable_velocity_cautions = [r for r in caution_records if not str(r["official_special_caution"]["kind"]).startswith("strong_motion")]
    sampling_counts = Counter((r["sampling_rate_hz"], r["npts"]) for r in records)
    response_counts = Counter(str(r["response_type"]) for r in records)
    unit_counts = Counter(str(r["unit"]) for r in records)
    theta_by_station: dict[str, float | None] = {}
    for r in horizontal:
        theta_by_station.setdefault(str(r["station"]), None if r["official_orientation"] is None else float(r["official_orientation"]["theta_deg"]))
    summary = {
        "expected_sac": expected,
        "observed_unique_expected_sac": len(records),
        "completeness_pass": len(records) == expected and not missing_keys and not duplicate_keys,
        "events": len(events),
        "unique_stations_group0": len(station_sets_by_group[0]),
        "unique_stations_group1": len(station_sets_by_group[1]),
        "group_station_overlap": len(station_sets_by_group[0] & station_sets_by_group[1]),
        "unique_stations_union": len(station_sets_by_group[0] | station_sets_by_group[1]),
        "sampling_npts_counts": {f"{key[0]}Hz_{key[1]}pts": value for key, value in sampling_counts.items()},
        "all_CH_coordinates_match": all(bool(r["CH_coordinates_match"]) for r in records),
        "all_nominal_orientation_headers_match": all(bool(r["nominal_orientation_match"]) for r in records),
        "all_header_starts_match_CNT_minute": all(bool(r["header_start_matches_CNT_minute"]) for r in records),
        "unit_counts": dict(unit_counts),
        "response_type_counts": dict(response_counts),
        "horizontal_records": len(horizontal),
        "horizontal_with_official_orientation": len(horizontal_with_official),
        "horizontal_actual_orientation_mismatch": len(horizontal_actual_mismatch),
        "nonzero_orientation_station_count": len(nonzero_orientation_stations),
        "missing_orientation_stations": missing_orientation_stations,
        "official_special_caution_record_count": len(caution_records),
        "applicable_velocity_caution_record_count": len(applicable_velocity_cautions),
        "full_instrument_response_removed_count": 0,
    }
    audit = {
        "schema": "hinet-official-calibration-audit-stage-c-v1103",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stage": "C_all_1872_completeness_headers_orientation_time",
        "official_primary_sources_only_for_normative_requirements": True,
        "known_structure_used": False,
        "machine_learning_used": False,
        "existing_products_modified": False,
        "inputs": {
            "cohort_manifest": str(COHORT.relative_to(ROOT)),
            "cohort_manifest_sha256": sha256(COHORT),
            "orientation_csv": str(ORIENTATION_CSV.relative_to(ROOT)),
            "orientation_csv_sha256": sha256(ORIENTATION_CSV),
            "official_cautions": "logs/nied_official_audit_v1098/sources/orientation_results.html",
        },
        "event_summaries": event_summaries,
        "station_orientation_theta_deg": theta_by_station,
        "nonzero_orientation_stations": nonzero_orientation_stations,
        "missing_keys": missing_keys,
        "duplicate_keys": duplicate_keys,
        "summary": summary,
        "records": records,
        "decision": "FAIL_REQUIRES_ISOLATED_STAGE_D" if summary["completeness_pass"] and (len(horizontal_actual_mismatch) or missing_orientation_stations or not summary["all_header_starts_match_CNT_minute"] or summary["full_instrument_response_removed_count"] == 0) else "REVIEW",
        "old_version_overwritten": False,
        "downstream_switch_allowed": False,
        "new_requirements": []
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = f"""# Hi-net公式補正監査 段階C（v1103）

判定: **{audit['decision']}**

- 完全性: {summary['observed_unique_expected_sac']}/{summary['expected_sac']}、欠損{len(missing_keys)}、重複{len(duplicate_keys)}、**{'PASS' if summary['completeness_pass'] else 'FAIL'}**。
- 観測点: group0={summary['unique_stations_group0']}、group1={summary['unique_stations_group1']}、重複={summary['group_station_overlap']}、合計={summary['unique_stations_union']}。
- sampling/npts: `{json.dumps(summary['sampling_npts_counts'], ensure_ascii=False)}`。
- CH緯度経度一致: {summary['all_CH_coordinates_match']}。名目CMPAZ/CMPINC一致: {summary['all_nominal_orientation_headers_match']}。
- CNTファイル名時刻とSACヘッダ開始分の一致: {summary['all_header_starts_match_CNT_minute']}。ただしUTC標準化は未実施。
- 水平成分{summary['horizontal_records']}件中、公式方位あり{summary['horizontal_with_official_orientation']}件、実方位未反映{summary['horizontal_actual_orientation_mismatch']}件。非ゼロ方位の観測点は{summary['nonzero_orientation_station_count']}点。
- 公式方位が得られない観測点: `{', '.join(missing_orientation_stations) if missing_orientation_stations else 'なし'}`。
- 公式個別注意事項に該当する速度波形レコード: {summary['applicable_velocity_caution_record_count']}件。
- 応答タイプ: `{json.dumps(summary['response_type_counts'], ensure_ascii=False)}`。完全装置応答除去は0件。

全1872レコードの入力パス、SHA-256、CH値、SACヘッダ、公式方位、有効期間、判定はJSONに保存した。旧成果物・Web・後段入力は変更していない。
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
