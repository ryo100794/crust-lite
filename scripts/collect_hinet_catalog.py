#!/usr/bin/env python3
"""Collect authenticated Hi-net event catalog rows into crust-lite CSV.

Hi-net event times are handled as JST by HinetPy.  The output keeps the JST
origin as metadata and writes the pipeline time_utc column as UTC.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from collect_hinet_waveforms import _load_env_file

CATALOG_FIELDS = [
    "event_id",
    "time_utc",
    "lat",
    "lon",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "catalog_source",
    "time_jst",
    "hinet_evid",
    "hinet_name",
    "hinet_name_en",
    "is_authenticated_source",
]


def _parse_date(value: str | None, default: str) -> date:
    return date.fromisoformat(value or default)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return data


def _credentials(env_file: Path | None) -> tuple[str, str]:
    if env_file:
        _load_env_file(env_file)
    user = os.environ.get("HINET_USER") or os.environ.get("HINET_USERNAME")
    password = os.environ.get("HINET_PASSWORD") or os.environ.get("HINET_PASS")
    if not user or not password:
        raise RuntimeError("Hi-net credentials are missing; set HINET_USER and HINET_PASSWORD in the env file.")
    return user, password


def _utc_text_from_hinet_jst(origin: datetime) -> str:
    utc_dt = (origin - timedelta(hours=9)).replace(tzinfo=UTC)
    return utc_dt.isoformat().replace("+00:00", "Z")


def _inside_bbox(lat: float, lon: float, bbox: list[float] | tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _event_row(event: Any) -> dict[str, Any]:
    return {
        "event_id": f"hinet_{event.evid}",
        "time_utc": _utc_text_from_hinet_jst(event.origin),
        "lat": f"{float(event.latitude):.5f}",
        "lon": f"{float(event.longitude):.5f}",
        "depth_km": f"{float(event.depth):.2f}",
        "magnitude": f"{float(event.magnitude):.2f}",
        "magnitude_type": "M_jma_hinet",
        "catalog_source": "NIED Hi-net authenticated event catalog",
        "time_jst": event.origin.isoformat(sep=" "),
        "hinet_evid": event.evid,
        "hinet_name": event.name,
        "hinet_name_en": event.name_en,
        "is_authenticated_source": True,
    }


def _existing_dates(path: Path) -> set[str]:
    meta = path.with_suffix(path.suffix + ".metadata.json")
    if not meta.exists():
        return set()
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return {str(item.get("date")) for item in data.get("daily_counts", []) if item.get("status") == "ok"}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(Path(args.config))
    region = config["region"]
    bbox = region["bbox"]
    start = _parse_date(args.start_date, str(region["start_date"]))
    end = _parse_date(args.end_date, str(region["end_date"]))
    user, password = _credentials(Path(args.env_file) if args.env_file else None)
    from HinetPy import Client  # type: ignore

    client = Client(
        user,
        password,
        timeout=args.timeout_s,
        retries=args.retries,
        sleep_time_in_seconds=args.sleep_time_s,
        max_sleep_count=args.max_sleep_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_dates(output) if args.resume else set()
    write_header = not output.exists() or output.stat().st_size == 0
    seen: set[str] = set()
    if output.exists() and output.stat().st_size > 0:
        with output.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                seen.add(str(row.get("event_id", "")))

    total_rows = 0
    daily_counts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cursor = start
    with output.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        while cursor <= end:
            day_key = cursor.isoformat()
            if day_key in completed:
                cursor += timedelta(days=1)
                continue
            try:
                events = client._search_event_by_day(
                    cursor.year,
                    cursor.month,
                    cursor.day,
                    region=args.hinet_region,
                    magmin=args.min_magnitude,
                    magmax=args.max_magnitude,
                    include_unknown_mag=args.include_unknown_magnitude,
                )
                rows = []
                for event in events:
                    if float(event.magnitude) < args.min_magnitude:
                        continue
                    if args.max_depth_km is not None and float(event.depth) > args.max_depth_km:
                        continue
                    if not _inside_bbox(float(event.latitude), float(event.longitude), bbox):
                        continue
                    row = _event_row(event)
                    if row["event_id"] in seen:
                        continue
                    seen.add(row["event_id"])
                    rows.append(row)
                writer.writerows(rows)
                f.flush()
                total_rows += len(rows)
                daily_counts.append({"date": day_key, "status": "ok", "raw_events": len(events), "written_rows": len(rows)})
                print(f"{day_key} raw={len(events)} written={len(rows)} total={total_rows}", flush=True)
            except Exception as exc:
                item = {"date": day_key, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                failures.append(item)
                daily_counts.append(item)
                print(f"{day_key} error={item['error']}", flush=True)
            cursor += timedelta(days=1)
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    meta = {
        "config": str(args.config),
        "output": str(output),
        "source": "NIED Hi-net authenticated event catalog via HinetPy event search",
        "region": region["name"],
        "bbox": bbox,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "min_magnitude": args.min_magnitude,
        "max_magnitude": args.max_magnitude,
        "max_depth_km": args.max_depth_km,
        "hinet_region": args.hinet_region,
        "written_rows_this_run": total_rows,
        "unique_event_rows_in_file": len(seen),
        "daily_counts": daily_counts,
        "failures": failures[: args.max_failures_recorded],
        "credential_source": "environment_or_env_file_without_logging_values",
        "not_prediction": True,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default="/workspace/equake/secrets/hinet.env")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--hinet-region", default="00")
    parser.add_argument("--min-magnitude", type=float, default=2.0)
    parser.add_argument("--max-magnitude", type=float, default=9.9)
    parser.add_argument("--include-unknown-magnitude", action="store_true")
    parser.add_argument("--max-depth-km", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-time-s", type=float, default=5.0)
    parser.add_argument("--max-sleep-count", type=int, default=30)
    parser.add_argument("--max-failures-recorded", type=int, default=200)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
