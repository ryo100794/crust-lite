#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
USGS_LIMIT = 20000


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year, 12, 31))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def fetch_chunk(params: dict[str, object], retries: int = 3) -> list[dict[str, str]]:
    url = f"{USGS_QUERY}?{urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=120) as response:  # noqa: S310 - user-triggered data acquisition
                text = response.read().decode("utf-8")
            return [dict(row) for row in csv.DictReader(text.splitlines())]
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"USGS request failed after {retries} attempts: {last_error}; url={url}")


def normalise_usgs_row(row: dict[str, str], source: str) -> dict[str, str]:
    event_id = row.get("id") or row.get("net", "usgs") + row.get("code", "unknown")
    return {
        "event_id": event_id,
        "time_utc": row["time"].replace(".000Z", "Z"),
        "lat": row["latitude"],
        "lon": row["longitude"],
        "depth_km": row["depth"],
        "magnitude": row["mag"],
        "magnitude_type": row.get("magType", ""),
        "catalog_source": source,
    }


def _query_params(
    base: dict[str, object],
    chunk_start: date,
    chunk_end: date,
    limit: int,
) -> dict[str, object]:
    return {
        **base,
        "starttime": chunk_start.isoformat(),
        "endtime": (chunk_end + timedelta(days=1)).isoformat(),
        "orderby": "time-asc",
        "limit": limit,
    }


def fetch_range_split(
    base_params: dict[str, object],
    chunk_start: date,
    chunk_end: date,
    raw_dir: Path,
    limit: int = USGS_LIMIT,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    params = _query_params(base_params, chunk_start, chunk_end, limit)
    rows = fetch_chunk(params)
    if len(rows) >= limit and chunk_start < chunk_end:
        span_days = (chunk_end - chunk_start).days
        mid = chunk_start + timedelta(days=max(0, span_days // 2))
        left_rows, left_meta = fetch_range_split(base_params, chunk_start, mid, raw_dir, limit=limit)
        right_rows, right_meta = fetch_range_split(base_params, mid + timedelta(days=1), chunk_end, raw_dir, limit=limit)
        return left_rows + right_rows, left_meta + right_meta
    name = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
    (raw_dir / name).write_text(
        json.dumps({"params": params, "count": len(rows), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return rows, [
        {
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "row_count": len(rows),
            "limit": limit,
            "file": str(raw_dir / name),
        }
    ]


def collect(config_path: Path, output: Path, min_magnitude: float | None = None) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    region = config["region"]
    filters = config["filters"]
    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    start = parse_date(str(region["start_date"]))
    end = parse_date(str(region["end_date"]))
    minmag = float(min_magnitude if min_magnitude is not None else filters["min_magnitude"])
    maxdepth = float(filters["max_depth_km"])
    base_params = {
        "format": "csv",
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "minmagnitude": minmag,
        "maxdepth": maxdepth,
    }
    rows: list[dict[str, str]] = []
    chunk_metadata: list[dict[str, object]] = []
    raw_dir = output.parent / "raw_chunks" / output.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start, chunk_end in year_chunks(start, end):
        chunk_rows, chunk_meta = fetch_range_split(base_params, chunk_start, chunk_end, raw_dir)
        rows.extend(
            normalise_usgs_row(row, "USGS ComCat")
            for row in chunk_rows
            if row.get("type", "earthquake") == "earthquake"
            and row.get("mag") not in {None, ""}
            and row.get("depth") not in {None, ""}
        )
        chunk_metadata.extend(chunk_meta)
        print(f"{region['name']} {chunk_start.year}: {len(chunk_rows)} rows in {len(chunk_meta)} chunk(s)")
    seen: set[str] = set()
    unique_rows = []
    for row in rows:
        if row["event_id"] in seen:
            continue
        seen.add(row["event_id"])
        unique_rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "event_id",
                "time_utc",
                "lat",
                "lon",
                "depth_km",
                "magnitude",
                "magnitude_type",
                "catalog_source",
            ],
        )
        writer.writeheader()
        writer.writerows(unique_rows)
    meta = {
        "config": str(config_path),
        "output": str(output),
        "source": "USGS ComCat FDSN event query.csv",
        "region": region["name"],
        "bbox": region["bbox"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "min_magnitude": minmag,
        "max_depth_km": maxdepth,
        "event_count": len(unique_rows),
        "raw_chunk_dir": str(raw_dir),
        "request_limit": USGS_LIMIT,
        "request_chunks": chunk_metadata,
        "split_on_limit": True,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-magnitude", type=float, default=None)
    args = parser.parse_args()
    meta = collect(args.config, args.output, min_magnitude=args.min_magnitude)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
