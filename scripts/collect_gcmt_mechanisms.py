#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml
from obspy import read_events  # type: ignore

BASE_NDK_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec20.ndk.gz"
MONTHLY_BASE_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY"
MONTH_CODES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _download(url: str, output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return True
    try:
        request = Request(url, headers={"Connection": "close", "User-Agent": "crust-lite/0.1"})
        with urlopen(request, timeout=120) as response:  # noqa: S310 - user-triggered data acquisition
            output.write_bytes(response.read())
        return True
    except Exception as exc:
        print(f"skip_download url={url} error={type(exc).__name__}: {exc}")
        return False


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)


def _event_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _time_seconds(value: datetime) -> float:
    return value.replace(tzinfo=timezone.utc).timestamp()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_event_index(rows: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        try:
            item = {
                **row,
                "_time_s": _time_seconds(_parse_time(str(row["time_utc"]))),
                "_lat": float(row["lat"]),
                "_lon": float(row["lon"]),
                "_mag": float(row["magnitude"]),
            }
        except Exception:
            continue
        enriched.append(item)
    enriched.sort(key=lambda row: float(row["_time_s"]))
    return [float(row["_time_s"]) for row in enriched], enriched


def _match_event(
    origin_time: datetime,
    lat: float,
    lon: float,
    mag: float,
    times: list[float],
    rows: list[dict[str, Any]],
    max_time_s: float = 900.0,
    max_distance_km: float = 250.0,
) -> tuple[str | None, float | None, float | None]:
    center = _time_seconds(origin_time)
    lo = bisect_left(times, center - max_time_s)
    hi = bisect_right(times, center + max_time_s)
    best: tuple[float, str, float, float] | None = None
    for row in rows[lo:hi]:
        distance = _haversine_km(lat, lon, float(row["_lat"]), float(row["_lon"]))
        if distance > max_distance_km:
            continue
        dt = abs(float(row["_time_s"]) - center)
        dm = abs(float(row["_mag"]) - mag)
        score = dt / 60.0 + distance / 25.0 + dm * 5.0
        if best is None or score < best[0]:
            best = (score, str(row["event_id"]), dt, distance)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def _iter_ndk_files(raw_dir: Path, start: date, end: date) -> list[Path]:
    files: list[Path] = []
    base = raw_dir / "jan76_dec20.ndk.gz"
    if end >= date(1976, 1, 1) and start <= date(2020, 12, 31):
        if _download(BASE_NDK_URL, base):
            files.append(base)
    cursor = max(start, date(2021, 1, 1))
    cursor = date(cursor.year, cursor.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        yy = str(cursor.year)[-2:]
        code = MONTH_CODES[cursor.month - 1]
        url = f"{MONTHLY_BASE_URL}/{cursor.year}/{code}{yy}.ndk"
        out = raw_dir / str(cursor.year) / f"{code}{yy}.ndk"
        if _download(url, out):
            files.append(out)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return files


def _read_catalog(path: Path):
    if path.suffix == ".gz":
        plain = path.with_suffix("")
        if not plain.exists():
            plain.write_bytes(gzip.decompress(path.read_bytes()))
        return read_events(str(plain), format="NDK")
    return read_events(str(path), format="NDK")


def collect(config_path: Path, output: Path, raw_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    region = config["region"]
    sources = config["data_sources"]
    start = date.fromisoformat(str(region["start_date"]))
    end = date.fromisoformat(str(region["end_date"]))
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in region["bbox"]]
    event_csv = Path(str(sources["event_csv"]))
    if not event_csv.is_absolute():
        event_csv = config_path.resolve().parents[1] / event_csv
    event_times, event_rows = _build_event_index(_event_rows(event_csv))
    files = _iter_ndk_files(raw_dir, start, end)
    rows: list[dict[str, Any]] = []
    unmatched = 0
    for ndk_file in files:
        try:
            catalog = _read_catalog(ndk_file)
        except Exception as exc:
            print(f"skip_parse file={ndk_file} error={type(exc).__name__}: {exc}")
            continue
        for event in catalog:
            origin = event.preferred_origin() or (event.origins[-1] if event.origins else None)
            focal = event.preferred_focal_mechanism() or (
                event.focal_mechanisms[0] if event.focal_mechanisms else None
            )
            magnitude = event.preferred_magnitude() or (event.magnitudes[-1] if event.magnitudes else None)
            if origin is None or focal is None or magnitude is None:
                continue
            lat = float(origin.latitude)
            lon = float(origin.longitude)
            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                continue
            origin_time = origin.time.datetime.replace(tzinfo=None)
            if not (start <= origin_time.date() <= end):
                continue
            mag = float(magnitude.mag or 0.0)
            event_id, dt_s, dist_km = _match_event(origin_time, lat, lon, mag, event_times, event_rows)
            if event_id is None:
                unmatched += 1
                continue
            planes = focal.nodal_planes
            np1 = planes.nodal_plane_1 if planes else None
            np2 = planes.nodal_plane_2 if planes else None
            if np1 is None or np2 is None:
                continue
            moment = 0.0
            if focal.moment_tensor and focal.moment_tensor.scalar_moment:
                moment = float(focal.moment_tensor.scalar_moment)
            rid = str(event.resource_id or f"gcmt_{event_id}")
            rows.append(
                {
                    "mechanism_id": rid.rsplit("/", 1)[-1],
                    "event_id": event_id,
                    "strike1": float(np1.strike),
                    "dip1": float(np1.dip),
                    "rake1": float(np1.rake),
                    "strike2": float(np2.strike),
                    "dip2": float(np2.dip),
                    "rake2": float(np2.rake),
                    "scalar_moment_nm": moment,
                    "source": f"Global CMT matched dt_s={dt_s:.1f} distance_km={dist_km:.1f}",
                }
            )
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[str(row["event_id"])] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mechanism_id",
        "event_id",
        "strike1",
        "dip1",
        "rake1",
        "strike2",
        "dip2",
        "rake2",
        "scalar_moment_nm",
        "source",
    ]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(dedup.values(), key=lambda row: str(row["event_id"])))
    meta = {
        "config": str(config_path),
        "source": "Global CMT NDK catalog",
        "source_urls": [BASE_NDK_URL, MONTHLY_BASE_URL],
        "output": str(output),
        "raw_dir": str(raw_dir),
        "ndk_file_count": len(files),
        "mechanism_count": len(dedup),
        "unmatched_gcmt_events": unmatched,
        "matching_policy": "nearest USGS event within 900 s and 250 km; score uses time, distance, magnitude",
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/gcmt"))
    args = parser.parse_args()
    collect(args.config, args.output, args.raw_dir)


if __name__ == "__main__":
    main()
