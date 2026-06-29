#!/usr/bin/env python3
"""Best-effort USGS focal-mechanism collector.

The script queries detail products in batches and appends successful mechanism
rows.  It is deliberately resumable because product coverage is sparse and
network/API failures should not discard already collected rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

USGS_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_LIMIT = 20_000


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def request_json(url: str, retries: int = 5) -> dict[str, Any]:
    """Fetch JSON with bounded retries for public API instability."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"Connection": "close", "User-Agent": "crust-lite/0.1"})
            with urlopen(request, timeout=30) as response:  # noqa: S310 - user-triggered data acquisition
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except HTTPError as exc:  # pragma: no cover - network path
            last_error = exc
            if exc.code == 429:
                time.sleep(min(30 * attempt, 180))
            else:
                time.sleep(min(2 * attempt, 8))
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"USGS request failed after {retries} attempts: {last_error}; url={url}")


def query_summary(config: dict[str, Any], min_magnitude: float) -> list[dict[str, Any]]:
    region = config["region"]
    filters = config["filters"]
    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    params = {
        "format": "geojson",
        "starttime": str(region["start_date"]),
        "endtime": (parse_date(str(region["end_date"])) + timedelta(days=1)).isoformat(),
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "minmagnitude": min_magnitude,
        "maxdepth": float(filters["max_depth_km"]),
        "producttype": "moment-tensor",
        "orderby": "time-asc",
        "limit": USGS_LIMIT,
    }
    data = request_json(f"{USGS_QUERY}?{urlencode(params)}")
    features = list(data.get("features", []))
    if len(features) >= USGS_LIMIT:
        raise RuntimeError("USGS mechanism query reached limit; add time splitting before using this result")
    return features


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _product_time(product: dict[str, Any]) -> int:
    return int(product.get("updateTime") or product.get("preferredWeight") or 0)


def mechanism_from_detail(feature: dict[str, Any]) -> dict[str, Any] | None:
    props = feature.get("properties", {})
    detail_url = props.get("detail")
    if not detail_url:
        return None
    detail = request_json(str(detail_url))
    detail_props = detail.get("properties", {})
    products = detail_props.get("products", {})
    candidates = list(products.get("moment-tensor", [])) + list(products.get("focal-mechanism", []))
    if not candidates:
        return None
    candidates.sort(key=_product_time, reverse=True)
    for product in candidates:
        product_props = product.get("properties", {})
        if "nodal-plane-1-strike" not in product_props:
            continue
        source = product.get("source") or product_props.get("eventsource") or "usgs"
        code = product.get("code") or product_props.get("eventsourcecode") or feature.get("id")
        scalar_moment = _float(product_props.get("scalar-moment"), 0.0)
        return {
            "mechanism_id": f"{source}_{code}",
            "event_id": str(feature.get("id")),
            "strike1": _float(product_props.get("nodal-plane-1-strike")),
            "dip1": _float(product_props.get("nodal-plane-1-dip")),
            "rake1": _float(product_props.get("nodal-plane-1-rake")),
            "strike2": _float(product_props.get("nodal-plane-2-strike")),
            "dip2": _float(product_props.get("nodal-plane-2-dip")),
            "rake2": _float(product_props.get("nodal-plane-2-rake")),
            "scalar_moment_nm": scalar_moment,
            "source": f"USGS ComCat {product.get('type', 'mechanism')}",
        }
    return None


def _existing_event_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()
    with output.open("r", encoding="utf-8", newline="") as fh:
        return {str(row.get("event_id")) for row in csv.DictReader(fh) if row.get("event_id")}


def _append_rows(output: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    # Append-only collection keeps long-running nationwide jobs resumable.
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def collect(
    config_path: Path,
    output: Path,
    min_magnitude: float,
    workers: int,
    batch_size: int = 100,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    features = query_summary(config, min_magnitude=min_magnitude)
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
    existing = _existing_event_ids(output)
    pending = [feature for feature in features if str(feature.get("id")) not in existing]
    failures: list[dict[str, str]] = []
    written = len(existing)
    print(f"mechanism summary events={len(features)}, existing={len(existing)}, pending={len(pending)}")
    for batch_index, batch in enumerate(_chunks(pending, max(1, batch_size)), start=1):
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(mechanism_from_detail, feature): feature for feature in batch}
            for future in as_completed(futures):
                feature = futures[future]
                try:
                    row = future.result()
                    if row:
                        rows.append(row)
                except Exception as exc:  # pragma: no cover - network path
                    failures.append({"event_id": str(feature.get("id")), "error": f"{type(exc).__name__}: {exc}"})
        rows.sort(key=lambda row: str(row["event_id"]))
        _append_rows(output, rows, fieldnames)
        written += len(rows)
        print(
            f"mechanism batch {batch_index}: fetched={min(batch_index * batch_size, len(pending))}/{len(pending)}, "
            f"written={written}, batch_rows={len(rows)}, failures={len(failures)}",
            flush=True,
        )
    meta = {
        "config": str(config_path),
        "output": str(output),
        "source": "USGS ComCat detail products: moment-tensor/focal-mechanism",
        "region": config["region"]["name"],
        "bbox": config["region"]["bbox"],
        "start_date": str(config["region"]["start_date"]),
        "end_date": str(config["region"]["end_date"]),
        "min_magnitude": min_magnitude,
        "queried_event_count": len(features),
        "pre_existing_count": len(existing),
        "mechanism_count": written,
        "failure_count": len(failures),
        "failures": failures[:100],
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-magnitude", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    meta = collect(args.config, args.output, args.min_magnitude, args.workers, batch_size=args.batch_size)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
