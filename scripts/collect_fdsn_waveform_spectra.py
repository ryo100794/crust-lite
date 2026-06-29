#!/usr/bin/env python3
"""Collect public FDSN waveform windows and preserve phase-aware spectra.

This script is resumable and writes compact CSV summaries plus MiniSEED cache
files.  It keeps amplitude, phase, and group delay so later array projection can
use timing information rather than amplitude-only spectra.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from obspy import UTCDateTime  # type: ignore
from obspy.clients.fdsn import Client  # type: ignore
from obspy.geodetics import gps2dist_azimuth  # type: ignore

FREQUENCIES_HZ = [0.5, 1.0, 2.0, 4.0, 8.0]
SPECTRA_FIELDS = [
    "event_id",
    "station_id",
    "network",
    "station",
    "location",
    "channel",
    "time_utc",
    "lat",
    "lon",
    "frequency_hz",
    "amplitude",
    "phase_rad",
    "group_delay_s",
    "p_residual_s",
    "s_residual_s",
    "source",
]
FEATURE_FIELDS = [
    "event_id",
    "station_id",
    "channel",
    "pga",
    "pgv",
    "psa_0p3",
    "psa_1p0",
    "psa_3p0",
    "p_residual_s",
    "s_residual_s",
    "amp_residual_log",
    "source",
]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def _read_events(path: Path, min_mag: float, max_events: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    filtered = []
    for row in rows:
        try:
            row["_mag"] = float(row.get("magnitude", 0.0) or 0.0)
            row["_time"] = _parse_time(str(row["time_utc"]))
            row["_lat"] = float(row["lat"])
            row["_lon"] = float(row["lon"])
            row["_depth_km"] = float(row.get("depth_km", 0.0) or 0.0)
        except Exception:
            continue
        if row["_mag"] >= min_mag:
            filtered.append(row)
    filtered.sort(key=lambda row: (float(row["_mag"]), row["_time"]), reverse=True)
    return filtered[:max_events] if max_events > 0 else filtered


def _event_csv_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value = config.get("data_sources", {}).get("event_csv")
    if not value:
        raise ValueError("data_sources.event_csv must be configured for waveform collection")
    path = Path(str(value))
    if not path.is_absolute():
        path = config_path.resolve().parents[1] / path
    return path


def _existing_keys(path: Path) -> set[tuple[str, str, str, float]]:
    """Return already collected spectrum keys for resumable downloads."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    keys: set[tuple[str, str, str, float]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                keys.add((str(row["event_id"]), str(row["station_id"]), str(row.get("channel", "")), float(row["frequency_hz"])))
            except Exception:
                pass
    return keys


def _append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _station_candidates(client: Client, event: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Choose nearby stations before requesting waveform payloads."""
    start = UTCDateTime(event["_time"] - timedelta(seconds=args.pre_seconds))
    end = UTCDateTime(event["_time"] + timedelta(seconds=args.duration_seconds))
    inventory = client.get_stations(
        network=args.network,
        station="*",
        location="*",
        channel=args.channel,
        starttime=start,
        endtime=end,
        latitude=float(event["_lat"]),
        longitude=float(event["_lon"]),
        maxradius=args.radius_km / 111.19,
        level="channel",
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for net in inventory:
        for station in net:
            for channel in station:
                key = (net.code, station.code, channel.location_code or "", channel.code)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    dist_m, _az, _baz = gps2dist_azimuth(float(event["_lat"]), float(event["_lon"]), station.latitude, station.longitude)
                except Exception:
                    dist_m = 0.0
                candidates.append(
                    {
                        "network": net.code,
                        "station": station.code,
                        "location": channel.location_code or "",
                        "channel": channel.code,
                        "lat": float(station.latitude),
                        "lon": float(station.longitude),
                        "distance_km": dist_m / 1000.0,
                    }
                )
    candidates.sort(key=lambda row: float(row["distance_km"]))
    return candidates[: args.max_stations_per_event]


def _prepare_trace(trace: Any) -> tuple[np.ndarray, float]:
    tr = trace.copy()
    tr.detrend("demean")
    tr.detrend("linear")
    try:
        tr.taper(max_percentage=0.05, type="hann")
    except Exception:
        pass
    data = np.asarray(tr.data, dtype=np.float64)
    if data.size < 16:
        raise ValueError("trace too short")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, float(tr.stats.sampling_rate)


def _spectrum_rows(data: np.ndarray, sampling_rate: float, event: dict[str, Any], station: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Summarize a waveform window without discarding Fourier phase."""
    if data.size < 16:
        return []
    window = np.hanning(data.size)
    fft = np.fft.rfft(data * window)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sampling_rate)
    phase_unwrapped = np.unwrap(np.angle(fft))
    rows: list[dict[str, Any]] = []
    for freq in FREQUENCIES_HZ:
        if freq <= freqs[0] or freq >= freqs[-1]:
            continue
        idx = int(np.argmin(np.abs(freqs - freq)))
        amp = float(np.abs(fft[idx]))
        phase = float(np.angle(fft[idx]))
        if 0 < idx < len(freqs) - 1:
            dphi = phase_unwrapped[idx + 1] - phase_unwrapped[idx - 1]
            df = freqs[idx + 1] - freqs[idx - 1]
            group_delay = float(-dphi / (2.0 * math.pi * max(df, 1e-12)))
        else:
            group_delay = 0.0
        rows.append(
            {
                "event_id": event["event_id"],
                "station_id": f"{station['network']}.{station['station']}.{station['location']}.{station['channel']}",
                "network": station["network"],
                "station": station["station"],
                "location": station["location"],
                "channel": station["channel"],
                "time_utc": event["time_utc"],
                "lat": station["lat"],
                "lon": station["lon"],
                "frequency_hz": freq,
                "amplitude": max(amp, 1e-30),
                "phase_rad": phase,
                "group_delay_s": group_delay,
                "p_residual_s": 0.0,
                "s_residual_s": 0.0,
                "source": source,
            }
        )
    return rows


def _feature_row(data: np.ndarray, sampling_rate: float, event: dict[str, Any], station: dict[str, Any], source: str) -> dict[str, Any]:
    dt = 1.0 / sampling_rate
    velocity_proxy = np.cumsum(data) * dt
    pga = float(np.max(np.abs(data)))
    pgv = float(np.max(np.abs(velocity_proxy)))
    amp_log = math.log(max(pga, 1e-30)) - float(event.get("_mag", 0.0))
    return {
        "event_id": event["event_id"],
        "station_id": f"{station['network']}.{station['station']}.{station['location']}",
        "channel": station["channel"],
        "pga": pga,
        "pgv": pgv,
        "psa_0p3": pga,
        "psa_1p0": pga,
        "psa_3p0": pga,
        "p_residual_s": 0.0,
        "s_residual_s": 0.0,
        "amp_residual_log": amp_log,
        "source": source,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Download event windows and append spectra/features in small batches."""
    config_path = Path(args.config)
    event_csv = Path(args.event_csv) if args.event_csv else _event_csv_from_config(config_path)
    events = _read_events(event_csv, args.min_magnitude, args.max_events)
    spectra_output = Path(args.output)
    feature_output = Path(args.feature_output)
    raw_dir = Path(args.raw_dir)
    existing = _existing_keys(spectra_output)
    clients = [Client(name.strip()) for name in args.clients.split(",") if name.strip()]
    totals = {
        "events_considered": len(events),
        "events_with_waveforms": 0,
        "station_attempts": 0,
        "trace_count": 0,
        "spectra_rows": 0,
        "feature_rows": 0,
        "failures": [],
    }
    for event_index, event in enumerate(events, start=1):
        event_written = 0
        for client in clients:
            try:
                stations = _station_candidates(client, event, args)
            except Exception as exc:
                totals["failures"].append({"event_id": event.get("event_id"), "client": str(client), "stage": "stations", "error": f"{type(exc).__name__}: {exc}"})
                continue
            for station in stations:
                totals["station_attempts"] += 1
                station_id = f"{station['network']}.{station['station']}.{station['location']}.{station['channel']}"
                if all((str(event["event_id"]), station_id, str(station["channel"]), freq) in existing for freq in FREQUENCIES_HZ):
                    continue
                start = UTCDateTime(event["_time"] - timedelta(seconds=args.pre_seconds))
                end = UTCDateTime(event["_time"] + timedelta(seconds=args.duration_seconds))
                try:
                    stream = client.get_waveforms(
                        station["network"], station["station"], station["location"] or "*", station["channel"], start, end
                    )
                    stream.merge(method=1, fill_value="interpolate")
                    if args.bandpass:
                        stream.filter("bandpass", freqmin=args.freqmin, freqmax=args.freqmax, corners=2, zerophase=True)
                except Exception as exc:
                    if len(totals["failures"]) < args.max_failures_recorded:
                        totals["failures"].append({"event_id": event.get("event_id"), "station_id": station_id, "stage": "waveform", "error": f"{type(exc).__name__}: {exc}"})
                    continue
                event_dir = raw_dir / str(event["_time"].year) / str(event["event_id"])
                event_dir.mkdir(parents=True, exist_ok=True)
                try:
                    stream.write(str(event_dir / f"{station_id}.mseed"), format="MSEED")
                except Exception:
                    pass
                for trace in stream:
                    try:
                        data, sampling_rate = _prepare_trace(trace)
                    except Exception:
                        continue
                    station_for_trace = {**station, "channel": str(trace.stats.channel), "location": str(trace.stats.location or station["location"])}
                    source = f"FDSN {client.base_url}; raw_mseed={event_dir / f'{station_id}.mseed'}"
                    spectra_rows = _spectrum_rows(data, sampling_rate, event, station_for_trace, source)
                    spectra_rows = [
                        row for row in spectra_rows
                        if (str(row["event_id"]), str(row["station_id"]), str(row["channel"]), float(row["frequency_hz"])) not in existing
                    ]
                    if spectra_rows:
                        _append_csv(spectra_output, spectra_rows, SPECTRA_FIELDS)
                        for row in spectra_rows:
                            existing.add((str(row["event_id"]), str(row["station_id"]), str(row["channel"]), float(row["frequency_hz"])))
                    feature = _feature_row(data, sampling_rate, event, station_for_trace, source)
                    _append_csv(feature_output, [feature], FEATURE_FIELDS)
                    totals["trace_count"] += 1
                    totals["spectra_rows"] += len(spectra_rows)
                    totals["feature_rows"] += 1
                    event_written += len(spectra_rows)
                    if totals["trace_count"] >= args.max_traces:
                        break
                if totals["trace_count"] >= args.max_traces:
                    break
                if event_written >= args.max_spectra_rows_per_event:
                    break
                time.sleep(args.sleep_s)
            if event_written:
                break
        if event_written:
            totals["events_with_waveforms"] += 1
        print(f"event {event_index}/{len(events)} id={event.get('event_id')} spectra_rows_added={event_written} trace_count={totals['trace_count']}", flush=True)
        if totals["trace_count"] >= args.max_traces:
            break
    meta = {
        **totals,
        "config": str(config_path),
        "event_csv": str(event_csv),
        "output": str(spectra_output),
        "feature_output": str(feature_output),
        "raw_dir": str(raw_dir),
        "clients": args.clients,
        "network": args.network,
        "channel": args.channel,
        "min_magnitude": args.min_magnitude,
        "max_events": args.max_events,
        "max_traces": args.max_traces,
        "representation": "complex spectra retaining phase_rad and group_delay_s plus raw MiniSEED windows",
        "not_prediction": True,
    }
    spectra_output.with_suffix(spectra_output.suffix + ".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    feature_output.with_suffix(feature_output.suffix + ".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-csv", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-output", required=True)
    parser.add_argument("--raw-dir", default="data/raw/waveforms/fdsn")
    parser.add_argument("--clients", default="IRIS")
    parser.add_argument("--network", default="*")
    parser.add_argument("--channel", default="BH?,HH?,HN?")
    parser.add_argument("--min-magnitude", type=float, default=6.0)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-stations-per-event", type=int, default=25)
    parser.add_argument("--max-traces", type=int, default=1000)
    parser.add_argument("--max-spectra-rows-per-event", type=int, default=500)
    parser.add_argument("--radius-km", type=float, default=1200.0)
    parser.add_argument("--pre-seconds", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=float, default=900.0)
    parser.add_argument("--bandpass", action="store_true")
    parser.add_argument("--freqmin", type=float, default=0.05)
    parser.add_argument("--freqmax", type=float, default=20.0)
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--max-failures-recorded", type=int, default=200)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
