#!/usr/bin/env python3
"""Collect authenticated NIED Hi-net waveform windows.

Credentials are read from an external env file that must stay outside Git.  The
output schema matches the public FDSN collector so both sources can be merged
without changing downstream transfer-function or array-projection code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from obspy import read  # type: ignore

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


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _credentials(env_file: Path | None) -> tuple[str, str]:
    """Load Hi-net credentials without ever printing secret values."""
    if env_file:
        _load_env_file(env_file)
    user = os.environ.get("HINET_USER") or os.environ.get("HINET_USERNAME")
    password = os.environ.get("HINET_PASSWORD") or os.environ.get("HINET_PASS")
    if not user or not password:
        raise RuntimeError(
            "Hi-net credentials are not configured. Set HINET_USER and HINET_PASSWORD, "
            "or create /workspace/equake/secrets/hinet.env outside Git."
        )
    return user, password


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def _read_events(path: Path, min_mag: float, max_events: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    out = []
    for row in rows:
        try:
            row["_mag"] = float(row.get("magnitude", 0.0) or 0.0)
            row["_time"] = _parse_time(str(row["time_utc"]))
            row["_lat"] = float(row["lat"])
            row["_lon"] = float(row["lon"])
        except Exception:
            continue
        if row["_mag"] >= min_mag:
            out.append(row)
    out.sort(key=lambda row: (float(row["_mag"]), row["_time"]), reverse=True)
    return out[:max_events] if max_events > 0 else out


def _event_csv_from_config(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value = config.get("data_sources", {}).get("event_csv")
    if not value:
        raise ValueError("data_sources.event_csv must be configured")
    path = Path(str(value))
    if not path.is_absolute():
        path = config_path.resolve().parents[1] / path
    return path


def _append(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _existing_keys(path: Path) -> set[tuple[str, str, str, float]]:
    """Return existing spectrum keys so repeated authenticated runs are resumable."""
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


def _parse_station_groups(value: str) -> list[list[str]]:
    """Parse semicolon-separated station groups, each with comma/colon stations."""
    groups: list[list[str]] = []
    for raw_group in value.split(";"):
        names = [item.strip() for item in raw_group.replace(":", ",").split(",") if item.strip()]
        if names:
            groups.append(names)
    return groups


def _group_label(stations: list[str] | None) -> str:
    if not stations:
        return "all"
    cleaned = [name.replace(".", "_").replace("/", "_") for name in stations]
    return "stations_" + "_".join(cleaned[:6])


def _prepend_tool_path(path_value: str) -> None:
    """Use workspace-local Hi-net conversion tools without touching system PATH."""
    if not path_value:
        return
    path = Path(path_value).expanduser()
    if path.exists():
        os.environ["PATH"] = str(path.resolve()) + os.pathsep + os.environ.get("PATH", "")


def _select_stations(client: Any, code: str, stations: list[str] | None) -> None:
    """Select a small station subset; all stations are used when stations is None."""
    payload = {
        "net": code,
        "stcds": ":".join(stations) if stations else None,
        "mode": "1",
    }
    client.session.post(client._CONT_SELECT, data=payload, timeout=client.timeout)


def _download_hinet_window(
    client: Any,
    event: dict[str, Any],
    outdir: Path,
    code: str,
    minutes: int,
    pre_seconds: int,
    time_offset_hours: float,
    threads: int,
) -> list[Path]:
    """Download one event-centered continuous window from Hi-net."""
    outdir.mkdir(parents=True, exist_ok=True)
    # Catalogs are stored as UTC. Hi-net continuous waveform requests use local
    # Japan time in the web UI, so the default offset is +9 hours.
    start = event["_time"] + timedelta(hours=time_offset_hours) - timedelta(seconds=pre_seconds)
    start_text = start.strftime("%Y%m%d%H%M")
    before = set(outdir.glob("**/*"))
    result = client.get_continuous_waveform(code, start_text, minutes, outdir=str(outdir), threads=threads)
    after = [path for path in outdir.glob("**/*") if path.is_file() and path not in before]
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, str | Path):
                path = Path(item)
                if path.exists() and path.is_file():
                    after.append(path)
    elif isinstance(result, str | Path):
        path = Path(result)
        if path.exists() and path.is_file():
            after.append(path)
    return sorted(set(after))


def _extract_sac_with_hinetpy(files: list[Path], outdir: Path) -> list[Path]:
    try:
        from HinetPy import win32  # type: ignore
    except Exception:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    sac_files: list[Path] = []
    existing_sac = [path for path in files if path.suffix.lower() == ".sac"]
    sac_files.extend(existing_sac)
    ctable_candidates = [
        path for path in files
        if path.suffix.lower() in {".ch", ".ctable", ".txt"}
        or "ch" in path.name.lower()
        or "table" in path.name.lower()
    ]
    data_candidates = [
        path for path in files
        if path.is_file()
        and path not in ctable_candidates
        and path.suffix.lower() not in {".sac", ".pz"}
    ]
    if not ctable_candidates:
        ctable_candidates = [
            path for parent in {p.parent for p in files} for path in parent.glob("*")
            if path.is_file() and ("ch" in path.name.lower() or "table" in path.name.lower())
        ]
    ctable = ctable_candidates[0] if ctable_candidates else None
    if ctable is None:
        return sorted(set(sac_files))
    for data_file in data_candidates:
        try:
            win32.extract_sac(str(data_file), str(ctable), outdir=str(outdir))
        except Exception:
            continue
    sac_files.extend(outdir.glob("*.SAC"))
    sac_files.extend(outdir.glob("*.sac"))
    return sorted(set(sac_files))


def _maybe_convert_with_win2sac(raw_dir: Path, sac_dir: Path, command: str) -> list[Path]:
    if not command:
        return []
    sac_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([command, str(raw_dir), str(sac_dir)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    files = list(sac_dir.glob("*.SAC")) + list(sac_dir.glob("*.sac"))
    return sorted(set(files))


def _prepare_trace(path: Path) -> tuple[Any, np.ndarray, float]:
    stream = read(str(path))
    trace = stream[0]
    trace.detrend("demean")
    trace.detrend("linear")
    try:
        trace.taper(max_percentage=0.05, type="hann")
    except Exception:
        pass
    data = np.asarray(trace.data, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.size < 16:
        raise ValueError("trace too short")
    return trace, data, float(trace.stats.sampling_rate)


def _station_id(trace: Any, path: Path) -> tuple[str, str, str, str, str]:
    network = str(getattr(trace.stats, "network", "NIED") or "NIED")
    station = str(getattr(trace.stats, "station", path.stem) or path.stem)
    location = str(getattr(trace.stats, "location", "") or "")
    channel = str(getattr(trace.stats, "channel", "") or "")
    station_id = f"{network}.{station}.{location}.{channel}"
    return station_id, network, station, location, channel


def _spectra(data: np.ndarray, sampling_rate: float, trace: Any, path: Path, event: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Convert SAC traces into the shared phase-aware spectrum schema."""
    station_id, network, station, location, channel = _station_id(trace, path)
    window = np.hanning(data.size)
    fft = np.fft.rfft(data * window)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sampling_rate)
    phase_unwrapped = np.unwrap(np.angle(fft))
    lat = float(getattr(trace.stats, "sac", {}).get("stla", 0.0)) if hasattr(trace.stats, "sac") else 0.0
    lon = float(getattr(trace.stats, "sac", {}).get("stlo", 0.0)) if hasattr(trace.stats, "sac") else 0.0
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
                "station_id": station_id,
                "network": network,
                "station": station,
                "location": location,
                "channel": channel,
                "time_utc": event["time_utc"],
                "lat": lat,
                "lon": lon,
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


def _feature(data: np.ndarray, sampling_rate: float, trace: Any, path: Path, event: dict[str, Any], source: str) -> dict[str, Any]:
    station_id, _network, _station, _location, channel = _station_id(trace, path)
    dt = 1.0 / sampling_rate
    vel = np.cumsum(data) * dt
    pga = float(np.max(np.abs(data)))
    pgv = float(np.max(np.abs(vel)))
    return {
        "event_id": event["event_id"],
        "station_id": station_id,
        "channel": channel,
        "pga": pga,
        "pgv": pgv,
        "psa_0p3": pga,
        "psa_1p0": pga,
        "psa_3p0": pga,
        "p_residual_s": 0.0,
        "s_residual_s": 0.0,
        "amp_residual_log": math.log(max(pga, 1e-30)) - float(event.get("_mag", 0.0)),
        "source": source,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run authenticated collection with resumable append-only outputs."""
    _prepend_tool_path(args.tool_path)
    user, password = _credentials(Path(args.env_file) if args.env_file else None)
    try:
        from HinetPy import Client  # type: ignore
    except Exception as exc:
        raise RuntimeError("HinetPy is required. Install it in the project .venv with: python -m pip install HinetPy") from exc
    client = Client(
        user,
        password,
        timeout=args.timeout_s,
        retries=args.retries,
        sleep_time_in_seconds=args.sleep_time_s,
        max_sleep_count=args.max_sleep_count,
    )
    config_path = Path(args.config)
    event_csv = Path(args.event_csv) if args.event_csv else _event_csv_from_config(config_path)
    events = _read_events(event_csv, args.min_magnitude, args.max_events)
    spectra_output = Path(args.output)
    feature_output = Path(args.feature_output)
    raw_root = Path(args.raw_dir)
    existing = _existing_keys(spectra_output)
    station_groups = _parse_station_groups(args.stations)
    if not station_groups:
        station_groups = [None]  # type: ignore[list-item]
    totals: dict[str, Any] = {
        "events_considered": len(events),
        "events_with_raw_download": 0,
        "station_group_requests": 0,
        "sac_trace_count": 0,
        "spectra_rows": 0,
        "feature_rows": 0,
        "failures": [],
    }
    for index, event in enumerate(events, start=1):
        event_dir = raw_root / str(event["_time"].year) / str(event["event_id"])
        event_had_files = False
        for stations in station_groups:
            label = _group_label(stations)
            raw_dir = event_dir / label / "raw"
            sac_dir = event_dir / label / "sac"
            try:
                _select_stations(client, args.network_code, stations)
                totals["station_group_requests"] += 1
                files = _download_hinet_window(
                    client,
                    event,
                    raw_dir,
                    args.network_code,
                    args.minutes,
                    args.pre_seconds,
                    args.time_offset_hours,
                    args.threads,
                )
                if files:
                    event_had_files = True
                sac_files = _extract_sac_with_hinetpy(files, sac_dir)
                if not sac_files:
                    sac_files = _maybe_convert_with_win2sac(raw_dir, sac_dir, args.win2sac_command)
            except Exception as exc:
                if len(totals["failures"]) < args.max_failures_recorded:
                    totals["failures"].append(
                        {
                            "event_id": event.get("event_id"),
                            "stage": "download_or_extract",
                            "station_group": stations,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            if args.max_traces_per_event > 0:
                sac_files = sac_files[: args.max_traces_per_event]
            for sac in sac_files:
                try:
                    trace, data, sampling_rate = _prepare_trace(sac)
                    source = f"NIED authenticated archive; network={args.network_code}; station_group={label}; raw_dir={raw_dir}; sac={sac}"
                    spectra_rows = _spectra(data, sampling_rate, trace, sac, event, source)
                    spectra_rows = [
                        row for row in spectra_rows
                        if (str(row["event_id"]), str(row["station_id"]), str(row["channel"]), float(row["frequency_hz"])) not in existing
                    ]
                    if not spectra_rows:
                        continue
                    feature_row = _feature(data, sampling_rate, trace, sac, event, source)
                except Exception as exc:
                    if len(totals["failures"]) < args.max_failures_recorded:
                        totals["failures"].append({"event_id": event.get("event_id"), "stage": "spectra", "file": str(sac), "error": f"{type(exc).__name__}: {exc}"})
                    continue
                _append(spectra_output, spectra_rows, SPECTRA_FIELDS)
                _append(feature_output, [feature_row], FEATURE_FIELDS)
                for row in spectra_rows:
                    existing.add((str(row["event_id"]), str(row["station_id"]), str(row["channel"]), float(row["frequency_hz"])))
                totals["sac_trace_count"] += 1
                totals["spectra_rows"] += len(spectra_rows)
                totals["feature_rows"] += 1
                if args.max_total_traces > 0 and totals["sac_trace_count"] >= args.max_total_traces:
                    break
            if args.max_total_traces > 0 and totals["sac_trace_count"] >= args.max_total_traces:
                break
        if event_had_files:
            totals["events_with_raw_download"] += 1
        print(f"event {index}/{len(events)} id={event.get('event_id')} traces={totals['sac_trace_count']} spectra_rows={totals['spectra_rows']}", flush=True)
        if args.max_total_traces > 0 and totals["sac_trace_count"] >= args.max_total_traces:
            break
    meta = {
        **totals,
        "config": str(config_path),
        "event_csv": str(event_csv),
        "output": str(spectra_output),
        "feature_output": str(feature_output),
        "raw_dir": str(raw_root),
        "network_code": args.network_code,
        "stations": args.stations,
        "minutes": args.minutes,
        "pre_seconds": args.pre_seconds,
        "time_offset_hours": args.time_offset_hours,
        "tool_path": args.tool_path,
        "min_magnitude": args.min_magnitude,
        "max_events": args.max_events,
        "representation": "Hi-net waveform windows converted to phase-preserving complex spectra",
        "credential_source": "environment_or_env_file_without_logging_values",
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
    parser.add_argument("--raw-dir", default="data/raw/waveforms/hinet")
    parser.add_argument("--env-file", default="/workspace/equake/secrets/hinet.env")
    parser.add_argument("--min-magnitude", type=float, default=5.5)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--network-code", default="0101")
    parser.add_argument("--stations", default="", help="Semicolon-separated station groups; each group uses comma or colon-separated station names.")
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--pre-seconds", type=int, default=60)
    parser.add_argument("--time-offset-hours", type=float, default=9.0, help="Offset from UTC catalog time to Hi-net request time; Japan local time is +9.")
    parser.add_argument("--max-traces-per-event", type=int, default=200)
    parser.add_argument("--max-total-traces", type=int, default=5000)
    parser.add_argument("--tool-path", default=".deps/hinet-win32tools/bin")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-time-s", type=float, default=5.0)
    parser.add_argument("--max-sleep-count", type=int, default=30)
    parser.add_argument("--win2sac-command", default="")
    parser.add_argument("--max-failures-recorded", type=int, default=200)
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
