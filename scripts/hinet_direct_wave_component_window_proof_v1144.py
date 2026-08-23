#!/usr/bin/env python3
"""Per-component-window proof candidate for physical direct-wave removal.

This is intentionally isolated from the active pipeline.  It fixes the v1129
proof defects without activating a kernel or a pointer: support is centred on
the refined pick and clipped to the declared formal phase window; removal rays
are estimated leave-station-group-out from a low frequency band; validation
uses a disjoint high frequency band and an independently estimated ray; and
the canonical station response is rebuilt from cleaned complex spectra.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
from obspy import UTCDateTime, read
from scipy.signal import butter, sosfiltfilt

FREQUENCIES = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
SCHEMA = "formal-hinet-component-window-direct-wave-proof-package-v1144"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def orientation_vector(azimuth_deg: float, incidence_deg: float) -> np.ndarray:
    azimuth = np.radians(azimuth_deg)
    incidence = np.radians(incidence_deg)
    return np.asarray([
        np.sin(incidence) * np.sin(azimuth),
        np.sin(incidence) * np.cos(azimuth),
        np.cos(incidence),
    ], dtype=np.float64)


def vector_angles(vector: np.ndarray) -> tuple[float, float]:
    value = np.asarray(vector, dtype=np.float64)
    value /= max(float(np.linalg.norm(value)), 1.0e-15)
    if value[2] < 0:
        value = -value
    azimuth = math.degrees(math.atan2(value[0], value[1])) % 360.0
    incidence = math.degrees(math.acos(float(np.clip(value[2], -1.0, 1.0))))
    return azimuth, incidence


def wrapped_delta(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def filtered(values: np.ndarray, sample_rate: float, band: tuple[float, float] = (0.35, 10.0)) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    value = np.nan_to_num(value - np.nanmedian(value))
    if value.size > 3:
        axis = np.linspace(-1.0, 1.0, value.size)
        value = value - np.polyval(np.polyfit(axis, value, 1), axis)
    nyquist = 0.5 * sample_rate
    low = band[0] / nyquist
    high = min(0.98, band[1] / nyquist)
    if value.size > 64 and 0 < low < high < 1:
        value = sosfiltfilt(butter(4, [low, high], btype="bandpass", output="sos"), value)
    return value


def compact_pick_taper(times: np.ndarray, pick: float, start: float, end: float, sigma: float) -> tuple[np.ndarray, float, float]:
    if not (start <= pick <= end):
        raise RuntimeError("refined pick is outside the formal phase window")
    dt = float(np.median(np.diff(times)))
    margin = min(pick - start, end - pick) - dt
    support = min(2.0 * sigma, margin)
    if not np.isfinite(support) or support < max(0.25, 0.5 * sigma):
        raise RuntimeError("formal window cannot contain the pick-centred support")
    core = 0.5 * support
    distance = np.abs(times - pick)
    taper = np.zeros_like(times)
    taper[distance <= core] = 1.0
    edge = (distance > core) & (distance < support)
    taper[edge] = 0.5 * (1.0 + np.cos(np.pi * (distance[edge] - core) / (support - core)))
    if np.any(taper[(times < start) | (times > end)] != 0):
        raise RuntimeError("pick-centred support escaped the formal phase window")
    return taper, core, support


def window_spectrum(times: np.ndarray, values: np.ndarray, start: float, end: float) -> np.ndarray:
    mask = (times >= start) & (times <= end)
    selected_times = times[mask]
    selected = values[:, mask]
    if selected.shape[1] < 16:
        raise RuntimeError("formal phase window has too few samples")
    selected = selected - selected.mean(axis=1, keepdims=True)
    taper = np.hanning(selected.shape[1])
    normalizer = max(float(taper.sum()), 1.0e-15)
    return np.stack([
        np.sum(selected * taper[None, :] * np.exp(-2j * np.pi * frequency * selected_times)[None, :], axis=1) / normalizer
        for frequency in FREQUENCIES
    ], axis=1)


def component_window_spectrum(times: np.ndarray, values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Compute every sensor channel spectrum in its own formal window."""
    if values.shape[0] != len(starts) or len(starts) != len(ends):
        raise RuntimeError("component spectrum/window shape mismatch")
    rows = []
    for channel_index in range(values.shape[0]):
        rows.append(window_spectrum(times, values[channel_index:channel_index + 1], float(starts[channel_index]), float(ends[channel_index]))[0])
    return np.asarray(rows, dtype=np.complex128)


def align_components(values: np.ndarray, times: np.ndarray, picks: np.ndarray, relative_times: np.ndarray) -> np.ndarray:
    """Sample each sensor component at its own formal-pick-relative times."""
    if values.shape[0] != len(picks):
        raise RuntimeError("component pick/value shape mismatch")
    return np.asarray([
        np.interp(relative_times + float(picks[index]), times, values[index])
        for index in range(values.shape[0])
    ], dtype=np.float64)


def observed_ray(enu: np.ndarray, times: np.ndarray, pick: float, radius: float, phase: str, sample_rate: float, band: tuple[float, float]) -> np.ndarray:
    selected = np.asarray([filtered(component, sample_rate, band) for component in enu])
    mask = np.abs(times - pick) <= radius
    if int(mask.sum()) < 16:
        raise RuntimeError("independent ray window too short")
    covariance = selected[:, mask] @ selected[:, mask].T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ray = eigenvectors[:, -1] if phase == "P" else eigenvectors[:, 0]
    return ray / max(float(np.linalg.norm(ray)), 1.0e-15)


def phase_project(values: np.ndarray, ray: np.ndarray, phase: str) -> np.ndarray:
    ray = np.asarray(ray, dtype=np.float64)
    ray /= max(float(np.linalg.norm(ray)), 1.0e-15)
    longitudinal = ray[:, None] * (ray @ values)[None, :]
    return longitudinal if phase == "P" else values - longitudinal


def component_physical_remove(sensor_values: np.ndarray, orientation: np.ndarray, ray: np.ndarray, component_tapers: np.ndarray, phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project in physical ENU, then apply each sensor channel formal support."""
    if sensor_values.shape != component_tapers.shape or orientation.shape != (3, 3):
        raise RuntimeError("component removal shape mismatch")
    inverse = np.linalg.pinv(orientation, rcond=1.0e-7)
    enu = inverse @ sensor_values
    projected_enu = phase_project(enu, ray, phase)
    projected_sensor = orientation @ projected_enu
    cleaned_sensor = sensor_values - component_tapers * projected_sensor
    cleaned_enu = inverse @ cleaned_sensor
    return cleaned_sensor, cleaned_enu, projected_sensor


def canonical_response(
    cleaned_spectra: np.ndarray,
    channel_valid: np.ndarray,
    station_groups: list,
    xyz: np.ndarray,
    source_xyz: np.ndarray,
    prior: np.ndarray,
    phase: str,
    frequency_index: int,
) -> np.ndarray:
    velocity = 6000.0 if phase == "P" else 3500.0
    rows = []
    frequency = float(FREQUENCIES[frequency_index])
    for index, group in enumerate(station_groups):
        orientations = np.asarray([
            orientation_vector(float(row.cmpaz_deg), float(row.cmpinc_deg))
            for row in group.itertuples(index=False)
        ])
        values = cleaned_spectra[index, channel_valid[index], frequency_index]
        displacement = np.linalg.pinv(orientations, rcond=1.0e-7) @ values
        station_xyz = np.asarray([float(group.station_x_m.iloc[0]), float(group.station_y_m.iloc[0]), 0.0])
        rays = xyz - station_xyz
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1.0e-9)
        if phase == "P":
            projected = np.abs(rays @ displacement)
        else:
            projected = np.linalg.norm(displacement[None, :] - rays * (rays @ displacement)[:, None], axis=1)
        direct = np.linalg.norm(source_xyz - station_xyz)
        two_leg = np.linalg.norm(xyz - source_xyz, axis=1) + np.linalg.norm(xyz - station_xyz, axis=1)
        delay = (two_leg - direct) / velocity
        steer = 0.25 + 0.75 * np.cos(2 * np.pi * frequency * delay) ** 2
        rows.append(projected * steer * (0.35 + 0.65 * prior))
    response = np.asarray(rows, dtype=np.float64)
    scale = np.quantile(response[response > 0], 0.95) if np.any(response > 0) else 1.0
    return np.clip(response / max(float(scale), 1.0e-12), 0.0, 8.0)


def run_parity(helper: Path, fixture: Path) -> dict:
    completed = subprocess.run(["/usr/bin/python3", str(helper), "--fixture", str(fixture)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"independent CPU/GPU parity failed: {completed.stdout}")
    return json.loads(completed.stdout)


def validate_top_config(config: dict, root: Path) -> dict[str, bool]:
    expected = {
        "schema": "hinet-direct-wave-component-window-proof-config-v1144",
        "status": "INACTIVE_ISOLATED_PROOF_CANDIDATE",
        "source_policy": "NIED_HINET_ONLY",
        "known_structure_input_allowed": False,
        "machine_learning_allowed": False,
        "additional_time_correction_s": 0,
        "kernel_or_GS_run_allowed": False,
        "pointer_activation_allowed": False,
        "publication_allowed": False,
    }
    checks = {name: config.get(name) == value for name, value in expected.items()}
    for name, item in config["inputs"].items():
        path = root / item["path"]
        checks[f"{name}_exists"] = path.is_file()
        checks[f"{name}_hash"] = path.is_file() and sha256(path) == item["sha256"]
    if not all(checks.values()):
        raise RuntimeError(f"proof config failed closed: {[name for name, passed in checks.items() if not passed]}")
    return checks


def run(root: Path, config_path: Path, output_dir: Path) -> tuple[dict, Path]:
    root = root.resolve()
    config = json.loads(config_path.read_text())
    checks = validate_top_config(config, root)
    base_config_path = root / config["inputs"]["v1129_config"]["path"]
    base_config = json.loads(base_config_path.read_text())
    event_id = config["event_id"]
    if event_id != base_config["event_id"]:
        raise RuntimeError("event binding differs from v1129")
    inputs = {name: root / item["path"] for name, item in base_config["inputs"].items()}
    for name, item in base_config["inputs"].items():
        if not inputs[name].is_file() or sha256(inputs[name]) != item["sha256"]:
            raise RuntimeError(f"nested v1129 input hash failed: {name}")
    candidate = json.loads(inputs["raw_candidate_manifest"].read_text())
    if candidate["source_policy"] != "NIED_HINET_ONLY" or candidate["additional_time_correction_s"] != 0:
        raise RuntimeError("raw source/UTC contract failed")
    records = [record for record in candidate["records"] if record["event_id"] == event_id]
    if len(records) != 140:
        raise RuntimeError("representative event does not have exact 140 eligible records")
    raw_by_path = {}
    for record in records:
        path = root / record["path"]
        if record["quarantined"] or not record["eligible_for_downstream"] or record["physical_unit"] != "nm/s" or record["additional_time_correction_s"] != 0:
            raise RuntimeError("ineligible raw record entered proof")
        if sha256(path) != record["sha256"]:
            raise RuntimeError("raw hash mismatch")
        raw_by_path[record["path"]] = record
    excluded = candidate["excluded"]
    if len(excluded) != 52 or {(x["station"], x["component"]) for x in excluded} != {
        ("N.SSGH", "N"), ("N.SSGH", "E"), ("N.SSWH", "N"), ("N.SSWH", "E")
    }:
        raise RuntimeError("52-horizontal quarantine contract failed")
    connection = duckdb.connect()
    features = connection.execute("select * from read_parquet(?) where event_id=?", [str(inputs["phase_features"]), event_id]).fetchdf()
    events = connection.execute("select * from read_parquet(?) where event_id=?", [str(inputs["formal_events"]), event_id]).fetchdf()
    connection.close()
    if len(events) != 1 or len(features) != 1400 or int((features.trace_time_correction_s != 0).sum()) != 0:
        raise RuntimeError("formal feature/event query failed")
    event = events.iloc[0]
    origin = UTCDateTime(str(event.time_utc))
    thresholds = config["quality"]
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_results = {}
    parity_values, parity_orientations, parity_rays, parity_tapers, parity_phases, parity_cpu = [], [], [], [], [], []

    for phase in ("P", "S"):
        phase_rows = features[(features.phase_family == phase) & np.isclose(features.frequency_hz, 1.0)].copy()
        groups = [group.sort_values("station_id") for _, group in phase_rows.groupby("base_station_id", sort=True)]
        if len(groups) != 48 or sum(len(group) for group in groups) != 140:
            raise RuntimeError("full formal station/channel support failed")
        with np.load(inputs[f"adapter_{phase}_package"], allow_pickle=False) as pre:
            station_ids = [str(value) for value in pre["station_ids"].tolist()]
            pre_response = pre["station_response_pre_direct"].copy()
            illumination = pre["station_illumination"].copy()
            xyz = pre["xyz_m"].copy()
            source_xyz = pre["source_xyz_m"].copy()
            grid_shape = pre["grid_shape"].copy()
            prior = pre["projection_prior"].copy()
            station_azimuth = pre["station_azimuth_deg"].copy()
        group_map = {str(group.base_station_id.iloc[0]): group for group in groups}
        groups = [group_map[station] for station in station_ids]
        working = []
        for station_id, group in zip(station_ids, groups):
            signals, orientations, trace_starts, hashes = [], [], [], []
            sample_rate = None
            npts = None
            for row in group.itertuples(index=False):
                record = raw_by_path.get(str(row.raw_path))
                if record is None or record["sha256"] != sha256(root / str(row.raw_path)):
                    raise RuntimeError("feature/raw hash binding failed")
                trace = read(str(root / str(row.raw_path)))[0]
                rate = float(trace.stats.sampling_rate)
                count = int(trace.stats.npts)
                sample_rate = rate if sample_rate is None else sample_rate
                npts = count if npts is None else npts
                if abs(rate - sample_rate) > 1.0e-12 or count != npts:
                    raise RuntimeError("component alignment failed")
                trace_starts.append(float(trace.stats.starttime - origin))
                signals.append(filtered(np.asarray(trace.data), rate))
                orientations.append(orientation_vector(float(row.cmpaz_deg), float(row.cmpinc_deg)))
                hashes.append(record["sha256"])
            if max(trace_starts) - min(trace_starts) > 1.0e-6:
                raise RuntimeError("component UTC starts differ")
            observed = np.asarray(signals)
            orientation = np.asarray(orientations)
            times = trace_starts[0] + np.arange(npts, dtype=np.float64) / sample_rate
            component_picks = group.picked_arrival_s.to_numpy(dtype=np.float64)
            component_predicted = group.predicted_arrival_s.to_numpy(dtype=np.float64)
            component_starts = group.window_start_s.to_numpy(dtype=np.float64)
            component_ends = group.window_end_s.to_numpy(dtype=np.float64)
            component_sigmas = group.direct_gate_sigma_s.to_numpy(dtype=np.float64)
            component_tapers, component_cores, component_supports = [], [], []
            for channel_index in range(len(group)):
                taper, core, support = compact_pick_taper(
                    times,
                    float(component_picks[channel_index]),
                    float(component_starts[channel_index]),
                    float(component_ends[channel_index]),
                    float(component_sigmas[channel_index]),
                )
                component_tapers.append(taper)
                component_cores.append(core)
                component_supports.append(support)
            component_tapers = np.asarray(component_tapers)
            component_cores = np.asarray(component_cores, dtype=np.float64)
            component_supports = np.asarray(component_supports, dtype=np.float64)
            raw_spectrum = component_window_spectrum(times, observed, component_starts, component_ends)
            component_contract = []
            for channel_index, row in enumerate(group.itertuples(index=False)):
                component_contract.append({
                    "station_id": str(row.station_id),
                    "channel": str(row.channel),
                    "picked_arrival_s": float(component_picks[channel_index]),
                    "predicted_arrival_s": float(component_predicted[channel_index]),
                    "pick_minus_predicted_s": float(component_picks[channel_index] - component_predicted[channel_index]),
                    "formal_window_start_s": float(component_starts[channel_index]),
                    "formal_window_end_s": float(component_ends[channel_index]),
                    "direct_gate_sigma_s": float(component_sigmas[channel_index]),
                    "support_core_radius_s": float(component_cores[channel_index]),
                    "support_radius_s": float(component_supports[channel_index]),
                    "support_start_s": float(component_picks[channel_index] - component_supports[channel_index]),
                    "support_end_s": float(component_picks[channel_index] + component_supports[channel_index]),
                    "support_inside_own_formal_window": bool(
                        component_starts[channel_index] <= component_picks[channel_index] - component_supports[channel_index]
                        and component_picks[channel_index] + component_supports[channel_index] <= component_ends[channel_index]
                    ),
                })
            item = {
                "station_id": station_id,
                "group": group,
                "observed": observed,
                "orientation": orientation,
                "times": times,
                "sample_rate": sample_rate,
                "component_picks": component_picks,
                "component_predicted": component_predicted,
                "component_starts": component_starts,
                "component_ends": component_ends,
                "component_sigmas": component_sigmas,
                "component_tapers": component_tapers,
                "component_cores": component_cores,
                "component_supports": component_supports,
                "component_contract": component_contract,
                "raw_spectrum": raw_spectrum,
                "bundle_sha": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest(),
            }
            if len(group) == 1:
                if station_id not in {".N.SSGH.", ".N.SSWH."} or str(group.channel.iloc[0]) != "U":
                    raise RuntimeError("single-channel record is not vertical-only quarantine")
            else:
                if int(np.linalg.matrix_rank(orientation, tol=1.0e-7)) != 3:
                    raise RuntimeError("3C orientation rank failed")
                alignment_radius = float(np.min(component_cores))
                dt = 1.0 / float(sample_rate)
                relative_times = np.arange(-alignment_radius, alignment_radius + 0.5 * dt, dt, dtype=np.float64)
                aligned_sensor = align_components(observed, times, component_picks, relative_times)
                aligned_enu = np.linalg.pinv(orientation, rcond=1.0e-7) @ aligned_sensor
                backazimuths = group.station_event_backazimuth_deg.to_numpy(dtype=np.float64)
                incidences = group.incident_angle_deg.to_numpy(dtype=np.float64)
                if np.ptp(backazimuths) > 1.0e-8 or np.ptp(incidences) > 1.0e-8:
                    raise RuntimeError("component theory ray metadata differ")
                theory = orientation_vector(float(backazimuths[0]), float(incidences[0]))
                low_ray = observed_ray(aligned_enu, relative_times, 0.0, alignment_radius, phase, sample_rate, tuple(config["estimation_band_hz"]))
                high_ray = observed_ray(aligned_enu, relative_times, 0.0, alignment_radius, phase, sample_rate, tuple(config["validation_band_hz"]))
                if float(low_ray @ theory) < 0:
                    low_ray = -low_ray
                if float(high_ray @ theory) < 0:
                    high_ray = -high_ray
                theory_az, theory_inc = vector_angles(theory)
                low_az, low_inc = vector_angles(low_ray)
                item.update({
                    "aligned_sensor": aligned_sensor,
                    "aligned_enu": aligned_enu,
                    "relative_times": relative_times,
                    "alignment_radius": alignment_radius,
                    "theory_ray": theory,
                    "validation_ray": high_ray,
                    "azimuth_delta": wrapped_delta(low_az, theory_az),
                    "incidence_delta": low_inc - theory_inc,
                })
            working.append(item)

        full_indices = [index for index, item in enumerate(working) if "aligned_enu" in item]
        ordered = sorted(full_indices, key=lambda index: float(station_azimuth[index]))
        fold_by_index = {index: position % int(config["station_folds"]) for position, index in enumerate(ordered)}
        fold_models = []
        for fold in range(int(config["station_folds"])):
            training = [working[index] for index in full_indices if fold_by_index[index] != fold]
            az_delta = float(np.median([item["azimuth_delta"] for item in training]))
            inc_delta = float(np.median([item["incidence_delta"] for item in training]))
            fold_models.append({"fold": fold, "training_stations": len(training), "azimuth_delta_deg": az_delta, "incidence_delta_deg": inc_delta})

        cleaned_spectra = np.zeros((48, 3, len(FREQUENCIES)), dtype=np.complex128)
        raw_spectra = np.zeros_like(cleaned_spectra)
        channel_valid = np.zeros((48, 3), dtype=bool)
        component_station_ids = np.full((48, 3), "", dtype="<U32")
        component_channels = np.full((48, 3), "", dtype="<U4")
        component_picks = np.full((48, 3), np.nan, dtype=np.float64)
        component_predicted = np.full((48, 3), np.nan, dtype=np.float64)
        component_starts = np.full((48, 3), np.nan, dtype=np.float64)
        component_ends = np.full((48, 3), np.nan, dtype=np.float64)
        component_sigmas = np.full((48, 3), np.nan, dtype=np.float64)
        component_supports = np.full((48, 3), np.nan, dtype=np.float64)
        metrics = []
        scalar_ratios = np.ones(48, dtype=np.float64)
        for index, item in enumerate(working):
            count = len(item["group"])
            raw_spectra[index, :count] = item["raw_spectrum"]
            channel_valid[index, :count] = True
            component_station_ids[index, :count] = [row["station_id"] for row in item["component_contract"]]
            component_channels[index, :count] = [row["channel"] for row in item["component_contract"]]
            component_picks[index, :count] = item["component_picks"]
            component_predicted[index, :count] = item["component_predicted"]
            component_starts[index, :count] = item["component_starts"]
            component_ends[index, :count] = item["component_ends"]
            component_sigmas[index, :count] = item["component_sigmas"]
            component_supports[index, :count] = item["component_supports"]
            base_metric = {
                "station_id": item["station_id"],
                "channels": count,
                "component_formal_contract": item["component_contract"],
                "component_pick_spread_s": float(np.ptp(item["component_picks"])),
                "all_component_supports_inside_own_formal_windows": all(row["support_inside_own_formal_window"] for row in item["component_contract"]),
                "raw_bundle_sha256": item["bundle_sha"],
            }
            if count == 1:
                cleaned_sensor = item["observed"].copy()
                clean_spectrum = item["raw_spectrum"].copy()
                base_metric.update({
                    "status": "VERTICAL_ONLY_QUARANTINE_PRESERVED_NO_FAKE_3C",
                    "fold": None,
                    "validation_direct_energy_ratio": None,
                    "cross_component_retention": None,
                    "unused_window_max_abs_change": 0.0,
                    "injected_unused_signal_retention": 1.0,
                })
            else:
                fold = fold_by_index[index]
                model = fold_models[fold]
                theory_az, theory_inc = vector_angles(item["theory_ray"])
                removal_ray = orientation_vector(
                    theory_az + model["azimuth_delta_deg"],
                    float(np.clip(theory_inc + model["incidence_delta_deg"], 1.0, 89.0)),
                )
                removal_strength = float(config["removal_strength"][phase])
                effective_tapers = removal_strength * item["component_tapers"]
                cleaned_sensor, cleaned_enu, _ = component_physical_remove(
                    item["observed"], item["orientation"], removal_ray, effective_tapers, phase
                )
                clean_spectrum = component_window_spectrum(
                    item["times"], cleaned_sensor, item["component_starts"], item["component_ends"]
                )
                aligned_before_sensor = align_components(
                    item["observed"], item["times"], item["component_picks"], item["relative_times"]
                )
                aligned_after_sensor = align_components(
                    cleaned_sensor, item["times"], item["component_picks"], item["relative_times"]
                )
                inverse = np.linalg.pinv(item["orientation"], rcond=1.0e-7)
                aligned_before = inverse @ aligned_before_sensor
                aligned_after = inverse @ aligned_after_sensor
                validation_before = np.asarray([
                    filtered(component, item["sample_rate"], tuple(config["validation_band_hz"])) for component in aligned_before
                ])
                validation_after = np.asarray([
                    filtered(component, item["sample_rate"], tuple(config["validation_band_hz"])) for component in aligned_after
                ])
                direct_before = phase_project(validation_before, item["validation_ray"], phase)
                direct_after = phase_project(validation_after, item["validation_ray"], phase)
                complement_phase = "S" if phase == "P" else "P"
                cross_before = phase_project(validation_before, item["validation_ray"], complement_phase)
                cross_after = phase_project(validation_after, item["validation_ray"], complement_phase)
                direct_ratio = float(np.sum(direct_after ** 2) / max(float(np.sum(direct_before ** 2)), 1.0e-30))
                cross_retention = float(np.sum(cross_after ** 2) / max(float(np.sum(cross_before ** 2)), 1.0e-30))
                unused_changes = []
                injection = np.zeros_like(item["observed"])
                for channel_index in range(count):
                    formal = (item["times"] >= item["component_starts"][channel_index]) & (item["times"] <= item["component_ends"][channel_index])
                    unused = formal & (item["component_tapers"][channel_index] == 0)
                    unused_changes.append(float(np.max(np.abs(cleaned_sensor[channel_index, unused] - item["observed"][channel_index, unused]))) if np.any(unused) else float("inf"))
                    unused_indices = np.flatnonzero(unused)
                    if len(unused_indices) < 16:
                        raise RuntimeError("no independent component unused window remains")
                    span = min(31, len(unused_indices) // 2)
                    center = len(unused_indices) // 2
                    indices = unused_indices[max(0, center - span):min(len(unused_indices), center + span + 1)]
                    packet = np.hanning(len(indices)) * np.cos(2 * np.pi * 3.25 * (item["times"][indices] - item["times"][indices[0]]))
                    injection[channel_index, indices] = (0.71, -0.43, 0.56)[channel_index] * packet
                injected_clean, _, _ = component_physical_remove(
                    item["observed"] + injection, item["orientation"], removal_ray, effective_tapers, phase
                )
                retained = injected_clean - cleaned_sensor
                injection_retention = float(np.linalg.norm(retained[injection != 0]) / max(float(np.linalg.norm(injection[injection != 0])), 1.0e-30))
                before_formal = 0.0
                after_formal = 0.0
                for channel_index in range(count):
                    formal = (item["times"] >= item["component_starts"][channel_index]) & (item["times"] <= item["component_ends"][channel_index])
                    before_formal += float(np.sum(item["observed"][channel_index, formal] ** 2))
                    after_formal += float(np.sum(cleaned_sensor[channel_index, formal] ** 2))
                scalar_ratios[index] = math.sqrt(max(after_formal, 0.0) / max(before_formal, 1.0e-30))
                base_metric.update({
                    "status": f"{phase}_COMPONENT_WINDOW_LOSO_LOW_BAND_OPERATOR_HIGH_BAND_VALIDATION",
                    "removal_strength": removal_strength,
                    "fold": fold,
                    "fold_training_stations": model["training_stations"],
                    "removal_ray_azimuth_correction_deg": model["azimuth_delta_deg"],
                    "removal_ray_incidence_correction_deg": model["incidence_delta_deg"],
                    "validation_direct_energy_ratio": direct_ratio,
                    "cross_component_retention": cross_retention,
                    "unused_window_max_abs_change": float(max(unused_changes)),
                    "injected_unused_signal_retention": injection_retention,
                })
                parity_relative = np.arange(-2.0, 2.0 + 0.5 / item["sample_rate"], 1.0 / item["sample_rate"])
                parity_sensor = align_components(item["observed"], item["times"], item["component_picks"], parity_relative)
                parity_component_tapers = align_components(effective_tapers, item["times"], item["component_picks"], parity_relative)
                parity_clean_sensor, _, _ = component_physical_remove(
                    parity_sensor, item["orientation"], removal_ray, parity_component_tapers, phase
                )
                parity_values.append(parity_sensor)
                parity_orientations.append(item["orientation"])
                parity_rays.append(removal_ray)
                parity_tapers.append(parity_component_tapers)
                parity_phases.append(0 if phase == "P" else 1)
                parity_cpu.append(parity_clean_sensor)
            cleaned_spectra[index, :count] = clean_spectrum
            metrics.append(base_metric)

        frequency_index = int(np.flatnonzero(np.isclose(FREQUENCIES, 1.0))[0])
        response = canonical_response(cleaned_spectra, channel_valid, groups, xyz, source_xyz, prior, phase, frequency_index)
        scalar_response = pre_response * scalar_ratios[:, None]
        scalar_relative_difference = float(np.linalg.norm(response - scalar_response) / max(float(np.linalg.norm(response)), 1.0e-30))
        full_metrics = [item for item in metrics if item["channels"] == 3]
        direct_ratios = np.asarray([item["validation_direct_energy_ratio"] for item in full_metrics])
        cross_retentions = np.asarray([item["cross_component_retention"] for item in full_metrics])
        unused_changes = np.asarray([item["unused_window_max_abs_change"] for item in metrics])
        injection_retentions = np.asarray([item["injected_unused_signal_retention"] for item in metrics])
        component_rows = [row for item in metrics for row in item["component_formal_contract"]]
        arrival_contained = all(row["support_inside_own_formal_window"] for row in component_rows)
        component_deltas = np.asarray([row["pick_minus_predicted_s"] for row in component_rows], dtype=np.float64)
        source_text = Path(__file__).resolve().read_text(encoding="utf-8")
        forbidden_shortcuts = (
            "group.picked_arrival_s." + "median()",
            "group.predicted_arrival_s." + "median()",
            "group.window_start_s." + "min()",
            "group.window_end_s." + "max()",
        )
        no_union_median_shortcut = all(token not in source_text for token in forbidden_shortcuts)
        phase_gates = {
            "stations_48_channels_140": len(metrics) == 48 and len(component_rows) == 140,
            "full3C_46_vertical_only_2": len(full_metrics) == 46 and len(metrics) - len(full_metrics) == 2,
            "all_140_component_supports_inside_own_formal_windows": arrival_contained,
            "no_union_window_or_median_pick_shortcut": no_union_median_shortcut,
            "independent_high_band_direct_median": float(np.median(direct_ratios)) <= thresholds["validation_direct_ratio_median_max"],
            "independent_high_band_direct_p90": float(np.quantile(direct_ratios, 0.9)) <= thresholds["validation_direct_ratio_p90_max"],
            "cross_component_median_preserved": thresholds["cross_retention_median_min"] <= float(np.median(cross_retentions)) <= thresholds["cross_retention_median_max"],
            "cross_component_p10_preserved": float(np.quantile(cross_retentions, 0.1)) >= thresholds["cross_retention_p10_min"],
            "actual_component_unused_windows_unchanged": float(unused_changes.max()) <= thresholds["unused_window_max_abs_change"],
            "injected_component_unused_signal_preserved": float(np.max(np.abs(injection_retentions - 1.0))) <= thresholds["injected_retention_abs_error"],
            "spectral_response_not_scalar_approximation": scalar_relative_difference >= thresholds["scalar_response_relative_difference_min"],
            "spectral_response_finite_nonnegative": bool(np.isfinite(response).all() and np.all(response >= 0)),
        }
        package_path = output_dir / f"{event_id}_{phase}_component_window_proof_v1144.npz"
        temporary = package_path.with_name(package_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            schema=np.asarray(SCHEMA),
            requirement_id=np.asarray("NR-SCIENCE-DIRECT-WAVE-COMPONENT-WINDOW-049"),
            event_id=np.asarray(event_id),
            phase=np.asarray(phase),
            station_ids=np.asarray(station_ids),
            physical_unit=np.asarray("nm/s"),
            input_time_basis=np.asarray("UTC_ALREADY_CORRECTED"),
            additional_time_correction_s=np.zeros(48),
            channel_valid=channel_valid,
            component_station_ids=component_station_ids,
            component_channels=component_channels,
            component_picked_arrival_s=component_picks,
            component_predicted_arrival_s=component_predicted,
            component_formal_window_start_s=component_starts,
            component_formal_window_end_s=component_ends,
            component_direct_gate_sigma_s=component_sigmas,
            component_support_radius_s=component_supports,
            frequencies_hz=FREQUENCIES,
            raw_component_window_spectrum_real=raw_spectra.real,
            raw_component_window_spectrum_imag=raw_spectra.imag,
            cleaned_component_window_spectrum_real=cleaned_spectra.real,
            cleaned_component_window_spectrum_imag=cleaned_spectra.imag,
            station_response_direct_removed_spectral=response,
            station_illumination=illumination,
            station_azimuth_deg=station_azimuth,
            xyz_m=xyz,
            source_xyz_m=source_xyz,
            grid_shape=grid_shape,
            projection_prior=prior,
            formal_features_sha256=np.asarray(sha256(inputs["phase_features"])),
            raw_candidate_manifest_sha256=np.asarray(sha256(inputs["raw_candidate_manifest"])),
            v1129_config_sha256=np.asarray(sha256(base_config_path)),
            failed_independent_035_sha256=np.asarray(sha256(root / config["inputs"]["failed_independent_audit_035"]["path"])),
            proof_code_sha256=np.asarray(sha256(Path(__file__).resolve())),
            direct_removal_proof_status=np.asarray("COMPONENT_WINDOW_PROOF_CANDIDATE_NOT_ACTIVATED"),
            known_structure_input_count=np.asarray(0, dtype=np.int8),
            machine_learning_operator_count=np.asarray(0, dtype=np.int8),
            kernel_or_GS_called=np.asarray(False),
            publication_allowed=np.asarray(False),
        )
        os.replace(temporary, package_path)
        phase_results[phase] = {
            "package": str(package_path.relative_to(root)),
            "package_sha256": sha256(package_path),
            "gates": phase_gates,
            "quality": {
                "component_formal_rows": len(component_rows),
                "component_window_or_pick_distinct_stations": sum(
                    1 for item in metrics if item["component_pick_spread_s"] > 1.0e-10
                ),
                "max_component_pick_spread_s": float(max(item["component_pick_spread_s"] for item in metrics)),
                "support_outside_component_formal_window_channels": sum(not row["support_inside_own_formal_window"] for row in component_rows),
                "validation_direct_ratio_median": float(np.median(direct_ratios)),
                "validation_direct_ratio_p90": float(np.quantile(direct_ratios, 0.9)),
                "cross_retention_median": float(np.median(cross_retentions)),
                "cross_retention_p10": float(np.quantile(cross_retentions, 0.1)),
                "unused_window_max_abs_change": float(unused_changes.max()),
                "injected_retention_abs_error_max": float(np.max(np.abs(injection_retentions - 1.0))),
                "scalar_response_relative_difference": scalar_relative_difference,
                "component_pick_minus_predicted_s_min_median_max": [
                    float(component_deltas.min()),
                    float(np.median(component_deltas)),
                    float(component_deltas.max()),
                ],
            },
            "fold_models": fold_models,
            "station_metrics": metrics,
        }

    parity_values_array = np.asarray(parity_values)
    parity_orientations_array = np.asarray(parity_orientations)
    parity_rays_array = np.asarray(parity_rays)
    parity_tapers_array = np.asarray(parity_tapers)
    parity_phases_array = np.asarray(parity_phases, dtype=np.int8)
    parity_cpu_array = np.asarray(parity_cpu)
    parity_fixture = output_dir / f"{event_id}_component_window_parity_fixture_v1144.npz"
    np.savez_compressed(parity_fixture, values=parity_values_array, orientations=parity_orientations_array, rays=parity_rays_array, component_tapers=parity_tapers_array, phase_code=parity_phases_array, clean_cpu=parity_cpu_array)
    parity_result = run_parity(root / config["inputs"]["component_gpu_parity_helper"]["path"], parity_fixture)
    parity_error = float(parity_result["numpy_vs_cuda_max_abs"])
    parity_gate = bool(parity_result["pass"] and (parity_error <= float(thresholds["cpu_gpu_atol"]) or float(parity_result["numpy_vs_cuda_max_rel"]) <= float(thresholds["cpu_gpu_rtol"])))
    negative_tests = {
        "pick_outside_window_rejected": False,
        "vertical_only_not_expanded_to_fake3C": all(
            len([item for item in phase_results[phase]["station_metrics"] if item["channels"] == 1]) == 2 for phase in ("P", "S")
        ),
    }
    try:
        compact_pick_taper(np.arange(100) / 10.0, -1.0, 0.0, 5.0, 0.4)
    except RuntimeError:
        negative_tests["pick_outside_window_rejected"] = True
    gates = {
        "config_and_nested_hashes": all(checks.values()),
        "phase_P_all_gates": all(phase_results["P"]["gates"].values()),
        "phase_S_all_gates": all(phase_results["S"]["gates"].values()),
        "cpu_gpu_parity": parity_gate,
        "negative_tests": all(negative_tests.values()),
        "quarantine52_vertical_only_no_fake3C": True,
        "known_structure_zero": True,
        "machine_learning_zero": True,
        "additional_UTC_zero": True,
        "kernel_GS_pointer_public_zero": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    manifest = {
        "schema": "formal-hinet-component-window-direct-wave-proof-manifest-v1144",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "requirement_id": "NR-SCIENCE-DIRECT-WAVE-COMPONENT-WINDOW-049",
        "status": status,
        "decision": "PASS_COMPONENT_WINDOW_PROOF_CANDIDATE_NOT_ACTIVATED" if status == "PASS" else "FAIL_COMPONENT_WINDOW_PROOF_REMAINS_BLOCKED",
        "event_id": event_id,
        "source_policy": "NIED_HINET_ONLY",
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        "proof_code": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256(Path(__file__).resolve())},
        "base_v1129": {"config": str(base_config_path.relative_to(root)), "config_sha256": sha256(base_config_path)},
        "gates": gates,
        "negative_tests": negative_tests,
        "phase_results": phase_results,
        "CPU_GPU_parity": {**parity_result, "cases": len(parity_values), "samples_per_case": 401, "max_abs_error": parity_error, "atol": thresholds["cpu_gpu_atol"], "rtol": thresholds["cpu_gpu_rtol"], "pass": parity_gate, "fixture": str(parity_fixture.relative_to(root)), "fixture_sha256": sha256(parity_fixture)},
        "proof_independence": {
            "operator_estimation": "low-band covariance ray correction from the other five station folds",
            "validation": "held-out station high-band covariance ray; never the removal projector",
            "unused_signal": "actual and injected formal-window samples outside removal support",
            "response": "canonical grid response recomputed directly from cleaned complex component spectra; scalar approximation prohibited",
        },
        "activation_state": "INACTIVE_PROOF_CANDIDATE",
        "known_structure_input_count": 0,
        "machine_learning_operator_count": 0,
        "additional_time_correction_s": 0,
        "kernel_or_GS_called": False,
        "pointer_modified": False,
        "publication_allowed": False,
    }
    manifest_path = output_dir / f"{event_id}_component_window_direct_wave_proof_v1144.manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    manifest, path = run(arguments.project, arguments.config, arguments.output_dir)
    print(json.dumps({
        "status": manifest["status"],
        "decision": manifest["decision"],
        "P": manifest["phase_results"]["P"]["quality"],
        "S": manifest["phase_results"]["S"]["quality"],
        "CPU_GPU_parity": manifest["CPU_GPU_parity"],
        "manifest": str(path),
    }, indent=2))
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
