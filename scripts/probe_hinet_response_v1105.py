#!/usr/bin/env python3
"""Numerical probe of NIED full RESP removal on one isolated Hi-net trace."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from HinetPy import win32
from obspy import read, read_inventory


ROOT = Path("/workspace/equake/crust-lite")
RESP = {
    "type3": ROOT / "logs/nied_official_audit_v1104/sources/seed_type3.txt",
    "type4": ROOT / "logs/nied_official_audit_v1098/sources/seed_type4.txt",
}
CH = ROOT / "data/interim/phase_aware_gs_20260811/hinet_maintenance_20260822/raw/group0_raw/2026/hinet_20260804000046/stations_N_NYOH_N_ASBH_N_SSGH_N_KJSH_N_KGRH_N_OGAH/raw/0101_20260804.ch"
SAC = ROOT / "logs/nied_official_audit_v1102/representative/official_download_binary/N.ASBH.N.SAC"
OUT = ROOT / "logs/nied_official_audit_v1105/response_probe_N_ASBH_N.SAC"
AUDIT = ROOT / "logs/nied_official_audit_v1105/response_probe_v1105.audit.json"
PREFILT = [0.05, 0.10, 35.0, 45.0]
WATER_LEVEL_DB = 60.0


def main() -> None:
    channel = next(c for c in win32.read_ctable(str(CH)) if c.name == "N.ASBH" and c.component == "N")
    kind = "type4" if abs(channel.lsb_value - 1.021e-7) < 5e-12 else "type3"
    inv = read_inventory(str(RESP[kind]), format="RESP")
    template = inv[0][0][0].response
    response = copy.deepcopy(template)
    a0 = float(response.response_stages[0].normalization_factor)
    response.response_stages[0].stage_gain = float(channel.gain) / a0
    digitizer_gain = float(response.response_stages[1].stage_gain)
    response.instrument_sensitivity.value = response.response_stages[0].stage_gain * digitizer_gain

    trace = read(str(SAC))[0]
    nmps = np.asarray(trace.data, dtype=np.float64)
    counts_per_mps_ch = float(channel.gain) / float(channel.lsb_value)
    counts = nmps * 1.0e-9 * counts_per_mps_ch
    trace.data = counts.astype(np.float64)
    trace.stats.response = response
    trace.remove_response(
        output="VEL", pre_filt=PREFILT, water_level=WATER_LEVEL_DB,
        zero_mean=True, taper=True, taper_fraction=0.05,
    )
    corrected_mps = np.asarray(trace.data, dtype=np.float64)
    trace.data = (corrected_mps * 1.0e9).astype(np.float32)
    trace.stats.sac.kuser0 = "NIEDRESP"
    trace.stats.sac.kuser1 = "TYPE4" if kind == "type4" else "TYPE3"
    trace.stats.sac.user0 = float(channel.gain)
    trace.stats.sac.user1 = float(channel.lsb_value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trace.write(str(OUT), format="SAC", byteorder="<")

    frequencies = np.array([0.1, 0.35, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0])
    response_spectrum, response_freq = response.get_evalresp_response(
        t_samp=0.01, nfft=65536, output="VEL"
    )
    indices = [int(np.argmin(np.abs(response_freq - f))) for f in frequencies]
    record = {
        "schema": "hinet-official-full-response-probe-v1105",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "station": channel.name,
        "component": channel.component,
        "response_type": kind,
        "official_resp": str(RESP[kind].relative_to(ROOT)),
        "official_A0": a0,
        "official_stage1_gain_replaced_with_CH_gain_over_A0": float(response.response_stages[0].stage_gain),
        "official_digitizer_gain": digitizer_gain,
        "official_final_sensitivity_replaced": float(response.instrument_sensitivity.value),
        "CH_counts_per_mps_used_to_reconstruct_counts": counts_per_mps_ch,
        "research_deconvolution": {
            "output": "VEL", "pre_filt_hz": PREFILT, "water_level_db": WATER_LEVEL_DB,
            "zero_mean": True, "taper": True, "taper_fraction": 0.05,
        },
        "response_samples": [
            {"frequency_hz": float(frequencies[i]), "amplitude_counts_per_mps": float(abs(response_spectrum[idx])), "phase_rad": float(np.angle(response_spectrum[idx]))}
            for i, idx in enumerate(indices)
        ],
        "input_nmps": {"finite": bool(np.isfinite(nmps).all()), "rms": float(np.sqrt(np.mean(nmps * nmps))), "peak": float(np.max(np.abs(nmps)))},
        "reconstructed_counts": {"finite": bool(np.isfinite(counts).all()), "rms": float(np.sqrt(np.mean(counts * counts))), "peak": float(np.max(np.abs(counts)))},
        "corrected_nmps": {"finite": bool(np.isfinite(trace.data).all()), "rms": float(np.sqrt(np.mean(np.asarray(trace.data, dtype=np.float64) ** 2))), "peak": float(np.max(np.abs(trace.data)))},
        "output_sac": str(OUT.relative_to(ROOT)),
        "pass": bool(np.isfinite(trace.data).all() and trace.stats.npts == 30000),
        "known_structure_used": False,
        "machine_learning_used": False,
    }
    AUDIT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
