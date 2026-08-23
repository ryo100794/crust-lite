#!/usr/bin/env python3
"""Quantify known-fault feedback in the legacy 832-row SA artifact."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    features = json.loads(args.input.read_text())["features"]
    rows = []
    for feature in features:
        p = feature["properties"]
        support = clamp(
            0.70 * math.log1p(max(0.0, float(p.get("n_support", 0)))) / math.log1p(80.0)
            + 0.30
            * math.log1p(max(1.0, float(p.get("frequency_count", 1))))
            / math.log1p(8.0)
        )
        independent = (
            0.26 * clamp(float(p.get("synthetic_aperture_linearity_score", 0.0)))
            + 0.28 * clamp(float(p.get("surface_wave_anomaly_score", 0.0)))
            + 0.22 * clamp(float(p.get("scattering_lineament_score", 0.0)))
            + 0.10 * support
        )
        known = clamp(float(p["known_fault_feedback_weight"]))
        reconstructed = clamp(independent + 0.14 * known)
        known_free = clamp(independent / 0.86)
        rows.append(
            {
                "segment_id": p["segment_id"],
                "stored": float(p["fault_score"]),
                "reconstructed": reconstructed,
                "known_free": known_free,
                "known_weight": known,
            }
        )
    by_stored = sorted(rows, key=lambda x: (-x["stored"], x["segment_id"]))
    by_clean = sorted(rows, key=lambda x: (-x["known_free"], x["segment_id"]))
    rank_old = {row["segment_id"]: idx for idx, row in enumerate(by_stored)}
    rank_new = {row["segment_id"]: idx for idx, row in enumerate(by_clean)}
    shifts = [abs(rank_old[key] - rank_new[key]) for key in rank_old]
    deltas = [row["stored"] - row["known_free"] for row in rows]
    reconstruction_errors = [abs(row["stored"] - row["reconstructed"]) for row in rows]
    result = {
        "schema": "nr-known-leak-008-legacy-sa-counterfactual-v1107",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "rows": len(rows),
        "legacy_formula": "0.26 linearity + 0.28 surface + 0.22 scattering + 0.10 support + 0.14 known_feedback",
        "known_free_counterfactual": "renormalize independent 0.86 subtotal to unit weight",
        "max_formula_reconstruction_error": max(reconstruction_errors),
        "score_delta_stored_minus_known_free": {
            "min": min(deltas),
            "max": max(deltas),
            "mean": sum(deltas) / len(deltas),
        },
        "rank_shift": {
            "rows_changed": sum(shift > 0 for shift in shifts),
            "max_absolute_positions": max(shifts),
            "mean_absolute_positions": sum(shifts) / len(shifts),
            "top_100_overlap": len(
                {row["segment_id"] for row in by_stored[:100]}
                & {row["segment_id"] for row in by_clean[:100]}
            ),
        },
        "pass": max(reconstruction_errors) < 1e-12,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
