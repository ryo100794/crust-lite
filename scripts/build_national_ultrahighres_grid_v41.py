#!/usr/bin/env python3
"""Subdivide the complete national 3.75 km domain to 1.875 km."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEPTHS = np.asarray([0.5,1,1.5,2,2.5,3,4,5,6,7.5,9,11,13,15,18,21,24,27,30], float)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("source", type=Path); ap.add_argument("output", type=Path); args = ap.parse_args()
    old = pd.read_parquet(args.source, columns=["x_m", "y_m"]).drop_duplicates()
    offsets = (-937.5, 937.5)
    xy = np.asarray([(float(r.x_m)+dx, float(r.y_m)+dy) for r in old.itertuples(index=False) for dx in offsets for dy in offsets], np.float64)
    xy = np.unique(xy, axis=0)
    # Avoid building a Python tuple list for ~12 million nodes.
    x = np.repeat(xy[:, 0], len(DEPTHS)); y = np.repeat(xy[:, 1], len(DEPTHS)); z = np.tile(DEPTHS, len(xy))
    out = pd.DataFrame({"x_m": x, "y_m": y, "z_km": z})
    args.output.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(args.output, compression="zstd", index=False)
    audit = {
        "grid_km": 1.875, "source_horizontal_cells": int(len(old)), "horizontal_cells": int(len(xy)),
        "depths_km": DEPTHS.tolist(), "fixed_nodes": int(len(out)), "analysis_footprint_changed": False,
        "shallow_vertical_spacing_km": .5, "deep_vertical_spacing_km_max": 3.0,
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
