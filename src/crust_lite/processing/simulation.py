from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from crust_lite.config import AppConfig
from crust_lite.geo import clamp01
from crust_lite.io.database import materialize_known_tables, stress_sum_by_segment
from crust_lite.io.geopackage import read_features
from crust_lite.io.parquet import read_table, write_table
from crust_lite.logging import get_logger
from crust_lite.paths import ProjectPaths

LOGGER = get_logger(__name__)


def _segment_ids(paths: ProjectPaths) -> list[str]:
    ids: list[str] = []
    for path in (paths.data_processed / "fault_segment.gpkg", paths.data_processed / "inferred_faults.gpkg"):
        if path.exists():
            ids.extend(str(feature.get("properties", {}).get("segment_id")) for feature in read_features(path))
    return [segment_id for segment_id in ids if segment_id and segment_id != "None"]


def run_simulation(config: AppConfig, paths: ProjectPaths) -> dict[str, Any]:
    segment_ids = _segment_ids(paths)
    if not segment_ids:
        raise ValueError("No fault segments are available for simulation")
    stress_sums = stress_sum_by_segment(paths)
    stress_rows: list[dict[str, Any]] = []
    stress_by_segment: dict[str, list[float]] = defaultdict(list)
    if stress_sums is not None:
        for seg, value in stress_sums.items():
            stress_by_segment[seg].append(value)
    else:
        stress_rows = read_table(paths.data_processed / "stress_state.parquet") if (
            paths.data_processed / "stress_state.parquet"
        ).exists() else []
        for row in stress_rows:
            seg = str(row["segment_id"])
            if str(row.get("cfs_pa", "")) not in {"", "nan", "None"}:
                stress_by_segment[seg].append(float(row["cfs_pa"]))
            else:
                # Explicitly a proxy for scenario ranking, not a physical stress.
                stress_by_segment[seg].append(float(row.get("cfs_score_approx", 0.0)) * 1.0e5)

    rng = np.random.default_rng(config.simulation.random_seed)
    years = list(range(0, config.simulation.years + 1))
    rows: list[dict[str, Any]] = []
    for segment_id in segment_ids:
        friction = rng.uniform(*config.simulation.effective_friction_range, config.simulation.n_ensemble)
        cohesion = rng.uniform(*config.simulation.cohesion_pa_range, config.simulation.n_ensemble)
        normal = rng.uniform(
            *config.simulation.effective_normal_stress_pa_range,
            config.simulation.n_ensemble,
        )
        strength = cohesion + friction * normal
        stress_rate = rng.lognormal(mean=np.log(1.0e4), sigma=0.6, size=config.simulation.n_ensemble)
        initial = rng.uniform(0.0, 0.25, config.simulation.n_ensemble) * strength
        cfs_values = stress_by_segment.get(segment_id, [0.0])
        cfs_past = float(np.sum(cfs_values))
        uncertainty = clamp01(float(np.std(stress_rate) / (np.mean(stress_rate) + 1e-9)))
        for year in years:
            tau = initial + stress_rate * year
            failure_index = np.maximum(0.0, tau + cfs_past) / strength
            rows.append(
                {
                    "segment_id": segment_id,
                    "year": year,
                    "failure_index_p05": float(np.quantile(failure_index, 0.05)),
                    "failure_index_p50": float(np.quantile(failure_index, 0.50)),
                    "failure_index_p95": float(np.quantile(failure_index, 0.95)),
                    "prob_index_gt_1": float(np.mean(failure_index > 1.0)),
                    "stress_rate_pa_per_yr": float(np.mean(stress_rate)),
                    "uncertainty_score": uncertainty,
                    "simulation_notes": (
                        "failure_index is dimensionless; values above 1 are model-threshold "
                        "exceedance, not an earthquake occurrence statement"
                    ),
                }
            )
    write_table(
        rows,
        paths.outputs_tables / "failure_scenarios.parquet",
        {
            "is_sample_data": any(str(row.get("is_sample_data", "")).lower() == "true" for row in stress_rows),
            "years": config.simulation.years,
            "n_ensemble": config.simulation.n_ensemble,
        },
    )
    _write_ranking(rows, paths)
    materialize_known_tables(paths)
    LOGGER.info("Wrote %d scenario rows", len(rows))
    return {"scenario_rows": len(rows), "segment_count": len(segment_ids)}


def _write_ranking(rows: list[dict[str, Any]], paths: ProjectPaths) -> None:
    targets = {0: "latest", 10: "10yr", 30: "30yr", 50: "50yr", 100: "100yr"}
    selected = [row for row in rows if int(row["year"]) in targets]
    ranked = sorted(selected, key=lambda row: (int(row["year"]), -float(row["failure_index_p50"])))
    output_rows: list[dict[str, Any]] = []
    rank_by_year: dict[int, int] = defaultdict(int)
    for row in ranked:
        year = int(row["year"])
        rank_by_year[year] += 1
        output_rows.append(
            {
                "horizon": targets[year],
                "rank": rank_by_year[year],
                "segment_id": row["segment_id"],
                "failure_index_p50": row["failure_index_p50"],
                "failure_index_p95": row["failure_index_p95"],
                "prob_index_gt_1": row["prob_index_gt_1"],
                "uncertainty_score": row["uncertainty_score"],
                "interpretation": "relative model state only; not an occurrence prediction",
            }
        )
    write_table(output_rows, paths.outputs_tables / "fault_ranking.csv", {"physical_format": "csv"})
