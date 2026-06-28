from __future__ import annotations

from crust_lite.geo import clamp01


def neutral_if_missing(value: float | None, neutral: float = 0.5) -> float:
    if value is None:
        return neutral
    return clamp01(float(value))


def fault_score(
    seismicity_planarity_score: float,
    mechanism_consistency_score: float | None,
    gnss_strain_gradient_score: float | None,
    waveform_residual_score: float | None,
    distance_from_known_fault_score: float,
) -> float:
    score = (
        0.35 * clamp01(seismicity_planarity_score)
        + 0.25 * neutral_if_missing(mechanism_consistency_score)
        + 0.20 * neutral_if_missing(gnss_strain_gradient_score)
        + 0.10 * neutral_if_missing(waveform_residual_score)
        + 0.10 * clamp01(distance_from_known_fault_score)
    )
    return clamp01(score)


def confidence_from_score(score: float, n_events: int) -> float:
    sample_factor = min(1.0, max(0.1, n_events / 20.0))
    return clamp01(0.25 + 0.75 * score * sample_factor)
