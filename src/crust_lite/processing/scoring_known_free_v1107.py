from __future__ import annotations

from crust_lite.geo import clamp01


FAULT_SCORE_WEIGHTS = {
    "seismicity_planarity": 7.0 / 18.0,
    "mechanism_consistency": 5.0 / 18.0,
    "gnss_strain_gradient": 2.0 / 9.0,
    "waveform_residual": 1.0 / 9.0,
}


def neutral_if_missing(value: float | None, neutral: float = 0.5) -> float:
    if value is None:
        return neutral
    return clamp01(float(value))


def fault_score(
    seismicity_planarity_score: float,
    mechanism_consistency_score: float | None,
    gnss_strain_gradient_score: float | None,
    waveform_residual_score: float | None,
) -> float:
    """Score inferred faults from observation-derived evidence only.

    External known faults, plates, and slabs are deliberately excluded. They
    may be used only by posthoc correlation or viewer-overlay code.
    """
    score = (
        FAULT_SCORE_WEIGHTS["seismicity_planarity"] * clamp01(seismicity_planarity_score)
        + FAULT_SCORE_WEIGHTS["mechanism_consistency"]
        * neutral_if_missing(mechanism_consistency_score)
        + FAULT_SCORE_WEIGHTS["gnss_strain_gradient"]
        * neutral_if_missing(gnss_strain_gradient_score)
        + FAULT_SCORE_WEIGHTS["waveform_residual"]
        * neutral_if_missing(waveform_residual_score)
    )
    return clamp01(score)


def confidence_from_score(score: float, n_events: int) -> float:
    sample_factor = min(1.0, max(0.1, n_events / 20.0))
    return clamp01(0.25 + 0.75 * score * sample_factor)
