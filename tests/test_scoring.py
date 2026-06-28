from __future__ import annotations

from crust_lite.processing.scoring import fault_score


def test_score_range() -> None:
    score = fault_score(1.4, None, None, None, -1.0)
    assert 0.0 <= score <= 1.0
