from __future__ import annotations

import pytest

from crust_lite.processing.fault_inference import infer_faults


def test_legacy_known_structure_path_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="legacy known-structure-contaminated"):
        infer_faults(None, None)  # type: ignore[arg-type]
