from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from crust_lite.config import AppConfig


@dataclass(frozen=True)
class ExecutionPlan:
    operation: str
    engine: str
    memory_mode: str
    available_memory_bytes: int
    memory_budget_bytes: int
    estimated_rows: int
    estimated_row_bytes: int
    estimated_bytes: int
    use_in_memory: bool
    batch_rows: int
    reason: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "engine": self.engine,
            "memory_mode": self.memory_mode,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "estimated_rows": self.estimated_rows,
            "estimated_row_bytes": self.estimated_row_bytes,
            "estimated_bytes": self.estimated_bytes,
            "use_in_memory": self.use_in_memory,
            "batch_rows": self.batch_rows,
            "reason": self.reason,
        }


def available_memory_bytes() -> int:
    meminfo = "/proc/meminfo"
    try:
        with open(meminfo, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return 1_000_000_000


def choose_execution_plan(
    config: AppConfig,
    operation: str,
    engine: str,
    estimated_rows: int,
    estimated_row_bytes: int = 512,
) -> ExecutionPlan:
    resources = config.resources
    available = max(1, available_memory_bytes())
    budget = max(64 * 1024 * 1024, int(available * resources.max_memory_fraction))
    estimated = max(0, int(estimated_rows)) * max(1, int(estimated_row_bytes))
    row_limit = resources.in_memory_row_limit
    mode = resources.memory_mode

    if mode == "low":
        use_memory = estimated_rows <= min(row_limit, 10_000) and estimated <= budget // 4
        reason = "low_memory_mode"
    elif mode == "high":
        use_memory = estimated_rows <= max(row_limit, row_limit * 4) and estimated <= budget
        reason = "high_memory_mode_budget_allows" if use_memory else "high_memory_mode_budget_exceeded"
    else:
        use_memory = estimated_rows <= row_limit and estimated <= budget
        reason = "within_in_memory_budget" if use_memory else "database_preferred_for_size"

    if estimated_rows >= resources.db_bulk_load_row_threshold:
        use_memory = False
        reason = "bulk_load_threshold_exceeded"

    row_bytes = max(1, estimated_row_bytes)
    batch_by_budget = max(resources.batch_rows_min, min(resources.batch_rows_max, budget // max(row_bytes, 1)))
    batch_rows = int(max(resources.batch_rows_min, min(resources.batch_rows_max, batch_by_budget)))

    return ExecutionPlan(
        operation=operation,
        engine=engine,
        memory_mode=mode,
        available_memory_bytes=available,
        memory_budget_bytes=budget,
        estimated_rows=int(estimated_rows),
        estimated_row_bytes=int(estimated_row_bytes),
        estimated_bytes=int(estimated),
        use_in_memory=bool(use_memory),
        batch_rows=batch_rows,
        reason=reason,
    )
