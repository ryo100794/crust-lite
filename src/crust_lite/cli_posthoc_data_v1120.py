"""Separate explicit posthoc external-geology inspection entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crust_lite.config import load_config
from crust_lite.external_geology import POSTHOC_MODE, resolve_external_geology
from crust_lite.io.geopackage import read_features
from crust_lite.paths import ProjectPaths


def inspect_dataset(
    config_path: str,
    dataset_id: str,
    member: str,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode != POSTHOC_MODE:
        raise ValueError(f"explicit mode={POSTHOC_MODE} is required")
    config = load_config(config_path)
    paths = ProjectPaths.from_config(config)
    with resolve_external_geology(paths, dataset_id, mode=mode) as resolved:
        path = resolved.member_path(member)
        feature_count: int | None = None
        if path.suffix.lower() == ".gpkg":
            feature_count = len(read_features(path))
        result = {
            "schema": "external-geology-posthoc-inspection-v1120",
            "mode": POSTHOC_MODE,
            "dataset_id": dataset_id,
            "member": member,
            "feature_count": feature_count,
            "normal_artifacts_created": 0,
            "analysis_input": False,
            "allowed_uses": ["posthoc correlation", "viewer overlay"],
            "provenance": resolved.provenance,
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crust-lite-posthoc-data-v1120")
    parser.add_argument("--config", default="configs/formal_hinet_only_v1120.yml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--mode", required=True, choices=[POSTHOC_MODE])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            inspect_dataset(
                args.config,
                args.dataset,
                args.member,
                mode=args.mode,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
