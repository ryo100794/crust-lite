"""Proposed single canonical CLI for the formal normal data plane."""

from __future__ import annotations

import argparse
import json
from typing import Any

from crust_lite.io.formal_input_resolver_v1120 import FormalInputError, resolve_formal_inputs


def command_fetch(config_path: str, *, sample: bool = False) -> dict[str, Any]:
    """Verify existing formal inputs; normal fetch performs no network or writes."""
    if sample:
        raise FormalInputError(
            "sample data is never a formal normal fallback; use an explicitly isolated "
            "development workflow"
        )
    return resolve_formal_inputs(config_path)



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crust-lite-data-v1120")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "formal-input-status"):
        item = sub.add_parser(name)
        item.add_argument("--config", default="configs/formal_hinet_only_v1120.yml")
        if name == "fetch":
            item.add_argument("--sample", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = command_fetch(args.config, sample=getattr(args, "sample", False))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
