#!/usr/bin/env python3
"""Fetch the official NIED Hi-net Type-3 RESP without logging credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from HinetPy import Client

URL = "https://hinetwww11.bosai.go.jp/auth/seed/dlDialogue.php?type=hinet_2006&stcd="


def load_env(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    load_env(args.env_file)
    user = os.environ.get("HINET_USER") or os.environ.get("HINET_USERNAME")
    password = os.environ.get("HINET_PASSWORD") or os.environ.get("HINET_PASS")
    if not user or not password:
        raise RuntimeError("Hi-net credentials not configured")
    client = Client(user, password, timeout=60, retries=1)
    response = client.session.get(URL, timeout=60)
    response.raise_for_status()
    body = response.content
    if b"CHANNEL RESPONSE DATA" not in body:
        raise RuntimeError("official RESP marker missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "seed_type3.txt"
    path.write_bytes(body)
    record = {
        "schema": "nied-official-response-type3-fetch-v1104",
        "credentials_logged": False,
        "url": URL,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "saved_path": str(path),
    }
    manifest = args.output_dir / "seed_type3_fetch_v1104.audit.json"
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
