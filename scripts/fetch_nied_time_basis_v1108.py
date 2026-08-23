#!/usr/bin/env python3
"""Archive the official public NIED page that labels waveform time as JST."""

from __future__ import annotations

import hashlib
import json
import ssl
from datetime import UTC, datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager


URL = "https://www.hinet.bosai.go.jp/strace/?LANG=en"
OUT = Path("/workspace/equake/crust-lite/logs/nied_official_audit_v1108/sources")


class LegacyTLSAdapter(HTTPAdapter):
    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = context
        self.poolmanager = PoolManager(num_pools=connections, maxsize=maxsize, block=block, **kwargs)


def main() -> None:
    session = requests.Session()
    session.mount("https://www.hinet.bosai.go.jp", LegacyTLSAdapter())
    response = session.get(URL, timeout=60)
    response.raise_for_status()
    body = response.content
    decoded = body.decode("euc_jp", "replace")
    if "Date and Time (JST)" not in decoded:
        raise RuntimeError("official JST label missing")
    OUT.mkdir(parents=True, exist_ok=True)
    page = OUT / "continuous_waveform_images_en.html"
    page.write_bytes(body)
    audit = {
        "schema": "nied-official-time-basis-source-v1108",
        "url": URL,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "saved_path": str(page),
        "official_spec": "The official NIED Hi-net continuous-waveform page labels its Date and Time selector as JST.",
        "marker_verified": "Date and Time (JST)",
    }
    (OUT / "time_basis_v1108.audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
