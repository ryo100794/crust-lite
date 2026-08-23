#!/usr/bin/env python3
"""Isolated standalone progressive viewer dashboard; never public by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path("/workspace/equake/crust-lite")
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from viewer_progressive_selector_flat_v1158 import Selector

VIEWER = SCRIPTS / "viewer_10101_canonical_v1120_r4.html"
VIEWER_SHA256 = "621162ff53781a4314d66d477ecc590475d1d7c99d80de3ef9053f2e973a9d36"


def viewer() -> bytes:
    body = VIEWER.read_bytes()
    if hashlib.sha256(body).hexdigest() != VIEWER_SHA256:
        raise RuntimeError("viewer asset hash mismatch")
    return body


def build_selector(root: Path = ROOT) -> Selector:
    return Selector(
        root,
        root / "data/interim/viewer_progressive_v1148/manifests",
        root / "configs/viewer_progressive_authority_candidate_v1148.json",
        root / "data/operations/agent_queue.sqlite",
        root / "data/interim/viewer_progressive_flat_v1158/runtime",
    )


SELECTOR: Selector | None = None


def selector() -> Selector:
    if SELECTOR is None:
        raise RuntimeError("progressive selector is not initialized")
    return SELECTOR


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_raw(self, body, content_type, status=200, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_guarded(self, producer):
        current = selector()
        try:
            with current.authorized_snapshot() as snapshot:
                body, content_type, status, cache = producer(snapshot)
                current.confirm(snapshot)
                self.send_raw(body, content_type, status, cache)
        except Exception:
            self.send_raw(
                b"authoritative artifact unavailable\n",
                "text/plain; charset=utf-8",
                503,
                "no-store",
            )

    def do_GET(self):
        parsed = urlsplit(self.path)
        current = selector()

        def response(snapshot):
            if parsed.path in ("/", "/viewer-single-event", "/viewer-single-event/"):
                return viewer(), "text/html; charset=utf-8", 200, "no-store"
            if parsed.path == "/api/single-event-gs/meta":
                meta = dict(snapshot.meta)
                meta["progressive_selection"] = current.status_for(snapshot)
                return json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode(), "application/json; charset=utf-8", 200, "no-store"
            if parsed.path == "/api/single-event-gs/display":
                phase = parse_qs(parsed.query).get("phase", ["P"])[0]
                if phase not in snapshot.packets:
                    return b"bad request\n", "text/plain", 400, "no-store"
                return snapshot.packets[phase], "application/octet-stream", 200, "no-store"
            if parsed.path == "/api/viewer-progressive-status":
                body = json.dumps(current.status_for(snapshot), ensure_ascii=False, separators=(",", ":")).encode()
                return body, "application/json; charset=utf-8", 200, "no-store"
            if parsed.path == "/artifacts/japan-v987-reference.png":
                return snapshot.poster, "image/png", 200, "public,max-age=300"
            return b"not found\n", "text/plain", 404, "no-store"

        self.send_guarded(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18188)
    args = parser.parse_args()
    viewer()
    global SELECTOR
    SELECTOR = build_selector()
    threading.Thread(target=SELECTOR.preload, kwargs={"interval": 3.0}, daemon=True).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
