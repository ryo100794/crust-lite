from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from crust_lite.config import AppConfig
from crust_lite.io.metadata import read_metadata, write_metadata
from crust_lite.paths import ProjectPaths

NOTICE_3D = (
    "この3D可視化は、入力データと仮定に基づく研究用の状態表示です。"
    "地震の発生日、発生場所、規模を断定的に予測するものではありません。"
    "防災判断には公的機関の情報を使用してください。"
)


def wrap_plotly_html(title: str, body_html: str, metadata: dict[str, Any]) -> str:
    metadata_text = html.escape(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; }}
    header {{ padding: 16px 20px; border-bottom: 1px solid #d9e2ec; background: #f8fafc; }}
    main {{ padding: 12px; }}
    .notice {{ font-weight: 600; color: #7c2d12; margin-top: 8px; }}
    .meta {{ font-size: 12px; color: #52606d; white-space: pre-wrap; margin: 12px 20px 24px; }}
    a {{ color: #0b5cad; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="notice">{NOTICE_3D}</div>
    <div>is_sample_data={str(metadata.get("is_sample_data", False)).lower()} / vertical_exaggeration={metadata.get("vertical_exaggeration")}</div>
  </header>
  <main>{body_html}</main>
  <section class="meta">{metadata_text}</section>
</body>
</html>
"""


def fallback_plotly_html(title: str, metadata: dict[str, Any]) -> str:
    payload = html.escape(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    body = f"""
<div id="crust-lite-fallback" style="height: 420px; border: 1px solid #d9e2ec; padding: 16px;">
  <h2>Plotly fallback renderer</h2>
  <p>Plotly Python is not installed in the local dependency directory. The metadata below was still generated.</p>
  <pre>{payload}</pre>
</div>
<script>
window.Plotly = window.Plotly || {{}};
window.Plotly.newPlot = window.Plotly.newPlot || function(id, data, layout) {{
  document.getElementById(id).setAttribute("data-plotly-fallback", "true");
}};
Plotly.newPlot("crust-lite-fallback", [], {{}});
</script>
"""
    return wrap_plotly_html(title, body, metadata)


def write_index(config: AppConfig, paths: ProjectPaths, metadata: dict[str, Any]) -> Path:
    links = [
        ("events_faults_timeseries.html", "震源・既知断層・推定断層 3D時系列"),
        ("stress_timeseries_3d.html", "応力変化 3D時系列"),
        ("failure_scenarios_3d.html", "100年シナリオ 3D"),
        ("japan_archipelago_context.html", "日本列島コンテキスト地図"),
    ]
    items = "\n".join(
        f'<li><a href="{html.escape(path)}">{html.escape(label)}</a></li>' for path, label in links
    )
    body = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>crust-lite 3D index</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 960px; margin: 32px auto; padding: 0 20px; color: #1f2933; }}
    .notice {{ padding: 12px 14px; background: #fff7ed; border-left: 4px solid #c2410c; font-weight: 600; }}
    code {{ background: #edf2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>crust-lite 3D outputs</h1>
  <p class="notice">{NOTICE_3D}</p>
  <p>region={html.escape(config.region.name)} / period={config.region.start_date} to {config.region.end_date} / is_sample_data={str(metadata.get("is_sample_data", False)).lower()}</p>
  <ul>{items}</ul>
  <h2>Metadata</h2>
  <pre>{html.escape(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))}</pre>
</body>
</html>
"""
    out = paths.outputs_3d / "index.html"
    out.write_text(body, encoding="utf-8")
    write_metadata(paths.outputs_3d / "metadata.json", metadata)
    return out


def load_3d_metadata(paths: ProjectPaths) -> dict[str, Any]:
    return read_metadata(paths.outputs_3d / "metadata.json")
