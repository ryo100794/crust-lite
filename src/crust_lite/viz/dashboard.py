from __future__ import annotations

import html
from pathlib import Path

from crust_lite.config import load_config
from crust_lite.io.metadata import read_metadata
from crust_lite.paths import ProjectPaths

DASHBOARD_NOTICE = (
    "このプロトタイプは地震発生日を予測するものではありません。"
    "表示される指標は、入力データと仮定に基づく相対的な断層候補スコアおよび破壊接近度です。"
    "防災判断には公的機関の情報を使用してください。"
)


def write_dashboard_stub(paths: ProjectPaths) -> Path:
    paths.outputs_dashboard.mkdir(parents=True, exist_ok=True)
    index_3d = paths.outputs_3d / "index.html"
    body = f"""<!doctype html>
<html lang="ja">
<head><meta charset="utf-8"><title>crust-lite dashboard</title></head>
<body>
  <h1>crust-lite dashboard</h1>
  <p>{html.escape(DASHBOARD_NOTICE)}</p>
  <p>Streamlit is not installed in local dependencies. Use the generated static outputs instead.</p>
  <p><a href="../3d/index.html">3D HTML index</a></p>
  <p>3D index exists: {str(index_3d.exists()).lower()}</p>
</body>
</html>
"""
    out = paths.outputs_dashboard / "dashboard_stub.html"
    out.write_text(body, encoding="utf-8")
    return out


def run_streamlit_app(config_path: str) -> None:
    try:
        import streamlit as st  # type: ignore
        import streamlit.components.v1 as components  # type: ignore
    except Exception:
        config = load_config(config_path)
        write_dashboard_stub(ProjectPaths.from_config(config))
        return

    config = load_config(config_path)
    paths = ProjectPaths.from_config(config)
    st.set_page_config(page_title="crust-lite", layout="wide")
    st.title("crust-lite")
    st.warning(DASHBOARD_NOTICE)
    meta = read_metadata(paths.outputs_3d / "metadata.json")
    st.write({"region": config.region.name, "is_sample_data": meta.get("is_sample_data", False)})
    index = paths.outputs_3d / "index.html"
    if index.exists() and index.stat().st_size < 8_000_000:
        components.html(index.read_text(encoding="utf-8"), height=720, scrolling=True)
    elif index.exists():
        st.markdown(f"[Open 3D HTML index]({index.as_posix()})")
    else:
        st.info("3D HTML has not been generated yet.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_streamlit_app(args.config)
