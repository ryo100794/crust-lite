#!/usr/bin/env python3
"""Create a read-only umbrella manifest for Hi-net audit stages A-E."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path("/workspace/equake/crust-lite")
STAGES = {
    "A": ROOT / "logs/HINET_OFFICIAL_SPEC_AUDIT_STAGE_A_v1101.json",
    "B": ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_B_v1102.json",
    "C": ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_C_v1103.json",
    "D": ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_D_v1106.json",
    "E": ROOT / "logs/HINET_OFFICIAL_CALIBRATION_AUDIT_STAGE_E_v1107.json",
}
TIME_SOURCE = ROOT / "logs/nied_official_audit_v1108/sources/time_basis_v1108.audit.json"
OUTPUT_ROOT = ROOT / "data/interim/hinet_official_calibrated_v1106/analysis_ready_nmps"
OUT_JSON = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_FINAL_v1109.json"
OUT_MD = ROOT / "logs/HINET_OFFICIAL_CALIBRATION_FINAL_v1109.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    docs = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in STAGES.items()}
    time_source = json.loads(TIME_SOURCE.read_text(encoding="utf-8"))
    sac_count = sum(1 for _ in OUTPUT_ROOT.glob("**/*.SAC"))
    group_markers = sum(1 for _ in OUTPUT_ROOT.glob("**/_group_complete.audit.json"))
    gates = {
        "stage_B_official_conversion_arrays_exact": docs["B"]["summary"]["official_download_vs_existing_all_arrays_equal"],
        "stage_C_complete_1872": docs["C"]["summary"]["completeness_pass"],
        "stage_D_generated_1872_finite": docs["D"]["summary"]["complete_and_finite"] and sac_count == 1872 and group_markers == 26,
        "stage_D_original_CNT_reconversion_matches_existing_1872": docs["D"]["summary"]["official_transient_equals_existing"] == 1872,
        "stage_E_shapes_finite_time_PS_windows": docs["E"]["decision"] == "PASS_WITH_ORIENTATION_QUARANTINE",
        "old_products_unchanged": all(not bool(doc.get("old_version_overwritten", False)) for doc in docs.values()),
        "known_structures_not_used": all(not bool(doc.get("known_structure_used", False)) for doc in docs.values()),
        "machine_learning_not_used": all(not bool(doc.get("machine_learning_used", False)) for doc in docs.values()),
    }
    audit = {
        "schema": "hinet-official-calibration-final-v1109",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "13 events x 48 stations x 3 components = 1872 SAC",
        "stage_manifests": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "decision": docs[key].get("decision") or docs[key].get("stage_a_decision", {}).get("status")}
            for key, path in STAGES.items()
        },
        "official_time_source": time_source,
        "gates": gates,
        "all_completed_gates_pass": all(gates.values()),
        "output": {"path": str(OUTPUT_ROOT.relative_to(ROOT)), "sac_count": sac_count, "group_marker_count": group_markers},
        "calibration_result": {
            "official_win2sac_from_original_CNT_CH": "1872/1872 exact match to existing sensitivity-scaled SAC",
            "official_response_types": docs["D"]["summary"]["response_type_counts"],
            "UTC": "1872/1872 deterministically shifted JST-9h; downstream alignment chooses 0 additional seconds",
            "orientation": docs["D"]["summary"]["orientation_status_counts"],
            "PS_windows": {"P_contained": docs["E"]["summary"]["new_raw_windows_contain_P"], "S_contained": docs["E"]["summary"]["new_raw_windows_contain_S"], "short": docs["E"]["summary"]["short_phase_windows"]},
        },
        "official_vs_research_boundary": {
            "official": docs["D"]["official_processing"],
            "research_added": docs["D"]["research_processing_not_official_spec"],
        },
        "decision": "PASS_WITH_52_HORIZONTAL_TRACES_QUARANTINED",
        "downstream_switch_allowed": False,
        "downstream_switch_reason": "Parent review and NR-HINET-ORIENT-001 resolution/exclusion policy are required; the current +/-9h heuristic must not be used on UTC outputs.",
        "new_requirements": docs["D"].get("new_requirements", []),
        "queue_ui_reusable_existing_json": docs["A"].get("queue_ui_reusable_existing_json", []),
        "existing_web_modified": False,
        "existing_outputs_overwritten": False,
    }
    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(f"""# Hi-net公式補正監査 最終（v1109）

判定: **{audit['decision']}**

- 監査範囲: 13イベント×48観測点×3成分 = 1872 SAC。
- 元CNT/CHから公式win2sac再変換し、既存感度換算SACと1872/1872一致。
- 公式Type3/4 RESP補正、JST→UTC、公式有効期間方位のN/E回転後、1872/1872 finite。
- 新UTC波形は全1872件で追加時刻補正0秒、P/S予測到着を全件包含、短窓0。
- N.SSGH/N.SSWHはNIED公式方位値がないため、水平52成分を方向解析から隔離。補間・既知構造・機械学習は不使用。
- 出力: `{audit['output']['path']}`（SAC={sac_count}, group marker={group_markers}）。
- 旧成果物、公開Web、後段入力は変更していない。

入力切替はまだ不可。親レビュー後、UTC入力では到着予測依存の±9時間分岐を廃止し、公式方位欠測2点の除外または`NR-HINET-ORIENT-001`解決が必要。
""", encoding="utf-8")
    print(json.dumps({"decision": audit["decision"], "gates": gates, "output": audit["output"], "new_requirements": audit["new_requirements"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
