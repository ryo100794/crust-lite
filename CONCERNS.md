# crust-lite 懸案事項ログ

更新日時: 2026-06-30 10:43 UTC

このファイルは、H200/5090での本計算前に潰すべき懸案、検証待ち、既知の簡略化を継続的に記録するためのログです。ここに書く数値は作業時点の状態であり、研究用の相対評価プロトタイプの状態記録です。地震の発生日・場所・規模を断定的に予測するものではありません。

## 現在の確認済み状態

- 高密度WebGLスプラットHTMLを再生成済み。
- `array_projection_splats.html` は Plotly ではなく WebGL2 point sprite 方式。
- 表示スプラット数: `113,504`
- 地形コンテキストメッシュ: 無効。地下スプラットを隠さないため、地表面は描画せず輪郭線のみ。
- 日本列島輪郭頂点数: `18,949`、Natural Earth 10m由来を約1km目標で高密度化。
- Canvas device pixel ratio 上限: `4`
- 点スプライト上限: `384 px`
- スマホ向け操作: pointer events による1本指回転、2本指パン・ピンチズームを実装済み。
- 使用中の波形由来スペクトルDB: `289,590` 行、`4,121` イベント、`794` 観測点。
- 高密度投影結果: `113,504` array projection rows / Gaussian splat primitives。

## データ取得・網羅性

- Hi-netの「全期間・全国・全イベント・全観測点」波形は未取得。現在は登録・接続確認後に取得できた範囲を処理している。
- 現在のスプラットは取得済みスペクトルから高スコア候補を抽出した圧縮表現であり、生波形の全サンプルをそのまま全部描画しているわけではない。
- 古い時代の国内データは粒度・精度・欠測率が下がるため、長期データとして取り込む場合は `source`, `era`, `location_uncertainty`, `magnitude_type`, `network_coverage`, `quality_class` を明示して層別解析する必要がある。
- JMA/F-net/GEONET/J-SHIS/K-NET/KiK-net/Hi-net を全国・長期で統合するには、取得状況、ライセンス、認証、再配布可否、引用条件をデータソース別に管理する必要がある。

## DB・データ構造

- 現状の表形式データ、スペクトル、特徴量、ランキングは DuckDB + Parquet が妥当。
- 大規模3Dボリューム、時系列メッシュ、Gaussian splattingの多段LODには DuckDB単独では不足する可能性がある。
- v1候補として、表形式は DuckDB/Parquet、チャンク化グリッド・ボリュームは Zarr または TileDB、GPU入力シャードは Arrow/Parquet の併用を検討する。
- CPU前処理でイベント単位のシャードを作り、GPU側では必要なシャードだけを読む構造にする。

## アルゴリズム・科学的妥当性

- 現在のスプラットは単なる震源点重ね描きではなく、位相・群遅延を使った合成開口風の相対投影と、direct / reflected / scattered / residual の簡易分類を含む。
- ただし、これは一意な地下構造インバージョンではない。反射・散乱は遅延相の代理指標であり、実構造の断定には使えない。
- 現在の速度モデルは簡略化されている。多層速度構造、観測点補正、P/S相分離、到達時刻ピッキング品質、反射面候補の幾何拘束を追加する必要がある。
- 既知断層との比較は、スプラット密度・応力近似・地震活動クラスタ・GNSS勾配を重ねた相対評価であり、断層位置の確定ではない。
- WebGLスプラットの層状構造は、現時点では実地下構造の確定シグナルではない。初期診断では、震源深さの整数km一致率が `0.460389`、遅延相の探索窓30秒クリップが `24,611` 件、遅延相中 `0.434187`。10km、33km、35km、62km付近の層はカタログ深さ丸めと `event_depth + 0.5 * late_delay * 3.5 km/s` のモデル由来が強かった。対策として、構造スプラット生成では情報を削除せず、全 `113,504` 候補を保持し、`splat_role` と `structure_amplitude` で構造寄与を分離するよう変更済み。内訳は structure `32,072`、source_anchor `56,821`、diagnostic_rejected `24,611`。GPU/voxel密度は `structure_amplitude > 0` の候補だけを展開し、現状 `554,843` voxel 行、`543` shard。残る層状傾向は遅延相モデル、均質速度 `3.5 km/s`、投影グリッド、周波数窓の影響として検証する。

## 可視化

- WebGL版はPlotly版より軽く、現時点ではさらに高密度化可能。
- スマホ操作は実装済みだが、実機でのドラッグ、ピンチ、視点保持、Canvasサイズ、DPR=4時のメモリ使用を確認する必要がある。
- `events_faults_timeseries.html` / `stress_timeseries_3d.html` / `failure_scenarios_3d.html` は、フレーム変更時の視点維持、古い描画状態の残留、スライダー時間変化の実質的な見え方を再検証する。
- 日本列島オーバーレイは Natural Earth 10m 由来の約1km目標高密度輪郭。地表面・海面は地下スプラットを隠すため描画しない。
- `array_projection_splats.html` に `depth diagnostics` 色モードを追加済み。amber は整数kmカタログ深さの直接波、red は遅延探索窓上限クリップ、cyan は遅延相モデル由来の深さを表す。現在のスプラット出力では direct と red も診断情報として保持し、構造密度には `structure_amplitude=0` として寄与させない。
- プレート境界・沈み込み面オーバーレイは `schematic_japan_plate_context_v0` の概略線・ワイヤーフレームであり、Slab2等の定量的プレートモデルではない。太平洋プレート、フィリピン海プレートの文脈表示として扱う。

## 性能・H200/5090準備

- CPU前処理はDuckDBで一部効率化しているが、array projection のイベント処理はまだ並列度が不足している。
- GPU投入前に、CPU側で波形からスペクトル、群遅延、投影候補、圧縮スプラット、LOD、シャードを作り切る必要がある。
- 次の性能改善は、イベント単位の ProcessPoolExecutor 化、DuckDB PRAGMA threads の明示、進捗ログ、メモリ上限、途中再開可能なシャード出力。
- GPU側は、WebGL表示とH200/5090計算を分ける。H200/5090では最終Web表示ではなく、多数投影の融合、3D密度場、異常領域抽出、バックテスト評価を優先する。

## セキュリティ・運用

- Hi-net等の認証情報は `secrets/` またはローカルテンプレートから扱い、Gitには入れない。
- ランタイム設定 `_runtime_*.yml` は作業用であり、公開設定に混ぜない。
- `/mnt/slam/equake` へは成果物だけを同期し、コード本体はGitHubにpushする運用を維持する。

## 次に処理する事項

- 高密度WebGL成果物を `/mnt/slam/equake/outputs/3d/` に同期済み。
- WebGL高密度化、白黒強度デフォルト、分類色オーバーレイ、スマホ操作対応、概略プレート構造オーバーレイをGitHubへpush済み。
- `array_projection` のCPU並列化とシャード化を実装する。
- Hi-net取得済み範囲と未取得範囲をDBで明示する取得台帳を作る。
- 実DEM・高解像度海岸線・プレート境界を追加し、地図オーバーレイを実データ化する。
- 100年シナリオHTMLの時間フレーム更新ロジックを再点検する。

## 2026-06-30 11:40 UTC - Plate overlay provenance

- Current WebGL plate/slab overlay is schematic hand-built context, not a literature-calibrated slab or plate-boundary dataset.
- It appeared spatially shifted and must not be used for analytical comparison with splats, faults, or stress.
- Mitigation implemented: schematic overlay default hidden; metadata marks `tectonic_overlay_literature_based=false`.
- Required v1 fix: import a quantitative published/open dataset such as Slab2/plate-boundary polylines and preserve citation/provenance in output metadata.

## 2026-06-30 12:05 UTC - Hi-net waveform units and orientation

- Current Hi-net/FDSN combined spectra should be treated as relative spectra, not calibrated physical ground motion.
- Existing merged CSV lacks `calibration_applied`, `physical_unit`, `cmpaz_deg`, and `cmpinc_deg`; regenerated sidecars now record `physical_calibration_applied=false` and `station_orientation_columns_present=false`.
- Hi-net collector now preserves SAC `calib`, `scale`, `cmpaz`, `cmpinc`, station elevation/depth when available, but does not apply official response removal or component rotation yet.
- Required v1 fix: load authoritative NIED channel/response metadata, apply AD-count-to-physical-unit conversion per station/component/time, validate against SAC PZ/StationXML, and rotate components using installation orientation before absolute-amplitude analyses.

## 2026-06-30 Hi-net Small-Event Ingest

- USGS ComCat nationwide M2 query is not a sufficient small-earthquake source around Japan: the 2000-2026 M2/depth700 pull returned 40,494 events but only 3 rows in M2-3. Hi-net/JMA authenticated catalogs are required for dense small-event work.
- Hi-net 0101 all-station continuous waveform download failed as one request. The working path is to build station groups from the authenticated station CSV; East Japan currently resolves to 563 Hi-net stations split into 8 groups of about 80 stations each.
- Hi-net WIN32 conversion tools (`catwin32`, `win2sac_32`) must exist under the workspace, e.g. `.deps/hinet-win32tools/bin`; missing tools caused downloads to fail after request preparation.
- HinetPy/win2sac SAC output is treated as sensitivity-removed and scaled by 1e9. Do not multiply SAC data again by `calib * scale`; keep orientation metadata and defer component rotation.
- USGS mechanism-detail collection may stall on public API detail requests. Use append-only CSV, smaller worker counts, and resume rather than discarding partial mechanism rows.
