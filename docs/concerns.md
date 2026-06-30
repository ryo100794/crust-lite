# crust-lite 懸案事項ログ

更新日時: 2026-06-30 01:58 UTC

このファイルは、H200/5090での本計算前に潰すべき懸案、検証待ち、既知の簡略化を継続的に記録するためのログです。ここに書く数値は作業時点の状態であり、研究用の相対評価プロトタイプの状態記録です。地震の発生日・場所・規模を断定的に予測するものではありません。

## 現在の確認済み状態

- 高密度WebGLスプラットHTMLを再生成済み。
- `array_projection_splats.html` は Plotly ではなく WebGL2 point sprite 方式。
- 表示スプラット数: `113,504`
- 地形面描画: `disabled_surface_outline_only`。地下スプラットを隠さないため、日本列島は輪郭のみ表示。
- 日本列島輪郭頂点数: `13,773`
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

## 可視化

- WebGL版はPlotly版より軽く、現時点ではさらに高密度化可能。
- スマホ操作は実装済みだが、実機でのドラッグ、ピンチ、視点保持、Canvasサイズ、DPR=4時のメモリ使用を確認する必要がある。
- `events_faults_timeseries.html` / `stress_timeseries_3d.html` / `failure_scenarios_3d.html` は、フレーム変更時の視点維持、古い描画状態の残留、スライダー時間変化の実質的な見え方を再検証する。
- 日本列島・地形オーバーレイは現状「表示用の高密度輪郭 + 合成地形コンテキスト」。実DEM、海岸線、行政境界、プレート境界を使った本格オーバーレイは未完了。

## 性能・H200/5090準備

- CPU前処理はDuckDB中心。array projection はイベント単位で2 worker並列化済みだが、Podの実効割当は2CPU。
- GPU投入前に、CPU側で波形からスペクトル、群遅延、投影候補、圧縮スプラット、LOD、シャードを作り切る必要がある。
- 次の性能改善は、balanced partition、DuckDB PRAGMA threads の明示、進捗ログ、メモリ上限、GPUミニバッチローダ。
- GPU側は、WebGL表示とH200/5090計算を分ける。H200/5090では最終Web表示ではなく、多数投影の融合、3D密度場、異常領域抽出、バックテスト評価を優先する。

## セキュリティ・運用

- Hi-net等の認証情報は `secrets/` またはローカルテンプレートから扱い、Gitには入れない。
- ランタイム設定 `_runtime_*.yml` は作業用であり、公開設定に混ぜない。
- `/mnt/slam/equake` へは成果物だけを同期し、コード本体はGitHubにpushする運用を維持する。

## 次に処理する事項

- 高密度WebGL成果物を `/mnt/slam/equake/outputs/3d/` に同期する。
- WebGL高密度化とスマホ操作対応をGitHubへコミット・pushする。
- `array_projection` のCPU並列化とシャード化を実装する。
- Hi-net取得済み範囲と未取得範囲をDBで明示する取得台帳を作る。
- 実DEM・高解像度海岸線・プレート境界を追加し、地図オーバーレイを実データ化する。
- 100年シナリオHTMLの時間フレーム更新ロジックを再点検する。


## 2026-06-30 CPU pre-GPU実行結果

- RunPodコンテナの実効CPU割当は `cpuset.cpus.effective=84,180`、`nproc=2`。ホストは多数コアに見えても、このPodでは2CPU制限が実効上限。
- `array_projection` はイベント単位の `ProcessPoolExecutor` 化済み。取得済みスペクトルから `113,504` array projection / Gaussian splat primitive を生成。
- WebGLスプラットは地表面メッシュを描かず、日本列島は高解像度輪郭のみ表示する方式へ変更。地下スプラットを地形面で隠さないため。
- 3D表示用WebGL成果物は `113,504` splats、outline vertices `13,773`、point sprite上限 `384 px`。これは研究用状態表示であり、地震発生日・場所・規模の断定ではない。
- CPU pre-GPU `gpu-prep` radius12: `601,854,017` view-image pixel rows、`271,292` views、`128` parts、出力約 `10GB`。
- CPU pre-GPU `gpu-prep` radius20: `1,264,851,772` view-image pixel rows、`271,296` views、`256` parts、出力約 `20GB`。
- radius20実測時間: `real 4171.731s`, `user 2815.822s`, `sys 209.066s`。2CPU制限下で約69.5分のCPU前処理。
- radius20最大part: `part_id=84`, `23,366,551` pixel rows。次はevent hashではなく、推定pixel行数で均衡化するbalanced partitionが必要。
- 最新GPU handoffは `outputs/gpu_prep/manifest.json` に記録。`view_image_partitions` は `data/processed/splat_view_image_parts_r20_p2500_s256/part-*.parquet` を指す。
- 現在のGPU前処理入力は、観測点別の `xy` と `range_depth` の2種類の投影画像を含む。これは単なる震源位置の重ね描きではない。
- H200/5090側では、全part一括ロードではなく、manifestのpart indexを読んでバッチ単位でGPUへ投入する。

## 次のGPU前の実装課題

- `splat_view_image_part_index` を使ったGPUミニバッチローダを実装する。
- balanced partition: event/station/frequencyごとの推定展開行数を事前計算し、partごとのpixel行数を平準化する。
- view-imageから3D Gaussian parameterへ変換する学習/最適化入力仕様を固定する。最低限 `view_id`, `event_id`, `station_id`, `frequency_hz`, `image_plane`, `pixel_u`, `pixel_v`, `intensity`, `phase_alignment_mean`, `group_delay_s_mean` を使う。
- GPUランでは、反射・散乱・残差成分を別channelまたは別loss重みで扱い、震源からの直接波だけに閉じない評価にする。
- 生成したGaussian密度場と既知断層・プレート境界・GNSS歪み・過去地震活動の相関評価を、バックテスト用に分離して実装する。
