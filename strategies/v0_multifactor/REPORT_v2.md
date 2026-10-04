# REPORT — P2 戦略本体 v2 の受け入れチェック

実データはこの環境に無いので、性能（Sharpe・実行時間）はすべて `{要実行}`。以下は、コード検査と
**合成データ**（60銘柄×約1,300営業日、ALL_COLUMNS 94列。性能の参考にはならない）での機能確認の結果。

## 受け入れチェックリスト（§6）

- [x] **§1 の4関数があり、`tools/walkforward.py` が import できる**: リポジトリの walkforward.py を
  取得し、そのまま合成データで実行した（`--module` で alpha_v2 を指定、fold 2010〜2013）。全フォールドが完走した。
  fold 2010 はラベル 0 行なので、ブロック合成だけで予測し、エラーは出ない。
- [x] **`make_label` は Valid のラベルを読まない**: ファイル参照は `LABEL_FILE = "target_1day_train.parquet"`
  の1箇所だけ。`make_label` の中に `valid` という文字列は無い（ファイル全体でも、特徴量名
  `valid_ratio20` と `np.errstate(invalid=...)` だけ）。
- [x] **params で切替できる**: seeds / block_weights / model_weight / sector_weight / label_clip /
  sector_neutral / sector_rank_blocks / regime_mix / block_set / model_features / label_transform /
  horizon_combine / fill_missing_blocks。合成データで11通りの切替がすべて動き、出力の index が入力と一致した。
- [x] **MODEL_DESIGN.md に設計判断の理由を記録した**（ラベル・列の取捨・中立化・重み・局面・計算量・アブレーション案）。
- [x] **禁止パターンが無い**: `tools/check_lookahead.py` の `scan_file` を alpha_v2.py にかけて、指摘は 0 件。
  逆順 rolling とラベルファイル参照の行には `# check_lookahead: train-ok` を付けた。フォルダ全体の
  `python tools/check_lookahead.py --submission strategies/v0_multifactor` はローカルで実行する（`{要実行}`）。
- [x] **1フォールド 5 分以内の見積もり**: MODEL_DESIGN.md §7。2 vCPU の合成データ計測で、
  LightGBM 1本（45万行×65列・350木）18.7秒 → 6本と前処理で約2〜2.5分/フォールド。

## 合成データでの機能確認（25項目すべて PASS）

- inf → NaN（合成で入れた 1,184 個の ±inf を全部置換。件数は `attrs["inf_replaced"]` に残る）
- `make_label`: 窓が end を超える最初の日から NaN。手計算の前方平均と一致。clip が効く。
  上場廃止銘柄で窓の観測が半分未満の行は NaN
- `fit_model`: 6本を学習。REGIME / FAST 列は入力から外れ、業種列は素通し。ラベルが無ければ空モデルで
  ブロックだけを返す
- **因果性**: 予測期間の先頭41日だけで作ったシグナルが、全期間で作ったシグナルの同じ行と完全一致（差 0）
- 業種中心化後の (Date, sector33) 平均は ≈0（1.5e-16）。業種内順位は、業種コードの欠損と小業種で NaN

## 未実施（`{要実行}`）と次にやること

1. 実データの walk-forward: `python tools/walkforward.py --strategy strategies/v0_multifactor --module alpha_v2 --json work/reports/wf_v2_alpha.json`
2. MODEL_DESIGN.md §8 のアブレーション A〜H。v1（+1.993）を上回る構成を選ぶ
3. 選んだ構成で `train.py` / `submission.py` を v2 に接続し、Valid を最終確認する（P3 以降）
