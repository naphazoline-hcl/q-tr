# CHANGELOG v3 → v4 候補（P5 改善ラウンド2: アンサンブル・低速化・局面配分）

土台は K1（S4 採用: I1 + `params.model_smoothing_span: 10`）。**この環境には入力データが無いため、
実データでの改善効果は 1 つも実測していない。** 実測値は `{要実行}`。ここに書く数字は
S4 の実測値・その算術、または合成データでの動作確認・時間計測（実データの性能ではない）だけ。

## 1. 変更点

| ファイル | 変更 | 既定値 / K1 との関係 |
|---|---|---|
| `ensemble.py`（新規） | **A**: Ridge（ブロック z を説明変数にした全期間プールのパネル回帰。係数はホライズンごとに 1 本＝時系列方向に学習）と LightGBM rank モデル（`lambdarank` / `rank_xendcg` / `quantile`）。各予測を**同日断面 z にしてから** `ensemble_weights` で加重合成 | lgbm だけが非ゼロなら合成をせず素通し（K1 とビット一致） |
| `regime_v4.py`（新規） | **B**: マクロ2変数（消費者態度指数の水準 z と 63 日平均変化、USD/JPY 60 日変化）＋市場2変数（`topix_cum60`、`topix_vol60`）の因果ルールで risk_off / neutral / expansion。新局面は 5 日連続で確定（debounce）、所属度を EWMA(20) で混ぜてブロック重みを滑らかに切替 | `regime_mix: false`（既定 OFF） |
| `slowdown.py`（新規） | **C**: 採点と同じ重み・回転率をシグナルだけから計算する `quintile_turnover`（ラベル不使用）、span 5 / 10 / 20 とラベル設計のセット `SLOW_PROFILES`、cap を満たす最速 span を選ぶ `choose_span` | `turnover_control: "off"`、`slow_profile: null` |
| `alpha_v2.py` | 新キー `ensemble_weights` / `ridge` / `rank_model` / `regime_mode` / `regime_v4` / `regime_weights` / `turnover_cap` / `turnover_control` / `turnover_window` / `slow_profile` を `DEFAULT_PARAMS` に追加。`fit_extras()`（Ridge・rank・auto span）を `fit_model` の最後で呼ぶ。`predict_signal` を `block_part` / `signal_parts` / `add_model_terms` に分割（`submission.py` と共有） | 旧キーは削除・改名なし。API 4 関数のシグネチャ不変 |
| `submission.py` | ブロック部分を `alpha_v2.block_part` に置換。Ridge 係数（meta の JSON）と `models_v2/rank_*.txt` を読み、`ensemble.combine` で合成。重み 0 のモデルは読まない | 旧 meta（`regime_mode` 無し）は `regime_mix` を P2 の 2 局面として扱う |
| `train_v2.py` | LGBM の後に `fit_extras` を 1 回だけ実行し、Ridge 係数・rank モデル・`regime_v4` しきい値・`turnover_estimates` を meta に保存。auto で選んだ span を `params.model_smoothing_span` に書き戻す | `--resume` は LGBM を読み直して extras を作り直す |
| `walkforward_config.json` | 新キーを全部明示。**候補 E1: `ensemble_weights.ridge: 0.3`**（rank 0、局面 OFF、auto OFF、`turnover_cap: 0.017` 宣言） | K1 に戻すには `ridge: 0.0` だけ |
| `tools/sweep_improve2.py`（新規） | fold ごとに 1 回だけ学習（LGBM + Ridge [+ rank: `--rank`]）し、成分から 18 候補を Train OOS と Valid で評価（T / V / TO 判定つき） | 提出物ではない（ローカル評価専用） |
| `selftest_p5.py` / `bench_p5.py`（新規） | 合成データでの切替・因果性・K1 一致の検査と、学習・推論時間の計測 | 実データ不要 |

## 2. 根拠（S4 の実測結果から）

### ① 回転率の余地: span だけでは足りない → A でモデル項そのものを遅くする

- K1 の Valid 回転率 0.0161 → コスト `0.1% × 0.0161 × 252 = 0.406%/年`（実測 0.41% と一致）。cap 0.017 まで
  の余地は `0.0009 × 0.252 = 0.023%/年` しかない。**回転率を増やさずにグロスを上げる**必要がある。
- K1 の vol は `(3.39 − 0.41) / 0.7591 ≈ 3.93%/年`。Valid +1.2 には同じ vol で年 `1.2 × 3.93 + 0.41 ≈ 5.1%` の
  グロス（K1 の約 1.5 倍）が要る。span や重みの微調整だけでは届かない水準なので、モデルの多様化（A）を優先した。
- Ridge の説明変数は 60 日窓・財務のブロック z だけで、モデル入力の 1〜10 日窓の列（r1, rev1, rcc3/5/10 など）を
  含まない。**Ridge 予測は構造的に遅い**ので、モデル項に混ぜると「同じモデル重みのまま回転率を下げる」方向に効く
  （LGBM の比率が 1.0 → 1/1.3 に下がる）。回転率がいくつになるかは `{要実行}`。

### ② size ブロック単独 +0.847・neutral_only +0.814・mw025 +0.773 が回転率 0.0198〜0.0207 で不採用

- 3 つとも「LGBM の非中立項を減らす / size に寄せる」と Valid が上がり、回転率が上がる。Ridge は過去全期間の
  パネルで**ブロックの時系列上の稼ぎ**に係数を付けるので、size が強ければ size に重く乗る（合成データでも
  size に最大係数を付けることを確認）。ブロック重みを手で size に寄せる（Valid の結果を見た調整）より、
  Train だけで学習した係数で寄せる方が過適合しにくい。
- 回転率 0.02 前後の候補は、モデル項の span を 10 → 20 にすると平均遅れが 4.5 → 9.5 営業日になり
  （k126 窓の 7.5%）、回転の主因であるモデル項の入れ替わりが減る。`sweep_improve2` に
  `neutral_only_span20` / `mw025_span20` / `size_heavy_ridge03_span20` を入れて、**TO ≤ 0.017 を満たしたまま
  Valid を保てるか**を実測で確かめる（「本丸」の検証）。`{要実行}`

### ③ 2020 −0.40 / 2026 −0.22（B は既定 OFF・実装のみ）

- 2020 はコロナ急落（TOPIX ボラ急上昇・消費者態度指数の急低下）、2026 は年初からの短い区間。どちらもブロックの
  重みが局面に合っていない仮説と整合するので、B の risk_off で lowrisk 0.3 → 0.6・quality 0.5 → 0.7 に上げる
  既定重みを置いた（**未検証の初期値**）。局面判定は同日値と学習期間のしきい値だけ、切替は debounce + EWMA
  で回転率の跳ねを抑える。`regime_v4` / `regime_v4_ridge03` をスイープで実測し、Train OOS が悪化するなら採らない。

### ④ rank モデル（既定 OFF）

- 採点は 5 分位なので順位損失の寄与は検証価値があるが、Train OOS の改善が strict ラベル由来で horizon z の
  寄与がほぼ 0 だった（S4）ことから、**ラベルの形を変えるだけでは伸びしろが小さい**可能性がある。既定は OFF にし、
  `--rank` 付きのスイープで `rank03` / `ridge03_rank03` を実測する。lambdarank は上位 30 位だけを見る NDCG なので、
  ショート側も効かせるため反転ラベルでもう 1 本学習し `z(long) − z(short)` を使う（`two_sided`）。

## 3. 実測結果（ローカル検収済み・2026-10-04）

出荷時の既定 E1（ridge 0.3）は実データでは Valid が K1 未満だったため不採用。成分保存スイープ
（`tools/sweep_improve2.py`、25 候補）で **mw025_ridge06_span20** を採用した。

| 指標 | K1（S4 実測） | E1（ridge 0.3、出荷時既定） | **採用: mw025_ridge06_span20** |
|---|---|---|---|
| Train OOS Sharpe | +2.0391 | +2.0688 | **+2.1240** |
| Valid Sharpe | +0.7591 | +0.7470 | **+0.8030** |
| Valid 回転率 | 0.0161 | 0.0145 | **0.0133**（コスト 0.33%/年） |
| Valid 2020 / 2026 | −0.40 / −0.22 | −0.13 / −0.17 | **−0.39 / −0.14** |
| Valid グロス | +3.39% | — | **+3.27%** |

- 採用ルール（T: Train OOS > +2.0391 / V: Valid > +0.7591 / TO: Valid 回転率 ≤ 0.017）をすべて満たす。
- 実測 JSON: `work/reports/improve2.json`（E1 walkforward）/ `improve2_final.json`（採用 walkforward）/
  `improve2_final_valid.json`（採用 score valid）/ `sweep_improve2.json` + `.md`（全候補）。
- 参考: ridge10（ridge 1.0・他は K1 のまま）も +2.109 / +0.794 / 0.0132 で合格。単一変更では最良。
  mw025_ridge07_span20（+2.110 / +0.804）・mw02_ridge06_span20（+2.086 / +0.806）は同水準。
  rank モデル（`--rank`）・slow_profile のラベル設計・局面配分の再調整は未実測（`{要実行}`）。
- 検証: py_compile / 静的 lookahead / selftest_p5（11 ケース・因果性）/ walkforward / train_v2 /
  score valid（--guard）/ lookahead --runtime / truncation すべて OK。runtime 予測 shape=(1227148, 1)。

## 4. 採用ルールと戻し方

- 採用は `work/reports/sweep_improve2.md` の T / V / TO がすべて OK の候補だけ。**Valid だけ良い候補は採らない**。
- K1 に戻すには `walkforward_config.json` の `"ensemble_weights": {"lgbm": 1.0, "ridge": 0.0, "rank": 0.0}` だけ。
- 予測側の設定を変えたら **`train_v2.py` を再実行して meta_v2.json を作り直す**（submission は meta の params を読む）。
- 提出 zip には `ensemble.py` / `regime_v4.py` / `slowdown.py` を同梱する（`alpha_v2.py` が import する）。
