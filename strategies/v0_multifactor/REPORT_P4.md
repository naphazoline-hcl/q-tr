# REPORT — P4 診断と改善1（improve1）

## 同梱物（`improve1_package.zip`）

| パス（リポジトリ直下からの相対） | 種別 |
|---|---|
| `strategies/v0_multifactor/alpha_v2.py` | 更新（model_smoothing_span / label_window / add_model_terms） |
| `strategies/v0_multifactor/submission.py` | 更新（モデル項を alpha_v2.add_model_terms に統一） |
| `strategies/v0_multifactor/walkforward_config.json` | 更新（K1 = I1 + model_smoothing_span 10） |
| `tools/sweep_improve1.py` | 新規（①②③の切り分け。ローカル評価専用） |
| `CHANGELOG_v2v3.md` / `REPORT.md` | 変更点・根拠 / 本書 |

## ローカルでの検収手順

```bash
python strategies/v0_multifactor/train_v2.py --strategy strategies/v0_multifactor   # meta_v2.json を作り直す
python tools/walkforward.py --strategy strategies/v0_multifactor --module alpha_v2 --json work/reports/improve1.json
python tools/score.py --submission strategies/v0_multifactor --split train
python tools/score.py --submission strategies/v0_multifactor --split valid
python tools/check_lookahead.py --submission strategies/v0_multifactor --runtime
python tools/check_truncation.py --strategy strategies/v0_multifactor --module alpha_v2 --cut 2014-06-30
python tools/sweep_improve1.py --strategy strategies/v0_multifactor --module alpha_v2   # ①②③（再開可）
```

`sweep_improve1.py` は strict / partial の 2 種類のラベルで学習する（各 6 fold と Train 全期間）。
学習時間は walkforward 2 回分と train_v2 2 回分に相当する見込みで、実測は `{要実行}`。
途中で止まっても保存済みの成分は読み飛ばす。結果は `work/reports/sweep_improve1.md` / `.json`。

## この環境で実行した確認（入力データなし・合成データ）

| 確認 | 結果 |
|---|---|
| `py_compile`（alpha_v2.py / submission.py / sweep_improve1.py） | OK |
| `check_lookahead.py`（静的検査のみ。`--runtime` はデータが無く未実行） | OK（ERROR パターンなし、8 ファイル） |
| model_smoothing_span=1 の predict_signal が元の alpha_v2（I1）と一致 | OK（合成パネル 260 日 × 60 銘柄、atol 1e-6） |
| model_smoothing_span=10 が因果的（未来の日付を切っても過去行が不変） | OK |
| label_window=strict が元の make_label と一致 / partial が v1 の式と一致 | OK |
| sweep の再合成が predict_signal + EWMA(5) と一致（span 1 と 10） | OK（atol 1e-5） |
| 実データでの Train OOS / Valid / 回転率 / runtime lookahead / truncation | **未実行 `{要実行}`** |

## 受け入れチェックリスト（§4）

- [x] 変更点を箇条書きで記録した（`CHANGELOG_v2v3.md`）
- [x] API 契約・設定キーを維持した（4 関数のシグネチャは不変。`make_label` に省略可能な `window`、
      params に省略可能な 2 キーを追加しただけ。既存キーの削除・改名なし。未指定なら I1 と同一）
- [x] 禁止パターンが無い（静的検査 OK。EWMA は過去方向のみ、順位・z は同一 Date 内のみ。
      逆順 rolling は従来どおり make_label だけで許可マーカー付き）
- [x] 変更の根拠を【実測結果】の数字に結びつけた（CHANGELOG §2: コスト 0.544% の再計算、
      グロス 3.47% vs 3.42%、A→I1 の回転率 0.0182→0.0216、分位からのグロス再構成、年別差 0.12〜0.21）
- [x] 25 分で終わる独立したファイル単位で分割した（4 ファイル + 文書 2。各ファイル単独で差し替え可能）
- [ ] 改善効果の実測（Train OOS +2.0 以上・Valid +0.754 超え・回転率 0.017 以下）→ `{要実行}`

## Assumptions

1. 入力データ（`input/`）はこの環境に無い。改善効果は算術と合成データの検証だけで、実測値は書いていない。
2. 「キー構成の維持」は既存キーの削除・改名をしないことと解釈し、省略可能な params キー
   （`model_smoothing_span` / `label_window`）の追加は許容とした。未指定時は I1 と同一の挙動。
3. 「strict ラベル」は make_label の「窓 t..t+k-1 が end を超える行は NaN」の規則、比較対象は v1 の
   alpha.make_label（min_periods=1・打ち切り判定なし）と解釈して `partial` として再現した。
4. 「A（中立化 OFF）」は I1 から `sector_neutral` だけを false にした構成と解釈した（sweep の `A_like`）。
   `v1_like` はモデル特徴量・ハイパーパラメータが v2 基盤のままなので、v1 の厳密な再現ではない。
5. 既定の span 10 は、ラベル窓に対する平均遅れ（4.5 / 126 日）から選んだ。最良値はスイープの実測で決める。
6. submission.py は meta_v2.json の params を使うため、設定変更後は train_v2.py の再実行が前提。

## 次にやること

1. 上の検収手順を実行し、`sweep_improve1.md` の T / V / TO がすべて OK の候補を選ぶ（Train OOS 優先）。
2. ②: 業種間／業種内分解で、2016・2017・2019・2025 の I1 の負けが業種間グロスから来ているかを確認する。
   業種間なら `no_sector_rank_blocks` 系、業種内なら単独成分の年別 Sharpe で崩れたブロックを見直す。
3. ③: 2×2 の z / strict 寄与が Train OOS と Valid で符号が逆なら、その設定を v1 側（mean / partial）に戻す候補にする。
