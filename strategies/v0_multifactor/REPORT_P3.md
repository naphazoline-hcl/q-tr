# REPORT_P3.md: P3 学習と推論の検収メモ

## 1. 受け入れチェック（§5）

| 項目 | 結果 | 根拠 |
|---|---|---|
| `train_v2.py` が `--resume` 対応・進捗ファイル書き出しつき | OK | `work/progress/train_v2.json`（JSONL）と `STATE.md`。合成データで、削除したモデルと途中で切れたモデルだけ再学習され、残りは読み飛ばされた |
| `meta_v2.json` の `features` が学習時と推論時の列順序と一致する | OK | `features` = `fit_model` が返す列順（照合つき）。推論では各 Booster の列名・列順序と照合し、2 列入れ替えた meta では停止した |
| `submission.py` が 30 分以内・低メモリで終わる実装 | 設計 OK・実測 {要実行} | 下の §2 の対策。実データでの時間・ピークメモリは未計測 |
| 予測が採点対象 index を完全に覆う設計 | OK | raw_return ∪ beta の index に reindex。beta にだけある行を含む合成データで `score.py` の align を通過（不足 0 行） |
| 予測コードに `target` / `raw_target` の読み込みが無い | OK | `check_lookahead.py --runtime`（ERROR 0・runtime guard OK）と `score.py --guard` を通過 |
| `TRAINING.md` にコマンド・所要時間・再開方法を書いた | OK | 行数 × 本数の見積もりとサンドボックス実測（参考値） |

## 2. submission.py のメモリ・時間対策

- 特徴量は `alpha_v2.build_features(splits=("train", "valid"), start=2014-06-01)`。P1 の `alpha_features` が
  pyarrow の `columns=[...]` と `filters=[("Date", ">=", start)]` で読み込み時に絞り、`listed_info` は 5 列だけ読む
- ブロック合成のあと、モデル列（51 列）だけを float32 の C 連続行列 1 つに列ごとに順位化して書き込み、生の特徴量パネルは即解放
- `lgb.Booster(model_file=...)` で読み、meta の列順の DataFrame と列名を照合してから、裏の行列をコピーせず `predict` に渡す
- 最後に銘柄ごとの EWMA（`smoothing_span` = 5）。数式は `alpha_v2.predict_signal` + walk-forward の平滑化と同一

## 3. サンドボックスでの検証（実データなし）

- 環境: Python 3.11 + pandas 3.0.3 / numpy 2.4.6 / pyarrow 24.0.0 / lightgbm 4.1.0 / scikit-learn 1.9.0（採点環境と同じ版）。
  Python 3.13 + pandas 3.0.6 でも同じ結果
- データ: 同じファイル名・列名の合成データ（`alpha_features` は同じ API・列名で値だけ合成のスタブ。`alpha_v2.py`・
  `walkforward_config.json`・`tools/*.py`・`evaluate_script.py` は q-tr の実物）
- 実行したコマンド: `train_v2.py`（6 本学習）→ `score.py --split train` → `score.py --split valid --guard` →
  `check_lookahead.py --runtime`。すべてエラーなく完走。合成データなのでスコアの数値には意味がなく、記載しない
- 同値性: 保存した 6 本を `alpha_v2.predict_signal` に渡した結果（+ walk-forward と同じ EWMA）と `submission.predict()` の
  差の最大値は 0.0、NaN の位置も一致
- 学習時間の参考: 乱数データ 753,294 行 × 51 列で学習 23.9 秒/本、順位化 12.3 秒（2 vCPU）

## 4. Assumptions（前提と不明点）

1. **`score.py --split train` 対応**: 公式の `align_prediction` は target の全行を覆うことを要求する。既定（公式採点）は
   Valid のみを覆い、`--split train` を argv で検出したとき（または `V0_PREDICT_SPLITS=train,valid`）だけ Train も予測する。
   Train は学習期間と重なる in-sample なので、スコアは楽観的（パイプライン確認用）
2. **採点対象 index**: target を読まずに再現するため raw_return ∪ beta を使う（`docs/data_schema.md` では target_1day_valid の
   1,227,148 行はすべて raw_return と beta にある）。特徴量の無い行は NaN（採点では 0 = 中立）
3. **学習行の定義**: walk-forward（+2.186 を出した手順）と同じく「どちらかのラベルがある行」を `fit_model` に渡す。
   モデル用の断面順位もその行集合で計算される。`training_rows` / `training_from` / `training_to` はこの行集合の値
4. **1 本ずつ学習**: `fit_model` を (k, seed) ごとに `seeds=[seed]`・単一ラベル列で呼ぶ。同じデータ・同じ `random_state` なので
   一括呼び出しと同じモデルになり、1 本ごとの保存と再開ができる。代わりに順位化を毎回やり直す（参考 12 秒/回）
5. **推論時の設定**: `walkforward_config.json` ではなく、`meta_v2.json` に保存した学習時の解決済み `params` と `blocks` を使う。
   config を後で変えても、再学習するまで提出物は変わらない
6. **in-place 順位化**: pandas 3 の Copy-on-Write では特徴量 DataFrame 自体を安全に上書きできないため、事前確保した float32
   行列へ列ごとに書き込む方式にした（float64 の全体コピーや順位化済み DataFrame の複製は作らない）
7. **ファイル名**: 既存の `strategies/v0_multifactor/REPORT.md`（S1）を ZIP 展開で上書きしないよう `REPORT_P3.md` とした
8. **progress.py の CLI**: docstring には `progress.py show <path>` とあるが、実装は `<path>` だけを受け付けるため、
   TRAINING.md では `python tools/progress.py work/progress/train_v2.json` と書いた
9. **ラベル日数不足**: `min_label_days` 未満のホライズンは学習せず `meta_v2.json` の `skipped` に記録する。推論は残りのモデル
   （無ければブロック合成のみ）で動き、`alpha_v2.predict_signal` と同じ扱いになる。最終学習では発生しない見込み
