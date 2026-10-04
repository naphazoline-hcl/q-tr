# TRAINING.md: v0_multifactor v2 の学習手順（P3）

## 0. 前提

- Python 3.11 / pandas 3.0.3 / numpy 2.4.6 / pyarrow 24.0.0 / lightgbm 4.1.0 / scikit-learn 1.9.0（採点環境と同じ）
- `input/` に Train の配布 parquet（ラベル用の `target_1day_train.parquet` を含む）
- `strategies/v0_multifactor/` に `alpha_v2.py` / `alpha_features.py` / `walkforward_config.json`（S2 採用構成・push 済み）

## 1. 配置

`train_submit_package.zip` をリポジトリのルートで展開すると `strategies/v0_multifactor/` に
`train_v2.py` / `submission.py`（v1 を置き換え。v1 は git 履歴に残る）/ `TRAINING.md` / `REPORT_P3.md` が入る。

```bash
python -m py_compile strategies/v0_multifactor/train_v2.py strategies/v0_multifactor/submission.py
```

## 2. 学習

```bash
python strategies/v0_multifactor/train_v2.py --strategy strategies/v0_multifactor
python tools/progress.py work/progress/train_v2.json   # 別ターミナルで進捗確認（STATE.md も同じ場所）
```

- 出力: `models_v2/lgbm_k126_s{0,1,2}.txt`・`lgbm_k250_s{0,1,2}.txt`（6 本。1 本終わるごとに保存）と、最後に `meta_v2.json`
- 設定はすべて `walkforward_config.json` から読む（`horizons` / `params.seeds` / `params` / `history_start` / `smoothing_span`）

## 3. 中断からの再開

```bash
python strategies/v0_multifactor/train_v2.py --strategy strategies/v0_multifactor --resume
```

- 保存済みで、読み込めて、列名・列順序が今回と一致するモデルは読み飛ばす。保存は `*.tmp` → 置換なので、書きかけの本体は残らない
- 特徴量構築とラベル作成は再開時も毎回やり直す
- `meta_v2.json` は全ジョブの後に書く。無ければ未完了なので `--resume` で再実行する
- `walkforward_config.json` のハイパーパラメータを変えたら `--resume` を付けずに全部学習し直す（列の変化は自動検出するが、パラメータの変化は検出しない）

## 4. 採点・検査

```bash
python tools/score.py --submission strategies/v0_multifactor --split train          # in-sample（学習期間と重なるので楽観的）
python tools/score.py --submission strategies/v0_multifactor --split valid --guard
python tools/check_lookahead.py --submission strategies/v0_multifactor --runtime
```

- `submission.py` は既定（公式採点）では Valid だけを覆う。`--split train` で呼ばれたとき（argv で検出）か
  `V0_PREDICT_SPLITS=train,valid` のときだけ Train も覆う

## 5. 所要時間の目安（行数 × 本数）

| 項目 | 値 | 根拠 |
|---|---|---|
| Train 行数 | 809,636 行 / 1,814 営業日（約 446 行/日） | `docs/data_schema.md` |
| k126 の学習行（strict） | 約 75 万行（見積もり） | 446 × (1,814 − 125)。`label_min_frac` の分だけ少し減る |
| k250 の学習行（strict） | 約 70 万行（見積もり） | 446 × (1,814 − 249) |
| 学習本数 | 6 本（2 ホライズン × 3 seed） | `walkforward_config.json` |
| 1 本あたり（手元） | {要実行} | `work/progress/train_v2.json` の `seconds` に記録される |
| 1 本あたり（参考） | 約 36 秒 = 学習 23.9 秒 + `fit_model` 内の順位化 12.3 秒 | v0 サンドボックス（2 vCPU・lightgbm 4.1.0）で乱数データ 753,294 行 × 51 列、同じ `model_params` の実測 |
| 特徴量構築 | {要実行}（P1 の build は 609,960 行で 28.7 秒 → 約 81 万行なら 40 秒前後の見積もり） | `v0/実行ログ.md` |
| 合計 | 特徴量 + ラベル + 6 ×（順位化 + 学習）= {要実行}（参考値では 6 × 36 秒 ≒ 3.6 分 + 特徴量・ラベル） | |

- 学習行数は walk-forward の fold 2015（S2 実測 約 49.1 万行）の約 1.5 倍。fold 2015 の 1 本あたり時間 × 1.5 が目安になる
- メモリの目安: 特徴量（約 75 万行 × 94 列 × float32 ≒ 0.3 GB）＋ 順位化行列（51 列で約 0.15 GB）＋ LightGBM の作業領域

## 6. この構成で学習する根拠（S2 実測）

- Train OOS Sharpe: v1 +1.993 → 採用構成 **+2.186**（2010 +0.89 / 2011 +2.21 / 2012 +0.89 / 2013 +2.68 / 2014 +2.35 / 2015 +4.97）
- グロス / コスト / 回転率: +8.86% / 0.57% / 0.0227（v1 は +8.52% / 0.81% / 0.0320）
- 採用値: `model_features="v1"` / `block_set="v1"` / `sector_rank_blocks=[]` / `sector_neutral=false` /
  `label_transform="rank"` / `horizon_combine="z"`
- strict ラベルのため walk-forward の fold 2010 は学習 0 行（ブロック合成のみ）、fold 2011 は k126 のみ。
  エラーではない。最終学習では両ホライズンとも `min_label_days=100` を大きく超えるので 6 本すべて学習される見込み
