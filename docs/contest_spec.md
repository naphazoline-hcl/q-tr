# コンペ仕様とルール（v0.app に渡す参照資料）

このファイルは配布 README・`evaluate_script.py`・`input_data_explorer.ipynb` から
**採点に効く事実だけ**を抜き出したものである。v0 は入力データを取得できないため、
ここに書かれた数式・制約が判断の根拠になる。

## 1. 何を予測するか

- 日本株 **498銘柄の固定ユニバース**。各銘柄は上場期間だけ行がある。
- 時点 `t` までに利用可能な情報から、**翌営業日の寄付から翌々営業日の寄付まで**に
  実現する **1日リターンの市場残差** を予測する。

```
   t−1        t          t+1         t+2
    │    シグナル日        │           │
    │         │        寄付で建玉    寄付で評価
    └─────────┘           └───────────┘
   raw_return[t]           target[t] = この区間のリターンの市場残差
 （見てよい過去）           （予測すべき未来）
```

- `raw_target[t] = raw_return[t+2]`（実測で完全一致を確認済み）
- `target[t] = raw_target[t] - beta[t+2] * topix_return[t+2]`（実測で完全一致を確認済み）
- `topix_return` は TOPIX 連動 ETF 4本から合成した近似系列（指数との相関 約0.90）。
  数式は `0.6 × 平均ETF(始値→始値)[t] + 0.4 × 平均ETF(終値→終値)[t−1]`。

| 区分 | 期間 | ファイル接尾辞 |
|---|---|---|
| train | 2008-11-04 〜 2016-03-31 | `*_train.parquet` |
| valid | 2016-04-01 〜 2026-07-31 | `*_valid.parquet` |

配布データの `Date` は「その行の情報を利用できた日」を意味し、ファイルごとに意味が違う:
株価・リターン系・`listed_info` は取引日、`fins_statements` は開示日、
マクロ2表（ecb / consumer）は**日本時間で利用可能になった日時**。

## 2. 採点（`evaluate_script.py` と完全同一）

予測 `signal` は営業日ごとに 5 分位のロング・ショートへ変換され、
片道 0.1% の取引コスト控除後の日次損益から年率 Sharpe を計算する。

```python
TRANSACTION_COST_RATE = 0.1 * 0.01

def compute_weight(signal):
    quantile = (signal.sort_index().fillna(0).groupby("Date").rank(method="first")
                .groupby("Date").transform(lambda x: pd.qcut(x, 5, labels=False)))
    return (quantile - 2) / quantile.groupby("Date").count() / 1.2

def compute_pl(signal, target_1day):
    weight = compute_weight(signal).iloc[:, 0]
    return (weight * target_1day.iloc[:, 0]
            - TRANSACTION_COST_RATE * weight.groupby("Code").diff().abs().fillna(weight))

def compute_sr(pl):
    daily_return = pl.groupby("Date").sum()
    return float(daily_return.mean() / daily_return.std() * np.sqrt(252))
```

採点の含意（重要）:

- 重みは `(quantile − 2) / その分位の銘柄数 / 1.2`。**分位内はすべて等ウェイト**で、
  シグナルの水準（値の大きさ）は結果に効かない。**順位だけが効く**。
- 1日のグロスエクスポージャ（|重み|合計）は常に約 5.0（ロング2.5 + ショート2.5）。
- コストは `0.001 × |前日との重みの差|`。**建玉が入れ替わる銘柄数に比例**する。
  したがって「予測精度（RankIC）」より **回転率（売買の少なさ）** が支配的な局面がある。
- 年率ボラティリティは 4〜5% 程度しかない。グロス年率 +3〜4% に対して
  年率コストが 10% を超えると Sharpe は簡単に負になる。
- `signal.fillna(0)` されるので **NaN は「中立」として扱われる**（0 は中央分位側）。

### 実測済みの対比（配布 README より）

| 戦略 | 日次回転率 | Valid スコア | 備考 |
|---|---|---|---|
| sample01 短期リバーサル | 1.337 | **−6.35** | RankIC +0.0072 と高いのにコスト年率 33.7% で負ける |
| sample02 規模・流動性 | 0.013 | +1.12 | RankIC +0.0053 で低いのに勝つ |
| sample03 財務 | 0.033 | +0.55 | |
| sample04 マルチファクター | 0.030 | +1.14 | |
| sample05 マクロ局面 | 0.036 | +0.80 | |
| （仮提出）低速マルチファクター + GBDT | 0.017 | **+0.754** | Train OOS（purged walk-forward）は約 +2.4 |

**教訓: 回転率を落とす設計（遅いシグナル・長期平均ラベル・平滑化）が最優先。**

## 3. 提出形式と実行環境

- 提出は `submission.py` を含む戦略フォルダの zip。同梱ファイルも可。
- `predict()` は `(Date, Code)` を index に持つ **1列の DataFrame** を返す。
  採点対象の index を **完全に覆う**必要がある（1行でも欠けるとエラー）。
- 採点は `predict()` の実行ディレクトリがデータ展開先になる。配布 parquet は
  **ベース名で相対読み**する（例: `pd.read_parquet("target...")` は不可、`prices_daily_quotes_train.parquet` は可）。
  同梱ファイルは `Path(__file__).resolve().parent` 基準で読む。
- 採点は **30分以内**に完了する必要がある。**ネットワーク不可**。
- 環境は Python 3.11 + 固定5パッケージ（pandas 3.0.3 / numpy 2.4.6 / pyarrow 24.0.0 /
  lightgbm **4.1.0** / scikit-learn 1.9.0）。配布 `requirements.txt` の記載は 4.6.0 だが、
  実際の Web 採点環境は 4.1.0 と判明した（仮提出の作業で確認。配布 README の GPU 欄も 4.1.0 と記載）。
  開発機の venv も 4.1.0 に合わせている（モデルファイル形式 v4 は両バージョンで互換）。
- GPU は保証されない。**CPUのみで完結**させること（GPU必須の実装は採点されない可能性がある）。

## 4. 禁止事項（違反すると失格・0点になりうる）

1. 予測コードから `target_1day_valid.parquet` / `raw_target_1day_*` を読まない
   （`target_1day_train.parquet` も学習コード以外では読まない）。
   ローカル検査: `python tools/check_lookahead.py --submission <フォルダ> --runtime`
2. 特徴量で **負の `shift`・`center` 付き rolling・`bfill`** を使わない。
3. **別日付を混ぜた順位化・標準化**をしない（順位・標準化は同一日付の断面のみ）。
4. 統計の利用可能日時より前に使わない（後から改定される統計は初報値のみ）。
5. 現在の指数構成・属性をそのまま過去に当てはめない（`listed_info` は同日付結合）。

## 5. ローカル開発の流れ（人間側の作業）

```bash
python tools/profile_data.py                      # docs/data_schema.md を再生成
python tools/score.py  --submission strategies/v0_multifactor --split valid --json work/reports/x.json
python tools/score.py  --submission strategies/v0_multifactor --split train
python tools/check_lookahead.py --submission strategies/v0_multifactor --runtime
python tools/walkforward.py --strategy strategies/v0_multifactor
python tools/make_zip.py --name v0_multifactor --score
```
