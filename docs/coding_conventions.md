# コーディング規約（v0 が生成するコードの必須仕様）

v0 は入力データを持たない。**この規約に従わないコードはローカルで動かない**ので必ず守ること。
不明点は質問せず Assumptions 節に記録したうえで、この規約を優先する。

## 1. 戦略フォルダの構成（v0_multifactor の例）

```text
strategies/v0_multifactor/
├── submission.py            # 提出対象。predict() を定義する
├── alpha.py                 # 戦略本体（特徴量・学習・シグナル合成）— walkforward と共有
├── rf_features.py           # 既存の特徴量定義（v1）
├── train.py                 # モデル学習（Train のみ使用）
├── meta.json                # 学習済みモデルの設定・特徴量リスト
├── walkforward_config.json  # purged walk-forward の設定
├── models/*.txt             # LightGBM モデル（テキスト形式）
└── README.md                # 戦略レポート（仮説・処理・検証結果）
```

## 2. API 契約（新コードはこの形に合わせる）

`alpha.py` は次を定義する（`tools/walkforward.py` が呼ぶ）:

```python
build_features(splits=("train",), start=None, end=None) -> pd.DataFrame
    # index=(Date, Code) の特徴量パネル。各行はその Date までの情報だけを使う（因果的）。
    # end を渡されたら Date <= end の行だけ返す。

make_label(k, start=None, end=None) -> pd.Series
    # 翌日から k 営業日の平均残差リターン。ラベルは target_1day_train.parquet からのみ作る。
    # 窓（t 〜 t+k-1）が end を超える行は NaN にする（purge を呼び出し側で保証できるように）。

fit_model(features, labels, params) -> object
    # labels は列が "k126", "k250" のような DataFrame。学習済みモデルを返す。

predict_signal(features, model, params) -> pd.Series
    # 合成済みの生シグナル（平滑化前）。index は features と同じ。
```

`submission.py` は次を満たす:

```python
def predict() -> pd.DataFrame:
    # 返り値: index=(Date, Code) / 列は1つ（列名は任意）の DataFrame。
    # 採点対象（target_1day_valid.parquet の index）を完全に覆うこと。
```

## 3. 時間（先読み）に関する絶対ルール

- 特徴量は **各時点 `t` の行について、`t` までに確定した情報だけ**を使う。
  - `rolling` は過去方向のみ（`center=False`）
  - `shift` は正の値のみ（`shift(-1)` 禁止）
  - `bfill` / `backfill` / `limit_area='both'` 禁止
  - 逆順処理（`[::-1]`）は **ラベル生成のみ** 許可（学習コード内）
- 順位化・標準化は **同一 `Date` の断面のみ**（`groupby("Date")`）。
  日付をまたいだ `rank` / `zscore` / `StandardScaler` は禁止。
  時系列の正規化は「過去方向の expanding」のみ許される。
- 決算短信（`fins_statements`）は **開示日基準の backward as-of join**。
  `Date` は tz-aware（Asia/Tokyo）なので `tz_localize(None)` してから結合する。
- マクロ統計（`ecb_fx_rates` / `consumer_attitude_index`）は `Date` =
  「日本時間で利用可能になった日時」。同じく backward as-of のみ。
- 予測コードは `target_1day_*` / `raw_target_1day_*` を読まない。
- 検算: `python tools/check_lookahead.py --submission <フォルダ> --runtime`

## 4. データの読み方（採点環境で動かすための必須事項）

- 配布 parquet は **ベース名で相対読み**する。`predict()` は cwd = データ展開先で実行される。
  開発時は `input/` があるので、`alpha.py` のようにカレントを切り替えるヘルパーを置く。
- 大きなファイルは **pyarrow の predicate pushdown** で絞る:
  `pd.read_parquet(path, columns=[...], filters=[("Date", ">=", start)])`
- `listed_info_*.parquet` は 1,000万行あるため:
  - `columns=["ScaleCategory", "Sector17Code", "Sector33Code", "MarketCode", "MarginCode"]` のみ読む
  - `Date` で絞る（例: 必要な期間だけ）
- ピークメモリは **2GB 以下**を目標（仮提出は約1GB）。float32 を使い、
  中間 DataFrame を増やさない（in-place 順位化など）。
- 実行時間は **採点30分以内**。ローカル想定: 特徴量構築 1〜3分、推論 1分以内。

## 5. 財務データの基準そろえ（重要・罠）

- `AdjustmentClose` は配布時点（2026-07-31）の株式数基準へ遡って調整済み。
- 一方 `fins_statements` の1株当たり値（`BookValuePerShare`・`EarningsPerShare`・
  `ForecastEarningsPerShare`・`ForecastDividendPerShareAnnual`）は**開示時点の基準**。
- そのまま割ると分割・併合の倍率だけ指標がずれる。比率 `R = AdjustmentClose / Close`
  を使い、`1株当たり値（生成時点基準） = 財務値 × R(開示日)`、
  `株式数（生成時点基準） = 株数 ÷ R(開示日)` と換算する。
  （実測: 換算を忘れるとサイズファクターが +2.86 → +2.16 に水増しされる）

## 6. 進捗ファイル（中断対応・必須）

25分を超えうる処理（学習・グリッド探索）は **必ず** `tools/progress.py` の `Progress` を使い、
**N ステップごとに JSONL へ書き出す**。モデルは学習が終わったものから順に保存し、
`--resume` で既存ファイルを読み飛ばせるようにする。

```python
from progress import Progress            # tools/ を sys.path に追加して import
prog = Progress("work/progress/train_v2.json", total=len(jobs), meta={"strategy": "v0_multifactor"})
for i, job in enumerate(jobs, start=1):
    prog.update(i, f"{job} 開始")
    if already_done(job):                 # 再開時はスキップ
        continue
    model = fit(job)
    save(model, job)                      # 成果物は逐次保存
    prog.update(i, f"{job} 完了", artifact=...)
prog.done()
```

## 7. 出力形式（v0 の応答形式）

- 応答は **単一の React Web アプリ（1つのコードブロック）**。素の文章だけの応答は禁止。
- アプリには生成したファイル全文を埋め込み、**ファイルごとのコピー**と
  **JSZip による一括 ZIP ダウンロード**を付ける。
- 断絶に備え、**ファイルごとに独立したダウンロードボタン**を用意する
  （一部が不完全でも、完成したファイルは回収できる）。
- すべて日本語で出力（コード内コメントのみ英語可）。
- 数値を捏造しない。不明な数値は `{要実行}` と明示する。
