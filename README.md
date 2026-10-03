# Stock Competition 2026

## コンペ概要

日本株498銘柄について、時点 $t$ までに利用可能な情報から、**翌営業日の寄付から
翌々営業日の寄付までに実現する1日残差リターン**を予測するコンペです。市場全体の
値動きはターゲットから控除済みで、銘柄間の相対的な強弱を予測します。

提出する `predict()` のシグナルは営業日ごとに5分位のロング・ショートへ変換され、
片道0.1%の取引コスト控除後の日次PLから計算した**年率Sharpe**で評価されます。
スコアは高いほど上位です。提出形式は `submission.py`、または関連ファイルや学習済み
モデルを含む戦略フォルダのzipです。
提出物の採点は30分以内に完了する必要があります。

このフォルダには、コンペのデータ仕様、予測契約、ローカル採点コード、サンプル戦略、
提出zip作成用Notebookが入っています。詳細なターゲットと採点方法は「Data」、提出条件は
「Zip Submission」を参照してください。

## ローカル実行の流れ

Python 3.11で同梱 `requirements.txt` の5パッケージを**バージョンまで揃えて**
入れます。これはローカル採点のための最小構成です。オムニキャンパス上で利用できる
追加ライブラリは「Environment」を参照してください。`input/` には
`input_manifest.json` と一致する20個のparquetが同梱されています。Notebook
（`input_data_explorer.ipynb`・`strategies/create_zip.ipynb`）を使う場合は、同じ環境に
Jupyterも用意してください。

1. `strategies/` にサンプルをコピーするなどして、自分の戦略フォルダを作る
2. `strategies/<あなたの戦略>/submission.py` に `predict()` を実装する
3. `evaluate_script.py` でローカル採点する（「Smoke Test」参照）
4. `strategies/create_zip.ipynb` で提出zipを作る（「Zip Submission」参照）

配布データの中身と出典は `input_data_explorer.ipynb` にまとまっています。

## Layout

```text
stock_comp_2026/
├── input/                 # 配布データ。*.parquet を相対パスで読む
├── input_manifest.json    # 配布データの name / size / sha256（as_of つき）
├── requirements.txt       # ローカル採点用の最小5パッケージ（Python 3.11）
├── evaluate_script.py     # ローカル採点・疎通確認スクリプト
├── input_data_explorer.ipynb # 配布データの概要・出典・注意点のNotebook
└── strategies/
    ├── create_zip.ipynb   # 提出zip作成用Notebook
    ├── sample01_short_reversal/         # 短期リバーサル（速いシグナル）
    ├── sample02_size_liquidity/         # 規模・流動性プレミアム（遅いシグナル）
    ├── sample03_fundamental_longhorizon/# 財務ファンダメンタル
    ├── sample04_multifactor/            # 複数プレミアムの合成
    ├── sample05_macro_regime/           # マクロ局面によるファクター配分切替
    └── <あなたの戦略>/
        ├── submission.py  # 提出対象。predict() を実装する
        ├── README.md      # 戦略の説明（任意）
        └── ...            # model.json など同梱ファイルも可
```

`submission.py` 以外のファイル（学習済みモデル、README、レポート等）も同じフォルダに
入れて構いません。採点時は `predict()` の実行ディレクトリがデータ展開先になるため、
同梱ファイルは `Path(__file__).resolve().parent` 基準の相対パスで読んでください。

## Sample Strategies

5本のサンプルが入っています。それぞれ配布データの別の部分を主役にし、
「何を仮説として、どの枠組みで検証するか」を書いています。
各フォルダの `README.md` に、仮説・処理・実測して考えたことが入っています。

| サンプル | 主に使うデータ | 仮説・主題 | 日次回転率 | スコア |
|---|---|---|---|---|
| `sample01_short_reversal` | `raw_return` / `beta` / `topix` | 売られすぎは戻る。速いシグナルの例 | 1.337 | −6.35 |
| `sample02_size_liquidity` | `prices` / `fins` / `listed_info` | 規模・流動性プレミアム。遅いシグナルの例 | 0.013 | +1.12 |
| `sample03_fundamental_longhorizon` | `fins` / `prices` | 財務はゆっくり動く＝実質ロングホライズン | 0.033 | +0.55 |
| `sample04_multifactor` | `prices` / `fins` / `listed_info` | 相関を見てブロック化してから合成する | 0.030 | +1.14 |
| `sample05_macro_regime` | `consumer_attitude_index` / `fins` / `prices` | マクロは断面シグナルの配分としてのみ効く | 0.036 | +0.80 |

### 最初に読んでほしい対比

`sample01`（速い）と `sample02`（遅い）を並べてください。

```
sample01 :  グロス +4.1%  −  コスト 33.7%  =  ネット −29.6%   RankIC +0.0072
sample02 :  グロス +3.9%  −  コスト  0.3%  =  ネット  +3.6%   RankIC +0.0053
```

予測精度（RankIC）は `sample01` のほうが高いのに、スコアは最下位になります。

年率コストは `片道0.1% × 日次回転率 × 252営業日` で見積もれます。
5分位ロング・ショートは市場リスクを除いてあるので年率ボラティリティが4〜5%しかなく、
回転率を意識しない戦略は、どれだけ当たっていてもコストを回収できません。

予測を当てることと、コスト控除後に残すことは別の問題です。

### すべてのサンプルに共通のルール

`target_1day_valid.parquet` と `raw_target_1day_*` を予測コードから読みません。
特徴量では負の`shift`、center付きrolling、`bfill`、別日付を混ぜた順位化・標準化を
使わないでください。

## Data

### 期間とユニバース

| 区分 | 期間 | 用途 |
|---|---|---|
| 訓練 `*_train.parquet` | 2008-11（実データ先頭 2008-11-04）〜 2016-03-31 | モデルの学習 |
| 検証/テスト `*_valid.parquet` | 2016-04-01 〜 2026-07-31 | 採点（バックテスト）の対象期間 |

ユニバースは日本株 **498 銘柄**。株価・財務情報は
J-Quantsから、為替はECB、消費者態度指数は内閣府のe-Stat原本を使用しています。
銘柄の上場期間だけ行が存在します。

### ファイル一覧

| ファイル | index | 内容 |
|---|---|---|
| `prices_daily_quotes_*` | (Date, Code) | 日次株価 40 列: 生 OHLC・制限値幅・出来高/売買代金・`AdjustmentFactor`・調整後 OHLCV（`Adjustment*`）、それぞれ通日/前場 `Morning*`/後場 `Afternoon*` |
| `fins_statements_*` | (Date, Code) | 決算短信サマリ 106 列（実績 PL/BS/CF・配当・業績予想・単体系・株式数ほか）。**開示日ベースの疎データ** |
| `listed_info_*` | (Date, Code) | 各取引日時点の銘柄属性（全上場約4,400銘柄）: 17/33業種・規模・市場区分・信用区分。Train/Valid別のPoint-in-Timeデータ |
| `raw_return_1day_*` | (Date, Code) | `AdjustmentOpen[t] / AdjustmentOpen[t−1] − 1`（**過去**の始値→始値リターン。特徴量用） |
| `topix_return_1day_*` | Date | 市場リターン系列（下記「市場リターンの定義」参照） |
| `beta_1day_*` | (Date, Code) | 対市場 β。raw_return と topix_return のローリング共分散/分散（窓 120 営業日） |
| `raw_target_1day_*` | (Date, Code) | `raw_return[t+2]`（**未来**の 2 営業日先リターン。ラベル用・残差化前） |
| **`target_1day_*`** | (Date, Code) | **予測対象** = `raw_target[t] − beta[t+2] × topix_return[t+2]` |
| `ecb_fx_rates_*` | (Date, Pair) | ECB参照レートのEUR/JPY・EUR/USDと、その2系列から算出したUSD/JPY。`Rate`、観測日、出典系列、派生区分を収録 |
| `consumer_attitude_index_*` | Date | 内閣府「消費動向調査」の消費者態度指数（原数値）。観測月、公表日時、調査方式、e-Stat原本IDを収録 |

### 銘柄属性の時点管理

`listed_info_train.parquet` と `listed_info_valid.parquet` は、各取引日にJ-Quantsで
公開されていた銘柄属性をその日の `(Date, Code)` に記録しています。2022年の市場区分変更や
業種変更を含むため、全期間を最新1時点の属性へ置き換えないでください。

価格などの日次パネルへは、同じsplitのファイルを**完全一致の `(Date, Code)`**で左結合します。
必要列だけ読むとメモリを抑えられます。

```python
features = pd.read_parquet("raw_return_1day_train.parquet")
listed = pd.read_parquet(
    "listed_info_train.parquet",
    columns=["Sector17Code", "Sector33Code"],
)
original_index = features.index
features = features.join(listed, how="left", validate="one_to_one")
assert features.index.equals(original_index)
assert features[["Sector17Code", "Sector33Code"]].notna().all().all()
```

Trainの特徴量にはTrain、Validの特徴量にはValidを使います。`Code` ごとの末尾1行へ潰して
静的マスタ化したり、未来方向の `bfill`、別日の属性を使う結合、outer/right joinを行うと、
先読みまたは行集合の変化につながるため禁止です。

### 公開特徴量の時点管理

公開特徴量2表のindex `Date` は、観測日ではなく**利用可能日時（Asia/Tokyo）**です。
Trainは `[2008-11-01, 2016-04-01)`、Validは
`[2016-04-01, 2026-08-01)` の半開区間で、公表日時から分割しています。

- ECBはフランクフルト16:00公表をJSTへ変換するため、夏時間は通常23:00、冬時間は
  翌日00:00になります。
- e-Statは過去原本に公表日しかないため、同日先読みを避けて23:59:59 JSTから利用可能と
  しています。
- 日次株価へ結合するときは `Date <= prediction_time` のbackward as-of joinを使い、
  観測日で結合したり未来方向の`bfill`をしたりしないでください。
- 消費者態度指数は原数値だけです。季節調整値、構成指標、景気ウォッチャーは含みません。
- 2016年3月の「一般世帯」から「二人以上の世帯」への変更は名称変更のみのため、
  `HouseholdScope` は全期間 `two_or_more_person_households` に統一しています。

出典: [ECB Data Portal](https://data.ecb.europa.eu/data/datasets/EXR)、
[e-Stat 消費動向調査](https://www.e-stat.go.jp/stat-search/files?toukei=00100405&tstat=000001014549)。
USD/JPYは本配布で `EUR/JPY ÷ EUR/USD` を小数9桁に丸めた加工値です。
再利用条件は[e-Stat利用規約](https://www.e-stat.go.jp/terms-of-use)と
[ECB statistics usage policy](https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html)
を参照してください。出典元は本配布による加工や、これを用いた取引・予測結果を保証しません。

### target のタイミング構造

```
   t−1        t          t+1         t+2
    │    シグナル日        │           │
    │         │        寄付で建玉    寄付で評価
    └─────────┘           └───────────┘
   raw_return[t]           target[t] = この区間のリターンの市場残差
  （見てよい過去）          （予測すべき未来）
```

t 日までのデータでシグナルを出し、**翌営業日の寄付で約定・翌々営業日の寄付で評価**する
1 日保有リターン（の市場中立版）を予測します。`raw_target[t] = raw_return[t+2]` の関係です。

### 市場リターン（topix_return）の定義

TOPIX 指数値そのものではなく、**TOPIX 連動 ETF 4 本（1305/1306/1308/1348）の価格から
全期間統一の式で合成した近似系列**です（指数実測値との相関 約 0.90）:

```
topix_return[t] = 0.6 × 平均ETF(始値→始値リターン)[t] + 0.4 × 平均ETF(終値→終値リターン)[t−1]
```

前日終値項を混ぜるのは、指数の「始値」が寄り付き前の銘柄を前日値で合成した値である
性質を近似するためです。1348 は 2009-05 上場のため、それ以前は 3 本平均です。
beta と target もこの系列から計算しており、期間による定義の切り替えはありません。

### 注意事項

- **NaN 構造**: `target`/`raw_target` はデータ**末尾 2 営業日**が NaN（未来の始値が未確定）。
  `beta` と `target` は上場直後など、銘柄ごとの履歴が足りない行も NaN。
  売買停止等で価格が無い日の行は NaN のまま入っています。
- **コード再利用・再上場**: 同一コードが上場廃止後に別の上場として再登場する銘柄は、
  上場区間ごとに独立してリターン・ラベル・beta を計算しています（20 営業日を超える
  行の欠落を区間境界とみなす）。区間の切れ目では、直前区間の末尾 2 営業日の
  `target`/`raw_target` と再開初日の `raw_return` が NaN になり、`beta` は区間内の
  履歴が溜まるまで NaN です（新規上場と同じ扱い。リターンが区間を跨ぐことはありません）。
- **2020-10-01**（東証システム障害による全銘柄売買停止日）は行ごと存在しません。
  翌営業日のリターンは直前取引日（2020-09-30）比です。
- **調整後価格の基準**: `Adjustment*` はデータ生成時点の最新日基準で、併合・分割は過去に
  遡って調整済みです（週次更新で基準日が進むと、対象銘柄の過去の調整後値も変わります。
  リターン系列は影響を受けません）。
- **fins の結合**: Date が tz-aware（Asia/Tokyo）なので、価格系列と結合するときは
  `tz_localize(None)` してから as-of 結合（`groupby("Code").ffill()` 等）してください。
- **index の整合**: ファイル間で (Date, Code) の行集合は完全一致しません。特徴量と
  ターゲットは index の intersection で揃えてください（`strategies/sample02_size_liquidity` が実例）。
- **採点**: `evaluate_script.py` と同じ — シグナルを日次 5 分位ロングショートに変換し、
  片道 0.1% のコスト控除後の日次PLで年率 Sharpe（×√252）を計算します。

## Environment

### GPU・追加ライブラリ

GPUに関連する利用可能なライブラリとバージョンは次のとおりです。

- PyTorch 2.6.0は、GPUホスト上でCUDAを利用でき、GPUがない場合はCPUで実行されます。
- Darts 0.46.1は、PyTorch系モデルでPyTorch経由のGPUを利用できます。
- XGBoost 2.1.4は、`device="cuda"` を指定するとGPUを利用できます。デフォルトはCPU実行です。
- CatBoostは、`task_type="GPU"` を指定するとGPUを利用できます。バージョンは固定されていません。
- TensorFlow 2.20.0、TensorFlow Probability 0.25.0、GPflow 2.9.2はCPU版です。
- LightGBM 4.1.0はCPU版です。

GPU環境の数には限りがあるため、GPUの割り当ては保証されません。
**GPUを必須とする提出は、GPU環境が不足した場合は実行されず、採点されません。**
確実に採点を受けるために、CPUでも実行できる実装またはフォールバックを推奨します。
また、GPU上での全処理の動作や実行速度も保証されません。

必要なライブラリが講義環境に用意されていない場合は、実装・提出の前に運営へ問い合わせてください。
採点中はネットワークを利用できません。

## Smoke Test

```bash
cd /path/to/stock_comp_2026
python3 evaluate_script.py --submission strategies/<あなたの戦略>
```

`strategies/` に戦略フォルダが 1 つしかない場合は `--submission` を省略できます。

`evaluate_script.py` は `input/` 内の target を自動検出します。valid データがある場合は
`target_1day_valid.parquet` を優先して使います。別ファイルを使う場合は次のように指定できます。

```bash
python3 evaluate_script.py --submission strategies/<あなたの戦略> --target target_1day_valid.parquet
```

## Zip Submission

提出時は自分の戦略フォルダを zip にします。`strategies/create_zip.ipynb` の `NAME` を
自分の戦略フォルダ名へ変えて実行すると、`__pycache__` や `.DS_Store`、`._*` などの
不要ファイルを除いた `strategies/<あなたの戦略>.zip` が作られます。

CLIで作る場合:

```bash
cd /path/to/stock_comp_2026/strategies
zip -r <あなたの戦略>.zip <あなたの戦略>
```

評価コード側は zip 内を探索して `submission.py` を見つけるため、フォルダごと
zip にしたもの（Finder の「圧縮」）でもそのまま採点できます。

zip でもローカル採点を確認できます。

```bash
cd /path/to/stock_comp_2026
python3 evaluate_script.py --submission strategies/<あなたの戦略>.zip
```
