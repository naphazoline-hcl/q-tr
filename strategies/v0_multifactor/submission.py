"""robust_multifactor: 低速マルチファクター + 勾配ブースティングの合成。

仮説
----
このコンペのスコアは「売買コスト控除後の年率Sharpe」なので、予測精度だけでなく
**回転率（売買の少なさ）**が本質的に効く。速いシグナルはRankICが高くても、
年率30%級のコストに負ける（サンプル01）。

戦略の枠組み
------------
1. 経済的に持続する4ブロックを日次断面で順位化して合成する（`rf_features.py`）
   - 規模・流動性 : -log(時価総額) + Amihud非流動性 + -log(売買代金)
   - バリュー     : B/P + S/P + 予想E/P + 実績E/P + CF利回り + 配当利回り
   - クオリティ   : ROE + 営業CF/総資産 + 自己資本比率
   - 低リスク     : -β + -ボラティリティ
2. 学習モデル（LightGBM）を **k営業日の平均残差リターン** をラベルに学習し、
   合成の一部（重み0.25）として加える。長いラベルで学習するとシグナル自体が
   ゆっくり動き、コストを踏まずに銘柄間の優劣を拾える
3. 最終シグナルを銘柄ごとに EWMA(span=5) で平滑化して回転率をさらに下げる

先読み禁止の遵守
----------------
- 決算短信は開示日基準の backward as-of join のみ
- rolling は過去方向のみ、`shift` は正方向のみ
- 公開特徴量・target/raw_target は予測コードから読まない
- 学習ラベルは Train の target のみ。Valid のラベルは使わない

メモリ対策
----------
採点環境のリソース上限対策として、parquet の predicate pushdown、float32、
順位化の in-place 化、推論行列の numpy 直渡しを行っている（数式は変えない）。
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rf_features import (
    FACTOR_BLOCKS,
    build_features,
    rank_model_features,
    signed_block,
    xsec_z,
)


HERE = Path(__file__).resolve().parent

# Valid の rolling 履歴を確保するための読み込み開始日（250営業日超の余裕）
HISTORY_START = pd.Timestamp("2014-06-01")


def load_meta() -> dict:
    return json.loads((HERE / "meta.json").read_text(encoding="utf-8"))


def predict_model_signal(meta: dict, ranked: pd.DataFrame) -> pd.Series:
    columns = [c for c in meta["features"] if c in ranked.columns]
    if columns != list(meta["features"]):
        raise RuntimeError("meta.json の特徴量と build_features の出力が一致しません")
    matrix = np.ascontiguousarray(ranked[columns].to_numpy(dtype=np.float32))
    total = np.zeros(len(matrix), dtype=np.float64)
    for filename in meta["model_files"]:
        booster = lgb.Booster(model_file=str(HERE / "models" / filename))
        total += booster.predict(matrix)
    total /= len(meta["model_files"])
    return pd.Series(total, index=ranked.index)


def smooth_by_code(signal: pd.Series, span: int) -> pd.Series:
    if span <= 1:
        return signal
    return signal.groupby(level="Code", sort=False).transform(
        lambda x: x.ewm(span=span, min_periods=1).mean()
    )


def predict() -> pd.DataFrame:
    meta = load_meta()

    features = build_features(splits=("train", "valid"), start=HISTORY_START)

    signal = 0.0
    for block, weight in meta["block_weights"].items():
        columns = FACTOR_BLOCKS[block]
        signal = signal + weight * xsec_z(signed_block(features, columns))

    ranked = rank_model_features(features)
    del features
    model_signal = xsec_z(predict_model_signal(meta, ranked))
    del ranked
    gc.collect()

    signal = signal + meta["model_weight"] * model_signal
    signal = smooth_by_code(signal, meta["smoothing_span"])

    valid_index = pd.read_parquet("raw_return_1day_valid.parquet").index
    result = signal.reindex(valid_index)
    return result.rename("Return").to_frame()


if __name__ == "__main__":
    frame = predict()
    print(f"prediction shape: {frame.shape}")
