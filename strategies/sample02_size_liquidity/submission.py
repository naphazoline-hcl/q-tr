"""規模・流動性プレミアム戦略（遅いシグナルの例）。

仮説
----
小型で売買しにくい銘柄を保有する投資家は、売買コスト・情報の非対称性・
アナリストカバレッジの薄さを負担する。その対価としてリスクプレミアムが要求される
（Banz 1981 の規模効果、Amihud 2002 の非流動性プレミアム）。
このプレミアムは TOPIX500 級のユニバースの内側でも観測できる。

戦略の枠組み
------------
1. 「規模の小ささ」と「売買のしにくさ」を測る（定義は features.py）
   - 規模      : 時価総額 = 終値 × 発行済株式数（決算短信から as-of で取得）
   - 非流動性  : Amihud ILLIQ = 60営業日平均の |日次リターン| / 売買代金
2. どちらも日次の断面でZ標準化して尺度を揃える
3. 2つをどんな比率で足すかは、**Train 期間の回帰で決めた係数**を使う
   （`train.py` で学習し `model.json` に保存済み。Valid のラベルは使わない）

重みを手で置かない理由
----------------------
線形に足し合わせるだけなのだから、比率は学習で決まる。
Valid のスコアを見比べて良さそうな比率を選ぶのは、特徴量選択にラベルを使うのと同じで、
このコンペで禁止されている行為にあたる。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, build_features


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.json"


def predict() -> pd.DataFrame:
    features = build_features()
    valid_index = pd.read_parquet("raw_return_1day_valid.parquet").index

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if tuple(model["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("model.jsonとfeatures.pyの特徴量定義が一致しません")

    x = features.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    signal = pd.Series(x @ coef, index=features.index)

    return signal.rename("Return").to_frame().loc[valid_index]
