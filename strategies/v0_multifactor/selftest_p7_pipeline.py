#!/usr/bin/env python3
"""selftest_p7_pipeline.py — P7 全部 ON で train_v2.py -> meta_v2.json -> submission.py が一貫するかを合成データで検査。

    python strategies/v0_multifactor/selftest_p7_pipeline.py

実データ不要。このフォルダの *.py と tools/progress.py を一時フォルダへコピーし、その中で実行する
（本物の models_v2/ と meta_v2.json には一切書かない）。build_features / make_label / scoring_index は
selftest_p5.synthetic_panel の合成パネルに差し替える。確認すること:
- meta_v2.json の "features" = feature_selection.columns = fit_model（2 パス）の選択列、rank モデルも同じ列。
- submission.predict_split の出力 = fit_model + predict_signal + 最終 EWMA（max|diff| <= 1e-6、欠損なし）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P7_ALL_ON = {"block_set": "v3", "sample_decay_halflife": 250, "feature_top_k": 20,
             "ensemble_weights": {"lgbm": 1.0, "ridge": 0.6, "rank": 0.2},
             "seeds": [0, 1], "model_params": {"n_estimators": 25}, "rank_model": {"lgbm_params": {"n_estimators": 15}}}


def outer() -> int:
    tools = HERE.parent.parent / "tools" / "progress.py"
    with tempfile.TemporaryDirectory(prefix="p7_pipeline_") as tmp:
        strategy = Path(tmp) / "strategies" / HERE.name
        strategy.mkdir(parents=True)
        (Path(tmp) / "tools").mkdir()
        shutil.copy2(tools, Path(tmp) / "tools" / "progress.py")
        for path in HERE.glob("*.py"):
            shutil.copy2(path, strategy / path.name)
        config = json.loads((HERE / "walkforward_config.json").read_text(encoding="utf-8"))
        sys.path.insert(0, str(HERE))
        import alpha_v2  # noqa: PLC0415

        config["params"] = alpha_v2._deep_merge(config.get("params") or {}, P7_ALL_ON)
        (strategy / "walkforward_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        return subprocess.call([sys.executable, str(strategy / Path(__file__).name), "--inner"])


def inner() -> int:
    sys.path.insert(0, str(HERE))
    import numpy as np  # noqa: PLC0415

    import alpha_v2  # noqa: PLC0415
    import slowdown  # noqa: PLC0415
    from selftest_p5 import synthetic_panel  # noqa: PLC0415

    features, labels = synthetic_panel()
    dates = features.index.get_level_values("Date")
    train_mask = dates < dates.unique()[280]
    labels = labels.loc[features.index[train_mask]]
    alpha_v2.build_features = lambda splits=("train",), start=None, end=None, codes=None: (
        features.copy() if "valid" in splits else features.loc[train_mask].copy())
    alpha_v2.make_label = lambda k, start=None, end=None, clip=None, window=None: labels[f"k{int(k)}"].copy()

    import train_v2  # noqa: PLC0415

    if train_v2.main(["--strategy", str(HERE), "--progress", str(HERE / "progress.json")]) != 0:
        return 1
    import submission  # noqa: PLC0415

    predict_index = features.index[~train_mask]
    submission.scoring_index = lambda split: predict_index
    meta = submission.load_meta()
    config = dict(meta["params"], blocks=meta["blocks"])
    rank_models = submission.load_rank_models(meta, config)
    served = submission.predict_split("valid", meta, config, submission.load_boosters(meta), rank_models)

    params = json.loads((HERE / "walkforward_config.json").read_text(encoding="utf-8"))["params"]
    # Same training rows as train_v2.py / walkforward: only rows with at least one label. The decay
    # reference date (last training Date) therefore is the last labelled date, in both paths.
    labelled = labels.dropna(how="all")
    train = features.loc[features.index.intersection(labelled.index)]
    model = alpha_v2.fit_model(train, labelled.reindex(train.index), params)
    expected = slowdown.ewm_by_code(alpha_v2.predict_signal(features, model, params),
                                    int(meta["smoothing_span"])).reindex(predict_index)
    diff = float(np.nanmax(np.abs(served.to_numpy(np.float64) - expected.to_numpy(np.float64))))
    rank_columns = all(list(b.feature_name()) == meta["features"] for b in rank_models.values())
    checks = {
        "meta features == feature_selection.columns": (meta["feature_selection"] or {}).get("columns") == meta["features"],
        "meta features == fit_model 2-pass columns": model["columns"] == meta["features"],
        "rank models use the selected columns": bool(rank_models) and rank_columns,
        "ridge uses the v3 blocks": meta["ridge"]["columns"] == list(alpha_v2.BLOCK_SETS["v3"]),
        "sample_decay recorded": bool(meta["sample_decay"]),
        f"submission == fit_model+predict_signal (max|diff| = {diff:.2e})": diff <= 1e-6,
        "no NaN in the served signal": int(served.isna().sum()) == 0,
    }
    print(f"selected {len(meta['features'])} columns ({meta['feature_selection']['top_k']} + sector), "
          f"rank models={len(rank_models)}, sample_decay={meta['sample_decay']}")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'NG'}] {name}")
    print("RESULT:", "OK" if all(checks.values()) else "NG")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(inner() if "--inner" in sys.argv[1:] else outer())
