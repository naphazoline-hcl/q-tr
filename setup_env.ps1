# 環境構築（Windows PowerShell）

配布 `requirements.txt` の固定バージョンを venv に導入する。パッケージのダウンロードは
数分かかることがあるため、バックグラウンド実行してログで確認する。

```powershell
cd C:\Users\UMA\Documents\stock_comp_2026_v0
python -m venv .venv
$p = Start-Process -FilePath '.venv\Scripts\python.exe' `
    -ArgumentList '-m pip install -r requirements.txt' `
    -RedirectStandardOutput 'work\logs\pip_install.log' `
    -RedirectStandardError 'work\logs\pip_install.err' -NoNewWindow -PassThru
```

完了の確認:

```powershell
& '.\.venv\Scripts\python.exe' -c "import pandas,numpy,pyarrow,lightgbm,sklearn;print(pandas.__version__,numpy.__version__,pyarrow.__version__,lightgbm.__version__,sklearn.__version__)"
# 期待: 3.0.3 2.4.6 24.0.0 4.1.0 1.9.0
```

疎通の確認（配布サンプルの採点・README 記載値と一致するはず）:

```powershell
& '.\.venv\Scripts\python.exe' evaluate_script.py --submission strategies/sample02_size_liquidity
# 期待: SHARPE: 1.12826446
```

注意:

- **lightgbm は 4.1.0**: 配布 `requirements.txt` の記載は 4.6.0 だったが、仮提出の作業で
  実際の Web 採点環境は 4.1.0 と判明した（配布 README の GPU 欄にも 4.1.0 と記載）。
  本ディレクトリの `requirements.txt` は 4.1.0 に修正済み。モデルファイル形式 v4 は
  4.1.0 / 4.6.0 のどちらでも読み書きでき、既存モデルはそのまま使える。
- 配布要件は Python 3.11 だが、現在の開発機は 3.13 のみ登録済み。
  パッケージがバージョンまで同一のため挙動は等価（仮提出の学習・採点もこの構成）。
- `py -0p` で 3.11 が追加された場合は 3.11 の venv を作り直して同じ手順で検証すること。
