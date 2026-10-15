# dl_forecast_lab

**深層学習での時系列予測**を、合成データで学ぶプロジェクト。日次の太陽光発電量を、予報と過去の観測から予測する。
「深層学習は、単純な方法（持続予報・直線回帰・ランダムフォレスト）に**どれだけ勝てるのか、勝てないのはいつか**」を、自分の手で作って確かめる。

詳しくは [MANUAL.md](MANUAL.md) / [DATA_SPEC.md](DATA_SPEC.md) / [EVALUATION.md](EVALUATION.md) / [PLAN.md](PLAN.md) を参照。

> 🔒 学習・評価は**合成データ**。自宅の発電量などの実データ（生データ）は、最終フェーズでだけ**手元で読み取り専用**に使い、コピー・コミットしない（集計した結果の表だけ `outputs/real/` にコミットしてある）。

## なぜこのプロジェクトか

[ai_core](../ai_core/PROJECT_MAP.md) で「未経験」だった分野の1つが**深層学習での時系列予測**。実データの [ml_forecast](../ml_forecast/README.md)（RandomForest）は、約1年半のデータしかない。深層学習は**大量のデータ**で強くなるので、実データだけでは「深層学習が効かない」のか「データが足りない」のか分からない。
そこで、**長い合成系列**（14年）で、データ量・隠れた状態・ノイズを変えながら、手法を比べる。

## データの構造（合成）
```
発電量 = 0.85 × 真の日射量 × 温度の低下 × (1 − よごれ) × (1 + ノイズ3%)
```
- **日射の予報**（リード1〜5日）: 先の日ほど不正確。予報が最も効く情報。
- **温度の低下**（気温から推測できる）と**よごれ**（雨が降らない日が続くと増え、雨で洗い流される。**隠れた状態**）が、過去の並びを読むモデルの**伸びしろ**。

## 進め方（Phase）
```
Phase0 合成データ・指標・基準（持続予報・平年値・予報の直線回帰・上限）
 ─▶ Phase1 古典的な機械学習（ラグ特徴量・リッジ・ランダムフォレスト・勾配ブースティング）
 ─▶ Phase2 ニューラルネット（MLP）。学習のコツ（正規化・早期終了・シード間のぶれ）
 ─▶ Phase3 系列モデル（GRU・1D-CNN）。隠れた状態（よごれ）を読めるか
 ─▶ Phase4 データ量・ノイズの効果。実データ並みの短さ（1.5年）では？
 ─▶ Phase5 確率予測（分位点）・複数リード
 ─▶ Phase6 最終評価（新しい世界でtestを1回だけ）・実データ適用（手元のみ）・最終レポート
```

## 現在の状況
**Phase0（合成データ・指標・基準）・Phase1（古典的な機械学習）・Phase2（ニューラルネット MLP）・Phase3（系列モデル GRU/1D-CNN）・Phase4（データ量・ノイズ）・Phase5（確率予測）・Phase6（最終評価・実データ）完了。全Phase完了。** 最終レポートは [REPORT.md](REPORT.md)、Notebook実行結果のまとめは [notebooks/README.md](notebooks/README.md)。進め方は [PLAN.md](PLAN.md)。pc_docs運用マニュアルのコピーは [dl-forecast-lab.md](dl-forecast-lab.md)。

## 使い方
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests   # テスト（42件）
.venv/bin/python src/run_phase0.py               # 基準の比較（約10秒）
.venv/bin/python src/run_phase1.py               # 古典的な機械学習（約1.5分）
.venv/bin/python src/run_phase2.py               # ニューラルネット MLP（約3分）
.venv/bin/python src/run_phase3.py               # 系列モデル GRU・1D-CNN（約15分）
.venv/bin/python src/run_phase4.py               # データ量とノイズ（約20分）
.venv/bin/python src/run_phase5.py               # 確率予測（約10分）
.venv/bin/python src/final_eval.py --split val   # 最終評価の予行演習（約12分）
.venv/bin/python src/final_eval.py --split test  # 最終評価（1回だけ。使用済みだと拒否される）
.venv/bin/python src/real_eval.py --split val    # 実データ（手元のみ。集計表を outputs/real/ に出力）
```
Notebook は `notebooks/`（初心者向け、出力を消してコミット）。カーネルは「Python (dl_forecast_lab)」。

## Notebook を読むとき
はじめての方は、まず [notebooks/00_basics.ipynb](notebooks/00_basics.ipynb)（基礎知識・用語集・読む順）から。各Notebookの冒頭には「やさしい導入」があります。
