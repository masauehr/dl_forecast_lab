"""Phase6: 実データ（自宅の発電量・手元のみ）への適用。合成データで固定した設定を、そのまま使う（チューニングしない）。

🔒 結果は outputs/real/ にだけ書く（git管理外）。日付・個別の値は出さず、集計（MAE・カバー率）だけを表示する。
設定: final_eval.FINAL_CONFIG（MLP=(32)・GRU=窓30・HGB既定）。NN系は5つの初期値の平均（実データは短いので多めに）。
分割: 時間順に 学習60%・検証20%・テスト20%（約309／103／103日）。
  --split val : 学習で学習し、検証で評価（予行演習。何度でも可）
  --split test: 学習＋検証で学習し直し、テストで評価（1回だけ。outputs/real/REAL_TEST_USED.json で拒否する）

予報の代わり（fc1）は、対象日の日射量（ほぼ完璧な予報。合成データの世界Bに相当）。実運用の予報ではない。
使い方: .venv/bin/python src/real_eval.py --split val
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_eval import FINAL_CONFIG, QS  # noqa: E402
from real_data import load_real, split_real  # noqa: E402
from tsf_eval import coverage, errors, linear_fc, pinball, residual_quantiles, width  # noqa: E402
from tsf_models import fit_predict, mlp_fit_predict, seq_fit_predict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    lock = OUT / "REAL_TEST_USED.json"
    if a.split == "test" and lock.exists():
        sys.exit(f"実データのtestは使用済みです（{lock}）。")
    df = load_real()
    tr, va, te = split_real(df)
    if a.split == "val":
        fit, ev, hist = tr, va, va.iloc[:0]
    else:
        fit, ev, hist = pd.concat([tr, va], ignore_index=True), te, te.iloc[:0]
    c, n = FINAL_CONFIG, 5
    L = 1
    print(f"学習 {len(fit)}日／評価 {len(ev)}日（実データ・{a.split}）")
    ev_all = ev                                              # 評価期間は学習の直後に連続しているので、履歴は fit から取れる
    preds = {"予報(=日射量)の直線回帰": linear_fc(fit, ev, L),
             "HGB（全特徴量）": fit_predict("hgb", fit, ev, L, seed=0, **c["hgb"]),
             "HGB（予報・季節だけ）": fit_predict("hgb", fit, ev, L, ("base",), seed=0, **c["hgb"]),
             "MLP（予報・季節だけ）": mlp_fit_predict(fit, ev, L, groups=("base",), seed=0, n_seeds=n, **c["mlp"]),
             "GRU（生の窓）": seq_fit_predict("gru", fit, ev, L, seed=0, n_seeds=n, **c["gru"])}
    y = ev["gen_kwh"].to_numpy()
    ok = ~np.isnan(preds["GRU（生の窓）"])
    base = errors(np.where(ok, preds["予報(=日射量)の直線回帰"], np.nan), y)["MAE"]
    rows = []
    for m, p in preds.items():
        e = errors(np.where(ok, p, np.nan), y)
        rows.append({"手法": m, "MAE": round(e["MAE"], 3), "RMSE": round(e["RMSE"], 3), "bias": round(e["bias"], 3), "直線回帰との差": round(e["MAE"] - base, 3)})
    mean_mae = errors(np.where(ok, float(fit["gen_kwh"].mean()), np.nan), y)["MAE"]
    rows.append({"手法": "（参考）学習期間の平均", "MAE": round(mean_mae, 3), "RMSE": np.nan, "bias": np.nan, "直線回帰との差": round(mean_mae - base, 3)})
    res = pd.DataFrame(rows)
    # 確率予測（MLP分位点 vs 線形＋一定幅）
    p_fit, p_ev = linear_fc(fit, fit, L), linear_fc(fit, ev, L)
    Ql = p_ev[:, None] + residual_quantiles(fit["gen_kwh"].to_numpy() - p_fit, QS)[None, :]
    Qm = mlp_fit_predict(fit, ev, L, groups=("base",), seed=0, n_seeds=n, quantiles=QS, **c["mlp"])
    Qg = seq_fit_predict("gru", fit, ev, L, seed=0, n_seeds=n, quantiles=QS, **c["gru"])
    prob = pd.DataFrame([{"手法": m, "平均pinball損失": round(pinball(y, np.where(ok[:, None], Q, np.nan), QS), 3),
                          "80%区間のカバー率": round(coverage(y[ok], Q[ok, 0], Q[ok, 8]), 3), "80%区間の幅": round(width(Q[ok, 0], Q[ok, 8]), 3)}
                         for m, Q in (("線形＋一定幅", Ql), ("MLP分位点", Qm), ("GRU分位点", Qg))])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    res.to_csv(OUT / f"real_{a.split}_point.csv", index=False); prob.to_csv(OUT / f"real_{a.split}_prob.csv", index=False)
    print(res.to_string(index=False)); print(prob.to_string(index=False))
    if a.split == "test":
        lock.write_text(json.dumps(dict(used_at=datetime.now().isoformat(timespec="seconds"), note="実データのtestは使用済み"), ensure_ascii=False))
        print("REAL_TEST_USED.json を作成しました（以後 --split test は拒否されます）")


if __name__ == "__main__":
    main()
