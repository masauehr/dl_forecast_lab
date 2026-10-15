"""Phase5: 確率予測（分位点回帰・pinball loss）。点の予測ではなく「どのくらいの範囲に入るか」を出す。

分位点: 0.1〜0.9 の9点（80%区間＝10%点〜90%点）。世界A（普通）で、L=1（E1・E2）／L=1,3,5（E3）。
E1 手法の比較: 線形＋一定幅／HGB分位点回帰（全特徴量・予報と季節だけ）／MLP分位点／GRU分位点。
   指標: 平均pinball損失、80%区間のカバー率と幅、校正（実測が10%点以下・90%点以下だった割合）
E2 CQRによる校正: 学習期間の最後の2年を校正用に取り分け、区間の幅を足し引きしてカバー率が合うか
E3 リードごとの区間の幅と、天気ごとのカバー率（不確かさは、天気やリードで違うか）

使い方: .venv/bin/python src/run_phase5.py   （約10分）
出力:   outputs/phase5_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_eval import calibration, cqr_shift, coverage, linear_fc, pinball, residual_quantiles, width  # noqa: E402
from tsf_models import fit_predict_quantiles, mlp_fit_predict, seq_fit_predict  # noqa: E402
from tsf_synth import make_series, split_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_SEEDS = [1_000_001 + i for i in range(4)]
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LO, HI = 0, 8                                   # 80%区間の添字（0.1 と 0.9）
HGB = dict(max_iter=200, learning_rate=0.05, max_depth=4)


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def _linear_q(tr, va, L):
    """線形＋一定幅: 直線回帰の予測に、学習期間の残差の経験分位点を足す"""
    p_tr, p_va = linear_fc(tr, tr, L), linear_fc(tr, va, L)
    return p_va[:, None] + residual_quantiles(tr["gen_kwh"].to_numpy() - p_tr, QS)[None, :]


def _models(tr, va, L, sd, n_seeds=3):
    return {"線形＋一定幅": _linear_q(tr, va, L),
            "HGB分位点（全特徴量）": fit_predict_quantiles(tr, va, L, QS, seed=sd, **HGB),
            "HGB分位点（予報・季節だけ）": fit_predict_quantiles(tr, va, L, QS, ("base",), seed=sd, **HGB),
            "MLP分位点（予報・季節だけ）": mlp_fit_predict(tr, va, L, groups=("base",), hidden=(32,), seed=sd, n_seeds=n_seeds, quantiles=QS),
            "GRU分位点（生の窓）": seq_fit_predict("gru", tr, va, L, seed=sd, n_seeds=n_seeds, quantiles=QS)}


def e1(seeds=EVAL_SEEDS, L=1):
    acc = {}
    for sd in seeds:
        sp = split_series(make_series(sd)); tr, va = sp["train"], sp["val"]
        ms = _models(tr, va, L, sd)
        ok = ~np.isnan(ms["GRU分位点（生の窓）"]).any(axis=1)
        y = va["gen_kwh"].to_numpy()
        for m, Q in ms.items():
            Q = np.where(ok[:, None], Q, np.nan)
            cal = calibration(y, Q, QS)
            acc.setdefault(m, []).append((pinball(y, Q, QS), coverage(y, Q[:, LO], Q[:, HI]), width(Q[:, LO], Q[:, HI]), cal[LO], cal[HI]))
    rows = []
    for m, v in acc.items():
        a = np.array(v)
        rows.append({"手法": m, "平均pinball損失": _stat(a[:, 0])[0], "標準誤差": _stat(a[:, 0])[1], "80%区間のカバー率": _stat(a[:, 1])[0],
                     "80%区間の幅": _stat(a[:, 2])[0], "実測が10%点以下の割合": _stat(a[:, 3])[0], "実測が90%点以下の割合": _stat(a[:, 4])[0]})
    return pd.DataFrame(rows)


def e2(seeds=EVAL_SEEDS, L=1, cal_days=730):
    """学習期間の最初の8年で学習し、最後の2年で校正（CQR）。val で、校正の前後を比べる"""
    acc = {}
    for sd in seeds:
        sp = split_series(make_series(sd)); tr_all, va = sp["train"], sp["val"]
        cut = len(tr_all) - cal_days
        tr = tr_all.iloc[:cut].reset_index(drop=True)
        # 校正用データ（最後の2年）の予測: 学習は tr だけ。特徴量は、tr＋校正期間をつないで作る
        cal = tr_all.iloc[cut:].reset_index(drop=True)
        y_cal, y_va = cal["gen_kwh"].to_numpy(), va["gen_kwh"].to_numpy()
        for m in ("HGB分位点（全特徴量）", "MLP分位点（予報・季節だけ）", "GRU分位点（生の窓）"):
            def run(target):
                if m.startswith("HGB"):
                    return fit_predict_quantiles(tr, target, L, QS, seed=sd, **HGB)
                if m.startswith("MLP"):
                    return mlp_fit_predict(tr, target, L, groups=("base",), hidden=(32,), seed=sd, n_seeds=3, quantiles=QS)
                return seq_fit_predict("gru", tr, target, L, seed=sd, n_seeds=3, quantiles=QS)
            Qc = run(cal)
            both = pd.concat([cal, va], ignore_index=True)           # 検証期間の特徴量・窓は、校正期間の末尾の履歴を使えるようにつなぐ
            Qv = run(both)[len(cal):]
            s = cqr_shift(y_cal, Qc[:, LO], Qc[:, HI], 0.2)
            ok = ~np.isnan(Qv).any(axis=1)
            lo, hi = Qv[:, LO], Qv[:, HI]
            acc.setdefault(m, []).append((coverage(y_va[ok], lo[ok], hi[ok]), width(lo[ok], hi[ok]), coverage(y_va[ok], lo[ok] - s, hi[ok] + s),
                                          width(lo[ok] - s, hi[ok] + s), s, coverage(y_cal, Qc[:, LO], Qc[:, HI])))
    rows = []
    for m, v in acc.items():
        a = np.array(v)
        rows.append({"手法": m, "校正期間でのカバー率": _stat(a[:, 5])[0], "校正前のカバー率（val）": _stat(a[:, 0])[0], "校正前の幅": _stat(a[:, 1])[0],
                     "広げる量（両側）": _stat(a[:, 4])[0], "校正後のカバー率（val）": _stat(a[:, 2])[0], "校正後の幅": _stat(a[:, 3])[0]})
    return pd.DataFrame(rows)


def e3(seeds=EVAL_SEEDS, leads=(1, 3, 5)):
    acc, wacc = {}, {}
    for L in leads:
        for sd in seeds:
            sp = split_series(make_series(sd)); tr, va = sp["train"], sp["val"]
            y = va["gen_kwh"].to_numpy()
            ms = {"線形＋一定幅": _linear_q(tr, va, L),
                  "MLP分位点（予報・季節だけ）": mlp_fit_predict(tr, va, L, groups=("base",), hidden=(32,), seed=sd, n_seeds=3, quantiles=QS)}
            for m, Q in ms.items():
                acc.setdefault((m, L), []).append((pinball(y, Q, QS), coverage(y, Q[:, LO], Q[:, HI]), width(Q[:, LO], Q[:, HI])))
                if L == 1:
                    for w, name in enumerate(["晴", "曇", "雨"]):
                        k = va["weather_class"].to_numpy() == w
                        wacc.setdefault((m, name), []).append((coverage(y[k], Q[k, LO], Q[k, HI]), width(Q[k, LO], Q[k, HI])))
    r1 = pd.DataFrame([{"手法": m, "リードL": L, "平均pinball損失": _stat(np.array(v)[:, 0])[0], "80%区間のカバー率": _stat(np.array(v)[:, 1])[0],
                        "80%区間の幅": _stat(np.array(v)[:, 2])[0]} for (m, L), v in acc.items()])
    r2 = pd.DataFrame([{"手法": m, "対象日の天気": w, "80%区間のカバー率": _stat(np.array(v)[:, 0])[0], "80%区間の幅": _stat(np.array(v)[:, 1])[0]}
                       for (m, w), v in wacc.items()])
    return r1, r2


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3)]:
        out = fn()
        for j, df in enumerate(out if isinstance(out, tuple) else (out,)):
            print(f"\n== {name}-{j} ==", flush=True)
            print(df.round(3).to_string(index=False), flush=True)
            df.to_csv(ROOT / "outputs" / f"phase5_{name}_{j}.csv", index=False)
