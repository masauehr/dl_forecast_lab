"""Phase1: 古典的な機械学習（ラグ特徴量・リッジ・ランダムフォレスト・勾配ブースティング）。

E1 特徴量のグループを足していくと、どう良くなるか（L=1。リッジとHGB）
E2 手法の比較（L=1〜5。10世界の平均±標準誤差）。基準・上限と並べる
E3 ハイパーパラメータを val で選ぶ（1つの世界で選び、別の9世界で評価）
E4 どの特徴量が効いているか（グループごとの並べ替え重要度）
E5 予報が完璧だったら（fc を真の日射量に置き換える）、過去の並びから隠れた状態（温度・よごれ）を読めるか

使い方: .venv/bin/python src/run_phase1.py
出力:   outputs/phase1_*.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_eval import errors, linear_fc, oracle_full, oracle_physical  # noqa: E402
from tsf_models import build_features, fit_predict  # noqa: E402
from tsf_synth import make_series, split_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
CUM = [("base（予報・季節）", ("base",)), ("＋gen（過去の発電量・性能比）", ("base", "gen")),
       ("＋weather（天気・気温）", ("base", "gen", "weather")), ("＋dry（雨の降らない日数）", ("base", "gen", "weather", "dry"))]
HGB_DEFAULT = dict(max_iter=300, learning_rate=0.05, max_depth=4)


def _mae(p, va):
    return errors(p, va["gen_kwh"])["MAE"]


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def e1(seeds=SEEDS, L=1):
    acc = {}
    for sd in seeds:
        sp = split_series(make_series(sd)); tr, va = sp["train"], sp["val"]
        for gname, g in CUM:
            for m in ("ridge", "hgb"):
                acc.setdefault((gname, m), []).append(_mae(fit_predict(m, tr, va, L, g, **(HGB_DEFAULT if m == "hgb" else {})), va))
        acc.setdefault(("基準：予報の直線回帰", "-"), []).append(_mae(linear_fc(tr, va, L), va))
    return pd.DataFrame([{"特徴量": g, "手法": m, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1]} for (g, m), v in acc.items()])


def e2(seeds=SEEDS, leads=(1, 2, 3, 4, 5)):
    acc = {}
    for sd in seeds:
        sp = split_series(make_series(sd)); tr, va = sp["train"], sp["val"]
        for L in leads:
            ms = {"予報の直線回帰": linear_fc(tr, va, L), "リッジ": fit_predict("ridge", tr, va, L),
                  "ランダムフォレスト": fit_predict("rf", tr, va, L, seed=sd), "勾配ブースティング": fit_predict("hgb", tr, va, L, seed=sd, **HGB_DEFAULT),
                  "上限1（予報が完璧）": oracle_physical(va), "上限2（隠れた状態まで）": oracle_full(va)}
            ok = ~np.isnan(ms["勾配ブースティング"])                     # 特徴量が作れない先頭の行は、全手法で除く
            for name, p in ms.items():
                acc.setdefault((name, L), []).append(errors(np.where(ok, p, np.nan), va["gen_kwh"])["MAE"])
    rows = []
    for (name, L), v in acc.items():
        base = np.array(acc[("予報の直線回帰", L)])
        d = np.array(v) - base
        rows.append({"手法": name, "リードL": L, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1], "直線回帰との差": _stat(d)[0], "差の標準誤差": _stat(d)[1]})
    return pd.DataFrame(rows)


def e3(select_seed=SEEDS[0], eval_seeds=SEEDS[1:], L=1):
    """HGB の max_depth × learning_rate × max_iter を、1つの世界の val で選び、別の世界で評価する"""
    sp = split_series(make_series(select_seed)); tr, va = sp["train"], sp["val"]
    rows = []
    for depth, lr, it in itertools.product((2, 3, 4, 6), (0.02, 0.05, 0.1), (100, 300, 600)):
        p = fit_predict("hgb", tr, va, L, seed=0, max_depth=depth, learning_rate=lr, max_iter=it)
        rows.append(dict(max_depth=depth, lr=lr, iters=it, MAE=_mae(p, va)))
    g = pd.DataFrame(rows)
    best = g.loc[g.MAE.idxmin()]
    worst = g.loc[g.MAE.idxmax()]
    res = {"選んだ設定": [], "既定": [], "最悪の設定": []}
    for sd in eval_seeds:
        sp2 = split_series(make_series(sd)); tr2, va2 = sp2["train"], sp2["val"]
        res["選んだ設定"].append(_mae(fit_predict("hgb", tr2, va2, L, seed=0, max_depth=int(best.max_depth), learning_rate=best.lr, max_iter=int(best.iters)), va2))
        res["既定"].append(_mae(fit_predict("hgb", tr2, va2, L, seed=0, **HGB_DEFAULT), va2))
        res["最悪の設定"].append(_mae(fit_predict("hgb", tr2, va2, L, seed=0, max_depth=int(worst.max_depth), learning_rate=worst.lr, max_iter=int(worst.iters)), va2))
    out = pd.DataFrame([{"設定": k, "内容": {"選んだ設定": f"depth={int(best.max_depth)}・lr={best.lr}・iters={int(best.iters)}", "既定": "depth=4・lr=0.05・iters=300",
                                          "最悪の設定": f"depth={int(worst.max_depth)}・lr={worst.lr}・iters={int(worst.iters)}"}[k],
                         "選んだ世界のval MAE": round({"選んだ設定": best.MAE, "最悪の設定": worst.MAE, "既定": g[(g.max_depth == 4) & (g.lr == 0.05) & (g.iters == 300)].MAE.iloc[0]}[k], 3),
                         "別の9世界のval MAE": _stat(v)[0], "標準誤差": _stat(v)[1]} for k, v in res.items()])
    return out, g


def e4(seeds=SEEDS, L=1):
    """グループごとの並べ替え重要度: val で、そのグループの列だけをシャッフルしたときのMAEの増加"""
    groups = {"base": ["fc", "sin_doy", "cos_doy"], "gen（過去の発電量）": [f"gen_lag{k}" for k in range(7)] + ["gen_mean7", "gen_mean30"],
              "gen（性能比）": ["pr_lag0", "pr_mean3", "pr_mean7"], "weather": ["weather_lag0", "weather_lag1", "weather_lag2", "sunshine_lag0", "gsr_lag0", "temp_lag0", "temp_mean7"],
              "dry": ["dry_days", "rain_frac14"]}
    acc = {k: [] for k in groups}
    base_mae = []
    for sd in seeds:
        sp = split_series(make_series(sd)); tr, va = sp["train"], sp["val"]
        both = pd.concat([tr, va], ignore_index=True)
        X = build_features(both, L)
        Xtr, Xva = X.iloc[: len(tr)], X.iloc[len(tr):].reset_index(drop=True)
        ok_tr = ~Xtr.isna().any(axis=1).to_numpy(); ok_va = ~Xva.isna().any(axis=1).to_numpy()
        m = HistGradientBoostingRegressor(random_state=sd, **HGB_DEFAULT).fit(Xtr[ok_tr], tr["gen_kwh"].to_numpy()[ok_tr])
        y = va["gen_kwh"].to_numpy()[ok_va]
        b = np.abs(m.predict(Xva[ok_va]) - y).mean(); base_mae.append(b)
        rng = np.random.default_rng(sd)
        for k, cols in groups.items():
            Xp = Xva[ok_va].copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            acc[k].append(np.abs(m.predict(Xp) - y).mean() - b)
    return pd.DataFrame([{"シャッフルしたグループ": k, "MAEの増加": _stat(v)[0], "標準誤差": _stat(v)[1]} for k, v in acc.items()] +
                        [{"シャッフルしたグループ": "（何もしない MAE）", "MAEの増加": _stat(base_mae)[0], "標準誤差": _stat(base_mae)[1]}])


def e5(seeds=SEEDS, L=1):
    """診断用。fc を真の日射量に置き換えた世界（現実にはありえない）で、特徴量のグループを足したときの効きを見る"""
    acc = {}
    for sd in seeds:
        sp = split_series(make_series(sd))
        tr, va = sp["train"].copy(), sp["val"].copy()
        tr["fc1"], va["fc1"] = tr["gsr_true"], va["gsr_true"]
        for gname, g in CUM:
            for m in ("ridge", "hgb"):
                acc.setdefault((gname, m), []).append(_mae(fit_predict(m, tr, va, L, g, **(HGB_DEFAULT if m == "hgb" else {})), va))
        acc.setdefault(("上限2（隠れた状態まで）", "-"), []).append(_mae(oracle_full(va), va))
    return pd.DataFrame([{"特徴量": g, "手法": m, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1]} for (g, m), v in acc.items()])


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5)]:
        out = fn()
        for j, df in enumerate(out if isinstance(out, tuple) else (out,)):
            print(f"\n== {name}-{j} ==")
            print(df.round(3).to_string(index=False))
            df.to_csv(ROOT / "outputs" / f"phase1_{name}_{j}.csv", index=False)
