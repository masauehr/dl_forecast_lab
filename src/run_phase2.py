"""Phase2: ニューラルネット（MLP）。同じ特徴量で、木と比べる。

E1 学習のコツ: 学習曲線（過学習）、入力の標準化の有無、ネットの大きさ
E2 設定（隠れ層・ドロップアウト・重み減衰）を val で選ぶ（1つの世界で選び、別の世界で評価）
E3 学習シード（初期値）によるぶれ。1つ vs 5つの平均（アンサンブル）
E4 手法の比較（L=1,3,5。選択に使った世界を除く9世界の平均±標準誤差）

使い方: .venv/bin/python src/run_phase2.py
出力:   outputs/phase2_*.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_eval import errors, linear_fc  # noqa: E402
from tsf_models import fit_predict, mlp_fit_predict  # noqa: E402
from tsf_synth import make_series, split_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
MLP_DEFAULT = dict(hidden=(64, 64), dropout=0.0, wd=1e-4)
MLP_SELECTED = dict(hidden=(32,), dropout=0.0, wd=1e-4)       # E2 で、開発用の世界（SEEDS[0]）の val により選んだ設定
HGB_DEFAULT = dict(max_iter=300, learning_rate=0.05, max_depth=4)


def _mae(p, va):
    return errors(p, va["gen_kwh"])["MAE"]


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def e1(seed=SEEDS[0], L=1):
    """学習曲線（標準化済みの目的変数の二乗誤差）と、標準化の有無・ネットの大きさ"""
    sp = split_series(make_series(seed)); tr, va = sp["train"], sp["val"]
    rows, curves = [], {}
    cases = {"小（32）": dict(hidden=(32,)), "中（64,64）": dict(hidden=(64, 64)), "大（128,128,128）": dict(hidden=(128, 128, 128)),
             "中・標準化なし": dict(hidden=(64, 64), standardize=False)}
    for name, kw in cases.items():
        p, info = mlp_fit_predict(tr, va, L, max_epochs=150, patience=1000, return_info=True, **kw)      # 早期終了なしの曲線を見るため patience を大きく
        tl, el = info["curves"][0]
        curves[name] = (tl, el)
        best = int(np.argmin(el)) + 1
        rows.append({"ネット": name, "内側検証が最小のエポック": best, "そのときの内側検証の損失": round(min(el), 3), "最後のエポックの内側検証の損失": round(el[-1], 3),
                     "最後の学習損失": round(tl[-1], 3), "検証MAE（そのエポック数で学習し直し）": round(_mae(p, va), 3)})
    return pd.DataFrame(rows), curves


def e2(select_seed=SEEDS[0], eval_seeds=SEEDS[1:], L=1):
    sp = split_series(make_series(select_seed)); tr, va = sp["train"], sp["val"]
    rows = []
    for hidden, do, wd in itertools.product(((16,), (32,), (64, 64), (128, 128)), (0.0, 0.1, 0.3), (1e-4, 1e-2)):
        rows.append(dict(hidden=hidden, dropout=do, wd=wd, MAE=_mae(mlp_fit_predict(tr, va, L, hidden=hidden, dropout=do, wd=wd), va)))
    g = pd.DataFrame(rows)
    best, worst = g.loc[g.MAE.idxmin()], g.loc[g.MAE.idxmax()]
    res = {"選んだ設定": [], "既定": [], "最悪の設定": []}
    cfg = {"選んだ設定": best, "最悪の設定": worst}
    for sd in eval_seeds:
        sp2 = split_series(make_series(sd)); t2, v2 = sp2["train"], sp2["val"]
        for k, c in cfg.items():
            res[k].append(_mae(mlp_fit_predict(t2, v2, L, hidden=c.hidden, dropout=c.dropout, wd=c.wd), v2))
        res["既定"].append(_mae(mlp_fit_predict(t2, v2, L, **MLP_DEFAULT), v2))
    desc = lambda c: f"{c.hidden}・dropout={c.dropout}・wd={c.wd}"      # noqa: E731
    out = pd.DataFrame([{"設定": k, "内容": {"選んだ設定": desc(best), "最悪の設定": desc(worst), "既定": "(64, 64)・dropout=0.0・wd=0.0001"}[k],
                         "別の9世界のval MAE": _stat(v)[0], "標準誤差": _stat(v)[1]} for k, v in res.items()])
    return out, g.sort_values("MAE").reset_index(drop=True)


def e3(seed=SEEDS[0], L=1, n=10):
    """同じ世界・同じデータで、初期値（学習シード）だけを変える"""
    sp = split_series(make_series(seed)); tr, va = sp["train"], sp["val"]
    maes = [_mae(mlp_fit_predict(tr, va, L, seed=s, **MLP_DEFAULT), va) for s in range(n)]
    ens5 = [_mae(mlp_fit_predict(tr, va, L, seed=100 + s, n_seeds=5, **MLP_DEFAULT), va) for s in range(5)]
    rf = [_mae(fit_predict("rf", tr, va, L, seed=s), va) for s in range(5)]
    return pd.DataFrame([{"条件": f"MLP 1つ（初期値{n}通り）", "MAEの平均": round(np.mean(maes), 3), "標準偏差": round(np.std(maes, ddof=1), 3), "最小": round(min(maes), 3), "最大": round(max(maes), 3)},
                         {"条件": "MLP 5つの平均（5通り）", "MAEの平均": round(np.mean(ens5), 3), "標準偏差": round(np.std(ens5, ddof=1), 3), "最小": round(min(ens5), 3), "最大": round(max(ens5), 3)},
                         {"条件": "ランダムフォレスト（乱数5通り）", "MAEの平均": round(np.mean(rf), 3), "標準偏差": round(np.std(rf, ddof=1), 3), "最小": round(min(rf), 3), "最大": round(max(rf), 3)}])


def e4(seeds=SEEDS[1:], leads=(1, 3, 5), cfg=None):
    """選択に使った世界（SEEDS[0]）を除く9世界で比べる"""
    cfg = cfg or MLP_SELECTED
    acc = {}
    for sd in seeds:
        sp = split_series(make_series(sd)); tr, va = sp["train"], sp["val"]
        for L in leads:
            ms = {"予報の直線回帰": linear_fc(tr, va, L), "リッジ": fit_predict("ridge", tr, va, L), "ランダムフォレスト": fit_predict("rf", tr, va, L, seed=sd),
                  "勾配ブースティング": fit_predict("hgb", tr, va, L, seed=sd, **HGB_DEFAULT),
                  "MLP（1つ）": mlp_fit_predict(tr, va, L, seed=sd, **cfg), "MLP（5つの平均）": mlp_fit_predict(tr, va, L, seed=sd, n_seeds=5, **cfg),
                  "MLP（予報・季節だけ）": mlp_fit_predict(tr, va, L, groups=("base",), seed=sd, **cfg),
                  "ランダムフォレスト（予報・季節だけ）": fit_predict("rf", tr, va, L, ("base",), seed=sd),
                  "勾配ブースティング（予報・季節だけ）": fit_predict("hgb", tr, va, L, ("base",), seed=sd, **HGB_DEFAULT)}
            ok = ~np.isnan(ms["勾配ブースティング"])
            for name, p in ms.items():
                acc.setdefault((name, L), []).append(errors(np.where(ok, p, np.nan), va["gen_kwh"])["MAE"])
    rows = []
    for (name, L), v in acc.items():
        d = np.array(v) - np.array(acc[("予報の直線回帰", L)])
        rf = np.array(v) - np.array(acc[("ランダムフォレスト", L)])
        rows.append({"手法": name, "リードL": L, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1], "直線回帰との差": _stat(d)[0], "ランダムフォレストとの差": _stat(rf)[0], "RFとの差の標準誤差": _stat(rf)[1]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)]:
        out = fn()
        for j, df in enumerate(out if isinstance(out, tuple) else (out,)):
            if isinstance(df, dict):
                continue
            print(f"\n== {name}-{j} ==")
            print(df.head(12).round(3).to_string(index=False) if name == "E2" and j == 1 else df.round(3).to_string(index=False))
            df.to_csv(ROOT / "outputs" / f"phase2_{name}_{j}.csv", index=False)
