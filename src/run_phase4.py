"""Phase4: データ量とノイズ。深層学習はデータが多いほど有利か。実データ並みの短さ（約1.5年）ではどうか。

世界: A 普通（予報は通常）／B 予報が完璧（診断用。隠れた状態を読む価値が出る世界）
E1 学習期間（1.5・3・5・10年）を変える: 直線回帰・HGB・MLP・GRU
E2 発電量のノイズ（1%・3%・6%・10%）を変える（B世界・学習10年）: HGB（全部）・HGB（予報・季節だけ）・GRU
※ 学習期間が短いときは、早期終了のための内側の検証を、学習期間の1/4までにする（短い学習期間では、エポック数の決定が不安定になる）

使い方: .venv/bin/python src/run_phase4.py   （約20分）
出力:   outputs/phase4_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_eval import errors, linear_fc, oracle_full  # noqa: E402
from tsf_models import fit_predict, mlp_fit_predict, seq_fit_predict  # noqa: E402
from tsf_synth import make_series, split_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_SEEDS = [1_000_001 + i for i in range(4)]
HGB = dict(max_iter=300, learning_rate=0.05, max_depth=4)


def _make(seed, perfect, train_years=None, **kw):
    sp = split_series(make_series(seed, **kw), train_years)
    tr, va = sp["train"].copy(), sp["val"].copy()
    if perfect:
        for d in (tr, va):
            d["fc1"] = d["gsr_true"]
    return tr, va


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def e1(seeds=EVAL_SEEDS, years=(1.5, 3, 5, 10), n_seeds=3):
    acc = {}
    for name, perfect in (("A 普通", False), ("B 予報が完璧", True)):
        for k in years:
            for sd in seeds:
                tr, va = _make(sd, perfect, k)
                ms = {"予報の直線回帰": linear_fc(tr, va, 1), "HGB（全特徴量）": fit_predict("hgb", tr, va, 1, seed=sd, **HGB),
                      "HGB（予報・季節だけ）": fit_predict("hgb", tr, va, 1, ("base",), seed=sd, **HGB),
                      "MLP（予報・季節だけ）": mlp_fit_predict(tr, va, 1, groups=("base",), hidden=(32,), seed=sd, n_seeds=n_seeds),
                      "GRU（生の窓）": seq_fit_predict("gru", tr, va, 1, seed=sd, n_seeds=n_seeds)}
                ok = ~np.isnan(ms["GRU（生の窓）"])
                for m, p in ms.items():
                    acc.setdefault((name, k, m), []).append(errors(np.where(ok, p, np.nan), va["gen_kwh"])["MAE"])
                acc.setdefault((name, k, "上限2"), []).append(errors(oracle_full(va), va["gen_kwh"])["MAE"])
    rows = []
    for (name, k, m), v in acc.items():
        h = np.array(acc[(name, k, "HGB（全特徴量）")])
        d = np.array(v) - h
        rows.append({"世界": name, "学習期間（年）": k, "手法": m, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1], "HGB（全特徴量）との差": _stat(d)[0], "差の標準誤差": _stat(d)[1]})
    return pd.DataFrame(rows)


def e2(seeds=EVAL_SEEDS, noises=(0.01, 0.03, 0.06, 0.10), n_seeds=3):
    acc = {}
    for nz in noises:
        for sd in seeds:
            tr, va = _make(sd, True, None, gen_noise=nz)
            ms = {"HGB（全特徴量）": fit_predict("hgb", tr, va, 1, seed=sd, **HGB), "HGB（予報・季節だけ）": fit_predict("hgb", tr, va, 1, ("base",), seed=sd, **HGB),
                  "GRU（生の窓）": seq_fit_predict("gru", tr, va, 1, seed=sd, n_seeds=n_seeds)}
            ok = ~np.isnan(ms["GRU（生の窓）"])
            for m, p in ms.items():
                acc.setdefault((nz, m), []).append(errors(np.where(ok, p, np.nan), va["gen_kwh"])["MAE"])
            acc.setdefault((nz, "上限2"), []).append(errors(oracle_full(va), va["gen_kwh"])["MAE"])
    rows = []
    for (nz, m), v in acc.items():
        h = np.array(acc[(nz, "HGB（全特徴量）")])
        d = np.array(v) - h
        rows.append({"発電量のノイズ": nz, "手法": m, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1], "HGB（全特徴量）との差": _stat(d)[0], "差の標準誤差": _stat(d)[1]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    for name, fn in [("E1", e1), ("E2", e2)]:
        df = fn()
        print(f"\n== {name} ==", flush=True)
        print(df.round(3).to_string(index=False), flush=True)
        df.to_csv(ROOT / "outputs" / f"phase4_{name}.csv", index=False)
