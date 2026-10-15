"""Phase3: 系列モデル（GRU・1D-CNN）。手作りの特徴量なしで、生の時系列の窓から隠れた状態を読めるか。

世界（予報の質と隠れた状態を、変えて比べる）:
  A 普通              : 通常の予報（リードが長いほど不正確）。隠れた状態（温度・よごれ）あり
  B 予報が完璧        : fc を真の日射量に置き換える（診断用）。隠れた状態あり
  C 予報が完璧・隠れた状態なし: B から、よごれ（soiling_rate=0）と温度の低下（temp_coef=0）を消す。伸びしろが無い対照実験
E1 窓の長さ（W=7,14,30,60）の効果（GRU・世界A,B）
E2 手法の比較（世界A,B,C。L=1）: 直線回帰／HGB／MLP／窓MLP／GRU／1D-CNN
E3 GRUは、よごれを読んでいるか: 予報が完璧なB世界で、GRUの予測の誤差と、真のよごれ・温度の低下との関係

使い方: .venv/bin/python src/run_phase3.py   （約15分）
出力:   outputs/phase3_*.csv
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
SEEDS = [1_000_000 + i for i in range(10)]
EVAL_SEEDS = SEEDS[1:6]                      # E2・E3 で使う世界（Phase2 の選択世界 SEEDS[0] を除く5つ）
HGB = dict(max_iter=300, learning_rate=0.05, max_depth=4)
WORLDS = {"A 普通": dict(perfect=False, kw={}), "B 予報が完璧": dict(perfect=True, kw={}),
          "C 予報が完璧・隠れた状態なし": dict(perfect=True, kw=dict(soiling_rate=0.0, temp_coef=0.0))}


def _world(seed, name, L=1):
    cfg = WORLDS[name]
    sp = split_series(make_series(seed, **cfg["kw"]))
    tr, va = sp["train"].copy(), sp["val"].copy()
    if cfg["perfect"]:
        for d in (tr, va):
            d[f"fc{L}"] = d["gsr_true"]
    return tr, va


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def e1(seeds=SEEDS[1:4], windows=(7, 14, 30, 60), n_seeds=2):
    rows = []
    for name in ("A 普通", "B 予報が完璧"):
        acc = {W: [] for W in windows}
        for sd in seeds:
            tr, va = _world(sd, name)
            for W in windows:
                p = seq_fit_predict("gru", tr, va, 1, window=W, seed=sd, n_seeds=n_seeds)
                acc[W].append(errors(p, va["gen_kwh"])["MAE"])
        for W in windows:
            rows.append({"世界": name, "窓の長さW（日）": W, "GRUのMAE": _stat(acc[W])[0], "標準誤差": _stat(acc[W])[1]})
    return pd.DataFrame(rows)


def e2(seeds=EVAL_SEEDS, L=1, n_seeds=3):
    acc = {}
    for name in WORLDS:
        for sd in seeds:
            tr, va = _world(sd, name)
            ms = {"予報の直線回帰": linear_fc(tr, va, L), "HGB（手作り特徴量・全部）": fit_predict("hgb", tr, va, L, seed=sd, **HGB),
                  "HGB（予報・季節だけ）": fit_predict("hgb", tr, va, L, ("base",), seed=sd, **HGB),
                  "MLP（予報・季節だけ）": mlp_fit_predict(tr, va, L, groups=("base",), hidden=(32,), seed=sd, n_seeds=n_seeds),
                  "窓MLP（生の窓を並べただけ）": seq_fit_predict("flat", tr, va, L, seed=sd, n_seeds=n_seeds),
                  "GRU（生の窓）": seq_fit_predict("gru", tr, va, L, seed=sd, n_seeds=n_seeds),
                  "1D-CNN（生の窓）": seq_fit_predict("cnn", tr, va, L, seed=sd, n_seeds=n_seeds),
                  "上限2（隠れた状態まで知っている）": oracle_full(va)}
            ok = ~np.isnan(ms["GRU（生の窓）"])
            for m, p in ms.items():
                acc.setdefault((name, m), []).append(errors(np.where(ok, p, np.nan), va["gen_kwh"])["MAE"])
    rows = []
    for (name, m), v in acc.items():
        h = np.array(acc[(name, "HGB（手作り特徴量・全部）")])
        d = np.array(v) - h
        rows.append({"世界": name, "手法": m, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1], "HGB（全部）との差": _stat(d)[0], "差の標準誤差": _stat(d)[1]})
    return pd.DataFrame(rows)


def e3(seeds=EVAL_SEEDS, L=1, n_seeds=3):
    """B世界（予報が完璧）で、予測の誤差（予測−実測）が、よごれ・温度の低下とどれだけ相関するか。
    よごれを読めているモデルは、誤差がよごれと無相関になる。予報を直線回帰しただけのモデルは、よごれの分だけ高めに外れる"""
    acc = {}
    for sd in seeds:
        tr, va = _world(sd, "B 予報が完璧")
        ms = {"予報の直線回帰": linear_fc(tr, va, L), "HGB（予報・季節だけ）": fit_predict("hgb", tr, va, L, ("base",), seed=sd, **HGB),
              "GRU（生の窓）": seq_fit_predict("gru", tr, va, L, seed=sd, n_seeds=n_seeds)}
        ok = ~np.isnan(ms["GRU（生の窓）"])
        # 比較しやすいよう「割合の誤差」にする: (予測−実測) / 実測（雨の日などの小さい値を除く）
        keep = ok & (va["gen_kwh"].to_numpy() > 2.0)
        soil, hot = va["soiling"].to_numpy()[keep], (1 - va["derate"].to_numpy())[keep]
        for m, p in ms.items():
            rel = (p[keep] - va["gen_kwh"].to_numpy()[keep]) / va["gen_kwh"].to_numpy()[keep]
            acc.setdefault(m, []).append((np.corrcoef(rel, soil)[0, 1], np.corrcoef(rel, hot)[0, 1], np.mean(np.abs(rel))))
    rows = []
    for m, v in acc.items():
        a = np.array(v)
        rows.append({"手法": m, "誤差とよごれの相関": _stat(a[:, 0])[0], "誤差と温度の低下の相関": _stat(a[:, 1])[0], "割合の誤差の平均（絶対値）": _stat(a[:, 2])[0]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3)]:
        df = fn()
        print(f"\n== {name} ==", flush=True)
        print(df.round(3).to_string(index=False), flush=True)
        df.to_csv(ROOT / "outputs" / f"phase3_{name}.csv", index=False)
