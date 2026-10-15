"""Phase6: 最終評価（合成データ）。設定は開発用の世界（1.0M番台）で選んで固定した（Phase1〜5）。test の世界（3.0M番台）は1回だけ使う。

結果を見る前に決めた予想（H）:
  H1 世界A（普通・学習10年・L=1）: HGB・MLP・GRU は、どれも予報の直線回帰より良い
  H2 世界A: MLP（予報・季節だけ）と GRU は、HGB（全特徴量）より良いか、区別できない（HGB より悪くならない）
  H3 世界B（予報が完璧）: GRU は HGB（予報・季節だけ）より 0.04 以上良く、HGB（全特徴量）と 0.01 以内
  H4 世界C（予報が完璧・隠れた状態なし）: GRU は直線回帰より良くならない（差 ≧ −0.005）
  H5 世界A・学習1.5年: HGB（全特徴量）は直線回帰より悪く、MLP（予報・季節だけ）は直線回帰より良い
  H6 確率予測（世界A・L=1）: MLP・GRU の平均pinball損失は、線形＋一定幅より 0.03 以上小さく、80%区間のカバー率は 0.77〜0.83。HGB分位点（全特徴量）のカバー率は 0.79 未満
  H7 リード: MLP（予報・季節だけ）の直線回帰に対する相対改善は、L=5 のほうが L=1 より小さい

学習は train（10年。1.5年の実験では新しい側1.5年）、評価は test（最後の2年）。NN系は5つの初期値の平均。
シード（世界）: 予行演習 val=2.2M番台（4世界）、test=3.0M番台（10世界）。開発用（1.0M番台）とは別。結果を見て設定を変えない。

使い方:
  .venv/bin/python src/final_eval.py --split val    # 予行演習（約12分）
  .venv/bin/python src/final_eval.py --split test   # 本番（1回だけ。使用済みだと拒否する。約30分）
出力: outputs/final/final_{split}_*.csv（test は TEST_USED.json も作る）
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_eval import (calibration, coverage, errors, linear_fc, oracle_full, pinball, residual_quantiles,  # noqa: E402
                      width)
from tsf_models import fit_predict, fit_predict_quantiles, mlp_fit_predict, seq_fit_predict  # noqa: E402
from tsf_synth import make_series, split_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "final"

# Phase1〜5 で、開発用の世界を使って選んだ設定。最終評価の前に固定した
FINAL_CONFIG = dict(hgb=dict(max_iter=300, learning_rate=0.05, max_depth=4), hgb_q=dict(max_iter=200, learning_rate=0.05, max_depth=4),
                    mlp=dict(hidden=(32,), dropout=0.0, wd=1e-4), gru=dict(window=30, hidden=32), n_seeds=3, short_years=1.5, leads=(1, 3, 5))
SPLITS = {"val": [2_200_000 + i for i in range(4)], "test": [3_000_000 + i for i in range(10)]}
QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WORLDS = {"A 普通": dict(perfect=False, kw={}), "B 予報が完璧": dict(perfect=True, kw={}),
          "C 予報が完璧・隠れた状態なし": dict(perfect=True, kw=dict(soiling_rate=0.0, temp_coef=0.0))}


def _data(seed, name, L=1, train_years=None):
    cfg = WORLDS[name]
    sp = split_series(make_series(seed, **cfg["kw"]), train_years)
    tr, te = sp["train"].copy(), sp["test"].copy()               # 最終評価は test（最後の2年）
    if cfg["perfect"]:
        for d in (tr, te):
            d[f"fc{L}"] = d["gsr_true"]
    # 学習期間の直後が test ではない（valの2年が間に入る）ので、特徴量の履歴のため val を間に挟む（学習には使わない）
    return tr, te, sp["val"].copy() if not cfg["perfect"] else _perfect(sp["val"], L)


def _perfect(d, L):
    d = d.copy(); d[f"fc{L}"] = d["gsr_true"]; return d


def _predict_all(tr, va, te, L, sd, cfg, which):
    """va を挟んで te を予測する。履歴は va（学習に使わないが、観測としては既知）から作る。
    実装: train と test の間の val を「履歴」として使うため、train → (val+test) をつなげて予測し、test の部分を返す"""
    vt = pd.concat([va, te], ignore_index=True)
    sl = slice(len(va), None)
    n = cfg["n_seeds"]
    out = {}
    if "linear" in which:
        out["予報の直線回帰"] = linear_fc(tr, te, L)
    if "hgb" in which:
        out["HGB（全特徴量）"] = fit_predict("hgb", tr, vt, L, seed=sd, **cfg["hgb"])[sl]
    if "hgb_base" in which:
        out["HGB（予報・季節だけ）"] = fit_predict("hgb", tr, vt, L, ("base",), seed=sd, **cfg["hgb"])[sl]
    if "mlp" in which:
        out["MLP（予報・季節だけ）"] = mlp_fit_predict(tr, vt, L, groups=("base",), seed=sd, n_seeds=n, **cfg["mlp"])[sl]
    if "gru" in which:
        out["GRU（生の窓）"] = seq_fit_predict("gru", tr, vt, L, seed=sd, n_seeds=n, **cfg["gru"])[sl]
    return out


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def part_main(seeds, cfg):
    acc = {}
    def add(setting, preds, te, ref="予報の直線回帰"):
        ok = ~np.isnan(preds["GRU（生の窓）"]) if "GRU（生の窓）" in preds else np.ones(len(te), bool)
        for m, p in preds.items():
            acc.setdefault((setting, m), []).append(errors(np.where(ok, p, np.nan), te["gen_kwh"])["MAE"])
        acc.setdefault((setting, "上限2（隠れた状態まで）"), []).append(errors(np.where(ok, oracle_full(te), np.nan), te["gen_kwh"])["MAE"])
    for sd in seeds:
        for name in WORLDS:
            tr, te, va = _data(sd, name)
            add(f"{name}・学習10年・L=1", _predict_all(tr, va, te, 1, sd, cfg, ("linear", "hgb", "hgb_base", "mlp", "gru")), te)
        tr, te, va = _data(sd, "A 普通", train_years=cfg["short_years"])
        add(f"A 普通・学習{cfg['short_years']}年・L=1", _predict_all(tr, va, te, 1, sd, cfg, ("linear", "hgb", "hgb_base", "mlp", "gru")), te)
        for L in cfg["leads"][1:]:
            tr, te, va = _data(sd, "A 普通", L)
            add(f"A 普通・学習10年・L={L}", _predict_all(tr, va, te, L, sd, cfg, ("linear", "hgb", "mlp")), te)
    rows = []
    for (setting, m), v in acc.items():
        lin = np.array(acc[(setting, "予報の直線回帰")])
        d = np.array(v) - lin
        rows.append({"設定": setting, "手法": m, "MAE": _stat(v)[0], "標準誤差": _stat(v)[1], "直線回帰との差": _stat(d)[0], "差の標準誤差": _stat(d)[1]})
    return pd.DataFrame(rows)


def part_prob(seeds, cfg, L=1):
    acc = {}
    for sd in seeds:
        tr, te, va = _data(sd, "A 普通")
        vt = pd.concat([va, te], ignore_index=True)
        y = te["gen_kwh"].to_numpy()
        p_tr, p_te = linear_fc(tr, tr, L), linear_fc(tr, te, L)
        Qs = {"線形＋一定幅": p_te[:, None] + residual_quantiles(tr["gen_kwh"].to_numpy() - p_tr, QS)[None, :],
              "HGB分位点（全特徴量）": fit_predict_quantiles(tr, vt, L, QS, seed=sd, **cfg["hgb_q"])[len(va):],
              "MLP分位点（予報・季節だけ）": mlp_fit_predict(tr, vt, L, groups=("base",), seed=sd, n_seeds=cfg["n_seeds"], quantiles=QS, **cfg["mlp"])[len(va):],
              "GRU分位点（生の窓）": seq_fit_predict("gru", tr, vt, L, seed=sd, n_seeds=cfg["n_seeds"], quantiles=QS, **cfg["gru"])[len(va):]}
        ok = ~np.isnan(Qs["GRU分位点（生の窓）"]).any(axis=1)
        for m, Q in Qs.items():
            Q = np.where(ok[:, None], Q, np.nan)
            acc.setdefault(m, []).append((pinball(y, Q, QS), coverage(y, Q[:, 0], Q[:, 8]), width(Q[:, 0], Q[:, 8])))
    lin = np.array(acc["線形＋一定幅"])[:, 0]
    rows = []
    for m, v in acc.items():
        a = np.array(v)
        d = a[:, 0] - lin
        rows.append({"手法": m, "平均pinball損失": _stat(a[:, 0])[0], "標準誤差": _stat(a[:, 0])[1], "線形との差": _stat(d)[0], "差の標準誤差": _stat(d)[1],
                     "80%区間のカバー率": _stat(a[:, 1])[0], "80%区間の幅": _stat(a[:, 2])[0]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    lock = OUT / "TEST_USED.json"
    if a.split == "test" and lock.exists():
        sys.exit(f"testは使用済みです（{lock}）。再評価が必要なら、新しいシード（4M番台など）を使ってください。")
    seeds = SPLITS[a.split]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    print(f"== {a.split}（世界のシード {seeds[0]}〜、{len(seeds)}個）==", flush=True)
    for name, fn in (("main", part_main), ("prob", part_prob)):
        df = fn(seeds, FINAL_CONFIG)
        df.to_csv(OUT / f"final_{a.split}_{name}.csv", index=False)
        print(f"\n-- {name} --", flush=True)
        print(df.to_string(index=False), flush=True)
    if a.split == "test":
        lock.write_text(json.dumps(dict(used_at=datetime.now().isoformat(timespec="seconds"), config=FINAL_CONFIG, seeds=seeds[:1],
                                        note="testは最終評価で使用済み。設定は開発用の世界で選んで固定した"), ensure_ascii=False, indent=2, default=list))
        print("\nTEST_USED.json を作成しました（以後 --split test は拒否されます）")


if __name__ == "__main__":
    main()
