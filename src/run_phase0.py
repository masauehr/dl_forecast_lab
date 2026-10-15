"""Phase0: 合成データと基準。

E1 データの様子（隠れた状態の大きさ・予報の誤差のリード依存）
E2 基準の比較（リード L=1〜5。10世界の平均±標準誤差）
E3 基準の誤差の内訳（天気別・季節別）

使い方: .venv/bin/python src/run_phase0.py
出力:   outputs/phase0_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsf_eval import (breakdown, climatology, errors, linear_fc, oracle_full, oracle_physical, persistence,  # noqa: E402
                      persistence_week, physical_fc, skill)
from tsf_synth import KWH_PER_MJ, make_series, split_series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]


def e1(seed=SEEDS[0]):
    df = make_series(seed)
    sp = split_series(df)
    rows = [{"項目": "系列の長さ（日）", "値": len(df)}, {"項目": "train／val／test（日）", "値": f"{len(sp['train'])}／{len(sp['val'])}／{len(sp['test'])}"},
            {"項目": "発電量の平均 kWh/日", "値": round(df.gen_kwh.mean(), 2)},
            {"項目": "晴／曇／雨の割合", "値": "／".join(f"{(df.weather_class == k).mean():.2f}" for k in range(3))},
            {"項目": "よごれの平均（％）", "値": round(100 * df.soiling.mean(), 2)}, {"項目": "よごれの最大（％）", "値": round(100 * df.soiling.max(), 1)},
            {"項目": "温度による低下の平均（％）", "値": round(100 * (1 - df.derate.mean()), 2)}, {"項目": "温度による低下の最大（％）", "値": round(100 * (1 - df.derate.min()), 1)}]
    for L in range(1, 6):
        e = errors(KWH_PER_MJ * df[f"fc{L}"], KWH_PER_MJ * df["gsr_true"])
        rows.append({"項目": f"リード{L}の予報の誤差 MAE（kWh換算）", "値": round(e["MAE"], 3)})
    return pd.DataFrame(rows)


def e2(seeds=SEEDS, leads=(1, 2, 3, 4, 5)):
    acc = {}
    for sd in seeds:
        df = make_series(sd)
        sp = split_series(df)
        tr, va = sp["train"], sp["val"]
        for L in leads:
            ms = {"持続予報（L日前の値）": persistence(va, L), "持続予報（7日平均）": persistence_week(va, L), "平年値（日付ごとの平均）": climatology(tr, va),
                  "予報×0.85（学習なし）": physical_fc(va, L), "予報の直線回帰": linear_fc(tr, va, L),
                  "上限1（真の日射量×0.85）": oracle_physical(va), "上限2（温度・よごれまで知っている）": oracle_full(va)}
            for name, p in ms.items():
                acc.setdefault((name, L), []).append(errors(p, va["gen_kwh"])["MAE"])
    rows = []
    for (name, L), v in acc.items():
        ref = np.array(acc[("持続予報（L日前の値）", L)])
        rows.append({"手法": name, "リードL": L, "MAE": round(float(np.mean(v)), 3), "標準誤差": round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 3),
                     "持続予報に対するスキル": round(float(1 - np.mean(v) / ref.mean()), 3)})
    return pd.DataFrame(rows)


def e3(seed=SEEDS[0], L=1):
    df = make_series(seed)
    sp = split_series(df)
    tr, va = sp["train"], sp["val"]
    out = []
    for name, p in {"予報の直線回帰": linear_fc(tr, va, L), "上限1（真の日射量×0.85）": oracle_physical(va)}.items():
        for by in ("weather", "season"):
            b = breakdown(p, va, by)
            b.insert(0, "手法", name); b.insert(1, "区分", by); b = b.rename(columns={by: "値"})
            out.append(b)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3)]:
        df = fn()
        print(f"\n== {name} ==")
        print(df.to_string(index=False))
        df.to_csv(ROOT / "outputs" / f"phase0_{name}.csv", index=False)
