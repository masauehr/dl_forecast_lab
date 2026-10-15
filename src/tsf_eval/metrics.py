"""予測の評価指標。

  MAE  : 平均絶対誤差（kWh/日）。外れ値に強い。主指標
  RMSE : 二乗平均平方根誤差。大きな外れを重く見る
  bias : 平均誤差（予測 − 実測）。系統的に高め／低めか
  skill: 1 − MAE/MAE_基準。基準（持続予報など）より何割良いか。0=同じ、正=良い、負=悪い
予測できない先頭の日（過去が足りない）は nan になるので、評価から除く。
"""
import numpy as np
import pandas as pd


def errors(pred, actual) -> dict:
    pred, actual = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    ok = ~np.isnan(pred) & ~np.isnan(actual)
    e = pred[ok] - actual[ok]
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()), "n": int(ok.sum())}


def skill(mae: float, mae_ref: float) -> float:
    """基準との比べ方。1 − MAE/MAE基準"""
    return 1.0 - mae / mae_ref


def breakdown(pred, df: pd.DataFrame, by: str = "weather") -> pd.DataFrame:
    """区分ごとの誤差。by='weather'（対象日の天気）／'season'（季節）／'month'"""
    pred = np.asarray(pred, dtype=float)
    if by == "weather":
        key = df["weather_class"].map({0: "晴", 1: "曇", 2: "雨"}).to_numpy()
    elif by == "season":
        key = pd.Series(df["date"].dt.month.to_numpy()).map(lambda m: "春" if m in (3, 4, 5) else "夏" if m in (6, 7, 8) else "秋" if m in (9, 10, 11) else "冬").to_numpy()
    else:
        key = df["date"].dt.month.to_numpy()
    rows = []
    for k in pd.unique(key):
        m = key == k
        e = errors(pred[m], df["gen_kwh"].to_numpy()[m])
        rows.append({by: k, "MAE": round(e["MAE"], 3), "RMSE": round(e["RMSE"], 3), "bias": round(e["bias"], 3), "n": e["n"]})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)
