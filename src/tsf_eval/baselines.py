"""非機械学習の基準（まず、これを作る）。学習の価値は、基準との比で測る。

リード L の予測は、「L日前の夜までの観測」と「L日前に出た予報 fc{L}」だけを使う。
  persistence      : L日前の発電量をそのまま使う（持続予報）
  persistence_week : L日前までの7日間の平均
  climatology      : 同じ日付（前後7日）の、学習期間の平均
  physical_fc      : 予報の日射量 × 0.85（物理の換算だけ。学習なし）
  linear_fc        : 予報の日射量に対する直線回帰（学習期間で当てはめる）
  oracle_physical  : 真の日射量 × 0.85（予報が完璧だったら。温度・よごれは無視）
  oracle_full      : 真の日射量・温度の低下・よごれまで知っていたら（ノイズ3%だけが残る。上限）
"""
import numpy as np
import pandas as pd

from tsf_synth import KWH_PER_MJ


def persistence(df: pd.DataFrame, L: int) -> np.ndarray:
    return df["gen_kwh"].shift(L).to_numpy()


def persistence_week(df: pd.DataFrame, L: int) -> np.ndarray:
    return df["gen_kwh"].rolling(7).mean().shift(L).to_numpy()


def climatology(train: pd.DataFrame, df: pd.DataFrame, window: int = 7) -> np.ndarray:
    """日付（1〜366）ごとの学習期間の平均。前後 window 日で滑らかにする"""
    doy = train["date"].dt.dayofyear.to_numpy()
    mean = np.array([train["gen_kwh"].to_numpy()[np.abs(((doy - d + 183) % 366) - 183) <= window].mean() for d in range(1, 367)])
    return mean[df["date"].dt.dayofyear.to_numpy() - 1]


def physical_fc(df: pd.DataFrame, L: int) -> np.ndarray:
    return KWH_PER_MJ * df[f"fc{L}"].to_numpy()


def linear_fc(train: pd.DataFrame, df: pd.DataFrame, L: int) -> np.ndarray:
    b, a = np.polyfit(train[f"fc{L}"].to_numpy(), train["gen_kwh"].to_numpy(), 1)
    return a + b * df[f"fc{L}"].to_numpy()


def oracle_physical(df: pd.DataFrame) -> np.ndarray:
    return KWH_PER_MJ * df["gsr_true"].to_numpy()


def oracle_full(df: pd.DataFrame) -> np.ndarray:
    return KWH_PER_MJ * df["gsr_true"].to_numpy() * df["derate"].to_numpy() * (1.0 - df["soiling"].to_numpy())
