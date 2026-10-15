"""特徴量の作成（リード L）。**リークしない**ことが最重要。

予測の起点は「対象日 t の L日前の夜」（=起点日 o = t − L）。使ってよいのは:
  ・起点日までの観測（天気・日照率・日射・気温・発電量）
  ・対象日 t の予報 fcL（L日前に出たもの）と、対象日の日付（カレンダーは未来でも分かる）
対象日 t の観測（天気・気温・発電量）は使わない。ORACLE_COLUMNS も使わない。

特徴量のグループ（どれを足すと効くかを、比べるために分ける）:
  base    : fcL、対象日の季節（sin, cos）
  gen     : 起点日までの発電量（ラグ0〜6、7日・30日の平均）、**性能比**（発電量 ÷ (0.85×日射の観測)。温度とよごれの影響が出る）
  weather : 起点日の天気・日照率・日射・気温、気温の7日平均
  dry     : 起点日までの「雨の降らない日の連続日数」、直近14日の雨の割合（よごれの手がかり）
"""
import numpy as np
import pandas as pd

from tsf_synth import KWH_PER_MJ

FEATURE_GROUPS = ("base", "gen", "weather", "dry")
WARMUP = 40                                           # 30日平均などに必要な先頭の日数。この日数ぶんの行は捨てる


def _dry_days(weather: np.ndarray) -> np.ndarray:
    """各日の時点での「雨の降らない日の連続日数」（その日が雨なら0）"""
    out = np.zeros(len(weather))
    for i in range(1, len(weather)):
        out[i] = 0 if weather[i] == 2 else out[i - 1] + 1
    return out


def build_features(df: pd.DataFrame, L: int, groups=FEATURE_GROUPS) -> pd.DataFrame:
    """リードLの特徴量。行は df と同じ並び。先頭 WARMUP+L 行は欠損（評価・学習から除く）"""
    f = {}
    doy = df["date"].dt.dayofyear.to_numpy()
    if "base" in groups:
        f["fc"] = df[f"fc{L}"].to_numpy()
        f["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        f["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    gen, gsr = df["gen_kwh"], df["gsr"]
    pr = (gen / (KWH_PER_MJ * gsr.clip(lower=0.5))).clip(upper=1.5)          # 性能比（起点日の値）
    if "gen" in groups:
        for k in range(7):
            f[f"gen_lag{k}"] = gen.shift(L + k).to_numpy()                   # 起点日=t−L をラグ0とする
        f["gen_mean7"] = gen.rolling(7).mean().shift(L).to_numpy()
        f["gen_mean30"] = gen.rolling(30).mean().shift(L).to_numpy()
        f["pr_lag0"] = pr.shift(L).to_numpy()
        f["pr_mean3"] = pr.rolling(3).mean().shift(L).to_numpy()
        f["pr_mean7"] = pr.rolling(7).mean().shift(L).to_numpy()
    if "weather" in groups:
        for k in range(3):
            f[f"weather_lag{k}"] = df["weather_class"].shift(L + k).to_numpy()
        f["sunshine_lag0"] = df["sunshine_ratio"].shift(L).to_numpy()
        f["gsr_lag0"] = gsr.shift(L).to_numpy()
        f["temp_lag0"] = df["temp"].shift(L).to_numpy()
        f["temp_mean7"] = df["temp"].rolling(7).mean().shift(L).to_numpy()
    if "dry" in groups:
        f["dry_days"] = pd.Series(_dry_days(df["weather_class"].to_numpy())).shift(L).to_numpy()
        f["rain_frac14"] = (df["weather_class"] == 2).astype(float).rolling(14).mean().shift(L).to_numpy()
    X = pd.DataFrame(f, index=df.index)
    X.iloc[: WARMUP + L] = np.nan
    return X
