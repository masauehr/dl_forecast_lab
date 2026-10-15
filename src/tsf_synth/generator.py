"""合成データ生成器（日次の太陽光発電量）。

発電量は、日射量から決まる「単純な部分」と、**隠れた状態**（見えない部分）でできている:

    発電量 = 0.85 × 真の日射量 × 温度の低下 × (1 − よごれ) × (1 + ノイズ3%)
      温度の低下 : パネルが熱いと効率が下がる（非線形。25℃を超えた分に比例）
      よごれ     : 晴れが続くとパネルが少しずつよごれ、雨の日に洗い流される（隠れた状態。直接は見えない。過去の天気の並びから推測できる）

予報: 日射量の予報は、先の日ほど不正確（リード L=1〜5 日前に出した予報）。リードが長いほど、平年並みの値に寄っていく。

予測の課題（リードL）: 「L日前の夜までの観測」と「L日前に出た、対象日の日射予報」から、対象日の発電量を当てる。
手法に渡してよい列: HISTORY_COLUMNS（過去の観測）と fc{L}（予報）。ORACLE_COLUMNS（真の日射量・よごれなど）は答えなので渡さない。
"""
import numpy as np
import pandas as pd

KWH_PER_MJ = 0.85                                   # 日射量(MJ/m2)→発電量(kWh)の換算（約4.5kWp）
FORECAST_LEADS = (1, 2, 3, 4, 5)                    # 予報のリード（日前）
HISTORY_COLUMNS = ["weather_class", "sunshine_ratio", "gsr", "temp", "gen_kwh"]   # 過去について、観測できる列
ORACLE_COLUMNS = ["gsr_true", "soiling", "derate"]  # 答えそのもの。手法に渡してはいけない
SPLIT_YEARS = {"train": 10, "val": 2, "test": 2}    # 時間順に、train → val → test
START = "2008-01-01"


def make_series(seed: int, n_years: int = 14, soiling_rate: float = 0.006, soiling_cap: float = 0.20,
                temp_coef: float = 0.004, gsr_noise: float = 0.04, gen_noise: float = 0.03) -> pd.DataFrame:
    """1つの世界の、長い日次系列を作る（同じseedなら同じ系列）。

    soiling_rate: 晴れ・曇りの日に増えるよごれ（1日あたり）。0にすると隠れた状態が消える（感度分析用）
    temp_coef   : 温度による低下の強さ。0にすると温度の影響が消える
    """
    rng = np.random.default_rng(seed)
    n = int(round(365.25 * n_years))
    dates = pd.date_range(START, periods=n, freq="D")
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    season = np.sin(2 * np.pi * (doy - 80) / 365.25)
    clear_gsr = 18.0 + 8.0 * season                                  # 晴天時の日射量（冬約10、夏約26）

    # 天気: 0=晴 1=曇 2=雨（マルコフ連鎖。5〜7月は雨が増える）
    rain_boost = np.where((month >= 5) & (month <= 7), 0.15, 0.0)
    weather = np.zeros(n, dtype=int)
    for i in range(1, n):
        prev = weather[i - 1]
        p_rain = np.clip({0: 0.10, 1: 0.25, 2: 0.45}[prev] + rain_boost[i], 0, 0.9)
        p_cloud = {0: 0.25, 1: 0.45, 2: 0.30}[prev]
        u = rng.random()
        weather[i] = 2 if u < p_rain else (1 if u < p_rain + p_cloud else 0)
    base = np.array([0.92, 0.55, 0.20])[weather]
    spread = np.array([0.04, 0.12, 0.08])[weather]
    ratio = np.clip(base + spread * rng.standard_normal(n), 0.03, 1.0)
    gsr_true = clear_gsr * ratio

    # 気温（日平均）: 季節＋日射が多い日は高め＋ゆるやかに変わる揺らぎ
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 1.2)
    temp = 17.0 + 10.0 * season + 2.5 * (ratio - 0.6) + ar

    # 温度による低下（パネルは気温より熱い。25℃を超えた分に比例）
    derate = 1.0 - temp_coef * np.maximum(temp + 18.0 * ratio - 25.0, 0.0)

    # よごれ（隠れた状態）: 雨以外の日に少しずつ増え、雨の日に大半が洗い流される
    soil = np.zeros(n)
    for i in range(1, n):
        s = soil[i - 1] + soiling_rate
        s = s * 0.35 if weather[i] == 2 else s
        soil[i] = min(s, soiling_cap)

    gen = KWH_PER_MJ * gsr_true * derate * (1.0 - soil) * (1.0 + gen_noise * rng.standard_normal(n))
    gsr_obs = gsr_true * (1.0 + gsr_noise * rng.standard_normal(n))
    sunshine = np.clip(ratio + 0.05 * rng.standard_normal(n), 0, 1)

    cols = {"date": dates, "weather_class": weather, "sunshine_ratio": sunshine.round(3), "gsr": gsr_obs.round(2),
            "temp": temp.round(1), "gen_kwh": np.round(gen, 2)}
    # 日射の予報: リードが長いほど、平年並み（晴天比0.62）に寄り、誤差が増える
    w = {1: 0.90, 2: 0.80, 3: 0.65, 4: 0.50, 5: 0.40}
    for L in FORECAST_LEADS:
        r = np.clip(w[L] * ratio + (1 - w[L]) * 0.62 + (0.05 + 0.02 * L) * rng.standard_normal(n), 0.03, 1.0)
        cols[f"fc{L}"] = (clear_gsr * r).round(2)
    cols.update({"gsr_true": gsr_true.round(2), "soiling": soil.round(4), "derate": derate.round(4)})
    return pd.DataFrame(cols)


def split_series(df: pd.DataFrame, train_years: int | None = None) -> dict:
    """時間順に train → val → test に分ける（未来を学習に使わない）。
    train_years を指定すると、train の「新しい側」だけを使う（データ量の実験用。valの直前までの k 年）"""
    n_tr = int(round(365.25 * SPLIT_YEARS["train"]))
    n_va = int(round(365.25 * SPLIT_YEARS["val"]))
    tr, va, te = df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_va], df.iloc[n_tr + n_va:]
    if train_years is not None:
        tr = tr.iloc[max(0, n_tr - int(round(365.25 * train_years))):]
    return {"train": tr.reset_index(drop=True), "val": va.reset_index(drop=True), "test": te.reset_index(drop=True)}
