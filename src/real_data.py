"""Phase6: 自宅の実データ（日次発電量＋気象）を、合成データと同じ形に変換する。

🔒 個人データ（自宅の発電量）の扱い:
  - ml_forecast/data/ を**読み取り専用**で参照する。このリポジトリへコピーしない。
  - 結果は outputs/real/（git管理外）にだけ書く。コミットしない。Notebookには日付や個別の値を出さない。
予報について: 実データには、長い期間の「L日前の予報」が無い。そこで、対象日の農研機構メッシュ農業気象データの日射量（GSR。解析値に近い）を
**予報の代わり（ほぼ完璧な予報）**として fc1 に入れる。これは合成データの「世界B（予報が完璧）」に相当する診断用の設定で、実運用の予報ではない。
"""
from pathlib import Path

import pandas as pd

REAL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_forecast" / "data"
WEATHER_MAP = {"快晴": 0, "晴れ": 0, "薄曇り〜曇り": 1, "曇り・雨": 2}           # 合成データの 0=晴 1=曇 2=雨 に合わせる


def load_real(base: Path = REAL_DIR) -> pd.DataFrame:
    daily = pd.read_csv(base / "hatuden_daily.csv", parse_dates=["date"])
    amd = pd.read_csv(base / "hatuden_amd.csv", parse_dates=["date"])
    m = daily.merge(amd, on="date", how="inner").sort_values("date").reset_index(drop=True)
    unknown = set(m["weather_class"].dropna()) - set(WEATHER_MAP)
    if unknown:
        raise ValueError(f"未知の天気分類: {unknown}")
    gaps = m["date"].diff().dropna().dt.days
    if (gaps != 1).any():
        raise ValueError("日付が連続していません（ラグ特徴量がずれるので、先に欠測を処理すること）")
    return pd.DataFrame({
        "date": m["date"], "weather_class": m["weather_class"].map(WEATHER_MAP).astype(int),
        "sunshine_ratio": (m["sunshine_ratio"] / 100.0).clip(0, 1), "gsr": m["GSR"],
        "temp": m["TMP_max"],                                                  # 日平均気温は欠測が多いので、日最高気温を使う
        "gen_kwh": m["generation_kwh"], "fc1": m["GSR"],                       # fc1 = 予報の代わり（ほぼ完璧な予報）
    })


def split_real(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)
