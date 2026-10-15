"""古典的な機械学習の手法（scikit-learn）。特徴量の作成・標準化は学習期間だけの統計で行う。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .features import build_features


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame, L: int, groups=("base", "gen", "weather", "dry"),
                seed: int = 0, **params) -> np.ndarray:
    """train で学習し、test の予測を返す（学習・評価に使えない先頭の行は nan）。
    model: 'ridge' | 'rf' | 'hgb'。test の特徴量は、test 自身の過去（と、直前の履歴）から作る
    ※ test の先頭の行は、直前の train の履歴が必要。ここでは train＋test をつなげて特徴量を作り、test の部分だけ返す"""
    both = pd.concat([train, test], ignore_index=True)
    X = build_features(both, L, groups)
    Xtr, Xte = X.iloc[: len(train)], X.iloc[len(train):]
    ytr = train["gen_kwh"].to_numpy()
    ok = ~Xtr.isna().any(axis=1).to_numpy()
    if model == "ridge":
        sc = StandardScaler().fit(Xtr[ok])
        m = Ridge(alpha=params.get("alpha", 1.0)).fit(sc.transform(Xtr[ok]), ytr[ok])
        pred = m.predict(sc.transform(Xte.fillna(0)))
    elif model == "rf":
        m = RandomForestRegressor(n_estimators=params.get("n_estimators", 200), min_samples_leaf=params.get("min_samples_leaf", 3),
                                  max_features=params.get("max_features", 0.5), n_jobs=-1, random_state=seed).fit(Xtr[ok], ytr[ok])
        pred = m.predict(Xte.fillna(0))
    elif model == "hgb":
        m = HistGradientBoostingRegressor(max_iter=params.get("max_iter", 300), learning_rate=params.get("learning_rate", 0.05),
                                          max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 20),
                                          l2_regularization=params.get("l2", 1.0), random_state=seed).fit(Xtr[ok], ytr[ok])
        pred = m.predict(Xte.fillna(0))
    else:
        raise ValueError(model)
    pred = pred.astype(float)
    pred[Xte.isna().any(axis=1).to_numpy()] = np.nan
    return pred


def fit_predict_quantiles(train: pd.DataFrame, test: pd.DataFrame, L: int, qs, groups=("base", "gen", "weather", "dry"),
                          seed: int = 0, **params) -> np.ndarray:
    """勾配ブースティングの分位点回帰（分位点ごとに別のモデル）。(n, 分位点の数)。小さい順に並べ替える（交差の防止）"""
    both = pd.concat([train, test], ignore_index=True)
    X = build_features(both, L, groups)
    Xtr, Xte = X.iloc[: len(train)], X.iloc[len(train):]
    ytr = train["gen_kwh"].to_numpy()
    ok = ~Xtr.isna().any(axis=1).to_numpy()
    out = np.zeros((len(test), len(qs)))
    for j, q in enumerate(qs):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=params.get("max_iter", 300), learning_rate=params.get("learning_rate", 0.05),
                                          max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 20),
                                          l2_regularization=params.get("l2", 1.0), random_state=seed).fit(Xtr[ok], ytr[ok])
        out[:, j] = m.predict(Xte.fillna(0))
    out = np.sort(out, axis=1)
    out[Xte.isna().any(axis=1).to_numpy()] = np.nan
    return out
