"""確率予測の評価。予測を分位点 Q[:, j]（分位点 qs[j]）で表す。

  平均pinball損失 : すべての分位点で平均した分位点損失。予測分布の良さ（小さいほど良い。CRPSの約半分）
  カバー率       : 実測が [下側, 上側] の区間に入る割合。80%区間（10%点〜90%点）なら、0.8に近いほど「正直」
  区間の幅       : 狭いほど情報がある。ただしカバー率が足りないなら、狭いだけでは意味がない（先にカバー率）
  校正の表       : 実測が各分位点以下だった割合。分位点 τ なら τ に近いほど良い
  CQR            : 校正用データで、区間の幅を足し引きしてカバー率を合わせる（共形予測の一種）
"""
import numpy as np


def pinball(y, Q, qs) -> float:
    y, Q, qs = np.asarray(y, dtype=float), np.asarray(Q, dtype=float), np.asarray(qs, dtype=float)
    ok = ~np.isnan(Q).any(axis=1) & ~np.isnan(y)
    d = y[ok, None] - Q[ok]
    return float(np.maximum(qs * d, (qs - 1.0) * d).mean())


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    ok = ~np.isnan(lo) & ~np.isnan(hi) & ~np.isnan(y)
    return float(((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])).mean())


def width(lo, hi) -> float:
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ok = ~np.isnan(lo) & ~np.isnan(hi)
    return float((hi[ok] - lo[ok]).mean())


def calibration(y, Q, qs):
    """各分位点 τ について、実測が予測以下だった割合"""
    y, Q = np.asarray(y, dtype=float), np.asarray(Q, dtype=float)
    ok = ~np.isnan(Q).any(axis=1) & ~np.isnan(y)
    return (y[ok, None] <= Q[ok]).mean(axis=0)


def residual_quantiles(resid, qs):
    """点予測の残差（予測−実測の符号を逆にした 実測−予測）の経験分位点。一定幅の区間を作る基準"""
    return np.quantile(np.asarray(resid, dtype=float), qs)


def cqr_shift(y_cal, lo_cal, hi_cal, alpha: float = 0.2) -> float:
    """CQR: 校正データでの適合スコア max(lo−y, y−hi) の、(1−α) 分位点。
    これを区間の両側に足す（正なら広げる、負なら狭める）と、校正データでのカバー率がほぼ 1−α になる"""
    y_cal, lo_cal, hi_cal = (np.asarray(a, dtype=float) for a in (y_cal, lo_cal, hi_cal))
    s = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
