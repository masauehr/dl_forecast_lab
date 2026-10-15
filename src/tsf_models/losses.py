"""損失関数。点予測は二乗誤差、確率予測は pinball loss（分位点損失）。

pinball loss（分位点 τ）: 予測 q、実測 y のとき  max(τ(y − q), (τ − 1)(y − q))
  τ=0.9 なら、「実測が予測を上回る」ことを重く罰し、予測は実測の上位10%の位置に寄る。
  すべての分位点で平均した値は、予測分布の良さ（CRPS の近似）になる。
"""
import torch


def loss_fn(out: torch.Tensor, y: torch.Tensor, qs=None) -> torch.Tensor:
    """out: (n, 1) か (n, Q)、y: (n,)、qs: 分位点の tensor (Q,) または None（二乗誤差）"""
    if qs is None:
        return ((out.squeeze(1) - y) ** 2).mean()
    d = y[:, None] - out
    return torch.maximum(qs * d, (qs - 1.0) * d).mean()
