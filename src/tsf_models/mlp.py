"""ニューラルネット（MLP）。PyTorch。特徴量は classic と同じ（build_features）。

学習の手順（検証データを、評価に使い回さないための2段階）:
  1) 学習期間の最後の2年を「内側の検証」として取り分け、残りで学習 → 内側の検証の誤差が最小のエポック数を決める（早期終了）
  2) 学習期間の全部で、そのエポック数だけ学習し直す → test（評価用）の予測
標準化（平均・標準偏差）は、学習期間の統計だけで行う。目的変数も標準化して学習し、予測は元の単位に戻す。
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import build_features
from .losses import loss_fn

torch.set_num_threads(4)
INNER_DAYS = 730


def _net(n_in: int, hidden: tuple, dropout: float, n_out: int = 1) -> nn.Module:
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def _train(Xtr, ytr, Xev, yev, hidden, dropout, wd, lr, batch, epochs, seed, patience=None, qs=None):
    """学習して、(モデル, エポックごとの学習損失, 評価損失) を返す。Xev があれば毎エポック評価し、patience で早期終了"""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = _net(Xtr.shape[1], hidden, dropout, 1 if qs is None else len(qs))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    tr_loss, ev_loss, best, bad = [], [], np.inf, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tot = 0.0
        for s in range(0, len(Xt), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx], qs)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        tr_loss.append(tot / len(Xt))
        if Xev is not None:
            net.eval()
            with torch.no_grad():
                e = float(loss_fn(net(torch.tensor(Xev, dtype=torch.float32)), torch.tensor(yev, dtype=torch.float32), qs))
            ev_loss.append(e)
            if e < best - 1e-5:
                best, bad = e, 0
            else:
                bad += 1
                if patience is not None and bad >= patience:
                    break
    return net, tr_loss, ev_loss


def mlp_fit_predict(train: pd.DataFrame, test: pd.DataFrame, L: int, groups=("base", "gen", "weather", "dry"), hidden=(64, 64),
                    dropout=0.0, wd=1e-4, lr=1e-3, batch=128, max_epochs=200, patience=20, seed=0, n_seeds=1,
                    standardize=True, epochs=None, return_info=False, quantiles=None):
    """train で学習し、test の予測を返す。epochs を指定すると、早期終了を使わずその回数だけ学習する。
    return_info=True なら (予測, 情報) を返す（情報: 選ばれたエポック数、学習曲線）
    quantiles（例 [0.1, 0.5, 0.9]）を指定すると、pinball loss で学習し、(n, 分位点の数) の予測を返す（小さい順に並べ替え済み）"""
    qs = None if quantiles is None else torch.tensor(quantiles, dtype=torch.float32)
    both = pd.concat([train, test], ignore_index=True)
    X = build_features(both, L, groups)
    Xtr, Xte = X.iloc[: len(train)], X.iloc[len(train):]
    ok = ~Xtr.isna().any(axis=1).to_numpy()
    Xtr_v, ytr_v = Xtr[ok].to_numpy(dtype=float), train["gen_kwh"].to_numpy()[ok]
    mu_x, sd_x = (Xtr_v.mean(0), Xtr_v.std(0) + 1e-8) if standardize else (0.0, 1.0)
    mu_y, sd_y = ytr_v.mean(), ytr_v.std()
    f = lambda a: (a - mu_x) / sd_x                                  # noqa: E731
    Xs, ys = f(Xtr_v), (ytr_v - mu_y) / sd_y
    Xe = f(Xte.fillna(0).to_numpy(dtype=float))
    preds, info = [], {"epochs": [], "curves": []}
    for k in range(n_seeds):
        sd = seed * 1000 + k
        if epochs is None:
            cut = len(Xs) - min(INNER_DAYS, len(Xs) // 4)           # 学習期間が短いときは、内側の検証を学習期間の1/4までにする
            _, tl, el = _train(Xs[:cut], ys[:cut], Xs[cut:], ys[cut:], hidden, dropout, wd, lr, batch, max_epochs, sd, patience, qs)
            n_ep = int(np.argmin(el)) + 1
            info["curves"].append((tl, el))
        else:
            n_ep = epochs
        info["epochs"].append(n_ep)
        net, _, _ = _train(Xs, ys, None, None, hidden, dropout, wd, lr, batch, n_ep, sd, None, qs)
        net.eval()
        with torch.no_grad():
            o = net(torch.tensor(Xe, dtype=torch.float32)).numpy() * sd_y + mu_y
            preds.append(o[:, 0] if qs is None else np.sort(o, axis=1))
    pred = np.mean(preds, axis=0)
    pred[Xte.isna().any(axis=1).to_numpy()] = np.nan
    return (pred, info) if return_info else pred
