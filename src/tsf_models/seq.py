"""系列モデル（GRU・1D-CNN・窓を並べただけのMLP）。手作りの特徴量を使わず、**生の時系列の窓**から学ぶ。

入力:
  窓      : 起点日（対象日のL日前）までの W 日。1日あたり9チャンネル
            [天気（晴・曇・雨の one-hot）、日照率、日射の観測、気温、発電量、季節 sin・cos]
  静的入力: 対象日の予報 fcL、対象日の季節 sin・cos（予報は未来の情報だが、L日前に出たものなので使ってよい）
学習の手順・標準化・早期終了・アンサンブルは mlp.py と同じ（学習期間の統計だけで標準化。最後の2年を内側の検証にして
エポック数を決め、全学習期間で学び直す）。
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(4)
INNER_DAYS = 730
N_CH = 9


def channels(df: pd.DataFrame) -> np.ndarray:
    """日ごとのチャンネル（日数 × 9）"""
    w = df["weather_class"].to_numpy()
    doy = df["date"].dt.dayofyear.to_numpy()
    return np.column_stack([w == 0, w == 1, w == 2, df["sunshine_ratio"], df["gsr"], df["temp"], df["gen_kwh"],
                            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]).astype(float)


def build_windows(df: pd.DataFrame, L: int, W: int):
    """(窓 n×W×9, 静的 n×3, 有効な行の印)。対象日 t の窓は、起点日 o=t−L までの W 日（o−W+1〜o）"""
    n = len(df)
    ch = channels(df)
    seq = np.zeros((n, W, N_CH))
    valid = np.zeros(n, dtype=bool)
    for t in range(n):
        o = t - L
        if o - W + 1 >= 0:
            seq[t] = ch[o - W + 1: o + 1]
            valid[t] = True
    doy = df["date"].dt.dayofyear.to_numpy()
    stat = np.column_stack([df[f"fc{L}"], np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]).astype(float)
    return seq, stat, valid


class GRUNet(nn.Module):
    def __init__(self, hidden=32, W=30):
        super().__init__()
        self.rnn = nn.GRU(N_CH, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, stat):
        _, h = self.rnn(seq)
        return self.head(torch.cat([h[-1], stat], dim=1))


class CNNNet(nn.Module):
    def __init__(self, hidden=32, W=30):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_CH, hidden, 3, padding=1), nn.ReLU(), nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden + 3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, stat):
        z = self.conv(seq.transpose(1, 2))                              # (n, hidden, W)
        return self.head(torch.cat([z.mean(dim=2), z[:, :, -1], stat], dim=1))


class FlatMLP(nn.Module):
    """窓を1列に並べただけのMLP（時間の並びの構造を使わない）"""
    def __init__(self, hidden=32, W=30):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W * N_CH + 3, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, stat):
        return self.net(torch.cat([seq.flatten(1), stat], dim=1))


NETS = {"gru": GRUNet, "cnn": CNNNet, "flat": FlatMLP}


def _train(make, S, T, y, Sev, Tev, yev, lr, wd, batch, epochs, seed, patience=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = make()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    S_, T_, y_ = (torch.tensor(a, dtype=torch.float32) for a in (S, T, y))
    tr_loss, ev_loss, best, bad = [], [], np.inf, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(S_), generator=gen)
        tot = 0.0
        for s in range(0, len(S_), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = ((net(S_[idx], T_[idx]).squeeze(1) - y_[idx]) ** 2).mean()
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        tr_loss.append(tot / len(S_))
        if Sev is not None:
            net.eval()
            with torch.no_grad():
                e = float(((net(torch.tensor(Sev, dtype=torch.float32), torch.tensor(Tev, dtype=torch.float32)).squeeze(1)
                            - torch.tensor(yev, dtype=torch.float32)) ** 2).mean())
            ev_loss.append(e)
            if e < best - 1e-5:
                best, bad = e, 0
            else:
                bad += 1
                if patience is not None and bad >= patience:
                    break
    return net, tr_loss, ev_loss


def seq_fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame, L: int, window: int = 30, hidden: int = 32,
                    wd: float = 1e-4, lr: float = 2e-3, batch: int = 128, max_epochs: int = 100, patience: int = 10,
                    seed: int = 0, n_seeds: int = 1, epochs=None, return_info: bool = False):
    both = pd.concat([train, test], ignore_index=True)
    seq, stat, valid = build_windows(both, L, window)
    ntr = len(train)
    ok = valid[:ntr] & (np.arange(ntr) >= 40 + L)                     # classic・MLPと同じ学習行（先頭の40+L行は使わない）
    S, T, y = seq[:ntr][ok], stat[:ntr][ok], train["gen_kwh"].to_numpy()[ok]
    mu_s, sd_s = S.reshape(-1, N_CH).mean(0), S.reshape(-1, N_CH).std(0) + 1e-8
    mu_t, sd_t = T.mean(0), T.std(0) + 1e-8
    mu_y, sd_y = y.mean(), y.std()
    Sn, Tn, yn = (S - mu_s) / sd_s, (T - mu_t) / sd_t, (y - mu_y) / sd_y
    Se, Te = (seq[ntr:] - mu_s) / sd_s, (stat[ntr:] - mu_t) / sd_t
    make = lambda: NETS[model](hidden, window)                         # noqa: E731
    preds, info = [], {"epochs": [], "curves": []}
    for k in range(n_seeds):
        sd = seed * 1000 + k
        if epochs is None:
            cut = len(Sn) - INNER_DAYS
            _, tl, el = _train(make, Sn[:cut], Tn[:cut], yn[:cut], Sn[cut:], Tn[cut:], yn[cut:], lr, wd, batch, max_epochs, sd, patience)
            n_ep = int(np.argmin(el)) + 1
            info["curves"].append((tl, el))
        else:
            n_ep = epochs
        info["epochs"].append(n_ep)
        net, _, _ = _train(make, Sn, Tn, yn, None, None, None, lr, wd, batch, n_ep, sd)
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.tensor(Se, dtype=torch.float32), torch.tensor(Te, dtype=torch.float32)).squeeze(1).numpy() * sd_y + mu_y)
    pred = np.mean(preds, axis=0)
    pred[~valid[ntr:]] = np.nan
    return (pred, info) if return_info else pred
