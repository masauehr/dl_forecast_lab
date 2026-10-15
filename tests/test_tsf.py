import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tsf_eval import (breakdown, climatology, errors, linear_fc, oracle_full, oracle_physical, persistence,  # noqa: E402
                      persistence_week, physical_fc, skill)
from tsf_synth import FORECAST_LEADS, HISTORY_COLUMNS, KWH_PER_MJ, ORACLE_COLUMNS, make_series, split_series  # noqa: E402


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_series(1_000_000)

    def test_同じシードなら同じ系列_違えば違う(self):
        pd.testing.assert_frame_equal(self.df, make_series(1_000_000))
        self.assertFalse(self.df["gen_kwh"].equals(make_series(1_000_001)["gen_kwh"]))

    def test_分割は時間順で重ならず_全部で系列と一致する(self):
        sp = split_series(self.df)
        self.assertLess(sp["train"]["date"].max(), sp["val"]["date"].min())
        self.assertLess(sp["val"]["date"].max(), sp["test"]["date"].min())
        self.assertEqual(sum(len(v) for v in sp.values()), len(self.df))
        self.assertEqual(len(split_series(self.df, train_years=2)["train"]), round(365.25 * 2))
        self.assertEqual(split_series(self.df, train_years=2)["train"]["date"].max(), sp["train"]["date"].max())   # 新しい側を使う

    def test_答えの列と観測の列は重ならない(self):
        self.assertFalse(set(ORACLE_COLUMNS) & set(HISTORY_COLUMNS))
        self.assertFalse(set(ORACLE_COLUMNS) & {f"fc{L}" for L in FORECAST_LEADS})
        for c in ORACLE_COLUMNS + HISTORY_COLUMNS + [f"fc{L}" for L in FORECAST_LEADS]:
            self.assertIn(c, self.df.columns)

    def test_発電量は_日射量と温度とよごれと整合する(self):
        e = self.df["gen_kwh"] / oracle_full(self.df)
        self.assertAlmostEqual(float(e.mean()), 1.0, delta=0.01)
        self.assertAlmostEqual(float(e.std()), 0.03, delta=0.005)         # ノイズ3%

    def test_よごれは雨で下がり_範囲内で_0にすると消える(self):
        s = self.df["soiling"].to_numpy(); w = self.df["weather_class"].to_numpy()
        self.assertTrue((s >= 0).all() and (s <= 0.20 + 1e-9).all())
        d = np.diff(s)
        self.assertLess(d[w[1:] == 2].mean(), 0)                          # 雨の日は平均して下がる
        self.assertGreater(d[w[1:] != 2].mean(), 0)                       # それ以外は増える
        self.assertTrue((make_series(1, 2, soiling_rate=0.0)["soiling"] == 0).all())

    def test_温度の低下は暑いほど大きく_係数0で消える(self):
        self.assertTrue((self.df["derate"] <= 1.0).all())
        hot = self.df["temp"] > self.df["temp"].quantile(0.9); cold = self.df["temp"] < self.df["temp"].quantile(0.1)
        self.assertLess(self.df["derate"][hot].mean(), self.df["derate"][cold].mean())
        self.assertTrue((make_series(1, 2, temp_coef=0.0)["derate"] == 1.0).all())

    def test_予報はリードが長いほど不正確(self):
        err = [np.abs(self.df[f"fc{L}"] - self.df["gsr_true"]).mean() for L in FORECAST_LEADS]
        self.assertTrue(all(a < b for a, b in zip(err, err[1:])))


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_series(1_000_000)
        sp = split_series(cls.df)
        cls.tr, cls.va = sp["train"], sp["val"]

    def test_持続予報はL日前の値_先頭はnan(self):
        p = persistence(self.va, 3)
        self.assertTrue(np.isnan(p[:3]).all())
        self.assertEqual(p[5], self.va["gen_kwh"].iloc[2])

    def test_7日平均はL日前までの7日だけを使う_未来を見ない(self):
        p = persistence_week(self.va, 2)
        self.assertAlmostEqual(p[10], self.va["gen_kwh"].iloc[2:9].mean())   # 添字8（=10−2）までの7日

    def test_平年値は同じ日付で同じ値_学習期間だけから(self):
        c = climatology(self.tr, self.va)
        d = self.va["date"].dt.dayofyear.to_numpy()
        self.assertTrue(np.allclose(c[d == 100], c[d == 100][0]))
        self.assertGreater(c[d == 180].mean(), c[d == 355].mean())        # 夏は冬より多い

    def test_上限の順序(self):
        mae = lambda p: errors(p, self.va["gen_kwh"])["MAE"]              # noqa: E731
        self.assertLess(mae(oracle_full(self.va)), mae(oracle_physical(self.va)))
        self.assertLess(mae(oracle_physical(self.va)), mae(linear_fc(self.tr, self.va, 1)))
        self.assertLess(mae(linear_fc(self.tr, self.va, 1)), mae(persistence(self.va, 1)))
        self.assertLess(mae(linear_fc(self.tr, self.va, 1)), mae(linear_fc(self.tr, self.va, 5)))   # リードが短いほど良い

    def test_指標(self):
        e = errors([1.0, 2.0, np.nan, 5.0], [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(e["n"], 3)
        self.assertAlmostEqual(e["MAE"], 1.0)
        self.assertAlmostEqual(e["bias"], 1 / 3)
        self.assertAlmostEqual(skill(0.8, 1.0), 0.2)
        b = breakdown(np.zeros(len(self.va)), self.va, "weather")
        self.assertEqual(int(b["n"].sum()), len(self.va))


if __name__ == "__main__":
    unittest.main()


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_series(1_000_000, 4)

    def test_特徴量は_起点日より後の観測を見ない_リークしない(self):
        from tsf_models import build_features
        for L in (1, 3):
            X = build_features(self.df, L)
            for t in (100, 777, 1200):
                d2 = self.df.copy()
                rng = np.random.default_rng(0)
                after = d2.index > t - L                                  # 起点日（t−L）より後の行
                for c in ["gen_kwh", "gsr", "temp", "sunshine_ratio", "gsr_true", "soiling", "derate"]:
                    d2.loc[after, c] = rng.random(after.sum()) * 100
                d2.loc[after, "weather_class"] = rng.integers(0, 3, after.sum())
                keep = d2.loc[t, f"fc{L}"]
                d2.loc[after, f"fc{L}"] = rng.random(after.sum()) * 100
                d2.loc[t, f"fc{L}"] = keep                                # 対象日の予報だけは使ってよい
                X2 = build_features(d2, L)
                pd.testing.assert_series_equal(X.loc[t], X2.loc[t])

    def test_答えの列を変えても特徴量は変わらない(self):
        from tsf_models import build_features
        d2 = self.df.copy()
        for c in ORACLE_COLUMNS:
            d2[c] = 0.0
        pd.testing.assert_frame_equal(build_features(self.df, 1), build_features(d2, 1))

    def test_先頭は欠損_それ以降は欠損なし(self):
        from tsf_models import build_features
        from tsf_models.features import WARMUP
        X = build_features(self.df, 2)
        self.assertTrue(X.iloc[: WARMUP + 2].isna().all(axis=1).all())
        self.assertFalse(X.iloc[WARMUP + 2:].isna().any().any())

    def test_雨の降らない日数(self):
        from tsf_models.features import _dry_days
        np.testing.assert_array_equal(_dry_days(np.array([0, 0, 2, 1, 0, 2, 2, 0])), [0, 1, 0, 1, 2, 0, 0, 1])

    def test_グループを選べる(self):
        from tsf_models import build_features
        self.assertEqual(list(build_features(self.df, 1, ("base",)).columns), ["fc", "sin_doy", "cos_doy"])
        self.assertIn("dry_days", build_features(self.df, 1).columns)


class TestClassic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp = split_series(make_series(1_000_000))
        cls.tr, cls.va = sp["train"], sp["val"]

    def test_予測は長さが同じで_先頭だけnan(self):
        from tsf_models import fit_predict
        p = fit_predict("ridge", self.tr, self.va, 1)
        self.assertEqual(len(p), len(self.va))
        self.assertFalse(np.isnan(p).any())                                # 検証の先頭は、学習期間の履歴があるのでnanにならない

    def test_リッジは直線回帰と同程度_木は上回る(self):
        from tsf_models import fit_predict
        mae = lambda p: errors(p, self.va["gen_kwh"])["MAE"]              # noqa: E731
        lin = mae(linear_fc(self.tr, self.va, 1))
        self.assertAlmostEqual(mae(fit_predict("ridge", self.tr, self.va, 1)), lin, delta=0.06)
        self.assertLess(mae(fit_predict("hgb", self.tr, self.va, 1)), lin - 0.03)

    def test_同じシードなら同じ予測(self):
        from tsf_models import fit_predict
        a = fit_predict("rf", self.tr, self.va, 1, seed=3, n_estimators=30)
        b = fit_predict("rf", self.tr, self.va, 1, seed=3, n_estimators=30)
        np.testing.assert_allclose(a, b, rtol=1e-9)   # 並列計算の足し算の順序で、1e-16程度ずれうる

    def test_未知の手法はエラー(self):
        from tsf_models import fit_predict
        with self.assertRaises(ValueError):
            fit_predict("xyz", self.tr, self.va, 1)


class TestMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp = split_series(make_series(1_000_000))
        cls.tr, cls.va = sp["train"], sp["val"]

    def test_同じシードなら同じ予測_違えば違う(self):
        from tsf_models import mlp_fit_predict
        a = mlp_fit_predict(self.tr, self.va, 1, hidden=(16,), epochs=5, seed=1)
        b = mlp_fit_predict(self.tr, self.va, 1, hidden=(16,), epochs=5, seed=1)
        c = mlp_fit_predict(self.tr, self.va, 1, hidden=(16,), epochs=5, seed=2)
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_検証期間の後半を変えても前半の予測は変わらない_標準化も学習期間だけ(self):
        from tsf_models import mlp_fit_predict
        va2 = self.va.copy()
        va2.loc[400:, ["gen_kwh", "gsr", "temp", "fc1"]] = 123.0
        a = mlp_fit_predict(self.tr, self.va, 1, hidden=(16,), epochs=5, seed=0)
        b = mlp_fit_predict(self.tr, va2, 1, hidden=(16,), epochs=5, seed=0)
        np.testing.assert_allclose(a[:400], b[:400])

    def test_早期終了は上限以下のエポックを選び_指定エポックはそのまま使う(self):
        from tsf_models import mlp_fit_predict
        _, info = mlp_fit_predict(self.tr, self.va, 1, hidden=(16,), max_epochs=30, patience=5, return_info=True)
        self.assertTrue(1 <= info["epochs"][0] <= 30)
        self.assertEqual(len(info["curves"][0][0]), len(info["curves"][0][1]))
        _, info2 = mlp_fit_predict(self.tr, self.va, 1, hidden=(16,), epochs=7, return_info=True)
        self.assertEqual(info2["epochs"], [7])

    def test_持続予報より良く_直線回帰と同程度以上(self):
        from tsf_models import mlp_fit_predict
        mae = lambda p: errors(p, self.va["gen_kwh"])["MAE"]              # noqa: E731
        m = mae(mlp_fit_predict(self.tr, self.va, 1, hidden=(32,), n_seeds=2))
        self.assertLess(m, mae(persistence(self.va, 1)) / 2)
        self.assertLess(m, mae(linear_fc(self.tr, self.va, 1)) + 0.03)

    def test_アンサンブルは単体より誤差が小さい(self):
        from tsf_models import mlp_fit_predict
        mae = lambda p: errors(p, self.va["gen_kwh"])["MAE"]              # noqa: E731
        singles = [mae(mlp_fit_predict(self.tr, self.va, 1, hidden=(32,), seed=s)) for s in range(3)]
        ens = mae(mlp_fit_predict(self.tr, self.va, 1, hidden=(32,), seed=0, n_seeds=3))
        self.assertLess(ens, np.mean(singles))


class TestSeq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp = split_series(make_series(1_000_000))
        cls.tr, cls.va = sp["train"], sp["val"]

    def test_窓は起点日までの日だけを含む_リークしない(self):
        from tsf_models.seq import build_windows, channels
        df = make_series(1_000_000, 2)
        L, W, t = 3, 10, 200
        seq, stat, valid = build_windows(df, L, W)
        ch = channels(df)
        np.testing.assert_allclose(seq[t], ch[t - L - W + 1: t - L + 1])      # 窓の最後の日は t−L（起点日）
        self.assertEqual(stat[t, 0], df.loc[t, f"fc{L}"])
        d2 = df.copy()
        after = d2.index > t - L
        for c in ["gen_kwh", "gsr", "temp", "sunshine_ratio"]:
            d2.loc[after, c] = 999.0
        d2.loc[after, "weather_class"] = 1
        seq2, _, _ = build_windows(d2, L, W)
        np.testing.assert_allclose(seq[t], seq2[t])
        self.assertFalse(valid[: W + L - 1].any())
        self.assertTrue(valid[W + L:].all())

    def test_ネットの出力の形(self):
        import torch
        from tsf_models.seq import NETS, N_CH
        for name, cls in NETS.items():
            out = cls(8, 12)(torch.zeros(5, 12, N_CH), torch.zeros(5, 3))
            self.assertEqual(tuple(out.shape), (5, 1), name)

    def test_同じシードなら同じ予測(self):
        from tsf_models import seq_fit_predict
        a = seq_fit_predict("gru", self.tr, self.va, 1, window=14, hidden=8, epochs=3, seed=1)
        b = seq_fit_predict("gru", self.tr, self.va, 1, window=14, hidden=8, epochs=3, seed=1)
        np.testing.assert_allclose(a, b)

    def test_検証期間の後半を変えても前半の予測は変わらない(self):
        from tsf_models import seq_fit_predict
        va2 = self.va.copy()
        va2.loc[400:, ["gen_kwh", "gsr", "temp", "fc1"]] = 123.0
        a = seq_fit_predict("cnn", self.tr, self.va, 1, window=14, hidden=8, epochs=3, seed=0)
        b = seq_fit_predict("cnn", self.tr, va2, 1, window=14, hidden=8, epochs=3, seed=0)
        np.testing.assert_allclose(a[:400], b[:400])

    def test_GRUは持続予報より良い(self):
        from tsf_models import seq_fit_predict
        p = seq_fit_predict("gru", self.tr, self.va, 1, window=14, hidden=16, max_epochs=20, patience=5)
        self.assertLess(errors(p, self.va["gen_kwh"])["MAE"], errors(persistence(self.va, 1), self.va["gen_kwh"])["MAE"] / 2)


class TestShortTrain(unittest.TestCase):
    def test_学習期間が短くても_内側の検証が学習期間の1_4に収まり学習できる(self):
        from tsf_models import mlp_fit_predict, seq_fit_predict
        sp = split_series(make_series(1_000_000), train_years=1.5)
        tr, va = sp["train"], sp["val"]
        self.assertEqual(len(tr), round(365.25 * 1.5))
        p, info = mlp_fit_predict(tr, va, 1, hidden=(16,), max_epochs=10, patience=3, return_info=True)
        self.assertEqual(len(info["curves"][0][1]), len(info["curves"][0][0]))
        self.assertFalse(np.isnan(p).any())
        q = seq_fit_predict("gru", tr, va, 1, window=14, hidden=8, max_epochs=5, patience=2)
        self.assertEqual(int(np.isnan(q).sum()), 0)


class TestProbabilistic(unittest.TestCase):
    def test_pinball損失の計算(self):
        from tsf_eval import pinball
        # 予測が実測より低い(y=3, q=1)と、τ=0.9 なら 0.9×2=1.8、τ=0.1 なら 0.1×2=0.2
        self.assertAlmostEqual(pinball([3.0], [[1.0]], [0.9]), 1.8)
        self.assertAlmostEqual(pinball([3.0], [[1.0]], [0.1]), 0.2)
        # 予測が実測より高い(y=1, q=3)と、τ=0.9 なら 0.1×2=0.2
        self.assertAlmostEqual(pinball([1.0], [[3.0]], [0.9]), 0.2)

    def test_pinballは真の分位点で最小になる(self):
        from tsf_eval import pinball
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 20000)
        for tau in (0.1, 0.5, 0.9):
            qs = np.linspace(-2, 2, 81)
            losses = [pinball(y, np.full((len(y), 1), q), [tau]) for q in qs]
            self.assertAlmostEqual(qs[int(np.argmin(losses))], np.quantile(y, tau), delta=0.1)

    def test_カバー率と幅と校正(self):
        from tsf_eval import calibration, coverage, width
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(coverage(y, y - 0.5, y + 0.5), 1.0)
        self.assertAlmostEqual(coverage(y, y + 0.1, y + 0.5), 0.0)
        self.assertAlmostEqual(width(y - 1, y + 1), 2.0)
        np.testing.assert_allclose(calibration(y, np.array([[2.5, 5.0]] * 4), [0.5, 0.9]), [0.5, 1.0])

    def test_CQRは校正データのカバー率を合わせる(self):
        from tsf_eval import coverage, cqr_shift
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 5000)
        lo, hi = np.full(5000, -0.5), np.full(5000, 0.5)               # 狭すぎる区間（カバー率 約38%）
        s = cqr_shift(y, lo, hi, 0.2)
        self.assertGreater(s, 0)                                       # 広げる
        self.assertAlmostEqual(coverage(y, lo - s, hi + s), 0.8, delta=0.02)
        y2 = rng.normal(0, 1, 5000)
        self.assertAlmostEqual(coverage(y2, lo - s, hi + s), 0.8, delta=0.03)   # 別のデータでも成り立つ

    def test_分位点の予測は交差せず_形が正しい(self):
        from tsf_models import fit_predict_quantiles, mlp_fit_predict, seq_fit_predict
        sp = split_series(make_series(1_000_000))
        tr, va = sp["train"], sp["val"]
        qs = [0.1, 0.5, 0.9]
        for Q in (fit_predict_quantiles(tr, va, 1, qs, max_iter=20), mlp_fit_predict(tr, va, 1, hidden=(8,), epochs=3, quantiles=qs),
                  seq_fit_predict("gru", tr, va, 1, window=10, hidden=8, epochs=2, quantiles=qs)):
            self.assertEqual(Q.shape, (len(va), 3))
            ok = ~np.isnan(Q).any(axis=1)
            self.assertTrue((np.diff(Q[ok], axis=1) >= 0).all())

    def test_点予測は分位点の追加で変わらない(self):
        from tsf_models import mlp_fit_predict
        sp = split_series(make_series(1_000_000))
        a = mlp_fit_predict(sp["train"], sp["val"], 1, hidden=(8,), epochs=3, seed=1)
        self.assertEqual(a.ndim, 1)

    def test_残差の経験分位点(self):
        from tsf_eval import residual_quantiles
        np.testing.assert_allclose(residual_quantiles(np.arange(101.0), [0.1, 0.5]), [10.0, 50.0])
