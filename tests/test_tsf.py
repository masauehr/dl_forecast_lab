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
