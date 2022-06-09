import logging
import unittest

import numpy as np
import pandas as pd
from sktime.datasets import load_airline, load_shampoo_sales, load_lynx

from autosktime.data.benchmark.m4 import load_timeseries
from autosktime.metalearning.kND import KNearestDataSets


class kNDTest(unittest.TestCase):

    def setUp(self):
        self.airline = load_airline()
        self.shampoo_sales = load_shampoo_sales()
        self.lynx = load_lynx()
        self.logger = logging.getLogger()

    def test_kneighbors(self):
        kND = KNearestDataSets(logger=self.logger)
        kND.fit(pd.DataFrame({
            'shampoo_sales': self.shampoo_sales.reset_index(drop=True),
            'lynx': self.lynx.reset_index(drop=True)
        }))

        neighbor, _, _ = kND.kneighbors(self.airline, 1)
        self.assertEqual(['shampoo_sales'], neighbor)
        neighbor, distance, _ = kND.kneighbors(self.airline, 1)
        self.assertEqual(['shampoo_sales'], neighbor)
        np.testing.assert_array_almost_equal([146303.195], distance)

        neighbors, _, _ = kND.kneighbors(self.airline, 2)
        np.testing.assert_array_equal(['shampoo_sales', 'lynx'], neighbors)
        neighbors, distance, _ = kND.kneighbors(self.airline, 2)
        np.testing.assert_array_equal(['shampoo_sales', 'lynx'], neighbors)
        np.testing.assert_array_almost_equal([146303.195, 115793162.625], distance)

        neighbors, _, _ = kND.kneighbors(self.airline, -1)
        np.testing.assert_array_equal(['shampoo_sales', 'lynx'], neighbors)
        neighbors, distance, _ = kND.kneighbors(self.airline, -1)
        np.testing.assert_array_equal(['shampoo_sales', 'lynx'], neighbors)
        np.testing.assert_array_almost_equal([146303.195, 115793162.625], distance)

        self.assertRaises(ValueError, kND.kneighbors, self.airline, 0)
        self.assertRaises(ValueError, kND.kneighbors, self.airline, -2)

    def test_m4(self):
        kND = KNearestDataSets(logger=self.logger)

        time_series = [f'Y{i}' for i in range(1, 1000)]
        ys, _ = load_timeseries(time_series)

        kND.fit(pd.DataFrame({ts: y for ts, y in zip(time_series, ys)}))

        x, _ = load_timeseries('Y100')
        names, distances, idx = kND.kneighbors(x, 1)
        np.testing.assert_array_equal(['Y100'], names)
        np.testing.assert_array_equal([0.0], distances)
        np.testing.assert_array_equal([99], idx)

        x, _ = load_timeseries('Y1001')
        names, distances, idx = kND.kneighbors(x, 2)
        np.testing.assert_array_equal(['Y105', 'Y931'], names)
        np.testing.assert_array_equal([624054.5248466857, 653325.0], distances)
        np.testing.assert_array_equal([104, 930], idx)
