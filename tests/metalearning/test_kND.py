import logging
import unittest

import numpy as np
import pandas as pd
from sktime.datasets import load_airline, load_shampoo_sales, load_lynx

from autosktime.data.benchmark.m4 import load_timeseries
from autosktime.metalearning.kND import KNearestDataSets


class kNDTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.airline = load_airline()
        self.shampoo_sales = load_shampoo_sales()
        self.lynx = load_lynx()

        self.runs = pd.DataFrame({
            'airline': [0.1, 0.5, 0.7],
            'shampoo_sales': [np.NaN, 0.1, 0.7],
            'lynx': [0.5, 0.7, 0.1]
        })
        self.logger = logging.getLogger()

    def test_fit_l1_distance(self):
        kND = KNearestDataSets(logger=self.logger)

        kND.fit(pd.DataFrame({
            'airline': self.airline.reset_index(drop=True),
            'shampoo_sales': self.shampoo_sales.reset_index(drop=True),
            'lynx': self.lynx.reset_index(drop=True)
        }), self.runs)
        self.assertEqual(kND.best_configuration_per_dataset_['airline'], 0)
        self.assertEqual(kND.best_configuration_per_dataset_['shampoo_sales'], 1)
        self.assertEqual(kND.best_configuration_per_dataset_['lynx'], 2)

    def test_kneighbors(self):
        kND = KNearestDataSets(logger=self.logger)
        kND.fit(pd.DataFrame({
            'shampoo_sales': self.shampoo_sales.reset_index(drop=True),
            'lynx': self.lynx.reset_index(drop=True)
        }), self.runs.loc[:, ['shampoo_sales', 'lynx']])

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

    def test_k_best_suggestions(self):
        kND = KNearestDataSets(logger=self.logger)
        kND.fit(pd.DataFrame({
            'shampoo_sales': self.shampoo_sales.reset_index(drop=True),
            'lynx': self.lynx.reset_index(drop=True)
        }), self.runs.loc[:, ['shampoo_sales', 'lynx']])
        neighbor = kND.k_best_suggestions(self.airline, 1)
        self.assertEqual([('shampoo_sales', 146303.19499999995, 1)], neighbor)
        neighbors = kND.k_best_suggestions(self.airline, 2)
        self.assertEqual([('shampoo_sales', 146303.19499999995, 1), ('lynx', 115793162.625, 2)], neighbors)
        neighbors = kND.k_best_suggestions(self.airline, -1)
        self.assertEqual([('shampoo_sales', 146303.19499999995, 1), ('lynx', 115793162.625, 2)], neighbors)

        self.assertRaises(ValueError, kND.k_best_suggestions, self.airline, 0)
        self.assertRaises(ValueError, kND.k_best_suggestions, self.airline, -2)

    def test_m4(self):
        kND = KNearestDataSets(logger=self.logger)

        time_series = [f'Y{i}' for i in range(1, 1000)]
        ys, _ = load_timeseries(time_series)

        kND.fit(pd.DataFrame({ts: y for ts, y in zip(time_series, ys)}),
                pd.DataFrame({ts: [i] for i, ts in enumerate(time_series)}))

        x, _ = load_timeseries('Y100')
        neighbor = kND.k_best_suggestions(x, 1)
        self.assertEqual([('Y100', 0.0, 0)], neighbor)

        x, _ = load_timeseries('Y1001')
        neighbor = kND.k_best_suggestions(x, 2)
        self.assertEqual([('Y105', 624054.5248466857, 0), ('Y931', 653325., 0)], neighbor)
