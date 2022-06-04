import logging
import unittest
import numpy as np

import pandas as pd

from autosktime.metalearning.kND import KNearestDataSets


class kNDTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.anneal = pd.Series({'number_of_instances': 898., 'number_of_classes': 5., 'number_of_features': 38.},
                                name='anneal')
        self.krvskp = pd.Series({'number_of_instances': 3196., 'number_of_classes': 2., 'number_of_features': 36.},
                                name='krvskp')
        self.labor = pd.Series({'number_of_instances': 57., 'number_of_classes': 2., 'number_of_features': 16.},
                               name='labor')
        self.runs = pd.DataFrame({
            'anneal': [0.1, 0.5, 0.7],
            'krvskp': [np.NaN, 0.1, 0.7],
            'labor': [0.5, 0.7, 0.1]
        })
        self.logger = logging.getLogger()

    def test_fit_l1_distance(self):
        kND = KNearestDataSets(logger=self.logger)

        kND.fit(pd.DataFrame([self.anneal, self.krvskp, self.labor]), self.runs)
        self.assertEqual(kND.best_configuration_per_dataset_['anneal'], 0)
        self.assertEqual(kND.best_configuration_per_dataset_['krvskp'], 1)
        self.assertEqual(kND.best_configuration_per_dataset_['labor'], 2)

    def test_kneighbors(self):
        kND = KNearestDataSets(logger=self.logger)
        kND.fit(pd.DataFrame([self.krvskp, self.labor]), self.runs.loc[:, ['krvskp', 'labor']])

        neighbor, _ = kND.kneighbors(self.anneal, 1)
        self.assertEqual(['krvskp'], neighbor)
        neighbor, distance = kND.kneighbors(self.anneal, 1)
        self.assertEqual(['krvskp'], neighbor)
        np.testing.assert_array_almost_equal([3.83208], distance)

        neighbors, _ = kND.kneighbors(self.anneal, 2)
        self.assertEqual(['krvskp', 'labor'], neighbors)
        neighbors, distance = kND.kneighbors(self.anneal, 2)
        self.assertEqual(['krvskp', 'labor'], neighbors)
        np.testing.assert_array_almost_equal([3.83208, 4.367919], distance)

        neighbors, _ = kND.kneighbors(self.anneal, -1)
        self.assertEqual(['krvskp', 'labor'], neighbors)
        neighbors, distance = kND.kneighbors(self.anneal, -1)
        self.assertEqual(['krvskp', 'labor'], neighbors)
        np.testing.assert_array_almost_equal([3.832080, 4.367919], distance)

        self.assertRaises(ValueError, kND.kneighbors, self.anneal, 0)
        self.assertRaises(ValueError, kND.kneighbors, self.anneal, -2)

    def test_k_best_suggestions(self):
        kND = KNearestDataSets(logger=self.logger)
        kND.fit(pd.DataFrame([self.krvskp, self.labor]), self.runs.loc[:, ['krvskp', 'labor']])
        neighbor = kND.k_best_suggestions(self.anneal, 1)
        self.assertEqual([('krvskp', 3.8320802803440586, 1)], neighbor)
        neighbors = kND.k_best_suggestions(self.anneal, 2)
        self.assertEqual([('krvskp', 3.8320802803440586, 1), ('labor', 4.367919719655942, 2)], neighbors)
        neighbors = kND.k_best_suggestions(self.anneal, -1)
        self.assertEqual([('krvskp', 3.8320802803440586, 1), ('labor', 4.367919719655942, 2)], neighbors)

        self.assertRaises(ValueError, kND.k_best_suggestions, self.anneal, 0)
        self.assertRaises(ValueError, kND.k_best_suggestions, self.anneal, -2)
