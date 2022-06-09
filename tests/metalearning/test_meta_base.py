import unittest

import pandas as pd
from sktime.datasets import load_airline

from autosktime.constants import UNIVARIATE_FORECAST
from autosktime.data import DatasetProperties
from autosktime.metalearning.meta_base import MetaBase
from autosktime.pipeline.templates import util


class MetaBaseTest(unittest.TestCase):

    def setUp(self) -> None:
        properties = DatasetProperties(index_type=pd.RangeIndex)
        self.base = MetaBase(util.get_configuration_space(properties), UNIVARIATE_FORECAST, 'mape', base_dir='./files/')

    def test_load_datasets(self):
        self.assertEqual(['Y1', 'Y2', 'Y3'], self.base._timeseries.columns.to_list())
        self.assertEqual((37, 3), self.base._timeseries.shape)

    def test_get_configuration(self):
        actual = self.base.get_configuration('Y1', 0)
        expected = [{
            'forecaster:__choice__': 'theta',
            'forecaster:theta:deseasonalize': True,
            'forecaster:theta:sp': 1.0,
            'imputation:method': 'random',
            'normalizer:__choice__': 'log',
            'outlier:n_sigma': 2.3176485168449408,
            'outlier:window_length': 17
        }, {
            'forecaster:__choice__': 'theta',
            'forecaster:theta:deseasonalize': False,
            'forecaster:theta:sp': 7.0,
            'imputation:method': 'ffill',
            'normalizer:__choice__': 'noop',
            'outlier:n_sigma': 2.376676502499649,
            'outlier:window_length': 11
        }]

        self.assertEqual(expected[0], actual.get_dictionary())

        actual = self.base.get_configuration('Y1', 1)
        self.assertEqual(expected[1], actual.get_dictionary())

        actual = self.base.get_configuration('Y1', 0.12)
        self.assertEqual(expected, [a.get_dictionary() for a in actual])

    def test_suggest_best_configs(self):
        y = load_airline()
        actual = self.base.suggest_configs(y, 2)

        expected = [
            {
                'forecaster:__choice__': 'theta',
                'forecaster:theta:deseasonalize': True,
                'forecaster:theta:sp': 1.0,
                'imputation:method': 'random',
                'normalizer:__choice__': 'log',
                'outlier:n_sigma': 2.3176485168449408,
                'outlier:window_length': 17
            }, {
                'forecaster:__choice__': 'tbats',
                'forecaster:tbats:sp': 4.0,
                'forecaster:tbats:use_arma_errors': True,
                'forecaster:tbats:use_box_cox': False,
                'forecaster:tbats:use_damped_trend': False,
                'forecaster:tbats:use_trend': True,
                'imputation:method': 'bfill',
                'normalizer:__choice__': 'log',
                'outlier:n_sigma': 3.689896803394823,
                'outlier:window_length': 11
            }]

        self.assertEqual(expected, [a.get_dictionary() for a in actual])
