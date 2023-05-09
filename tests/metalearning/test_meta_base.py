import os
import pathlib
import unittest

import pandas as pd
from sktime.datasets import load_airline

from autosktime.constants import UNIVARIATE_FORECAST
from autosktime.data import DatasetProperties
from autosktime.metalearning.meta_base import MetaBase
from autosktime.pipeline.templates import util
from autosktime.smac.prior import Prior


class MetaBaseTest(unittest.TestCase):

    def setUp(self) -> None:
        properties = DatasetProperties(task=UNIVARIATE_FORECAST, index_type=pd.RangeIndex(0, 10, 1))
        base_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), 'files')
        self.base = MetaBase(util.get_configuration_space(properties), UNIVARIATE_FORECAST, 'mape', base_dir=base_dir)

    def test_load_datasets(self):
        self.assertEqual(['Y1', 'Y2', 'Y3'], self.base._timeseries.columns.to_list())
        self.assertEqual((37, 3), self.base._timeseries.shape)

    def test_get_configuration(self):
        actual = self.base._get_configuration('Y1', 0)
        expected = [{
            '__choice__': 'linear',
            'linear:forecaster:__choice__': 'theta',
            'linear:forecaster:theta:deseasonalize': True,
            'linear:forecaster:theta:sp': 1.0,
            'linear:imputation:method': 'random',
            'linear:normalizer:__choice__': 'log',
            'linear:outlier:n_sigma': 2.3176485168449408,
            'linear:outlier:window_length': 5
        }, {
            '__choice__': 'linear',
            'linear:forecaster:__choice__': 'theta',
            'linear:forecaster:theta:deseasonalize': False,
            'linear:forecaster:theta:sp': 7.0,
            'linear:imputation:method': 'ffill',
            'linear:normalizer:__choice__': 'noop',
            'linear:outlier:n_sigma': 2.376676502499649,
            'linear:outlier:window_length': 5
        }]

        self.assertEqual(expected[0], actual.get_dictionary())

        actual = self.base._get_configuration('Y1', 1)
        self.assertEqual(expected[1], actual.get_dictionary())

    @unittest.skip('fix test after finalizing search space')
    def test_suggest_univariate_prior(self):
        y = load_airline()
        actual = self.base.suggest_univariate_prior(y, 2)
        self.assertEqual(105, len(actual))
        self.assertEqual([True] * len(actual), [isinstance(v, Prior) for v in actual.values()])

    def test_suggest_best_configs(self):
        y = load_airline()
        actual = self.base.suggest_configs(y, 2)

        expected = [
            {
                '__choice__': 'linear',
                'linear:forecaster:__choice__': 'theta',
                'linear:forecaster:theta:deseasonalize': True,
                'linear:forecaster:theta:sp': 1.0,
                'linear:imputation:method': 'random',
                'linear:normalizer:__choice__': 'log',
                'linear:outlier:n_sigma': 2.3176485168449408,
                'linear:outlier:window_length': 5
            }, {
                '__choice__': 'linear',
                'linear:forecaster:__choice__': 'tbats',
                'linear:forecaster:tbats:sp': 4.0,
                'linear:forecaster:tbats:use_arma_errors': True,
                'linear:forecaster:tbats:use_box_cox': False,
                'linear:forecaster:tbats:use_damped_trend': False,
                'linear:forecaster:tbats:use_trend': True,
                'linear:imputation:method': 'bfill',
                'linear:normalizer:__choice__': 'log',
                'linear:outlier:n_sigma': 3.689896803394823,
                'linear:outlier:window_length': 5
            }]

        self.assertEqual(expected, [a.get_dictionary() for a in actual])
