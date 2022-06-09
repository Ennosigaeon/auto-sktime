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
        self.assertEqual(self.base._timeseries.columns.to_list(), ['Y1', 'Y2', 'Y3'])
        self.assertEqual(self.base._timeseries.shape, (37, 3))

    def test_get_configuration(self):
        actual = self.base.get_configuration('Y1', 0)
        self.assertEqual(actual.get_dictionary(), {
            'forecaster:__choice__': 'naive', 'forecaster:naive:sp': 7.0,
            'forecaster:naive:strategy': 'last',
            'imputation:method': 'bfill', 'normalizer:__choice__': 'scaled_logit',
            'normalizer:scaled_logit:lower_bound': 0.1669434000354257,
            'normalizer:scaled_logit:upper_bound': 1.4780700820643673,
            'outlier:n_sigma': 4.373124452515206, 'outlier:window_length': 14
        })

    def test_suggest_best_configs(self):
        y = load_airline()
        actual = self.base.suggest_configs(y, 3)

        expected = [
            {
                'forecaster:__choice__': 'naive',
                'forecaster:naive:sp': 7.0,
                'forecaster:naive:strategy': 'last',
                'imputation:method': 'bfill',
                'normalizer:__choice__': 'scaled_logit',
                'normalizer:scaled_logit:lower_bound': 0.1669434000354257,
                'normalizer:scaled_logit:upper_bound': 1.4780700820643673,
                'outlier:n_sigma': 4.373124452515206,
                'outlier:window_length': 14,
            },
            {
                'forecaster:__choice__': 'bats',
                'forecaster:bats:sp': 7.0,
                'forecaster:bats:use_arma_errors': True,
                'forecaster:bats:use_box_cox': False,
                'forecaster:bats:use_damped_trend': True,
                'forecaster:bats:use_trend': True,
                'imputation:method': 'nearest',
                'normalizer:__choice__': 'box_cox',
                'normalizer:box_cox:lower_bound': -1.7664640121167117,
                'normalizer:box_cox:method': 'mle',
                'normalizer:box_cox:sp': 2.0,
                'normalizer:box_cox:upper_bound': -1.0,
                'outlier:n_sigma': 4.435139345734089,
                'outlier:window_length': 4,
            },
            {
                'forecaster:__choice__': 'ets',
                'forecaster:ets:damped_trend': True,
                'forecaster:ets:error': 'mul',
                'forecaster:ets:seasonal': 'mul',
                'forecaster:ets:sp': 4.0,
                'forecaster:ets:trend': 'add',
                'imputation:method': 'random',
                'normalizer:__choice__': 'box_cox',
                'normalizer:box_cox:lower_bound': -1.1855787081349818,
                'normalizer:box_cox:method': 'mle',
                'normalizer:box_cox:sp': 12.0,
                'normalizer:box_cox:upper_bound': 2.0,
                'outlier:n_sigma': 2.5301957481757715,
                'outlier:window_length': 8,
            }
        ]

        self.assertEqual([a.get_dictionary() for a in actual], expected)
