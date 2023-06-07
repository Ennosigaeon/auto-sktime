import numpy as np
import pandas as pd
import shutil
import unittest
from sktime.datasets import load_airline, load_longley
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils._testing.hierarchical import _bottom_hier_datagen

from autosktime.automl import AutoML
from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.data.benchmark import FiltrationBenchmark
from autosktime.data.splitter import PanelHoldoutSplitter
from autosktime.metrics import calculate_loss
from autosktime.util import resolve_index


def train_test_split(y: SUPPORTED_Y_TYPES, X: pd.DataFrame = None, test_size: float = None):
    train_split, test_split = next(PanelHoldoutSplitter(test_size, random_state=np.random.RandomState(42)).split(y))

    y_train = y.iloc[train_split]
    y_test = y.iloc[test_split]

    y_train.index = y_train.index.remove_unused_levels()
    y_test.index = y_test.index.remove_unused_levels()

    if X is None:
        return y_train, y_test
    else:
        X_train = X.iloc[train_split, :]
        X_test = X.iloc[test_split, :]
        X_train.index = X_train.index.remove_unused_levels()
        X_test.index = X_test.index.remove_unused_levels()

        return y_train, y_test, X_train, X_test


def fit_and_predict(
        y: SUPPORTED_Y_TYPES,
        X: pd.DataFrame = None,
        panel: bool = False,
        fh: ForecastingHorizon = None,
        y_test_metric: np.ndarray = None
):
    try:
        shutil.rmtree('tmp')
        shutil.rmtree('tmp__tmp__')
        shutil.rmtree('output')
    except FileNotFoundError:
        pass

    splitter = train_test_split if panel else temporal_train_test_split

    if X is not None:
        y_train, y_test, X_train, X_test = splitter(y, X, test_size=0.2)
    else:
        y_train, y_test = splitter(y, test_size=0.2)
        X_train = None
        X_test = None

    if fh is None:
        fh = ForecastingHorizon(resolve_index(y_test.index), is_relative=False)

    if y_test_metric is None:
        y_test_metric = y_test

    automl = AutoML(
        runcount_limit=10,
        time_left_for_this_task=60,
        per_run_time_limit=10,
        working_directory='tmp',
        resampling_strategy='panel-holdout' if panel else 'temporal-holdout',
        seed=0
    )

    automl.fit(y_train, X_train, dataset_name='test')
    y_pred = automl.predict(fh, X_test)
    loss = calculate_loss(y_test_metric, y_pred, automl._task, automl._metric)

    return automl, loss


class AutoMLTest(unittest.TestCase):

    def test_reproducible_results(self):
        y = load_airline()

        incumbents = [{
            '__choice__': 'linear',
            'linear:forecaster:__choice__': 'naive',
            'linear:forecaster:naive:sp': 1,
            'linear:forecaster:naive:strategy': 'last',
            'linear:imputation:method': 'drift',
            'linear:normalizer:__choice__': 'box_cox',
            'linear:normalizer:box_cox:lower_bound': -2.0,
            'linear:normalizer:box_cox:method': 'mle',
            'linear:normalizer:box_cox:sp': 0,
            'linear:normalizer:box_cox:upper_bound': 2.0,
            'linear:outlier:n_sigma': 3.0,
            'linear:outlier:window_length': 5,
        }, {'__choice__': 'regression',
            'regression:detrend:degree': 1,
            'regression:detrend:with_intercept': True,
            'regression:imputation:method': 'nearest',
            'regression:outlier:n_sigma': 3.8145365592351377,
            'regression:outlier:window_length': 11,
            'regression:reduction:preprocessing:rescaling:__choice__': 'none',
            'regression:reduction:preprocessing:selection:__choice__': 'pca',
            'regression:reduction:preprocessing:selection:pca:keep_variance': 0.5447925568158788,
            'regression:reduction:preprocessing:selection:pca:whiten': 'False',
            'regression:reduction:regression:__choice__': 'passive_aggressive',
            'regression:reduction:regression:passive_aggressive:C': 2.2394334138366582e-05,
            'regression:reduction:regression:passive_aggressive:average': 'False',
            'regression:reduction:regression:passive_aggressive:fit_intercept': 'True',
            'regression:reduction:regression:passive_aggressive:loss': 'squared_epsilon_insensitive',
            'regression:reduction:regression:passive_aggressive:tol': 0.00014831194342106356,
            'regression:reduction:strategy': 'recursive',
            'regression:reduction:window_length': 7
            }, {
            '__choice__': 'regression',
            'regression:detrend:degree': 1,
            'regression:detrend:with_intercept': True,
            'regression:imputation:method': 'mean',
            'regression:outlier:n_sigma': 2.3561831568627323,
            'regression:outlier:window_length': 4,
            'regression:reduction:preprocessing:rescaling:__choice__': 'minmax',
            'regression:reduction:preprocessing:selection:__choice__': 'none',
            'regression:reduction:regression:__choice__': 'sgd',
            'regression:reduction:regression:sgd:alpha': 0.018854220341167446,
            'regression:reduction:regression:sgd:epsilon': 0.03910654323716942,
            'regression:reduction:regression:sgd:fit_intercept': 'True',
            'regression:reduction:regression:sgd:loss': 'epsilon_insensitive',
            'regression:reduction:regression:sgd:penalty': 'l2',
            'regression:reduction:regression:sgd:tol': 0.0022644276012169754,
            'regression:reduction:strategy': 'recursive',
            'regression:reduction:window_length': 2
        }]
        perf = [0.1219190910993419, 0.12043029777776044, 0.11097340803946293]

        for _ in range(3):
            automl, _ = fit_and_predict(y)

            for i in range(3):
                self.assertEqual(incumbents[i],
                                 automl.runhistory_.ids_config[automl.trajectory_[i].config_ids[0]].get_dictionary())
                self.assertEqual(perf[i], automl.trajectory_[i].costs[0])

    def test_univariate_endogenous(self):
        y = load_airline()

        _, loss = fit_and_predict(y)
        self.assertAlmostEqual(0.15817264895091435, loss)

    def test_univariate_exogenous(self):
        y, X = load_longley()

        _, loss = fit_and_predict(y, X)
        self.assertAlmostEqual(0.03530212266123195, loss)

    @unittest.skip('Panel without exogenous data not supported')
    def test_panel_endogenous(self):
        y = _bottom_hier_datagen(no_levels=1, random_seed=0)

        _, loss = fit_and_predict(y, panel=True)
        self.assertAlmostEqual(0.09982033042925743, loss)

    @unittest.skip('Panel without exogenous data not supported')
    def test_panel_endogenous_different_size(self):
        X, y = FiltrationBenchmark().get_data()
        automl, loss = fit_and_predict(y, panel=True)

        if len(automl.runhistory_.get_all_configs()) == 3:
            self.assertAlmostEqual(6.643866020291058e-06, loss)
        else:
            self.assertAlmostEqual(0.07232810928942789, loss)

    def test_panel_exogenous(self):
        X, y = FiltrationBenchmark().get_data()
        automl, loss = fit_and_predict(y, X, panel=True)

        self.assertAlmostEqual(0.31632010197592714, loss)

    @unittest.skip('Panel without exogenous data not supported')
    def test_panel_relative_forecast_horizon(self):
        X, y = FiltrationBenchmark().get_data()
        _, loss = fit_and_predict(
            y,
            panel=True,
            fh=ForecastingHorizon(np.arange(1, 20, 1), is_relative=True),
            y_test_metric=np.ones(38) * 0.26
        )

        self.assertAlmostEqual(0.0, loss)
