import os.path
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sktime.datasets import load_airline, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.utils._testing.hierarchical import _bottom_hier_datagen

from autosktime.automl import AutoML
from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.data.benchmark.rul import load_rul
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


def fit_and_predict(y: SUPPORTED_Y_TYPES, X: pd.DataFrame = None, panel: bool = False):
    try:
        shutil.rmtree('tmp')
        shutil.rmtree('output')
    except FileNotFoundError:
        pass

    splitter = train_test_split if panel else temporal_train_test_split

    if X is not None:
        y_train, y_test, X_train, X_test = splitter(y, X, test_size=0.2)
        if panel:
            X_test[y_test.columns[0]] = y_test
    else:
        y_train, y_test = splitter(y, test_size=0.2)
        X_train = None
        X_test = None

    automl = AutoML(
        time_left_for_this_task=10,
        per_run_time_limit=10,
        temporary_directory='tmp',
        resampling_strategy='panel-holdout' if panel else 'temporal-holdout',
        seed=0
    )

    automl.fit(y_train, X_train, dataset_name='test')
    y_pred = automl.predict(resolve_index(y_test.index), X_test)
    loss = calculate_loss(y_test, y_pred, automl._task, automl._metric)

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
        }, {
            '__choice__': 'regression',
            'regression:detrend:degree': 2,
            'regression:detrend:with_intercept': False,
            'regression:imputation:method': 'linear',
            'regression:outlier:n_sigma': 4.1524233479073684,
            'regression:outlier:window_length': 6,
            'regression:reduction:preprocessing:rescaling:__choice__': 'minmax',
            'regression:reduction:preprocessing:selection:__choice__': 'pca',
            'regression:reduction:preprocessing:selection:pca:keep_variance': 0.9750518438961764,
            'regression:reduction:preprocessing:selection:pca:whiten': 'False',
            'regression:reduction:regression:__choice__': 'random_forest',
            'regression:reduction:regression:random_forest:bootstrap': 'True',
            'regression:reduction:regression:random_forest:criterion': 'friedman_mse',
            'regression:reduction:regression:random_forest:max_depth': 'None',
            'regression:reduction:regression:random_forest:max_features': 0.22502824032754642,
            'regression:reduction:regression:random_forest:max_leaf_nodes': 'None',
            'regression:reduction:regression:random_forest:min_impurity_decrease': 0.0,
            'regression:reduction:regression:random_forest:min_samples_leaf': 7,
            'regression:reduction:regression:random_forest:min_samples_split': 8,
            'regression:reduction:regression:random_forest:min_weight_fraction_leaf': 0.0,
            'regression:reduction:strategy': 'recursive',
            'regression:reduction:window_length': 3,
        }]
        perf = [2147483648., 0.1609, 0.1378]

        for i in range(3):
            automl, _ = fit_and_predict(y)
            self.assertEqual(incumbents[0], automl.trajectory_[0].incumbent.get_dictionary())
            self.assertEqual(perf[0], automl.trajectory_[0].train_perf)

            self.assertEqual(incumbents[0], automl.trajectory_[1].incumbent.get_dictionary())
            self.assertEqual(perf[1], automl.trajectory_[1].train_perf)

            # self.assertEqual(incumbents[1], automl.trajectory_[2].incumbent.get_dictionary())
            # self.assertEqual(perf[1], automl.trajectory_[1].train_perf)

    def test_univariate_endogenous(self):
        y = load_airline()

        _, loss = fit_and_predict(y)
        self.assertAlmostEqual(0.17396103403628044, loss)

    def test_univariate_exogenous(self):
        y, X = load_longley()

        _, loss = fit_and_predict(y, X)
        self.assertAlmostEqual(0.04423657419818824, loss)

    def test_panel_endogenous(self):
        y = _bottom_hier_datagen(no_levels=1, random_seed=0)

        _, loss = fit_and_predict(y)
        self.assertAlmostEqual(0.16764395883774477, loss)

    def test_panel_exogenous(self):
        X, y = load_rul(
            os.path.join(Path(__file__).parent.resolve(), 'data', 'rul')
        )
        _, loss = fit_and_predict(y, X, panel=True)

        self.assertAlmostEqual(5.17433814792307e-05, loss)
