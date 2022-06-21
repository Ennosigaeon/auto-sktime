import shutil
import unittest

from autosktime.metrics import calculate_loss
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split

from autosktime.automl import AutoML
from autosktime.constants import SUPPORTED_Y_TYPES
from autosktime.util import resolve_index
from sktime.utils._testing.hierarchical import _bottom_hier_datagen


def fit_and_predict(y: SUPPORTED_Y_TYPES):
    try:
        shutil.rmtree('tmp')
        shutil.rmtree('output')
    except FileNotFoundError:
        pass

    y_train, y_test = temporal_train_test_split(y, test_size=0.2)

    automl = AutoML(
        time_left_for_this_task=10,
        per_run_time_limit=10,
        temporary_directory='tmp',
        seed=0
    )

    automl.fit(y_train, dataset_name='test')
    y_pred = automl.predict(resolve_index(y_test.index))
    loss = calculate_loss(y_test, y_pred, automl._task, automl._metric)

    return automl, loss


class AutoMLTest(unittest.TestCase):

    def test_reproducible_results(self):
        y = load_airline()

        incumbents = [{
            'forecaster:__choice__': 'arima',
            'forecaster:arima:d': 0,
            'forecaster:arima:maxiter': 50,
            'forecaster:arima:p': 1,
            'forecaster:arima:q': 0,
            'forecaster:arima:sp': 0,
            'forecaster:arima:with_intercept': 'True',
            'imputation:method': 'drift',
            'normalizer:__choice__': 'box_cox',
            'normalizer:box_cox:lower_bound': -2.0,
            'normalizer:box_cox:method': 'mle',
            'normalizer:box_cox:sp': 0,
            'normalizer:box_cox:upper_bound': 2.0,
            'outlier:n_sigma': 3.0,
            'outlier:window_length': 10,
        }, {
            'forecaster:__choice__': 'ets',
            'forecaster:ets:damped_trend': True,
            'forecaster:ets:error': 'mul',
            'forecaster:ets:seasonal': 'mul',
            'forecaster:ets:sp': 7,
            'forecaster:ets:trend': 'mul',
            'imputation:method': 'ffill',
            'normalizer:__choice__': 'noop',
            'outlier:n_sigma': 2.3617555184501295,
            'outlier:window_length': 39,
        }]
        perf = [2147483648., 0.3026, 0.1378]

        for i in range(3):
            automl, _ = fit_and_predict(y)
            self.assertEqual(incumbents[0], automl.trajectory_[0].incumbent.get_dictionary())
            self.assertEqual(perf[0], automl.trajectory_[0].train_perf)

            self.assertEqual(incumbents[0], automl.trajectory_[1].incumbent.get_dictionary())
            self.assertEqual(perf[1], automl.trajectory_[1].train_perf)

            self.assertEqual(incumbents[1], automl.trajectory_[2].incumbent.get_dictionary())
            self.assertEqual(perf[1], automl.trajectory_[1].train_perf)

    def test_series(self):
        y = load_airline()

        _, loss = fit_and_predict(y)
        self.assertAlmostEqual(0.1414420215112278, loss)

    def test_multi_index(self):
        y = _bottom_hier_datagen(no_levels=1, random_seed=0)

        _, loss = fit_and_predict(y)
        self.assertAlmostEqual(0.387640331022832, loss)
