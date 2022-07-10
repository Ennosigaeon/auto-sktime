import unittest
from typing import Dict, Type

import numpy as np
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from autosktime.metrics import RootMeanSquaredError, WeightedRootMeanSquaredError, WeightedMeanAbsoluteError, \
    MeanAbsoluteError, MeanError, MeanArctangentAbsoluteRelativeError, MeanAbsolutePercentageError, \
    MedianAbsolutePercentageError, MeanAbsoluteScaledError, OverallWeightedAverage, RelativePrognosticHorizon, \
    CumulativeRelativeAccuracy, PrognosticHorizonRate, StandardDeviationError


def exclude_base(func):
    def wrapper(self, *args, **kwargs):
        if self.__class__ == BaseMetricTest:
            return

        try:
            return func(self, *args, **kwargs)
        except ModuleNotFoundError:
            return

    return wrapper


class BaseMetricTest(unittest.TestCase):
    module: Type[BaseForecastingErrorMetric] = None
    res: Dict[str, float] = None
    y_train: bool = False

    @exclude_base
    def test_identical(self):
        np.random.seed(42)
        a = np.ones(10)
        b = np.ones(10)
        c = np.ones(10)

        if self.y_train:
            res = self.module()(a, b, y_train=c)
        else:
            res = self.module()(a, b)

        self.assertAlmostEqual(self.res["test_identical"], res)

    @exclude_base
    def test_offset_one(self):
        np.random.seed(42)
        a = np.zeros(10)
        b = np.ones(10)
        c = np.ones(10)

        if self.y_train:
            res = self.module()(a, b, y_train=c)
        else:
            res = self.module()(a, b)

        self.assertAlmostEqual(self.res["test_offset_one"], res)

    @exclude_base
    def test_random_univariate(self):
        np.random.seed(42)
        a = np.random.rand(10)
        b = np.random.rand(10)
        c = np.random.rand(10)

        if self.y_train:
            res = self.module()(a, b, y_train=c)
        else:
            res = self.module()(a, b)

        self.assertAlmostEqual(self.res["test_random_univariate"], res)

    @exclude_base
    def test_random_multivariate(self):
        np.random.seed(42)
        a = np.random.rand(10, 2)
        b = np.random.rand(10, 2)
        c = np.random.rand(10, 2)

        if self.y_train:
            res = self.module()(a, b, y_train=c)
        else:
            res = self.module()(a, b)

        self.assertAlmostEqual(self.res["test_random_multivariate"], res)


class RMSETest(BaseMetricTest):
    module = RootMeanSquaredError
    res = {
        'test_random_univariate': 0.25805839598665065,
        'test_random_multivariate': 0.5155226217982779,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
    }


class wRMSETest(BaseMetricTest):
    module = WeightedRootMeanSquaredError
    res = {
        'test_random_univariate': 0.31319988981348584,
        'test_random_multivariate': 0.472302358402945,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
    }


class MAETest(BaseMetricTest):
    module = MeanAbsoluteError
    res = {
        'test_random_univariate': 0.20867273347264428,
        'test_random_multivariate': 0.44707585479024214,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
    }


class wMAETest(BaseMetricTest):
    module = WeightedMeanAbsoluteError
    res = {
        'test_random_univariate': 0.28498883481671194,
        'test_random_multivariate': 0.38445299078287437,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
    }


class METest(BaseMetricTest):
    module = MeanError
    res = {
        'test_random_univariate': -0.12486889510977861,
        'test_random_multivariate': -0.0028733241269221013,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
    }


class STDTest(BaseMetricTest):
    module = StandardDeviationError
    res = {
        'test_random_univariate': 0.16003411457234076,
        'test_random_multivariate': 0.28548096275330204,
        'test_identical': 0.0,
        'test_offset_one': 0.0,
    }


class MAARETest(BaseMetricTest):
    module = MeanArctangentAbsoluteRelativeError
    res = {
        'test_random_univariate': 0.43455848653626566,
        'test_random_multivariate': 0.7754344161537233,
        'test_identical': 0.0,
        'test_offset_one': np.pi / 2,
    }


class RelPHTest(BaseMetricTest):
    module = RelativePrognosticHorizon

    res = {
        'test_random_univariate': 1.0,
        'test_random_multivariate': 0.2,
        'test_identical': 1.0,
        'test_offset_one': 0.0,
    }


class PHRateTest(BaseMetricTest):
    module = PrognosticHorizonRate

    res = {
        'test_random_univariate': 1.0,
        'test_random_multivariate': 0.7,
        'test_identical': 1.0,
        'test_offset_one': 0.0,
    }


class CRATest(BaseMetricTest):
    module = CumulativeRelativeAccuracy

    res = {
        'test_random_univariate': 0.24088151644147335,
        'test_random_multivariate': -2.994246126474539,
        'test_identical': 1.0,
        'test_offset_one': -np.inf,
    }


class MAPETest(BaseMetricTest):
    module = MeanAbsolutePercentageError
    res = {
        'test_random_univariate': 0.6218485628785994,
        'test_random_multivariate': 1.0056419955697928,
        'test_identical': 0.0,
        'test_offset_one': 2.,
    }


class MeAPETest(BaseMetricTest):
    module = MedianAbsolutePercentageError
    res = {
        'test_random_univariate': 0.33779795789128786,
        'test_random_multivariate': 0.808231484254992,
        'test_identical': 0.0,
        'test_offset_one': 4503599627370496.,
    }


class MASETest(BaseMetricTest):
    module = MeanAbsoluteScaledError
    y_train = True
    res = {
        'test_random_univariate': 0.7107790103959116,
        'test_random_multivariate': 1.7202314823505105,
        'test_identical': 0.0,
        'test_offset_one': 4503599627370496.,
    }


@unittest.skip('Requires frequency information')
class OWATest(BaseMetricTest):
    module = OverallWeightedAverage
    y_train = True
    res = {
        'test_random_univariate': 0.,
        'test_random_multivariate': 0.,
        'test_identical': 0.,
        'test_offset_one': 0.,
    }
