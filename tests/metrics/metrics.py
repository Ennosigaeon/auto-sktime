import os.path
import unittest
from pathlib import Path
from typing import Dict, Type, List, Union

import numpy as np
import pandas as pd
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
    res: Dict[str, Union[float, List[float]]] = None
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
        a = np.ones(10)
        b = np.ones(10) * 2
        c = np.ones(10) * 2

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

    @exclude_base
    def test_rul_data(self):
        for i in range(1, 5):
            df = pd.read_csv(
                os.path.join(Path(__file__).parent.resolve(), '..', 'data', 'metrics', f'{i}_rul.csv'),
                index_col='Time'
            )
            df.index = df.index.astype(int)

            a = df['RULtrue']
            b = df['RULpred']
            c = a

            if self.y_train:
                res = self.module()(a, b, y_train=c)
            else:
                res = self.module()(a, b)

            self.assertAlmostEqual(self.res['test_rul_data'][i - 1], res, places=4)

    @exclude_base
    def test_multiindex(self):
        a = [pd.DataFrame(np.random.rand(10)), pd.DataFrame(np.random.rand(20))]
        b = [pd.DataFrame(np.random.rand(10)), pd.DataFrame(np.random.rand(20))]
        c = [pd.DataFrame(np.random.rand(10)), pd.DataFrame(np.random.rand(20))]

        a = pd.concat(a, axis=0, keys=[1, 2])
        b = pd.concat(b, axis=0, keys=[1, 2])
        c = pd.concat(c, axis=0, keys=[1, 2])

        if self.y_train:
            res = self.module()(a, b, y_train=c)
        else:
            res = self.module()(a, b)

        self.assertAlmostEqual(self.res['test_multiindex'], res)


class RMSETest(BaseMetricTest):
    module = RootMeanSquaredError
    res = {
        'test_random_univariate': 0.25805839598665065,
        'test_random_multivariate': 0.5155226217982779,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
        'test_rul_data': [11.95315, 12.0863, 19.17031, 19.09333],
        'test_multiindex': 0.48125311004879096
    }


class wRMSETest(BaseMetricTest):
    module = WeightedRootMeanSquaredError
    res = {
        'test_random_univariate': 0.31319988981348584,
        'test_random_multivariate': 0.472302358402945,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
        'test_rul_data': [7.671201380303419, 7.801224149975564, 16.11790546216795, 11.403906477222359],
        'test_multiindex': 0.4012428148812932
    }


class MAETest(BaseMetricTest):
    module = MeanAbsoluteError
    res = {
        'test_random_univariate': 0.20867273347264428,
        'test_random_multivariate': 0.44707585479024214,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
        'test_rul_data': [9.418613, 10.36906, 18.39041, 17.36921],
        'test_multiindex': 0.4070179135278025
    }


class wMAETest(BaseMetricTest):
    module = WeightedMeanAbsoluteError
    res = {
        'test_random_univariate': 0.28498883481671194,
        'test_random_multivariate': 0.38445299078287437,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
        'test_rul_data': [6.5607001926612725, 6.970551494372769, 15.304509907138861, 10.148936735375614],
        'test_multiindex': 0.3356117234871766
    }


class METest(BaseMetricTest):
    module = MeanError
    res = {
        'test_random_univariate': -0.12486889510977861,
        'test_random_multivariate': -0.0028733241269221013,
        'test_identical': 0.0,
        'test_offset_one': 1.0,
        'test_rul_data': [-8.297101, -10.34674, 18.39041, -17.36492],
        'test_multiindex': 0.04063873414466394
    }


class STDTest(BaseMetricTest):
    module = StandardDeviationError
    res = {
        'test_random_univariate': 0.2380520362658041,
        'test_random_multivariate': 0.45378875706396976,
        'test_identical': 0.0,
        'test_offset_one': 0.0,
        'test_rul_data': [8.644898331677966, 6.279017609960558, 5.443719768623921, 7.977768332499385],
        'test_multiindex': 0.4960844634969046
    }


class MAARETest(BaseMetricTest):
    module = MeanArctangentAbsoluteRelativeError
    res = {
        'test_random_univariate': 0.43455848653626566,
        'test_random_multivariate': 0.7754344161537233,
        'test_identical': 0.0,
        'test_offset_one': np.pi / 4,
        'test_rul_data': [0.1254494213418507, 0.14048126957742357, 0.28124283087870233, 0.20448637292673436],
        'test_multiindex': 0.7345646563264796
    }


class RelPHTest(BaseMetricTest):
    module = RelativePrognosticHorizon

    res = {
        'test_random_univariate': 1.0,
        'test_random_multivariate': 0.2,
        'test_identical': 1.0,
        'test_offset_one': 0.0,
        'test_rul_data': [0.411214953271028, 0.3469387755102041, 0.011494252873563218, 0.09900990099009901],
        'test_multiindex': 0.6
    }


class PHRateTest(BaseMetricTest):
    module = PrognosticHorizonRate

    res = {
        'test_random_univariate': 1.0,
        'test_random_multivariate': 0.7,
        'test_identical': 1.0,
        'test_offset_one': 0.0,
        'test_rul_data': [0.822429906542056, 0.744897959183674, 0.0919540229885057, 0.326732673267327],
        'test_multiindex': 0.75
    }


class CRATest(BaseMetricTest):
    module = CumulativeRelativeAccuracy

    res = {
        'test_random_univariate': 0.13753542936668478,
        'test_random_multivariate': -4.446269118117955,
        'test_identical': 1.0,
        'test_offset_one': 0.0,
        'test_rul_data': [0.8322938631251892, 0.8554756986236439, 0.6122893593964682, 0.7628956560882032],
        'test_multiindex': -0.655485313943879
    }


class MAPETest(BaseMetricTest):
    module = MeanAbsolutePercentageError
    res = {
        'test_random_univariate': 0.6218485628785994,
        'test_random_multivariate': 1.0056419955697928,
        'test_identical': 0.0,
        'test_offset_one': 2 / 3,
        'test_rul_data': [0.1305847660536306, 0.16554514862677863, 0.2604303982757675, 0.2608817965391625],
        'test_multiindex': 0.8911941155176933
    }


class MeAPETest(BaseMetricTest):
    module = MedianAbsolutePercentageError
    res = {
        'test_random_univariate': 0.33779795789128786,
        'test_random_multivariate': 0.808231484254992,
        'test_identical': 0.0,
        'test_offset_one': 1.,
        'test_rul_data': [0.08123111111111116, 0.09312747685185188, 0.19369777777777777, 0.13868322580645165],
        'test_multiindex': 0.7213723322621854
    }


class MASETest(BaseMetricTest):
    module = MeanAbsoluteScaledError
    y_train = True
    res = {
        'test_random_univariate': 0.7107790103959116,
        'test_random_multivariate': 1.7202314823505105,
        'test_identical': 0.0,
        'test_offset_one': 4503599627370496.,
        'test_rul_data': [3.767445308411215, 4.147624163265306, 7.356165324137931, 6.9476817199999985],
        'test_multiindex': 1.0467053143980116
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
        'test_rul_data': [0.0, 0.0, 0.0, 0.0],
        'test_multiindex': 0.
    }
