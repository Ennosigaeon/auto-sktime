import unittest
from typing import Dict, Any, Type

from autosktime.metrics import MeanAbsolutePercentageError
from autosktime.pipeline.components.base import AutoSktimePredictor
from autosktime.pipeline.components.forecast.arima import ARIMAComponent
from autosktime.pipeline.components.forecast.bats import BATSComponent
from autosktime.pipeline.components.forecast.ets import ETSComponent
from autosktime.pipeline.components.forecast.exp_smoothing import ExponentialSmoothingComponent
from autosktime.pipeline.components.forecast.naive import NaiveForecasterComponent
from autosktime.pipeline.components.forecast.tbats import TBATSComponent
from autosktime.pipeline.components.forecast.theta import ThetaComponent
from tests.pipeline.components.forecast.util import _test_forecaster


def exclude_base(func):
    def wrapper(self, *args, **kwargs):
        if self.__class__ == BaseForecastComponentTest:
            return

        try:
            self.module.check_dependencies()
            return func(self, *args, **kwargs)
        except ModuleNotFoundError:
            return

    return wrapper


class BaseForecastComponentTest(unittest.TestCase):
    module: Type[AutoSktimePredictor] = None
    res: Dict[str, Any] = None

    @exclude_base
    def test_airline(self):
        predictions, targets = _test_forecaster(dataset="airline", forecaster=self.module)
        self.assertAlmostEqual(self.res["default_airline"], MeanAbsolutePercentageError()(targets, predictions))

    @exclude_base
    def test_shampoo_sales(self):
        predictions, targets = _test_forecaster(dataset="shampoo_sales", forecaster=self.module)
        self.assertAlmostEqual(self.res["default_shampoo_sales"], MeanAbsolutePercentageError()(targets, predictions))

    @exclude_base
    def test_lynx(self):
        predictions, targets = _test_forecaster(dataset="lynx", forecaster=self.module)
        self.assertAlmostEqual(self.res["default_lynx"], MeanAbsolutePercentageError()(targets, predictions))

    # @exclude_base
    # def test_longley(self):
    #     predictions, targets = _test_forecaster(dataset="longley", forecaster=self.module)
    #     self.assertAlmostEqual(self.res["default_longley"], MeanAbsolutePercentageError()(targets, predictions))

    # @exclude_base
    # def test_uschange(self):
    #     predictions, targets = _test_forecaster(dataset="uschange", forecaster=self.module)
    #     self.assertAlmostEqual(self.res["default_uschange"], MeanAbsolutePercentageError()(targets, predictions))


class ARIMAComponentTest(BaseForecastComponentTest):
    module = ARIMAComponent
    res = {
        "default_airline": 0.4810660528764208,
        "default_shampoo_sales": 0.6515283536142408,
        "default_longley": 0.,
        "default_lynx": 0.8918009981494803,
        "default_uschange": 0.
    }


class BATSComponentTest(BaseForecastComponentTest):
    module = BATSComponent
    res = {
        "default_airline": 0.20740638743659737,
        "default_shampoo_sales": 0.39885629017674823,
        "default_longley": 0.,
        "default_lynx": 0.8769087014853324,
        "default_uschange": 0.
    }


class ETSComponentTest(BaseForecastComponentTest):
    module = ETSComponent
    res = {
        "default_airline": 0.28081929857986004,
        "default_shampoo_sales": 0.4015054717430467,
        "default_longley": 0.,
        "default_lynx": 0.9529550413236142,
        "default_uschange": 0.
    }


class ExpSmoothingComponentTest(BaseForecastComponentTest):
    module = ExponentialSmoothingComponent
    res = {
        "default_airline": 0.28080912828661697,
        "default_shampoo_sales": 0.40173687984238277,
        "default_longley": 0.,
        "default_lynx": 0.9532172046656211,
        "default_uschange": 0.
    }


class NaiveComponentTest(BaseForecastComponentTest):
    module = NaiveForecasterComponent
    res = {
        "default_airline": 0.28080912677075504,
        "default_shampoo_sales": 0.31614122022703284,
        "default_longley": 0.,
        "default_lynx": 0.952949547168119,
        "default_uschange": 0.
    }


# class ProphetComponentTest(BaseForecastComponentTest):
#     module = ProphetComponent
#     res = {
#         # TODO PeriodIndex is not supported for input, use type: DatetimeIndex or integer pd.Index instead.
#         "default_airline": 0.,
#         "default_shampoo_sales": 0.,
#         "default_longley": 0.,
#         "default_lynx": 0.,
#         "default_uschange": 0.
#     }


class TBATSComponentTest(BaseForecastComponentTest):
    module = TBATSComponent
    res = {
        "default_airline": 0.20740638743659737,
        "default_shampoo_sales": 0.39885629017674823,
        "default_longley": 0.,
        "default_lynx": 0.8769087014853324,
        "default_uschange": 0.
    }


class ThetaComponentTest(BaseForecastComponentTest):
    module = ThetaComponent
    res = {
        "default_airline": 0.19768187834986364,
        "default_shampoo_sales": 0.313356747471033,
        "default_longley": 0.,
        "default_lynx": 0.9486242161269473,
        "default_uschange": 0.
    }
