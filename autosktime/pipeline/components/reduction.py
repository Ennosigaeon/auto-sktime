from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.compose._reduce import _get_forecaster, _Reducer

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter, Configuration
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import COMPONENT_PROPERTIES, AutoSktimePredictor
from autosktime.pipeline.components.data_preprocessing import DataPreprocessingPipeline
from autosktime.pipeline.components.regression import RegressorChoice


def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
    return pd.Series(np.zeros_like(fh._values, dtype=float), index=fh._values)


class ReductionComponent(AutoSktimePredictor):

    def __init__(
            self,
            estimator: AutoSktimePredictor = None,
            strategy: str = 'recursive',
            window_length: int = 10,
            random_state=None
    ):
        super().__init__()
        self.strategy = strategy
        self.window_length = window_length
        self.random_state = random_state

        self._estimator_class = _get_forecaster('tabular-regressor', strategy)
        self.estimator = estimator

    def _fit(self, y: pd.Series, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        if self.estimator is None:
            raise ValueError('Call set_hyperparameter before fitting')

        # Monkey-patch _predict_in_sample which is not implemented as of sktime 0.12.0
        self.estimator._predict_in_sample = _predict_in_sample.__get__(self.estimator, _Reducer)
        self.estimator.fit(y, X=X, fh=fh)
        return self

    def set_hyperparameters(self, configuration: Configuration, init_params: Dict[str, Any] = None):
        preprocessing_params = {}
        regression_params = {}

        def parse_dict(d: Dict[str, Any]):
            for param, value in d.items():
                if param.startswith('preprocessing'):
                    param = param.replace('preprocessing:', '')
                    preprocessing_params[param] = value
                elif param.startswith('regression'):
                    param = param.replace('regression:', '')
                    regression_params[param] = value
                else:
                    setattr(self, param, value)

        parse_dict(configuration.get_dictionary())
        if init_params is not None:
            parse_dict(init_params)

        pipeline = Pipeline(steps=[
            ('preprocessing',
             DataPreprocessingPipeline(random_state=self.random_state).set_hyperparameters(preprocessing_params)),
            ('regression', RegressorChoice(random_state=self.random_state).set_hyperparameters(regression_params))
        ])

        self.estimator = make_reduction(
            pipeline,
            window_length=self.window_length,
            strategy=self.strategy,
            scitype='tabular-regressor'
        )

        # Copy tags from selected estimator
        tags = self.estimator.get_tags()
        self.set_tags(**tags)

        return self

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        strategy = CategoricalHyperparameter('strategy', choices=['recursive'])  # , 'direct', 'dirrec'])
        window_length = UniformIntegerHyperparameter('window_length', lower=3, upper=20, default_value=10)

        # TODO include and exclude missing
        preprocessing = DataPreprocessingPipeline() \
            .get_hyperparameter_search_space(dataset_properties=dataset_properties)
        regression = RegressorChoice().get_hyperparameter_search_space(dataset_properties=dataset_properties)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([strategy, window_length])
        cs.add_configuration_space('preprocessing', preprocessing)
        cs.add_configuration_space('regression', regression)

        return cs
