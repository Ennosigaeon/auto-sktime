from typing import List, Union, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.compose._reduce import RecursiveTabularRegressionForecaster

from ConfigSpace import ConfigurationSpace, Configuration, UniformIntegerHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES, AutoSktimeTransformer
from autosktime.pipeline.templates.base import set_pipeline_configuration, get_pipeline_search_space
from autosktime.pipeline.util import NotVectorizedMixin


class RecursivePanelReducer(NotVectorizedMixin, RecursiveTabularRegressionForecaster, AutoSktimeComponent):
    configspace: ConfigurationSpace = None

    _tags = {
        'scitype:transform-input': 'Panel',
        'scitype:transform-output': 'Panel',
        "scitype:y": "both",
        "y_inner_mtype": 'pd.DataFrame',
        "X_inner_mtype": 'pd.DataFrame',
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
        'X-y-must-have-same-index': True,
        'fit_is_empty': False
    }

    def __init__(
            self,
            estimator: Pipeline,
            dataset_properties: DatasetProperties,
            window_length: int = 5,
            transformers: List[Tuple[str, AutoSktimeTransformer]] = None,
            random_state: np.random.RandomState = None
    ):
        super(NotVectorizedMixin, self).__init__(estimator, window_length, transformers)
        # Just to make type explicit for type checker
        self.transformers: List[Tuple[str, AutoSktimeTransformer]] = transformers
        self.dataset_properties = dataset_properties
        self.random_state = random_state

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: True,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, ]
        }

    def get_fitted_params(self):
        pass

    def _fit(self, y, X=None, fh=None):
        self._target_column = y.name if isinstance(y, pd.Series) else y.columns[0]

        if self.transformers is None:
            return super()._fit(y, X, fh)

        transformers = self.transformers

        yt = y
        Xt = X
        for _, t in transformers:
            yt, Xt = t.fit_transform(X=yt, y=Xt)
        self.transformers = None

        res = super()._fit(yt, Xt, fh)

        self.transformers = transformers
        return res

    def _transform(self, y, X=None):
        if isinstance(y.index, pd.MultiIndex):
            Xt_complete = []
            yt_complete = []
            for idx in y.index.remove_unused_levels().levels[0]:
                X_ = X.loc[idx] if X is not None else None
                y_ = y.loc[idx]

                yt, Xt = super()._transform(y_, X_)
                yt_complete.append(yt)
                Xt_complete.append(Xt)

            return np.concatenate(yt_complete), np.concatenate(Xt_complete)
        else:
            return super()._transform(y, X)

    def _predict(self, fh, X=None):
        if X is not None and isinstance(X.index, pd.MultiIndex):
            y_pred_complete = []
            keys = X.index.remove_unused_levels().levels[0]

            for idx in keys:
                X_ = X.loc[idx]
                y_pred = super()._predict(fh, X_)
                y_pred_complete.append(y_pred)

            return pd.concat(y_pred_complete, keys=keys)
        else:
            return super()._predict(fh, X)

    def _predict_last_window(
            self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        return self._predict_in_sample(fh, X, return_pred_int, alpha)

    def _predict_in_sample(self, fh: ForecastingHorizon, X: pd.DataFrame = None, return_pred_int=False, alpha=None):
        if X is None:
            y_pred = np.zeros_like(fh._values, dtype=float)
        else:
            y = X.loc[fh.to_pandas(), [self._target_column]]
            X_ = X.loc[fh.to_pandas()].drop(columns=[self._target_column])

            _, Xt = self._transform(y, X_)
            y_pred = self.estimator_.predict(Xt)

            padding = np.ones(self.window_length_) * y_pred[0]
            y_pred = np.concatenate((padding, y_pred))

        if self._y_mtype_last_seen == 'pd-multiindex':
            return pd.DataFrame(y_pred, columns=self._y.columns, index=fh.to_pandas())
        else:
            return pd.Series(y_pred, name=self._y.name, index=fh.to_pandas())

    def set_hyperparameters(
            self,
            configuration: Union[Configuration, Dict[str, Any]],
            init_params: Dict[str, Any] = None
    ):
        # RecursiveReducer is a hybrid between Component and Pipeline which makes Configuration handling a bit messy
        if isinstance(configuration, Configuration):
            params = configuration.get_dictionary()
        else:
            params = configuration

        def split_dict(d: Dict) -> Dict[str, Dict]:
            res = {'own': {}, 'estimator': {}, 'transformers': {}}
            for param, value in d.items():
                if param.startswith('estimator:'):
                    new_name = param.replace('estimator:', '', 1)
                    res['estimator'][new_name] = value
                elif param.startswith('transformers:'):
                    new_name = param.replace('transformers:', '', 1)
                    res['transformers'][new_name] = value
                else:
                    res['own'][param] = value
            return res

        grouped_hps = split_dict(params)
        grouped_init_params = split_dict(init_params if init_params is not None else {})

        super().set_hyperparameters(grouped_hps['own'], grouped_init_params['own'])
        set_pipeline_configuration(grouped_hps['estimator'], self.estimator.steps, self.dataset_properties,
                                   grouped_init_params['estimator'])

        if self.transformers is not None:
            set_pipeline_configuration(grouped_hps['transformers'], self.transformers, self.dataset_properties,
                                       grouped_init_params['transformers'])

        return self

    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        # RecursiveReducer is a hybrid between Component and Pipeline which makes Configuration handling a bit messy
        window_length = UniformIntegerHyperparameter('window_length', lower=3, upper=20, default_value=5)
        estimator = get_pipeline_search_space(self.estimator.steps, dataset_properties=dataset_properties)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([window_length])
        cs.add_configuration_space('estimator', estimator)

        if self.transformers is not None:
            transformers = ConfigurationSpace()
            for name, t in self.transformers:
                transformers.add_configuration_space(name, t.get_hyperparameter_search_space(dataset_properties))
            cs.add_configuration_space('transformers', transformers)

        return cs
