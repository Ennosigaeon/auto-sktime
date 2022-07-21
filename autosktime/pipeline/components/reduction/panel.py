from typing import List, Union, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.compose._reduce import RecursiveTabularRegressionForecaster
from sktime.utils.datetime import _shift

from ConfigSpace import ConfigurationSpace, Configuration, UniformIntegerHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, HANDLES_PANEL, IGNORES_EXOGENOUS_X, \
    SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, COMPONENT_PROPERTIES, AutoSktimeTransformer, \
    UpdatablePipeline
from autosktime.pipeline.components.downsampling import DownsamplerChoice, BaseDownSampling
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
            estimator: UpdatablePipeline,
            dataset_properties: DatasetProperties,
            window_length: int = 6,
            step_size: Union[int, float] = 0.5,
            transformers: List[Tuple[str, AutoSktimeTransformer]] = None,
            random_state: np.random.RandomState = None
    ):
        super(NotVectorizedMixin, self).__init__(estimator, window_length, transformers)
        self.step_size = step_size
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

        if isinstance(self.step_size, float):
            self.step_size_ = int(self.window_length * self.step_size)
        else:
            self.step_size_ = int(self.step_size)

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

    def _transform(self, y: pd.Series, X: pd.DataFrame = None):
        if y is None and X is None:
            raise ValueError('Provide either X or y')
        elif y is None:
            y = pd.Series(np.zeros(X.shape[0]), X.index)

        if isinstance(y.index, pd.MultiIndex):
            Xt_complete = []
            yt_complete = []
            for idx in y.index.remove_unused_levels().levels[0]:
                X_ = X.loc[idx] if X is not None else None
                y_ = y.loc[idx]

                yt, Xt = self._transform(y_, X_)
                yt_complete.append(yt)
                Xt_complete.append(Xt)

            return np.concatenate(yt_complete), np.concatenate(Xt_complete)
        else:
            yt, Xt = super()._transform(y, X)

            # Remove encoded y data
            Xt = Xt[:, self.window_length:]

            # Reshape to (num_samples, seq_length, num_features)
            Xt = Xt.T.reshape(X.shape[1], self.window_length, -1).T

            return yt[::self.step_size_], Xt[::self.step_size_]

    def _predict(self, fh: ForecastingHorizon, X: pd.DataFrame = None):
        if X is not None and isinstance(X.index, pd.MultiIndex):
            y_pred_complete = []
            keys = X.index.remove_unused_levels().levels[0]

            for idx in keys:
                X_ = X.loc[idx]
                original_cutoff = self._cutoff
                self._set_cutoff(X_.index[[-1]])

                y_pred = self._predict(fh, X_)
                y_pred_complete.append(y_pred)

                self._set_cutoff(original_cutoff)

            return pd.concat(y_pred_complete, keys=keys)
        else:
            Xt = X
            for _, t in self.transformers:
                # Skip down-sampling for predictions
                if isinstance(t, DownsamplerChoice) or isinstance(t, BaseDownSampling):
                    continue

                _, Xt = t.transform(X=None, y=Xt)

            return super()._predict(fh, Xt)

    def _update(self, y: pd.Series, X: pd.DataFrame = None, update_params: bool = True):
        yt = y
        Xt = X
        for _, t in self.transformers:
            yt, Xt = t.transform(X=yt, y=Xt)

        yt, Xt = self._transform(yt, Xt)

        self.estimator_.update(yt, Xt)

    # noinspection PyMethodOverriding
    def _get_last_window(self, X: pd.DataFrame) -> np.ndarray:
        """Select last window."""
        # Get the start and end points of the last window.
        cutoff = self.cutoff
        start = _shift(cutoff, by=-self.window_length_ + 1)

        # Get the last window of the endogenous variable.
        X = X.loc[start:cutoff].to_numpy()

        return X

    def _predict_last_window(
            self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        if X is None:
            raise ValueError('`X` must be passed to `predict`.')

            # Get last window of available data.
        X_last = self._get_last_window(X)

        # Pre-allocate arrays.
        if X is None:
            n_columns = 1
        else:
            n_columns = X.shape[1] + 1
        window_length = self.window_length_
        fh_max = fh.to_relative(self.cutoff)[-1]

        y_pred = np.zeros(fh_max)
        last = np.zeros((1, n_columns, window_length + fh_max))

        # Fill pre-allocated arrays with available data.
        last[:, 0, :window_length] = np.zeros(1, window_length)
        if X is not None:
            last[:, 1:, :window_length] = X_last.T
            last[:, 1:, window_length:] = X.T

        # Recursively generate predictions by iterating over forecasting horizon.
        for i in range(fh_max):
            # Slice prediction window.
            X_pred = last[:, :, i: window_length + i]

            # Reshape data into tabular array.
            X_pred = X_pred.reshape(1, -1)

            # Generate predictions.
            y_pred[i] = self.estimator_.predict(X_pred)

            # Update last window with previous prediction.
            last[:, 0, window_length + i] = y_pred[i]

        # While the recursive strategy requires to generate predictions for all steps
        # until the furthest step in the forecasting horizon, we only return the
        # requested ones.
        fh_idx = fh.to_indexer(self.cutoff)

        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_return = y_pred.iloc[fh_idx]
        else:
            y_return = y_pred[fh_idx]
            if self._y_mtype_last_seen == 'pd-multiindex':
                y_return = pd.DataFrame(y_return, columns=self._y.columns, index=fh.to_pandas())
            else:
                y_return = pd.Series(y_return, name=self._y.name, index=fh.to_pandas())

        return y_return

    def _predict_in_sample(self, fh: ForecastingHorizon, X: pd.DataFrame = None, return_pred_int=False, alpha=None):
        if X is None:
            raise ValueError('`X` must be passed to `predict`.')

        start = max(_shift(fh.to_pandas()[0], by=-self.window_length_ + 1), X.index[0])
        cutoff = fh.to_pandas()[-1] + 1

        X = X.loc[start: cutoff, :]

        # noinspection PyTypeChecker
        _, Xt = self._transform(None, X)
        y_pred = self.estimator_.predict(Xt)
        y_pred = np.repeat(y_pred, self.step_size_)

        if y_pred.shape > fh.to_pandas().shape:
            y_pred = y_pred[:fh.to_pandas().shape[0]]
        elif y_pred.shape < fh.to_pandas().shape:
            padding = np.ones(fh.to_pandas().shape[0] - y_pred.shape[0]) * y_pred[0]
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

    def supports_iterative_fit(self) -> bool:
        forecaster = self.estimator.steps[-1][1]
        return forecaster.supports_iterative_fit()

    def get_max_iter(self) -> Optional[int]:
        forecaster = self.estimator.steps[-1][1]
        return forecaster.get_max_iter()

    def set_desired_iterations(self, iterations: int):
        self.estimator.steps[-1][1].set_desired_iterations(iterations)
        if hasattr(self.estimator, 'steps_'):
            self.estimator.steps_[-1][1].set_desired_iterations(iterations)
        if hasattr(self, 'estimator_'):
            self.estimator_.steps[-1][1].set_desired_iterations(iterations)
            if hasattr(self.estimator_, 'steps_'):
                self.estimator_.steps_[-1][1].set_desired_iterations(iterations)
