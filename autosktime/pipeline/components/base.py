import abc
import importlib
import inspect
import pkgutil
import sys
from abc import ABC
from collections import OrderedDict
from typing import Dict, Type, List, Any, Union, Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sktime.base import BaseEstimator
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from sktime.transformations.base import BaseTransformer

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import SUPPORTED_INDEX_TYPES, UNIVARIATE_TASKS, MULTIVARIATE_TASKS, PANEL_TASKS, \
    HANDLES_PANEL, HANDLES_MULTIVARIATE, HANDLES_UNIVARIATE
from autosktime.data import DatasetProperties
from autosktime.pipeline.util import sub_configuration
from autosktime.sktime_._utilities import get_cutoff
from autosktime.util.backend import ConfigContext, ConfigId

COMPONENT_PROPERTIES = Any


class AutoSktimeComponent(BaseEstimator):
    _estimator_class: Type[BaseEstimator] = None
    estimator: BaseEstimator = None
    config_id: ConfigId = None

    _tags = {
        'fit_is_empty': False,
        'X-y-must-have-same-index': True
    }

    @staticmethod
    @abc.abstractmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        """Get the properties of the underlying algorithm.

        Find more information at :ref:`get_properties`

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    @staticmethod
    def check_dependencies():
        """Raises an exception is missing dependencies are not installed"""
        pass

    def get_tags(self) -> Dict:
        try:
            estimator = self.estimator if self.estimator is not None else self._estimator_class()
            tags = estimator.get_tags()
        except (TypeError, AttributeError):
            tags = super(AutoSktimeComponent, self).get_tags()
        tags.update(self._tags)
        return tags

    def set_hyperparameters(self, configuration: Union[Configuration, Dict], init_params: Dict[str, Any] = None):
        if isinstance(configuration, Configuration):
            params = configuration.get_dictionary()
        else:
            params = configuration

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError(
                    f'Cannot set hyperparameter {param} for {self} because the hyperparameter does not exist.')
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError(f'Cannot set init param {param} for {self} because the init param does not exist.')
                setattr(self, param, value)

        return self

    def supports_iterative_fit(self) -> bool:
        return self.get_max_iter() is not None

    def get_max_iter(self) -> Optional[int]:
        return None

    def set_config_id(self, config_id: ConfigId):
        self.config_id = config_id

    def __hash__(self):
        return hash(self.estimator)


class AutoSktimePredictor(AutoSktimeComponent, BaseForecaster, ABC):
    _estimator_class: Type[BaseForecaster] = None
    estimator: BaseForecaster = None

    # noinspection PyUnresolvedReferences
    def get_fitted_params(self):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.get_fitted_params()

    @abc.abstractmethod
    def _fit(self, y: pd.Series, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        """The fit function calls the fit function of the underlying
        sktime model and returns `self`.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters' horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError()

    # noinspection PyUnresolvedReferences
    def _predict(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(fh=fh, X=X)

    def _update(self, y: pd.Series, X: pd.Series = None, update_params: bool = True):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.update(y, X=X, update_params=update_params)

    def _set_cutoff_from_y(self, y):
        # Use own patched version
        cutoff_idx = get_cutoff(y, self.cutoff, return_index=True)
        self._cutoff = cutoff_idx


class AutoSktimeTransformer(AutoSktimeComponent, BaseTransformer, ABC):
    _estimator_class: Type[BaseTransformer] = None
    estimator: BaseTransformer = None

    @abc.abstractmethod
    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        raise NotImplementedError()

    # noinspection PyUnresolvedReferences
    def _transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.transform(X, y=y)

    # noinspection PyUnresolvedReferences
    def _inverse_transform(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.inverse_transform(X, y=y)


def find_components(package, directory, base_class) -> Dict[str, Type[AutoSktimeComponent]]:
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = f'{package}.{module_name}'
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    component = obj
                    components[module_name] = component

    return components


class AutoSktimeChoice(AutoSktimeComponent, ABC):

    def __init__(
            self,
            estimator: AutoSktimeComponent = None,
            config_id: ConfigId = None,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.estimator: AutoSktimeComponent = estimator
        self.config_id = config_id
        self.random_state = random_state

    @classmethod
    @abc.abstractmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        raise NotImplementedError()

    def get_available_components(
            self,
            dataset_properties: DatasetProperties = None,
            include: List[str] = None,
            exclude: List[str] = None
    ) -> Dict[str, Type[AutoSktimeComponent]]:
        if include is not None and exclude is not None:
            raise ValueError('The argument include and exclude cannot be used together.')

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError(f'Trying to include unknown component: {incl}')

        components_dict = OrderedDict()
        for name, entry in available_comp.items():
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            if dataset_properties is not None:
                props = entry.get_properties()
                if dataset_properties.index_type not in props[SUPPORTED_INDEX_TYPES]:
                    continue
                if dataset_properties.task in UNIVARIATE_TASKS and not props[HANDLES_UNIVARIATE]:
                    continue
                if dataset_properties.task in MULTIVARIATE_TASKS and not props[HANDLES_MULTIVARIATE]:
                    continue
                if dataset_properties.task in PANEL_TASKS and not props[HANDLES_PANEL]:
                    continue

            try:
                available_comp[name].check_dependencies()
            except ModuleNotFoundError:
                continue

            components_dict[name] = available_comp[name]

        return components_dict

    def set_hyperparameters(
            self,
            configuration: Union[Configuration, Dict[str, Any]],
            init_params: Dict[str, Any] = None
    ):
        params = configuration.get_dictionary() if isinstance(configuration, Configuration) else configuration
        choice, new_params = sub_configuration(params, init_params)
        new_params['random_state'] = self.random_state

        self.new_params = new_params
        try:
            # noinspection PyArgumentList
            self.estimator = self.get_components()[choice](**new_params)
        except TypeError as ex:
            try:
                # Try to provide hyper-parameters as dictionary
                # noinspection PyArgumentList
                self.estimator = self.get_components()[choice](new_params)
            except TypeError as ex:
                # Provide selected type as additional info in message
                raise TypeError(f'{self.get_components()[choice]}.{ex}')

        # Copy tags from selected estimator
        tags = self.estimator.get_tags()
        self.set_tags(**tags)

        return self

    def get_hyperparameter_search_space(
            self,
            dataset_properties: DatasetProperties = None,
            default: str = None,
            include: List[str] = None,
            exclude: List[str] = None
    ) -> ConfigurationSpace:
        if include is not None and exclude is not None:
            raise ValueError('The arguments include and exclude cannot be used together.')

        cs = ConfigurationSpace()

        # Compile a list of all objects for this problem
        available_components = self.get_available_components(dataset_properties, include, exclude)

        if len(available_components) == 0:
            raise ValueError('No estimators found')

        if default is None or default not in available_components.keys():
            for default_ in available_components.keys():
                if include is not None and default_ not in include:
                    continue
                if exclude is not None and default_ in exclude:
                    continue
                default = default_
                break

        # noinspection PyArgumentList
        estimator = CategoricalHyperparameter('__choice__', list(available_components.keys()), default_value=default)
        cs.add_hyperparameter(estimator)

        for comp_name in available_components.keys():
            comp_cs = available_components[comp_name].get_hyperparameter_search_space(dataset_properties)
            parent_hyperparameter = {'parent': estimator, 'value': comp_name}
            # noinspection PyTypeChecker
            cs.add_configuration_space(comp_name, comp_cs, parent_hyperparameter=parent_hyperparameter)

        self.config_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def supports_iterative_fit(self) -> bool:
        if self.estimator is None:
            return False
        return self.estimator.supports_iterative_fit()

    def get_max_iter(self) -> Optional[int]:
        if self.estimator is None:
            return None
        return self.estimator.get_max_iter()

    def set_config_id(self, config_id: ConfigId):
        if self.estimator is None:
            return
        self.estimator.set_config_id(config_id)

    def update(self, X: pd.DataFrame, y: pd.Series, update_params: bool = True):
        # noinspection PyUnresolvedReferences
        self.estimator.update(X, y)
        return self


class AutoSktimeRegressionAlgorithm(AutoSktimeComponent, ABC):
    _estimator_class: Type[RegressorMixin] = None
    estimator: RegressorMixin = None
    iterations: int = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise NotImplementedError
        # noinspection PyUnresolvedReferences
        return self.estimator.predict(X)

    def get_iterations(self):
        config: ConfigContext = ConfigContext.instance()
        return self.iterations or config.get_config(self.config_id, 'iterations') or self.get_max_iter()

    def update(self, X: pd.DataFrame, y: pd.Series):
        if self.estimator is None:
            # noinspection PyUnresolvedReferences
            return self.fit(X, y)
        else:
            self._update()
            # noinspection PyUnresolvedReferences
            return self._fit(X, y)

    @abc.abstractmethod
    def _update(self):
        pass


class AutoSktimePreprocessingAlgorithm(TransformerMixin, AutoSktimeComponent, ABC):
    _estimator_class: Type[TransformerMixin] = None
    estimator: Union[TransformerMixin, str] = None

    def __init__(self, random_state: np.random.RandomState = None):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.estimator is None:
            raise NotFittedError()

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.estimator is None:
            raise NotFittedError()

        # noinspection PyUnresolvedReferences
        transformed_X = self.estimator.transform(X)

        return transformed_X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        return ConfigurationSpace()


class UpdatablePipeline(Pipeline):

    def update(self, y: pd.Series, X: pd.DataFrame = None, update_params: bool = True):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].update(Xt, y, update_params=update_params)

    def set_config_id(self, config_id: ConfigId):
        for name, est in self.steps:
            est.set_config_id(config_id)
        if hasattr(self, 'steps_'):
            for name, est in self.steps_:
                est.set_config_id(config_id)


class SwappedInput(AutoSktimePreprocessingAlgorithm, AutoSktimeTransformer):
    _estimator_class: Type[AutoSktimePreprocessingAlgorithm] = None
    estimator: AutoSktimePreprocessingAlgorithm = None

    def __init__(self, estimator: AutoSktimePreprocessingAlgorithm, random_state: np.random.RandomState = None):
        super().__init__(random_state)
        self.estimator = estimator

    def transform(self, X, y=None, **fit_params):
        yt = super(SwappedInput, self).transform(y)

        if isinstance(y, pd.Series) and not isinstance(yt, pd.Series):
            yt = pd.Series(yt, index=y.index, name=y.name)
        elif isinstance(y, pd.DataFrame) and not isinstance(yt, pd.DataFrame):
            yt = pd.DataFrame(yt, index=y.index, columns=y.columns)

        return X, yt

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(y, X).transform(X, y)

    def _fit(self, X: Union[pd.Series, pd.DataFrame], y: pd.Series = None):
        return

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return SwappedInput._estimator_class.get_properties(dataset_properties)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        raise ValueError('Do not use the static version of this method.')

    # noinspection PyRedeclaration
    def get_hyperparameter_search_space(self, dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        # This method is not static to allow access to self.estimator
        return self.estimator.get_hyperparameter_search_space(dataset_properties)

    def set_hyperparameters(self, configuration: Union[Configuration, Dict], init_params: Dict[str, Any] = None):
        self.estimator.set_hyperparameters(configuration, init_params)
