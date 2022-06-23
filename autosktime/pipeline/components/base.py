import abc
import importlib
import inspect
import pkgutil
import sys
from abc import ABC
from collections import OrderedDict
from typing import Dict, Type, List, Any, Union

import pandas as pd
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sktime.base import BaseEstimator
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from sktime.transformations.base import BaseTransformer

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties

COMPONENT_PROPERTIES = Any


class AutoSktimeComponent(BaseEstimator):
    # TODO check which methods really have to be wrapped

    _estimator_class: Type[BaseEstimator] = None
    estimator: BaseEstimator = None

    _tags = {
        'fit_is_empty': False
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
            tags = {}
        tags.update(self._tags)
        return tags

    def set_hyperparameters(self, configuration: Configuration, init_params: Dict[str, Any] = None):
        params = configuration.get_dictionary()

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


class AutoSktimePredictor(AutoSktimeComponent, BaseForecaster, ABC):
    # TODO check which methods really have to be wrapped

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


class AutoSktimeTransformer(AutoSktimeComponent, BaseTransformer, ABC):
    # TODO check which methods really have to be wrapped

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
    _tags = {
        'requires-fh-in-fit': False
    }

    def __init__(self, estimator: AutoSktimeComponent = None, random_state=None):
        super().__init__()
        self.estimator = estimator
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
                if (dataset_properties.index_type is not None and
                        dataset_properties.index_type not in entry.get_properties()[SUPPORTED_INDEX_TYPES]):
                    continue

            try:
                available_comp[name].check_dependencies()
            except ModuleNotFoundError:
                continue

            components_dict[name] = available_comp[name]

        return components_dict

    def set_hyperparameters(self, configuration: Configuration, init_params: Dict[str, Any] = None):
        new_params = {}

        params = configuration.get_dictionary()
        choice = params['__choice__']

        for param, value in params.items():
            if param == '__choice__':
                continue

            param = param.replace(choice, '').replace(':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, '').replace(':', '')
                new_params[param] = value

        new_params['random_state'] = self.random_state

        self.new_params = new_params
        try:
            # noinspection PyArgumentList
            self.estimator = self.get_components()[choice](**new_params)
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

        if default is None:
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
            cs.add_configuration_space(comp_name, comp_cs, parent_hyperparameter=parent_hyperparameter)

        self.config_space = cs
        self.dataset_properties = dataset_properties
        return cs


class AutoSktimeRegressionAlgorithm(AutoSktimeComponent, ABC):
    _estimator_class: Type[RegressorMixin] = None
    estimator: RegressorMixin = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise NotImplementedError
        # noinspection PyUnresolvedReferences
        return self.estimator.predict(X)


class AutoSktimePreprocessingAlgorithm(TransformerMixin, AutoSktimeComponent, ABC):
    _estimator_class: Type[TransformerMixin] = None
    estimator: TransformerMixin = None

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
