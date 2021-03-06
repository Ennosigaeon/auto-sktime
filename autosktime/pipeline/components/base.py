import abc
import importlib
import inspect
import pkgutil
import sys
from abc import ABC
from collections import OrderedDict
from typing import Dict, Type, List, Any

import pandas as pd
from sklearn.base import BaseEstimator

from autosktime.constants import SUPPORTED_INDEX_TYPES
from autosktime.data import DatasetProperties
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import Configuration, ConfigurationSpace

COMPONENT_PROPERTIES = Any


class AutoSktimeComponent(BaseEstimator):
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

    def fit(self, y: pd.Series, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
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

    def set_hyperparameters(self, configuration: Configuration, init_params: Dict[str, Any] = None):
        params = configuration.get_dictionary()

        for param, value in params.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter {} for {} because '
                                 'the hyperparameter does not exist.'.format(param, self))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param {} for {} because '
                                     'the init param does not exist.'.format(param, self))
                setattr(self, param, value)

        return self


class AutoSktimePredictor(AutoSktimeComponent, ABC):

    # noinspection PyUnresolvedReferences
    def predict(self, fh: ForecastingHorizon = None, X: pd.DataFrame = None):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(fh=fh, X=X)


def find_components(package, directory, base_class) -> Dict[str, AutoSktimeComponent]:
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "{}.{}".format(package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    component = obj
                    components[module_name] = component

    return components


class AutoSktimeChoice(AutoSktimePredictor, ABC):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.choice = None

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
            raise ValueError("The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: {}".format(incl))

        components_dict = OrderedDict()
        for name, entry in available_comp.items():
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            if dataset_properties is not None:
                if dataset_properties.index_type not in entry.get_properties()[SUPPORTED_INDEX_TYPES]:
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
        # noinspection PyArgumentList
        self.estimator = self.get_components()[choice](**new_params)

        return self

    @abc.abstractmethod
    def get_hyperparameter_search_space(
            self,
            dataset_properties: DatasetProperties = None,
            default: str = None,
            include: List[str] = None,
            exclude: List[str] = None
    ) -> Configuration:
        raise NotImplementedError()

    def fit(self, y: pd.Series, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        self.fitted_ = True
        self.estimator.fit(y, X=X, fh=fh)
        return self
