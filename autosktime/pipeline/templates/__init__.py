from collections import OrderedDict
from typing import Dict, Union, Any, List, Type

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL, UNIVARIATE_TASKS, MULTIVARIATE_TASKS, PANEL_TASKS
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES
from autosktime.pipeline.components.util import sub_configuration
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster
from autosktime.pipeline.templates.regression import RegressionPipeline
from autosktime.pipeline.templates.univariate_endogenous import UnivariateEndogenousPipeline


class TemplateChoice(AutoSktimePredictor):

    def __init__(
            self,
            config: Configuration = None,
            init_params: Dict[str, Any] = None,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.config = config
        self.init_params = init_params
        self.random_state = random_state

        if config is not None:
            self.set_hyperparameters(config, init_params=init_params)

    def set_hyperparameters(
            self,
            configuration: Union[Configuration, Dict[str, Any]],
            init_params: Dict[str, Any] = None
    ):
        params = configuration.get_dictionary() if isinstance(configuration, Configuration) else configuration
        choice, sub_config = sub_configuration(params, init_params)
        self.estimator = self.get_components()[choice](config=sub_config, random_state=self.random_state)
        return self

    def get_hyperparameter_search_space(
            self,
            dataset_properties: DatasetProperties = None,
            default: str = None,
            include: Dict[str, Dict[str, List[str]]] = None,
            exclude: Dict[str, Dict[str, List[str]]] = None
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        # Compile a list of all objects for this problem
        available_components_ = self.get_available_components(dataset_properties, include, exclude)

        if len(available_components_) == 0:
            raise ValueError('No estimators found')

        if default not in available_components_.keys():
            default = None
        if default is None:
            for default_ in available_components_.keys():
                if include is not None and default_ not in include.keys():
                    continue
                if exclude is not None and default_ in exclude.keys():
                    continue
                default = default_
                break

        # noinspection PyArgumentList
        estimator = CategoricalHyperparameter('__choice__', list(available_components_.keys()), default_value=default)
        cs.add_hyperparameter(estimator)

        for comp_name in available_components_.keys():
            comp_cs = available_components_[comp_name].get_hyperparameter_search_space()
            parent_hyperparameter = {'parent': estimator, 'value': comp_name}
            cs.add_configuration_space(comp_name, comp_cs, parent_hyperparameter=parent_hyperparameter)

        self.config_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def get_available_components(
            self,
            dataset_properties: DatasetProperties = None,
            include: Dict[str, Dict[str, List[str]]] = None,
            exclude: Dict[str, Dict[str, List[str]]] = None
    ) -> Dict[str, ConfigurableTransformedTargetForecaster]:
        if include is not None and exclude is not None:
            raise ValueError('The argument include and exclude cannot be used together.')

        available_comp = self.get_components()

        if include is not None:
            for incl in include.keys():
                if incl not in available_comp:
                    raise ValueError(f'Trying to include unknown component: {incl}')

        components_dict = OrderedDict()
        for name, entry in available_comp.items():
            if include is not None and name not in include.keys():
                continue
            elif exclude is not None and name in exclude.keys():
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

            components_dict[name] = available_comp[name](
                dataset_properties=dataset_properties,
                include=include.get(name) if include is not None else None,
                exclude=exclude.get(name) if exclude is not None else None,
                random_state=self.random_state
            )

        return components_dict

    def get_components(self) -> Dict[str, Type[ConfigurableTransformedTargetForecaster]]:
        return {
            'linear': UnivariateEndogenousPipeline,
            'regression': RegressionPipeline
        }

    def _fit(self, y: pd.Series, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        if self.estimator is None:
            raise NotImplementedError
        # noinspection PyUnresolvedReferences
        return self.estimator.fit(y, X=X, fh=fh)

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }
