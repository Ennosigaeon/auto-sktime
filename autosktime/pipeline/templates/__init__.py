from collections import OrderedDict
from typing import Dict, Union, Any, List, Type, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter
from autosktime.constants import HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, SUPPORTED_INDEX_TYPES, \
    HANDLES_PANEL, UNIVARIATE_TASKS, MULTIVARIATE_TASKS, PANEL_TASKS
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimePredictor, COMPONENT_PROPERTIES
from autosktime.pipeline.templates.base import ConfigurableTransformedTargetForecaster
from autosktime.pipeline.templates.panel_regression import PanelRegressionPipeline
from autosktime.pipeline.templates.nn_panel_regression import NNPanelRegressionPipeline
from autosktime.pipeline.templates.preconstructed.cnn import CNNRegressionPipeline
from autosktime.pipeline.templates.preconstructed.lstm import LSTMRegressionPipeline
from autosktime.pipeline.templates.preconstructed.random_forest import RandomForestPipeline
from autosktime.pipeline.templates.preconstructed.svm import SVMRegressionPipeline
from autosktime.pipeline.templates.preconstructed.transformers import TransformerRegressionPipeline
from autosktime.pipeline.templates.regression import RegressionPipeline
from autosktime.pipeline.templates.univariate_endogenous import UnivariateEndogenousPipeline
from autosktime.pipeline.util import sub_configuration, NotVectorizedMixin, Int64Index
from autosktime.util.backend import ConfigId


class TemplateChoice(NotVectorizedMixin, AutoSktimePredictor):

    def __init__(
            self,
            estimator: AutoSktimePredictor = None,
            config: Configuration = None,
            budget: float = None,
            dataset_properties: DatasetProperties = None,
            init_params: Dict[str, Any] = None,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.estimator = estimator
        self.config = config
        self.budget = budget
        self.dataset_properties = dataset_properties
        self.init_params = init_params
        self.random_state = random_state

        if estimator is None and config is not None:
            self.set_hyperparameters(config, init_params=init_params)

    def set_hyperparameters(
            self,
            configuration: Union[Configuration, Dict[str, Any]],
            init_params: Dict[str, Any] = None
    ):
        params = configuration.get_dictionary() if isinstance(configuration, Configuration) else configuration
        choice, sub_config = sub_configuration(params, init_params)
        available_components = self.get_components()
        available_components.update(self.get_baseline_components())
        self.estimator: AutoSktimePredictor = available_components[choice](
            config=sub_config,
            dataset_properties=self.dataset_properties,
            random_state=self.random_state
        )
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
            # noinspection PyTypeChecker
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
            # Only consider baseline components if explicitly included
            available_comp.update(self.get_baseline_components())
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

    @staticmethod
    def get_components() -> Dict[str, Type[ConfigurableTransformedTargetForecaster]]:
        return {
            'statistical': UnivariateEndogenousPipeline,
            'regression': RegressionPipeline,
            'panel-regression': PanelRegressionPipeline,
            'nn-panel-regression': NNPanelRegressionPipeline,
        }

    @staticmethod
    def get_baseline_components() -> Dict[str, Type[ConfigurableTransformedTargetForecaster]]:
        return {
            'baseline_rf': RandomForestPipeline,
            'baseline_lstm': LSTMRegressionPipeline,
            'baseline_cnn': CNNRegressionPipeline,
            'baseline_transformer': TransformerRegressionPipeline,
            'baseline_svm': SVMRegressionPipeline,
        }

    def _fit(self, y: pd.Series, X: pd.DataFrame = None, fh: ForecastingHorizon = None):
        if self.estimator is None:
            raise NotImplementedError
        # noinspection PyUnresolvedReferences
        return self.estimator.fit(y, X=X, fh=fh)

    def _update(self, y: pd.Series, X: pd.Series = None, update_params: bool = True):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.update(y, X=X, update_params=update_params)

    def supports_iterative_fit(self) -> bool:
        # noinspection PyUnresolvedReferences
        return self.estimator.supports_iterative_fit()

    def get_max_iter(self) -> Optional[int]:
        # noinspection PyUnresolvedReferences
        return self.estimator.get_max_iter()

    def set_config_id(self, config_id: ConfigId):
        # noinspection PyUnresolvedReferences
        self.estimator.set_config_id(config_id)

    def supports_pynisher(self) -> bool:
        # noinspection PyUnresolvedReferences
        return self.estimator.supports_pynisher()

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: True,
            HANDLES_PANEL: True,
            IGNORES_EXOGENOUS_X: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex, Int64Index]
        }

    @staticmethod
    def from_config(
            configuration: Configuration,
            budget: float = None,
            dataset_properties: DatasetProperties = None,
            random_state: np.random.RandomState = None
    ) -> 'TemplateChoice':
        return TemplateChoice(
            config=configuration,
            budget=budget,
            dataset_properties=dataset_properties,
            random_state=random_state
        )
