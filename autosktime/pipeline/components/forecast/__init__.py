import os
from collections import OrderedDict
from typing import List, Dict, Type

import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosktime.constants import SUPPORTED_INDEX_TYPES, HANDLES_UNIVARIATE, HANDLES_MULTIVARIATE, IGNORES_EXOGENOUS_X, \
    HANDLES_MISSING
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, find_components, AutoSktimeChoice, \
    AutoSktimePredictor

forecaster_directory = os.path.split(__file__)[0]
_forecasters = find_components(__package__, forecaster_directory, AutoSktimePredictor)


class ForecasterChoice(AutoSktimeChoice):

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None):
        return {
            HANDLES_UNIVARIATE: True,
            HANDLES_MULTIVARIATE: False,
            IGNORES_EXOGENOUS_X: False,
            HANDLES_MISSING: False,
            SUPPORTED_INDEX_TYPES: [pd.RangeIndex, pd.DatetimeIndex, pd.PeriodIndex]
        }

    @classmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        components = OrderedDict()
        components.update(_forecasters)
        return components

    @classmethod
    def get_available_components(
            cls,
            dataset_properties: DatasetProperties = None,
            include: List[str] = None,
            exclude: List[str] = None
    ) -> Dict[str, Type[AutoSktimeComponent]]:
        available_comp = cls.get_components()
        components_dict = OrderedDict()

        if include is not None and exclude is not None:
            raise ValueError("The argument include and exclude cannot be used together.")

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: {}".format(incl))

        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            entry = available_comp[name]

            # Avoid infinite loop
            if entry == ForecasterChoice:
                continue

            if dataset_properties is not None:
                if dataset_properties.index_type not in entry.get_properties()[SUPPORTED_INDEX_TYPES]:
                    continue

            components_dict[name] = entry

        return components_dict

    def get_hyperparameter_search_space(
            self,
            dataset_properties: DatasetProperties = None,
            default: str = None,
            include: List[str] = None,
            exclude: List[str] = None
    ) -> ConfigurationSpace:
        if include is not None and exclude is not None:
            raise ValueError("The arguments include and exclude cannot be used together.")

        cs = ConfigurationSpace()

        # Compile a list of all objects for this problem
        available_components = self.get_available_components(dataset_properties, include, exclude)

        if len(available_components) == 0:
            raise ValueError("No forecasters found")

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

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs
