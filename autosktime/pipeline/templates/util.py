from typing import Optional, Dict, List, Union

import numpy as np

from ConfigSpace import ConfigurationSpace
from autosktime.data import DatasetProperties
from autosktime.pipeline.templates import TemplateChoice


def get_configuration_space(
        dataset_properties: DatasetProperties,
        include: Optional[Dict[str, List[str]]] = None,
        exclude: Optional[Dict[str, List[str]]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
) -> ConfigurationSpace:
    return TemplateChoice(random_state=random_state).get_hyperparameter_search_space(
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude
    )
