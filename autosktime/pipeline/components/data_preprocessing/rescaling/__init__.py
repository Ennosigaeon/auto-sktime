import os
from collections import OrderedDict
from typing import Dict, Type

from autosktime.pipeline.components.base import find_components, AutoSktimeChoice, AutoSktimePreprocessingAlgorithm, \
    AutoSktimeComponent

rescaling_directory = os.path.split(__file__)[0]
_rescalers = find_components(__package__, rescaling_directory, AutoSktimePreprocessingAlgorithm)


class RescalingChoice(AutoSktimeChoice, AutoSktimePreprocessingAlgorithm):

    @classmethod
    def get_components(cls) -> Dict[str, Type[AutoSktimeComponent]]:
        components = OrderedDict()
        components.update(_rescalers)
        return components
