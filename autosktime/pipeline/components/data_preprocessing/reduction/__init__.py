import os
from collections import OrderedDict

from autosktime.pipeline.components.base import AutoSktimeChoice, find_components, AutoSktimePreprocessingAlgorithm

classifier_directory = os.path.split(__file__)[0]
_preprocessors = find_components(__package__, classifier_directory, AutoSktimePreprocessingAlgorithm)


class ReductionChoice(AutoSktimeChoice, AutoSktimePreprocessingAlgorithm):

    @classmethod
    def get_components(cls):
        components = OrderedDict()
        components.update(_preprocessors)
        return components
