import numpy as np
import pandas as pd
from sklearn.svm import SVR
from typing import Tuple, List

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from autosktime.data import DatasetProperties
from autosktime.pipeline.components.base import AutoSktimeComponent, UpdatablePipeline, \
    SwappedInput, COMPONENT_PROPERTIES, AutoSktimeRegressionAlgorithm
from autosktime.pipeline.components.data_preprocessing import VarianceThresholdComponent
from autosktime.pipeline.components.data_preprocessing.rescaling.standardize import StandardScalerComponent
from autosktime.pipeline.components.features.flatten import FlatteningFeatureGenerator
from autosktime.pipeline.templates import PanelRegressionPipeline
from autosktime.pipeline.templates.preconstructed import FixedRecursivePanelReducer


class SVM(AutoSktimeRegressionAlgorithm):

    def __init__(
            self,
            C: float = 0.9251,
            epsilon: float = 0.1951,
            gamma: float = 0.0702,
            random_state: np.random.RandomState = None
    ):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.estimator = SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma, kernel='rbf')
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        if y.shape[0] > 50000:
            stride = max(1, int(X.shape[0] / 50000))
            y = y[::stride]
            X = X[::stride]

        # noinspection PyUnresolvedReferences
        self.estimator.fit(X, y)
        return self

    def _update(self):
        pass

    @staticmethod
    def get_properties(dataset_properties: DatasetProperties = None) -> COMPONENT_PROPERTIES:
        pass

    def get_max_iter(self):
        return 1

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties: DatasetProperties = None) -> ConfigurationSpace:
        C = UniformFloatHyperparameter('C', 0.001, 100, default_value=0.9251, log=True)
        gamma = UniformFloatHyperparameter('gamma', 1e-6, 1, default_value=0.0702, log=True)
        epsilon = UniformFloatHyperparameter('epsilon', 0.1, 1000, default_value=0.1951, log=True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, epsilon, gamma])
        return cs


class SVMRegressionPipeline(PanelRegressionPipeline):
    """
    Implementation of "Hybrid PSO–SVM-based method for forecasting of the remaining useful life for aircraft engines and evaluation of its reliability"
    Set runcount_limit to 100, ensemble_size to 1 and disable pynisher (use_pynisher) when selecting this template.
    """

    def _get_pipeline_steps(self) -> List[Tuple[str, AutoSktimeComponent]]:
        pipeline = UpdatablePipeline(steps=[
            ('feature_generation', FlatteningFeatureGenerator(random_state=self.random_state)),
            ('variance_threshold', VarianceThresholdComponent()),
            ('regression', SVM(random_state=self.random_state))
        ])

        steps = [
            ('reduction',
             FixedRecursivePanelReducer(
                 transformers=[
                     ('scaling', SwappedInput(StandardScalerComponent())),
                 ],
                 estimator=pipeline,
                 random_state=self.random_state,
                 dataset_properties=self.dataset_properties)
             ),
        ]
        return steps

    def _get_hyperparameter_search_space(self) -> ConfigurationSpace:
        return super()._get_hyperparameter_search_space()
